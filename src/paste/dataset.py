import os
import numpy as np
import cv2
import random
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import albumentations as A
from src.paste.utils.feature_extractors import TextureExtractor, ColorHistogramExtractor, ShapeFeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.ndimage
try:
    from skimage.segmentation import slic
except ImportError:
    slic = None

class CustomCropDataset(Dataset):
    def __init__(self, root_path, image_size=(640, 640), transform=None, hyp=None, apply_aug=True):
        """
        Dataset otimizado para ROI Align - retorna apenas imagens, máscaras e coordenadas
        
        Args:
            root_path: Caminho para pasta com imagens e máscaras
            image_size: Tamanho padrão das imagens (640, 640)
            transform: Transformações adicionais
            hyp: Hiperparâmetros para augmentação
            apply_aug: Se True, aplica augmentação (só deve ser True para treino)
        """
        self.root_path = root_path
        self.image_size = image_size
        self.hyp = hyp
        self.apply_aug = apply_aug  # Novo parâmetro
        self.to_tensor = transforms.ToTensor()
        self.resize_transform = transforms.Resize(self.image_size)
        self.debug_vis = True   
        
        # Define transformações de augmentação conforme o código existente
        if self.hyp is not None:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.3),
                A.CLAHE(p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
                A.Blur(blur_limit=5, p=0.2),
                A.MedianBlur(blur_limit=5, p=0.1),
                A.MotionBlur(blur_limit=5, p=0.1),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
                A.GridDistortion(p=0.2),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=30, p=0.2),
                A.RandomResizedCrop(height=self.image_size[0], width=self.image_size[1], scale=(0.7, 1.0), p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=int(self.hyp.get('hsv_h', 0.015)*180),
                    sat_shift_limit=int(self.hyp.get('hsv_s', 0.7)*255),
                    val_shift_limit=int(self.hyp.get('hsv_v', 0.4)*255),
                    p=0.5
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.2,
                    rotate_limit=10,
                    p=0.7
                ),
                A.Perspective(scale=(0.0, 0.01), p=0.5),
            ])
        else:
            self.transform = None
        
        # Carregar caminhos de imagens e máscaras
        self.lesion_image_paths, self.lesion_masks_paths, self.lesion_labels = self._data_load('metaplasia')
        
        # Para guardar a época atual para mosaic scheduling
        self.current_epoch = 0
        self.total_epochs = 100
        
        # Inicialize extratores fora do loop
        self.texture_extractor = TextureExtractor().eval()
        self.color_extractor = ColorHistogramExtractor().eval()
        self.shape_extractor = ShapeFeatureExtractor().eval()
        
        # Calcular estatísticas das lesões
        self.lesion_stats = self._compute_lesion_stats()
        self.context_stats = self._compute_context_stats()
        
    def _extract_cluster_stats(self, image, mask):
        """
        Extrai estatísticas dos clusters para análise e avaliação
        
        Args:
            image: Imagem como array numpy
            mask: Máscara binária de lesões
            
        Returns:
            Dicionário com estatísticas dos clusters
        """
        try:
            # Extrai propriedades das lesões
            lesion_features = self._extract_lesion_properties(image, mask.astype(np.uint8))
            
            # Se não há lesões, retornar None
            if not lesion_features:
                return None
            
            # Gera superpixels
            n_segments = 100
            if slic is not None:
                segments = slic(image, n_segments=n_segments, compactness=10, sigma=1)
            else:
                # Fallback - grid simples
                img_h, img_w = image.shape[:2]
                segments = np.zeros((img_h, img_w), dtype=np.int32)
                grid_size = 10
                cell_h, cell_w = img_h // grid_size, img_w // grid_size
                for i in range(grid_size):
                    for j in range(grid_size):
                        segments[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = i*grid_size + j
            
            # Extrai características para cada segmento
            segment_features = []
            for segment_id in range(np.max(segments) + 1):
                segment_mask = (segments == segment_id)
                # Pular segmentos vazios ou muito pequenos
                if not np.any(segment_mask) or np.sum(segment_mask) < 100:
                    continue
                    
                # Evitar áreas de lesão
                if np.sum(segment_mask & mask) > 0:
                    continue
                    
                # Calcular centroide
                y_indices, x_indices = np.where(segment_mask)
                cx, cy = np.mean(x_indices), np.mean(y_indices)
                    
                # Características de cor e textura
                segment_color = np.mean(image[segment_mask], axis=0) if np.sum(segment_mask) > 0 else np.zeros(3)
                
                if len(image.shape) == 3:
                    gray = cv2.cvtColor((image * np.expand_dims(segment_mask, 2)).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray = image * segment_mask
                    
                segment_texture = np.std(gray[segment_mask > 0]) if np.sum(segment_mask) > 0 else 0
                segment_luminance = np.mean(gray[segment_mask > 0]) if np.sum(segment_mask) > 0 else 0
                
                # Armazenar
                segment_features.append({
                    'id': segment_id,
                    'centroid': (cx, cy),
                    'color': segment_color,
                    'texture': segment_texture,
                    'luminance': segment_luminance,
                    'size': np.sum(segment_mask)
                })
            
            # Executar clustering
            if len(segment_features) < 10:
                return None
                
            # Criar matriz de características para clustering
            feature_vectors = []
            for segment in segment_features:
                # Combinar cor, textura e luminância
                feature_vec = np.concatenate([
                    segment['color'] / 255.0,
                    [segment['texture'] / 50.0],
                    [segment['luminance'] / 255.0]
                ])
                feature_vectors.append(feature_vec)
                
            feature_matrix = np.array(feature_vectors)
            
            # Executar K-means
            n_clusters = 3
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(feature_matrix)
            
            # Associar segmentos aos clusters
            for i, segment in enumerate(segment_features):
                segment['cluster'] = int(cluster_labels[i])
            
            # Calcular estatísticas por cluster
            cluster_stats = {}
            for i in range(n_clusters):
                segments_in_cluster = [s for s in segment_features if s['cluster'] == i]
                if segments_in_cluster:
                    cluster_stats[i] = {
                        'size': len(segments_in_cluster),
                        'luminance': np.mean([s['luminance'] for s in segments_in_cluster]),
                        'texture': np.mean([s['texture'] for s in segments_in_cluster])
                    }
            
            # Classificar clusters
            sorted_clusters = sorted(cluster_stats.keys(), 
                                    key=lambda c: cluster_stats[c]['luminance'], 
                                    reverse=True)
            
            if len(sorted_clusters) >= 2:
                positive_cluster = sorted_clusters[0]
                negative_cluster = sorted_clusters[-1]
                
                # Separação entre clusters (qualidade do agrupamento)
                separation = abs(cluster_stats[positive_cluster]['luminance'] - 
                            cluster_stats[negative_cluster]['luminance'])
                
                return {
                    'positive_size': cluster_stats[positive_cluster]['size'],
                    'negative_size': cluster_stats[negative_cluster]['size'],
                    'positive_luminance': cluster_stats[positive_cluster]['luminance'],
                    'negative_luminance': cluster_stats[negative_cluster]['luminance'],
                    'separation_quality': min(1.0, separation / 50.0)
                }
            
            return None
        except Exception as e:
            print(f"Erro na extração de estatísticas de cluster: {e}")
            return None
        
    def _data_load(self, subset=None):
        """
        Carrega caminhos para imagens e máscaras do dataset.
        
        Args:
            subset: Nome do subconjunto (ex: 'metaplasia')
            
        Returns:
            Tupla (imagens, máscaras, labels)
        """
        # Correção para lidar com a pasta duplicada
        if subset:
            images_dir = os.path.join(self.root_path, subset, 'images')
            masks_dir = os.path.join(self.root_path, subset, 'masks')
            
            # Se não existir, tenta com caminho duplicado
            if not os.path.exists(images_dir):
                images_dir = os.path.join(self.root_path, subset, subset, 'images')
                masks_dir = os.path.join(self.root_path, subset, subset, 'masks')
        else:
            # Fallback para caminho padrão
            images_dir = os.path.join(self.root_path, 'images')
            masks_dir = os.path.join(self.root_path, 'masks')
        
        # Verificar se os diretórios existem
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Diretório de imagens não encontrado: {images_dir}")
        if not os.path.exists(masks_dir):
            raise FileNotFoundError(f"Diretório de máscaras não encontrado: {masks_dir}")
            
        # Resto do código permanece o mesmo
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Construir listas de caminhos
        image_paths = [os.path.join(images_dir, f) for f in image_files]
        mask_paths = [os.path.join(masks_dir, f.replace('.jpg', '.png').replace('.jpeg', '.png')) for f in image_files]
        
        # Verificar se cada máscara existe
        valid_paths = []
        valid_masks = []
        valid_labels = []
        
        for img_path, mask_path in zip(image_paths, mask_paths):
            if os.path.exists(mask_path):
                valid_paths.append(img_path)
                valid_masks.append(mask_path)
                valid_labels.append(0)  # Default label = 0
        
        return valid_paths, valid_masks, valid_labels
        
    def __len__(self):
        return len(self.lesion_image_paths)
    
    def __getitem__(self, idx):
        import os, random, numpy as np, torch, cv2
        from PIL import Image

        # Carrega imagem/máscara desta amostra
        img_path = self.lesion_image_paths[idx]
        mask_path = self.lesion_masks_paths[idx]
        image = Image.open(img_path).convert("RGB").resize(self.image_size)
        mask = Image.open(mask_path).convert("L").resize(self.image_size)

        # Mosaic scheduling (opcional)
        mosaic_prob = self.hyp.get('mosaic', 1.0) if self.hyp else 0.0
        use_mosaic = random.random() < mosaic_prob
        if hasattr(self, 'current_epoch') and hasattr(self, 'total_epochs'):
            if self.current_epoch >= max(0, int(0.9 * self.total_epochs)):
                use_mosaic = False

        if use_mosaic:
            # Mosaico com 4 imagens
            import numpy as np
            import random
            indices = [random.randint(0, len(self.lesion_image_paths)-1) for _ in range(4)]
            try:
                mosaic_img, mosaic_mask, _ = mosaic_augmentation(self, indices, image_size=self.image_size)
                image = mosaic_img
                mask = mosaic_mask
            except Exception:
                # fallback para single
                pass

        # Augmentations (apenas se apply_aug True)
        image_np = np.array(image)
        mask_np = np.array(mask)
        if self.transform and self.apply_aug:
            # Exemplo simples: apenas garantir que transform aceita dicts/tensores já convertidos
            # Caso contrário, deixe como está
            try:
                aug = self.transform(image=image_np, mask=mask_np)
                image_np, mask_np = aug["image"], aug["mask"]
            except Exception:
                pass

        image = Image.fromarray(image_np)
        mask = Image.fromarray(mask_np)

        # Extrai boxes
        lesion_boxes = self._extract_instance_boxes(image, mask)  # lista [cls, cx, cy, w, h]
        pos_boxes, neg_boxes = self.region_grow_background_boxes(image, mask, expand_ratio=1.5, max_steps=5, similarity_threshold=3.5)

        # Garante formato
        lesion_boxes = lesion_boxes or []
        pos_boxes = pos_boxes or []
        neg_boxes = neg_boxes or []

        # Combina boxes e labels em tensor [N,6] => [cls, cx, cy, w, h, label]
        all_rows = []
        for b in lesion_boxes:
            # label=1 para lesão
            all_rows.append([b[0], b[1], b[2], b[3], b[4], 1.0])
        for b in pos_boxes:
            all_rows.append([b[0], b[1], b[2], b[3], b[4], 2.0])
        for b in neg_boxes:
            all_rows.append([b[0], b[1], b[2], b[3], b[4], 0.0])

        if all_rows:
            box_data = torch.tensor(all_rows, dtype=torch.float32)
        else:
            box_data = torch.zeros((0, 6), dtype=torch.float32)

        # Tensores de saída
        image_tensor = self.to_tensor(image)
        mask_tensor = self.to_tensor(mask)

        return image_tensor, mask_tensor, box_data

    def _compute_lesion_stats(self):
        """
        Calcula médias e desvios das características das lesões reais do dataset.
        """
        texture_feats = []
        color_feats = []
        illumination_vals = []
        for img_path, mask_path in zip(self.lesion_image_paths, self.lesion_masks_paths):
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            img = self.resize_transform(img)
            mask = self.resize_transform(mask)
            mask_np = np.array(mask)
            img_np = np.array(img)
            # Extrai regiões de lesão
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_np > 0).astype(np.uint8) * 255, connectivity=8)
            for i in range(1, num_labels):
                x, y, w, h, _ = stats[i]
                region = img_np[y:y+h, x:x+w]
                if region.size == 0:
                    continue
                region_tensor = torch.tensor(region / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                texture_feats.append(self.texture_extractor(region_tensor).detach().cpu().numpy().flatten())
                color_feats.append(self.color_extractor(region_tensor).detach().cpu().numpy().flatten())
                illumination_vals.append(region_tensor.mean().item())
        return {
            'mean_texture': np.mean(texture_feats, axis=0) if texture_feats else 0,
            'std_texture': np.std(texture_feats, axis=0) if texture_feats else 1,
            'mean_color': np.mean(color_feats, axis=0) if color_feats else 0,
            'std_color': np.std(color_feats, axis=0) if color_feats else 1,
            'mean_illumination': np.mean(illumination_vals) if illumination_vals else 0,
            'std_illumination': np.std(illumination_vals) if illumination_vals else 1
        }

    def _compute_context_stats(self, expand_ratio=2.0):
        texture_feats = []
        color_feats = []
        illumination_vals = []
        for img_path, mask_path in zip(self.lesion_image_paths, self.lesion_masks_paths):
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            img = self.resize_transform(img)
            mask = self.resize_transform(mask)
            mask_np = np.array(mask)
            img_np = np.array(img)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_np > 0).astype(np.uint8) * 255, connectivity=8)
            for i in range(1, num_labels):
                x, y, w, h, _ = stats[i]
                # Expande box para contexto
                cx, cy = x + w/2, y + h/2
                bg_w, bg_h = w * expand_ratio, h * expand_ratio
                bg_x1 = max(0, int(cx - bg_w / 2))
                bg_y1 = max(0, int(cy - bg_h / 2))
                bg_x2 = min(img_np.shape[1], int(cx + bg_w / 2))
                bg_y2 = min(img_np.shape[0], int(cy + bg_h / 2))
                region = img_np[bg_y1:bg_y2, bg_x1:bg_x2]
                if region.size == 0:
                    continue
                region_tensor = torch.tensor(region / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                texture_feats.append(self.texture_extractor(region_tensor).detach().cpu().numpy().flatten())
                color_feats.append(self.color_extractor(region_tensor).detach().cpu().numpy().flatten())
                illumination_vals.append(region_tensor.mean().item())
        return {
            'mean_texture': np.mean(texture_feats, axis=0) if texture_feats else 0,
            'std_texture': np.std(texture_feats, axis=0) if texture_feats else 1,
            'mean_color': np.mean(color_feats, axis=0) if color_feats else 0,
            'std_color': np.std(color_feats, axis=0) if color_feats else 1,
            'mean_illumination': np.mean(illumination_vals) if illumination_vals else 0,
            'std_illumination': np.std(illumination_vals) if illumination_vals else 1
        }

    def _compute_cluster_feature_stats(self, img_np, mask):
        """
        Calcula média e desvio dos features (Lab mean, laplacian std, luminância mean, contraste std)
        para a máscara (mask>0). Retorna (mu, sigma).
        """
        mask = (mask > 0)
        if not np.any(mask):
            return None, None
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

        L = lab[:, :, 0][mask] / 255.0
        A = (lab[:, :, 1][mask] - 128.0) / 128.0
        B = (lab[:, :, 2][mask] - 128.0) / 128.0
        tex = np.std(lap[mask]) / 50.0
        lum = np.mean(L)
        contrast = np.std(gray[mask]) / 50.0

        feat = np.array([np.mean(L), np.mean(A), np.mean(B), tex, lum, contrast], dtype=np.float32)
        # Para sigma, calculamos feature-wise onde aplicável
        # (para tex, lum, contrast usamos valores escalares => sigma mínimo)
        tex_vals = np.array([tex])
        lum_vals = np.array([lum])
        contrast_vals = np.array([contrast])
        sigma = np.array([
            np.std(L) + 1e-6,
            np.std(A) + 1e-6,
            np.std(B) + 1e-6,
            max(1e-6, np.std(tex_vals)),
            max(1e-6, np.std(lum_vals)),
            max(1e-6, np.std(contrast_vals))
        ], dtype=np.float32)
        return feat, sigma
    
    def _compute_box_features(self, img_np, x1, y1, x2, y2):
        """
        Extrai o mesmo vetor de características para uma janela (x1,y1,x2,y2).
        """
        patch = img_np[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

        L = lab[:, :, 0] / 255.0
        A = (lab[:, :, 1] - 128.0) / 128.0
        B = (lab[:, :, 2] - 128.0) / 128.0
        if L.size < 25:
            return None

        tex = np.std(lap) / 50.0
        lum = np.mean(L)
        contrast = np.std(gray) / 50.0
        feat = np.array([
            np.mean(L),
            np.mean(A),
            np.mean(B),
            tex,
            lum,
            contrast
        ], dtype=np.float32)
        return feat
    
    
    

    def _extract_instance_boxes(self, image, mask):
        """
        Extrai as coordenadas YOLO das instâncias de lesão na máscara.
        
        Args:
            image: Imagem PIL
            mask: Máscara PIL
        
        Returns:
            Lista de coordenadas YOLO [class_idx, x_center, y_center, width, height]
        """
        mask_np = np.array(mask)
        if len(mask_np.shape) == 3:
            mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
        mask_np = (mask_np > 0).astype(np.uint8) * 255
        
        # Encontrar componentes conectados (instâncias)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        
        yolo_boxes = []
        img_w, img_h = image.size
        
        # Para cada instância, extrair coordenadas normalizadas
        for i in range(1, num_labels):  # Skip label 0 (background)
            x, y, w, h, _ = stats[i]
            
            # Coordenadas YOLO normalizadas
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            bbox_width = w / img_w
            bbox_height = h / img_h
            yolo_boxes.append([0, x_center, y_center, bbox_width, bbox_height])  # class_idx=0
        
        return yolo_boxes

    def _extract_background_boxes(self, image, mask, expand_ratio=1.5):
        img_w, img_h = image.size
        lesion_boxes = self._extract_instance_boxes(image, mask)
        bg_boxes = []

        # Para cada lesão, cria uma região expandida ao redor (ACEITA SEM FILTRO)
        for box in lesion_boxes:
            _, cx, cy, w, h = box
            abs_cx = cx * img_w
            abs_cy = cy * img_h
            abs_w = w * img_w
            abs_h = h * img_h

            bg_w = abs_w * expand_ratio
            bg_h = abs_h * expand_ratio

            bg_x1 = max(0, int(abs_cx - bg_w / 2))
            bg_y1 = max(0, int(abs_cy - bg_h / 2))
            bg_x2 = min(img_w, int(abs_cx + bg_w / 2))
            bg_y2 = min(img_h, int(abs_cy + bg_h / 2))

            bg_boxes.append([0, bg_x1/img_w, bg_y1/img_h, bg_x2/img_w, bg_y2/img_h])
        
        return bg_boxes
    
    def _generate_candidate_boxes_from_mask(self, mask, img_w, img_h,
                                            box_size_factors=(0.18, 0.13, 0.09),
                                            stride_factors=(0.5, 0.4, 0.3),
                                            min_coverage=0.75,
                                            max_raw=200):
        """
        Varre a imagem gerando candidatos dentro da máscara (mask>0) com cobertura mínima.
        Retorna lista de dicts: [{'cx':..,'cy':..,'w':..,'h':..,'rect':(x1,y1,x2,y2)}]
        """
        mask = (mask > 0).astype(np.uint8)
        candidates = []
        for size_f, stride_f in zip(box_size_factors, stride_factors):
            bw = int(img_w * size_f)
            bh = int(img_h * size_f)
            if bw < 10 or bh < 10:
                continue
            sx = max(1, int(bw * stride_f))
            sy = max(1, int(bh * stride_f))
            for y in range(0, img_h - bh + 1, sy):
                for x in range(0, img_w - bw + 1, sx):
                    win = mask[y:y+bh, x:x+bw]
                    if win.shape != (bh, bw):
                        continue
                    cov = np.sum(win) / (bw * bh)
                    if cov >= min_coverage:
                        cx = (x + bw / 2) / img_w
                        cy = (y + bh / 2) / img_h
                        candidates.append({
                            "cx": cx, "cy": cy,
                            "w": bw / img_w, "h": bh / img_h,
                            "rect": (x, y, x + bw, y + bh)
                        })
                        if len(candidates) >= max_raw:
                            return candidates
        return candidates
    
    

    def _grid_based_boxes(self, image, mask, similarity_threshold=5.5):
        """
        Versão simplificada do método original baseado em grid.
        Usado como fallback quando a amostragem baseada em distância não gera boxes suficientes.
        """
        img_w, img_h = image.size
        img_np = np.array(image)
        
        # Estatísticas globais da imagem
        global_mean_color = np.mean(img_np, axis=(0, 1))
        global_std_color = np.std(img_np, axis=(0, 1))
        luminance = np.mean(img_np, axis=2)
        global_mean_luminance = np.mean(luminance)
        global_std_luminance = np.std(luminance)
        
        positive_boxes = []
        negative_boxes = []
        
        # Grid simples 6x6
        grid_size = 6
        cell_w, cell_h = img_w / grid_size, img_h / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Extrair coordenadas da célula
                x1 = int(i * cell_w)
                y1 = int(j * cell_h)
                x2 = int((i + 1) * cell_w)
                y2 = int((j + 1) * cell_h)
                
                # Coordenadas YOLO da célula (centro, largura, altura)
                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                width = 0.8 * cell_w / img_w
                height = 0.8 * cell_h / img_h
                
                # Verificar validez
                if x1 >= x2 or y1 >= y2:
                    continue
                    
                # Extrair região para análise
                try:
                    region = img_np[y1:y2, x1:x2]
                except IndexError:
                    continue
                
                # Ignorar células muito pequenas
                if region.shape[0] < 20 or region.shape[1] < 20:
                    continue
                
                # Classificar com critério simplificado
                region_luminance = np.mean(np.mean(region, axis=2))
                
                # Regiões muito escuras são negativas
                if region_luminance < 30:
                    negative_boxes.append([0, cx, cy, width, height])
                # Regiões com luminância próxima à média são positivas
                elif abs(region_luminance - global_mean_luminance) < 2.0 * global_std_luminance:
                    positive_boxes.append([0, cx, cy, width, height])
                # Outras são negativas
                else:
                    negative_boxes.append([0, cx, cy, width, height])
        
        return positive_boxes, negative_boxes
    
    def _visualize_clusters(self, image, segments, segment_features, positive_cluster, negative_cluster, has_lesion_context=False):
        """
        Cria visualização dos clusters para debug
        
        Args:
            image: Imagem original (array numpy)
            segments: Resultado da segmentação
            segment_features: Lista de dicionários de características dos segmentos
            positive_cluster: ID do cluster positivo
            negative_cluster: ID do cluster negativo
            has_lesion_context: Se True, indica que usou contexto de lesões reais
        """
        # Criar visualização dos clusters
        cluster_vis = np.zeros_like(image)
        
        # Cores para clusters
        colors = {
            positive_cluster: [0, 255, 0],  # Verde para cluster positivo
            negative_cluster: [255, 0, 0],  # Vermelho para cluster negativo
        }
        
        # Preencher visualização de cluster
        for segment in segment_features:
            cluster_id = segment['cluster']
            mask = segment['mask']
            
            # Visualizar apenas clusters positivo e negativo
            if cluster_id in colors:
                color = colors[cluster_id]
                for c in range(min(3, cluster_vis.shape[2])):
                    cluster_vis[:, :, c] += (mask * color[c]).astype(np.uint8)
        
        # Criar figura com imagem original e visualização de cluster
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Imagem original
        axes[0].imshow(image)
        axes[0].set_title("Imagem Original")
        axes[0].axis('off')
        
        # Visualização de cluster
        axes[1].imshow(cluster_vis)
        title = "Visualização de Clusters"
        if has_lesion_context:
            title += " (Baseada no Contexto das Lesões)"
        axes[1].set_title(title)
        axes[1].axis('off')
        
        # Adicionar legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Cluster Positivo (Contexto Adequado)'),
            Patch(facecolor='red', label='Cluster Negativo (Contexto Inadequado)')
        ]
        axes[1].legend(handles=legend_elements, loc='upper right')
        
        # Salvar visualização
        debug_dir = 'debug_clusters'
        os.makedirs(debug_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, f"clusters_{np.random.randint(1000)}.png"))
        plt.close(fig)
    
    def _extract_lesion_context_features(self, image, binary_mask, lesion_boxes, expand_ratio=1.8):
        """
        Extrai características do contexto ao redor das lesões reais.
        
        Args:
            image: Imagem como array numpy
            binary_mask: Máscara binária de lesões
            lesion_boxes: Lista de boxes de lesões [cls_idx, cx, cy, w, h]
            expand_ratio: Quanto expandir a box da lesão para capturar o contexto
            
        Returns:
            Lista de vetores de características do contexto
        """
        img_h, img_w = image.shape[:2]
        context_features = []
        
        for box in lesion_boxes:
            _, cx, cy, w, h = box
            
            # Converte para coordenadas absolutas
            abs_cx = cx * img_w
            abs_cy = cy * img_h
            abs_w = w * img_w
            abs_h = h * img_h
            
            # Expande a box
            ctx_w = abs_w * expand_ratio
            ctx_h = abs_h * expand_ratio
            
            # Calcula coordenadas da box expandida
            x1 = max(0, int(abs_cx - ctx_w/2))
            y1 = max(0, int(abs_cy - ctx_h/2))
            x2 = min(img_w, int(abs_cx + ctx_w/2))
            y2 = min(img_h, int(abs_cy + ctx_h/2))
            
            # Cria máscara para isolar apenas o contexto (não a lesão)
            context_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            context_mask[y1:y2, x1:x2] = 1
            
            # Exclui a lesão da máscara de contexto
            context_mask = context_mask & ~binary_mask
            
            # Se há poucos pixels de contexto, continua
            if np.sum(context_mask) < 100:
                continue
                
            # Extrai características do contexto
            context_color = np.mean(image[context_mask > 0], axis=0) if np.sum(context_mask) > 0 else np.zeros(3)
            
            # Extrai características de textura
            if len(image.shape) == 3:
                context_gray = cv2.cvtColor(
                    (image * np.expand_dims(context_mask, 2).astype(np.uint8)),
                    cv2.COLOR_RGB2GRAY
                )
            else:
                context_gray = image * context_mask
                
            context_texture = np.std(context_gray[context_mask > 0]) if np.sum(context_mask) > 0 else 0
            context_luminance = np.mean(context_gray[context_mask > 0]) if np.sum(context_mask) > 0 else 0
            
            # Combina características
            feature_vec = np.concatenate([
                context_color / 255.0,
                [context_texture / 50.0],
                [context_luminance / 255.0]
            ])
            
            context_features.append(feature_vec)
        
        return context_features
    
    def _uniformize_image(self, img_np):
        """
        Uniformiza a imagem para reduzir variações de cor e iluminação.
        Aplica CLAHE no canal L do espaço Lab e normaliza média/desvio.
        """
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
        img_norm = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        img_norm = (img_norm - np.mean(img_norm, axis=(0,1))) / (np.std(img_norm, axis=(0,1)) + 1e-6)
        img_norm = (img_norm * 32 + 128).clip(0,255).astype(np.uint8)
        return img_norm
    
    def region_grow_background_boxes(self, image, mask,
                                expand_ratio=None,          # <- compatibilidade antiga (antes usado p/ expansão)
                                max_steps=None,             # <- compat: antes usado (agora mapeado p/ max_core_steps)
                                n_segments=500,
                                max_core_steps=60,
                                similarity_threshold=5.0,
                                consensus_ratio=0.35,
                                inflation_factor=1.5,
                                frontier_z_margin=0.85,
                                target_ratio=0.18,
                                relax_factor_max=1.8,
                                min_frontier_gain=0.005,
                                max_frontier_passes=6,
                                neg_margin=0.6,
                                luminance_sigma_gate=2.2,
                                max_boxes=10,
                                **_ignore):
        """
        Versão com retro‑compatibilidade:
        - expand_ratio: usado agora como sinal para aumentar target_ratio (cobertura desejada)
        - max_steps: mapeado para max_core_steps se fornecido
        Outros kwargs antigos são ignorados.
        """
        # Ajustes de compatibilidade
        if expand_ratio is not None:
            # Escala simples: maior expand_ratio -> maior target_ratio (limitado)
            target_ratio = min(0.30, 0.10 + 0.08 * float(expand_ratio))
        if max_steps is not None:
            # Garante no mínimo o valor anterior informado
            max_core_steps = max_core_steps if max_core_steps is not None else 60
            # Aumenta ligeiramente para dar chance de crescer mais
            max_core_steps = max(max_core_steps, int(max_steps) * 40)

        import numpy as np, cv2
        from collections import deque

        img_w, img_h = image.size
        img_np = np.array(image)
        mask_np = np.array(mask)
        if len(mask_np.shape) == 3:
            mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
        lesion_mask = (mask_np > 0).astype(np.uint8)

        gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        valid_mask = (gray_full > 12).astype(np.uint8)

        if slic is not None:
            segments = slic(img_np, n_segments=n_segments, compactness=14, sigma=1)
        else:
            segments = np.zeros((img_h, img_w), np.int32)
            step = max(8, int(np.sqrt((img_h * img_w) / n_segments)))
            sid = 0
            for y in range(0, img_h, step):
                for x in range(0, img_w, step):
                    segments[y:y+step, x:x+step] = sid
                    sid += 1
        seg_ids = np.unique(segments)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        gray = gray_full
        lap_full = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

        def seg_feature(sm_bool):
            idx = sm_bool
            L = img_lab[:, :, 0][idx] / 255.0
            a = (img_lab[:, :, 1][idx] - 128) / 128.0
            b = (img_lab[:, :, 2][idx] - 128) / 128.0
            lap = lap_full[idx]
            return np.array([
                L.mean(), a.mean(), b.mean(),
                L.std() + 1e-6,
                lap.std() / 50.0,
                gray[idx].std() / 50.0
            ], dtype=np.float32)

        seg_masks = {sid: (segments == sid) for sid in seg_ids}
        seg_feats = {sid: seg_feature(sm) for sid, sm in seg_masks.items()}

        kernel3 = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(lesion_mask, kernel3, 1)
        ring = (dil > 0) & (lesion_mask == 0)
        pos_seed_ids = [sid for sid, sm in seg_masks.items() if np.any(sm & ring)]

        border = np.zeros_like(lesion_mask)
        border[0:10, :], border[-10:, :], border[:, 0:10], border[:, -10:] = 1, 1, 1, 1
        neg_seed_ids = [sid for sid, sm in seg_masks.items()
                        if np.any(sm & border) and not np.any(sm & lesion_mask)]

        if not pos_seed_ids or not neg_seed_ids:
            return [], []

        seed_feats = np.stack([seg_feats[s] for s in pos_seed_ids], 0)
        seed_mu = seed_feats.mean(0)
        seed_sigma = seed_feats.std(0) + 1e-6

        pos_set = set(pos_seed_ids)
        pos_mu = seed_mu.copy()
        pos_sigma = seed_sigma.copy()

        neg_initial = np.stack([seg_feats[s] for s in neg_seed_ids], 0)
        neg_mu = neg_initial.mean(0)
        neg_sigma = neg_initial.std(0) + 1e-6
        neg_set = set(neg_seed_ids)

        def zdist(vec, mu, sigma):
            return float(np.linalg.norm((vec - mu) / sigma))

        adj = self._build_segment_adjacency(segments)

        from collections import deque
        visited = set(pos_seed_ids + neg_seed_ids)
        q = deque(pos_seed_ids + neg_seed_ids)

        seed_L_mean = seed_mu[0]
        seed_L_std = seed_sigma[0]

        def luminance_ok(Lcand):
            return abs(Lcand - seed_L_mean) <= luminance_sigma_gate * seed_L_std + 1e-6

        total_valid_area = np.sum(valid_mask)
        aimed_pixels = max(np.sum(lesion_mask) + int(total_valid_area * target_ratio),
                            np.sum(lesion_mask) + 50)

        # Fase 1
        steps = 0
        while q and steps < max_core_steps * len(seg_ids):
            cur = q.popleft()
            for nb in adj.get(int(cur), []):
                if nb in visited:
                    continue
                sm = seg_masks[nb]
                if np.any(sm & lesion_mask) or np.any(sm & (valid_mask == 0)):
                    visited.add(nb); continue
                feat = seg_feats[nb]
                if not luminance_ok(feat[0]):
                    visited.add(nb); continue
                d_pos_seed = zdist(feat, seed_mu, seed_sigma)
                d_pos = zdist(feat, pos_mu, pos_sigma)
                d_neg = zdist(feat, neg_mu, neg_sigma)
                nb_neigh = adj.get(int(nb), [])
                pos_neigh = sum((n in pos_set) for n in nb_neigh)
                neg_neigh = sum((n in neg_set) for n in nb_neigh)
                tot = max(1, len(nb_neigh))
                if (pos_neigh / tot >= consensus_ratio and
                    d_pos + 0.5 < d_neg and
                    d_pos_seed < similarity_threshold):
                    current_pos_feats = np.vstack([seed_feats, *(seg_feats[s] for s in pos_set)])
                    new_std = np.vstack([current_pos_feats, feat]).std(0) + 1e-6
                    if np.all(new_std <= seed_sigma * inflation_factor):
                        pos_set.add(nb)
                        current_pos_feats = np.vstack([current_pos_feats, feat])
                        pos_mu = current_pos_feats.mean(0)
                        pos_sigma = current_pos_feats.std(0) + 1e-6
                        q.append(nb)
                        visited.add(nb)
                        continue
                if (neg_neigh / tot >= consensus_ratio and
                    d_neg + 0.5 < d_pos):
                    neg_set.add(nb)
                    neg_feats_all = np.vstack([neg_initial, *(seg_feats[s] for s in neg_set)])
                    neg_mu = neg_feats_all.mean(0)
                    neg_sigma = neg_feats_all.std(0) + 1e-6
                    q.append(nb)
                    visited.add(nb)
                    continue
                visited.add(nb)
            steps += 1

        pos_mask_core = np.zeros((img_h, img_w), np.uint8)
        for sid in pos_set:
            pos_mask_core |= seg_masks[sid].astype(np.uint8)
        pos_mask_core[lesion_mask > 0] = 0

        # Fase 2
        def frontier_candidates():
            fc = set()
            for sid in pos_set:
                for nb in adj.get(int(sid), []):
                    if nb in pos_set or nb in neg_set:
                        continue
                    fc.add(nb)
            return list(fc)

        relax_factor = 1.0
        prev_area = np.sum(pos_mask_core)
        passes = 0
        while passes < max_frontier_passes and (prev_area + np.sum(lesion_mask)) < aimed_pixels:
            added_any = 0
            cand_list = frontier_candidates()
            if not cand_list:
                break
            for nb in cand_list:
                if nb in pos_set or nb in neg_set:
                    continue
                sm = seg_masks[nb]
                if np.any(sm & lesion_mask) or np.any(sm & (valid_mask == 0)):
                    continue
                feat = seg_feats[nb]
                if not luminance_ok(feat[0]):
                    continue
                d_seed = zdist(feat, seed_mu, seed_sigma)
                d_cur = zdist(feat, pos_mu, pos_sigma)
                if (d_seed <= similarity_threshold * relax_factor and
                    d_cur <= similarity_threshold * relax_factor + frontier_z_margin):
                    current_pos_feats = np.vstack([seed_feats, *(seg_feats[s] for s in pos_set)])
                    new_std = np.vstack([current_pos_feats, feat]).std(0) + 1e-6
                    if np.all(new_std <= seed_sigma * inflation_factor * 1.15):
                        pos_set.add(nb)
                        current_pos_feats = np.vstack([current_pos_feats, feat])
                        pos_mu = current_pos_feats.mean(0)
                        pos_sigma = current_pos_feats.std(0) + 1e-6
                        added_any += np.sum(sm)
            new_area = 0
            if added_any > 0:
                pos_mask_core = np.zeros((img_h, img_w), np.uint8)
                for sid in pos_set:
                    pos_mask_core |= seg_masks[sid].astype(np.uint8)
                pos_mask_core[lesion_mask > 0] = 0
                new_area = np.sum(pos_mask_core)
            gain = (new_area - prev_area) / max(1, prev_area)
            if gain < min_frontier_gain:
                if relax_factor < relax_factor_max:
                    relax_factor *= 1.25
                else:
                    break
            prev_area = new_area if new_area > 0 else prev_area
            passes += 1

        ring_mask = ring.astype(np.uint8)
        cc_num, cc_lab = cv2.connectedComponents(pos_mask_core)
        keep = np.zeros(cc_num, np.uint8)
        for c in range(1, cc_num):
            comp = (cc_lab == c)
            if np.any(comp & ring_mask):
                keep[c] = 1
        clean_pos = (keep[cc_lab] > 0).astype(np.uint8)
        pos_mask_total = ((clean_pos > 0) | (lesion_mask > 0)).astype(np.uint8)

        neg_mask_plain = (valid_mask > 0) & (pos_mask_total == 0)
        neg_mask = neg_mask_plain.astype(np.uint8)

        positive_boxes = self._create_strictly_contained_boxes(
            pos_mask_total.astype(np.uint8), img_w, img_h,
            min_coverage=0.80, max_boxes=max_boxes
        )

        neg_cc, neg_lab = cv2.connectedComponents(neg_mask.astype(np.uint8))
        neg_comp_areas = []
        for c in range(1, neg_cc):
            comp = (neg_lab == c)
            area = np.sum(comp)
            if area < 80:
                continue
            neg_comp_areas.append((area, c))
        neg_comp_areas.sort(reverse=True)
        neg_boxes = []
        for _, c in neg_comp_areas[:max_boxes * 3]:
            comp_mask = (neg_lab == c).astype(np.uint8)
            cb = self._create_strictly_contained_boxes(
                comp_mask, img_w, img_h, min_coverage=0.80, max_boxes=1
            )
            if cb:
                neg_boxes.extend(cb)
            if len(neg_boxes) >= max_boxes:
                break
        if len(neg_boxes) < max_boxes:
            extra = self._create_strictly_contained_boxes(
                neg_mask.astype(np.uint8), img_w, img_h,
                min_coverage=0.80, max_boxes=(max_boxes - len(neg_boxes))
            )
            neg_boxes.extend(extra)

        if hasattr(self, 'debug_vis') and self.debug_vis:
            self._visualize_region_growing(img_np, lesion_mask, pos_mask_total, neg_mask.astype(np.uint8))

        return positive_boxes, neg_boxes
    
    def _extract_lesion_properties(self, image, mask):
        """
        Extrai cor, textura e outras propriedades das áreas de lesão
        
        Args:
            image: Imagem RGB como array numpy
            mask: Máscara binária de lesões
            
        Returns:
            Lista de dicionários de características para cada lesão
        """
        # Encontrar componentes conectados na máscara
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask * 255, connectivity=8)
        
        lesion_features = []
        
        # Pular fundo (label 0)
        for i in range(1, num_labels):
            # Obter bounding box
            x, y, w, h, area = stats[i]
            
            # Pular áreas muito pequenas
            if area < 50:
                continue
                
            # Criar máscara para esta lesão específica
            lesion_mask = (labels == i).astype(np.uint8)
            
            # Extrair região
            lesion_region = np.zeros_like(image)
            for c in range(min(3, lesion_region.shape[2])):
                lesion_region[:, :, c] = image[:, :, c] * lesion_mask
                
            # Calcular características de cor
            lesion_color = np.mean(image[lesion_mask > 0], axis=0) if np.sum(lesion_mask) > 0 else np.zeros(3)
            
            # Calcular textura (variância simples)
            if len(image.shape) == 3:
                gray_lesion = cv2.cvtColor(
                    (image * np.expand_dims(lesion_mask, 2)).astype(np.uint8),
                    cv2.COLOR_RGB2GRAY
                )
            else:
                gray_lesion = image * lesion_mask
                
            lesion_texture = np.std(gray_lesion[lesion_mask > 0]) if np.sum(lesion_mask) > 0 else 0
            
            # Calcular luminância
            lesion_luminance = np.mean(gray_lesion[lesion_mask > 0]) if np.sum(lesion_mask) > 0 else 0
            
            # Armazenar características
            lesion_features.append({
                'mask': lesion_mask,
                'bbox': (x, y, w, h),
                'color': lesion_color,
                'texture': lesion_texture,
                'luminance': lesion_luminance
            })
            
        return lesion_features
    
    def _region_grow(self, image, seed_mask, reference_color, reference_texture, threshold=0.3, max_iterations=30):
        """
        Implementa o algoritmo de region growing com expansão melhorada.
        """
        h, w = image.shape[:2]
        grown_mask = seed_mask.copy()
        
        # Kernel maior para dilatação (vizinhança mais ampla)
        kernel = np.ones((5, 5), np.uint8)  # Aumentado de (3,3) para (5,5)
        
        # Region growing iterativo com mais iterações
        for iteration in range(max_iterations):
            # Dilatar a máscara atual para encontrar vizinhos
            dilated = cv2.dilate(grown_mask.astype(np.uint8), kernel, iterations=1)
            
            # Encontrar novos candidatos (vizinhos que ainda não estão na máscara)
            candidates = np.logical_and(dilated, np.logical_not(grown_mask))
            
            if not np.any(candidates):
                break  # Sem novos candidatos
            
            # Para cada candidato, verificar similaridade
            new_pixels = np.zeros_like(grown_mask)
            
            # Coordenadas dos candidatos
            candidate_coords = np.where(candidates)
            
            # Limitar número de candidatos para eficiência
            max_candidates = 10000
            if len(candidate_coords[0]) > max_candidates:
                indices = np.random.choice(len(candidate_coords[0]), max_candidates, replace=False)
                y_coords = candidate_coords[0][indices]
                x_coords = candidate_coords[1][indices]
            else:
                y_coords = candidate_coords[0]
                x_coords = candidate_coords[1]
            
            added_pixels = 0
            for i in range(len(y_coords)):
                y, x = y_coords[i], x_coords[i]
                
                # Extrair cor deste pixel
                pixel_color = image[y, x]
                
                # Calcular similaridade de cor (distância euclidiana normalizada)
                color_distance = np.linalg.norm(pixel_color - reference_color) / np.sqrt(3 * 255**2)
                
                # Threshold adaptativo - vai ficando mais permissivo com o tempo
                adaptive_threshold = threshold * (1 + 0.05 * iteration)
                
                # Se for similar o suficiente, adicionar
                if color_distance < adaptive_threshold:
                    new_pixels[y, x] = 1
                    added_pixels += 1
            
            # Se poucos novos pixels foram adicionados, considerar parar
            if added_pixels < 10:
                break
                
            # Atualizar máscara
            grown_mask = np.logical_or(grown_mask, new_pixels)
        
        # Opcional: Preencher pequenos buracos na máscara final
        grown_mask = cv2.morphologyEx(
            grown_mask.astype(np.uint8), 
            cv2.MORPH_CLOSE, 
            np.ones((7, 7), np.uint8)
        )
        
        return grown_mask
    
    def _select_context_boxes(self, img_np, pos_mask, neg_mask,
                              img_w, img_h,
                              max_pos=10, max_neg=10,
                              pos_cov=0.75, neg_cov=0.75,
                              pos_sim_thr=1.8, neg_dist_thr=2.0):
        """
        Seleciona boxes positivas e negativas usando similaridade ao cluster positivo.
        - Positivas: z-dist <= pos_sim_thr
        - Negativas: z-dist >= neg_dist_thr (mais distantes do cluster positivo)
        Retorna (positive_boxes, negative_boxes) em formato YOLO [0,cx,cy,w,h].
        """
        # Estatísticas do cluster positivo (incluindo lesão se já combinada no pos_mask)
        ref_mu, ref_sigma = self._compute_cluster_feature_stats(img_np, pos_mask)
        if ref_mu is None:
            return [], []

        def zdist(feat):
            return float(np.linalg.norm((feat - ref_mu) / ref_sigma))

        # Candidatos positivos
        pos_candidates = self._generate_candidate_boxes_from_mask(
            pos_mask, img_w, img_h, min_coverage=pos_cov
        )
        scored_pos = []
        for c in pos_candidates:
            x1, y1, x2, y2 = c["rect"]
            feat = self._compute_box_features(img_np, x1, y1, x2, y2)
            if feat is None:
                continue
            dist = zdist(feat)
            if dist <= pos_sim_thr:
                scored_pos.append((dist, c))
        scored_pos.sort(key=lambda x: x[0])
        positive_boxes = [
            [0, c["cx"], c["cy"], c["w"], c["h"]]
            for dist, c in scored_pos[:max_pos]
        ]

        # Candidatos negativos: INDEPENDENTEMENTE do cluster negativo,
        # usamos distância alta ao cluster positivo e contidos em neg_mask
        neg_candidates = self._generate_candidate_boxes_from_mask(
            neg_mask, img_w, img_h, min_coverage=neg_cov
        )
        scored_neg = []
        for c in neg_candidates:
            x1, y1, x2, y2 = c["rect"]
            feat = self._compute_box_features(img_np, x1, y1, x2, y2)
            if feat is None:
                continue
            dist = zdist(feat)
            if dist >= neg_dist_thr:
                scored_neg.append((dist, c))
        # Ordena decrescente (mais distante = mais negativo)
        scored_neg.sort(key=lambda x: x[0], reverse=True)
        negative_boxes = [
            [0, c["cx"], c["cy"], c["w"], c["h"]]
            for dist, c in scored_neg[:max_neg]
        ]

        # Fallback se vazio: amostra simples
        if len(positive_boxes) == 0:
            positive_boxes = self._sample_boxes_from_mask((pos_mask > 0), img_w, img_h, max_samples=max_pos)
        if len(negative_boxes) == 0:
            negative_boxes = self._sample_boxes_from_mask((neg_mask > 0), img_w, img_h, max_samples=max_neg)

        return positive_boxes, negative_boxes
    
    def _segment_feature_vector(self, img_np, img_lab, gray, seg_mask):
        """
        Vetor de características robusto do superpixel:
        [L*, a*, b* (Lab) médios, textura (Laplaciano std), luminância (L*), contraste (std gray)]
        Todos normalizados aproximadamente para [0,1].
        """
        area = np.sum(seg_mask)
        if area == 0:
            return np.zeros(6, dtype=np.float32)

        L = img_lab[:, :, 0][seg_mask]
        A = img_lab[:, :, 1][seg_mask]
        B = img_lab[:, :, 2][seg_mask]

        Lm = np.mean(L) / 255.0
        Am = (np.mean(A) - 128.0) / 128.0
        Bm = (np.mean(B) - 128.0) / 128.0

        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        tex = float(np.std(lap[seg_mask])) / 50.0
        lum = Lm
        contrast = float(np.std(gray[seg_mask])) / 50.0

        return np.array([Lm, Am, Bm, tex, lum, contrast], dtype=np.float32)

    def _build_segment_adjacency(self, segments):
        """
        Grafo de adjacência 4-conectado entre superpixels.
        """
        h, w = segments.shape
        adj = {int(s): set() for s in np.unique(segments)}
        right_diff = segments[:, :-1] != segments[:, 1:]
        down_diff = segments[:-1, :] != segments[1:, :]

        ys, xs = np.where(right_diff)
        for y, x in zip(ys, xs):
            a = int(segments[y, x]); b = int(segments[y, x+1])
            adj[a].add(b); adj[b].add(a)

        ys, xs = np.where(down_diff)
        for y, x in zip(ys, xs):
            a = int(segments[y, x]); b = int(segments[y+1, x])
            adj[a].add(b); adj[b].add(a)

        return adj

    def _grow_positive_context_by_superpixels(self, img_np, lesion_mask, n_segments=240,
                                         ring_width_px=14, max_iters=12,
                                         base_threshold=1.8, neg_percentile=75):
        """
        Cresce UMA região positiva conectada ao anel ao redor das lesões.
        Em seguida, define a região negativa como o COMPLEMENTO (fora da lesão e fora das bordas pretas).
        Retorna (pos_mask, neg_mask, debug).
        """
        h, w = img_np.shape[:2]
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 1. Cria o anel de contexto: dilatação menos a própria lesão
        ring_kernel = np.ones((max(3, ring_width_px), max(3, ring_width_px)), np.uint8)
        dilated = cv2.dilate(lesion_mask.astype(np.uint8), ring_kernel, iterations=1)
        ring = (dilated > 0) & (~lesion_mask.astype(bool))

        # 2. Segmentação (SLIC ou grid fallback)
        if slic is not None:
            segments = slic(img_np, n_segments=n_segments, compactness=12, sigma=1)
        else:
            segments = np.zeros((h, w), dtype=np.int32)
            grid = 12
            gh, gw = h // grid, w // grid
            for i in range(grid):
                for j in range(grid):
                    segments[i*gh:(i+1)*gh, j*gw:(j+1)*gw] = i*grid + j

        seg_ids = np.unique(segments)
        seg_masks, seg_feats = {}, {}
        for sid in seg_ids:
            sm = segments == sid
            if np.sum(sm) < 60:
                continue
            seg_masks[sid] = sm
            seg_feats[sid] = self._segment_feature_vector(img_np, img_lab, gray, sm)

        # 3. Sementes: superpixels que tocam o anel (vizinhança das lesões)
        seed_ids = []
        for sid, sm in seg_masks.items():
            ov = np.sum(sm & ring)
            if ov / (np.sum(sm) + 1e-6) >= 0.20 or (ov > 0 and not seed_ids):
                seed_ids.append(sid)
        if not seed_ids:
            # Fallback: usa todos os superpixels que tocam o anel
            for sid, sm in seg_masks.items():
                if np.any(sm & ring):
                    seed_ids.append(sid)
        if not seed_ids:
            # Se ainda não há sementes, retorna vazio
            return np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8), {"seed_count": 0}

        seed_feats = np.stack([seg_feats[sid] for sid in seed_ids], axis=0)
        mu = seed_feats.mean(axis=0)
        sigma = seed_feats.std(axis=0) + 1e-6

        def zdist(vec):
            z = (vec - mu) / sigma
            return float(np.linalg.norm(z))

        # z-dist de todos os superpixels válidos
        all_ids = list(seg_feats.keys())
        all_dists = np.array([zdist(seg_feats[s]) for s in all_ids])

        # limiar automático: mediana + 0.6*desvio (mais permissivo)
        auto_thr = float(np.median(all_dists) + 0.6 * np.std(all_dists))
        # garante que não seja mais restrito que base_threshold
        thr0 = max(base_threshold, auto_thr)

        adj = self._build_segment_adjacency(segments)

        def grow(thr):
            eligible = {sid for sid in all_ids if zdist(seg_feats[sid]) <= thr}
            blocked = {sid for sid, sm in seg_masks.items() if np.any(sm & lesion_mask)}
            eligible = eligible - blocked

            # BFS pelos elegíveis conectados às sementes (ANEL)
            positive_set = set()
            frontier = list(seed_ids)
            visited = set(frontier)
            while frontier:
                cur = frontier.pop()
                if cur in eligible:
                    positive_set.add(cur)
                    for nb in adj.get(int(cur), []):
                        if nb not in visited:
                            visited.add(nb)
                            frontier.append(nb)
            return positive_set

        # 4. Crescimento com limiar auto-ajustado
        positive_set = grow(thr0)

        # 5. Se ficou pequeno, relaxa mais o limiar
        if len(positive_set) < max(4, len(seed_ids)):
            positive_set = grow(thr0 * 1.25)

        # 6. Monta máscara positiva e faz limpeza morfológica
        pos_mask = np.zeros((h, w), np.uint8)
        for sid in positive_set:
            pos_mask |= seg_masks[sid].astype(np.uint8)

        pos_mask = cv2.morphologyEx(pos_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        pos_mask[lesion_mask > 0] = 0  # nunca incluir a própria lesão

        # Preencher buracos/gaps dentro da região positiva
        pos_mask = scipy.ndimage.binary_fill_holes(pos_mask).astype(np.uint8)

        # 7. Cria máscara de área válida (remove cantos pretos/escuros)
        valid_mask = (cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) > 8).astype(np.uint8)

        # 8. Negativo = complemento (bipartição)
        neg_mask = (valid_mask > 0).astype(np.uint8)
        neg_mask[pos_mask > 0] = 0
        neg_mask[lesion_mask > 0] = 0
        neg_mask = cv2.morphologyEx(neg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        dbg = {
            "seed_count": len(seed_ids),
            "pos_count": int(len(positive_set)),
            "thr_used": float(thr0)
        }
        return pos_mask, neg_mask, dbg

    def _extract_pixel_features(self, image, mask):
        """
        Extrai características de pixels de uma região da imagem.
        
        Args:
            image: Imagem como array numpy
            mask: Máscara binária indicando quais pixels considerar
            
        Returns:
            Tupla (features, coordinates)
        """
        # Limitar número de pixels para eficiência
        max_pixels = 10000
        nonzero = np.nonzero(mask)
        num_pixels = len(nonzero[0])
        
        if num_pixels > max_pixels:
            # Amostrar aleatoriamente
            indices = np.random.choice(num_pixels, max_pixels, replace=False)
            y_coords = nonzero[0][indices]
            x_coords = nonzero[1][indices]
        else:
            y_coords = nonzero[0]
            x_coords = nonzero[1]
        
        # Extrair características
        features = []
        coordinates = []
        
        for y, x in zip(y_coords, x_coords):
            # Cor RGB
            pixel = image[y, x]
            
            # Adicionar às listas
            features.append(pixel)
            coordinates.append((y, x))
        
        return np.array(features), np.array(coordinates)
    
    def _create_boxes_from_mask(self, mask, img_w, img_h, max_boxes=10):
        """
        Cria boxes YOLO a partir de uma máscara binária, garantindo que ficam contidas.
        """
        # Encontrar componentes conectados
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(mask * 255, connectivity=8)
        
        # Se não há componentes suficientes, criar grid de candidatos
        if num_components <= 1:
            return []
        
        boxes = []
        
        # Para cada componente conectado, criar boxes contidas
        for i in range(1, num_components):  # Ignorar background (0)
            # Extrair a máscara deste componente
            component_mask = (labels == i).astype(np.uint8)
            
            # Encontrar contornos da região
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            # Para cada contorno, encontrar o maior retângulo inscrito
            for contour in contours:
                # Obter bounding rect do contorno
                x, y, w, h = cv2.boundingRect(contour)
                
                # Ignorar regiões muito pequenas
                if w < 20 or h < 20 or w * h < 100:
                    continue
                    
                # Para regiões grandes, encontrar múltiplas boxes inscritas
                if w > img_w/5 and h > img_h/5:
                    # Dividir em sub-boxes (grid 2x2)
                    for row in range(2):
                        for col in range(2):
                            sub_x = x + col * (w // 2)
                            sub_y = y + row * (h // 2)
                            sub_w = w // 2
                            sub_h = h // 2
                            
                            # Verificar se a sub-box está contida na máscara
                            sub_mask = component_mask[sub_y:sub_y+sub_h, sub_x:sub_x+sub_w]
                            coverage = np.sum(sub_mask) / (sub_w * sub_h)
                            
                            # Só aceitar se pelo menos 70% da box está na máscara
                            if coverage < 0.7:
                                continue
                                
                            # Adicionar box no formato YOLO
                            cx = (sub_x + sub_w / 2) / img_w
                            cy = (sub_y + sub_h / 2) / img_h
                            width = sub_w / img_w
                            height = sub_h / img_h
                            
                            boxes.append([0, cx, cy, width, height])
                else:
                    # Para regiões menores, verificar cobertura
                    mask_region = component_mask[y:y+h, x:x+w]
                    coverage = np.sum(mask_region) / (w * h)
                    
                    # Só aceitar se pelo menos 80% da box está na máscara
                    if coverage < 0.8:
                        # Tentar encontrar uma box menor que tenha melhor cobertura
                        found_better = False
                        
                        # Tentar reduzir a box para melhorar cobertura
                        for scale in [0.9, 0.8, 0.7]:
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            
                            if new_w < 20 or new_h < 20:
                                continue
                                
                            # Tentar posicionar no centro
                            new_x = x + (w - new_w) // 2
                            new_y = y + (h - new_h) // 2
                            
                            new_region = component_mask[new_y:new_y+new_h, new_x:new_x+new_w]
                            new_coverage = np.sum(new_region) / (new_w * new_h)
                            
                            if new_coverage >= 0.8:
                                x, y, w, h = new_x, new_y, new_w, new_h
                                found_better = True
                                break
                        
                        if not found_better:
                            continue  # Pular esta box se não conseguimos melhorar cobertura
                    
                    # Adicionar box no formato YOLO
                    cx = (x + w / 2) / img_w
                    cy = (y + h / 2) / img_h
                    width = w / img_w
                    height = h / img_h
                    
                    boxes.append([0, cx, cy, width, height])
        
        # Limitar número de boxes
        if len(boxes) > max_boxes:
            # Garantir diversidade - selecionar boxes espalhadas
            selected_indices = np.linspace(0, len(boxes)-1, max_boxes, dtype=int)
            boxes = [boxes[i] for i in selected_indices]
        
        return boxes

    def _visualize_region_growing(self, image, lesion_mask, positive_regions, negative_regions):
        """
        Visualiza os resultados do region growing com boxes corretamente contidas e lesões originais.
        """
        # Importar patches se ainda não estiver disponível
        import matplotlib.patches as patches
        
        # Criar visualização colorida
        vis_image = np.copy(image)
        
        # Criar máscara de visualização
        overlay = np.zeros_like(image)
        
        # Cor verde para lesões
        overlay[lesion_mask > 0] = [0, 255, 0]
        
        # Cor azul para regiões positivas (contexto bom)
        overlay[positive_regions > 0] = [0, 0, 255]
        
        # Cor vermelha para regiões negativas (contexto ruim)
        overlay[negative_regions > 0] = [255, 0, 0]
        
        # Combinar imagem original com overlay
        alpha = 0.5
        vis_image = cv2.addWeighted(vis_image, 1.0, overlay, alpha, 0)
        
        # Criar figura
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Imagem original com lesões marcadas
        axes[0].imshow(image)
        axes[0].set_title("Imagem Original")
        axes[0].axis('off')
        
        # Desenhar contornos das lesões na imagem original
        contours, _ = cv2.findContours(lesion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            axes[0].add_patch(rect)
        
        # Visualização dos resultados
        axes[1].imshow(vis_image)
        axes[1].set_title("Region Growing + Clustering")
        axes[1].axis('off')
        
        # Dimensões da imagem
        img_h, img_w = image.shape[:2]
        
        # NOVO: Gerar caixas positivas ESTRITAMENTE contidas na região positiva
        positive_boxes = self._create_strictly_contained_boxes(
            positive_regions, img_w, img_h, min_coverage=0.95, max_boxes=15
        )
        
        # Desenhar caixas positivas
        for box in positive_boxes:
            _, cx, cy, w, h = box
            x1 = int((cx - w/2) * img_w)
            y1 = int((cy - h/2) * img_h)
            width = int(w * img_w)
            height = int(h * img_h)
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='cyan', facecolor='none', linestyle='-'
            )
            axes[1].add_patch(rect)
        
        # NOVO: Gerar caixas negativas ESTRITAMENTE contidas na região negativa
        negative_boxes = self._create_strictly_contained_boxes(
            negative_regions, img_w, img_h, min_coverage=0.95, max_boxes=10
        )
        
        # Desenhar caixas negativas
        for box in negative_boxes:
            _, cx, cy, w, h = box
            x1 = int((cx - w/2) * img_w)
            y1 = int((cy - h/2) * img_h)
            width = int(w * img_w)
            height = int(h * img_h)
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='magenta', facecolor='none', linestyle='--'
            )
            axes[1].add_patch(rect)
        
        # NOVO: Desenhar as lesões originais também na visualização de contexto
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            axes[1].add_patch(rect)
        
        # Legenda atualizada
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Lesão Original'),
            Patch(facecolor='blue', label='Contexto Positivo'),
            Patch(facecolor='red', label='Contexto Negativo'),
            Patch(facecolor='none', edgecolor='yellow', label='Lesões Originais (Boxes)'),
            Patch(facecolor='none', edgecolor='cyan', label='Boxes Positivas'),
            Patch(facecolor='none', edgecolor='magenta', label='Boxes Negativas')
        ]
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize='small')
        
        # Salvar visualização
        debug_dir = 'debug_region_growing'
        os.makedirs(debug_dir, exist_ok=True)
        plt.tight_layout()
        filename = f"region_growing_{len(positive_boxes)}p_{len(negative_boxes)}n_{np.random.randint(1000)}.png"
        plt.savefig(os.path.join(debug_dir, filename), dpi=150)
        plt.close(fig)
        
    def _create_strictly_contained_boxes(self, mask, img_w, img_h, min_coverage=0.85, max_boxes=12, box_size_factors=(0.18, 0.13, 0.09)):
        """
        Gera boxes YOLO estritamente contidas na máscara fornecida, permitindo sobreposição moderada.
        - As boxes nunca podem sair da máscara.
        - Permite sobreposição de até 40% entre boxes.
        - Garante boxes mesmo em regiões grandes ou fragmentadas.
        """
        mask = (mask > 0).astype(np.uint8)
        area = np.sum(mask)
        if area < 100:
            return []

        boxes = []
        used_mask = np.zeros_like(mask, dtype=np.uint8)
        stride_factors = [0.5, 0.4, 0.3]  # Quanto menor, mais sobreposição

        for box_size_factor, stride_factor in zip(box_size_factors, stride_factors):
            box_w = int(img_w * box_size_factor)
            box_h = int(img_h * box_size_factor)
            stride_x = int(box_w * stride_factor)
            stride_y = int(box_h * stride_factor)

            for y in range(0, img_h - box_h + 1, stride_y):
                for x in range(0, img_w - box_w + 1, stride_x):
                    window = mask[y:y+box_h, x:x+box_w]
                    if window.shape != (box_h, box_w):
                        continue
                    coverage = np.sum(window) / (box_w * box_h)
                    # Só aceita se a maior parte da box está dentro da máscara e não ultrapassa o limite
                    if coverage >= min_coverage:
                        # Garante que não sai da máscara (bordas)
                        if np.any(mask[y:y+box_h, x:x+box_w] == 0):
                            # Checa se a box está totalmente dentro da máscara delimitadora
                            continue
                        # Permite sobreposição moderada (até 40%)
                        overlap = np.sum(used_mask[y:y+box_h, x:x+box_w]) / (box_w * box_h)
                        if overlap > 0.4:
                            continue
                        # Marca região como usada
                        used_mask[y:y+box_h, x:x+box_w] = 1
                        # YOLO format
                        cx = (x + box_w / 2) / img_w
                        cy = (y + box_h / 2) / img_h
                        width = box_w / img_w
                        height = box_h / img_h
                        boxes.append([0, cx, cy, width, height])
                        if len(boxes) >= max_boxes:
                            return boxes

        # Fallback: se não encontrou boxes suficientes, amostra pontos aleatórios
        if len(boxes) < max_boxes:
            points = np.argwhere(mask)
            np.random.shuffle(points)
            for y, x in points:
                box_w = int(img_w * box_size_factors[-1])
                box_h = int(img_h * box_size_factors[-1])
                x1 = max(0, x - box_w // 2)
                y1 = max(0, y - box_h // 2)
                x2 = min(img_w, x1 + box_w)
                y2 = min(img_h, y1 + box_h)
                window = mask[y1:y2, x1:x2]
                if window.shape != (box_h, box_w):
                    continue
                coverage = np.sum(window) / (box_w * box_h)
                if coverage >= min_coverage:
                    cx = (x1 + box_w / 2) / img_w
                    cy = (y1 + box_h / 2) / img_h
                    width = box_w / img_w
                    height = box_h / img_h
                    boxes.append([0, cx, cy, width, height])
                    if len(boxes) >= max_boxes:
                        break

        return boxes
        
    def _create_box_from_segment(self, segment, img_w, img_h):
        """
        Cria uma bounding box a partir de um segmento
        
        Args:
            segment: Dicionário com informações do segmento
            img_w: Largura da imagem
            img_h: Altura da imagem
            
        Returns:
            Box no formato YOLO [cls_idx, cx, cy, w, h] ou None se inválida
        """
        # Obter máscara do segmento
        mask = segment['mask']
        y_indices, x_indices = np.where(mask)
        
        # Se poucos pixels, retornar None
        if len(y_indices) < 20 or len(x_indices) < 20:
            return None
            
        # Calcular bounding box
        x1, x2 = np.min(x_indices), np.max(x_indices)
        y1, y2 = np.min(y_indices), np.max(y_indices)
        
        # Criar box no formato YOLO [cls_idx, cx, cy, w, h]
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        
        # Pular se box é muito grande ou muito pequena
        if width > 0.5 or height > 0.5 or width < 0.05 or height < 0.05:
            return None
            
        return [0, cx, cy, width, height]

    def _extract_instance_features(self, img_np, mask_np):
        """
        Extrai características das instâncias existentes (lesões) na imagem
        de forma agnóstica ao domínio.
        """
        features = []
        
        # Encontrar componentes conectados (instâncias)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np * 255, connectivity=8)
        
        for i in range(1, num_labels):  # Pular label 0 (background)
            x, y, w, h, area = stats[i]
            
            # Ignorar instâncias muito pequenas
            if w < 10 or h < 10:
                continue
                
            # Extrair região da instância
            instance_mask = (labels == i).astype(np.uint8)
            instance_region = img_np.copy()
            
            # Aplicar máscara (zerar pixels fora da instância)
            for c in range(min(3, instance_region.shape[2])):
                instance_region[:, :, c] = instance_region[:, :, c] * instance_mask
            
            # 1. Características de cor (histograma normalizado)
            color_hist = []
            for c in range(min(3, instance_region.shape[2])):
                hist = cv2.calcHist([instance_region], [c], instance_mask * 255, [8], [0, 256])
                color_hist.extend(hist.flatten() / max(1, np.sum(hist)))
            
            # 2. Características de textura (LBP simplificado ou gradientes)
            if instance_region.shape[0] > 10 and instance_region.shape[1] > 10:
                try:
                    # Converter para escala de cinza
                    if len(instance_region.shape) == 3:
                        gray = cv2.cvtColor(instance_region, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = instance_region
                    
                    # Calcular gradientes como proxy para textura
                    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
                    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
                    grad_mag = np.sqrt(gx**2 + gy**2)
                    grad_dir = np.arctan2(gy, gx)
                    
                    # Histograma de gradientes (simplificado)
                    hist_mag = np.histogram(grad_mag[instance_mask > 0], bins=8, range=(0, np.max(grad_mag)))[0]
                    hist_dir = np.histogram(grad_dir[instance_mask > 0], bins=8, range=(-np.pi, np.pi))[0]
                    
                    # Normalizar
                    if np.sum(hist_mag) > 0:
                        hist_mag = hist_mag / np.sum(hist_mag)
                    if np.sum(hist_dir) > 0:
                        hist_dir = hist_dir / np.sum(hist_dir)
                    
                    texture_features = np.concatenate([hist_mag, hist_dir])
                except:
                    # Fallback para medidas básicas
                    std_dev = np.std(gray[instance_mask > 0]) if np.sum(instance_mask) > 0 else 0
                    texture_features = np.array([std_dev])
            else:
                texture_features = np.array([0])
            
            # 3. Características geométricas normalizadas
            aspect_ratio = w / h if h > 0 else 1
            roundness = 4 * np.pi * area / ((2*w + 2*h)**2) if (w > 0 and h > 0) else 0
            
            # Combinar todas as características
            combined_features = np.concatenate([
                np.array(color_hist),
                texture_features,
                np.array([aspect_ratio, roundness])
            ])
            
            features.append(combined_features)
        
        return features

    def _calculate_region_score(self, region, reference_features, is_instance_region=False):
        """
        Calcula um score de qualidade para uma região baseado em sua similaridade
        com as instâncias de referência e outros critérios gerais.
        """
        if len(reference_features) == 0:
            return 0.5  # Score neutro se não houver referências
        
        # 1. Extrair as mesmas características da região
        region_mask = np.ones((region.shape[0], region.shape[1]), dtype=np.uint8)
        region_features = self._extract_instance_features(region, region_mask)
        
        if not region_features:
            # Se não conseguir extrair características, usar medidas básicas
            score = 0.0
            
            # Visibilidade: evitar áreas muito escuras ou muito claras
            mean_intensity = np.mean(region)
            if mean_intensity > 20 and mean_intensity < 235:  # Não muito escuro nem claro
                score += 0.5
            
            # Contraste: prefere áreas com algum contraste
            contrast = np.std(region)
            score += min(0.5, contrast / 50)  # Normalizado, máximo 0.5
            
            return score
        
        # 2. Calcular similaridade com cada referência
        similarities = []
        for ref_feature in reference_features:
            for region_feature in region_features:
                # Garantir mesmo comprimento (pode precisar de padding)
                min_len = min(len(ref_feature), len(region_feature))
                similarity = cosine_similarity(
                    ref_feature[:min_len].reshape(1, -1),
                    region_feature[:min_len].reshape(1, -1)
                )[0, 0]
                similarities.append(similarity)
        
        # Para regiões positivas, queremos baixa similaridade com lesões
        # Para regiões negativas, queremos alta similaridade com lesões
        if is_instance_region:
            # Menor similaridade = melhor para negativos
            score = 1.0 - (np.mean(similarities) if similarities else 0.5)
        else:
            # Maior similaridade = melhor para positivos
            score = np.mean(similarities) if similarities else 0.5
        
        # Critérios adicionais gerais
        
        # Visibilidade: evitar áreas muito escuras
        mean_intensity = np.mean(region)
        if mean_intensity < 30:  # Muito escuro
            score *= 0.5
        
        # Nitidez/Foco: preferir áreas nítidas
        laplacian_var = cv2.Laplacian(cv2.cvtColor(region, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        score *= min(1.0, 0.5 + laplacian_var / 1000)  # Normalizado
        
        return score

    def _sample_boxes_from_mask(self, mask, img_w, img_h, max_samples=15, box_size_factor=0.12):
        """
        Amostra boxes de uma máscara binária.
        """
        boxes = []
        points = np.argwhere(mask)
        
        if len(points) == 0:
            return boxes
        
        # Amostrar pontos aleatoriamente
        indices = np.random.choice(len(points), min(max_samples * 2, len(points)), replace=False)
        
        for idx in indices:
            y, x = points[idx]
            
            # Tamanho proporcional à imagem
            box_size = int(min(img_w, img_h) * box_size_factor)
            half_size = box_size // 2
            
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(img_w, x + half_size)
            y2 = min(img_h, y + half_size)
            
            # Verificar tamanho mínimo
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
                
            # Criar box no formato YOLO [cls_idx, cx, cy, w, h]
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            
            boxes.append([0, cx, cy, width, height])
            
            # Limitar número de boxes
            if len(boxes) >= max_samples:
                break
        
        return boxes
    
    def evaluate_region_quality(self, image, box, reference_lesions=None):
        """
        Avalia a qualidade de uma região para colagem de lesões, retornando um score de 0-1.
        
        Args:
            image: Imagem PIL ou tensor [C,H,W]
            box: Coordenadas [x1,y1,x2,y2] ou [cx,cy,w,h,is_yolo]
            reference_lesions: Lista opcional de features de lesões para comparação
            
        Returns:
            Score de qualidade (0-1) e dict com scores individuais
        """
        # Converter para tensor se for PIL
        if isinstance(image, Image.Image):
            image = self.to_tensor(image).unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Padronizar formato da box para [x1,y1,x2,y2]
        if len(box) == 5 and box[4]:  # é YOLO
            cx, cy, w, h = box[:4]
            img_size = image.shape[-1]
            x1 = int((cx - w/2) * img_size)
            y1 = int((cy - h/2) * img_size)
            x2 = int((cx + w/2) * img_size)
            y2 = int((cy + h/2) * img_size)
        else:
            x1, y1, x2, y2 = [int(c) for c in box[:4]]
        
        # Garantir que a box está dentro dos limites da imagem
        img_h, img_w = image.shape[-2], image.shape[-1]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0, {"error": "Box inválida"}
        
        # Extrair região
        device = image.device
        region = image[:, :, y1:y2, x1:x2]
        
        # 1. Critério de tamanho adequado
        min_size, max_size = 16, img_w/2
        size_score = 1.0
        region_size = max(x2-x1, y2-y1)
        
        if region_size < min_size:
            size_score = 0.1
        elif region_size > max_size:
            size_score = 0.5
        
        # 2. Critério de luminância
        with torch.no_grad():
            luminance = torch.mean(region, dim=1)  # Média sobre canais
            mean_luminance = torch.mean(luminance)
            
            # Muito escuro ou muito claro é ruim
            if mean_luminance < 0.1:  # Muito escuro
                lum_score = 0.2
            elif mean_luminance > 0.9:  # Muito claro
                lum_score = 0.5
            else:
                # Escala linear entre 0.1-0.9
                lum_score = 1.0 - 0.6 * abs((mean_luminance - 0.5) / 0.4)
        
        # 3. Critério de similaridade com lesões de referência
        if reference_lesions is not None and len(reference_lesions) > 0:
            # Extrair características desta região
            with torch.no_grad():
                region_features = self.extract_region_features(image, torch.tensor([[x1, y1, x2, y2]], device=device))
                
                # Calcular similaridade com referências
                similarities = torch.nn.functional.cosine_similarity(
                    region_features.unsqueeze(1), 
                    reference_lesions.unsqueeze(0), 
                    dim=2
                )
                max_similarity = torch.max(similarities).item()
                
                # Escala para score (0.5 para neutro, até 1.0 para muito similar)
                similarity_score = 0.5 + max(0, min(0.5, max_similarity - 0.3))
        else:
            similarity_score = 0.5  # Neutro se não há referências
        
        # Combina scores com pesos
        final_score = 0.3 * size_score + 0.3 * lum_score + 0.4 * similarity_score
        
        # Detalhes para debug
        details = {
            "size_score": float(size_score),
            "luminance_score": float(lum_score),
            "similarity_score": float(similarity_score) if reference_lesions is not None else None,
            "region_size": float(region_size),
            "mean_luminance": float(mean_luminance)
        }
        
        return float(final_score), details
    
    # Adicionar após a função region_grow_background_boxes, aproximadamente linha 530

def adaptive_context_refinement(self, initial_boxes, images, features_extractor, refinement_steps=3):
    """
    Refinamento adaptativo de contextos baseado em feedback do recomendador
    para melhorar a qualidade dos contextos positivos e negativos.
    
    Args:
        initial_boxes: Lista inicial de boxes [positivas, negativas]
        images: Tensor de imagens
        features_extractor: Extrator de características
        refinement_steps: Número de iterações de refinamento
        
    Returns:
        Tupla (refined_positive_boxes, refined_negative_boxes)
    """
    positive_boxes, negative_boxes = initial_boxes
    
    # Converter para tensores se ainda não forem
    if not isinstance(positive_boxes, torch.Tensor) and positive_boxes:
        positive_boxes = torch.tensor(positive_boxes)
    if not isinstance(negative_boxes, torch.Tensor) and negative_boxes:
        negative_boxes = torch.tensor(negative_boxes)
    
    # Garantir que temos pelo menos alguns boxes para começar
    if len(positive_boxes) == 0 or len(negative_boxes) == 0:
        return positive_boxes, negative_boxes
    
    device = next(features_extractor.parameters()).device
    
    # Extrair características para todas as boxes
    with torch.no_grad():
        # Converter formato YOLO para xyxy
        if positive_boxes.size(0) > 0:
            pos_xyxy = self._yolo_to_xyxy(positive_boxes)
            pos_features = features_extractor(images, pos_xyxy)
        else:
            pos_features = torch.tensor([])
            
        if negative_boxes.size(0) > 0:
            neg_xyxy = self._yolo_to_xyxy(negative_boxes)
            neg_features = features_extractor(images, neg_xyxy)
        else:
            neg_features = torch.tensor([])
    
    # Refinamento iterativo
    for step in range(refinement_steps):
        # Passo 1: Identificar positivos ambíguos (próximos de negativos)
        if pos_features.size(0) > 0 and neg_features.size(0) > 0:
            # Calcula similaridade entre positivos e negativos
            sim_matrix = torch.mm(pos_features, neg_features.t())
            
            # Encontra positivos que são muito similares a negativos
            ambiguous_pos = torch.any(sim_matrix > 0.8, dim=1)
            
            # Atualiza positivos removendo ambíguos
            if torch.sum(ambiguous_pos) > 0 and torch.sum(ambiguous_pos) < len(ambiguous_pos):
                positive_boxes = positive_boxes[~ambiguous_pos]
                pos_features = pos_features[~ambiguous_pos]
        
        # Passo 2: Identificar negativos ambíguos (próximos de positivos)
        if pos_features.size(0) > 0 and neg_features.size(0) > 0:
            # Similaridade entre negativos e positivos
            sim_matrix = torch.mm(neg_features, pos_features.t())
            
            # Encontra negativos que são muito similares a positivos
            ambiguous_neg = torch.any(sim_matrix > 0.8, dim=1)
            
            # Atualiza negativos removendo ambíguos
            if torch.sum(ambiguous_neg) > 0 and torch.sum(ambiguous_neg) < len(ambiguous_neg):
                negative_boxes = negative_boxes[~ambiguous_neg]
                neg_features = neg_features[~ambiguous_neg]
    
    # Garantir que ainda temos boxes suficientes
    min_boxes = 5
    if len(positive_boxes) < min_boxes:
        # Complementar com a abordagem baseada em distância
        extra_pos = self._grid_based_boxes(images, None)[0]
        if isinstance(extra_pos, list) and extra_pos:
            extra_pos = torch.tensor(extra_pos)
            if len(extra_pos) > 0:
                positive_boxes = torch.cat([positive_boxes, extra_pos[:min_boxes-len(positive_boxes)]])
    
    if len(negative_boxes) < min_boxes:
        # Complementar com a abordagem baseada em distância
        _, extra_neg = self._grid_based_boxes(images, None)
        if isinstance(extra_neg, list) and extra_neg:
            extra_neg = torch.tensor(extra_neg)
            if len(extra_neg) > 0:
                negative_boxes = torch.cat([negative_boxes, extra_neg[:min_boxes-len(negative_boxes)]])
    
    return positive_boxes, negative_boxes

def _yolo_to_xyxy(self, boxes):
    """Converte formato YOLO para xyxy para uso interno"""
    if boxes.size(0) == 0:
        return boxes
        
    # Ignorar class_id na primeira coluna se houver
    if boxes.size(1) > 4:  # tem class_id
        coords = boxes[:, 1:5]
    else:
        coords = boxes
        
    # Extrair componentes (cx, cy, w, h)
    cx, cy, w, h = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    
    # Calcular coordenadas xyxy
    x1 = (cx - w/2) * 640
    y1 = (cy - h/2) * 640
    x2 = (cx + w/2) * 640
    y2 = (cy + h/2) * 640
    
    return torch.stack([x1, y1, x2, y2], dim=1)

def mine_similar_contexts(self, image, mask, lesion_boxes, max_contexts=20):
    """
    Extrai automaticamente contextos positivos e negativos baseados na similaridade
    com os contextos das lesões existentes.
    
    Args:
        image: Imagem PIL
        mask: Máscara binária
        lesion_boxes: Lista de boxes de lesões detectadas
        max_contexts: Número máximo de contextos a extrair
        
    Returns:
        Tupla (contextos_positivos, contextos_negativos)
    """
    img_w, img_h = image.size
    img_np = np.array(image)
    device = next(self.texture_extractor.parameters()).device
    
    # 1. Extrai características dos contextos das lesões existentes
    lesion_context_features = []
    lesion_positions = []
    for box in lesion_boxes:
        _, cx, cy, w, h = box
        # Converte para coordenadas absolutas
        abs_cx, abs_cy = cx * img_w, cy * img_h
        abs_w, abs_h = w * img_w, h * img_h
        
        # Expande para capturar o contexto
        exp_ratio = 1.8
        ctx_x1 = max(0, int(abs_cx - abs_w*exp_ratio/2))
        ctx_y1 = max(0, int(abs_cy - abs_h*exp_ratio/2))
        ctx_x2 = min(img_w, int(abs_cx + abs_w*exp_ratio/2))
        ctx_y2 = min(img_h, int(abs_cy + abs_h*exp_ratio/2))
        
        # Extrai região
        if ctx_x2 > ctx_x1 and ctx_y2 > ctx_y1:
            ctx_region = img_np[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
            if ctx_region.size == 0:
                continue
            
            # Converte para tensor e extrai características
            ctx_tensor = torch.from_numpy(ctx_region).permute(2, 0, 1).float() / 255.0
            ctx_tensor = ctx_tensor.unsqueeze(0).to(device)
            
            # Extrai características usando os extratores pré-definidos
            with torch.no_grad():
                texture_feat = self.texture_extractor(ctx_tensor).cpu().numpy().flatten()
                color_feat = self.color_extractor(ctx_tensor).cpu().numpy().flatten()
                features = np.concatenate([texture_feat, color_feat])
                
            lesion_context_features.append(features)
            lesion_positions.append((abs_cx/img_w, abs_cy/img_h))  # Posição normalizada
    
    if not lesion_context_features:
        return [], []
        
    # 2. Dividir imagem em grid e avaliar similaridade
    grid_size = 12  # Grid mais fino
    cell_w, cell_h = img_w / grid_size, img_h / grid_size
    
    candidate_regions = []
    candidate_positions = []
    candidate_features = []
    
    # Gerar grid de regiões candidatas
    for i in range(grid_size):
        for j in range(grid_size):
            # Coordenadas da célula
            x1 = int(i * cell_w)
            y1 = int(j * cell_h)
            x2 = int((i + 1) * cell_w)
            y2 = int((j + 1) * cell_h)
            
            # Ignorar células muito pequenas
            if x2 <= x1 or y2 <= y1 or x2 > img_w or y2 > img_h:
                continue
                
            # Extrair região
            try:
                region = img_np[y1:y2, x1:x2]
            except:
                continue
                
            # Verificar se a célula tem tamanho adequado
            if region.shape[0] < 20 or region.shape[1] < 20:
                continue
            
            # Extrair características
            region_tensor = torch.from_numpy(region).permute(2, 0, 1).float() / 255.0
            region_tensor = region_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                texture_feat = self.texture_extractor(region_tensor).cpu().numpy().flatten()
                color_feat = self.color_extractor(region_tensor).cpu().numpy().flatten()
                region_features = np.concatenate([texture_feat, color_feat])
            
            # Guardar candidato
            candidate_features.append(region_features)
            candidate_regions.append((x1, y1, x2, y2))
            candidate_positions.append(((x1 + cell_w/2)/img_w, (y1 + cell_h/2)/img_h))
    
    if not candidate_regions:
        return [], []
    
    # 3. Calcular similaridade entre candidatos e contextos de lesões
    from scipy.spatial.distance import cosine
    similarities = []
    for cand_feat in candidate_features:
        # Calcula similaridade com todos os contextos de lesões
        sim_scores = []
        for lesion_feat in lesion_context_features:
            sim = 1.0 - cosine(cand_feat, lesion_feat)  # 1.0 = idêntico, 0.0 = totalmente diferente
            sim_scores.append(sim)
        similarities.append(max(sim_scores))  # Usa a maior similaridade
    
    # 4. Classificar candidatos
    positive_boxes = []
    negative_boxes = []
    
    pos_threshold = 0.7  # Limiar para contexto positivo
    neg_threshold = 0.3  # Limiar para contexto negativo
    
    # Ordenar candidatos por similaridade
    sorted_indices = np.argsort(similarities)[::-1]  # Ordem decrescente
    
    pos_count = 0
    neg_count = 0
    
    for idx in sorted_indices:
        sim_score = similarities[idx]
        x1, y1, x2, y2 = candidate_regions[idx]
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        
        # Evitar sobreposição com lesões existentes
        overlap = False
        for les_cx, les_cy in lesion_positions:
            dist = np.sqrt((cx - les_cx)**2 + (cy - les_cy)**2)
            if dist < 0.1:  # Distância normalizada
                overlap = True
                break
        
        if overlap:
            continue
        
        if sim_score > pos_threshold and pos_count < max_contexts:
            positive_boxes.append([0, cx, cy, width, height])
            pos_count += 1
        elif sim_score < neg_threshold and neg_count < max_contexts:
            negative_boxes.append([0, cx, cy, width, height])
            neg_count += 1
            
        # Limitar número total de contextos
        if pos_count + neg_count >= 2 * max_contexts:
            break
    
    return positive_boxes, negative_boxes

def custom_collate(batch):
    """
    Retorna:
      images: [B,3,640,640]
      masks:  [B,1,640,640]
      all_boxes: [M,5] no formato YOLO [cls,cx,cy,w,h]
      all_labels: [M] rótulos (0=neg,1=lesão,2=pos)
      all_img_indices: [M] índice da imagem no batch
    """
    import torch
    images, masks, box_data = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)

    all_boxes = []
    all_labels = []
    all_img_indices = []
    for i, bd in enumerate(box_data):
        if bd.numel() == 0:
            continue
        # bd: [N,6] -> YOLO coords [0:5], label em [-1]
        all_boxes.append(bd[:, 0:5])
        all_labels.append(bd[:, 5].long())
        all_img_indices.append(torch.full((bd.size(0),), i, dtype=torch.long))

    if all_boxes:
        all_boxes = torch.cat(all_boxes, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_img_indices = torch.cat(all_img_indices, dim=0)
    else:
        all_boxes = torch.zeros((0, 5))
        all_labels = torch.zeros((0,), dtype=torch.long)
        all_img_indices = torch.zeros((0,), dtype=torch.long)

    return images, masks, all_boxes, all_labels, all_img_indices

def mosaic_augmentation(dataset, indices, image_size=(640, 640)):
    """
    Implementação de mosaic augmentation para datasets de segmentação.
    
    Args:
        dataset: Dataset de origem
        indices: Índices das 4 imagens a serem combinadas
        image_size: Tamanho final da imagem mosaico
    
    Returns:
        imagem mosaico, máscara mosaico, boxes de lesão
    """
    assert len(indices) == 4, "Mosaic augmentation requer exatamente 4 imagens"
    
    # Criar imagem e máscara mosaico em branco
    mosaic_img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    mosaic_mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    
    # Definir ponto central do mosaico (com jitter)
    center_x = int(random.uniform(image_size[0] // 3, image_size[0] * 2 // 3))
    center_y = int(random.uniform(image_size[1] // 3, image_size[1] * 2 // 3))
    
    # Definir as 4 regiões
    regions = [
        (0, 0, center_x, center_y),                            # top-left
        (center_x, 0, image_size[0], center_y),                # top-right
        (0, center_y, center_x, image_size[1]),                # bottom-left
        (center_x, center_y, image_size[0], image_size[1])     # bottom-right
    ]
    
    # Lista para armazenar boxes de lesões
    lesion_boxes = []
    
    # Processar cada imagem
    for i, idx in enumerate(indices):
        # Carregar imagem e máscara
        img_path = dataset.lesion_image_paths[idx]
        mask_path = dataset.lesion_masks_paths[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Redimensionar
        img = dataset.resize_transform(img)
        mask = dataset.resize_transform(mask)
        
        # Converter para numpy
        img_np = np.array(img)
        mask_np = np.array(mask)
        
        # Definir região de destino
        x1, y1, x2, y2 = regions[i]
        region_w, region_h = x2 - x1, y2 - y1
        
        # Redimensionar para a região
        img_region = cv2.resize(img_np, (region_w, region_h))
        mask_region = cv2.resize(mask_np, (region_w, region_h))
        
        # Colocar na imagem mosaico
        mosaic_img[y1:y2, x1:x2] = img_region
        mosaic_mask[y1:y2, x1:x2] = mask_region
        
        # Extrair boxes de lesão para esta região
        region_boxes = extract_boxes_for_region(mask_region, x1, y1, region_w, region_h, image_size[0], image_size[1])
        lesion_boxes.extend(region_boxes)
    
    # Converter de volta para PIL
    mosaic_img_pil = Image.fromarray(mosaic_img)
    mosaic_mask_pil = Image.fromarray(mosaic_mask)
    
    return mosaic_img_pil, mosaic_mask_pil, lesion_boxes

def extract_boxes_for_region(mask, x_offset, y_offset, region_w, region_h, img_w, img_h):
    """
    Extrai boxes de lesão de uma região do mosaico e as normaliza para coordenadas YOLO.
    """
    mask_np = (mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    
    boxes = []
    for i in range(1, num_labels):  # Pula o background (índice 0)
        x, y, w, h, _ = stats[i]
        
        # Converter para coordenadas globais
        global_x = x + x_offset
        global_y = y + y_offset
        
        # Normalizar para coordenadas YOLO
        x_center = (global_x + w / 2) / img_w
        y_center = (global_y + h / 2) / img_h
        width = w / img_w
        height = h / img_h
        
        boxes.append([0, x_center, y_center, width, height])  # class_idx=0
    
    return boxes