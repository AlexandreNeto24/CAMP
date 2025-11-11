import os
import sys
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Configuração de caminhos para garantir importações corretas
project_root = Path(__file__).parent.parent.parent  # src/paste/ → src/ → project_root
sys.path.insert(0, str(project_root))

class AttentionDataset(Dataset):
    """
    Dataset especializado para treinamento do AttentionRecommender.
    
    Esta classe:
    1. Carrega imagens e máscaras de lesões
    2. Extrai patches de lesão para usar como queries
    3. Gera mapas de calor gaussianos centrados nas posições originais
    4. Fornece pares (imagem, patch) → mapa_gaussiano para treinamento
    
    Args:
        root_path: Caminho para o diretório raiz dos dados
        image_size: Tamanho padrão das imagens
        patch_size: Tamanho fixo para redimensionar patches (None = manter original)
        transform: Transformações adicionais
        hyp: Hiperparâmetros para augmentação
        apply_aug: Se deve aplicar augmentação (True para treino, False para val)
        split_ratio: Proporção de validação (usado se split_mode='ratio')
        split_mode: Como dividir os dados ('ratio', 'preset', 'none')
        split: Qual split usar ('train', 'val', 'all') - usado se split_mode='preset'
    """
    def __init__(self, root_path, image_size=(512, 512), patch_size=None, 
                 transform=None, hyp=None, apply_aug=False,
                 split_ratio=0.2, split_mode='none', split='all', p_remove=0.0):
        self.root_path = root_path
        self.image_size = image_size
        self.patch_size = patch_size
        self.transform = transform
        self.hyp = hyp
        self.apply_aug = apply_aug
        # probabilidade de remover (inpaint/blur/local) a região do patch na imagem alvo
        # default=0.0 mantém comportamento antigo
        self.p_remove = float(p_remove)

        # # Caminho para imagens
        # img_dirs = [
        #     os.path.join(self.root_path, 'images'),
        #     os.path.join(self.root_path, 'airplane', 'images')
        # ]
        # img_dir = None
        # for i_dir in img_dirs:
        #     if os.path.exists(i_dir):
        #         img_dir = i_dir
        #         break

        # # Calcula mean/std automaticamente
        # means = []
        # stds = []
        # image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # for fname in image_files:
        #     img = np.array(Image.open(os.path.join(img_dir, fname)).convert('RGB')) / 255.0
        #     means.append(np.mean(img, axis=(0, 1)))
        #     stds.append(np.std(img, axis=(0, 1)))
        # mean = np.mean(means, axis=0)
        # std = np.mean(stds, axis=0)

        # Transforms
        self.to_tensor = transforms.ToTensor()
        # self.normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())
        self.resize_img = transforms.Resize(self.image_size)
        
        # Configurar transformação para augmentação se necessário
        self._setup_augmentation()
        
        # Carregar caminhos para imagens e máscaras
        self.image_paths, self.mask_paths = self._load_data_paths()
        
        # Dividir em treino/validação conforme configuração
        if split_mode == 'ratio':
            self._split_by_ratio(split_ratio, split)
        elif split_mode == 'preset':
            self._load_preset_split(split)
    
    def _setup_augmentation(self):
        """
        Configura transformações para data augmentation
        """
        # if self.apply_aug and self.hyp:
        if self.apply_aug:
            try:
                import albumentations as A
                self.aug_transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.20, rotate_limit=30, p=0.6),
                    # A.GridDistortion(p=0.4),
                    # A.ElasticTransform(alpha=2, sigma=80, alpha_affine=40, p=0.4),
                    A.RandomBrightnessContrast(p=0.7),
                    A.RandomGamma(p=0.5),
                    A.CLAHE(p=0.4),
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.4),
                    A.GaussNoise(var_limit=(10.0, 80.0), p=0.6),
                    A.Blur(blur_limit=7, p=0.3),
                    A.MedianBlur(blur_limit=7, p=0.2),
                    A.MotionBlur(blur_limit=7, p=0.2),
                    # A.CoarseDropout(max_holes=12, max_height=48, max_width=48, p=0.7),
                    # A.Cutout(num_holes=8, max_h_size=48, max_w_size=48, p=0.5),
                ])
            except ImportError:
                print("Albumentations não encontrado. Augmentação desativada.")
                self.aug_transform = None
        else:
            self.aug_transform = None
    
    def _load_data_paths(self):
        """
        Carrega os caminhos para imagens e máscaras, garantindo correspondência
        
        Returns:
            Tupla (imagem_paths, mask_paths)
        """
        # Procura nas estruturas de diretório comuns
        img_dirs = [
            os.path.join(self.root_path, 'images'),
            os.path.join(self.root_path, 'airplane', 'images')
        ]
        
        mask_dirs = [
            os.path.join(self.root_path, 'masks'),
            os.path.join(self.root_path, 'airplane', 'masks')
        ]
        
        # Encontra o primeiro diretório válido
        img_dir = None
        mask_dir = None
        
        for i_dir, m_dir in zip(img_dirs, mask_dirs):
            if os.path.exists(i_dir) and os.path.exists(m_dir):
                img_dir = i_dir
                mask_dir = m_dir
                break
        
        if not img_dir or not mask_dir:
            raise FileNotFoundError(f"Diretórios de imagens/máscaras não encontrados em {self.root_path}")
        
        # Lista todos os arquivos de imagem
        image_files = sorted([f for f in os.listdir(img_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        image_paths = []
        mask_paths = []
        
        # Garante que cada imagem tem uma máscara correspondente
        for img_file in image_files:
            mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)
        
        return image_paths, mask_paths
    
    def _split_by_ratio(self, split_ratio, split='train'):
        """
        Divide os dados em treino/validação usando proporção
        
        Args:
            split_ratio: Proporção para validação (ex: 0.2 = 20%)
            split: Qual split retornar ('train', 'val', 'all')
        """
        n = len(self.image_paths)
        indices = list(range(n))
        
        # Usa seed fixa para garantir splits consistentes
        random.seed(42)
        random.shuffle(indices)
        
        split_idx = int(n * (1 - split_ratio))
        
        if split == 'train':
            selected_indices = indices[:split_idx]
        elif split == 'val':
            selected_indices = indices[split_idx:]
        else:  # 'all'
            selected_indices = indices
        
        # Atualiza paths para conter apenas os do split selecionado
        self.image_paths = [self.image_paths[i] for i in selected_indices]
        self.mask_paths = [self.mask_paths[i] for i in selected_indices]
    
    def _load_preset_split(self, split='all'):
        """
        Carrega split predefinido (usando arquivos train.txt, val.txt)
        
        Args:
            split: Qual split carregar ('train', 'val', 'all')
        """
        if split == 'all':
            return  # Mantém todos os dados
        
        # Verifica se existem arquivos de split
        split_file = os.path.join(self.root_path, f'{split}.txt')
        
        if not os.path.exists(split_file):
            print(f"Arquivo de split {split_file} não encontrado. Usando todos os dados.")
            return
        
        # Carrega nomes de arquivo do split
        with open(split_file, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]
        
        # Filtra paths para conter apenas os do split
        filtered_img_paths = []
        filtered_mask_paths = []
        
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            img_filename = os.path.basename(img_path)
            if img_filename in filenames:
                filtered_img_paths.append(img_path)
                filtered_mask_paths.append(mask_path)
        
        self.image_paths = filtered_img_paths
        self.mask_paths = filtered_mask_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Retorna um item do dataset: (imagem, patches, mapas_alvo)
        
        Args:
            idx: Índice do item
            
        Returns:
            Dicionário com 'image', 'patches', 'patch_boxes', 'target_heatmaps'
        """
        # Carrega imagem e máscara
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Redimensiona para tamanho padrãos
        image = self.resize_img(image)
        mask = self.resize_img(mask)
        
        # Converte para numpy para processamento
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Aplica data augmentation se configurado
        if self.aug_transform and self.apply_aug:
            augmented = self.aug_transform(image=image_np, mask=mask_np)
            image_np = augmented['image']
            mask_np = augmented['mask']
        
        # Extrai patches de lesão (positivos) e de background (negativos)
        patches, patch_boxes, target_heatmaps, weight_masks, negative_mask, segmentor_target_mask = self._extract_patches_and_targets_with_negatives(
            image_np, mask_np
        )
        
        # --- opcional: remover realisticamente a(s) região(ões) dos patches na imagem alvo ---
        image_no_patch = image_np.copy()
        if self.p_remove > 0 and len(patch_boxes) > 0 and random.random() < self.p_remove:
            for box in patch_boxes:
                # box pode ser tensor/tuple/list
                if isinstance(box, torch.Tensor):
                    bx = int(box[0].item()); by = int(box[1].item()); bx2 = int(box[2].item()); by2 = int(box[3].item())
                else:
                    bx, by, bx2, by2 = map(int, box)
                bx = max(0, bx); by = max(0, by); bx2 = min(image_np.shape[1], bx2); by2 = min(image_np.shape[0], by2)
                if bx2 <= bx or by2 <= by:
                    continue
                h = by2 - by; w = bx2 - bx
                method = random.choice(['inpaint', 'blur', 'local_patch'])
                if method == 'inpaint':
                    try:
                        mask_roi = np.zeros((image_no_patch.shape[0], image_no_patch.shape[1]), dtype=np.uint8)
                        mask_roi[by:by2, bx:bx2] = 255
                        img_bgr = cv2.cvtColor(image_no_patch, cv2.COLOR_RGB2BGR)
                        inpainted = cv2.inpaint(img_bgr, mask_roi, 3, cv2.INPAINT_TELEA)
                        image_no_patch = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
                    except Exception:
                        # fallback para blur
                        roi = image_no_patch[by:by2, bx:bx2].copy()
                        roi = cv2.GaussianBlur(roi, (min(51, (max(3, (w//2)|1))),)*2, 0)
                        image_no_patch[by:by2, bx:bx2] = roi
                elif method == 'blur':
                    roi = image_no_patch[by:by2, bx:bx2].copy()
                    k = max(3, min(51, int(max(w,h)//2)|1))
                    roi = cv2.GaussianBlur(roi, (k, k), 0)
                    image_no_patch[by:by2, bx:bx2] = roi
                else:  # local_patch: amostra área próxima
                    dx = random.randint(-w, w)
                    dy = random.randint(-h, h)
                    sx1 = np.clip(bx + dx, 0, image_no_patch.shape[1]-w)
                    sy1 = np.clip(by + dy, 0, image_no_patch.shape[0]-h)
                    sample = image_no_patch[sy1:sy1+h, sx1:sx1+w].copy()
                    if sample.shape[0] == h and sample.shape[1] == w:
                        image_no_patch[by:by2, bx:bx2] = sample

        # Converte para tensores (usa image_no_patch que pode ter sido modificado)
        image_tensor = self.to_tensor(Image.fromarray(image_no_patch))
        # image_tensor = self.normalize(image_tensor)  # se usar normalização, ative aqui
        mask_tensor = self.to_tensor((mask_np > 0).astype(np.float32))
        
        # Dicionário com todos os dados necessários
        result = {
            'image': image_tensor,
            'mask': mask_tensor,
            'segmentor_target_mask': segmentor_target_mask.unsqueeze(0), # Adicionar
            'patches': patches,
            'patch_boxes': patch_boxes,
            'target_heatmaps': target_heatmaps,
            'negative_mask': negative_mask,
            'weight_masks': weight_masks,
            'image_path': img_path
        }
        
        return result
    
    def _extract_patches_and_targets_with_negatives(self, image_np, mask_np, num_neg_per_pos=1):
        """
        Extrai patches de lesão (positivos) e patches de background (negativos).
        Para patches positivos, também gera uma máscara de pesos para focar no contexto/fronteira.
        Agora o contexto é um anel natural à volta da lesão, criado por dilatação morfológica.
        """
        # Binariza a máscara
        binary_mask = (mask_np > 0).astype(np.uint8)
        
        # Dimensões da imagem
        img_h, img_w = image_np.shape[:2]

        # --- CONTEXTO ADAPTATIVO POR BACKGROUND (GENÉRICO) ---
        # Valores alvo: objeto = 1.0, contexto = 0.6 (soft label — ajusta conforme precisares)
        OBJECT_TARGET_VALUE = 1.0
        # Valores alvo: contexto = 0.6 (soft label — ajusta conforme precisares)
        CONTEXT_TARGET_VALUE = 0.6

        # prepara alvo vazio (float -> suporta soft labels)
        segmentor_target_mask_np = np.zeros_like(binary_mask, dtype=np.float32)

        # calcula mapa de clusters de background (uma vez por imagem) — usado para limitar expansão
        cluster_map = self._cluster_labels(image_np, n_clusters=4)

        # Encontra componentes conectados (lesões individuais)
        binary_mask_for_stats = binary_mask * 255
        num_labels, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_for_stats, connectivity=8)

        # Para cada instância, gera ANEL/TRANSIÇÃO: base_ring (raio fixo) + expansão adaptativa no mesmo background
        for inst_idx in range(1, num_labels):
            inst_mask = (cc_labels == inst_idx).astype(np.uint8)
            if inst_mask.sum() == 0:
                continue

            # ---- 1) base ring (zona de transição imediata junto ao contorno) ----
            # kernel base proporcional ao tamanho da instância / imagem (ajustar conforme necessário)
            x, y, bw, bh, area = stats[inst_idx]
            img_max_dim = max(img_h, img_w)
            # base_radius: tenta ser proporcional ao bbox, mas limitada para estabilidade
            base_radius = max(5, int(0.25 * max(bw, bh)))
            base_radius = min(base_radius, int(0.12 * img_max_dim))  # não extrapolar demasiado
            # garante kernel ímpar
            if base_radius % 2 == 0:
                base_radius += 1
            base_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_radius, base_radius))
            dil_base = cv2.dilate(inst_mask, base_kernel, iterations=1)
            base_ring = (dil_base - inst_mask).clip(0, 1).astype(np.uint8)

            # ---- 2) expansão adaptativa do contexto no mesmo background ----
            # _adaptive_context_by_background devolve dilated_inst - inst_mask até o limite permitido pelo cluster_map
            # converte intenção anterior (max_kernel=200, step=8) para new API:
            # max_steps ~= max_kernel / step => 200/8 = 25, step_kernel_size = 8
            ctx_adapt = self._adaptive_context_by_background(
                inst_mask,
                image_np,
                cluster_map=cluster_map,
                max_steps=70,
                step_kernel_size=6,
                color_thresh=14.0,
                max_area_frac=0.50
            )

            # Assegura que o anel base está incluído (caso a expansão adaptativa seja muito conservadora)
            final_ctx = np.logical_or(ctx_adapt, base_ring).astype(np.uint8)

            # Garante que o interior da instância NÃO é marcado
            final_ctx[inst_mask == 1] = 0

            # Escreve no alvo (se sobreposição entre instâncias, mantém o valor máximo)
            segmentor_target_mask_np[final_ctx == 1] = np.maximum(segmentor_target_mask_np[final_ctx == 1], CONTEXT_TARGET_VALUE)
        # --- FIM DA NOVA CONTEXTUALIZAÇÃO ---

        # Extrai patches de lesão (positivos) e de background (negativos)
        patches = []
        patch_boxes = []
        target_heatmaps = []
        weight_masks = []

        img_h, img_w = image_np.shape[:2]

        CONTEXT_WEIGHT = 1.0
        BORDER_WEIGHT = 1.5
        INTERIOR_WEIGHT = 0.1

        # --- Patches Positivos (Lesões) ---
        all_positive_boxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if w < 16 or h < 16 or area < 512:
                continue
            all_positive_boxes.append((x, y, w, h))

        for i, (x, y, w, h) in enumerate(all_positive_boxes):
            # Extrai o patch da imagem original usando a bounding box da instância
            patch = image_np[y:y+h, x:x+w].copy()
            box = [x, y, x+w, y+h]
            patch_boxes.append(torch.tensor(box, dtype=torch.float32))

            # Máscara da instância dentro do patch
            instance_mask_in_patch = (mask_np[y:y+h, x:x+w] > 0).astype(np.uint8)
            
            # # --- Data augmentation: padding para patches pequenos ---
            # if w < 64 or h < 64:
            #     pad_h = max(0, 64 - h)
            #     pad_w = max(0, 64 - w)
            #     patch = cv2.copyMakeBorder(patch, pad_h//2, pad_h-pad_h//2, pad_w//2, pad_w-pad_w//2, cv2.BORDER_REFLECT)
            #     instance_mask_in_patch = cv2.copyMakeBorder(instance_mask_in_patch, pad_h//2, pad_h-pad_h//2, pad_w//2, pad_w-pad_w//2, cv2.BORDER_REFLECT)

            # Kernel para contexto do patch: igual ao kernel global (ex: 80x80 elíptico)
            kernel_size_patch = min(80, max(5, int(0.8 * max(w, h))))
            kernel_patch = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_patch, kernel_size_patch))

            # Cria máscara de pesos para o patch (contexto, borda, interior)
            kernel_patch = np.ones((5, 5), np.uint8)
            dilated_patch_mask = cv2.dilate(instance_mask_in_patch, kernel_patch, iterations=1)
            eroded_patch_mask = cv2.erode(instance_mask_in_patch, kernel_patch, iterations=1)

            border_zone = dilated_patch_mask - eroded_patch_mask
            interior_zone = eroded_patch_mask
            context_zone = 1 - dilated_patch_mask

            weight_mask_np = np.ones_like(instance_mask_in_patch, dtype=np.float32)
            weight_mask_np[context_zone == 1] = CONTEXT_WEIGHT
            weight_mask_np[border_zone == 1] = BORDER_WEIGHT
            weight_mask_np[interior_zone == 1] = INTERIOR_WEIGHT
            weight_masks.append(torch.from_numpy(weight_mask_np))

            if self.patch_size:
                patch = cv2.resize(patch, self.patch_size)
                instance_mask_in_patch = cv2.resize(instance_mask_in_patch, self.patch_size, interpolation=cv2.INTER_NEAREST)
                weight_mask_np = cv2.resize(weight_mask_np, self.patch_size, interpolation=cv2.INTER_NEAREST)
            patch_tensor = self.to_tensor(patch)
            box = [x, y, x+w, y+h]

            cx, cy = centroids[i+1]
            target_heatmap = self._create_gaussian_heatmap(
                (img_h, img_w), (cx, cy), sigma=max(w, h) / 2
            )

            patches.append(patch_tensor)
            patch_boxes.append(torch.tensor(box, dtype=torch.float32))
            target_heatmaps.append(target_heatmap)

        # --- Máscara Negativa Global ---
        global_negative_mask_np = np.zeros((img_h, img_w), dtype=np.float32)
        for x, y, w, h in all_positive_boxes:
            global_negative_mask_np[y:y+h, x:x+w] = 1.0
        global_negative_mask = torch.from_numpy(global_negative_mask_np)

        if not patches:
            # aplica pós-processamento mesmo que não hajam patches (para consistência visual)
            try:
                # parâmetros adaptativos baseados na dimensão da imagem
                img_max_dim = max(img_h, img_w)
                k_close = max(7, int(0.02 * img_max_dim))  # kernel para closing/opening
                if k_close % 2 == 0: k_close += 1

                bin_mask = (segmentor_target_mask_np > 0.0).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
                # fecha pequenos buracos e depois remove pequenas ilhas
                closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

                # remove componentes muito pequenos (ruído)
                num_cc, cc_map, stats, centroids = cv2.connectedComponentsWithStats(opened.astype(np.uint8), connectivity=8)
                min_comp_area = max(64, int(0.001 * img_h * img_w))
                cleaned = np.zeros_like(opened, dtype=np.uint8)
                for cid in range(1, num_cc):
                    area = stats[cid, cv2.CC_STAT_AREA]
                    if area >= min_comp_area:
                        cleaned[cc_map == cid] = 1

                # suaviza (mantém soft labels) e re-escalona para o valor de contexto original (se presente)
                blur_k = max(3, int(0.01 * img_max_dim))
                if blur_k % 2 == 0: blur_k += 1
                smooth = cv2.GaussianBlur(cleaned.astype(np.float32), (blur_k, blur_k), 0)

                # atribui valor soft semelhante ao CONTEXT_TARGET_VALUE usado antes (0.6)
                context_val = 0.6
                smooth_soft = smooth * context_val
                segmentor_target_mask_np = np.maximum(segmentor_target_mask_np, smooth_soft.astype(np.float32))
            except Exception:
                # fallback: não altera se algo falhar
                pass
            return [], [], [], [], torch.zeros((img_h, img_w), dtype=torch.float32), torch.from_numpy(segmentor_target_mask_np)

        # --- Pós‑processamento global da máscara segmentor_target_mask_np ---
        try:
            img_max_dim = max(img_h, img_w)
            k_close = max(7, int(0.02 * img_max_dim))
            if k_close % 2 == 0: k_close += 1
            bin_mask = (segmentor_target_mask_np > 0.0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
            closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

            # remove componentes muito pequenos
            num_cc, cc_map, stats, centroids = cv2.connectedComponentsWithStats(opened.astype(np.uint8), connectivity=8)
            min_comp_area = max(64, int(0.001 * img_h * img_w))
            cleaned = np.zeros_like(opened, dtype=np.uint8)
            for cid in range(1, num_cc):
                area = stats[cid, cv2.CC_STAT_AREA]
                if area >= min_comp_area:
                    cleaned[cc_map == cid] = 1

            blur_k = max(3, int(0.01 * img_max_dim))
            if blur_k % 2 == 0: blur_k += 1
            smooth = cv2.GaussianBlur(cleaned.astype(np.float32), (blur_k, blur_k), 0)
            context_val = 0.6
            smooth_soft = smooth * context_val
            # combina mantendo valores mais fortes já existentes
            segmentor_target_mask_np = np.maximum(segmentor_target_mask_np, smooth_soft.astype(np.float32))
        except Exception:
            pass

        return patches, patch_boxes, target_heatmaps, weight_masks, global_negative_mask, torch.from_numpy(segmentor_target_mask_np)
    
    def _create_gaussian_heatmap(self, image_size, center, sigma=20):
        """
        Cria um mapa de calor gaussiano centrado em 'center'
        
        Args:
            image_size: Tamanho da imagem (H, W)
            center: Centro da gaussiana (x, y)
            sigma: Desvio padrão da gaussiana
            
        Returns:
            Tensor do mapa de calor [H, W]
        """
        h, w = image_size
        x_grid = np.arange(0, w, 1, float)
        y_grid = np.arange(0, h, 1, float)[:, np.newaxis]
        
        x0, y0 = center
        
        # Calcula gaussiana 2D
        gaussian = np.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * sigma ** 2))
        
        # Normaliza para max=1
        if gaussian.max() > 0:
            gaussian = gaussian / gaussian.max()
        
        # Converte para tensor
        return torch.tensor(gaussian, dtype=torch.float32)
    
    def _cluster_labels(self, image_np, n_clusters=4):
        """
        Agrupa pixels por cor (LAB) usando k-means e retorna mapa de rótulos HxW.
        Uso: delimitar sub-regiões de background (ex.: mar / céu / terra).
        """
        h, w = image_np.shape[:2]
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        data = lab.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        attempts = 3
        flags = cv2.KMEANS_PP_CENTERS
        try:
            _, labels, _ = cv2.kmeans(data, n_clusters, None, criteria, attempts, flags)
            labels = labels.flatten().reshape(h, w)
        except Exception:
            # fallback simples (luminosidade)
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            labels = (hsv[..., 2] > np.median(hsv[..., 2])).astype(np.int32)
        return labels

    def _adaptive_context_by_background(self, inst_mask, image_np, cluster_map=None,
                                        max_steps=60, step_kernel_size=3, color_thresh=18.0,
                                        max_area_frac=0.35):
        """
        Expansão adaptativa do contexto voltada para o mesmo background.
        Método:
          - converte a imagem para LAB e calcula o 'centro de cor' da vizinhança imediata da instância;
          - itera pequenos passos de dilatação (kernel pequeno) e só adiciona pixels cuja distância
            LAB ao centro de cor seja <= color_thresh;
          - garante conectividade (só pixels adjacentes à região corrente são considerados);
          - para quando não há mais candidatos, atinge max_steps ou atinge max_area_frac da imagem.
        Parâmetros:
          - inst_mask: uint8 0/1 máscara da instância
          - image_np: RGB uint8 HxW x3
          - cluster_map: (opcional) mapa de clusters já calculado (não usado diretamente aqui, mantido para compatibilidade)
          - max_steps: número máximo de iterações (cada passo cresce ~ step_kernel_size)
          - step_kernel_size: tamanho do kernel de crescimento (tipicamente 3)
          - color_thresh: limiar de distância LAB para aceitar pixel no contexto (experimentar 12..30)
          - max_area_frac: limita fração máxima da imagem que pode ser ocupada pelo contexto
        Retorna:
          - context_mask (uint8 0/1) = expansão_resultante - inst_mask
        """
        h, w = inst_mask.shape
        if inst_mask.sum() == 0:
            return np.zeros_like(inst_mask, dtype=np.uint8)

        # converte para LAB float32 (L:0-100, a,b: aproximados -128..127)
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)

        # calcula rótulo/centro de cor a partir do anel próximo à instância (mais robusto que interior)
        small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        ring = cv2.dilate(inst_mask, small_k, iterations=1) - inst_mask
        ring_idxs = np.where(ring == 1)
        if ring_idxs[0].size > 0:
            ring_pixels = lab[ring_idxs]
            center_color = ring_pixels.mean(axis=0)
        else:
            # fallback: usa média da própria instância
            inst_idxs = np.where(inst_mask == 1)
            center_color = lab[inst_idxs].mean(axis=0)

        # pré-computa distância LAB por pixel ao centro de cor
        diff = lab - center_color[None, None, :]
        dist_lab = np.linalg.norm(diff, axis=2)

        # allowed mask: candidatos por semelhança de cor
        allowed = (dist_lab <= color_thresh)

        # conectividade / crescimento iterativo
        current = inst_mask.copy().astype(np.uint8)
        max_pixels = int(max_area_frac * h * w)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (step_kernel_size, step_kernel_size))

        for step in range(max_steps):
            # frontier: pixels adjacentes a current (1-pixel dilatation) that are allowed and not yet included
            dil = cv2.dilate(current, kernel, iterations=1)
            frontier = (dil.astype(np.uint8) & (~current.astype(np.uint8))).astype(np.uint8)
            candidates = frontier & allowed.astype(np.uint8)

            if candidates.sum() == 0:
                break

            # adiciona apenas os candidatos conectados diretamente ao objeto corrente
            # (garante que não "salta" por zonas permitidas desconectadas)
            new_region = (current | candidates).astype(np.uint8)

            # prevenção de crescimento excessivo
            if new_region.sum() > max_pixels:
                # limita: adiciona apenas uma fracção dos candidatos por ordem de proximidade ao contorno
                cand_idxs = np.stack(np.nonzero(candidates), axis=1)
                # calcula distância ao contorno (pixels mais próximos primeiro)
                contoured = cv2.distanceTransform(1 - current, cv2.DIST_L2, 5)
                dists = contoured[candidates == 1]
                # ordena índices pelo valor de dists ascendente
                order = np.argsort(dists)
                # escolher quantos ainda podemos adicionar
                remaining = max_pixels - int(current.sum())
                if remaining <= 0:
                    break
                chosen = cand_idxs[order[:remaining]]
                # construir máscara escolhida
                chosen_mask = np.zeros_like(current, dtype=np.uint8)
                chosen_mask[chosen[:, 0], chosen[:, 1]] = 1
                current = (current | chosen_mask).astype(np.uint8)
                break

            # incorpora candidatos válidos e continua
            current = new_region

        # contexto final = região crescida menos a instância original
        context = (current - inst_mask).clip(0, 1).astype(np.uint8)
        return context    
    
    def visualize_sample(self, idx, save_path=None):
        """
        Visualiza uma amostra do dataset com patches, mapas de calor e máscaras de segmentação.
        """
        sample = self[idx]
        image = sample['image'].permute(1, 2, 0).numpy()
        patches = sample['patches']
        target_heatmaps = sample['target_heatmaps']
        segmentor_mask = sample['segmentor_target_mask'][0].numpy()  # [1,H,W] -> [H,W]
        mask_tensor = sample['mask'][0].numpy() if sample['mask'].ndim == 4 else sample['mask'].numpy()

        n_patches = len(patches)
        fig_rows = 3 if n_patches > 0 else 2
        fig, axes = plt.subplots(fig_rows, n_patches + 1, figsize=(3 * (n_patches + 1), 9), squeeze=False)

        # Linha 1: Imagem original
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Imagem Original")
        axes[0, 0].axis('off')

        # Linha 2: Mapa positivo combinado
        combined_heatmap = torch.zeros_like(target_heatmaps[0])
        for heatmap in target_heatmaps:
            if heatmap.max() > 0:
                combined_heatmap = torch.max(combined_heatmap, heatmap)
        axes[1, 0].imshow(combined_heatmap.numpy(), cmap='hot')
        axes[1, 0].set_title("Mapa Positivo")
        axes[1, 0].axis('off')

        # Linha 3: Máscara de segmentação geral
        axes[2, 0].imshow(segmentor_mask, cmap='gray')
        axes[2, 0].set_title("Máscara Segmentação Geral")
        axes[2, 0].axis('off')

        for i, (patch, heatmap) in enumerate(zip(patches, target_heatmaps)):
            patch_np = patch.permute(1, 2, 0).numpy()
            axes[0, i + 1].imshow(patch_np)
            is_positive = heatmap.max() > 0
            patch_type = "Positivo" if is_positive else "Negativo"
            axes[0, i + 1].set_title(f"Patch {i+1} ({patch_type})")
            axes[0, i + 1].axis('off')

            axes[1, i + 1].imshow(heatmap.numpy(), cmap='hot')
            axes[1, i + 1].set_title(f"Mapa Alvo {i+1}")
            axes[1, i + 1].axis('off')

            # Máscara de cada instância (patch)
            # Se quiser mostrar a máscara binária original da instância:
            mask_to_show = mask_tensor.squeeze()
            axes[2, i + 1].imshow(mask_to_show, cmap='gray')
            axes[2, i + 1].set_title(f"Máscara Instância {i+1}")
            axes[2, i + 1].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig


def attention_collate(batch):
    """
    Função de collate customizada para o AttentionDataset.
    Agrupa amostras em batch, lidando com número variável de patches por imagem.
    
    Args:
        batch: Lista de amostras retornadas pelo __getitem__
        
    Returns:
        Dicionário com tensores em batch
    """
    # Separa os componentes do batch
    images = []
    masks = []
    segmentor_target_masks = [] # Adicionar
    all_patches = []
    all_boxes = []
    all_heatmaps = []
    all_negative_masks = [] # Adicionar
    all_weight_masks = [] # Adicionar
    all_sample_indices = []  # Para rastrear de qual amostra vem cada patch
    
    for i, sample in enumerate(batch):
        images.append(sample['image'])
        masks.append(sample['mask'])
        segmentor_target_masks.append(sample['segmentor_target_mask']) # Adicionar

        # Adiciona a máscara negativa da imagem inteira
        if 'negative_mask' in sample and sample['patches']:
             all_negative_masks.append(sample['negative_mask'])

        # Processa cada patch na amostra
        for patch, box, heatmap, weight_mask in zip(sample['patches'], 
                                                    sample['patch_boxes'], 
                                                    sample['target_heatmaps'],
                                                    sample['weight_masks']): # Adicionar
            all_patches.append(patch)
            all_boxes.append(box)
            all_heatmaps.append(heatmap)
            all_weight_masks.append(weight_mask) # Adicionar
            all_sample_indices.append(i)
    
    # Stack das imagens em um único tensor
    if images:
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        segmentor_target_masks = torch.stack(segmentor_target_masks, dim=0) # Adicionar
    else:
        images = torch.zeros((0, 3, 512, 512), dtype=torch.float32)
        masks = torch.zeros((0, 1, 512, 512), dtype=torch.float32)
        segmentor_target_masks = torch.zeros((0, 1, 512, 512), dtype=torch.float32) # Adicionar
    
    # Se não há patches, retorna tensores vazios para patches e heatmaps
    if not all_patches:
        all_patches = torch.zeros((0, 3, 64, 64), dtype=torch.float32)  # Tamanho arbitrário
        all_boxes = torch.zeros((0, 4), dtype=torch.float32)
        all_heatmaps = torch.zeros((0, 512, 512), dtype=torch.float32)  # Assume tamanho padrão
        all_sample_indices = torch.zeros((0,), dtype=torch.long)
    else:
        # Stack de patches e heatmaps
        try:
            all_patches = torch.stack(all_patches, dim=0)
        except:
            # Se os patches têm tamanhos diferentes, usa padding para tamanho uniforme
            max_h = max(p.shape[1] for p in all_patches)
            max_w = max(p.shape[2] for p in all_patches)
            
            padded_patches = []
            for p in all_patches:
                # Padding para o tamanho máximo
                p_padded = torch.zeros((3, max_h, max_w), dtype=torch.float32)
                p_padded[:, :p.shape[1], :p.shape[2]] = p
                padded_patches.append(p_padded)
            
            all_patches = torch.stack(padded_patches, dim=0)
        
        all_boxes = torch.stack(all_boxes, dim=0)
        all_heatmaps = torch.stack(all_heatmaps, dim=0)
        all_sample_indices = torch.tensor(all_sample_indices, dtype=torch.long)
    
    if all_negative_masks:
        all_negative_masks = torch.stack(all_negative_masks, dim=0)
    else:
        all_negative_masks = torch.zeros((0, 512, 512), dtype=torch.float32)

    # Stack das máscaras de pesos
    if all_weight_masks:
        # Lidar com tamanhos diferentes via padding
        max_h_w = max(p.shape[0] for p in all_weight_masks)
        max_w_w = max(p.shape[1] for p in all_weight_masks)
        padded_weights = []
        for w_mask in all_weight_masks:
            p_padded = torch.ones((max_h_w, max_w_w), dtype=torch.float32) # Padding com 1.0 (peso neutro)
            p_padded[:w_mask.shape[0], :w_mask.shape[1]] = w_mask
            padded_weights.append(p_padded)
        all_weight_masks = torch.stack(padded_weights, dim=0)
    else:
        all_weight_masks = torch.zeros((0, 64, 64), dtype=torch.float32) # Tamanho arbitrário

    return {
        'images': images,
        'masks': masks,
        'segmentor_target_mask': segmentor_target_masks, # Adicionar
        'patches': all_patches,
        'boxes': all_boxes,
        'heatmaps': all_heatmaps,
        'negative_masks': all_negative_masks,
        'weight_masks': all_weight_masks, # Adicionar
        'sample_indices': all_sample_indices
    }


if __name__ == "__main__":
    # Código de teste para validar o dataset
    dataset_path = os.path.join(project_root, 'data', 'airplane')
    
    # Testa o dataset
    dataset = AttentionDataset(
        root_path=dataset_path,
        image_size=(512, 512),
        apply_aug=False
    )
    
    print(f"Dataset carregado com {len(dataset)} amostras")
    
    # Visualiza algumas amostras
    save_dir = os.path.join(project_root, 'data', 'debug_attn_dataset')
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(5, len(dataset))):
        print(f"Visualizando amostra {i}")
        dataset.visualize_sample(i, save_path=os.path.join(save_dir, f'sample_{i}.png'))
    
    # Testa o collate
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True,
        collate_fn=attention_collate
    )
    
    batch = next(iter(dataloader))
    print("Batch carregado com sucesso:")
    print(f"- Imagens: {batch['images'].shape}")
    print(f"- Máscaras: {batch['masks'].shape}") # Adicionar
    print(f"- Patches: {batch['patches'].shape if batch['patches'].numel() > 0 else '0'}")
    print(f"- Boxes: {batch['boxes'].shape if batch['boxes'].numel() > 0 else '0'}")
    print(f"- Heatmaps: {batch['heatmaps'].shape if batch['heatmaps'].numel() > 0 else '0'}")
    print(f"- Índices: {batch['sample_indices'].shape if batch['sample_indices'].numel() > 0 else '0'}")
