import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

try:
    import albumentations as A
except ImportError:
    A = None

# Configuração de caminhos para garantir importações corretas
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

class AttentionDataset(Dataset):
    """
    Dataset especializado para treinamento do AttentionRecommender.
    Implementa estratégias de 'Blindfold Training' para forçar aprendizagem de contexto.
    """
    def __init__(self, root_path, image_size=(512, 512), patch_size=None, 
                 transform=None, hyp=None, apply_aug=False,
                 split_ratio=0.2, split_mode='none', split='all', p_remove=0.8):
        self.root_path = root_path
        self.image_size = image_size
        self.patch_size = patch_size
        self.transform = transform
        self.hyp = hyp
        self.apply_aug = apply_aug
        self.split = split
        
        # Probabilidade de remover o objeto da imagem de destino (Inpainting)
        # Alto valor (0.8) força o modelo a adivinhar pelo contexto
        self.p_remove = float(p_remove)

        self.to_tensor = transforms.ToTensor()
        self.resize_img = transforms.Resize(self.image_size)
        
        self._setup_augmentation()
        self.image_paths, self.mask_paths = self._load_data_paths()
        
        if split_mode == 'ratio':
            self._split_by_ratio(split_ratio, split)
        elif split_mode == 'preset':
            self._load_preset_split(split)
    
    def _setup_augmentation(self):
        if self.apply_aug and A is not None:
            # Augmentação GLOBAL (Imagem + Máscara)
            self.aug_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.20, rotate_limit=30, p=0.6),
                A.RandomBrightnessContrast(p=0.7),
                A.RandomGamma(p=0.5),
                A.CLAHE(p=0.4),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.4),
                A.GaussNoise(var_limit=(10.0, 80.0), p=0.6),
                A.Blur(blur_limit=7, p=0.3),
            ])
            
            # Augmentação EXCLUSIVA DO PATCH (Quebra correlação de cor exata)
            self.patch_only_transform = A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ToGray(p=0.1),
            ])
            
            # ✅ NOVO: Augmentação do CONTEXTO
            self.context_aug = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.Blur(blur_limit=7, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.8),  # Aumenta probabilidade de ruído para o modelo não ver áreas "mortas"
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.6),
            ])
        else:
            self.aug_transform = None
            self.patch_only_transform = None
    
    def _load_data_paths(self):
        # Lógica robusta de procura de pastas
        possible_roots = [
            self.root_path,
            os.path.join(self.root_path, 'boat'),
            os.path.join(self.root_path, '..', 'boat')
        ]
        
        img_dir, mask_dir = None, None
        for root in possible_roots:
            i_dir = os.path.join(root, 'images')
            m_dir = os.path.join(root, 'masks')
            if os.path.exists(i_dir) and os.path.exists(m_dir):
                img_dir, mask_dir = i_dir, m_dir
                break
        
        if not img_dir:
            raise FileNotFoundError(f"Não foi possível encontrar pastas images/masks em {self.root_path}")
            
        image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        image_paths, mask_paths = [], []
        
        for f in image_files:
            m_name = f.replace('.jpg', '.png').replace('.jpeg', '.png')
            m_path = os.path.join(mask_dir, m_name)
            if os.path.exists(m_path):
                image_paths.append(os.path.join(img_dir, f))
                mask_paths.append(m_path)
                
        return image_paths, mask_paths

    def _split_by_ratio(self, split_ratio, split):
        n = len(self.image_paths)
        indices = list(range(n))
        random.seed(42)
        random.shuffle(indices)
        split_idx = int(n * (1 - split_ratio))
        
        if split == 'train': idxs = indices[:split_idx]
        elif split == 'val': idxs = indices[split_idx:]
        else: idxs = indices
        
        self.image_paths = [self.image_paths[i] for i in idxs]
        self.mask_paths = [self.mask_paths[i] for i in idxs]

    def _load_preset_split(self, split):
        if split == 'all': return
        split_file = os.path.join(self.root_path, f'{split}.txt')
        if not os.path.exists(split_file): return
        
        with open(split_file, 'r') as f:
            names = set(line.strip() for line in f)
            
        self.image_paths = [p for p in self.image_paths if os.path.basename(p) in names]
        self.mask_paths = [p for p in self.mask_paths if os.path.basename(p).replace('.png','.jpg') in names]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        image = self.resize_img(image)
        mask = self.resize_img(mask)
        
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        if self.aug_transform and self.apply_aug:
            aug = self.aug_transform(image=image_np, mask=mask_np)
            image_np, mask_np = aug['image'], aug['mask']
            
        # Extração de patches
        patches, patch_boxes, target_heatmaps, weight_masks, neg_mask, seg_target = \
            self._extract_patches_and_targets_with_negatives(image_np, mask_np)
            
        # 1. Limitar patches no treino (evita OOM)
        if self.split == 'train' and len(patches) > 0:
            idx = random.randint(0, len(patches) - 1)
            patches = [patches[idx]]
            patch_boxes = [patch_boxes[idx]]
            target_heatmaps = [target_heatmaps[idx]]
            weight_masks = [weight_masks[idx]]
            
            # ✅ NOVO: Aplica augmentação de contexto (AQUI!)
            if self.apply_aug and self.context_aug is not None:
                for i in range(len(patches)):
                    if isinstance(patches[i], dict):
                        # Converte tensor para numpy
                        ctx_tensor = patches[i]['with_context']
                        ctx_numpy = ctx_tensor.cpu().numpy().transpose(1, 2, 0)  # [h, w, 3]
                        
                        # Aplica augmentação
                        ctx_aug = self.context_aug(image=ctx_numpy)['image']
                        
                        # Converte de volta para tensor
                        patches[i]['with_context'] = self.to_tensor(ctx_aug)
                        
        # 2. INPAINTING: Remover o objeto da imagem de destino (Blindfold Training)
        # O modelo vê a imagem SEM o objeto e tem de adivinhar onde ele estava
        image_for_model = image_np.copy()
        
        # ✅ ATIVADO: Blindfold Training
        if self.split == 'train' and self.p_remove > 0 and len(patch_boxes) > 0:
            if random.random() < self.p_remove:
                box = patch_boxes[0]
                x1, y1, x2, y2 = map(int, box)
                
                # Escolhe estratégia aleatoriamente
                strategy = random.choice(['remove', 'remove', 'blur', 'noise']) # 50% remove, 25% blur, 25% noise
                
                if strategy == 'remove':
                    # ===== ESTRATÉGIA 1: Preenche com média da borda =====
                    # Calcula média de cores numa faixa ao redor do objeto
                    border_size = 10
                    border_y1 = max(0, y1 - border_size)
                    border_y2 = min(image_np.shape[0], y2 + border_size)
                    border_x1 = max(0, x1 - border_size)
                    border_x2 = min(image_np.shape[1], x2 + border_size)
                    
                    # Extrai pixels da borda (excluindo o objeto)
                    border_region = image_np[border_y1:border_y2, border_x1:border_x2].copy()
                    
                    # Remove objeto da região de borda para calcular cor "real" do fundo
                    border_mask = np.ones((border_y2-border_y1, border_x2-border_x1), dtype=bool)
                    obj_y1 = max(0, y1 - border_y1)
                    obj_y2 = min(border_region.shape[0], y2 - border_y1)
                    obj_x1 = max(0, x1 - border_x1)
                    obj_x2 = min(border_region.shape[1], x2 - border_x1)
                    border_mask[obj_y1:obj_y2, obj_x1:obj_x2] = False
                    
                    # Cor média apenas do fundo verdadeiro
                    if border_mask.sum() > 0:
                        mean_color = border_region[border_mask].mean(axis=0).astype(np.uint8)
                    else:
                        mean_color = image_np.mean(axis=(0, 1)).astype(np.uint8)
                    
                    # ===== ESTRATÉGIA 2: Preenche com textura (não cor sólida) =====
                    # Adiciona ruído Gaussiano para parecer natural
                    obj_h, obj_w = y2 - y1, x2 - x1
                    noise = np.random.normal(0, 8, (obj_h, obj_w, 3)).astype(np.int16)
                    filled = mean_color[np.newaxis, np.newaxis, :] + noise
                    filled = np.clip(filled, 0, 255).astype(np.uint8)
                    
                    # Suaviza para evitar picos
                    filled = cv2.GaussianBlur(filled, (7, 7), 1.5)
                    
                    # Aplica ao modelo
                    image_for_model[y1:y2, x1:x2] = filled

                elif strategy == 'blur':
                    # Apenas desfoca muito a região (mantém cores, perde detalhes)
                    roi = image_for_model[y1:y2, x1:x2]
                    # Blur pesado
                    roi = cv2.GaussianBlur(roi, (51, 51), 30)
                    image_for_model[y1:y2, x1:x2] = roi
                    
                elif strategy == 'noise':
                    # Adiciona ruído forte sobre o objeto existente
                    roi = image_for_model[y1:y2, x1:x2].astype(np.float32)
                    noise = np.random.normal(0, 50, roi.shape).astype(np.float32)
                    roi = roi + noise
                    image_for_model[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)

        image_tensor = self.to_tensor(image_for_model)
        mask_tensor = self.to_tensor((mask_np > 0).astype(np.float32))
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'segmentor_target_mask': seg_target.unsqueeze(0),
            'patches': patches,                        # Agora contém dicts com 'object_only' e 'with_context'
            'patches_with_context': [p['with_context'] if isinstance(p, dict) else p for p in patches],  # ✅ NOVO
            'patch_boxes': patch_boxes,
            'target_heatmaps': target_heatmaps,
            'negative_mask': neg_mask,
            'weight_masks': weight_masks,
            'image_path': img_path
        }

    def _extract_patches_and_targets_with_negatives(self, image_np, mask_np):
        """
        ✅ HIBRIDIZADO: Volta a usar cor LAB (funciona) + inpainting leve.
        """
        binary_mask = (mask_np > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        patches, patch_boxes, heatmaps, weights = [], [], [], []
        seg_target = np.zeros_like(binary_mask, dtype=np.float32)
        
        # ===== INPAINTING LEVE =====
        image_for_heatmap = image_np.copy()
        
        if self.split == 'train' and len(stats) > 1:
            # Seleciona um objeto aleatório
            idx_remove = np.random.randint(1, len(stats))
            inst_mask_check = (labels == idx_remove).astype(np.uint8)
            
            if inst_mask_check.sum() > 0:
                # Computa contraste deste objeto específico
                kernel_size = int(max(stats[idx_remove][2], stats[idx_remove][3]) * 0.5)
                kernel_size = max(5, min(kernel_size, 50))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                dilated = cv2.dilate(inst_mask_check, kernel)
                context_ring_check = (dilated - inst_mask_check).astype(bool)
                
                contrast_level = self._compute_contrast_level(image_np, inst_mask_check, context_ring_check)
                
                # ADAPTATIVO: Se contraste baixo (ambíguo), remove mais frequentemente
                # Se contraste alto (claro), remove menos frequentemente
                p_remove_adaptive = 0.8 if contrast_level < 0.3 else 0.2
                
                if random.random() < p_remove_adaptive:
                    # Preenche com cor média
                    if inst_mask_check.sum() > 0:
                        y1, x1, w, h = stats[idx_remove][1], stats[idx_remove][0], stats[idx_remove][2], stats[idx_remove][3]
                        y2, x2 = y1 + h, x1 + w
                        
                        # Cor do fundo
                        border_size = 10
                        by1 = max(0, y1 - border_size)
                        by2 = min(image_np.shape[0], y2 + border_size)
                        bx1 = max(0, x1 - border_size)
                        bx2 = min(image_np.shape[1], x2 + border_size)
                        
                        border_region = image_np[by1:by2, bx1:bx2]
                        mean_color = border_region.mean(axis=(0, 1)).astype(np.uint8)
                        
                        # Preenche
                        image_for_heatmap[y1:y2, x1:x2] = mean_color
        
        for i in range(1, num_labels):
            inst_mask = (labels == i).astype(np.uint8)
            if inst_mask.sum() < 100:
                continue
            
            # ===== CONTEXTO =====
            kernel_size = int(max(stats[i][2], stats[i][3]) * 0.5)
            kernel_size = max(5, min(kernel_size, 50))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            dilated = cv2.dilate(inst_mask, kernel)
            context_ring = (dilated - inst_mask).astype(bool)
            
            seg_target[context_ring] = 0.9
            
            # ===== HEATMAP COM COR LAB (O QUE FUNCIONAVA) =====
            hm = self._compute_color_heatmap(
                image_for_heatmap,
                inst_mask,
                context_ring
            )
            
            heatmaps.append(hm)
            
            # ===== BBOX =====
            x, y, w, h, area = stats[i]
            patch_boxes.append(torch.tensor([x, y, x+w, y+h], dtype=torch.float32))
            
            # ===== PATCHES =====
            patch_object_only = image_np[y:y+h, x:x+w].copy()
            
            expand_ratio = 1.5
            cx, cy = int(x + w // 2), int(y + h // 2)
            new_w, new_h = int(w * expand_ratio), int(h * expand_ratio)
            ctx_x1 = max(0, cx - new_w // 2)
            ctx_y1 = max(0, cy - new_h // 2)
            ctx_x2 = min(image_np.shape[1], cx + new_w // 2)
            ctx_y2 = min(image_np.shape[0], cy + new_h // 2)
            
            patch_with_context = image_np[ctx_y1:ctx_y2, ctx_x1:ctx_x2].copy()
            
            if self.patch_size:
                patch_with_context = cv2.resize(patch_with_context, self.patch_size)
                patch_object_only = cv2.resize(patch_object_only, self.patch_size)
            
            if self.apply_aug and self.patch_only_transform:
                patch_with_context = self.patch_only_transform(image=patch_with_context)['image']
            
            patches.append({
                'object_only': self.to_tensor(patch_object_only),
                'with_context': self.to_tensor(patch_with_context)
            })
            
            weights.append(torch.ones((64, 64)))
        
        neg_mask = torch.zeros(image_np.shape[:2])
        return patches, patch_boxes, heatmaps, weights, neg_mask, torch.from_numpy(seg_target)
    
    
    def _compute_color_heatmap(self, image_np, inst_mask, context_ring):
        """
        ✅ O QUE FUNCIONAVA: Compatibilidade por cor LAB.
        Mantém o que era bom, apenas adiciona estrutura extra.
        """
        
        # ===== 1. COR DO CONTEXTO =====
        if context_ring.sum() > 0:
            context_colors = image_np[context_ring]
        else:
            context_colors = image_np.reshape(-1, 3)
        
        # Converte para LAB
        context_rgb = context_colors.astype(np.uint8)
        context_lab = cv2.cvtColor(context_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)
        context_mean_lab = context_lab.mean(axis=1).squeeze()
        
        # ===== 2. DISTÂNCIA DE COR EM LAB =====
        image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        color_dist = np.sqrt(
            (image_lab[:, :, 0] - context_mean_lab[0]) ** 2 +
            (image_lab[:, :, 1] - context_mean_lab[1]) ** 2 +
            (image_lab[:, :, 2] - context_mean_lab[2]) ** 2
        )
        
        max_dist = color_dist.max() + 1e-6
        heatmap = 1.0 - (color_dist / max_dist)
        
        # ===== 3. ADICIONA ESTRUTURA (LEVE) =====
        # Não domina, mas ajuda em casos como boat
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad = cv2.Laplacian(gray, cv2.CV_32F)
        grad = np.abs(grad) / (grad.max() + 1e-6)
        
        # Peso baixo: 80% cor + 20% estrutura
        heatmap = 0.8 * heatmap + 0.2 * grad
        
        # ===== 4. COMBINA COM ESTRUTURA =====
        structural_hm = self._compute_structural_heatmap(image_np, inst_mask, context_ring)
        
        # Weight adaptativo baseado em contraste
        # Se baixo contraste (ambíguo), dá mais peso à estrutura
        # Se alto contraste (claro), mantém peso na cor
        contrast_level = np.sqrt(
            (image_lab[:, :, 0] - context_mean_lab[0]) ** 2 +
            (image_lab[:, :, 1] - context_mean_lab[1]) ** 2 +
            (image_lab[:, :, 2] - context_mean_lab[2]) ** 2
        ).mean() / 100.0
        contrast_level = min(contrast_level, 1.0)
        
        w_color = 0.6 + 0.2 * contrast_level  # [0.6, 0.8]
        w_struct = 0.4 - 0.2 * contrast_level  # [0.2, 0.4]
        
        heatmap = w_color * heatmap + w_struct * structural_hm.numpy()
        
        # ===== 5. ZER OBJETO =====
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        obj_dilated = cv2.dilate(inst_mask.astype(np.uint8), kernel_dilate)
        heatmap[obj_dilated > 0] = 0.0
        
        # ===== 6. SUAVIZA =====
        heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (5, 5), 1.0)
        
        # Clipa
        heatmap = np.clip(heatmap, 0, 1).astype(np.float32)
        
        return torch.tensor(heatmap, dtype=torch.float32)
    
    def _compute_contrast_level(self, image_np, inst_mask, context_ring):
        """
        Computa nível de contraste entre objeto e contexto.
        
        Returns:
            float entre 0 (baixo contraste, ambíguo) e 1 (alto contraste, claro)
        """
        import cv2
        
        if inst_mask.sum() == 0 or context_ring.sum() == 0:
            return 0.5  # Fallback
        
        # Cores do objeto e contexto em LAB
        obj_colors = image_np[inst_mask > 0]
        ctx_colors = image_np[context_ring]
        
        obj_lab = cv2.cvtColor(obj_colors.astype(np.uint8).reshape(1, -1, 3), cv2.COLOR_RGB2LAB)
        ctx_lab = cv2.cvtColor(ctx_colors.astype(np.uint8).reshape(1, -1, 3), cv2.COLOR_RGB2LAB)
        
        obj_mean = obj_lab.mean(axis=1).squeeze()
        ctx_mean = ctx_lab.mean(axis=1).squeeze()
        
        # Distância Euclidiana em LAB
        contrast = np.sqrt(
            (obj_mean[0] - ctx_mean[0]) ** 2 +
            (obj_mean[1] - ctx_mean[1]) ** 2 +
            (obj_mean[2] - ctx_mean[2]) ** 2
        )
        
        # Normaliza para [0, 1]
        contrast_norm = min(contrast / 100.0, 1.0)
        return contrast_norm
    
    def _compute_structural_heatmap(self, image_np, inst_mask, context_ring):
        """
        Computa heatmap baseado em características estruturais (não só cor).
        
        Features:
        - Gradientes (bordas)
        - Frequência (textura)
        - Contraste local
        
        Returns:
            Heatmap [H, W] com valores [0, 1]
        """
        import cv2
        
        # Converte para escala de cinza
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # 1. GRADIENTES (Sobel)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradients = np.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # Normaliza
        grad_norm = gradients / (gradients.max() + 1e-6)
        
        # 2. CONTRASTE LOCAL (diferença da vizinhança)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(gray, kernel)
        eroded = cv2.erode(gray, kernel)
        local_contrast = (dilated - eroded) / 255.0
        
        # 3. TEXTURA (Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        texture = np.abs(laplacian) / (np.abs(laplacian).max() + 1e-6)
        
        # Combina as 3 features
        structural_heatmap = (
            0.4 * grad_norm +  # Bordas
            0.4 * local_contrast +  # Contraste
            0.2 * texture  # Textura
        )
        
        # Suaviza
        structural_heatmap = cv2.GaussianBlur(structural_heatmap.astype(np.float32), (5, 5), 1.0)
        
        # Clipa
        structural_heatmap = np.clip(structural_heatmap, 0, 1).astype(np.float32)
        
        return torch.tensor(structural_heatmap, dtype=torch.float32)
    
def attention_collate(batch):
    # Collate simplificado e robusto
    images = torch.stack([b['image'] for b in batch])
    masks = torch.stack([b['mask'] for b in batch])
    seg_targets = torch.stack([b['segmentor_target_mask'] for b in batch])
    
    all_patches, all_patches_ctx, all_boxes, all_heatmaps, all_indices = [], [], [], [], []
    
    for i, b in enumerate(batch):
        for p, p_ctx, box, hm in zip(b['patches'], b['patches_with_context'], b['patch_boxes'], b['target_heatmaps']):
            # Extrai só o tensor do objeto (se é dict, senão usa direto)
            patch_obj = p['object_only'] if isinstance(p, dict) else p
            patch_ctx = p_ctx if isinstance(p_ctx, torch.Tensor) else p_ctx
            
            all_patches.append(patch_obj)
            all_patches_ctx.append(patch_ctx)
            all_boxes.append(box)
            all_heatmaps.append(hm)
            all_indices.append(i)
            
    if not all_patches:
        return {
            'images': images, 
            'patches': torch.empty(0), 
            'patches_with_context': torch.empty(0),  # ✅ NOVO
            'heatmaps': torch.empty(0),
            'sample_indices': torch.empty(0), 
            'segmentor_target_mask': seg_targets
        }
        
    # Padding para patches de tamanhos diferentes
    max_h = max(p.shape[1] for p in all_patches)
    max_w = max(p.shape[2] for p in all_patches)
    padded_patches = []
    padded_patches_ctx = []
    for p, p_ctx in zip(all_patches, all_patches_ctx):
        pad = torch.zeros((3, max_h, max_w))
        pad[:, :p.shape[1], :p.shape[2]] = p
        padded_patches.append(pad)
        
        pad_ctx = torch.zeros((3, max_h, max_w))
        pad_ctx[:, :p_ctx.shape[1], :p_ctx.shape[2]] = p_ctx
        padded_patches_ctx.append(pad_ctx)
        
    return {
        'images': images,
        'masks': masks,
        'segmentor_target_mask': seg_targets,
        'patches': torch.stack(padded_patches),
        'patches_with_context': torch.stack(padded_patches_ctx),  # ✅ NOVO
        'boxes': torch.stack(all_boxes),
        'heatmaps': torch.stack(all_heatmaps),
        'sample_indices': torch.tensor(all_indices, dtype=torch.long)
    }


if __name__ == "__main__":
    # Código de teste para validar o dataset
    dataset_path = os.path.join(project_root, 'data', 'boat')
    
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
