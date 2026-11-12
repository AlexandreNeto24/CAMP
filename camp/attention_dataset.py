"""Image and mask samples for CAMP training.

The segmentor reads the image and a context-ring target. The recommender
also reads patches, context crops, and a colour heatmap. ``for_segmentor``
skips the heatmap, the clean-image tensor, weight masks, and box tensors.
"""

import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

try:
    import albumentations as A
except ImportError:
    A = None

class AttentionDataset(Dataset):
    """Paired images and masks, with an optional lighter segmentor path.

    ``for_segmentor=True`` still builds the context ring (value 0.9) and still
    runs image augmentation, patch augmentation, and blindfold. It does not
    build LAB heatmaps, the clean-image tensor, weight masks, or box tensors.
    Patch crops stay so the augmentation random stream is unchanged.
    """
    def __init__(self, root_path, image_size=(512, 512), patch_size=None, 
                 transform=None, hyp=None, apply_aug=False,
                 split_ratio=0.2, split_mode='none', split='all', p_remove=0.0,
                 class_name=None, for_segmentor=False):
        self.root_path = root_path
        self.image_size = image_size
        self.patch_size = patch_size
        self.transform = transform
        self.hyp = hyp
        self.apply_aug = apply_aug
        self.split = split
        self.class_name = class_name
        
        # ✅ NOVO COMPORTAMENTO:
        # - Treino: Remove objeto 20% das vezes (para robreasttez)
        # - Validação: NUNCA remove (testa em cenário realista)
        self.p_remove = float(p_remove)
        self.for_segmentor = bool(for_segmentor)

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
            # Augmentação GLOBAL (Imagem + Másbreasta)
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
            
            # Augmentação EXCLUSIVA DO PATCH
            self.patch_only_transform = A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ToGray(p=0.1),
            ])
            
            # Augmentação do CONTEXTO
            self.context_aug = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.Blur(blur_limit=7, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.8),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.6),
            ])
        else:
            self.aug_transform = None
            self.patch_only_transform = None
            self.context_aug = None
    
    def _load_data_paths(self):
        possible_roots = [self.root_path]
        
        if self.class_name:
            possible_roots.extend([
                os.path.join(self.root_path, self.class_name),
                os.path.join(self.root_path, '..', self.class_name)
            ])
        else:
            # Fallback for retro-compatibility if needed, or just default to common ones
            # But better to rely on root_path or class_name
            possible_roots.extend([
                os.path.join(self.root_path, 'breast'),
                os.path.join(self.root_path, '..', 'breast')
            ])
        
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
            
        # 1. Limitar patches no treino
        if self.split == 'train' and len(patches) > 0:
            idx = random.randint(0, len(patches) - 1)
            patches = [patches[idx]]
            patch_boxes = [patch_boxes[idx]]
            if target_heatmaps:
                target_heatmaps = [target_heatmaps[idx]]
            if weight_masks:
                weight_masks = [weight_masks[idx]]
            
            # Augmentação de contexto
            if self.apply_aug and self.context_aug is not None:
                for i in range(len(patches)):
                    if isinstance(patches[i], dict):
                        ctx = patches[i]['with_context'].numpy().transpose(1, 2, 0)
                        ctx = (ctx * 255).astype(np.uint8)
                        aug_ctx = self.context_aug(image=ctx)['image']
                        patches[i]['with_context'] = self.to_tensor(aug_ctx)
                        
        # 2. INPAINTING: Remover o objeto da imagem de destino (Blindfold Training)
        image_for_model = image_np.copy()
        inpainting_mask = np.zeros(image_np.shape[:2], dtype=np.float32)
        
        # ✅ NOVA LÓGICA: Validação AGORA TAMBÉM remove (para avaliação honesta)
        # Antes: # self.split == 'train'  # ✅ APENAS no treino
        # Agora: removemos para todos se p_remove > 0
        should_blindfold = (
            self.p_remove > 0 and 
            len(patch_boxes) > 0 
        )
        
        if should_blindfold and random.random() < self.p_remove:
            box = patch_boxes[0]
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            
            img_h, img_w = image_np.shape[:2]
            obj_area = w * h
            img_area = img_h * img_w
            ratio = obj_area / img_area
            
            if ratio > 0.30:
                if self.split == 'train':
                    strategy = random.choice(['blur', 'central_cutout', 'central_cutout'])
                else:
                    strategy = 'central_cutout'
            else:
                if self.split == 'train':
                    strategy = random.choice(['remove', 'remove', 'blur', 'noise'])
                else:
                    strategy = 'remove' 
            
            # --- APLICAÇÃO DAS ESTRATÉGIAS ---
            
            if strategy == 'remove':
                color = np.array([128, 128, 128], dtype=np.uint8)
                image_for_model[y1:y2, x1:x2] = color
                inpainting_mask[y1:y2, x1:x2] = 1.0  # ✅ Marca a região
                
            elif strategy == 'noise':
                roi = image_for_model[y1:y2, x1:x2].astype(np.float32)
                noise = np.random.normal(0, 30, roi.shape)
                roi = np.clip(roi + noise, 0, 255).astype(np.uint8)
                image_for_model[y1:y2, x1:x2] = roi
                inpainting_mask[y1:y2, x1:x2] = 1.0  # ✅ Marca a região
                
            elif strategy == 'blur':
                roi = image_for_model[y1:y2, x1:x2]
                ksize = max(15, min(w, h) // 4 | 1) 
                roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                image_for_model[y1:y2, x1:x2] = roi
                inpainting_mask[y1:y2, x1:x2] = 1.0  # ✅ Marca a região
                
            elif strategy == 'central_cutout':
                margin_w = int(w * 0.2)
                margin_h = int(h * 0.2)
                cx1 = x1 + margin_w
                cy1 = y1 + margin_h
                cx2 = x2 - margin_w
                cy2 = y2 - margin_h
                
                if cx2 > cx1 and cy2 > cy1:
                    # ✅ INPAINTING em vez de Gray Box
                    # Cria másbreasta local para inpaint
                    mask_inpaint = np.zeros(image_np.shape[:2], dtype=np.uint8)
                    mask_inpaint[cy1:cy2, cx1:cx2] = 255
                    
                    # Usa Telea inpainting (rápido e eficaz para texturas simples)
                    # Radius 3 é típico
                    try:
                        image_inpainted = cv2.inpaint(image_for_model, mask_inpaint, 3, cv2.INPAINT_TELEA)
                        image_for_model = image_inpainted
                    except Exception as e:
                        # Fallback se falhar
                        image_for_model[cy1:cy2, cx1:cx2] = 128
                        
                    inpainting_mask[cy1:cy2, cx1:cx2] = 1.0  # ✅ Marca a região

        image_tensor = self.to_tensor(image_for_model)
        if self.for_segmentor:
            return {
                'image': image_tensor,
                'segmentor_target_mask': seg_target.unsqueeze(0),
                'image_path': img_path,
            }
        clean_image_tensor = self.to_tensor(image_np) # ✅ CLEAN IMAGE (Original Augmentada)
        mask_tensor = self.to_tensor((mask_np > 0).astype(np.float32))
        inpainting_mask_tensor = torch.from_numpy(inpainting_mask).float()  # ✅ Novo tensor
        
        return {
            'image': image_tensor,
            'clean_image': clean_image_tensor, # ✅ Retorna imagem limpa
            'mask': mask_tensor,
            'segmentor_target_mask': seg_target.unsqueeze(0),
            'inpainting_mask': inpainting_mask_tensor,  # ✅ RETORNA A MÁSbreastA
            'patches': patches,
            'patches_with_context': [p['with_context'] if isinstance(p, dict) else p for p in patches],
            'patches_context_masks': [p['context_mask'] if isinstance(p, dict) else torch.zeros((1, 1, 1)) for p in patches], # ✅ EXTRAI MÁSbreastA
            'patch_boxes': patch_boxes,
            'target_heatmaps': target_heatmaps,
             'negative_mask': neg_mask,
            'weight_masks': weight_masks,
            'image_path': img_path
        }

    def _extract_patches_and_targets_with_negatives(self, image_np, mask_np):
        binary_mask = (mask_np > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        patches, patch_boxes, heatmaps, weights = [], [], [], []
        seg_target = np.zeros_like(binary_mask, dtype=np.float32)
        
        image_for_heatmap = image_np.copy()
        
        for i in range(1, num_labels):
            inst_mask = (labels == i).astype(np.uint8)
            if inst_mask.sum() < 100:
                continue
            
            # Contexto
            kernel_size = int(max(stats[i][2], stats[i][3]) * 0.5)
            kernel_size = max(5, min(kernel_size, 50))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            dilated = cv2.dilate(inst_mask, kernel)
            context_ring = (dilated - inst_mask).astype(bool)
            
            seg_target[context_ring] = 0.9
            
            # Heatmap. The segmentor never reads it.
            if not self.for_segmentor:
                hm = self._compute_color_heatmap(image_for_heatmap, inst_mask, context_ring)
                heatmaps.append(hm)
            
            # Box. Segmentor training only needs the coordinates for blindfold.
            x, y, w, h, area = stats[i]
            if self.for_segmentor:
                patch_boxes.append((int(x), int(y), int(x + w), int(y + h)))
            else:
                patch_boxes.append(torch.tensor([x, y, x+w, y+h], dtype=torch.float32))
            
            # Patches
            patch_object_only = image_np[y:y+h, x:x+w].copy()
            
            # ✅ Create mask for visualization, BUT keep patch_object_only RAW for model input consistency
            obj_mask_tight = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
            # patch_object_only = cv2.bitwise_and(patch_object_only, patch_object_only, mask=obj_mask_tight) # REMOVIDO para não alterar input do modelo
            
            # ✅ AUMENTADO: De 1.5x para 3.0x para garantir que apanhamos
            # bastante fundo para o novo descritor de contexto.
            expand_ratio = 3.0 
            cx, cy = x + w // 2, y + h // 2
            new_w, new_h = int(w * expand_ratio), int(h * expand_ratio)
            ctx_x1 = max(0, cx - new_w // 2)
            ctx_y1 = max(0, cy - new_h // 2)
            ctx_x2 = min(image_np.shape[1], cx + new_w // 2)
            ctx_y2 = min(image_np.shape[0], cy + new_h // 2)
            
            patch_with_context = image_np[ctx_y1:ctx_y2, ctx_x1:ctx_x2].copy()
            
            # ✅ CORREÇÃO CRÍTICA: Extrair a másbreasta real do objeto no contexto
            # Usamos labels == i para pegar APENAS a másbreasta deste objeto específico
            mask_full_slice = (labels[ctx_y1:ctx_y2, ctx_x1:ctx_x2] == i).astype(np.uint8) * 255
            patch_mask_context = mask_full_slice.copy()
            
            if self.patch_size:
                patch_object_only = cv2.resize(patch_object_only, self.patch_size)
                # Resize da másbreasta objeto com NEAREST
                obj_mask_tight = cv2.resize(obj_mask_tight, self.patch_size, interpolation=cv2.INTER_NEAREST)
                
                patch_with_context = cv2.resize(patch_with_context, self.patch_size)
                # Resize da másbreasta com NEAREST para manter binário
                patch_mask_context = cv2.resize(patch_mask_context, self.patch_size, interpolation=cv2.INTER_NEAREST)
            
            if self.apply_aug and self.patch_only_transform:
                patch_object_only = self.patch_only_transform(image=patch_object_only)['image']
            
            patches.append({
                'object_only': self.to_tensor(patch_object_only),
                'object_mask': self.to_tensor(obj_mask_tight * 255), # ✅ MÁSCARA EXPLICITA
                'with_context': self.to_tensor(patch_with_context),
                'context_mask': self.to_tensor(patch_mask_context) # ✅ Passa a másbreasta real (1=Objeto, 0=Fundo)
            })
            
            if not self.for_segmentor:
                weights.append(torch.ones((64, 64)))
        
        neg_mask = torch.zeros(image_np.shape[:2])
        return patches, patch_boxes, heatmaps, weights, neg_mask, torch.from_numpy(seg_target)
    
    def _compute_color_heatmap(self, image_np, inst_mask, context_ring):
        if context_ring.sum() > 0:
            context_colors = image_np[context_ring]
        else:
            context_colors = image_np.reshape(-1, 3)
        
        context_rgb = context_colors.astype(np.uint8)
        context_lab = cv2.cvtColor(context_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)
        context_mean_lab = context_lab.mean(axis=1).squeeze()
        
        image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        color_dist = np.sqrt(
            (image_lab[:, :, 0] - context_mean_lab[0]) ** 2 +
            (image_lab[:, :, 1] - context_mean_lab[1]) ** 2 +
            (image_lab[:, :, 2] - context_mean_lab[2]) ** 2
        )
        
        max_dist = color_dist.max() + 1e-6
        heatmap = 1.0 - (color_dist / max_dist)
        
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad = cv2.Laplacian(gray, cv2.CV_32F)
        grad = np.abs(grad) / (grad.max() + 1e-6)
        
        heatmap = 0.8 * heatmap + 0.2 * grad
        
        heatmap[inst_mask > 0] = 0.0
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        obj_dilated = cv2.dilate(inst_mask.astype(np.uint8), kernel_dilate)
        heatmap[obj_dilated > 0] = 0.0
        
        heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (5, 5), 1.0)
        heatmap = np.clip(heatmap, 0, 1).astype(np.float32)
        
        return torch.tensor(heatmap, dtype=torch.float32)

def attention_collate(batch):
    images = torch.stack([b['image'] for b in batch])
    clean_images = torch.stack([b['clean_image'] for b in batch]) # ✅ CLEAN IMAGES
    masks = torch.stack([b['mask'] for b in batch])
    seg_targets = torch.stack([b['segmentor_target_mask'] for b in batch])
    inpainting_masks = torch.stack([b['inpainting_mask'] for b in batch])  # ✅ NOVO
    
    all_patches, all_patches_ctx, all_masks_ctx, all_boxes, all_heatmaps, all_indices = [], [], [], [], [], []
    
    for i, b in enumerate(batch):
        for p, p_ctx, box, hm in zip(b['patches'], b['patches_with_context'], b['patch_boxes'], b['target_heatmaps']):
            patch_obj = p['object_only'] if isinstance(p, dict) else p
            patch_ctx = p_ctx if isinstance(p_ctx, torch.Tensor) else p_ctx
            
            # ✅ Extrai a másbreasta se disponível
            mask_ctx = p['context_mask'] if isinstance(p, dict) and 'context_mask' in p else torch.zeros_like(patch_ctx[:1])
            
            all_patches.append(patch_obj)
            all_patches_ctx.append(patch_ctx)
            all_masks_ctx.append(mask_ctx)
            all_boxes.append(box)
            all_heatmaps.append(hm)
            all_indices.append(i)
            
    if not all_patches:
        return {
            'images': images, 
            'patches': torch.empty(0), 
            'patches_with_context': torch.empty(0),
            'heatmaps': torch.empty(0),
            'sample_indices': torch.empty(0),
            'segmentor_target_mask': seg_targets,
            'image_paths': [item.get('image_path', '') for item in batch],
        }
        
    max_h = max(p.shape[1] for p in all_patches)
    max_w = max(p.shape[2] for p in all_patches)
    padded_patches = []
    padded_patches_ctx = []
    padded_masks_ctx = []
    
    for p, p_ctx, m_ctx in zip(all_patches, all_patches_ctx, all_masks_ctx):
        pad = torch.zeros((3, max_h, max_w))
        pad[:, :p.shape[1], :p.shape[2]] = p
        padded_patches.append(pad)
        
        pad_ctx = torch.zeros((3, max_h, max_w))
        pad_ctx[:, :p_ctx.shape[1], :p_ctx.shape[2]] = p_ctx
        padded_patches_ctx.append(pad_ctx)

        pad_mask = torch.zeros((1, max_h, max_w))
        pad_mask[:, :m_ctx.shape[1], :m_ctx.shape[2]] = m_ctx
        padded_masks_ctx.append(pad_mask)
        
    return {
        'images': images,
        'clean_images': clean_images, # ✅ Retorna clean images
        'masks': masks,
        'segmentor_target_mask': seg_targets,
        'inpainting_masks': inpainting_masks,  # ✅ NOVO
        'patches': torch.stack(padded_patches),
        'patches_with_context': torch.stack(padded_patches_ctx),
        'patches_context_masks': torch.stack(padded_masks_ctx), # ✅ Retorna másbreastas
        'boxes': torch.stack(all_boxes),
        'heatmaps': torch.stack(all_heatmaps),
        'sample_indices': torch.tensor(all_indices, dtype=torch.long),
        'image_paths': [item.get('image_path', '') for item in batch],
    }


def segmentor_collate(batch):
    """Stack the image and the context-ring target used by ``SegmentorTrainer``."""
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'segmentor_target_mask': torch.stack(
            [item['segmentor_target_mask'] for item in batch]
        ),
        'image_paths': [item.get('image_path', '') for item in batch],
    }
