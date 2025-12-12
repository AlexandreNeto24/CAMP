import os
import cv2
import torch
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
        else:
            self.aug_transform = None
            self.patch_only_transform = None
    
    def _load_data_paths(self):
        # Lógica robusta de procura de pastas
        possible_roots = [
            self.root_path,
            os.path.join(self.root_path, 'metaplasia'),
            os.path.join(self.root_path, '..', 'metaplasia')
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
            
        # 2. INPAINTING: Remover o objeto da imagem de destino (Blindfold Training)
        # O modelo vê a imagem SEM o objeto e tem de adivinhar onde ele estava
        image_for_model = image_np.copy()
        
        # if self.split == 'train' and self.p_remove > 0 and len(patch_boxes) > 0:
        #     if random.random() < self.p_remove:
        #         box = patch_boxes[0]
        #         x1, y1, x2, y2 = map(int, box)
                
        #         # Preenche com a média da borda (simula fundo limpo)
        #         border_y1, border_y2 = max(0, y1-5), min(image_np.shape[0], y2+5)
        #         border_x1, border_x2 = max(0, x1-5), min(image_np.shape[1], x2+5)
                
        #         region = image_np[border_y1:border_y2, border_x1:border_x2]
        #         if region.size > 0:
        #             mean_color = np.mean(region, axis=(0,1)).astype(np.uint8)
                    
        #             # Adiciona ruído para não ficar uma cor sólida artificial
        #             noise = np.random.normal(0, 5, (y2-y1, x2-x1, 3)).astype(np.int16)
        #             filled = mean_color + noise
        #             image_for_model[y1:y2, x1:x2] = np.clip(filled, 0, 255).astype(np.uint8)

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
        ✅ NOVO: Extrai patches E gera targets baseados em COMPATIBILIDADE DE CONTEXTO
        (não em posição original)
        """
        binary_mask = (mask_np > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        patches, patch_boxes, heatmaps, weights = [], [], [], []
        seg_target = np.zeros_like(binary_mask, dtype=np.float32)
        
        for i in range(1, num_labels):
            inst_mask = (labels == i).astype(np.uint8)
            if inst_mask.sum() < 100:
                continue  # Ignora ruído
            
            # ===== PASSO 1: Extrair contexto original (dilatação) =====
            kernel_size = int(max(stats[i][2], stats[i][3]) * 0.5)
            kernel_size = max(5, min(kernel_size, 50))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            dilated = cv2.dilate(inst_mask, kernel)
            context_ring = (dilated - inst_mask).astype(bool)
            
            # Marca contexto para segmentação
            seg_target[context_ring] = 0.9
            
            # ===== PASSO 2: Extrair cor média do contexto (compatibilidade) =====
            if context_ring.sum() > 0:
                context_colors = image_np[context_ring]  # [N, 3]
                context_mean_color = context_colors.mean(axis=0)
            else:
                context_mean_color = image_np.mean(axis=(0, 1))
            
            # ===== PASSO 3: Calcular mapa de compatibilidade de COR =====
            # Converte para LAB para melhor perceptibilidade
            image_lab = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
            context_mean_lab = cv2.cvtColor(
                np.array([[[int(context_mean_color[0]), int(context_mean_color[1]), int(context_mean_color[2])]]], 
                        dtype=np.uint8),
                cv2.COLOR_RGB2LAB
            )[0, 0].astype(np.float32)
            
            # Distância Euclidiana em LAB
            color_dist = np.sqrt(
                (image_lab[:, :, 0] - context_mean_lab[0]) ** 2 +
                (image_lab[:, :, 1] - context_mean_lab[1]) ** 2 +
                (image_lab[:, :, 2] - context_mean_lab[2]) ** 2
            )
            
            # Converte distância em COMPATIBILIDADE (menor distância = maior compatibilidade)
            # Normaliza por máximo para ter valores entre 0-1
            max_dist = color_dist.max() + 1e-6
            hm = 1.0 - (color_dist / max_dist)
            
            # ✅ CRUCIAL: Zera o próprio objeto (queremos ativar FUNDO, não objeto)
            hm[inst_mask > 0] = 0.0
            
            # Suaviza um pouco para evitar picos
            hm = cv2.GaussianBlur(hm.astype(np.float32), (5, 5), 1.0)
            
            hm = torch.tensor(hm, dtype=torch.float32)
            heatmaps.append(hm)
            
            # ===== PASSO 4: Extrair bounding box =====
            x, y, w, h, area = stats[i]
            patch_boxes.append(torch.tensor([x, y, x+w, y+h], dtype=torch.float32))
            
            # ===== PASSO 5: Extrair patch COM CONTEXTO (NÃO mascarado) =====
            # Patch com contexto original (sem remover fundo)
            expand_ratio = 1.5
            cx, cy = int(x + w // 2), int(y + h // 2)
            new_w, new_h = int(w * expand_ratio), int(h * expand_ratio)
            ctx_x1 = max(0, cx - new_w // 2)
            ctx_y1 = max(0, cy - new_h // 2)
            ctx_x2 = min(image_np.shape[1], cx + new_w // 2)
            ctx_y2 = min(image_np.shape[0], cy + new_h // 2)
            
            patch_with_context = image_np[ctx_y1:ctx_y2, ctx_x1:ctx_x2].copy()
            
            # Apenas o objeto (região original, sem mascaramento)
            patch_object_only = image_np[y:y+h, x:x+w].copy()
            
            # Redimensiona se necessário
            if self.patch_size:
                patch_with_context = cv2.resize(patch_with_context, self.patch_size)
                patch_object_only = cv2.resize(patch_object_only, self.patch_size)
            
            # Aplica augmentação ao patch com contexto
            if self.apply_aug and self.patch_only_transform:
                patch_with_context = self.patch_only_transform(image=patch_with_context)['image']
            
            # Armazena ambos no dicionário
            patches.append({
                'object_only': self.to_tensor(patch_object_only),
                'with_context': self.to_tensor(patch_with_context)
            })
            
            # Weight mask (dummy)
            weights.append(torch.ones((64, 64)))
        
        neg_mask = torch.zeros(image_np.shape[:2])
        return patches, patch_boxes, heatmaps, weights, neg_mask, torch.from_numpy(seg_target)

    def _create_gaussian_heatmap(self, size, center, sigma):
        h, w = size
        x0, y0 = center
        x = np.arange(0, w, 1, float)
        y = np.arange(0, h, 1, float)[:, np.newaxis]
        return torch.tensor(np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2)), dtype=torch.float32)

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
    dataset_path = os.path.join(project_root, 'data', 'metaplasia')
    
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
