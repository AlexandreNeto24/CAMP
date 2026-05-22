import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import cv2

from src.blend.data.synthesis import create_composite, calculate_boundary_mask
from src.blend.data.synthesis import calculate_boundary_mask

# 1.5. Redefinição do Dataset com "Open Context Ring" + "Soft Matte" (Poisson-Style)
# Redefinimos a classe aqui para aumentar o anel e SUAVIZAR a máscara (Blur)

class HarmonizationDataset(Dataset):
    """
    Versão "Poisson-Style" do Dataset.
    - Anel Externo Largo (30%)
    - Máscara Suavizada (Gaussian Blur) para evitar cortes duros (Hard Cuts)
    """
    def __init__(self, root_path, image_size=(512, 512), split='train', split_ratio=0.2):
        self.root_path = root_path
        self.image_size = image_size
        self.split = split
        
        self.image_paths, self.mask_paths = self._load_data_paths()
        
        total = len(self.image_paths)
        val_size = int(total * split_ratio)
        indices = list(range(total))
        random.Random(42).shuffle(indices)
        
        if split == 'train':
            self.indices = indices[val_size:]
        else:
            self.indices = indices[:val_size]
            
        print(f"[{split.upper()}] HarmonizationDataset (Poisson-Style): {len(self.indices)} imagens.")

        self.composite_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ])
        
        self.to_tensor = transforms.ToTensor()

    def _load_data_paths(self):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        if os.path.isdir(os.path.join(self.root_path, 'images')):
             search_path = Path(self.root_path) / 'images'
        else:
             search_path = Path(self.root_path)
        for ext in exts:
            image_files.extend(list(search_path.glob(ext)))
            
        valid_images = []
        valid_masks = []
        for img_path in sorted(image_files):
            mask_path_candidate = getattr(img_path.parent.parent / 'masks', 'joinpath', None)
            if mask_path_candidate:
                 mask_path = mask_path_candidate(img_path.name)
            else:
                 mask_path = Path(str(img_path).replace('images', 'masks'))
                 
            if not mask_path.exists():
                 possible_names = [img_path.stem + '_mask' + img_path.suffix, img_path.stem + '.png']
                 for name in possible_names:
                     if (img_path.parent.parent / 'masks' / name).exists():
                         mask_path = img_path.parent.parent / 'masks' / name
                         break
            
            if mask_path.exists():
                valid_images.append(str(img_path))
                valid_masks.append(str(mask_path))
        return valid_images, valid_masks

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path = self.image_paths[real_idx]
        mask_path = self.mask_paths[real_idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') 
        
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)
        
        image_np = np.array(image)
        mask_np = np.array(mask)
        mask_bool = mask_np > 127
        
        if not np.any(mask_bool):
            real_tensor = self.to_tensor(image)
            return {
                "pixel_values": real_tensor,
                "ground_truth": real_tensor,
                "mask": torch.zeros((1,)+self.image_size),
                "boundary_mask": torch.zeros((1,)+self.image_size),
                "path": img_path
            }

        # Criar Composite
        perturbed_image = self.composite_transform(image)
        perturbed_np = np.array(perturbed_image)
        composite_np = image_np.copy()
        composite_np[mask_bool] = perturbed_np[mask_bool]
        
        # --- Boundary mask: valores altos na instância, gradiente suave na borda ---
        mask_binary = (mask_bool.astype(np.uint8) * 255)
        # Usar função avançada para soft mask
        # Novo modo: inclui expansão para fora da instância
        boundary_mask = calculate_boundary_mask(mask_binary, mode='dovenet_expanded', smooth=True)
        boundary_mask = np.clip(boundary_mask, 0, 1)
        composite_tensor = self.to_tensor(Image.fromarray(composite_np))
        real_tensor = self.to_tensor(image)
        mask_tensor = self.to_tensor(mask)
        boundary_tensor = torch.from_numpy(boundary_mask).unsqueeze(0)
        return {
            "pixel_values": composite_tensor,
            "ground_truth": real_tensor,
            "mask": mask_tensor, # Usada para loss local
            "boundary_mask": boundary_tensor, # Usada para mistura final suave
            "path": img_path
        }