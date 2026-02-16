import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class HarmonizationDataset(Dataset):
    """
    Dataset para treinar modelos de Harmonização (Blending).
    Gera pares (Composite Image, Real Image) usando perturbações sintéticas.
    """
    def __init__(self, root_path, image_size=(512, 512), split='train', split_ratio=0.2):
        self.root_path = root_path
        self.image_size = image_size
        self.split = split
        
        # Carrega paths (Reutilizando lógica simplificada do AttentionDataset)
        self.image_paths, self.mask_paths = self._load_data_paths()
        
        # Split treino/val
        total = len(self.image_paths)
        val_size = int(total * split_ratio)
        indices = list(range(total))
        # Seed fixa para consistência entre runs
        random.Random(42).shuffle(indices)
        
        if split == 'train':
            self.indices = indices[val_size:]
        else:
            self.indices = indices[:val_size]
            
        print(f"[{split.upper()}] HarmonizationDataset: {len(self.indices)} imagens de {total} totais.")

        # Transformações para DEGRADAÇÃO (Composite)
        # Mais agressivas para simular colagens ruins
        self.composite_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ])
        
        # Transformações Base (Resize/ToTensor)
        self.to_tensor = transforms.ToTensor()

    def _load_data_paths(self):
        # Lógica simplificada de loading: assume estrutura data/images e data/masks ou similar
        # Tenta encontrar imagens
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        if os.path.isdir(os.path.join(self.root_path, 'images')):
             search_path = Path(self.root_path) / 'images'
        else:
             search_path = Path(self.root_path)
             
        for ext in exts:
            image_files.extend(list(search_path.glob(ext)))
            
        # Filtra masks
        valid_images = []
        valid_masks = []
        
        for img_path in sorted(image_files):
            # Tenta achar máscara com mesmo nome
            # 1. Tenta na pasta 'masks'
            mask_path_candidate = getattr(img_path.parent.parent / 'masks', 'joinpath', None)
            if mask_path_candidate:
                 mask_path = mask_path_candidate(img_path.name)
            else:
                 mask_path = Path(str(img_path).replace('images', 'masks'))
                 
            # Fallback: tentar replace na extensão ou sufixo se necessário (ex: _mask.png)
            if not mask_path.exists():
                 # Tenta sufixo comum
                 possible_names = [
                     img_path.stem + '_mask' + img_path.suffix,
                     img_path.stem + '.png' # Máscaras costumam ser PNG
                 ]
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
        
        # 1. Carregar Dados
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # Grayscale
        
        # Resize garantido
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)
        
        # Converte para Numpy para processamento
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Binariza máscara
        mask_bool = mask_np > 127
        
        if not np.any(mask_bool):
            # Se máscara vazia, retorna original sem alteração (fallback)
            real_tensor = self.to_tensor(image)
            return real_tensor, real_tensor, torch.zeros((1,) + self.image_size), torch.zeros((1,) + self.image_size)

        # 2. Criar Composite (Fabricar a colagem ruim)
        # Extrai objeto
        object_crop = image.crop(image.getbbox()) # Apenas para aplicar transform no PIL se quisessemos, mas temos mask total
        
        # Aplica transformações apenas ao objeto
        # Truque: Aplicamos transform na imagem TODA, e depois usamos a máscara para misturar
        perturbed_image = self.composite_transform(image)
        perturbed_np = np.array(perturbed_image)
        
        # Mistura: Onde é máscara, usa perturbed. Onde é fundo, usa original.
        composite_np = image_np.copy()
        composite_np[mask_bool] = perturbed_np[mask_bool]
        
        # 50% das vezes, altera também as bordas com um pouco de deslocamento ou blur extra na fronteira
        # Para simular recorte ruim (aliasing)
        
        # 3. Criar Boundary Mask (Fita na borda ou Objeto Todo)
        # UPDATED: Conceito "Variable Soft Mask" com Dinâmica e Proteção (90%)
        
        # Garante máscara binária para morfologia limpa
        mask_binary = (mask_bool.astype(np.uint8) * 255)
        
        # A. CÁLCULO DINÂMICO DE KERNELS (Baseado na dimensão do objeto)
        # Fundamental para lidar com objetos pequenos vs grandes sem destruir o miolo
        coords = cv2.findNonZero(mask_binary)
        if coords is not None:
            _, _, w, h = cv2.boundingRect(coords)
            min_dim = min(w, h)
        else:
            min_dim = 50 # Fallback
            
        # Kernel de Dilatação (Borda Externa): ~15%
        k_dil = max(5, int(min_dim * 0.15))
        # Kernel de Erosão (Miolo Protegido): ~10%
        k_ero = max(3, int(min_dim * 0.10))
        
        # B. Cria as Máscaras
        kernel_out = np.ones((k_dil, k_dil), np.uint8) 
        dilated_full = cv2.dilate(mask_binary, kernel_out, iterations=1)
        
        kernel_inner = np.ones((k_ero, k_ero), np.uint8)
        eroded_center = cv2.erode(mask_binary, kernel_inner, iterations=1)
        
        # C. Constroi a Máscara Gradual (Float)
        # Começa tudo a 0.0 (Fundo intocável)
        boundary_float = np.zeros_like(dilated_full, dtype=np.float32)
        
        # 1. Borda (1.0): Liberdade total para redesenhar transição
        boundary_float[dilated_full > 0] = 1.0
        
        # 2. Miolo (0.1): PROTEÇÃO ESTRUTURAL (90%)
        # Apenas permite ajuste muito subtil de cor/iluminação.
        # Isto impede o modelo de alucinar ou destruir a identidade do objeto.
        boundary_float[eroded_center > 0] = 0.1
        
        # Converte para Tensor (mas mantém float, não binariza!)
        # ToTensor espera [H, W, C] ou [H, W]. Retorna [C, H, W]
        boundary_tensor = torch.from_numpy(boundary_float).unsqueeze(0)
        
        # 4. Converter para Tensores [0, 1]
        real_tensor = self.to_tensor(image)
        composite_tensor = transforms.ToTensor()(Image.fromarray(composite_np))
        mask_tensor = transforms.ToTensor()(Image.fromarray(mask_np))
        # boundary_tensor já foi calculado acima com a soft mask
        
        # Retorna: Composite (Input), Real (Target), Mask, Boundary
        return {
            "pixel_values": composite_tensor,      # Input para o modelo (Imagem "Ruim")
            "ground_truth": real_tensor,           # Target (Imagem "Boa")
            "mask": mask_tensor,                   # Máscara do objeto
            "boundary_mask": boundary_tensor,      # Máscara da borda (Foco)
            "path": img_path
        }
