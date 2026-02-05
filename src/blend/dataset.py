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
        # UPDATED: Conceito "Variable Soft Mask"
        # - Borda (Exterior + Interior Próximo): Valor 1.0 (Liberdade Total para Blend)
        # - Centro do Objeto: Valor 0.5 (Liberdade Parcial para Luz/Cor, mas preserva Estrutura)
        
        # Garante máscara binária para morfologia limpa
        mask_binary = (mask_bool.astype(np.uint8) * 255)
        
        # A. Cria a zona de Borda Completa (Expandida para fora)
        # Aumentamos o contexto exterior (periferia) para ajudar no blend
        kernel_out = np.ones((50, 50), np.uint8) 
        dilated_full = cv2.dilate(mask_binary, kernel_out, iterations=1)
        
        # B. Identifica o "Miolo" (Centro seguro do objeto)
        # Erodimos AINDA MENOS (kernel minúsculo) para garantir que mesmo ovelhas pequenas
        # tenham um "miolo" protegido.
        kernel_inner = np.ones((5, 5), np.uint8) # Reduzido de 15 para 5
        eroded_center = cv2.erode(mask_binary, kernel_inner, iterations=1)
        
        # C. Constroi a Máscara Gradual (Float)
        # Começa tudo a 0.0 (Fundo intocável)
        boundary_float = np.zeros_like(dilated_full, dtype=np.float32)
        
        # Onde é 'dilated' (Objeto + Borda externa), definimos base 1.0
        boundary_float[dilated_full > 0] = 1.0
        
        # MAS, onde é 'eroded_center' (o miolo profundo), reduzimos para preservar identidade
        # 0.5: Preservação média (50%), permitindo troca de iluminação/cor
        boundary_float[eroded_center > 0] = 0.5
        
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
