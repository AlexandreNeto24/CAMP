import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def calculate_boundary_mask(mask_numpy, mode='wide', smooth=False):
    """
    Calcula a Boundary Mask (Soft) para guiar a harmonização.
    
    Modos:
      - 'wide':   Treino DoveNet — anel largo (30%), valores discretos para guia espacial.
      - 'narrow': Inference Diffusion — anel estreito (12%), valores discretos.
      - 'blend':  Blending final — gradiente SUAVE baseado em distance transform.
                  Sem degraus, transição contínua para fusão natural com o fundo.
    
    Args:
        mask_numpy (np.array): Máscara binária.
        mode (str): 'wide', 'narrow' ou 'blend'.
        smooth (bool): Se True, aplica Gaussian blur no resultado final (compatível c/ modos antigos).
    """
    # Garante máscara binária (0-255 uint8)
    mask_uint8 = (mask_numpy > 127).astype(np.uint8) * 255
    
    # A. CÁLCULO DINÂMICO DE KERNELS (Baseado na dimensão do objeto)
    coords = cv2.findNonZero(mask_uint8)
    if coords is not None:
        _, _, w, h = cv2.boundingRect(coords)
        min_dim = min(w, h)
    else:
        min_dim = 100 # Fallback

    # =====================================================
    # MODO 'blend': Gradiente suave via distance transform
    # Cria transição contínua 1.0 (centro) → 0.0 (fundo)
    # SEM degraus — a instância funde-se naturalmente no fundo
    # =====================================================
    if mode == 'blend' or mode == 'dovenet_expanded':
        # Lógica igual ao blend_v4: anel largo (1.0), núcleo (0.1), fora (0.0), blur opcional
        mask_bin = (mask_uint8 > 0).astype(np.uint8)
        coords = cv2.findNonZero(mask_bin)
        if coords is not None:
            _, _, w, h = cv2.boundingRect(coords)
            min_dim = min(w, h)
        else:
            min_dim = 100
        # Anel largo (30% do objeto)
        k_dil = max(11, int(min_dim * 0.30))
        if k_dil % 2 == 0: k_dil += 1
        # Núcleo (10% do objeto)
        k_ero = max(3, int(min_dim * 0.10))
        kernel_out = np.ones((k_dil, k_dil), np.uint8)
        dilated_full = cv2.dilate(mask_uint8, kernel_out, iterations=1)
        kernel_inner = np.ones((k_ero, k_ero), np.uint8)
        eroded_center = cv2.erode(mask_uint8, kernel_inner, iterations=1)
        boundary_float = np.zeros_like(dilated_full, dtype=np.float32)
        # Anel largo: 1.0
        boundary_float[dilated_full > 0] = 1.0
        # Núcleo: 0.1
        boundary_float[eroded_center > 0] = 0.1
        # Blur opcional
        if smooth:
            k_blur = max(5, int(min_dim * 0.10))
            if k_blur % 2 == 0: k_blur += 1
            boundary_float = cv2.GaussianBlur(boundary_float, (k_blur, k_blur), 0)
        boundary_float = np.clip(boundary_float, 0, 1)
        return boundary_float
        
    if mode == 'wide':
        # --- MODO TREINO/DOVENET (Lógica V2 Original) ---
        # Anel Largo (30%) para o DoveNet ter contexto suficiente para corrigir cores.
        k_dil = max(11, int(min_dim * 0.30))
        if k_dil % 2 == 0: k_dil += 1
        
        # Erosão Média (10%) para proteger núcleo
        k_ero = max(3, int(min_dim * 0.10))
        
    else: # mode == 'narrow'
        # --- MODO INFERENCE/DIFFUSION (Lógica V2 Refinement) ---
        # Anel Estreito (12-15%) para apenas tratar a sombra de contacto.
        k_dil = max(10, int(min_dim * 0.12))
        
        # Erosão Mínima (3px) para Preservação MÁXIMA da instância.
        k_ero = 3 
    
    # B. Cria as Máscaras
    kernel_out = np.ones((k_dil, k_dil), np.uint8) 
    dilated_full = cv2.dilate(mask_uint8, kernel_out, iterations=1)
    
    kernel_inner = np.ones((k_ero, k_ero), np.uint8)
    eroded_center = cv2.erode(mask_uint8, kernel_inner, iterations=1)
    
    # C. Constroi a Máscara Gradual (Float)
    boundary_float = np.zeros_like(dilated_full, dtype=np.float32)
    
    # 1. Borda (1.0): Liberdade total
    boundary_float[dilated_full > 0] = 1.0
    
    # 2. Miolo (0.1): PROTEÇÃO ESTRUTURAL
    boundary_float[eroded_center > 0] = 0.1
    
    # D. Suavização opcional (para modos antigos usados em blending)
    if smooth:
        k_blur = max(5, int(min_dim * 0.10))
        if k_blur % 2 == 0: k_blur += 1
        boundary_float = cv2.GaussianBlur(boundary_float, (k_blur, k_blur), 0)
    
    return boundary_float

def create_composite(image, mask, jitter_transform=None):
    """
    Cria uma imagem composta 'falsa' (Composite) aplicando perturbações 
    no objeto definido pela máscara para simular colagem ruim.
    
    Args:
        image (PIL.Image): Imagem original harmonizada (Ground Truth).
        mask (PIL.Image): Máscara do objeto.
        jitter_transform (callable): Transformada de cor/blur.
        
    Returns:
        composite_tensor (torch.Tensor)
        mask_float (torch.Tensor): Máscara original binarizada [0,1]
    """
    image_np = np.array(image)
    mask_np = np.array(mask)
    mask_bool = mask_np > 127
    
    if not np.any(mask_bool):
        # Fallback sem alterações
        return transforms.ToTensor()(image), torch.from_numpy(mask_bool).float()

    # Aplica perturbações na imagem TODA (mas só usaremos a parte mascarada)
    if jitter_transform:
        perturbed_image = jitter_transform(image)
    else:
        # Default jitter if none provided (MATCHES V2 EXACTLY)
        default_jitter = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ])
        perturbed_image = default_jitter(image)
        
    perturbed_np = np.array(perturbed_image)
    
    # Mistura: Onde é máscara, usa perturbed. Onde é fundo, usa original.
    composite_np = image_np.copy()
    composite_np[mask_bool] = perturbed_np[mask_bool]
    
    return transforms.ToTensor()(Image.fromarray(composite_np)), transforms.ToTensor()(mask)
