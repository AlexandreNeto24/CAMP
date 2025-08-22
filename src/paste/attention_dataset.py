import os
import sys
import cv2
import random
import numpy as np
import torch
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
                 split_ratio=0.2, split_mode='none', split='all'):
        self.root_path = root_path
        self.image_size = image_size
        self.patch_size = patch_size
        self.transform = transform
        self.hyp = hyp
        self.apply_aug = apply_aug
        
        # Transformações padrão
        self.to_tensor = transforms.ToTensor()
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
        if self.apply_aug and self.hyp:
            try:
                import albumentations as A
                self.aug_transform = A.Compose([
                    # Geométricas
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.3),
                    A.RandomRotate90(p=0.3),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20, p=0.4),
                    A.GridDistortion(p=0.2),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=30, p=0.2),

                    # Fotométricas
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.3),
                    A.CLAHE(p=0.2),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
                    A.Blur(blur_limit=5, p=0.2),
                    A.MedianBlur(blur_limit=5, p=0.1),
                    A.MotionBlur(blur_limit=5, p=0.1),

                    # Oclusão e mistura
                    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
                    A.Cutout(num_holes=4, max_h_size=32, max_w_size=32, p=0.3),
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
        
        # Converte para tensores
        image_tensor = self.to_tensor(image_np)
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

        # --- NOVA LÓGICA DE CONTEXTO NATURAL ---
        CONTEXT_TARGET_VALUE = 1.0
        OBJECT_TARGET_VALUE = 0.5

        # Cria a máscara de contexto ("anel") usando dilatação morfológica
        kernel_size = 80  # Ajuste conforme necessário para o seu contexto
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        context_mask = (dilated_mask - binary_mask).astype(np.uint8)

        # Combina as máscaras para criar o alvo final ponderado
        segmentor_target_mask_np = np.zeros_like(binary_mask, dtype=np.float32)
        segmentor_target_mask_np[context_mask == 1] = CONTEXT_TARGET_VALUE
        segmentor_target_mask_np[binary_mask == 1] = OBJECT_TARGET_VALUE

        # Encontra componentes conectados (lesões individuais)
        binary_mask_for_stats = binary_mask * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_for_stats, connectivity=8)

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
            if w < 10 or h < 10 or area < 100:
                continue
            all_positive_boxes.append((x, y, w, h))

        for i, (x, y, w, h) in enumerate(all_positive_boxes):
            # Extrai o patch da imagem original usando a bounding box da instância
            patch = image_np[y:y+h, x:x+w].copy()

            # Máscara da instância dentro do patch
            instance_mask_in_patch = (mask_np[y:y+h, x:x+w] > 0).astype(np.uint8)

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
            patch_tensor = self.to_tensor(patch)
            box = [x, y, x+w, y+h]

            cx, cy = centroids[i+1]
            target_heatmap = self._create_gaussian_heatmap(
                (img_h, img_w), (cx, cy), sigma=max(w, h) / 2
            )

            patches.append(patch_tensor)
            patch_boxes.append(torch.tensor(box, dtype=torch.float32))
            target_heatmaps.append(target_heatmap)

        # --- Patches Negativos (Background) ---
        num_positives = len(all_positive_boxes)
        num_negatives_to_sample = num_positives * num_neg_per_pos

        if num_negatives_to_sample > 0:
            dilated_lesion_mask = cv2.dilate(binary_mask, np.ones((30, 30), np.uint8), iterations=1)
            safe_background_mask = (dilated_lesion_mask == 0).astype(np.uint8)
            background_contours, _ = cv2.findContours(safe_background_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            sampled_negatives = 0
            attempts = 0
            while sampled_negatives < num_negatives_to_sample and attempts < 50 and background_contours:
                attempts += 1
                contour = random.choice(background_contours)
                x_bg, y_bg, w_bg, h_bg = cv2.boundingRect(contour)

                patch_h = random.randint(min(30, h_bg if h_bg > 0 else 30), min(100, h_bg if h_bg > 0 else 100))
                patch_w = random.randint(min(30, w_bg if w_bg > 0 else 30), min(100, w_bg if w_bg > 0 else 100))

                if w_bg < patch_w or h_bg < patch_h:
                    continue

                rand_x = random.randint(x_bg, x_bg + w_bg - patch_w)
                rand_y = random.randint(y_bg, y_bg + h_bg - patch_h)

                patch = image_np[rand_y:rand_y+patch_h, rand_x:rand_x+patch_w].copy()

                if self.patch_size:
                    patch = cv2.resize(patch, self.patch_size)
                patch_tensor = self.to_tensor(patch)
                target_heatmap = torch.zeros((img_h, img_w), dtype=torch.float32)

                ph, pw = patch.shape[:2]
                weight_masks.append(torch.ones((ph, pw), dtype=torch.float32))

                patches.append(patch_tensor)
                patch_boxes.append(torch.tensor([0,0,0,0], dtype=torch.float32))
                target_heatmaps.append(target_heatmap)
                sampled_negatives += 1

        # --- Máscara Negativa Global ---
        global_negative_mask_np = np.zeros((img_h, img_w), dtype=np.float32)
        for x, y, w, h in all_positive_boxes:
            global_negative_mask_np[y:y+h, x:x+w] = 1.0
        global_negative_mask = torch.from_numpy(global_negative_mask_np)

        if not patches:
            return [], [], [], [], torch.zeros((img_h, img_w), dtype=torch.float32), torch.from_numpy(segmentor_target_mask_np)

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