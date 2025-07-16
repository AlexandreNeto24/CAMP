import os
import numpy as np
import cv2
import random
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import albumentations as A

class CustomCropDataset(Dataset):
    def __init__(self, root_path, image_size=(640, 640), transform=None, hyp=None):
        """
        Dataset otimizado para ROI Align - retorna apenas imagens, máscaras e coordenadas
        
        Args:
            root_path: Caminho para pasta com imagens e máscaras
            image_size: Tamanho padrão das imagens (640, 640)
            transform: Transformações adicionais
            hyp: Hiperparâmetros para augmentação
        """
        self.root_path = root_path
        self.image_size = image_size
        self.hyp = hyp
        self.to_tensor = transforms.ToTensor()
        self.resize_transform = transforms.Resize(self.image_size)
        
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
        
    def __len__(self):
        return len(self.lesion_image_paths)
    
    def __getitem__(self, idx):
        # Aplicar mosaic augmentation com probabilidade baseada em hiperparâmetros
        mosaic_prob = self.hyp.get('mosaic', 1.0) if self.hyp else 0.0
        use_mosaic = random.random() < mosaic_prob

        # Verificar se estamos perto do fim do treino para desativar mosaic
        if hasattr(self, 'current_epoch') and hasattr(self, 'total_epochs'):
            close_mosaic = self.hyp.get('close_mosaic', 10) if self.hyp else 0
            if self.current_epoch >= self.total_epochs - close_mosaic:
                use_mosaic = False

        # Aplicar mosaic se necessário
        if use_mosaic:
            # Seleciona 3 imagens aleatórias diferentes
            indices = [idx] + random.sample([i for i in range(len(self)) if i != idx], 3)
            image, mask, lesion_boxes = mosaic_augmentation(self, indices, image_size=self.image_size)
        else:
            # Carregar imagem e máscara normalmente
            image_path = self.lesion_image_paths[idx]
            mask_path = self.lesion_masks_paths[idx]
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            
            # Redimensionar para tamanho padrão
            image = self.resize_transform(image)
            mask = self.resize_transform(mask)
            
            # Extrair boxes das instâncias de lesão
            lesion_boxes = self._extract_instance_boxes(image, mask)

        # Converter para numpy para augmentations
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Aplicar augmentations se existirem
        if self.transform:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_np = augmented['image']
            mask_np = augmented['mask']

        # Converter de volta para PIL
        image = Image.fromarray(image_np)
        mask = Image.fromarray(mask_np)

        # Obter as coordenadas YOLO das instâncias de lesão
        lesion_boxes = self._extract_instance_boxes(image, mask)
        
        # Gerar coordenadas YOLO para regiões de background (expandindo as boxes da lesão)
        background_boxes = self._extract_background_boxes(image, mask, expand_ratio=1.5)
        
        # Combinar todas as boxes (lesão + background)
        all_boxes = torch.cat([torch.tensor(lesion_boxes), torch.tensor(background_boxes)], dim=0) if len(background_boxes) > 0 else torch.tensor(lesion_boxes)
        
        # Criar rótulos para as boxes (1=lesão, 0=background)
        box_labels = torch.cat([
            torch.ones(len(lesion_boxes)), 
            torch.zeros(len(background_boxes))
        ]) if len(background_boxes) > 0 else torch.ones(len(lesion_boxes))
        
        # Converter para tensor
        image_tensor = self.to_tensor(image)
        mask_tensor = self.to_tensor(mask)
        
        # Combinar boxes e labels num único tensor
        box_data = torch.cat([all_boxes, box_labels.unsqueeze(1)], dim=1)
        
        return image_tensor, mask_tensor, box_data

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
        """
        Extrai coordenadas para regiões de background expandindo as boxes da lesão.
        
        Args:
            image: Imagem PIL
            mask: Máscara PIL
            expand_ratio: Fator de expansão para região ao redor da lesão
        
        Returns:
            Lista de coordenadas YOLO para regiões de background
        """
        img_w, img_h = image.size
        
        # Primeiro obtém as boxes das lesões
        lesion_boxes = self._extract_instance_boxes(image, mask)
        bg_boxes = []
        
        # Para cada lesão, cria uma região expandida ao redor
        for box in lesion_boxes:
            _, cx, cy, w, h = box
            
            # Converte para coordenadas absolutas
            abs_cx = cx * img_w
            abs_cy = cy * img_h
            abs_w = w * img_w
            abs_h = h * img_h
            
            # Calcula região expandida (mais contexto ao redor)
            bg_w = abs_w * expand_ratio
            bg_h = abs_h * expand_ratio
            
            # Garante que está dentro dos limites da imagem
            bg_x1 = max(0, int(abs_cx - bg_w / 2))
            bg_y1 = max(0, int(abs_cy - bg_h / 2))
            bg_x2 = min(img_w, int(abs_cx + bg_w / 2))
            bg_y2 = min(img_h, int(abs_cy + bg_h / 2))
            
            # Converte de volta para formato YOLO
            bg_cx = (bg_x1 + bg_x2) / 2 / img_w
            bg_cy = (bg_y1 + bg_y2) / 2 / img_h
            bg_w = (bg_x2 - bg_x1) / img_w
            bg_h = (bg_y2 - bg_y1) / img_h
            
            bg_boxes.append([0, bg_cx, bg_cy, bg_w, bg_h])  # class_idx=0 para background
        
        return bg_boxes

    def _data_load(self, class_name):
        """
        Carrega caminhos de imagens e máscaras.
        
        Args:
            class_name: Nome da pasta de classe
            
        Returns:
            Tupla de (image_paths, mask_paths, labels)
        """
        lesion_filenames = os.listdir(os.path.join(self.root_path, class_name, 'images'))
        lesion_image_paths = [
            os.path.join(self.root_path, class_name, 'images', filename)
            for filename in lesion_filenames
        ]
        lesion_masks_paths = [
            os.path.join(self.root_path, class_name, 'masks', filename)
            for filename in lesion_filenames
        ]
        lesion_labels = [1] * len(lesion_image_paths)
        return lesion_image_paths, lesion_masks_paths, lesion_labels
    
    def _load_image_mask_boxes(self, idx):
        """
        Carrega apenas imagem, máscara e boxes para uso com mosaic.
        
        Args:
            idx: Índice da imagem
            
        Returns:
            Tupla (imagem, máscara, boxes)
        """
        image_path = self.lesion_image_paths[idx]
        mask_path = self.lesion_masks_paths[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = self.resize_transform(image)
        mask = self.resize_transform(mask)
        
        # Extrai apenas as coordenadas das boxes
        boxes = self._extract_instance_boxes(image, mask)
        return image, mask, boxes

def mosaic_augmentation(dataset, indices, image_size=(640, 640)):
    """
    Cria uma imagem mosaic a partir de 4 imagens do dataset.
    
    Args:
        dataset: Dataset para carregar imagens
        indices: Lista de 4 índices de imagem
        image_size: Tamanho da imagem resultante
        
    Returns:
        Tupla (imagem mosaic, máscara mosaic, boxes ajustadas)
    """
    assert len(indices) == 4, "Mosaic precisa de 4 imagens"
    imgs, masks, boxes = [], [], []

    for idx in indices:
        img, mask, img_boxes = dataset._load_image_mask_boxes(idx)
        imgs.append(np.array(img))
        masks.append(np.array(mask))
        boxes.append(img_boxes)

    h, w = image_size
    xc = random.randint(int(0.4 * w), int(0.6 * w))
    yc = random.randint(int(0.4 * h), int(0.6 * h))

    mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
    mosaic_mask = np.zeros((h, w), dtype=np.uint8)
    mosaic_boxes = []

    # Posições: [top-left, top-right, bottom-left, bottom-right]
    positions = [
        (0, 0, xc, yc),           # top-left
        (xc, 0, w, yc),           # top-right
        (0, yc, xc, h),           # bottom-left
        (xc, yc, w, h)            # bottom-right
    ]

    for i, (x1, y1, x2, y2) in enumerate(positions):
        img = imgs[i]
        mask = masks[i]
        orig_h, orig_w = img.shape[:2]

        # Resize para caber no quadrante
        scale_x = (x2 - x1) / orig_w
        scale_y = (y2 - y1) / orig_h
        img_resized = np.array(Image.fromarray(img).resize((x2 - x1, y2 - y1)))
        mask_resized = np.array(Image.fromarray(mask).resize((x2 - x1, y2 - y1)))

        # Coloca no mosaic
        mosaic_img[y1:y2, x1:x2] = img_resized
        mosaic_mask[y1:y2, x1:x2] = mask_resized

        # Ajusta boxes
        for box in boxes[i]:
            cls, x_c, y_c, bw, bh = box
            # Para coordenadas absolutas
            abs_x = x_c * orig_w
            abs_y = y_c * orig_h
            abs_w = bw * orig_w
            abs_h = bh * orig_h

            # Ajusta para o quadrante
            new_x = (abs_x * scale_x) + x1
            new_y = (abs_y * scale_y) + y1
            new_w = abs_w * scale_x
            new_h = abs_h * scale_y

            # Normaliza para o tamanho final
            norm_x = new_x / w
            norm_y = new_y / h
            norm_w = new_w / w
            norm_h = new_h / h

            # Só adiciona se a box estiver dentro da imagem
            if 0 <= norm_x <= 1 and 0 <= norm_y <= 1:
                mosaic_boxes.append([cls, norm_x, norm_y, norm_w, norm_h])

    mosaic_img = Image.fromarray(mosaic_img)
    mosaic_mask = Image.fromarray(mosaic_mask)
    return mosaic_img, mosaic_mask, mosaic_boxes

def custom_collate(batch):
    """
    Função de collate customizada que separa boxes de lesões e background.
    
    Args:
        batch: Lista de tuplas (imagem, máscara, box_data)
        
    Returns:
        Tupla de tensores (images, masks, all_boxes, all_labels, all_img_indices)
    """
    images, masks, box_data = zip(*batch)
    
    # Empilha imagens e máscaras
    images = torch.stack(images)
    masks = torch.stack(masks)
    
    # Coleta todas as boxes e seus índices de imagem
    all_boxes = []
    all_labels = []
    all_img_indices = []
    
    for idx, bd in enumerate(box_data):
        num_boxes = bd.size(0)
        
        # Separar boxes e labels
        boxes = bd[:, :5]  # [class_idx, x, y, w, h]
        labels = bd[:, 5]  # Label (0=background, 1=lesão)
        
        all_boxes.append(boxes)
        all_labels.append(labels)
        all_img_indices.extend([idx] * num_boxes)
    
    # Concatena tudo
    all_boxes = torch.cat(all_boxes, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_img_indices = torch.tensor(all_img_indices)
    
    return images, masks, all_boxes, all_labels, all_img_indices
