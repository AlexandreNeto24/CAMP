from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import torch
from PIL import Image
import random
import albumentations as A
import yaml


class CustomCropDataset(Dataset):
    def __init__(self, root_path, image_size=(640, 640), transform=None, pad=10, zoom_factor=0.5,
                 margin_factor_zoom=0.3, hyp=None 
                 ):
        """
        :param margin_factor_zoom: quanto queremos expandir a ROI do zoomed para manter mais background
        """
        self.root_path = root_path
        self.image_size = image_size
        self.pad = pad
        self.zoom_factor = zoom_factor
        self.margin_factor_zoom = margin_factor_zoom  # margem para zoomed
        self.hyp = hyp

        self.to_tensor = transforms.ToTensor()
        self.resize_transform = transforms.Resize(self.image_size)
        
        # # Transformação para a imagem global (mantém 640x640)
        # self.transform = transform if transform else transforms.Compose([
        #     transforms.Resize(self.image_size),
        #     transforms.ToTensor()
        # ])

        if self.hyp is not None:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(p=0.5),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.HorizontalFlip(p=self.hyp.get('fliplr', 0.5)),
                A.VerticalFlip(p=self.hyp.get('flipud', 0.0)),
                A.RandomBrightnessContrast(p=self.hyp.get('hsv_v', 0.4)),
                A.HueSaturationValue(
                    hue_shift_limit=int(self.hyp.get('hsv_h', 0.015)*180),
                    sat_shift_limit=int(self.hyp.get('hsv_s', 0.7)*255),
                    val_shift_limit=int(self.hyp.get('hsv_v', 0.4)*255),
                    p=0.5
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.05,  # mais suave
                    scale_limit=0.2,   # mais suave
                    rotate_limit=5,    # mais suave
                    p=0.7
                ),
                A.Perspective(scale=(0.0, 0.0005), p=0.5),  # mais suave
                # Adiciona outras se quiseres
            ])
        else:
            self.transform = None
        
        # Carregar caminhos de imagens e máscaras
        self.lesion_image_paths, self.lesion_masks_paths, self.lesion_labels = self._data_load('metaplasia')
        
    def __len__(self):
        return len(self.lesion_image_paths)
    
    def __getitem__(self, idx):
        # Mosaic augmentation
        mosaic_prob = self.hyp.get('mosaic', 1.0) if self.hyp else 1.0
        use_mosaic = random.random() < mosaic_prob

        if hasattr(self, 'current_epoch') and hasattr(self, 'total_epochs'):
            close_mosaic = self.hyp.get('close_mosaic', 10)
            if self.current_epoch >= self.total_epochs - close_mosaic:
                use_mosaic = False

        if use_mosaic:
            # Seleciona 3 imagens aleatórias diferentes
            indices = [idx] + random.sample([i for i in range(len(self)) if i != idx], 3)
            image, mask, yolo_boxes = mosaic_augmentation(self, indices, image_size=self.image_size)
        else:
            image_path = self.lesion_image_paths[idx]
            mask_path = self.lesion_masks_paths[idx]
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            image = self.resize_transform(image)
            mask = self.resize_transform(mask)
            _, _, _, _, yolo_boxes = self._extract_instances(image, mask)

        # 2. Converter para numpy para augmentations
        image_np = np.array(image)
        mask_np = np.array(mask)

        # 3. Augmentations (se existirem)
        if self.transform:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_np = augmented['image']
            mask_np = augmented['mask']

        # 4. Converter de volta para PIL
        image = Image.fromarray(image_np)
        mask = Image.fromarray(mask_np)

        # 5. Extrair instâncias, crops, boxes, etc. (já tudo no espaço 640x640)
        cropped_images, cropped_masks, zoomed_images, zoomed_masks, yolo_boxes = self._extract_instances(image, mask)
        
        # Agora, para cada patch (cropped e zoomed), vamos aplicar letterbox se quisermos padronizar para 640x640
        padded_cropped_imgs = []
        cropped_coords = []
        for img in cropped_images:
            padded_img, coords = letterbox_image(img, target_size=self.image_size)
            padded_cropped_imgs.append(self.to_tensor(padded_img))
            cropped_coords.append(coords)
        
        padded_zoomed_imgs = []
        zoomed_coords = []
        for img in zoomed_images:
            padded_img, coords = letterbox_image(img, target_size=self.image_size)
            padded_zoomed_imgs.append(self.to_tensor(padded_img))
            zoomed_coords.append(coords)
        
        # Converter coords para tensores
        cropped_coords = torch.tensor(cropped_coords, dtype=torch.float32)
        zoomed_coords = torch.tensor(zoomed_coords, dtype=torch.float32)
        
        # Adicionar coluna de classe 0
        cropped_coords = torch.cat([torch.zeros(cropped_coords.size(0), 1), cropped_coords], dim=1)
        zoomed_coords = torch.cat([torch.zeros(zoomed_coords.size(0), 1), zoomed_coords], dim=1)
        
        # Converter yolo_boxes para tensor
        yolo_boxes = torch.tensor(yolo_boxes, dtype=torch.float32)
        
        # # Por fim, criar imagem global (redimensionada para 640x640)
        # global_img = self.transform(image)
        # global_mask = self.transform(mask)
        # TODO: fazemos resize depois de retirarmos as yoloboxes. Isto pode comprometer a verdadeira localização das lesões
        global_img = self.resize_transform(image)
        global_mask = self.resize_transform(mask)
        global_img = self.to_tensor(global_img)
        global_mask = self.to_tensor(global_mask)
        
        return global_img, global_mask, padded_cropped_imgs, cropped_coords, padded_zoomed_imgs, zoomed_coords, yolo_boxes

    def _extract_instances(self, image, mask):
        """
        Extrai instâncias usando connectedComponentsWithStats, apaga o fundo e:
          - Para cropped, faz tight crop
          - Para zoomed, faz crop com margin (para reter mais background)
        Retorna as imagens recortadas (cropped_images, zoomed_images) e suas máscaras.
        """
        mask_np = np.array(mask)
        if len(mask_np.shape) == 3:
            mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
        mask_np = (mask_np > 0).astype(np.uint8) * 255
        assert len(mask_np.shape) == 2, f"Mask must be single-channel, got shape {mask_np.shape}"
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        
        cropped_images, cropped_masks = [], []
        zoomed_images, zoomed_masks = [], []
        yolo_boxes = []
        
        img_w, img_h = image.size
        
        for i in range(1, num_labels):
            x, y, w, h, _ = stats[i]
            
            # Máscara da instância
            instance_mask = (labels == i).astype(np.uint8) * 255
            instance_mask_pil = Image.fromarray(instance_mask)
            
            # Crop básico com padding
            x_min = max(0, x - self.pad)
            y_min = max(0, y - self.pad)
            x_max = min(mask_np.shape[1], x + w + self.pad)
            y_max = min(mask_np.shape[0], y + h + self.pad)
            
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            cropped_mask = instance_mask_pil.crop((x_min, y_min, x_max, y_max))
            
            # Apagar o fundo
            cropped_image_np = np.array(cropped_image)
            cropped_mask_np = np.array(cropped_mask)
            cropped_image_np[cropped_mask_np == 0] = 0
            cropped_image = Image.fromarray(cropped_image_np)
            
            # Agora, tight crop no objeto
            tight_cropped_img, tight_cropped_mask = self._tight_crop(cropped_image, cropped_mask)
            
            # Zoomed
            zoom_x_min = max(0, int(x_min - w * self.zoom_factor))
            zoom_y_min = max(0, int(y_min - h * self.zoom_factor))
            zoom_x_max = min(mask_np.shape[1], int(x_max + w * self.zoom_factor))
            zoom_y_max = min(mask_np.shape[0], int(y_max + h * self.zoom_factor))
            
            zoomed_image = image.crop((zoom_x_min, zoom_y_min, zoom_x_max, zoom_y_max))
            zoomed_mask = instance_mask_pil.crop((zoom_x_min, zoom_y_min, zoom_x_max, zoom_y_max))
            
            zoomed_image_np = np.array(zoomed_image)
            zoomed_mask_np = np.array(zoomed_mask)
            # Apaga o objeto (mantendo fundo) ou o contrário, dependendo da lógica
            if np.any(zoomed_mask_np > 0):
                zoomed_image_np[zoomed_mask_np > 0] = cv2.GaussianBlur(
                    zoomed_image_np, (5,5), 0
                )[zoomed_mask_np > 0]
            zoomed_image = Image.fromarray(zoomed_image_np)
            
            # Crop com margem para reter mais background
            expanded_zoomed_img, expanded_zoomed_mask = self._crop_with_margin(
                zoomed_image, zoomed_mask, margin_factor=self.margin_factor_zoom
            )
            
            # Salva
            cropped_images.append(tight_cropped_img)
            cropped_masks.append(tight_cropped_mask)
            zoomed_images.append(expanded_zoomed_img)
            zoomed_masks.append(expanded_zoomed_mask)
            
            # YOLO boxes normalizadas (no espaço original)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            bbox_width = w / img_w
            bbox_height = h / img_h
            yolo_boxes.append([0, x_center, y_center, bbox_width, bbox_height])
        
        return cropped_images, cropped_masks, zoomed_images, zoomed_masks, yolo_boxes

    def _tight_crop(self, pil_img, pil_mask):
        """
        Recorta justinho ao objeto.
        """
        img_np = np.array(pil_img)
        mask_np = np.array(pil_mask)
        
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            img_np = img_np[y:y+h, x:x+w]
            mask_np = mask_np[y:y+h, x:x+w]
            
            tight_img = Image.fromarray(img_np)
            tight_mask = Image.fromarray(mask_np)
        else:
            tight_img = pil_img
            tight_mask = pil_mask
        
        return tight_img, tight_mask
    
    def _crop_with_margin(self, pil_img, pil_mask, margin_factor=0.3):
        """
        Parecido com o tight_crop, mas expande a ROI para manter mais background.
        """
        img_np = np.array(pil_img)
        mask_np = np.array(pil_mask)
        
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            margin_x = int(w * margin_factor)
            margin_y = int(h * margin_factor)
            
            x_expanded = max(0, x - margin_x)
            y_expanded = max(0, y - margin_y)
            w_expanded = w + 2 * margin_x
            h_expanded = h + 2 * margin_y
            
            H, W = mask_np.shape[:2]
            if x_expanded + w_expanded > W:
                w_expanded = W - x_expanded
            if y_expanded + h_expanded > H:
                h_expanded = H - y_expanded
            
            img_np = img_np[y_expanded : y_expanded + h_expanded,
                            x_expanded : x_expanded + w_expanded]
            mask_np = mask_np[y_expanded : y_expanded + h_expanded,
                              x_expanded : x_expanded + w_expanded]
            
            out_img = Image.fromarray(img_np)
            out_mask = Image.fromarray(mask_np)
        else:
            out_img = pil_img
            out_mask = pil_mask
        
        return out_img, out_mask

    def _data_load(self, class_name):
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
        image_path = self.lesion_image_paths[idx]
        mask_path = self.lesion_masks_paths[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = self.resize_transform(image)
        mask = self.resize_transform(mask)
        # Extrai boxes (sem crops, só boxes)
        _, _, _, _, yolo_boxes = self._extract_instances(image, mask)
        return image, mask, None, None, None, None, yolo_boxes

def letterbox_image(img, target_size=(640, 640)):
    """
    Redimensiona a imagem mantendo a proporção (letterbox) e aplica padding
    para que a imagem final tenha exatamente o tamanho target_size.
    Retorna (padded_img, (xmin, ymin, xmax, ymax)) com as coords da
    região redimensionada dentro da imagem final.
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    h, w = img.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    padded_img = cv2.copyMakeBorder(resized_img, pad_top, pad_bottom,
                                    pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=(0,0,0))
    
    # Coordenadas absolutas dentro da imagem final
    xmin = pad_left
    ymin = pad_top
    xmax = pad_left + new_w
    ymax = pad_top + new_h

    return Image.fromarray(padded_img), (xmin, ymin, xmax, ymax)

# Define a transform to convert PIL images to tensors
to_tensor = transforms.ToTensor()

def custom_collate(batch):
    images, masks, cropped_imgs, cropped_coords, zoomed_imgs, zoomed_coords, bbs = zip(*batch)
    
    images = torch.stack([img for img in images])
    masks = torch.stack([mask for mask in masks])

    all_cropped_imgs, all_cropped_coords = [], []
    all_zoomed_imgs, all_zoomed_coords = [], []
    all_bbs = []
    all_img_indices = []

    for idx in range(len(cropped_imgs)):
        n_inst = len(cropped_imgs[idx])
        for i in range(n_inst):
            all_cropped_imgs.append(cropped_imgs[idx][i])
            all_cropped_coords.append(cropped_coords[idx][i])
            all_zoomed_imgs.append(zoomed_imgs[idx][i])
            all_zoomed_coords.append(zoomed_coords[idx][i])
            all_bbs.append(bbs[idx][i])
            all_img_indices.append(idx)  # índice da imagem original no batch

    all_cropped_imgs = torch.stack(all_cropped_imgs)
    all_cropped_coords = torch.stack(all_cropped_coords)
    all_zoomed_imgs = torch.stack(all_zoomed_imgs)
    all_zoomed_coords = torch.stack(all_zoomed_coords)
    all_bbs = torch.stack(all_bbs)
    all_img_indices = torch.tensor(all_img_indices)

    return images, masks, all_cropped_imgs, all_cropped_coords, all_zoomed_imgs, all_zoomed_coords, all_bbs, all_img_indices

def mosaic_augmentation(dataset, indices, image_size=(640, 640)):
    """
    Cria uma imagem mosaic a partir de 4 imagens do dataset.
    Retorna:
        - imagem mosaic (PIL)
        - máscara mosaic (PIL)
        - lista de boxes ajustadas (cada box: [class, x_center, y_center, w, h] normalizado)
    """
    assert len(indices) == 4, "Mosaic precisa de 4 imagens"
    imgs, masks, boxes = [], [], []

    for idx in indices:
        img, mask, _, _, _, _, yolo_boxes = dataset._load_image_mask_boxes(idx)
        imgs.append(np.array(img))
        masks.append(np.array(mask))
        boxes.append(yolo_boxes)

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
