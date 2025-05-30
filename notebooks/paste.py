import os
import sys
from torch.utils.data import DataLoader, random_split
import torch
from pathlib import Path

# paste.py está em <project_root>/notebooks/paste.py
project_root = Path(__file__).parent.parent  # notebooks → project_root
sys.path.insert(0, str(project_root))
from src.paste.dataset import CustomCropDataset, custom_collate
from src.paste.paste_model import PasteModel

data_path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'metaplasia'))
save_weights_path = os.path.abspath(os.path.join(os.getcwd(), 'models', 'paste', 'weights'))
hyp_path = os.path.abspath(os.path.join(os.getcwd(), 'cfg', 'default.yaml'))
yaml_path= os.path.abspath(os.path.join(os.getcwd(), 'cfg', 'default_cfg_paste.yaml'))

# load dataset
dataset = CustomCropDataset(root_path=data_path)

# # create the device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# build model
cim_yolo = PasteModel(yaml_path, 'yolov8s')  

cim_yolo.build_model()

# cim_yolo.model_summary()

# # Blocos principais do backbone/head (útil para ver índices para descongelar)
# first_block = list(cim_yolo.model.model.model.children())[0]
# print(first_block)

# for idx, (name, layer) in enumerate(cim_yolo.model.model.model.named_children()):
#     print(f"{idx}: {name} ({layer.__class__.__name__})")
cim_yolo.train(dataset)