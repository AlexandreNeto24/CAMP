
import os
import sys

# # Caminho até a pasta src
# project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if project_path not in sys.path:
#     sys.path.insert(0, project_path)

# Add the path to the ultralytics directory, which is one level up and then into the 'ultralytics' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ultralytics")))

# Agora o import deve ser direto
from ultralytics import YOLO
import yaml
import torch.nn as nn

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def build_model(hyp_path, device):
    model = YOLO('yolov8s.pt').to(device)
    # Access the detection head from the underlying sequential container.
    detect_layer = model.model.model[-1]  # This should be the Detect module

    # Set number of classes to 1 and recompute expected output channels.
    detect_layer.nc = 1
    # For a DFL head with reg_max (typically 16), expected output channels is:
    detect_layer.no = detect_layer.reg_max * 4 + detect_layer.nc  # e.g. 16*4 + 1 = 65
    print("Expected output channels should be:", detect_layer.no)

    # IMPORTANT: Do not change the regression branch (cv2) if you want to use DFL.
    # Update only the classification branch (cv3) so that it outputs 1 channel instead of 80.
    if hasattr(detect_layer, 'cv3'):
        for i in range(len(detect_layer.cv3)):
            seq = detect_layer.cv3[i]
            layers = list(seq.children())
            in_channels = layers[-1].in_channels
            # Reinitialize the last layer to output detect_layer.nc (which is 1) channel.
            layers[-1] = nn.Conv2d(in_channels, detect_layer.nc, kernel_size=1, stride=1).to(device)
            detect_layer.cv3[i] = nn.Sequential(*layers).to(device)
    else:
        print("cv3 branch not found.")

    # (Optional) If there's a flag for DFL, ensure it is enabled.
    if hasattr(detect_layer, 'use_dfl'):
        detect_layer.use_dfl = True

    # Update model class names.
    model.model.names = {0: 'Similar Region'}

    # Load hyperparameters
    with open(hyp_path, "r") as f:
        hyp = yaml.safe_load(f)

    model.model.criterion = model.model.init_criterion()  # Ensure criterion is initialized
    model.model.criterion.hyp = DictToObject(hyp)  # Update hyperparameters
    
    return model