import os
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# Configura caminhos
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.paste.dataset import CustomCropDataset, custom_collate
from src.paste.paste_recommender import PasteRecommender
import yaml
from torch.utils.data import DataLoader, random_split

# Carrega hiperparâmetros
hyp_path = os.path.abspath(os.path.join(project_root, 'cfg', 'default.yaml'))
with open(hyp_path, 'r') as f:
    hyp_dict = yaml.safe_load(f)

# Carrega dataset
data_path = os.path.abspath(os.path.join(project_root, 'data', 'metaplasia'))
dataset = CustomCropDataset(root_path=data_path, hyp=hyp_dict)

# Split dataset
val_split = 0.2
n_train = int(len(dataset) * (1 - val_split))
n_val = len(dataset) - n_train
_, val_dataset = random_split(dataset, [n_train, n_val])

# Carrega modelo treinado
# Substitua pelo caminho do seu modelo
model_path = os.path.abspath(os.path.join(project_root, 'models', 'paste', 'weights', '20250718_093137', 'detector', 'weights', 'best.pt'))
recommender_weights = os.path.abspath(os.path.join(project_root, 'models', 'paste', 'weights', '20250718_093137', 'best_recommender.pt'))

# Inicializa recomendador
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
recommender = PasteRecommender(detector_path=model_path, embedding_dim=128).to(device)
recommender.load_state_dict(torch.load(recommender_weights, map_location=device))

# Avalia
from notebooks.train_paste_two_stage import evaluate_paste_recommendations
evaluate_paste_recommendations(recommender, val_dataset, num_samples=10, save_dir='evaluation_results')