import os
import sys
import site

# Tenta importar cv2
try:
    import cv2
except ImportError:
    print("Tentando corrigir erro de importação do OpenCV...")
    # Adiciona caminhos do site-packages do ambiente conda
    conda_prefix = os.environ.get('CONDA_PREFIX', r'C:\Users\win11\anaconda3\envs\cim_yolo')
    site_packages = os.path.join(conda_prefix, 'Lib', 'site-packages')
    if site_packages not in sys.path and os.path.exists(site_packages):
        print(f"Adicionando ao path: {site_packages}")
        sys.path.insert(0, site_packages)
    
    # Tenta importar novamente
    import cv2

# Continua com os imports originais
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime
import yaml

# Configura caminhos
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# CustomCropDataset foi removido. Use AttentionDataset de src.paste.attention_dataset em seu lugar.

# GERA UM TIMESTAMP ÚNICO PARA TODO O EXPERIMENTO
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# DEFINE O DIRETÓRIO BASE PARA TODOS OS RESULTADOS
RUN_DIR = os.path.abspath(os.path.join(project_root, 'models', 'paste', 'weights', RUN_TIMESTAMP))
os.makedirs(RUN_DIR, exist_ok=True)

print(f"Todos os resultados serão salvos em: {RUN_DIR}")

# Cria subdiretórios para debug
DEBUG_DIR = os.path.join(RUN_DIR, 'debug')
DEBUG_AUG_DIR = os.path.join(RUN_DIR, 'debug_augmented')
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(DEBUG_AUG_DIR, exist_ok=True)

# Salva o diretório atual para ser usado pelo script de treinamento
with open(os.path.join(RUN_DIR, 'run_info.txt'), 'w') as f:
    f.write(RUN_DIR)

# Carrega hiperparâmetros
hyp_path = os.path.abspath(os.path.join(project_root, 'cfg', 'default.yaml'))
with open(hyp_path, 'r') as f:
    hyp_dict = yaml.safe_load(f)

# Salva uma cópia dos hiperparâmetros nesta pasta de experimento
with open(os.path.join(RUN_DIR, 'hyperparams.yaml'), 'w') as f:
    yaml.dump(hyp_dict, f)

# Caminho para dados
data_path = os.path.abspath(os.path.join(project_root, 'data', 'metaplasia'))

# Carrega datasets separados (com e sem augmentation)
print("Carregando e verificando datasets...")

# Dataset sem augmentation para estatísticas e visualização
clean_dataset = CustomCropDataset(root_path=data_path, hyp=None, apply_aug=False)

# Dataset com augmentation apenas para referência/comparação
aug_dataset = CustomCropDataset(root_path=data_path, hyp=hyp_dict, apply_aug=True)

print(f"Dataset carregado com {len(clean_dataset)} imagens")

# Conta número de lesões no dataset limpo
total_lesions = 0
total_backgrounds = 0

for i in range(min(100, len(clean_dataset))):  # Verifica as 100 primeiras imagens
    _, _, boxes = clean_dataset[i]
    if boxes.size(1) > 5:
        total_lesions += (boxes[:, 5] == 1).sum().item()
        total_backgrounds += (boxes[:, 5] == 0).sum().item()

print(f"Estatísticas (primeiras 100 imagens):")
print(f"- Total de lesões: {total_lesions}")
print(f"- Total de backgrounds: {total_backgrounds}")
print(f"- Proporção lesão:background = 1:{total_backgrounds/max(1,total_lesions):.1f}")

# Salva as estatísticas em um arquivo de texto
with open(os.path.join(RUN_DIR, 'dataset_stats.txt'), 'w') as f:
    f.write(f"Dataset carregado com {len(clean_dataset)} imagens\n")
    f.write(f"Estatísticas (primeiras 100 imagens):\n")
    f.write(f"- Total de lesões: {total_lesions}\n")
    f.write(f"- Total de backgrounds: {total_backgrounds}\n")
    f.write(f"- Proporção lesão:background = 1:{total_backgrounds/max(1,total_lesions):.1f}\n")

print("Dataset pronto para treinamento!")

def save_debug_examples(dataset, num_samples=5, save_dir=None):
    """
    Salva exemplos de imagens com suas instâncias e backgrounds para debug.
    
    Args:
        dataset: Dataset que contém as imagens e boxes
        num_samples: Número de amostras a salvar
        save_dir: Diretório onde salvar os exemplos
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Salvando exemplos de debug em: {save_dir}")
    
    # Seleciona amostras aleatórias
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for i, idx in enumerate(indices):
        img, mask, boxes = dataset[idx]
        
        if boxes.size(0) == 0:
            continue
            
        # Converte tensor para numpy para visualização
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)  # Limita valores entre 0 e 1
        
        mask_np = mask.permute(1, 2, 0).numpy() if mask.dim() > 2 else mask.numpy()
        
        # Cria figura com 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Imagem original
        axs[0].imshow(img_np)
        axs[0].set_title("Imagem Original")
        axs[0].axis('off')
        
        # 2. Máscara
        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title("Máscara")
        axs[1].axis('off')
        
        # 3. Imagem com boxes (instâncias e backgrounds)
        axs[2].imshow(img_np)
        axs[2].set_title("Boxes (Lesão = Verde, Background = Vermelho)")
        axs[2].axis('off')
        
        # Plota as boxes
        img_h, img_w = img_np.shape[:2]
        for box in boxes:
            # Formato [class_idx, x_center, y_center, width, height, label]
            if box.size(0) > 5:
                _, cx, cy, w, h, label = box
            else:
                _, cx, cy, w, h = box
                label = 1  # Assume como lesão se não tiver label
                
            # Converte para coordenadas absolutas
            x1 = int((cx - w/2) * img_w)
            y1 = int((cy - h/2) * img_h)
            width = int(w * img_w)
            height = int(h * img_h)
            
            # Cor baseada no label (verde=lesão, vermelho=background)
            color = 'g' if label == 1 else 'r'
            alpha = 0.5 if label == 0 else 0.7  # Background mais transparente
            
            # Adiciona retângulo
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                alpha=alpha
            )
            axs[2].add_patch(rect)
        
        # Salva figura
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"example_{i}.png"), dpi=150)
        plt.close(fig)
    
    print(f"Salvos {num_samples} exemplos para debug")

# Salva exemplos de debug usando os diretórios definidos
save_debug_examples(clean_dataset, num_samples=10, save_dir=DEBUG_DIR)
save_debug_examples(aug_dataset, num_samples=5, save_dir=DEBUG_AUG_DIR)

print(f"\nPreparo de dados concluído! Use este diretório para treinamento:")
print(f"RUN_DIR = '{RUN_DIR}'")

# Comando sugerido para o próximo passo
print(f"\nPara iniciar o treinamento, execute:")
print(f"python notebooks/train_paste_two_stage.py --run-dir \"{RUN_DIR}\"")