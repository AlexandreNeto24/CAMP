import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import datetime
from tqdm.auto import tqdm
import shutil

# Configuração de caminhos
project_root = Path(__file__).parent.parent  # notebooks → project_root
sys.path.insert(0, str(project_root))

# Adicione o caminho para o ultralytics clonado
ultralytics_path = os.path.abspath(os.path.join(project_root, "src", "ultralytics"))
sys.path.insert(0, ultralytics_path)

# Importações
from src.paste.dataset import CustomCropDataset
try:
    from ultralytics import YOLO
    print("Ultralytics importado com sucesso!")
except ImportError as e:
    print(f"Erro ao importar ultralytics: {e}")
    sys.exit(1)

# Configuração
data_path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'metaplasia'))
save_weights_path = os.path.abspath(os.path.join(os.getcwd(), 'models', 'baseline', 'weights'))
hyp_path = os.path.abspath(os.path.join(os.getcwd(), 'cfg', 'default.yaml'))
os.makedirs(save_weights_path, exist_ok=True)

# Carrega hiperparâmetros
with open(hyp_path, 'r') as f:
    hyp_dict = yaml.safe_load(f)

def prepare_yolo_dataset(dataset, output_dir):
    """
    Prepara um dataset no formato YOLO a partir do nosso CustomCropDataset
    """
    # Cria diretórios
    images_train_dir = os.path.join(output_dir, "images", "train")
    images_val_dir = os.path.join(output_dir, "images", "val")
    labels_train_dir = os.path.join(output_dir, "labels", "train")
    labels_val_dir = os.path.join(output_dir, "labels", "val")
    
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    
    # Split dataset
    val_split = 0.2
    n_train = int(len(dataset) * (1 - val_split))
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    print(f"Preparando dataset - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Processa treino
    for i, idx in enumerate(tqdm(range(len(train_dataset)), desc="Processando treino")):
        img, _, boxes_data = train_dataset[idx]
        
        # Salva imagem
        img_path = os.path.join(images_train_dir, f"{i:06d}.jpg")
        img_np = img.permute(1, 2, 0).numpy() * 255
        plt.imsave(img_path, img_np.astype(np.uint8))
        
        # Filtra apenas boxes de lesão (label=1)
        if boxes_data.size(1) > 5:  # Se temos coluna de labels
            boxes = boxes_data[:, :5]  # [class_idx, x, y, w, h]
            labels = boxes_data[:, 5]  # Label (0=background, 1=lesão)
            mask = labels == 1  # Filtrar apenas lesões
            boxes = boxes[mask]
        else:
            boxes = boxes_data  # Assume todas como lesão
        
        # Salva labels no formato YOLO
        label_path = os.path.join(labels_train_dir, f"{i:06d}.txt")
        with open(label_path, "w") as f:
            for box in boxes:
                # Formato YOLO: class_id, x_center, y_center, width, height
                f.write(f"0 {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
    
    # Processa validação
    for i, idx in enumerate(tqdm(range(len(val_dataset)), desc="Processando validação")):
        img, _, boxes_data = val_dataset[idx]
        
        # Salva imagem
        img_path = os.path.join(images_val_dir, f"{i:06d}.jpg")
        img_np = img.permute(1, 2, 0).numpy() * 255
        plt.imsave(img_path, img_np.astype(np.uint8))
        
        # Filtra apenas boxes de lesão (label=1)
        if boxes_data.size(1) > 5:  # Se temos coluna de labels
            boxes = boxes_data[:, :5]  # [class_idx, x, y, w, h]
            labels = boxes_data[:, 5]  # Label (0=background, 1=lesão)
            mask = labels == 1  # Filtrar apenas lesões
            boxes = boxes[mask]
        else:
            boxes = boxes_data  # Assume todas como lesão
        
        # Salva labels no formato YOLO
        label_path = os.path.join(labels_val_dir, f"{i:06d}.txt")
        with open(label_path, "w") as f:
            for box in boxes:
                # Formato YOLO: class_id, x_center, y_center, width, height
                f.write(f"0 {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                
    # Cria arquivo YAML de configuração
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # Número de classes
        'names': ['lesion']  # Nomes das classes
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
        
    return yaml_path

def plot_results(results_file, output_dir):
    """
    Plota os resultados do treinamento a partir do arquivo CSV gerado pelo YOLO
    """
    import pandas as pd
    
    # Carrega resultados
    results = pd.read_csv(results_file)
    
    # Imprime colunas disponíveis para debug
    print("Colunas disponíveis no results.csv:")
    print(results.columns.tolist())
    
    # Plota métricas
    plt.figure(figsize=(20, 16))
    
    # Loss - estas colunas geralmente estão presentes
    plt.subplot(3, 2, 1)
    plt.plot(results['epoch'], results['train/box_loss'], label='train')
    plt.plot(results['epoch'], results['val/box_loss'], label='val')
    plt.title('Box Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(3, 2, 2)
    plt.plot(results['epoch'], results['train/cls_loss'], label='train')
    plt.plot(results['epoch'], results['val/cls_loss'], label='val')
    plt.title('Class Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(3, 2, 3)
    plt.plot(results['epoch'], results['train/dfl_loss'], label='train')
    plt.plot(results['epoch'], results['val/dfl_loss'], label='val')
    plt.title('DFL Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Métricas - busca por nomes alternativos para lidar com diferentes versões do Ultralytics
    plt.subplot(3, 2, 4)
    
    # Tenta diferentes possíveis nomes de colunas para precision/recall
    precision_cols = ['metrics/precision', 'precision', 'val/precision']
    precision_col = next((col for col in precision_cols if col in results.columns), None)
    
    recall_cols = ['metrics/recall', 'recall', 'val/recall']
    recall_col = next((col for col in recall_cols if col in results.columns), None)
    
    if precision_col:
        plt.plot(results['epoch'], results[precision_col], label='precision')
    if recall_col:
        plt.plot(results['epoch'], results[recall_col], label='recall')
    
    plt.title('Precision/Recall')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(3, 2, 5)
    
    # Tenta diferentes possíveis nomes de colunas para mAP
    map50_cols = ['metrics/mAP50', 'mAP_0.5', 'val/mAP50', 'val/mAP_0.5']
    map50_col = next((col for col in map50_cols if col in results.columns), None)
    
    map5095_cols = ['metrics/mAP50-95', 'mAP_0.5:0.95', 'val/mAP50-95', 'val/mAP_0.5:0.95']
    map5095_col = next((col for col in map5095_cols if col in results.columns), None)
    
    if map50_col:
        plt.plot(results['epoch'], results[map50_col], label='mAP50')
    if map5095_col:
        plt.plot(results['epoch'], results[map5095_col], label='mAP50-95')
    
    plt.title('mAP')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close()

def main():
    # Cria diretório com timestamp
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_weights_path, now)
    os.makedirs(run_dir, exist_ok=True)
    
    # Carrega dataset
    print("Carregando dataset...")
    dataset = CustomCropDataset(root_path=data_path, hyp=hyp_dict)
    
    # Prepara dataset no formato YOLO
    print("Preparando dataset no formato YOLO...")
    dataset_dir = os.path.join(run_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    data_yaml_path = prepare_yolo_dataset(dataset, dataset_dir)
    
    print(f"Dataset preparado: {dataset_dir}")
    print(f"Configuração YAML: {data_yaml_path}")
    
    # Treina o modelo usando a API nativa do YOLO
    print("Iniciando treinamento...")
    model = YOLO("yolov8n.pt")  # Já está usando o menor, yolov8n
    
    # Ajuste os parâmetros para treinamento estendido
    results = model.train(
        data=data_yaml_path,
        epochs=500,      # Aumentado de 50 para 500 épocas
        patience=15,     # Aumentado de 10 para 15 para dar mais chance de melhoria
        batch=8,         # Mantido em 8 para evitar problemas de memória
        imgsz=640,
        save=True,
        device="0",      # GPU
        project=run_dir,
        name="baseline",
        exist_ok=True,
        cache=False,     # Desabilita cache para economizar memória
        amp=True,        # Mixed precision para reduzir uso de memória
        close_mosaic=10, # Desativa mosaico nas últimas 10 épocas
    )
    
    # Resultados
    print("Treinamento concluído!")
    
    # Plota resultados
    results_csv = os.path.join(run_dir, "baseline", "results.csv")
    plot_results(results_csv, os.path.join(run_dir, "baseline"))
    
    # Validação final
    best_model_path = os.path.join(run_dir, "baseline", "weights", "best.pt")
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        val_results = best_model.val(data=data_yaml_path)
        
        # Copia o melhor modelo para a raiz do diretório
        shutil.copy(best_model_path, os.path.join(run_dir, "best.pt"))
        
    print(f"Resultados e modelo salvo em: {run_dir}")

if __name__ == "__main__":
    main()