import os
import sys
import site

# Tenta importar torch
try:
    import torch
except ImportError:
    print("Tentando corrigir erro de importação do PyTorch...")
    # Adiciona caminhos do site-packages do ambiente conda
    conda_prefix = os.environ.get('CONDA_PREFIX', r'C:\Users\win11\anaconda3\envs\cim_yolo')
    site_packages = os.path.join(conda_prefix, 'Lib', 'site-packages')
    if site_packages not in sys.path and os.path.exists(site_packages):
        print(f"Adicionando ao path: {site_packages}")
        sys.path.insert(0, site_packages)
    
    # Tenta importar novamente
    import torch

# Continua com os imports originais
import os  # Re-import (não causa problema)
import sys  # Re-import
import yaml
import torch
import random  # Adicione esta linha
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import datetime
from tqdm.auto import tqdm
import shutil
import argparse
from sklearn.neighbors import KernelDensity
import json
from scipy.spatial.distance import cdist
from PIL import Image 
import seaborn as sns
import cv2
from src.paste.paste_recommender import RecommendationEvaluator

# Configuração de caminhos
project_root = Path(__file__).parent.parent  # notebooks → project_root
sys.path.insert(0, str(project_root))

# Adicione o caminho para o ultralytics clonado
ultralytics_path = os.path.abspath(os.path.join(project_root, "src", "ultralytics"))
sys.path.insert(0, ultralytics_path)

# Importações
from src.paste.dataset import CustomCropDataset, custom_collate
try:
    from ultralytics import YOLO
    print("Ultralytics importado com sucesso!")
except ImportError as e:
    print(f"Erro ao importar ultralytics: {e}")
    sys.exit(1)

from src.paste.paste_recommender import PasteRecommender
import torch.nn.functional as F

# Configurações
data_path = os.path.abspath(os.path.join(project_root, 'data', 'metaplasia'))
save_weights_path = os.path.abspath(os.path.join(project_root, 'models', 'paste', 'weights'))
hyp_path = os.path.abspath(os.path.join(project_root, "cfg", "default.yaml"))
os.makedirs(save_weights_path, exist_ok=True)

# Carrega hiperparâmetros
with open(hyp_path, 'r') as f:
    hyp_dict = yaml.safe_load(f)

def prepare_yolo_dataset(train_dataset, val_dataset, output_dir):
    """
    Gera o dataset para o detector YOLO:
      - Salva imagens (train/val)
      - Salva labels .txt apenas com as boxes de lesão (classe 0)
      - Cria data.yaml
    """
    import os
    from pathlib import Path
    import torch
    import numpy as np
    from PIL import Image
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, **kwargs): return x

    base = Path(output_dir)
    ds_root = base / "detector" / "dataset"
    images_train_dir = ds_root / "images" / "train"
    images_val_dir = ds_root / "images" / "val"
    labels_train_dir = ds_root / "labels" / "train"
    labels_val_dir = ds_root / "labels" / "val"
    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    def tensor_to_uint8_img(img_tensor):
        if isinstance(img_tensor, torch.Tensor):
            x = img_tensor.detach().cpu()
            if x.ndim == 3 and x.shape[0] in (1, 3):
                pass
            elif x.ndim == 3 and x.shape[-1] in (1, 3):
                x = x.permute(2, 0, 1)
            else:
                raise ValueError("Imagem com formato inesperado.")
            if x.max() <= 1.0:
                x = (x * 255.0).clamp(0, 255)
            x = x.byte().permute(1, 2, 0).numpy()
        else:
            x = np.array(img_tensor)
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        return x.astype(np.uint8)

    def save_split(split_name, dataset, img_dir, lbl_dir):
        n = len(dataset)
        saved = 0
        lesion_files = 0
        for i in tqdm(range(n), desc=f"Processando {split_name}"):
            image_tensor, _, boxes_data = dataset[i]
            img_np = tensor_to_uint8_img(image_tensor)
            Image.fromarray(img_np).save(img_dir / f"{i:06d}.jpg", quality=95)

            yolo_rows = []
            if isinstance(boxes_data, torch.Tensor) and boxes_data.numel() > 0 and boxes_data.size(1) >= 6:
                lesion_mask = (boxes_data[:, -1] == 1)  # label==1 => lesão
                lesion_boxes = boxes_data[lesion_mask][:, 0:5]  # [cls,cx,cy,w,h]
                for _, cx, cy, w, h in lesion_boxes.tolist():
                    cx = float(np.clip(cx, 0.0, 1.0))
                    cy = float(np.clip(cy, 0.0, 1.0))
                    w = float(np.clip(w, 0.0, 1.0))
                    h = float(np.clip(h, 0.0, 1.0))
                    yolo_rows.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                if lesion_boxes.shape[0] > 0:
                    lesion_files += 1

            with open(lbl_dir / f"{i:06d}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_rows))
            saved += 1
        return saved, lesion_files

    n_train, train_with_lesion = save_split("treino", train_dataset, images_train_dir, labels_train_dir)
    n_val, val_with_lesion = save_split("validação", val_dataset, images_val_dir, labels_val_dir)

    data_yaml = ds_root / "data.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {ds_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "nc: 1",
            "names: ['lesion']",
        ]),
        encoding="utf-8",
    )
    print(f"YOLO dataset salvo em: {ds_root}")
    print(f"Imagens treino: {n_train} (com lesão: {train_with_lesion})")
    print(f"Imagens validação: {n_val} (com lesão: {val_with_lesion})")
    print(f"YAML: {data_yaml}")

    return {
        "root": str(ds_root),
        "yaml": str(data_yaml),
        "train_images": n_train,
        "val_images": n_val,
        "train_with_lesion": train_with_lesion,
        "val_with_lesion": val_with_lesion,
    }

def get_args():
    parser = argparse.ArgumentParser(description='Treina sistema de recomendação de colagem em duas etapas')
    parser.add_argument('--run-dir', type=str, default=None, 
                      help='Diretório para salvar resultados (se não especificado, usa o mais recente)')
    parser.add_argument('--detector-epochs', type=int, default=20, 
                      help='Número de épocas para treinar o detector')
    parser.add_argument('--recommender-epochs', type=int, default=5, 
                      help='Número de épocas para treinar o recommender')
    parser.add_argument('--batch-size', type=int, default=4, 
                      help='Tamanho do batch')
    parser.add_argument('--device', type=str, default='0', 
                      help='Dispositivo para treinamento (0, 1, cpu)')
    return parser.parse_args()

def get_latest_run_dir():
    """Obtém o diretório do experimento mais recente"""
    weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'paste', 'weights')
    if not os.path.exists(weights_dir):
        return None
        
    dirs = [d for d in os.listdir(weights_dir) if os.path.isdir(os.path.join(weights_dir, d))]
    if not dirs:
        return None
    
    # Ordena por data (formato YYYYMMDD_HHMMSS)
    latest_dir = sorted(dirs, reverse=True)[0]
    return os.path.join(weights_dir, latest_dir)

# Modifique a função main() para usar os argumentos e garantir que não há augmentation no validation set
def main():
    """
    Função principal simplificada usando a abordagem Detector + Classificador.
    """
    args = get_args()
    
    # Configuração de diretórios
    if args.run_dir is None:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = os.path.join(save_weights_path, now)
    
    os.makedirs(args.run_dir, exist_ok=True)
    print(f"Usando diretório de execução: {args.run_dir}")
    
    # Salva os argumentos usados
    with open(os.path.join(args.run_dir, 'training_args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Carrega datasets
    print("Carregando datasets...")
    train_full_dataset = CustomCropDataset(root_path=data_path, hyp=hyp_dict, apply_aug=False)
    val_full_dataset = CustomCropDataset(root_path=data_path, hyp=None, apply_aug=False)
    
    # Dividir em treino/validação
    n = len(train_full_dataset)
    val_split = 0.2
    n_train = int(n * (1 - val_split))
    n_val = n - n_train
    
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n, generator=generator).tolist()
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices[:len(val_full_dataset)])
    
    # Salvar visualizações de debug
    debug_dir = os.path.join(args.run_dir, "debug_boxes")
    os.makedirs(debug_dir, exist_ok=True)
    print("Salvando visualizações de debug...")
    save_clustering_visualizations(train_full_dataset, args.run_dir, num_examples=8)
    
    # Fase 1: Treinar detector YOLO (apenas lesões)
    dataset_dir = os.path.join(args.run_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    print("Preparando dataset para detector...")
    
    data_info = prepare_yolo_dataset(train_dataset, val_dataset, dataset_dir)
    
    print("Treinando detector base (Etapa 1)...")
    detector = YOLO("yolov8n.pt")
    detector_results = detector.train(
        data=data_info['yaml'],
        epochs=args.detector_epochs,
        patience=15,
        batch=args.batch_size,
        imgsz=640,
        save=True,
        device=args.device,
        project=args.run_dir,
        name="detector",
        exist_ok=True,
        cache=False,
        amp=True,
        close_mosaic=10
    )
    
    # Caminho para o melhor modelo detector
    best_detector_path = os.path.join(args.run_dir, "detector", "weights", "best.pt")
    
    # Fase 2: Treinar classificador de contexto
    print("Treinando classificador de contexto (Etapa 2)...")
    classifier = train_simple_classifier(
        train_dataset, 
        val_dataset, 
        best_detector_path,
        args.run_dir,
        epochs=args.recommender_epochs,
        batch_size=args.batch_size,
        lr=0.0005,
        device=args.device
    )
    
    # Inicializar recomendador simples
    from src.paste.paste_recommender import SimplePasteRecommender
    print("Inicializando recomendador simplificado...")
    recommender = SimplePasteRecommender(best_detector_path)
    recommender.classifier = classifier
    
    # Avaliação final
    print("Avaliando o sistema completo...")
    evaluate_paste_recommendations(recommender, val_dataset, num_samples=20, save_dir=args.run_dir)
    
    # NOVO: Salva exemplos exógenos (patches de outras imagens)
    print("Salvando exemplos exógenos de recomendação...")
    save_exogenous_patch_recommendations(
        recommender, 
        val_dataset, 
        num_examples=10,
        save_dir=os.path.join(args.run_dir, 'exogenous_examples')
    )

    # Salva um arquivo de finalização
    with open(os.path.join(args.run_dir, 'training_complete.txt'), 'w') as f:
        f.write(f"Treinamento concluído em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nTreinamento concluído! Todos os resultados foram salvos em:")
    print(f"{args.run_dir}")

# NOVO: Função para visualização das boxes com labels
def visualize_boxes_with_labels(image, boxes, save_path=None):
    """
    Visualiza as boxes de lesão, contexto positivo e negativo com cores distintas e legenda.
    
    Args:
        image: Tensor da imagem [C, H, W]
        boxes: Tensor de boxes [N, 6] onde o último valor é o label (0=neg, 1=lesão, 2=pos)
        save_path: Caminho para salvar a imagem
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Converte imagem para numpy
    img_np = image.permute(1, 2, 0).cpu().numpy()
    
    # Cria figura
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_np)
    
    # Define cores e legendas
    box_colors = {
        0: ('red', 'Contexto Negativo'),
        1: ('lime', 'Lesão'),
        2: ('blue', 'Contexto Positivo')
    }
    
    # Adiciona boxes com cores diferentes conforme o label
    legend_handles = []
    for label in [1, 2, 0]:  # Ordem intencional para garantir que lesão apareça por cima
        for box in boxes:
            if box[-1] == label:
                cls, cx, cy, w, h, _ = box
                x1 = int((cx - w/2) * img_np.shape[1])
                y1 = int((cy - h/2) * img_np.shape[0])
                x2 = int((cx + w/2) * img_np.shape[1])
                y2 = int((cy + h/2) * img_np.shape[0])
                
                color, label_text = box_colors[label]
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                # Adiciona handle para a legenda apenas se ainda não existir para este label
                if not any(h.get_label() == label_text for h in legend_handles):
                    legend_handles.append(patches.Patch(color=color, label=label_text))
    
    # Adiciona legenda
    ax.legend(handles=legend_handles, loc='upper right')
    ax.set_title("Lesão (Verde), Contexto Positivo (Azul), Contexto Negativo (Vermelho)")
    ax.axis('off')
    
    # Salva ou mostra
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

def convert_yolo_to_xyxy(boxes, img_size=640):
    """
    Converte coordenadas YOLO [class_idx, center_x, center_y, width, height]
    para formato [x1, y1, x2, y2] necessário para ROI Align
    """
    # Ignora class_idx, pega apenas as coordenadas
    if boxes.size(0) == 0:
        return boxes
        
    # Extrair componentes: center_x, center_y, width, height
    center_x = boxes[:, 1] * img_size
    center_y = boxes[:, 2] * img_size
    width = boxes[:, 3] * img_size
    height = boxes[:, 4] * img_size
    
    # Calcular x1, y1, x2, y2
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    
    return torch.stack([x1, y1, x2, y2], dim=1)

def train_recommender(recommender, train_dataset, val_dataset, run_dir, epochs=50, batch_size=8, lr=0.001):
    """
    Treina o recomendador com foco em compreensão contextual e generalizabilidade
    """
    # Configuração de treinamento
    device = next(recommender.parameters()).device
    optimizer = torch.optim.Adam(recommender.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    
    # Analisa distribuição do domínio (agnóstica ao problema específico)
    domain_stats = analyze_domain_distribution(train_dataset)
    
    # Salva estatísticas e visualização
    with open(os.path.join(run_dir, 'domain_stats.json'), 'w') as f:
        json.dump(domain_stats, f, indent=2)
    
    visualize_domain_distribution(domain_stats, os.path.join(run_dir, 'domain_distribution.png'))
    
    # Dataloaders com batch_size reduzido para validação
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size // 2, shuffle=False, collate_fn=custom_collate)
    
    # Margem dinâmica: começa alto e diminui para afinar
    base_margin = 0.5
    
    # Lista para armazenar as losses
    train_losses = []
    val_losses = []
    
    # Loop de treinamento
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Ajusta margem ao longo do treinamento (de 0.5 para 0.3)
        current_margin = base_margin * (1.0 - 0.4 * epoch / epochs)
        
        # Limpa memória no início de cada época
        torch.cuda.empty_cache()
        
        # Treinamento
        recommender.train()
        train_loss = 0.0
        
        # Use enumerate para obter o índice do batch
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)")):
            # Limpa memória CUDA a cada 3 batches
            if torch.cuda.is_available() and batch_idx % 3 == 0:
                torch.cuda.empty_cache()
                
            # Extrai imagens e boxes
            images, _, boxes, labels, img_indices = batch
            images = images.to(device)
            
            # Pula batch se não houver boxes de lesão
            if torch.sum(labels == 1) == 0:
                continue
                
            # Extrai embeddings para triplet loss
            features = recommender.extract_features(images)
            
            # Separa lesões, contextos positivos e contextos negativos
            lesion_mask = labels == 1   # Lesão
            positive_mask = labels == 2 # Contexto positivo (novo)
            negative_mask = labels == 0 # Contexto negativo
            
            # Verifica se temos todos os tipos necessários
            if not lesion_mask.any() or not positive_mask.any() or not negative_mask.any():
                continue
                
            lesion_boxes = boxes[lesion_mask]
            lesion_indices = img_indices[lesion_mask]
            
            positive_boxes = boxes[positive_mask]
            positive_indices = img_indices[positive_mask]
            
            negative_boxes = boxes[negative_mask]
            negative_indices = img_indices[negative_mask]
            
            # Pula se não houver lesões, positivos ou negativos
            if lesion_boxes.size(0) == 0 or positive_boxes.size(0) == 0 or negative_boxes.size(0) == 0:
                continue
                
            # Converte boxes para formato correto (xyxy)
            lesion_boxes_xyxy = convert_yolo_to_xyxy(lesion_boxes)
            positive_boxes_xyxy = convert_yolo_to_xyxy(positive_boxes)
            negative_boxes_xyxy = convert_yolo_to_xyxy(negative_boxes)
            
            # Extrai embeddings (usando coordenadas convertidas)
            lesion_embeddings = recommender.extract_roi_features(features, lesion_boxes_xyxy, lesion_indices)
            positive_embeddings = recommender.extract_roi_features(features, positive_boxes_xyxy, positive_indices)
            negative_embeddings = recommender.extract_roi_features(features, negative_boxes_xyxy, negative_indices)
            
            
            # Usa o loss contextual com foco em hard negatives
            hard_negative_ratio = 0.3 + 0.4 * epoch / epochs  # Aumenta o foco em hard negatives ao longo do treino
            
            # Antes de calcular o triplet_loss, aplicar hardness mining
            lesion_emb_hard, positive_emb_hard, negative_emb_hard = hardness_mining(
                lesion_embeddings, positive_embeddings, negative_embeddings, hardness_ratio=hard_negative_ratio
            )

            triplet_loss = compute_contextual_triplet_loss(
                lesion_emb_hard, positive_emb_hard, negative_emb_hard,
                margin=current_margin,
                hard_negative_ratio=hard_negative_ratio
            )
            
            # Backprop
            optimizer.zero_grad()
            triplet_loss.backward()
            torch.nn.utils.clip_grad_norm_(recommender.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += triplet_loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)  # <-- ADICIONE ESTA LINHA
        
        # Validação
        recommender.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)")):
                # Limpa memória CUDA a cada batch na validação
                torch.cuda.empty_cache()
                
                # Extrai imagens e boxes
                images, _, boxes, labels, img_indices = batch
                images = images.to(device)
                
                # Pula batch se não houver boxes de lesão
                if torch.sum(labels == 1) == 0:
                    continue
                
                # Extrai embeddings para triplet loss
                features = recommender.extract_features(images)
                
                # Separa lesões e backgrounds
                lesion_mask = labels == 1
                lesion_boxes = boxes[lesion_mask]
                lesion_indices = img_indices[lesion_mask]
                
                bg_mask = labels == 0
                bg_boxes = boxes[bg_mask]
                bg_indices = img_indices[bg_mask]
                
                # Pula se não houver lesões ou backgrounds
                if lesion_boxes.size(0) == 0 or bg_boxes.size(0) == 0:
                    continue
                
                # Converte boxes para formato correto
                lesion_boxes_xyxy = convert_yolo_to_xyxy(lesion_boxes)
                bg_boxes_xyxy = convert_yolo_to_xyxy(bg_boxes)
                
                # Extrai embeddings (usando coordenadas convertidas)
                lesion_embeddings = recommender.extract_roi_features(features, lesion_boxes_xyxy, lesion_indices)
                bg_embeddings = recommender.extract_roi_features(features, bg_boxes_xyxy, bg_indices)
                
                # Para validação, usa o triplet loss simples
                triplet_loss = compute_triplet_loss(lesion_embeddings, bg_embeddings, margin=0.3)
                
                val_loss += triplet_loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)      # <-- ADICIONE ESTA LINHA
            
        # Atualiza learning rate
        scheduler.step(val_loss)
        
        # Salva modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(recommender.state_dict(), os.path.join(run_dir, "best_recommender.pt"))
            print(f"Salvo melhor modelo com loss {val_loss:.4f}")
            save_checkpoint(recommender, optimizer, epoch, val_loss, run_dir, is_best=True)
        else:
            # Salvar checkpoint regular
            save_checkpoint(recommender, optimizer, epoch, val_loss, run_dir, is_best=False)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Salva gráfico da loss do recomendador
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Evolução da Loss do Recomendador')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'recommender_loss.png'), dpi=150)
    plt.close()
    print(f"Gráfico da loss do recomendador salvo em {os.path.join(run_dir, 'recommender_loss.png')}")

    # Carrega melhor modelo
    recommender.load_state_dict(torch.load(os.path.join(run_dir, "best_recommender.pt")))
    
    # Extrai e salva contextos de referência
    print("Extraindo contextos de referência...")
    reference_contexts = recommender.extract_reference_contexts(train_dataset)

    if reference_contexts is not None:
        # Salva para uso posterior
        torch.save(reference_contexts, os.path.join(run_dir, 'reference_contexts.pt'))
        # Atribui ao recommender para uso imediato
        recommender.reference_contexts = reference_contexts
        print("Contextos de referência extraídos e salvos com sucesso")
    
    # Adicione depois de carregar o recommender e antes de iniciar a avaliação
    print("Testando novas funcionalidades...")
    test_dir = os.path.join(run_dir, "feature_tests")
    os.makedirs(test_dir, exist_ok=True)

    # Testa simulações de colagem
    for i in range(3):
        idx = random.randint(0, len(val_dataset)-1)
        img, _, boxes_data = val_dataset[idx]
        
        # Encontra uma lesão para usar como patch
        for box in boxes_data:
            if box.size(0) > 5 and box[5] == 1:  # Se é uma lesão
                # Extrai patch
                cx, cy, w, h = box[1:5]
                x1 = max(0, int((cx - w/2) * 640))
                y1 = max(0, int((cy - h/2) * 640))
                x2 = min(640, int((cx + w/2) * 640))
                y2 = min(640, int((cy + h/2) * 640))
                
                patch = img[:, y1:y2, x1:x2]
                
                # Interpola para tamanho adequado
                patch = F.interpolate(patch.unsqueeze(0), size=(128, 128), mode='bilinear').squeeze(0)
                
                # Obtém recomendações
                img = img.to(device)
                patch = patch.to(device)
                recommendations = recommender.recommend_placements(img.unsqueeze(0), patch.unsqueeze(0))
                
                if recommendations:
                    # Testa visualização com simulação
                    fig = recommender.visualize_recommendations(
                        img.unsqueeze(0), 
                        patch.unsqueeze(0), 
                        recommendations, 
                        show_simulated=True,
                        save_path=os.path.join(test_dir, f"sim_test_{i}.png")
                    )
                    plt.close(fig)
                    break
    
    # Avaliação final
    print("Avaliando o sistema completo...")
    evaluate_paste_recommendations(recommender, val_dataset, num_samples=20, save_dir=run_dir)

    # Avalia compreensão de contexto
    evaluate_context_understanding(
        recommender, 
        val_dataset, 
        num_samples=5, 
        save_dir=os.path.join(run_dir, 'context_eval'),
        domain_stats=domain_stats
    )
    
    print("Calculando métricas avançadas...")
    advanced_metrics = calculate_advanced_metrics(
        recommender, val_dataset, domain_stats=domain_stats, num_samples=20
    )
    # Salva métricas avançadas
    with open(os.path.join(run_dir, 'advanced_metrics.json'), 'w') as f:
        json.dump(advanced_metrics, f, indent=2)

    # NOVO: Salva exemplos exógenos (patches de outras imagens)
    print("Salvando exemplos exógenos de recomendação...")
    save_exogenous_patch_recommendations(
        recommender, 
        val_dataset, 
        num_examples=10,  # ajuste o número conforme desejar
        save_dir=os.path.join(run_dir, 'exogenous_examples')
    )

    # Salva um arquivo de finalização
    with open(os.path.join(run_dir, 'training_complete.txt'), 'w') as f:
        f.write(f"Treinamento concluído em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nTreinamento concluído! Todos os resultados foram salvos em:")
    print(f"{run_dir}")
    
# Adicione após a definição de train_recommender, aproximadamente linha 655

def save_checkpoint(model, optimizer, epoch, loss, run_dir, is_best=False):
    """
    Salva checkpoint do modelo para retomar treinamento se necessário
    
    Args:
        model: Modelo a ser salvo
        optimizer: Otimizador
        epoch: Época atual
        loss: Loss atual
        run_dir: Diretório do experimento
        is_best: Se é o melhor modelo até agora
    """
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Salva checkpoint regular
    torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    # Se for o melhor modelo, salva uma cópia separada
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_checkpoint.pt'))
    
    # Mantém apenas os últimos 3 checkpoints para economizar espaço
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if len(checkpoints) > 3:
        checkpoints.sort()
        oldest_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
        if os.path.exists(oldest_checkpoint):
            os.remove(oldest_checkpoint)

def load_checkpoint(model, optimizer, run_dir, resume_epoch=None):
    """
    Carrega checkpoint para continuar treinamento
    
    Args:
        model: Modelo a ser carregado
        optimizer: Otimizador
        run_dir: Diretório do experimento
        resume_epoch: Época específica para retomar (opcional)
        
    Returns:
        Tupla (modelo, otimizador, época_inicial, melhor_loss)
    """
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    
    # Se não há diretório de checkpoints, retorna o modelo como está
    if not os.path.exists(checkpoint_dir):
        return model, optimizer, 0, float('inf')
    
    # Decide qual checkpoint carregar
    if resume_epoch is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{resume_epoch}.pt')
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint da época {resume_epoch} não encontrado. Iniciando do zero.")
            return model, optimizer, 0, float('inf')
    else:
        # Tenta carregar o melhor checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
        if not os.path.exists(checkpoint_path):
            # Se não há melhor checkpoint, tenta carregar o mais recente
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
            if not checkpoints:
                return model, optimizer, 0, float('inf')
                
            checkpoints.sort(reverse=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
    
    # Carrega o checkpoint
    print(f"Carregando checkpoint de {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch + 1, loss

def compute_triplet_loss(anchor_embeddings, negative_embeddings, margin=0.3):
    """
    Calcula o triplet loss entre embeddings de âncora (lesões) e exemplos negativos.
    
    Args:
        anchor_embeddings: Tensor de embeddings de âncora [N, D]
        negative_embeddings: Tensor de embeddings negativos [M, D]
        margin: Margem para o triplet loss
    
    Returns:
        Tensor com o valor do triplet loss
    """
    # Se não houver embeddings suficientes, retorna zero
    if anchor_embeddings.size(0) == 0 or negative_embeddings.size(0) == 0:
        return torch.tensor(0.0, device=anchor_embeddings.device)
    
    # Calcula todas as combinações de distâncias entre âncoras e negativos
    # Usa squared L2 distance
    distances = torch.cdist(anchor_embeddings, negative_embeddings, p=2.0)**2
    
    # Seleciona os negativos mais difíceis (mais próximos) para cada âncora
    hard_negative_distances, _ = distances.min(dim=1)
    
    # Calcula a perda (usando uma versão simplificada do triplet loss para casos binários)
    losses = F.relu(margin - hard_negative_distances)
    
    return losses.mean()

def compute_contextual_triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin=0.3, hard_negative_ratio=0.5):
    """
    Calcula o triplet loss contextual usando exemplos positivos e negativos.
    
    Args:
        anchor_embeddings: Tensor de embeddings de âncora (lesões) [N, D]
        positive_embeddings: Tensor de embeddings positivos (contexto bom) [P, D]
        negative_embeddings: Tensor de embeddings negativos (contexto ruim) [M, D]
        margin: Margem para o triplet loss
        hard_negative_ratio: Proporção de negativos difíceis a usar
    
    Returns:
        Tensor com o valor do triplet loss contextual
    """
    # Se não houver embeddings suficientes, retorna zero
    if (anchor_embeddings.size(0) == 0 or positive_embeddings.size(0) == 0 
        or negative_embeddings.size(0) == 0):
        return torch.tensor(0.0, device=anchor_embeddings.device)
    
    # Calcula todas as combinações de distâncias
    # Distâncias âncora-positivo
    ap_distances = torch.cdist(anchor_embeddings, positive_embeddings, p=2.0)**2
    # Distâncias âncora-negativo
    an_distances = torch.cdist(anchor_embeddings, negative_embeddings, p=2.0)**2
    
    # Para cada âncora, seleciona o positivo mais difícil (mais distante)
    hard_positive_distances, _ = ap_distances.max(dim=1)
    
    # Para cada âncora, seleciona o negativo mais difícil (mais próximo)
    hard_negative_distances, _ = an_distances.min(dim=1)
    
    # Também seleciona alguns negativos semi-difíceis
    num_hard_negatives = max(1, int(negative_embeddings.size(0) * hard_negative_ratio))
    semi_hard_indices = an_distances.argsort(dim=1)[:, :num_hard_negatives]
    
    # Para cada âncora, calcula a média das distâncias para os negativos semi-difíceis
    semi_hard_distances = torch.gather(an_distances, 1, semi_hard_indices).mean(dim=1)
    
    # Combina hard negatives e semi-hard negatives
    negative_distances = 0.7 * hard_negative_distances + 0.3 * semi_hard_distances
    
    # Calcula triplet loss: max(0, d(a,p) - d(a,n) + margin)
    losses = F.relu(hard_positive_distances - negative_distances + margin)
    
    return losses.mean()

def hardness_mining(anchor_features, positive_features, negative_features, hardness_ratio=0.5):
    """
    Implementa técnicas avançadas de hardness mining para melhorar o treinamento
    
    Args:
        anchor_features: Features das âncoras
        positive_features: Features dos exemplos positivos
        negative_features: Features dos exemplos negativos
        hardness_ratio: Proporção de exemplos difíceis a manter
        
    Returns:
        Tupla (anchor_hard, positive_hard, negative_hard) com exemplos selecionados
    """
    if len(anchor_features) == 0 or len(positive_features) == 0 or len(negative_features) == 0:
        return anchor_features, positive_features, negative_features
    
    # Dispositivo atual
    device = anchor_features.device
    
    # 1. Encontrar positivos difíceis (mais distantes das âncoras)
    ap_dist = torch.cdist(anchor_features, positive_features, p=2.0)
    
    # Para cada âncora, ordena positivos por distância (decrescente)
    ap_hard_indices = torch.argsort(ap_dist, dim=1, descending=True)
    
    # Seleciona os top-k positivos mais difíceis para cada âncora
    k_pos = max(1, int(positive_features.size(0) * hardness_ratio))
    hard_positive_indices = ap_hard_indices[:, :k_pos].reshape(-1)
    hard_positive_indices = torch.unique(hard_positive_indices)
    
    # 2. Encontrar negativos difíceis (mais próximos das âncoras)
    an_dist = torch.cdist(anchor_features, negative_features, p=2.0)
    
    # Para cada âncora, ordena negativos por distância (crescente)
    an_hard_indices = torch.argsort(an_dist, dim=1, descending=False)
    
    # Seleciona os top-k negativos mais difíceis para cada âncora
    k_neg = max(1, int(negative_features.size(0) * hardness_ratio))
    hard_negative_indices = an_hard_indices[:, :k_neg].reshape(-1)
    hard_negative_indices = torch.unique(hard_negative_indices)
    
    # 3. Selecionando âncoras para focar (aquelas com positivos mais difíceis)
    anchor_hardness = torch.max(ap_dist, dim=1)[0] - torch.min(an_dist, dim=1)[0]
    anchor_hard_indices = torch.argsort(anchor_hardness, descending=True)
    
    # Seleciona as top-k âncoras mais difíceis
    k_anchor = max(1, int(anchor_features.size(0) * hardness_ratio))
    hard_anchor_indices = anchor_hard_indices[:k_anchor]
    
    # Retorna subconjuntos selecionados
    return (
        anchor_features[hard_anchor_indices],
        positive_features[hard_positive_indices],
        negative_features[hard_negative_indices]
    )

# Substitua a função analyze_domain_distribution existente por esta versão aprimorada

def analyze_domain_distribution(dataset, num_samples=100, grid_size=10):
    """
    Versão melhorada para analisar a distribuição espacial e visual das lesões no domínio.
    
    Args:
        dataset: Dataset para análise
        num_samples: Número de amostras para análise
        grid_size: Tamanho do grid para análise espacial
        
    Returns:
        Dictionary com estatísticas do domínio
    """
    print("Analisando distribuição do domínio (versão avançada)...")
    
    # Amostra aleatória de imagens (usando seed fixa para reprodutibilidade)
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Estatísticas para lesões
    lesion_positions = []  # Posições normalizadas das lesões
    lesion_sizes = []      # Tamanhos normalizados das lesões
    lesion_colors = []     # Cores médias das lesões
    lesion_textures = []   # Texturas das lesões (variância local)
    lesion_aspects = []    # Razão de aspecto (largura/altura)
    
    # Grid de frequência espacial multi-escala
    spatial_grid = np.zeros((grid_size, grid_size))
    valid_regions = [[False for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Adjacência de lesões (quantas lesões ocorrem próximas umas das outras)
    lesion_adjacency = []
    
    # Distribuição de bordas da imagem
    border_distribution = np.zeros(4)  # [top, right, bottom, left]
    border_threshold = 0.1  # 10% da borda é considerado "borda"
    
    # Loop pelos samples
    for idx in indices:
        img, _, boxes = dataset[idx]
        img_np = img.permute(1, 2, 0).numpy()
        h, w = img_np.shape[:2]
        
        if boxes.dim() < 2:
            continue
        
        # Lista de lesões nesta imagem
        image_lesions = []
        
        # Processa cada box
        for box in boxes:
            # Verifica se tem 6 elementos e se é uma lesão (label=1)
            if box.size(0) > 5 and box[5] == 1:
                # Extrai coordenadas normalizadas
                _, cx, cy, bw, bh, _ = box
                
                # Registra posição na grid
                grid_x = min(grid_size - 1, int(cx * grid_size))
                grid_y = min(grid_size - 1, int(cy * grid_size))
                spatial_grid[grid_y, grid_x] += 1
                valid_regions[grid_y][grid_x] = True
                
                # Guarda estatísticas básicas
                lesion_positions.append((float(cx), float(cy)))
                lesion_sizes.append((float(bw), float(bh)))
                lesion_aspects.append(float(bw/max(bh, 1e-5)))
                image_lesions.append((float(cx), float(cy)))
                
                # Verifica se está na borda
                if cy < border_threshold:
                    border_distribution[0] += 1  # top
                if cx > (1.0 - border_threshold):
                    border_distribution[1] += 1  # right
                if cy > (1.0 - border_threshold):
                    border_distribution[2] += 1  # bottom
                if cx < border_threshold:
                    border_distribution[3] += 1  # left
                
                # Extrai região da lesão
                x1 = int(max(0, (cx - bw/2) * w))
                y1 = int(max(0, (cy - bh/2) * h))
                x2 = int(min(w, (cx + bw/2) * w))
                y2 = int(min(h, (cy + bh/2) * h))
                
                if x2 > x1 and y2 > y1:  # Garantir que região é válida
                    region = img_np[y1:y2, x1:x2]
                    lesion_colors.append(np.mean(region, axis=(0, 1)).tolist())
                    
                    # Textura aproximada (análise mais detalhada)
                    gray_region = np.mean(region, axis=2)
                    if gray_region.size > 0:
                        # Variância simples
                        texture_var = np.var(gray_region)
                        
                        # Entropia (mede a "desordem" ou complexidade)
                        if gray_region.size > 100:  # Região grande o suficiente
                            try:
                                hist, _ = np.histogram(gray_region, bins=32, range=(0, 1))
                                hist = hist / hist.sum()
                                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                                texture_metric = (texture_var + entropy) / 2
                            except:
                                texture_metric = texture_var
                        else:
                            texture_metric = texture_var
                        
                        lesion_textures.append(float(texture_metric))
        
        # Calcula adjacência de lesões nesta imagem
        if len(image_lesions) > 1:
            positions = np.array(image_lesions)
            # Matriz de distâncias entre todas as lesões
            dist_matrix = np.sqrt(((positions[:, np.newaxis, :] - positions[np.newaxis, :, :]) ** 2).sum(axis=2))
            # Exclui a diagonal (distância de uma lesão para ela mesma)
            mask = ~np.eye(dist_matrix.shape[0], dtype=bool)
            distances = dist_matrix[mask]
            # Guarda distâncias médias entre lesões nesta imagem
            lesion_adjacency.append(float(np.mean(distances)))
    
    # Normaliza o grid de frequência
    if np.sum(spatial_grid) > 0:
        spatial_grid = spatial_grid / np.sum(spatial_grid)
    
    # Normaliza distribuição de bordas
    if np.sum(border_distribution) > 0:
        border_distribution = border_distribution / np.sum(border_distribution)
    
    # Calcula densidade de posição usando KDE
    try:
        if lesion_positions:
            kde = KernelDensity(bandwidth=0.05, kernel='gaussian')
            kde.fit(np.array(lesion_positions))
            
            # Amostra pontos na grid para visualizar distribuição
            x, y = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
            xy_sample = np.vstack([x.ravel(), y.ravel()]).T
            
            # Calcula log-densidade
            density = np.exp(kde.score_samples(xy_sample))
            density = density.reshape(grid_size, grid_size)
            
            # Normaliza para visualização
            if np.sum(density) > 0:
                density = density / np.sum(density)
        else:
            density = spatial_grid
    except:
        density = spatial_grid
    
    # Análise de co-ocorrência (padrões de disposição espacial)
    patterns = {}
    if len(lesion_positions) > 1:
        try:
            from sklearn.cluster import DBSCAN
            # Agrupamento de lesões para encontrar padrões
            clustering = DBSCAN(eps=0.15, min_samples=2).fit(np.array(lesion_positions))
            clusters = clustering.labels_
            
            # Conta quantas lesões estão em clusters vs. isoladas
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_isolated = list(clusters).count(-1)
            
            patterns = {
                'clusters': int(n_clusters),
                'isolated': int(n_isolated),
                'cluster_ratio': float(n_clusters) / (n_clusters + n_isolated + 1e-6)
            }
        except:
            pass
    
    # Compila estatísticas do domínio
    domain_stats = {
        'positions': lesion_positions,
        'sizes': lesion_sizes,
        'spatial_grid': spatial_grid.tolist(),
        'density': density.tolist(),
        'valid_regions': valid_regions,
        'colors': lesion_colors,
        'textures': lesion_textures,
        'aspects': lesion_aspects,
        'adjacency': lesion_adjacency,
        'border_distribution': border_distribution.tolist(),
        'patterns': patterns,
        'mean_color': np.mean(lesion_colors, axis=0).tolist() if lesion_colors else [0.5, 0.5, 0.5],
        'std_color': np.std(lesion_colors, axis=0).tolist() if lesion_colors else [0.1, 0.1, 0.1],
        'mean_size': np.mean(lesion_sizes, axis=0).tolist() if lesion_sizes else [0.1, 0.1],
        'std_size': np.std(lesion_sizes, axis=0).tolist() if lesion_sizes else [0.05, 0.05],
        'mean_aspect': float(np.mean(lesion_aspects)) if lesion_aspects else 1.0,
        'mean_texture': float(np.mean(lesion_textures)) if lesion_textures else 0.1
    }
    
    return domain_stats

def visualize_domain_distribution(domain_stats, save_path):
    """
    Visualiza a distribuição espacial de lesões no domínio.
    """
    grid_size = len(domain_stats['spatial_grid'])
    density = np.array(domain_stats['density'])
    
    # Criar visualização
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot da densidade espacial
    im = axes[0].imshow(density, cmap='viridis', origin='lower')
    axes[0].set_title("Densidade Espacial de Lesões")
    axes[0].set_xlabel("Coordenada X Normalizada")
    axes[0].set_ylabel("Coordenada Y Normalizada")
    
    # CORREÇÃO: Primeiro definir os ticks
    x_ticks = np.arange(0, grid_size, 2)
    y_ticks = np.arange(0, grid_size, 2)
    
    # Definir os ticks
    axes[0].set_xticks(x_ticks)
    axes[0].set_yticks(y_ticks)
    
    # Definir os labels correspondentes aos ticks (mesmo número)
    axes[0].set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, len(x_ticks))])
    axes[0].set_yticklabels([f"{y:.1f}" for y in np.linspace(0, 1, len(y_ticks))])
    
    fig.colorbar(im, ax=axes[0], label="Probabilidade Relativa")
    
    # Plot de tamanhos (se disponível)
    if domain_stats.get('sizes') and len(domain_stats['sizes']) > 1:
        sizes = np.array(domain_stats['sizes'])
        axes[1].scatter(sizes[:, 0], sizes[:, 1], alpha=0.5, s=50)
        axes[1].set_title("Distribuição de Tamanhos de Lesões")
        axes[1].set_xlabel("Largura Normalizada")
        axes[1].set_ylabel("Altura Normalizada")
        axes[1].grid(True, alpha=0.3)
        
        # Adicionar média
        mean_size = np.array(domain_stats['mean_size'])
        axes[1].scatter([mean_size[0]], [mean_size[1]], color='red', s=100, marker='x')
    else:
        axes[1].set_title("Distribuição de Tamanhos\n(Dados Insuficientes)")
        axes[1].set_axis_off()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def evaluate_paste_recommendations(recommender, val_dataset, num_samples=20, save_dir=None):
    """
    Avalia e visualiza as recomendações de colagem geradas pelo modelo.
    
    Args:
        recommender: Modelo PasteRecommender treinado
        val_dataset: Dataset de validação
        num_samples: Número de amostras para avaliação
        save_dir: Diretório para salvar resultados
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        vis_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
    if hasattr(recommender, 'parameters'):
        # Para nn.Module (PasteRecommender)
        device = next(recommender.parameters()).device
    elif hasattr(recommender, 'device'):
        # Para SimplePasteRecommender
        device = recommender.device
    else:
        # Fallback para CPU
        device = torch.device('cpu')
        print("AVISO: Não foi possível detectar o dispositivo do recomendador. Usando CPU.")
    
    # device = next(recommender.parameters()).device
    
    # Inicializa métricas
    scores = {
        'combined': [],
        'similarity': [],
        'informativity': [],
        'quality': []
    }
    
    # Seleciona índices aleatórios para avaliação
    indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
    
    all_recommendations = []  # Lista para armazenar recomendações para o evaluator
    
    for i, idx in enumerate(tqdm(indices, desc="Avaliando recomendações")):
        # Carrega imagem e boxes
        img, _, boxes_data = val_dataset[idx]
        img = img.to(device)
        
        # Procura por uma lesão para usar como patch
        for box in boxes_data:
            if box.size(0) > 5 and box[5] == 1:  # Se é uma lesão
                # Extrai patch
                cx, cy, w, h = box[1:5]
                x1 = max(0, int((cx - w/2) * 640))
                y1 = max(0, int((cy - h/2) * 640))
                x2 = min(640, int((cx + w/2) * 640))
                y2 = min(640, int((cy + h/2) * 640))
                
                patch = img[:, y1:y2, x1:x2]
                
                # Redimensiona patch para tamanho adequado
                patch = F.interpolate(patch.unsqueeze(0), size=(128, 128), mode='bilinear').squeeze(0)
                
                # Obtém recomendações
                recommendations = recommender.recommend_placements(img.unsqueeze(0), patch.unsqueeze(0))
                
                # Armazena para uso com evaluator
                if recommendations:
                    all_recommendations.append({
                        'img': img,
                        'patch': patch,
                        'recommendations': recommendations
                    })
                
                # Registra métricas
                if recommendations:
                    for rec in recommendations[:3]:  # Top-3 recomendações
                        scores['combined'].append(rec['combined_score'])
                        scores['similarity'].append(rec.get('similarity_score', 0))
                        scores['informativity'].append(rec.get('informativity_score', 0))
                        scores['quality'].append(rec.get('quality_score', 0))
                    
                    # Visualiza e salva resultado
                    if save_dir:
                        fig = recommender.visualize_recommendations(
                            img.unsqueeze(0), 
                            patch.unsqueeze(0), 
                            recommendations,
                            show_simulated=True,
                            save_path=os.path.join(vis_dir, f"recommendation_{i}.png")
                        )
                        plt.close(fig)
                break
    
    # Calcula estatísticas
    stats = {}
    for key, values in scores.items():
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    # Salva estatísticas
    if save_dir:
        with open(os.path.join(save_dir, 'recommendation_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
            
        # Gera gráficos de distribuição
        plt.figure(figsize=(10, 6))
        for key, values in scores.items():
            if values:
                sns.kdeplot(values, label=key)
        plt.title("Distribuição dos Scores de Recomendação")
        plt.xlabel("Score")
        plt.ylabel("Densidade")
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'score_distribution.png'), dpi=150)
        plt.close()
    
    # Inicializar avaliador para análise detalhada
    evaluator = RecommendationEvaluator(recommender)

    # Avaliar algumas recomendações
    for item in all_recommendations[:5]:  # Avalia até 5 exemplos
        img = item['img']
        patch = item['patch']
        for i, rec in enumerate(item['recommendations'][:3]):  # Top 3 recomendações
            metrics = evaluator.evaluate_recommendation(
                img.unsqueeze(0), patch.unsqueeze(0), rec
            )

    # Obter métricas agregadas
    performance_metrics = evaluator.get_performance_metrics()
    if save_dir:
        with open(os.path.join(save_dir, 'evaluator_metrics.json'), 'w') as f:
            json.dump(performance_metrics, f, indent=2)
    
    return stats

def evaluate_context_understanding(recommender, val_dataset, num_samples=5, save_dir=None, domain_stats=None):
    """
    Avalia a compreensão de contexto do modelo, analisando onde ele recomenda colar lesões.
    
    Args:
        recommender: Modelo PasteRecommender treinado
        val_dataset: Dataset de validação
        num_samples: Número de amostras para avaliação
        save_dir: Diretório para salvar resultados
        domain_stats: Estatísticas do domínio (opcional)
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    device = next(recommender.parameters()).device
    
    # Inicializa métricas de contexto
    context_distances = []
    spatial_consistency = []
    
    # Seleciona índices aleatórios para avaliação
    indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
    
    for i, idx in enumerate(tqdm(indices, desc="Avaliando compreensão de contexto")):
        # Carrega imagem e boxes
        img, _, boxes_data = val_dataset[idx]
        img = img.to(device)
        
        # Extrai todos os patches de lesão da imagem
        lesion_patches = []
        lesion_positions = []
        
        for box in boxes_data:
            if box.size(0) > 5 and box[5] == 1:  # Se é uma lesão
                # Extrai patch
                cx, cy, w, h = box[1:5]
                x1 = max(0, int((cx - w/2) * 640))
                y1 = max(0, int((cy - h/2) * 640))
                x2 = min(640, int((cx + w/2) * 640))
                y2 = min(640, int((cy + h/2) * 640))
                
                patch = img[:, y1:y2, x1:x2]
                
                # Redimensiona patch para tamanho adequado
                patch = F.interpolate(patch.unsqueeze(0), size=(128, 128), mode='bilinear').squeeze(0)
                
                lesion_patches.append(patch)
                lesion_positions.append((float(cx), float(cy)))
        
        # Pular se não há lesões
        if not lesion_patches:
            continue
            
        # Para cada patch, gerar recomendações
        all_recommendations = []
        for patch_idx, patch in enumerate(lesion_patches):
            recommendations = recommender.recommend_placements(img.unsqueeze(0), patch.unsqueeze(0))
            
            if recommendations:
                # Registra recomendações
                for rec_idx, rec in enumerate(recommendations[:3]):  # Top-3
                    box = rec['box']
                    cx = (box[0] + box[2]) / 2 / 640  # Centro normalizado
                    cy = (box[1] + box[3]) / 2 / 640
                    
                    all_recommendations.append({
                        'patch_idx': patch_idx,
                        'rec_idx': rec_idx,
                        'position': (cx, cy),
                        'score': rec['combined_score']
                    })
        
        # Analisa distribuição espacial de recomendações vs. lesões reais
        if all_recommendations and lesion_positions:
            # Calcula distância média às lesões reais
            for rec in all_recommendations:
                rec_pos = rec['position']
                
                # Calcula distâncias euclidianas para todas as lesões reais
                distances = [np.sqrt((rec_pos[0] - lp[0])**2 + (rec_pos[1] - lp[1])**2) for lp in lesion_positions]
                
                # Pega a menor distância (a lesão real mais próxima)
                min_dist = min(distances)
                context_distances.append(min_dist)
            
            # Se temos estatísticas do domínio, avalia consistência espacial
            if domain_stats and 'density' in domain_stats:
                density_grid = np.array(domain_stats['density'])
                grid_size = density_grid.shape[0]
                
                # Para cada recomendação, verifica consistência com a distribuição do domínio
                for rec in all_recommendations:
                    cx, cy = rec['position']
                    
                    # Mapeia para grade de densidade
                    grid_x = min(grid_size - 1, int(cx * grid_size))
                    grid_y = min(grid_size - 1, int(cy * grid_size))
                    
                    # Obtém score de densidade neste ponto
                    consistency = density_grid[grid_y, grid_x]
                    spatial_consistency.append(consistency)
    
    # Calcula estatísticas
    stats = {}
    if context_distances:
        stats['context_distance'] = {
            'mean': float(np.mean(context_distances)),
            'std': float(np.std(context_distances)),
            'min': float(np.min(context_distances)),
            'max': float(np.max(context_distances))
        }
    
    if spatial_consistency:
        stats['spatial_consistency'] = {
            'mean': float(np.mean(spatial_consistency)),
            'std': float(np.std(spatial_consistency)),
            'min': float(np.min(spatial_consistency)),
            'max': float(np.max(spatial_consistency))
        }
    
    # Salva estatísticas
    if save_dir:
        with open(os.path.join(save_dir, 'context_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
            
        # Gera histograma de distâncias
        if context_distances:
            plt.figure(figsize=(8, 5))
            plt.hist(context_distances, bins=20, alpha=0.7)
            plt.axvline(np.mean(context_distances), color='red', linestyle='dashed', linewidth=1)
            plt.title("Distribuição de Distâncias às Lesões Reais")
            plt.xlabel("Distância Normalizada")
            plt.ylabel("Frequência")
            plt.savefig(os.path.join(save_dir, 'distance_histogram.png'), dpi=150)
            plt.close()
    
    # Adicionar no final da função evaluate_context_understanding (antes do return stats)

    # Adicionar estatísticas de clusters à análise
    if hasattr(val_dataset.dataset, 'debug_vis') and isinstance(val_dataset, torch.utils.data.Subset):
        # Ativa temporariamente debug para extrair clusters
        val_dataset.dataset.debug_vis = True
        cluster_stats = {
            'positive_size': [],
            'negative_size': [],
            'positive_luminance': [],
            'negative_luminance': [],
            'separation_quality': []
        }
        
        # Analisa clusters de algumas imagens
        for _ in range(min(5, len(val_dataset))):
            idx = random.randint(0, len(val_dataset) - 1)
            img, mask, _ = val_dataset[idx]
            
            # Força geração e análise de clusters
            if hasattr(val_dataset.dataset, 'region_grow_background_boxes'):
                try:
                    # Acessa o _visualize_clusters para extrair estatísticas diretamente
                    clusters_data = val_dataset.dataset._extract_cluster_stats(
                        np.array(img.permute(1, 2, 0)),
                        np.array(mask.squeeze()) > 0
                    )
                    
                    if clusters_data:
                        cluster_stats['positive_size'].append(clusters_data.get('positive_size', 0))
                        cluster_stats['negative_size'].append(clusters_data.get('negative_size', 0))
                        cluster_stats['positive_luminance'].append(clusters_data.get('positive_luminance', 0))
                        cluster_stats['negative_luminance'].append(clusters_data.get('negative_luminance', 0))
                        cluster_stats['separation_quality'].append(clusters_data.get('separation_quality', 0))
                except Exception as e:
                    print(f"Erro ao analisar clusters: {e}")
        
        # Adiciona estatísticas médias ao resultado
        if cluster_stats['positive_size']:
            stats['cluster_analysis'] = {
                'positive_size_mean': np.mean(cluster_stats['positive_size']),
                'negative_size_mean': np.mean(cluster_stats['negative_size']),
                'positive_luminance_mean': np.mean(cluster_stats['positive_luminance']),
                'negative_luminance_mean': np.mean(cluster_stats['negative_luminance']),
                'separation_quality_mean': np.mean(cluster_stats['separation_quality']),
            }
        
        # Restaura configuração original
        val_dataset.dataset.debug_vis = False
    
    return stats

def save_exogenous_patch_recommendations(recommender, val_dataset, num_examples=10, save_dir=None):
    """
    Gera e salva exemplos de recomendações usando patches de uma imagem em outra imagem.
    
    Args:
        recommender: Modelo PasteRecommender treinado
        val_dataset: Dataset de validação
        num_examples: Número de exemplos para gerar
        save_dir: Diretório para salvar os resultados
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    else:
        return
    
    device = next(recommender.parameters()).device
    
    # Seleciona pares de imagens diferentes
    num_images = len(val_dataset)
    if num_images < 2:
        print("Dataset muito pequeno para exemplos exógenos")
        return
    
    # Gera pares aleatórios de imagens diferentes
    pairs = []
    for _ in range(num_examples):
        source_idx = np.random.randint(num_images)
        target_idx = np.random.randint(num_images)
        # Garante que são imagens diferentes
        while target_idx == source_idx:
            target_idx = np.random.randint(num_images)
        pairs.append((source_idx, target_idx))
    
    # Para cada par, gera recomendações
    for i, (source_idx, target_idx) in enumerate(tqdm(pairs, desc="Gerando recomendações exógenas")):
        # Carrega imagem fonte e extrai patch de lesão
        source_img, _, source_boxes = val_dataset[source_idx]
        source_img = source_img.to(device)
        
        # Procura uma lesão na imagem fonte
        patch = None
        for box in source_boxes:
            if box.size(0) > 5 and box[5] == 1:  # Se é uma lesão
                # Extrai patch
                cx, cy, w, h = box[1:5]
                x1 = max(0, int((cx - w/2) * 640))
                y1 = max(0, int((cy - h/2) * 640))
                x2 = min(640, int((cx + w/2) * 640))
                y2 = min(640, int((cy + h/2) * 640))
                
                patch = source_img[:, y1:y2, x1:x2]
                
                # Redimensiona patch para tamanho adequado
                patch = F.interpolate(patch.unsqueeze(0), size=(128, 128), mode='bilinear').squeeze(0)
                break
        
        # Se não encontrou patch, continua para próxima iteração
        if patch is None:
            continue
            
        # Carrega imagem alvo
        target_img, _, _ = val_dataset[target_idx]
        target_img = target_img.to(device)
        
        # Obtém recomendações
        recommendations = recommender.recommend_placements(
            target_img.unsqueeze(0), 
            patch.unsqueeze(0)
        )
        
        # Visualiza e salva
        if recommendations:
            fig = plt.figure(figsize=(15, 10))
            
            # Subplot para imagem fonte
            ax1 = fig.add_subplot(2, 2, 1)
            source_np = source_img.cpu().permute(1, 2, 0).numpy()
            ax1.imshow(source_np)
            ax1.set_title("Imagem Fonte (com lesão)")
            
            # Marca a lesão extraída
            for box in source_boxes:
                if box.size(0) > 5 and box[5] == 1:
                    cx, cy, w, h = box[1:5]
                    x1 = int((cx - w/2) * source_np.shape[1])
                    y1 = int((cy - h/2) * source_np.shape[0])
                    x2 = int((cx + w/2) * source_np.shape[1])
                    y2 = int((cy + h/2) * source_np.shape[0])
                    
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2, edgecolor='lime', facecolor='none'
                    )
                    ax1.add_patch(rect)
            ax1.axis('off')
            
            # Subplot para o patch
            ax2 = fig.add_subplot(2, 2, 2)
            patch_np = patch.cpu().permute(1, 2, 0).numpy()
            ax2.imshow(patch_np)
            ax2.set_title("Patch Extraído")
            ax2.axis('off')
            
            # Subplot para imagem alvo com recomendações
            ax3 = fig.add_subplot(2, 1, 2)
            target_np = target_img.cpu().permute(1, 2, 0).numpy()
            ax3.imshow(target_np)
            ax3.set_title("Imagem Alvo com Recomendações")
            
            # Desenha recomendações
            colors = ['lime', 'cyan', 'yellow', 'magenta', 'white']
            for j, rec in enumerate(recommendations[:5]):
                box = rec['box']
                x1, y1, x2, y2 = box
                color = colors[j % len(colors)]
                
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax3.add_patch(rect)
                
                # Adiciona o score
                ax3.text(
                    x1, y1-5, f"#{j+1}: {rec['combined_score']:.2f}", 
                    color='white', fontsize=10, weight='bold',
                    bbox=dict(facecolor=color, alpha=0.7)
                )
            
            ax3.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"exogenous_{i}.png"), dpi=150)
            plt.close(fig)
            
            # Também salva versão com simulação da colagem
            if hasattr(recommender, 'simulate_paste_result'):
                fig = plt.figure(figsize=(15, 10))
                
                # Imagem original
                ax1 = fig.add_subplot(2, 1, 1)
                ax1.imshow(target_np)
                ax1.set_title("Imagem Original")
                ax1.axis('off')
                
                # Imagem com colagem simulada (primeira recomendação)
                ax2 = fig.add_subplot(2, 1, 2)
                if recommendations:
                    box = recommendations[0]['box']
                    simulated = recommender.simulate_paste_result(
                        target_img, patch, box, blend_mode='poisson'
                    )
                    ax2.imshow(simulated)
                    ax2.set_title("Simulação da Colagem (Primeira Recomendação)")
                ax2.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"exogenous_simulated_{i}.png"), dpi=150)
                plt.close(fig)
                
# Adicione após a função save_exogenous_patch_recommendations, aproximadamente linha 1350

def calculate_advanced_metrics(recommender, val_dataset, domain_stats=None, num_samples=20):
    """
    Calcula métricas avançadas para avaliação do sistema de recomendação
    
    Args:
        recommender: Modelo treinado
        val_dataset: Dataset de validação
        domain_stats: Estatísticas do domínio
        num_samples: Número de amostras para avaliação
        
    Returns:
        Dicionário de métricas
    """
    print("Calculando métricas avançadas de desempenho...")
    device = next(recommender.parameters()).device
    
    # Métricas a coletar
    metrics = {
        'recommendation_quality': [],  # Score combinado das recomendações
        'anatomical_consistency': [],  # Consistência com estatísticas anatômicas
        'context_similarity': [],      # Similaridade de contexto
        'visual_quality': [],          # Qualidade visual da colagem
        'spatial_distribution': []     # Consistência com distribuição espacial
    }
    
    # Seleciona índices aleatórios
    indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
    
    # Para cada amostra
    for idx in tqdm(indices, desc="Calculando métricas"):
        img, _, boxes_data = val_dataset[idx]
        img = img.to(device)
        
        # Procura uma lesão para usar como patch
        for box in boxes_data:
            if box.size(0) > 5 and box[5] == 1:  # Se é uma lesão
                # Extrai patch
                cx, cy, w, h = box[1:5]
                x1 = max(0, int((cx - w/2) * 640))
                y1 = max(0, int((cy - h/2) * 640))
                x2 = min(640, int((cx + w/2) * 640))
                y2 = min(640, int((cy + h/2) * 640))
                
                patch = img[:, y1:y2, x1:x2]
                
                # Redimensiona patch
                patch = F.interpolate(patch.unsqueeze(0), size=(128, 128), mode='bilinear').squeeze(0)
                
                # Obtém recomendações
                recommendations = recommender.recommend_placements(
                    img.unsqueeze(0), patch.unsqueeze(0), domain_stats=domain_stats
                )
                
                # Métricas das recomendações
                if recommendations:
                    top_rec = recommendations[0]  # Melhor recomendação
                    
                    # Métricas básicas da recomendação
                    metrics['recommendation_quality'].append(top_rec.get('combined_score', 0))
                    
                    # Métricas anatômicas
                    if 'details' in top_rec:
                        details = top_rec['details']
                        anatomical_metrics = [
                            details.get('domain_score', 0),
                            details.get('anatomical_score', 0)
                        ]
                        metrics['anatomical_consistency'].append(np.mean(anatomical_metrics))
                    
                    # Similaridade de contexto
                    metrics['context_similarity'].append(top_rec.get('similarity_score', 0))
                    
                    # Consistência com distribuição espacial
                    if domain_stats and 'density' in domain_stats:
                        box = top_rec['box']
                        cx = (box[0] + box[2]) / 2 / 640
                        cy = (box[1] + box[3]) / 2 / 640
                        
                        # Mapeia para grade de densidade
                        density_grid = np.array(domain_stats['density'])
                        grid_size = density_grid.shape[0]
                        
                        grid_x = min(grid_size - 1, int(cx * grid_size))
                        grid_y = min(grid_size - 1, int(cy * grid_size))
                        
                        # Obtém score de densidade neste ponto
                        spatial_score = density_grid[grid_y, grid_x]
                        metrics['spatial_distribution'].append(spatial_score)
                    
                    # Qualidade visual da colagem
                    if hasattr(recommender, 'simulate_paste_result'):
                        # Simula colagem
                        simulated = recommender.simulate_paste_result(
                            img.unsqueeze(0), patch.unsqueeze(0), top_rec['box'], blend_mode='poisson'
                        )
                        
                        # Avalia suavidade da transição na borda
                        # Simplicação: usa desvio padrão de gradiente na borda como proxy
                        if isinstance(simulated, torch.Tensor):
                            simulated = simulated.cpu().numpy()
                            
                        # Extrai região da borda expandida
                        box = top_rec['box']
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        
                        # Expande um pouco
                        border = 5
                        x1b = max(0, x1 - border)
                        y1b = max(0, y1 - border)
                        x2b = min(simulated.shape[1], x2 + border)
                        y2b = min(simulated.shape[0], y2 + border)
                        
                        # Extrai região da borda
                        border_region = simulated[y1b:y2b, x1b:x2b]
                        if border_region.size > 0:
                            # Calcula gradiente
                            if len(border_region.shape) == 3:
                                border_region_gray = np.mean(border_region, axis=2)
                            else:
                                border_region_gray = border_region
                                
                            # Calcula gradiente usando Sobel
                            try:
                                grad_x = cv2.Sobel(border_region_gray, cv2.CV_64F, 1, 0, ksize=3)
                                grad_y = cv2.Sobel(border_region_gray, cv2.CV_64F, 0, 1, ksize=3)
                                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                                
                                # Normaliza para 0-1
                                smoothness = 1.0 - min(1.0, np.std(grad_mag) / 0.2)
                                metrics['visual_quality'].append(smoothness)
                            except:
                                # Fallback se falhar
                                metrics['visual_quality'].append(0.5)
                break
    
    # Calcula estatísticas para cada métrica
    results = {}
    for metric_name, values in metrics.items():
        if values:
            results[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
    
    return results

def save_clustering_visualizations(dataset, run_dir, num_examples=5):
    """
    Salva visualizações dos clusters gerados para avaliação qualitativa
    """
    cluster_dir = os.path.join(run_dir, 'cluster_visualizations')
    region_dir = os.path.join(run_dir, 'region_growing_visualizations')
    os.makedirs(cluster_dir, exist_ok=True)
    os.makedirs(region_dir, exist_ok=True)

    print("Gerando visualizações de clustering...")

    # Ativa temporariamente modo de debug
    orig_debug_state = getattr(dataset, 'debug_vis', False)
    dataset.debug_vis = True

    # Monkey-patch do join para redirecionar os diretórios de debug
    import builtins
    builtins._orig_join = os.path.join

    def custom_join(path, *parts):
        if path == 'debug_clusters':
            return builtins._orig_join(cluster_dir, *parts)
        if path == 'debug_region_growing':
            return builtins._orig_join(region_dir, *parts)
        return builtins._orig_join(path, *parts)

    os.path.join = custom_join

    # Gera exemplos
    for i in range(num_examples):
        try:
            idx = random.randint(0, len(dataset) - 1)
            img, mask, _ = dataset[idx]

            # força pipeline que chama as visualizações
            if hasattr(dataset, 'region_grow_background_boxes'):
                dataset.region_grow_background_boxes(
                    Image.fromarray(img.permute(1, 2, 0).numpy().astype(np.uint8)),
                    Image.fromarray(mask.squeeze().numpy().astype(np.uint8) * 255),
                    expand_ratio=1.5,
                    max_steps=5,
                    similarity_threshold=5.5
                )
        except Exception as e:
            print(f"Erro ao gerar visualização {i}: {e}")

    # Restaura
    os.path.join = builtins._orig_join
    dataset.debug_vis = orig_debug_state

    print(f"Visualizações de clusters salvas em {cluster_dir}")
    print(f"Visualizações de region growing salvas em {region_dir}")
    
def train_simple_classifier(train_dataset, val_dataset, detector_path, run_dir, 
                          epochs=5, batch_size=8, lr=0.0005, device='cuda'):
    """
    Treina o classificador simples de contexto para recomendar locais de colagem.
    
    Args:
        train_dataset: Dataset de treino
        val_dataset: Dataset de validação
        detector_path: Caminho do detector treinado
        run_dir: Diretório para salvar resultados
        epochs: Número de épocas
        batch_size: Tamanho do batch
        lr: Taxa de aprendizagem
        device: Dispositivo para treinamento
        
    Returns:
        Classificador treinado
    """
    from src.paste.paste_recommender import SimpleContextClassifier
    
    # Converter string de dispositivo para formato apropriado
    if device == '0' or device == '1' or device.isdigit():
        device = f'cuda:{device}'
    
    # Verificar se CUDA está disponível quando solicitado
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("AVISO: CUDA solicitado mas não disponível. Usando CPU.")
        device = 'cpu'
        
    device = torch.device(device)
    
    print(f"Inicializando classificador de contexto no dispositivo: {device}")
    classifier = SimpleContextClassifier(detector_path, embedding_size=256)
    classifier = classifier.to(device)
    
    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=lr)
    
    # Importar ReduceLROnPlateau do local correto
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    
    criterion = torch.nn.BCELoss()

    # Inicializa dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=custom_collate,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate,
        num_workers=2
    )
    
    # Cria diretório para checkpoints
    checkpoints_dir = os.path.join(run_dir, 'classifier_checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Loop de treinamento
    for epoch in range(epochs):
        print(f"\nÉpoca {epoch+1}/{epochs}")
        
        # === TREINAMENTO ===
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Coleta exemplos positivos e negativos para treinamento
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Treinando")):
            images, _, boxes, labels, img_indices = batch
            images = images.to(device)
            
            # Extrai boxes de lesão (para patches)
            lesion_mask = labels == 1
            lesion_boxes = boxes[lesion_mask]
            lesion_indices = img_indices[lesion_mask]
            
            # Extrai boxes de contexto positivo
            pos_mask = labels == 2
            pos_boxes = boxes[pos_mask]
            pos_indices = img_indices[pos_mask]
            
            # Extrai boxes de contexto negativo
            neg_mask = labels == 0
            neg_boxes = boxes[neg_mask]
            neg_indices = img_indices[neg_mask]
            
            # Pula batch se não tiver qualquer um dos tipos
            if not (lesion_mask.any() and pos_mask.any() and neg_mask.any()):
                continue
            
            # Converte para formato xyxy
            lesion_boxes_xyxy = convert_yolo_to_xyxy(lesion_boxes)
            pos_boxes_xyxy = convert_yolo_to_xyxy(pos_boxes)
            neg_boxes_xyxy = convert_yolo_to_xyxy(neg_boxes)
            
            # Extrai patches de lesões
            lesion_patches = []
            for i, (box, img_idx) in enumerate(zip(lesion_boxes_xyxy, lesion_indices)):
                try:
                    x1, y1, x2, y2 = [max(0, int(coord)) for coord in box]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Extrai patch
                    patch = images[img_idx, :, y1:y2, x1:x2]
                    if patch.numel() == 0:
                        continue
                    
                    # Redimensiona
                    patch = F.interpolate(patch.unsqueeze(0), size=(128, 128), mode='bilinear')
                    lesion_patches.append(patch)
                except Exception as e:
                    continue
            
            # Extrai regiões positivas e negativas
            pos_regions = []
            for i, (box, img_idx) in enumerate(zip(pos_boxes_xyxy, pos_indices)):
                try:
                    x1, y1, x2, y2 = [max(0, int(coord)) for coord in box]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Extrai região
                    region = images[img_idx, :, y1:y2, x1:x2]
                    if region.numel() == 0:
                        continue
                    
                    # Redimensiona
                    region = F.interpolate(region.unsqueeze(0), size=(128, 128), mode='bilinear')
                    pos_regions.append(region)
                except Exception as e:
                    continue
            
            neg_regions = []
            for i, (box, img_idx) in enumerate(zip(neg_boxes_xyxy, neg_indices)):
                try:
                    x1, y1, x2, y2 = [max(0, int(coord)) for coord in box]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Extrai região
                    region = images[img_idx, :, y1:y2, x1:x2]
                    if region.numel() == 0:
                        continue
                    
                    # Redimensiona
                    region = F.interpolate(region.unsqueeze(0), size=(128, 128), mode='bilinear')
                    neg_regions.append(region)
                except Exception as e:
                    continue
            
            # Pula batch se não tiver patches ou regiões suficientes
            if not lesion_patches or len(pos_regions) < 2 or len(neg_regions) < 2:
                continue
            
            # Prepara mini-batches de treino
            batch_loss = 0.0
            batch_size = min(8, len(lesion_patches))
            num_examples = 0
            
            # Zera gradientes
            optimizer.zero_grad()
            
            # Processa exemplos POSITIVOS (lesão + contexto positivo)
            for _ in range(batch_size):
                # Seleciona patch e região aleatória
                patch = random.choice(lesion_patches)
                region = random.choice(pos_regions)
                
                # Classificar (1.0 = compatível)
                output = classifier(region, patch)
                target = torch.ones_like(output)
                
                # Calcula perda
                loss = criterion(output, target)
                batch_loss += loss
                
                # Conta acerto
                pred = (output > 0.5).float()
                train_correct += (pred == target).sum().item()
                train_total += target.size(0)
                
                num_examples += 1
            
            # Processa exemplos NEGATIVOS (lesão + contexto negativo)
            for _ in range(batch_size):
                # Seleciona patch e região aleatória
                patch = random.choice(lesion_patches)
                region = random.choice(neg_regions)
                
                # Classificar (0.0 = incompatível)
                output = classifier(region, patch)
                target = torch.zeros_like(output)
                
                # Calcula perda
                loss = criterion(output, target)
                batch_loss += loss
                
                # Conta acerto
                pred = (output > 0.5).float()
                train_correct += (pred == target).sum().item()
                train_total += target.size(0)
                
                num_examples += 1
            
            # Backpropagation
            if num_examples > 0:
                batch_loss = batch_loss / num_examples
                batch_loss.backward()
                optimizer.step()
                
                train_loss += batch_loss.item()
        
        # Calcula média da loss
        train_loss = train_loss / max(1, len(train_loader))
        train_acc = train_correct / max(1, train_total)
        train_losses.append(train_loss)
        
        # === VALIDAÇÃO ===
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validando")):
                images, _, boxes, labels, img_indices = batch
                images = images.to(device)
                
                # Extrai boxes de lesão (para patches)
                lesion_mask = labels == 1
                lesion_boxes = boxes[lesion_mask]
                lesion_indices = img_indices[lesion_mask]
                
                # Extrai boxes de contexto positivo
                pos_mask = labels == 2
                pos_boxes = boxes[pos_mask]
                pos_indices = img_indices[pos_mask]
                
                # Extrai boxes de contexto negativo
                neg_mask = labels == 0
                neg_boxes = boxes[neg_mask]
                neg_indices = img_indices[neg_mask]
                
                # Pula batch se não tiver qualquer um dos tipos
                if not (lesion_mask.any() and pos_mask.any() and neg_mask.any()):
                    continue
                
                # Converte para formato xyxy
                lesion_boxes_xyxy = convert_yolo_to_xyxy(lesion_boxes)
                pos_boxes_xyxy = convert_yolo_to_xyxy(pos_boxes)
                neg_boxes_xyxy = convert_yolo_to_xyxy(neg_boxes)
                
                # Extrai patches de lesões
                lesion_patches = []
                for i, (box, img_idx) in enumerate(zip(lesion_boxes_xyxy, lesion_indices)):
                    try:
                        x1, y1, x2, y2 = [max(0, int(coord)) for coord in box]
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Extrai patch
                        patch = images[img_idx, :, y1:y2, x1:x2]
                        if patch.numel() == 0:
                            continue
                        
                        # Redimensiona
                        patch = F.interpolate(patch.unsqueeze(0), size=(128, 128), mode='bilinear')
                        lesion_patches.append(patch)
                    except Exception as e:
                        continue
                
                # Extrai regiões positivas e negativas
                pos_regions = []
                for i, (box, img_idx) in enumerate(zip(pos_boxes_xyxy, pos_indices)):
                    try:
                        x1, y1, x2, y2 = [max(0, int(coord)) for coord in box]
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Extrai região
                        region = images[img_idx, :, y1:y2, x1:x2]
                        if region.numel() == 0:
                            continue
                        
                        # Redimensiona
                        region = F.interpolate(region.unsqueeze(0), size=(128, 128), mode='bilinear')
                        pos_regions.append(region)
                    except Exception as e:
                        continue
                
                neg_regions = []
                for i, (box, img_idx) in enumerate(zip(neg_boxes_xyxy, neg_indices)):
                    try:
                        x1, y1, x2, y2 = [max(0, int(coord)) for coord in box]
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Extrai região
                        region = images[img_idx, :, y1:y2, x1:x2]
                        if region.numel() == 0:
                            continue
                        
                        # Redimensiona
                        region = F.interpolate(region.unsqueeze(0), size=(128, 128), mode='bilinear')
                        neg_regions.append(region)
                    except Exception as e:
                        continue
                
                # Pula batch se não tiver patches ou regiões suficientes
                if not lesion_patches or len(pos_regions) < 2 or len(neg_regions) < 2:
                    continue
                
                # Prepara mini-batches para avaliação
                batch_loss = 0.0
                batch_size = min(8, len(lesion_patches))
                num_examples = 0
                
                # Processa exemplos POSITIVOS (lesão + contexto positivo)
                for _ in range(batch_size):
                    # Seleciona patch e região aleatória
                    patch = random.choice(lesion_patches)
                    region = random.choice(pos_regions)
                    
                    # Classificar (1.0 = compatível)
                    output = classifier(region, patch)
                    target = torch.ones_like(output)
                    
                    # Calcula perda
                    loss = criterion(output, target)
                    batch_loss += loss
                    
                    # Conta acerto
                    pred = (output > 0.5).float()
                    val_correct += (pred == target).sum().item()
                    val_total += target.size(0)
                    
                    num_examples += 1
                
                # Processa exemplos NEGATIVOS (lesão + contexto negativo)
                for _ in range(batch_size):
                    # Seleciona patch e região aleatória
                    patch = random.choice(lesion_patches)
                    region = random.choice(neg_regions)
                    
                    # Classificar (0.0 = incompatível)
                    output = classifier(region, patch)
                    target = torch.zeros_like(output)
                    
                    # Calcula perda
                    loss = criterion(output, target)
                    batch_loss += loss
                    
                    # Conta acerto
                    pred = (output > 0.5).float()
                    val_correct += (pred == target).sum().item()
                    val_total += target.size(0)
                    
                    num_examples += 1
                
                # Acumula loss de validação
                if num_examples > 0:
                    val_loss += (batch_loss / num_examples).item()
                
        # Calcula média da loss de validação
        val_loss = val_loss / max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)
        val_losses.append(val_loss)
        
        # Atualiza scheduler
        scheduler.step(val_loss)
        
        # Salva melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(classifier.state_dict(), os.path.join(run_dir, "best_classifier.pt"))
            print(f"✓ Salvou melhor modelo com loss {val_loss:.4f}")
        
        print(f"Época {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    # Salva gráfico de loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Evolução do Treinamento')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'classifier_loss.png'))
    plt.close()
    
    # Carrega o melhor modelo
    classifier.load_state_dict(torch.load(os.path.join(run_dir, "best_classifier.pt")))
    
    return classifier
                
if __name__ == "__main__":
    main()