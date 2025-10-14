import os
import sys
import json
import torch
from pathlib import Path

# Configuração de caminhos
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.paste.attention_dataset import AttentionDataset
from src.paste.attention_recommender import AttentionRecommender
from src.paste.segmentation_model import PretrainedUNet

def main():
    # Caminhos dos pesos
    segmentor_path = r"D:\Projects\2023_24\yolo_CIM\YOLO CIM\models\paste\weights\20250825_191022_attention_unet\best_segmentor.pth"
    attention_path = r"D:\Projects\2023_24\yolo_CIM\YOLO CIM\models\paste\weights\20250825_191022_attention_unet\best_model.pt"
    data_path = r"D:\Projects\2023_24\yolo_CIM\YOLO CIM\data\metaplasia"
    results_dir = r"D:\Projects\2023_24\yolo_CIM\YOLO CIM\models\paste\weights\20250825_191022_attention_unet\test_results"
    os.makedirs(results_dir, exist_ok=True)

    # Carrega datasets
    val_dataset = AttentionDataset(
        root_path=data_path,
        image_size=(512, 512),
        patch_size=(64, 64),
        hyp=None,
        apply_aug=False,
        split_mode='ratio',
        split='val',
        split_ratio=0.2
    )

    # Carrega backbone e modelo
    backbone = PretrainedUNet(encoder_name='resnet34', encoder_weights=None)
    backbone.load_state_dict(torch.load(segmentor_path))
    model = AttentionRecommender(
        backbone_model=backbone,
        embedding_dim=2048,
        num_heads=32,
        dropout_rate=0.3
    )
    model.load_model(attention_path)
    model.eval()

    # Avaliação normal
    from train_attention_recommender import evaluate_model
    print("Avaliando modelo no conjunto de validação...")
    eval_dir = os.path.join(results_dir, "evaluation")
    metrics = evaluate_model(model, val_dataset, num_samples=20, save_dir=eval_dir)
    print(json.dumps(metrics, indent=2))

    # Avaliação exógena (patches de outras imagens)
    print("\nAvaliando com patches exógenos...")
    exogenous_eval_dir = os.path.join(results_dir, "exogenous_evaluation")
    exogenous_metrics = model.evaluate_exogenous_patches(
        val_dataset,
        num_samples=15,
        save_dir=exogenous_eval_dir
    )
    print(json.dumps(exogenous_metrics, indent=2))

    # Avaliação em imagens normais (colagem de instância)
    print("\nAvaliando colagem de patch em imagens normais...")
    normal_images_dir = os.path.join(data_path, "normal")
    normal_patch_eval_dir = os.path.join(results_dir, "normal_patch_evaluation")
    model.evaluate_patch_on_normal(
        val_dataset,
        normal_images_dir,
        normal_patch_eval_dir,
        num_samples=10
    )

    print("\nTestes concluídos. Resultados em:", results_dir)

if __name__ == "__main__":
    main()