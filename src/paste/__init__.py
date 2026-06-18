from .evaluation.evaluator import RecommenderEvaluator
from .evaluation.metrics import (
    calculate_iou,
    calculate_context_activation,
    calculate_object_activation     
) 

from .inference.extractors import extract_recommendations_from_heatmap
from .inference.recommender import RecommendationEngine

from .losses.contrastive_loss import compute_contrastive_loss
from .losses.heatmap_loss import compute_loss, compute_contrastive_loss

from .models.attention_recommender import AttentionRecommender
from .models.backbone import PositionalEncoding

from .utils.compute_similarity import (
    decode_boxes,
    select_top_confident_preds,
    extract_multi_scale_roi_features,
    compute_similarity,
    triplet_loss_from_embeddings,
    compute_similarity_enhanced,
    extract_image_patches,
    expand_boxes,
    compute_pairwise_similarity,
    extract_roi_features,
)

from .utils.trainer import AttentionRecommenderTrainer


from .attention_dataset import AttentionDataset
from .segmentation_model import PretrainedUNet


__all__ = [
    "RecommenderEvaluator",
    "calculate_iou",
    "calculate_context_activation",
    "calculate_object_activation",
    "extract_recommendations_from_heatmap",
    "RecommendationEngine",
    "compute_contrastive_loss",
    "compute_loss",
    "compute_contrastive_loss_in_loss",
    "AttentionRecommender",
    "PositionalEncoding",
    "decode_boxes",
    "select_top_confident_preds",
    "extract_multi_scale_roi_features",
    "compute_similarity",
    "triplet_loss_from_embeddings",
    "compute_similarity_enhanced",
    "extract_image_patches",
    "expand_boxes",
    "compute_pairwise_similarity",
    "extract_roi_features",
    "AttentionRecommenderTrainer",
    "AttentionDataset",
    "PretrainedUNet"
]