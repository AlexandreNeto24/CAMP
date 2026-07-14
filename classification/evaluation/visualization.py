"""High-level visual explanation helpers for classification runs."""

from pathlib import Path

from classification.evaluation.gradcam import export_gradcam_examples
from classification.evaluation.vit_gradcam import export_vit_gradcam_examples


def export_model_explanations(model, dataset, output_dir: Path, architecture: str, max_samples: int = 6, device: str = 'cpu'):
    architecture = architecture.lower().replace('_', '-').strip()
    explanation_dir = Path(output_dir)

    if architecture in {'resnet34', 'resnet-34'}:
        return export_gradcam_examples(model, dataset, explanation_dir, max_samples=max_samples, device=device)

    if architecture in {'vit-b-16', 'vitb16', 'vit_b_16'}:
        return export_vit_gradcam_examples(model, dataset, explanation_dir, max_samples=max_samples, device=device)

    raise ValueError(f'Unsupported architecture for visual explanations: {architecture}')
