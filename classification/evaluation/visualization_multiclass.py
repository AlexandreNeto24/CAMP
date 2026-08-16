"""High-level visual explanation helpers for multiclass classification runs."""

from pathlib import Path
from typing import Optional

from classification.evaluation.gradcam_multiclass import export_gradcam_examples
from classification.evaluation.vit_gradcam_multiclass import export_vit_gradcam_examples


def export_model_explanations(
    model,
    dataset,
    output_dir: Path,
    architecture: str,
    samples_per_class: Optional[dict] = None,
    device: str = 'cpu',
    return_per_class_metrics: bool = False,
):
    if samples_per_class is None:
        samples_per_class = {0: 3, 1: 3}

    architecture = architecture.lower().replace('_', '-').strip()
    explanation_dir = Path(output_dir)

    if architecture in {'resnet34', 'resnet-34'}:
        return export_gradcam_examples(
            model,
            dataset,
            explanation_dir,
            samples_per_class=samples_per_class,
            device=device,
            return_per_class_metrics=return_per_class_metrics,
        )

    if architecture in {'vit-b-16', 'vitb16', 'vit_b_16'}:
        return export_vit_gradcam_examples(
            model,
            dataset,
            explanation_dir,
            samples_per_class=samples_per_class,
            device=device,
            return_per_class_metrics=return_per_class_metrics,
        )

    raise ValueError(f'Unsupported architecture for visual explanations: {architecture}')