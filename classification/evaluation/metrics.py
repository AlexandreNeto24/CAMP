"""Evaluation metrics for binary classification."""

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from classification.training import tensor_to_device


def collect_predictions(model, loader, criterion, device='cpu'):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch in loader:
            images, labels = tensor_to_device(batch, device)
            logits = model(images)
            loss = criterion(logits, labels)

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())
            y_prob.extend(probabilities[:, 1].detach().cpu().tolist())

    return y_true, y_pred, y_prob, total_loss / max(total_samples, 1)


def compute_binary_metrics(y_true, y_pred, y_prob=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='binary',
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / max(tn + fp, 1)

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'confusion_matrix': matrix.tolist(),
    }

    if y_prob is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics['roc_auc'] = None

    return metrics


def evaluate_model(model, loader, criterion, device='cpu'):
    y_true, y_pred, y_prob, loss = collect_predictions(model, loader, criterion, device=device)
    metrics = compute_binary_metrics(y_true, y_pred, y_prob=y_prob)
    metrics['loss'] = float(loss)
    return metrics
