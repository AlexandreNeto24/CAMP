"""Training helpers for classification models."""

from copy import deepcopy

import torch


def tensor_to_device(batch: dict, device: torch.device):
    images = batch['image'].to(device, non_blocking=True)
    labels = batch['label'].to(device, non_blocking=True)
    return images, labels


def run_epoch(model, loader, criterion, optimizer=None, device='cpu'):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images, labels = tensor_to_device(batch, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    return {
        'loss': total_loss / max(total_samples, 1),
        'accuracy': total_correct / max(total_samples, 1),
    }


def train_stage(model, train_loader, val_loader, criterion, optimizer, epochs, device, stage_name, evaluate_fn):
    history = []
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        val_metrics = evaluate_fn(model, val_loader, criterion, device=device)

        epoch_record = {
            'stage': stage_name,
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
        }
        history.append(epoch_record)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_state = deepcopy(model.state_dict())

        print(
            f"[{stage_name}] epoch {epoch + 1}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f}"
        )

    return history, best_state, best_val_loss
