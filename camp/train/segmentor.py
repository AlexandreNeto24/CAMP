"""Segmentor training.

``PretrainedUNet`` returns logits. The training target is the context ring at
0.9, not the lesion mask. The checkpoint is chosen by the mean of per-image
Dice at threshold 0.5. While every encoder parameter has ``requires_grad``
False, the encoder runs under ``torch.no_grad`` and the decoder receives
detached features. After any encoder parameter is unfrozen, the forward is
the normal one and features are not detached.

The OneCycle scheduler is stepped after every optimizer step and once more at
the end of each freeze-phase epoch. Batch-norm is put in eval when fine-tuning
starts, then ``model.train()`` runs and a partially unfrozen encoder is set
back to train.
"""

import logging
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from camp.dataset import segmentor_collate

logger = logging.getLogger(__name__)


class TverskyFocalLoss(nn.Module):
    """Tversky focal loss as one ratio over the flattened batch."""

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        true_positive = (inputs * targets).sum()
        false_positive = ((1 - targets) * inputs).sum()
        false_negative = (targets * (1 - inputs)).sum()

        tversky = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        tversky_loss = 1 - tversky
        return torch.pow(tversky_loss, self.gamma)


class SegmentorTrainer:
    """Two-phase segmentor fit: frozen encoder, then gradual unfreeze."""

    def __init__(self, model):
        self.model = model

    def fit(
        self,
        train_dataset,
        val_dataset,
        run_dir,
        epochs=100,
        batch_size=4,
        lr=1e-3,
        device="cuda",
        accumulation_steps=4,
        freeze_epochs=20,
        full_finetune_lr=1e-5,
        early_stopping_patience=100,
        verbose=True,
    ):
        """Train and return the path of ``best_segmentor.pth``."""
        device = self._select_device(device)
        self.model.to(device)
        train_loader, val_loader = self._make_loaders(
            train_dataset, val_dataset, batch_size
        )
        criterion = TverskyFocalLoss(alpha=0.7, beta=0.3, gamma=0.75)

        save_path = os.path.join(run_dir, "best_segmentor.pth")
        train_losses = []
        val_losses = []
        train_dice_history = []
        val_dice_history = []
        best_val_dice = -float("inf")
        epochs_no_improve = 0

        initial_unfreeze_groups = 1
        gradual_unfreeze_every = 2
        encoder_warmup_epochs = 15
        encoder_lr_factor = 0.05
        encoder_children = (
            list(self.model.model.encoder.children())
            if hasattr(self.model.model, "encoder")
            else []
        )
        current_unfrozen_children = 0

        writer = SummaryWriter(log_dir=run_dir) if SummaryWriter is not None else None
        if writer is None and verbose:
            logger.info("TensorBoard is not available; skipping SummaryWriter.")
        if verbose:
            logger.info(
                "Training segmentor for %s epochs (freeze_epochs=%s). Checkpoint: %s",
                epochs,
                freeze_epochs,
                save_path,
            )

        optimizer = None
        scheduler = None
        scheduler_type = None
        optimizer_encoder = None
        scheduler_encoder = None
        encoder_base_lr = None

        for epoch in range(epochs):
            if epoch == 0:
                if verbose:
                    logger.info("Phase 1: training the decoder (encoder frozen).")
                optimizer, scheduler, scheduler_type = self._start_decoder_phase(
                    lr, train_loader, freeze_epochs, epochs, verbose
                )
                optimizer_encoder = None
                scheduler_encoder = None

            if epoch == freeze_epochs:
                if verbose:
                    logger.info("Phase 2: fine-tuning (encoder unfrozen gradually).")
                (
                    optimizer_encoder,
                    scheduler_encoder,
                    encoder_base_lr,
                    current_unfrozen_children,
                    scheduler,
                    scheduler_type,
                ) = self._start_finetune_phase(
                    epoch=epoch,
                    epochs=epochs,
                    save_path=save_path,
                    verbose=verbose,
                    encoder_children=encoder_children,
                    initial_unfreeze_groups=initial_unfreeze_groups,
                    full_finetune_lr=full_finetune_lr,
                    encoder_lr_factor=encoder_lr_factor,
                    optimizer=optimizer,
                    scheduler_type=scheduler_type,
                )

            if epoch > freeze_epochs and encoder_children:
                current_unfrozen_children = self._unfreeze_next_group(
                    epoch=epoch,
                    freeze_epochs=freeze_epochs,
                    gradual_unfreeze_every=gradual_unfreeze_every,
                    encoder_children=encoder_children,
                    current_unfrozen_children=current_unfrozen_children,
                    encoder_base_lr=encoder_base_lr,
                    full_finetune_lr=full_finetune_lr,
                    encoder_lr_factor=encoder_lr_factor,
                    optimizer=optimizer,
                    optimizer_encoder=optimizer_encoder,
                    verbose=verbose,
                )

            encoder_frozen = self._encoder_fully_frozen()
            self._set_epoch_mode(encoder_frozen)
            epoch_loss, train_dice_parts = self._train_one_epoch(
                train_loader,
                criterion,
                optimizer,
                optimizer_encoder,
                scheduler,
                scheduler_type,
                accumulation_steps,
                encoder_frozen,
                device,
                epoch,
                epochs,
            )

            self.model.eval()
            val_loss, val_iou_parts, val_dice_parts = self._validate(
                val_loader, criterion, device, epoch, epochs
            )

            avg_train_loss = epoch_loss / max(1, len(train_loader))
            avg_val_loss = val_loss / max(1, len(val_loader))
            avg_train_dice = _mean_metric(train_dice_parts)
            avg_val_iou = _mean_metric(val_iou_parts)
            avg_val_dice = _mean_metric(val_dice_parts)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_dice_history.append(avg_train_dice)
            val_dice_history.append(avg_val_dice)

            if writer is not None:
                writer.add_scalar("loss/train", avg_train_loss, epoch)
                writer.add_scalar("loss/val", avg_val_loss, epoch)
                writer.add_scalar("metrics/train_dice", avg_train_dice, epoch)
                writer.add_scalar("metrics/val_dice", avg_val_dice, epoch)
                writer.add_scalar("metrics/val_iou", avg_val_iou, epoch)

            if verbose:
                logger.info(
                    "Epoch %s: Train %.4f | Val %.4f | Train Dice %.4f | "
                    "Val Dice %.4f | IoU %.4f",
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    avg_train_dice,
                    avg_val_dice,
                    avg_val_iou,
                )

            self._step_schedulers(
                epoch=epoch,
                freeze_epochs=freeze_epochs,
                encoder_warmup_epochs=encoder_warmup_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                optimizer_encoder=optimizer_encoder,
                scheduler_encoder=scheduler_encoder,
                encoder_base_lr=encoder_base_lr,
                avg_val_dice=avg_val_dice,
                verbose=verbose,
            )

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                self._save_checkpoint(
                    save_path,
                    epoch,
                    optimizer,
                    optimizer_encoder,
                    best_val_dice,
                )
                epochs_no_improve = 0
                self._write_history(
                    run_dir,
                    train_losses,
                    val_losses,
                    train_dice_history,
                    val_dice_history,
                )
                if verbose:
                    logger.info(
                        "Epoch %s: val_dice improved to %.4f. Saved %s",
                        epoch + 1,
                        best_val_dice,
                        save_path,
                    )
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                if verbose:
                    logger.info(
                        "No val_dice improvement for %s epochs. Stopping at epoch %s.",
                        early_stopping_patience,
                        epoch + 1,
                    )
                break

        if train_losses:
            self._write_history(
                run_dir,
                train_losses,
                val_losses,
                train_dice_history,
                val_dice_history,
            )
        if writer is not None:
            writer.close()
        return save_path

    def _select_device(self, device):
        if isinstance(device, str) and device.isdigit():
            device = f"cuda:{device}"
        if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA device %s is not available. Using CPU.", device)
            device = "cpu"
        return torch.device(device)

    def _make_loaders(self, train_dataset, val_dataset, batch_size):
        train_workers = 0 if os.name == "nt" else 4
        val_workers = 0 if os.name == "nt" else 2
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=train_workers,
            pin_memory=True,
            collate_fn=segmentor_collate,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=True,
            collate_fn=segmentor_collate,
        )
        return train_loader, val_loader

    def _start_decoder_phase(self, lr, train_loader, freeze_epochs, epochs, verbose):
        if hasattr(self.model.model, "encoder"):
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False
        decoder_params = [
            param
            for name, param in self.model.named_parameters()
            if param.requires_grad and "encoder" not in name
        ]
        optimizer = torch.optim.AdamW(decoder_params, lr=lr, weight_decay=1e-5)
        try:
            steps_per_epoch = max(1, len(train_loader))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                steps_per_epoch=steps_per_epoch,
                epochs=max(1, min(freeze_epochs, epochs)),
                pct_start=0.3,
                anneal_strategy="cos",
                div_factor=10.0,
                final_div_factor=1e4,
            )
            scheduler_type = "onecycle"
            if verbose:
                logger.info(
                    "OneCycleLR configured (decoder) max_lr=%.1e, "
                    "steps_per_epoch=%s, epochs=%s",
                    lr,
                    steps_per_epoch,
                    min(freeze_epochs, epochs),
                )
        except Exception:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=7, factor=0.5
            )
            scheduler_type = "plateau"
        return optimizer, scheduler, scheduler_type

    def _start_finetune_phase(
        self,
        epoch,
        epochs,
        save_path,
        verbose,
        encoder_children,
        initial_unfreeze_groups,
        full_finetune_lr,
        encoder_lr_factor,
        optimizer,
        scheduler_type,
    ):
        if os.path.exists(save_path):
            try:
                checkpoint = torch.load(save_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                if verbose:
                    logger.info("Loaded best weights from %s", save_path)
            except Exception as exc:
                logger.warning("Could not load checkpoint %s: %s", save_path, exc)

        if encoder_children:
            n_children = len(encoder_children)
            groups_to_unfreeze = min(initial_unfreeze_groups, n_children)
            start_idx = max(0, n_children - groups_to_unfreeze)
            for idx in range(start_idx, n_children):
                for param in encoder_children[idx].parameters():
                    param.requires_grad = True
            current_unfrozen_children = groups_to_unfreeze
            if verbose:
                logger.info(
                    "Gradual unfreeze: initially unfroze %s of %s encoder groups.",
                    current_unfrozen_children,
                    n_children,
                )
        else:
            for name, param in self.model.named_parameters():
                if "encoder" in name:
                    param.requires_grad = True
            current_unfrozen_children = len(encoder_children)

        for module in self.model.model.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

        encoder_params = [
            param
            for name, param in self.model.named_parameters()
            if "encoder" in name and param.requires_grad
        ]
        encoder_base_lr = max(full_finetune_lr * encoder_lr_factor, 5e-6)
        if encoder_params:
            optimizer_encoder = torch.optim.AdamW(
                encoder_params, lr=encoder_base_lr, weight_decay=1e-5
            )
            scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_encoder, T_max=max(1, epochs - epoch - 1)
            )
            if verbose:
                logger.info("Encoder optimizer created with lr=%.2e", encoder_base_lr)
        else:
            optimizer_encoder = None
            scheduler_encoder = None

        if scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs - epoch - 1)
            )
            scheduler_type = "cosine_post"
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=10, factor=0.5
            )
        return (
            optimizer_encoder,
            scheduler_encoder,
            encoder_base_lr,
            current_unfrozen_children,
            scheduler,
            scheduler_type,
        )

    def _unfreeze_next_group(
        self,
        epoch,
        freeze_epochs,
        gradual_unfreeze_every,
        encoder_children,
        current_unfrozen_children,
        encoder_base_lr,
        full_finetune_lr,
        encoder_lr_factor,
        optimizer,
        optimizer_encoder,
        verbose,
    ):
        epochs_since_unfreeze = epoch - freeze_epochs
        if epochs_since_unfreeze % gradual_unfreeze_every != 0:
            return current_unfrozen_children
        n_children = len(encoder_children)
        if current_unfrozen_children >= n_children:
            return current_unfrozen_children

        next_idx = n_children - 1 - current_unfrozen_children
        newly_unfrozen = []
        for param in encoder_children[next_idx].parameters():
            if not param.requires_grad:
                param.requires_grad = True
                newly_unfrozen.append(param)
        if not newly_unfrozen:
            return current_unfrozen_children

        add_lr = (encoder_base_lr or (full_finetune_lr * encoder_lr_factor)) * 0.2
        param_group = {
            "params": newly_unfrozen,
            "lr": add_lr,
            "weight_decay": 1e-5,
            "name": f"encoder_g{current_unfrozen_children + 1}",
        }
        if optimizer_encoder is not None:
            optimizer_encoder.add_param_group(param_group)
        else:
            optimizer.add_param_group(param_group)
        current_unfrozen_children += 1
        if verbose:
            logger.info(
                "Gradual unfreeze: encoder child %s added with lr=%.2e "
                "(unfrozen groups=%s).",
                next_idx,
                add_lr,
                current_unfrozen_children,
            )
        return current_unfrozen_children

    def _encoder_fully_frozen(self):
        if not hasattr(self.model.model, "encoder"):
            return True
        encoder_params = list(self.model.model.encoder.parameters())
        if not encoder_params:
            return True
        return all(not param.requires_grad for param in encoder_params)

    def _set_epoch_mode(self, encoder_frozen):
        self.model.train()
        if not hasattr(self.model.model, "encoder"):
            return
        if encoder_frozen:
            self.model.model.encoder.eval()
        else:
            self.model.model.encoder.train()

    def _logits(self, images, encoder_frozen):
        if not encoder_frozen:
            return self.model(images)
        net = self.model.model
        net.check_input_shape(images)
        with torch.no_grad():
            encoded = net.encoder(images)
        features = [feature.detach() for feature in encoded]
        decoded = net.decoder(features)
        masks = net.segmentation_head(decoded)
        if net.classification_head is not None:
            labels = net.classification_head(features[-1])
            return masks, labels
        return masks

    def _train_one_epoch(
        self,
        train_loader,
        criterion,
        optimizer,
        optimizer_encoder,
        scheduler,
        scheduler_type,
        accumulation_steps,
        encoder_frozen,
        device,
        epoch,
        epochs,
    ):
        epoch_loss = 0.0
        dice_parts = []
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")):
            images = batch["images"].to(device)
            true_masks = batch["segmentor_target_mask"].to(device)
            pred_masks = self._logits(images, encoder_frozen)
            loss = criterion(pred_masks, true_masks)

            with torch.no_grad():
                dice, _iou = _per_image_dice_iou(pred_masks, true_masks)
                if dice.numel() > 0:
                    dice_parts.append(dice)

            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda param: param.requires_grad, self.model.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                if optimizer_encoder is not None:
                    try:
                        optimizer_encoder.step()
                    except Exception:
                        pass
                if scheduler is not None and scheduler_type == "onecycle":
                    try:
                        scheduler.step()
                    except Exception:
                        pass
                optimizer.zero_grad()
                if optimizer_encoder is not None:
                    try:
                        optimizer_encoder.zero_grad()
                    except Exception:
                        pass

            epoch_loss += loss.item() * accumulation_steps
        return epoch_loss, dice_parts

    def _validate(self, val_loader, criterion, device, epoch, epochs):
        val_loss = 0.0
        iou_parts = []
        dice_parts = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                images = batch["images"].to(device)
                true_masks = batch["segmentor_target_mask"].to(device)
                pred_masks = self.model(images)
                val_loss += criterion(pred_masks, true_masks).item()
                dice, iou = _per_image_dice_iou(pred_masks, true_masks)
                if dice.numel() > 0:
                    dice_parts.append(dice)
                if iou.numel() > 0:
                    iou_parts.append(iou)
        return val_loss, iou_parts, dice_parts

    def _step_schedulers(
        self,
        epoch,
        freeze_epochs,
        encoder_warmup_epochs,
        optimizer,
        scheduler,
        optimizer_encoder,
        scheduler_encoder,
        encoder_base_lr,
        avg_val_dice,
        verbose,
    ):
        prev_lr = optimizer.param_groups[0]["lr"]
        try:
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_dice)
                else:
                    # OneCycle during the freeze phase is stepped again here,
                    # in addition to the per-batch step.
                    try:
                        scheduler.step()
                    except Exception:
                        pass
        except Exception:
            pass

        if optimizer_encoder is not None:
            if epoch >= freeze_epochs and epoch < freeze_epochs + encoder_warmup_epochs:
                warm_step = (epoch - freeze_epochs + 1) / max(1, encoder_warmup_epochs)
                warm_lr = encoder_base_lr * (0.1 + 0.9 * warm_step)
                for group in optimizer_encoder.param_groups:
                    group["lr"] = warm_lr
                if verbose:
                    logger.info("Encoder warmup lr -> %.2e", warm_lr)
            else:
                try:
                    scheduler_encoder.step()
                except Exception:
                    pass

        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr and verbose:
            logger.info("Learning rate reduced: %.2e -> %.2e", prev_lr, new_lr)

    def _save_checkpoint(self, save_path, epoch, optimizer, optimizer_encoder, best_val_dice):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": best_val_dice,
        }
        if optimizer_encoder is not None:
            checkpoint["optimizer_encoder_state_dict"] = optimizer_encoder.state_dict()
        torch.save(checkpoint, save_path)

    def _write_history(
        self, run_dir, train_losses, val_losses, train_dice_history, val_dice_history
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(train_losses, label="Train Loss", color="C0")
        axes[0].plot(val_losses, label="Val Loss", color="C1")
        axes[0].set_title("Segmentor Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.2)
        axes[1].plot(train_dice_history, label="Train Dice", color="C2")
        axes[1].plot(val_dice_history, label="Val Dice", color="C3")
        axes[1].set_title("Dice")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dice")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].legend()
        axes[1].grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "segmentor_metrics_subplot.png"))
        plt.close(fig)
        try:
            np.save(os.path.join(run_dir, "train_losses.npy"), np.array(train_losses))
            np.save(os.path.join(run_dir, "val_losses.npy"), np.array(val_losses))
            np.save(
                os.path.join(run_dir, "train_dice_history.npy"),
                np.array(train_dice_history),
            )
            np.save(
                os.path.join(run_dir, "val_dice_history.npy"),
                np.array(val_dice_history),
            )
        except Exception:
            logger.warning("Could not write segmentor history arrays in %s", run_dir)


def _per_image_dice_iou(pred_logits, true_masks):
    """Per-image Dice and IoU at 0.5. Empty images are omitted from each list."""
    probabilities = torch.sigmoid(pred_logits)
    pred_bin = (probabilities > 0.5).float()
    true_bin = (true_masks > 0.5).float()
    reduce_dims = tuple(range(1, pred_bin.ndim))
    intersection = (pred_bin * true_bin).sum(dim=reduce_dims)
    total = pred_bin.sum(dim=reduce_dims) + true_bin.sum(dim=reduce_dims)
    union = ((pred_bin + true_bin) > 0).float().sum(dim=reduce_dims)
    dice = (2 * intersection) / (total + 1e-6)
    iou = intersection / (union + 1e-6)
    return dice[total > 0], iou[union > 0]


def _mean_metric(parts):
    if not parts:
        return 0.0
    values = torch.cat([part.reshape(-1) for part in parts])
    if values.numel() == 0:
        return 0.0
    return float(values.double().mean().item())
