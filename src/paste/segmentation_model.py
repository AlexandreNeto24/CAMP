import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
import segmentation_models_pytorch as smp 
import matplotlib.pyplot as plt
import numpy as np

# --- REMOVER AS CLASSES DE CONSTRUÇÃO DO U-NET (DoubleConv, Down, Up, OutConv) ---
# --- A biblioteca smp trata disto internamente. ---

# --- Modelo U-Net Completo (Substituído) ---
class PretrainedUNet(nn.Module):
    """
    Wrapper para um U-Net de segmentation-models-pytorch para compatibilidade
    com o pipeline existente.
    """
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, decoder_dropout=0.2):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_dropout=decoder_dropout
        )
        # Determina a dimensão das features do encoder
        # Isto é um valor conhecido para cada arquitetura.
        # Ex: resnet34 -> 512, resnet50 -> 2048, efficientnet-b0 -> 1280, efficientnet-b3 -> 1536
        self.feature_dim = self.model.encoder.out_channels[-1]

    def forward(self, x):
        return self.model(x)

    def get_encoder(self):
        """Retorna a parte do encoder para ser usada como feature extractor."""
        return self.model.encoder

# --- Lógica de Treino para o Segmentador (Permanece igual) ---
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        return BCE + dice_loss

class TverskyFocalLoss(nn.Module):
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

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        tversky_loss = 1 - tversky

        focal_loss = torch.pow(tversky_loss, self.gamma)
        return focal_loss
    
class LovaszLoss(nn.Module):
    """
    Lovász hinge loss para segmentação binária.
    Maximiza diretamente o IoU.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        # logits: [B, 1, H, W] (antes do sigmoid)
        # targets: [B, 1, H, W] (binário)
        logits = logits.squeeze(1)
        targets = targets.squeeze(1)
        losses = []
        for logit, target in zip(logits, targets):
            loss = self.lovasz_hinge_flat(logit, target)
            losses.append(loss)
        return torch.mean(torch.stack(losses))

    @staticmethod
    def lovasz_hinge_flat(logits, labels):
        """
        Lovász hinge loss para vetores planos.
        """
        labels = labels.view(-1)
        logits = logits.view(-1)
        signs = 2. * labels.float() - 1.
        errors = 1. - logits * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = LovaszLoss.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    @staticmethod
    def lovasz_grad(gt_sorted):
        """
        Calcula gradientes para a Lovász extension.
        """
        p = len(gt_sorted)
        gts = gt_sorted.float()
        intersection = gts.sum() - gts.cumsum(0)
        union = gts.sum() + (1 - gts).cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

def train_segmentor(
    model, train_dataset, val_dataset, run_dir, epochs=100, batch_size=4, lr=1e-3, device='cuda',
    accumulation_steps=4, freeze_epochs=20, full_finetune_lr=1e-5, early_stopping_patience=100, verbose=True
    ):
    """
    Função de treino com estratégia de fine-tuning em duas fases.
    Agora com gradual unfreeze do encoder, BN frozen temporariamente, LR discriminativo e warmup.
    """
    from src.paste.attention_dataset import attention_collate
    if isinstance(device, str) and device.isdigit(): device = f'cuda:{device}'
    if isinstance(device, str) and device.startswith('cuda') and not torch.cuda.is_available():
        if verbose: print(f"AVISO: Dispositivo CUDA '{device}' solicitado mas não disponível. Usando CPU.")
        device = 'cpu'
    device = torch.device(device)
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=attention_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=attention_collate)

    # criterion
    criterion = DiceBCELoss()

    save_path = os.path.join(run_dir, 'best_segmentor.pth')
    train_losses = []
    val_losses = []
    train_dice_history = []
    val_dice_history = []
    best_val_dice = -float('inf')
    epochs_no_improve = 0

    # --- parâmetros para fine‑tune suave / gradual unfreeze ---
    # quantos grupos (children) do encoder descongelar inicialmente e com que frequência
    initial_unfreeze_groups = 1
    gradual_unfreeze_every = 2       # a cada N épocas após freeze, desbloqueia mais 1 grupo
    encoder_warmup_epochs = 10       # warmup mais longo para encoder
    encoder_lr_factor = 0.01         # encoder lr = full_finetune_lr * encoder_lr_factor
    encoder_children = list(model.model.encoder.children()) if hasattr(model.model, 'encoder') else []
    current_unfrozen_children = 0    # controlo de quantos grupos do encoder estão descongelados
    # ----------------------------------------------------------------

    # TensorBoard writer (opcional)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=run_dir)
    except Exception:
        writer = None
        if verbose:
            print("TensorBoard não disponível - sem SummaryWriter.")

    if verbose:
        print(f"Iniciando treino por {epochs} épocas (freeze_epochs={freeze_epochs}). Checkpoint: {save_path}")

    # place-holder optimizer/scheduler variables (serão criados na fase 1)
    optimizer = None
    scheduler = None
    encoder_base_lr = None

    for epoch in range(epochs):
        # --- Fase: freeze / unfreeze logic ---
        if epoch == 0:
            if verbose: print("\n--- Fase 1: Treinando apenas o decoder (encoder congelado) ---")
            # congela todo o encoder
            if hasattr(model.model, 'encoder'):
                for p in model.model.encoder.parameters():
                    p.requires_grad = False
            # optimizer apenas com parâmetros que requerem grad (decoder/heads)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
            # scheduler monitora val_dice (mode='max')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=7, factor=0.2)

        if epoch == freeze_epochs:
            if verbose: print("\n--- Fase 2: Fine-tuning (encoder descongelado gradualmente) ---")
            # tenta carregar melhor checkpoint antes do fine-tune
            if os.path.exists(save_path):
                try:
                    ckpt = torch.load(save_path, map_location='cpu')
                    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                        model.load_state_dict(ckpt['model_state_dict'])
                    else:
                        model.load_state_dict(ckpt)
                    if verbose: print(f"Carregados melhores pesos de {save_path}")
                except Exception:
                    if verbose: print("Falha ao carregar checkpoint; continuando com pesos atuais.")

            # Inicial: descongela apenas os últimos `initial_unfreeze_groups` filhos do encoder
            if encoder_children:
                n_children = len(encoder_children)
                groups_to_unfreeze = min(initial_unfreeze_groups, n_children)
                # define quais índices serão inicialmente descongelados: últimos grupos
                start_idx = max(0, n_children - groups_to_unfreeze)
                for idx in range(start_idx, n_children):
                    module = encoder_children[idx]
                    for p in module.parameters():
                        p.requires_grad = True
                current_unfrozen_children = groups_to_unfreeze
                if verbose:
                    print(f"Gradual unfreeze: desbloqueados inicialmente {current_unfrozen_children} encoder groups (dos {n_children}).")
            else:
                # se encoder_children vazio, habilita todos os params de encoder
                for n, p in model.named_parameters():
                    if 'encoder' in n:
                        p.requires_grad = True
                current_unfrozen_children = len(encoder_children)

            # freeze BatchNorm running stats e desativa grad nas gammas/betas dos BNs
            for m in model.model.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for bn_p in m.parameters():
                        bn_p.requires_grad = False

            # adiciona os parâmetros actualmente descongelados ao optimizer (param_group)
            encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad]
            if encoder_params:
                encoder_base_lr = full_finetune_lr * encoder_lr_factor
                try:
                    optimizer.add_param_group({'params': encoder_params, 'lr': encoder_base_lr, 'weight_decay': 1e-5, 'name': 'encoder'})
                    if verbose:
                        print(f"Encoder params inicialmente adicionados ao optimizer com lr={encoder_base_lr:.2e} (preservando estado do decoder).")
                except Exception:
                    # fallback: recria optimizer com param groups
                    decoder_params = [p for n, p in model.named_parameters() if 'encoder' not in n and p.requires_grad]
                    optimizer = torch.optim.AdamW([
                        {'params': decoder_params, 'lr': full_finetune_lr, 'weight_decay': 1e-5},
                        {'params': encoder_params, 'lr': encoder_base_lr, 'weight_decay': 1e-5, 'name': 'encoder'}
                    ])
                    if verbose:
                        print("Optimizer recriado com param_groups (decoder + encoder).")
            else:
                # sem encoder params encontrados
                encoder_base_lr = full_finetune_lr

            # scheduler (continua a monitorizar val_dice)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.3)

        # Gradual unfreeze adicionais: a cada `gradual_unfreeze_every` épocas após freeze, descongela mais 1 grupo
        if epoch > freeze_epochs and encoder_children:
            epochs_since_unfreeze = epoch - freeze_epochs
            if epochs_since_unfreeze % gradual_unfreeze_every == 0:
                n_children = len(encoder_children)
                # se ainda houver grupos congelados, descongela o próximo (da direita para a esquerda)
                if current_unfrozen_children < n_children:
                    next_idx = n_children - 1 - current_unfrozen_children
                    module = encoder_children[next_idx]
                    newly_unfrozen = []
                    for p in module.parameters():
                        if not p.requires_grad:
                            p.requires_grad = True
                            newly_unfrozen.append(p)
                    if newly_unfrozen:
                        # adiciona esses params ao optimizer com lr = encoder_base_lr * small factor (inicia mais baixo)
                        add_lr = (encoder_base_lr or (full_finetune_lr * encoder_lr_factor)) * 0.2
                        optimizer.add_param_group({'params': newly_unfrozen, 'lr': add_lr, 'weight_decay': 1e-5, 'name': f'encoder_g{current_unfrozen_children+1}'})
                        current_unfrozen_children += 1
                        if verbose:
                            print(f"Gradual unfreeze: desbloqueado encoder child index {next_idx}; param_group adicionado com lr={add_lr:.2e} (total unfrozen groups={current_unfrozen_children}).")

        # train mode, but keep encoder eval() if still all frozen
        model.train()
        encoder_params_list = list(model.model.encoder.parameters()) if hasattr(model.model, 'encoder') else []
        encoder_frozen = all(not p.requires_grad for p in encoder_params_list) if encoder_params_list else True
        if encoder_frozen:
            # mantém encoder em eval para estabilidade se totalmente congelado
            model.model.encoder.eval()
        else:
            # se não totalmente congelado, mantém encoder em train (BNs foram colocados em eval manualmente)
            model.model.encoder.train()

        epoch_loss = 0.0
        train_epoch_dices = []

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            images = batch['images'].to(device)
            true_masks = batch['segmentor_target_mask'].to(device)

            pred_masks = model(images)
            loss = criterion(pred_masks, true_masks)

            # compute train-dice for this batch (cheap, same threshold as val)
            with torch.no_grad():
                preds_sig = torch.sigmoid(pred_masks).cpu()
                trues = true_masks.cpu()
                for p, t in zip(preds_sig, trues):
                    pbin = (p > 0.5).float()
                    tbin = (t > 0.5).float()
                    inter = (pbin * tbin).sum()
                    tot = pbin.sum() + tbin.sum()
                    if tot > 0:
                        train_epoch_dices.append((2 * inter / (tot + 1e-6)).item())

            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                # grad clipping (aplica a todos os params)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps

        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_ious = []
        val_dices = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = batch['images'].to(device)
                true_masks = batch['segmentor_target_mask'].to(device)
                pred_masks = model(images)
                val_loss += criterion(pred_masks, true_masks).item()
                preds_sig = torch.sigmoid(pred_masks).cpu()
                trues = true_masks.cpu()
                for p, t in zip(preds_sig, trues):
                    pbin = (p > 0.5).float()
                    tbin = (t > 0.5).float()
                    inter = (pbin * tbin).sum()
                    union = ((pbin + tbin) > 0).float().sum()
                    if union > 0:
                        val_ious.append((inter / (union + 1e-6)).item())
                    tot = pbin.sum() + tbin.sum()
                    if tot > 0:
                        val_dices.append((2 * inter / (tot + 1e-6)).item())

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_train_dice = float(np.mean(train_epoch_dices)) if train_epoch_dices else 0.0
        avg_val_iou = float(np.mean(val_ious)) if val_ious else 0.0
        avg_val_dice = float(np.mean(val_dices)) if val_dices else 0.0

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_dice_history.append(avg_train_dice)
        val_dice_history.append(avg_val_dice)

        # tensorboard logs
        if writer is not None:
            writer.add_scalar('loss/train', avg_train_loss, epoch)
            writer.add_scalar('loss/val', avg_val_loss, epoch)
            writer.add_scalar('metrics/train_dice', avg_train_dice, epoch)
            writer.add_scalar('metrics/val_dice', avg_val_dice, epoch)
            writer.add_scalar('metrics/val_iou', avg_val_iou, epoch)

        if verbose:
            print(f"Epoch {epoch+1}: Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | Train Dice {avg_train_dice:.4f} | Val Dice {avg_val_dice:.4f} | IoU {avg_val_iou:.4f}")

        # scheduler step (ReduceLROnPlateau monitora val_dice)
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_dice)

        # warmup linear para encoder param_groups nas primeiras encoder_warmup_epochs após freeze
        if encoder_base_lr is not None and epoch >= freeze_epochs and epoch < freeze_epochs + encoder_warmup_epochs:
            warm_step = (epoch - freeze_epochs + 1) / max(1, encoder_warmup_epochs)
            target_lr = encoder_base_lr * (0.1 + 0.9 * warm_step)  # começa em 0.1*base -> base
            for g in optimizer.param_groups:
                if g.get('name', '').startswith('encoder'):
                    old = g['lr']
                    g['lr'] = target_lr
                    if verbose and abs(old - target_lr) > 1e-12:
                        print(f"[Warmup] encoder param_group '{g.get('name')}' lr: {old:.2e} -> {g['lr']:.2e}")

        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr and verbose:
            print(f"[LR Scheduler] LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        # checkpoint by Val Dice
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice
            }, save_path)
            epochs_no_improve = 0
            if verbose:
                print(f"[ModelCheckpoint] Epoch {epoch+1}: val_dice improved to {best_val_dice:.4f}. Saved -> {save_path}")
        else:
            epochs_no_improve += 1

        # early stopping by Val Dice
        if epochs_no_improve >= early_stopping_patience:
            if verbose:
                print(f"[EarlyStopping] No improvement in val_dice for {early_stopping_patience} epochs. Stopping at epoch {epoch+1}.")
            break

        # SAVE: subplot com Loss (train/val) e Dice (train/val)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        # Loss subplot
        axes[0].plot(train_losses, label='Train Loss', color='C0')
        axes[0].plot(val_losses, label='Val Loss', color='C1')
        axes[0].set_title("Segmentor Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.2)
        # Dice subplot (train + val)
        axes[1].plot(train_dice_history, label='Train Dice', color='C2')
        axes[1].plot(val_dice_history, label='Val Dice', color='C3')
        axes[1].set_title("Dice")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dice")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].legend()
        axes[1].grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'segmentor_metrics_subplot.png'))
        plt.close(fig)

        # opcional: salvar históricos em .npy para análise posterior
        try:
            np.save(os.path.join(run_dir, 'train_losses.npy'), np.array(train_losses))
            np.save(os.path.join(run_dir, 'val_losses.npy'), np.array(val_losses))
            np.save(os.path.join(run_dir, 'train_dice_history.npy'), np.array(train_dice_history))
            np.save(os.path.join(run_dir, 'val_dice_history.npy'), np.array(val_dice_history))
        except Exception:
            pass

    if writer is not None:
        writer.close()

    return save_path