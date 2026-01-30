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
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        
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

    def load_adapted_state_dict(self, checkpoint_path, strict=False):
        """
        Carrega pesos de 3 canais num modelo de 5 canais (CoordConv).
        Estratégia: Copia RGB para 0-2 e inicializa X,Y (3-4) com zeros ou média.
        """
        device = next(self.parameters()).device
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        own_state = self.state_dict()
        
        adapted_state_dict = {}
        
        for name, param in state_dict.items():
            if name not in own_state:
                continue
                
            if own_state[name].shape != param.shape:
                # Caso clássico: Primeira convolução (ex: 64, 5, 3, 3) vs (64, 3, 3, 3)
                if (len(param.shape) == 4 and 
                    own_state[name].shape[0] == param.shape[0] and
                    own_state[name].shape[2:] == param.shape[2:] and
                    own_state[name].shape[1] == self.in_channels and 
                    param.shape[1] == 3):
                    
                    print(f"⚠️ Adaptando layer {name}: {param.shape} -> {own_state[name].shape} (CoordConv Injection)")
                    
                    new_weight = torch.zeros_like(own_state[name])
                    
                    # Copia RGB (canais 0,1,2)
                    new_weight[:, :3, :, :] = param
                    
                    # Inicializa X,Y (canais 3,4) com valores pequenos ou zero
                    # Zero é seguro para não estragar a inferência inicial
                    new_weight[:, 3:, :, :] = 0.0 
                    
                    adapted_state_dict[name] = new_weight
                else:
                    print(f"❌ Mismatch in {name}: {param.shape} vs {own_state[name].shape} - pulando")
            else:
                adapted_state_dict[name] = param
        
        # Carrega
        missing = self.load_state_dict(adapted_state_dict, strict=strict)
        return missing

    def forward(self, x):
        return self.model(x)

    def get_encoder(self):
        """Retorna a parte do encoder para ser usada como feature extractor."""
        return self.model.encoder

# --- Lógica de Treino para o Segmentador (Permanece igual) ---
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, logits, targets, smooth=1):
        """
        logits: raw network outputs (B,1,H,W)
        targets: float targets in [0,1] (B,1,H,W) - supports soft/context values
        Combina BCEWithLogits (estável numericamente) com Dice calculado sobre probabilidades.
        """
        # probabilidades para o termo Dice
        probs = torch.sigmoid(logits)

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1).float()

        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)

        # BCE estável a partir de logits
        bce = F.binary_cross_entropy_with_logits(logits.view(-1), targets_flat, reduction='mean')

        return bce + dice_loss

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
    # criterion = DiceBCELoss()
    criterion = TverskyFocalLoss(alpha=0.7, beta=0.3, gamma=0.75)

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
    encoder_warmup_epochs = 15       # warmup mais longo para encoder (increase)
    encoder_lr_factor = 0.05         # encoder lr = full_finetune_lr * encoder_lr_factor (increase)
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
            decoder_params = [p for n, p in model.named_parameters() if p.requires_grad and 'encoder' not in n]
            optimizer = torch.optim.AdamW(decoder_params, lr=lr, weight_decay=1e-5)

            # scheduler: OneCycleLR durante a fase de freeze (step por batch)
            try:
                steps_per_epoch = max(1, len(train_loader))
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=lr,
                    steps_per_epoch=steps_per_epoch,
                    epochs=max(1, min(freeze_epochs, epochs)),
                    pct_start=0.3,
                    anneal_strategy='cos',
                    div_factor=10.0,
                    final_div_factor=1e4,
                )
                scheduler_type = 'onecycle'
                if verbose:
                    print(f"OneCycleLR configured (decoder) max_lr={lr:.1e}, steps_per_epoch={steps_per_epoch}, epochs={min(freeze_epochs, epochs)}")
            except Exception:
                # fallback para ReduceLROnPlateau se OneCycle não for aplicável
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=7, factor=0.5)
                scheduler_type = 'plateau'

            # placeholders para optimizer/scheduler do encoder (serão criados no fine-tune)
            optimizer_encoder = None
            scheduler_encoder = None

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
                start_idx = max(0, n_children - groups_to_unfreeze)
                for idx in range(start_idx, n_children):
                    module = encoder_children[idx]
                    for p in module.parameters():
                        p.requires_grad = True
                current_unfrozen_children = groups_to_unfreeze
                if verbose:
                    print(f"Gradual unfreeze: desbloqueados inicialmente {current_unfrozen_children} encoder groups (dos {n_children}).")
            else:
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

            # CRIA um optimizer separado para o encoder (não inserir encoder no optimizer do decoder)
            encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad]
            encoder_base_lr = max(full_finetune_lr * encoder_lr_factor, 5e-6)
            if encoder_params:
                optimizer_encoder = torch.optim.AdamW(encoder_params, lr=encoder_base_lr, weight_decay=1e-5)
                # scheduler para encoder: Cosine (será combinado com warmup manual)
                scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_encoder, T_max=max(1, epochs - epoch - 1))
                if verbose:
                    print(f"Encoder optimizer criado com lr={encoder_base_lr:.2e}")
            else:
                optimizer_encoder = None
                scheduler_encoder = None

            # Para o decoder: se OneCycle foi usado, troca para um scheduler mais conservador pós-freeze
            if scheduler_type == 'onecycle':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - epoch - 1))
                scheduler_type = 'cosine_post'
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

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
                        # adiciona esses params ao optimizer_encoder com lr = encoder_base_lr * small factor (mantém decoder optimizer separado)
                        add_lr = (encoder_base_lr or (full_finetune_lr * encoder_lr_factor)) * 0.2
                        if 'optimizer_encoder' in locals() and optimizer_encoder is not None:
                            optimizer_encoder.add_param_group({'params': newly_unfrozen, 'lr': add_lr, 'weight_decay': 1e-5, 'name': f'encoder_g{current_unfrozen_children+1}'})
                        else:
                            # fallback inseguro: adiciona ao optimizer do decoder se optimizer_encoder não existir
                            optimizer.add_param_group({'params': newly_unfrozen, 'lr': add_lr, 'weight_decay': 1e-5, 'name': f'encoder_g{current_unfrozen_children+1}'})
                        current_unfrozen_children += 1
                        if verbose:
                            print(f"Gradual unfreeze: desbloqueado encoder child index {next_idx}; param_group adicionado ao optimizer_encoder com lr={add_lr:.2e} (total unfrozen groups={current_unfrozen_children}).")

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
                # Atualiza decoder optimizer
                optimizer.step()
                # Se existir optimizer para encoder, actualiza também (safe try/except)
                if 'optimizer_encoder' in locals() and optimizer_encoder is not None:
                    try:
                        optimizer_encoder.step()
                    except Exception:
                        pass
                # OneCycleLR (decoder) precisa de step por batch
                if scheduler is not None and scheduler_type == 'onecycle':
                    try:
                        scheduler.step()
                    except Exception:
                        pass
                # Zera gradientes em ambos os optimizers
                optimizer.zero_grad()
                if 'optimizer_encoder' in locals() and optimizer_encoder is not None:
                    try:
                        optimizer_encoder.zero_grad()
                    except Exception:
                        pass

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

        # Scheduler step: (a) se ReduceLROnPlateau usado para decoder, manda métrica;
        # (b) se post-cosine, chama step() por época; (c) também atualiza scheduler_encoder.
        prev_lr = optimizer.param_groups[0]['lr']
        try:
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_dice)
                else:
                    # para schedulers por-época (Cosine, etc.)
                    try:
                        scheduler.step()
                    except Exception:
                        pass
        except Exception:
            pass

        # Atualiza scheduler do encoder (se existir) por época + aplica warmup manual nos primeiros passos
        if 'optimizer_encoder' in locals() and optimizer_encoder is not None:
            # warmup manual durante encoder_warmup_epochs
            if epoch >= freeze_epochs and epoch < freeze_epochs + encoder_warmup_epochs:
                warm_step = (epoch - freeze_epochs + 1) / max(1, encoder_warmup_epochs)
                warm_lr = encoder_base_lr * (0.1 + 0.9 * warm_step)
                for g in optimizer_encoder.param_groups:
                    old_lr = g['lr']
                    g['lr'] = warm_lr
                if verbose:
                    print(f"[Encoder Warmup] lr -> {warm_lr:.2e}")
            else:
                # scheduler_encoder step normal (cosine)
                try:
                    scheduler_encoder.step()
                except Exception:
                    pass

        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr and verbose:
            print(f"[LR Scheduler] LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        # checkpoint by Val Dice
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice
            }
            if 'optimizer_encoder' in locals() and optimizer_encoder is not None:
                ckpt['optimizer_encoder_state_dict'] = optimizer_encoder.state_dict()
            torch.save(ckpt, save_path)
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