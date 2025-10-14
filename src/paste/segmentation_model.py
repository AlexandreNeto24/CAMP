import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
import segmentation_models_pytorch as smp # <--- ADICIONAR IMPORT
import matplotlib.pyplot as plt

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
        # Ex: resnet34 -> 512, resnet50 -> 2048, efficientnet-b0 -> 1280
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

def train_segmentor(model, train_dataset, val_dataset, run_dir, epochs=100, batch_size=4, lr=1e-3, device='cuda', accumulation_steps=4, freeze_epochs=20, full_finetune_lr=1e-5):
    """
    Função de treino com estratégia de fine-tuning em duas fases.
    """
    from src.paste.attention_dataset import attention_collate
    print(f"Iniciando treino do segmentador por {epochs} épocas...")
    print(f"Estratégia: {freeze_epochs} épocas com encoder congelado, depois fine-tuning completo.")

    if isinstance(device, str) and device.isdigit(): device = f'cuda:{device}'
    if isinstance(device, str) and device.startswith('cuda') and not torch.cuda.is_available():
        print(f"AVISO: Dispositivo CUDA '{device}' solicitado mas não disponível. Usando CPU."); device = 'cpu'
    device = torch.device(device)
    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=attention_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=attention_collate)
    
    # criterion = TverskyFocalLoss(alpha=0.7, beta=0.3, gamma=0.75)
    # criterion = DiceBCELoss()
    criterion = LovaszLoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 35
    save_path = os.path.join(run_dir, 'best_segmentor.pth') # Definir o caminho aqui
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # --- LÓGICA DE FINE-TUNING ---
        if epoch == 0:
            print("\n--- Fase 1: Treinando apenas o decoder (encoder congelado) ---")
            for param in model.model.encoder.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2, verbose=True)

        if epoch == freeze_epochs:
            # --- INÍCIO DA ALTERAÇÃO ---
            print("\n--- Carregando melhor modelo da Fase 1 antes do Fine-Tuning ---")
            if os.path.exists(save_path):
                model.load_state_dict(torch.load(save_path))
                print(f"Melhores pesos carregados de {save_path} (Val Loss: {best_val_loss:.4f})")
            else:
                print("AVISO: Nenhum modelo salvo encontrado. Continuando com os pesos atuais.")
            
            print("\n--- Fase 2: Fine-tuning do modelo completo (encoder descongelado) ---")
            # --- FIM DA ALTERAÇÃO ---
            for param in model.model.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=full_finetune_lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
        # --- FIM DA LÓGICA ---

        model.train()
        # O backbone deve estar sempre em modo eval() para desativar o comportamento específico de treino do BatchNorm
        model.model.encoder.eval()
        
        epoch_loss = 0
        # optimizer.zero_grad() # Limpa os gradientes no início da acumulação da época

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            images = batch['images'].to(device)
            true_masks = batch['segmentor_target_mask'].to(device)
            
            pred_masks = model(images)
            loss = criterion(pred_masks, true_masks)
            
            # --- LÓGICA DE ACUMULAÇÃO ---
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            # --- FIM DA LÓGICA ---
            
            epoch_loss += loss.item() * accumulation_steps

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = batch['images'].to(device)
                true_masks = batch['segmentor_target_mask'].to(device)
                pred_masks = model(images)
                val_loss += criterion(pred_masks, true_masks).item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # O save_path já está definido no início
            torch.save(model.state_dict(), save_path)
            print(f"Modelo de segmentação salvo em {save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping ativado na época {epoch+1}.")
            break
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.title("Segmentor Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(run_dir, 'segmentor_loss.png'))
        plt.close()
            
    return os.path.join(run_dir, 'best_segmentor.pth')
            
    return os.path.join(run_dir, 'best_segmentor.pth')