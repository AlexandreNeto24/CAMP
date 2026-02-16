import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

class DiffusionTrainer:
    def __init__(self, model, output_dir="experiments/diffusion_blend"):
        """
        Trainer para o modelo de Harmonização via Difusão.
        Args:
            model (DiffusionBlendingModel): Instância do wrapper do modelo.
            output_dir (str): Diretório para salvar checkpoints/logs.
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = model.device
        
    def train(self, train_loader, epochs=10, lr=1e-4, save_interval=5):
        """
        Loop de treino principal.
        """
        print(f"Starting generic Diffusion training for {epochs} epochs...")
        
        # Otimizador (apenas nos params treináveis da UNet - LoRA)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.unet.parameters()),
            lr=lr
        )
        
        loss_history = []
        
        for epoch in range(epochs):
            self.model.unet.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                # 1. Prepara Inputs (encapsulado no model)
                latent_input, noise, timesteps, text_embeddings = self.model.prepare_train_inputs(batch)
                
                # 2. Forward Pass (Predição de Ruído)
                music_pred = self.model.unet(
                    latent_input,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # 3. Loss (MSE entre Ruído Real e Predito)
                loss = F.mse_loss(music_pred, noise, reduction="mean")
                
                # 4. Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
            
            # Save Checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch+1)
                
        # Save Final
        self.save_checkpoint("final")
        self.plot_loss(loss_history)
        
        return self.model

    def save_checkpoint(self, tag):
        save_path = self.output_dir / f"checkpoint_{tag}"
        save_path.mkdir(exist_ok=True)
        # Salva apenas os pesos do LoRA (que são pequenos)
        self.model.unet.save_pretrained(save_path)
        print(f"Checkpoint saved: {save_path}")

    def plot_loss(self, losses):
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.savefig(self.output_dir / 'loss_curve.png')
        plt.close()
