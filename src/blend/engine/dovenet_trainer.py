import torch
import torch.nn as nn
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from src.blend.models.dovenet import GradientLoss

class DoveNetTrainer:
    def __init__(self, model, output_dir="experiments/dovenet_blend", device="cuda"):
        """
        Trainer específico para o DoveNet, replicando a estratégia de Loss Híbrida do blend_v2.
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Losses
        self.criterion_pixel = nn.MSELoss().to(device)
        self.criterion_grad = GradientLoss().to(device)
        
    def train(self, train_loader, epochs=50, lr=1e-4, save_interval=10, weights={'pixel': 1.0, 'grad': 0.5}):
        """
        Loop de treino Híbrido (Pixel + Gradient + BorderFocus).
        """
        print(f"Starting DoveNet training for {epochs} epochs...")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_history = []
        
        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                # Load Data
                composite = batch['pixel_values'].to(self.device) 
                real = batch['ground_truth'].to(self.device)      
                mask = batch['mask'].to(self.device)        
                boundary = batch['boundary_mask'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                output = self.model(composite, mask, boundary)
                
                # --- LOSS CALCULATION (Igual ao blend_v2) ---
                loss_pixel_global = self.criterion_pixel(output, real)
                
                # Foca a loss na zona da boundary e objeto
                loss_pixel_local = self.criterion_pixel(output * mask, real * mask)
                loss_boundary_focus = self.criterion_pixel(output * boundary, real * boundary)
                
                # Peso maior na borda (5.0) e objeto (2.0)
                loss_pixel = loss_pixel_global + (2.0 * loss_pixel_local) + (5.0 * loss_boundary_focus)
                
                # Gradient Loss (Smoothness)
                loss_grad = self.criterion_grad(output, real)
                
                # Total
                loss = (weights['pixel'] * loss_pixel) + (weights['grad'] * loss_grad)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'L_Pix': loss_pixel.item(), 'L_Grad': loss_grad.item()})
            
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
            
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch+1)
        
        self.save_checkpoint("final")
        self.plot_loss(loss_history)
        return self.model

    def save_checkpoint(self, tag):
        save_path = self.output_dir / f"dovenet_epoch_{tag}.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Checkpoint saved: {save_path}")

    def plot_loss(self, losses):
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Train Loss')
        plt.title('DoveNet Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.output_dir / 'loss_curve.png')
        plt.close()
