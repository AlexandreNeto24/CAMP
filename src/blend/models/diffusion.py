import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

class DiffusionBlendingModel:
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting", device="cuda"):
        self.device = device
        self.model_id = model_id
        
        print(f"Loading Stable Diffusion components from {model_id}...")
        
        # 1. Load Components
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Congelar pesos base
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # Move para device
        self.vae.to(device)
        self.text_encoder.to(device)
        self.unet.to(device)
        
    def setup_lora(self, rank=8):
        """
        Injeta adaptadores LoRA na U-Net para fine-tuning eficiente.
        """
        print(f"Injecting LoRA adapters (rank={rank})...")
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        self.unet.to(self.device)
        
        return self.unet

    def get_latents(self, image):
        """Converte imagem RGB [B, 3, H, W] para Latents [B, 4, h, w]"""
        # Imagem deve estar em [-1, 1]
        image = 2.0 * image - 1.0
        dist = self.vae.encode(image).latent_dist
        latents = dist.sample() * self.vae.config.scaling_factor
        return latents

    def get_masked_image_latents(self, image, mask):
        """
        Prepara a 'masked_image' para o inpainting.
        """
        masked_image = image * (1 - mask)
        return self.get_latents(masked_image)

    def prepare_train_inputs(self, batch):
        """
        Prepara todo o batch para um passo de treino.
        Batch contém: 'pixel_values' (composite), 'ground_truth' (real), 'mask' (objeto), 'boundary_mask' (fita)
        """
        bsz = batch['ground_truth'].shape[0]
        
        # Ground Truth (Target)
        target_img = batch['ground_truth'].to(self.device, dtype=self.vae.dtype) # Ensure dtype match (float32 usually)
        target_latents = self.get_latents(target_img) # [B, 4, 64, 64]
        
        # Noise
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)
        timesteps = timesteps.long()
        
        # Noisy Latents (p(x_t))
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
        
        # --- Condicionamento do Inpainting ---
        composite_img = batch['pixel_values'].to(self.device, dtype=self.vae.dtype)
        
        # 2. Máscara de Boundary (Zona a refazer)
        mask_boundary = batch['boundary_mask'].to(self.device, dtype=self.vae.dtype)
        # Redimensiona para tamanho dos latents (1/8 of 512 = 64)
        mask_boundary_down = F.interpolate(mask_boundary, size=target_latents.shape[-2:], mode="nearest")
        
        # 3. Masked Image Latents
        # Usa o 'composite' como base, e mascara a zona de boundary
        # A U-Net vai ver o interior do objeto (composite) e o fundo (composite), e tem de preencher a boundary
        masked_composite_latents = self.get_latents(composite_img * (1 - mask_boundary))
        
        # 4. Inputs concatenados para a U-Net Inpainting (9 canais padrão no SD 2-inpainting)
        # [Noisy_Latents (4), Mask (1), Masked_Image_Latents (4)]
        latent_model_input = torch.cat([noisy_latents, mask_boundary_down, masked_composite_latents], dim=1)
        
        # 5. Prompt Vazio
        empty_prompt_ids = self.tokenizer(
            [""] * bsz, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.to(self.device)
        encoder_hidden_states = self.text_encoder(empty_prompt_ids)[0]
        
        return latent_model_input, noise, timesteps, encoder_hidden_states
