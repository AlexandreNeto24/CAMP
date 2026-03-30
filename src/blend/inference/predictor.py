import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInpaintPipeline
from pathlib import Path

from src.blend.data.synthesis import calculate_boundary_mask


def poisson_blend(source_pil, target_pil, mask_pil, mode='mixed', erode_px=2):
    """
    Poisson blending (seamlessClone) — funde a instância no fundo 
    preservando gradientes, sem borda acinzentada.
    
    Args:
        source_pil: Imagem com o objeto harmonizado (ex: saída DoveNet).
        target_pil: Imagem de fundo original (onde o objeto deve fundir-se).
        mask_pil: Máscara binária do objeto (L mode).
        mode: 'mixed' (MIXED_CLONE — preserva gradiente mais forte entre
              source e target, bom para manter textura do fundo na transição)
              ou 'normal' (NORMAL_CLONE — transfere todos os gradientes do source).
        erode_px: Erodir máscara N pixels antes do clone. Evita artefactos
                  no bordo exacto da segmentação (que pode ter ruído).
    
    Returns:
        PIL.Image com blending suave.
    """
    source_np = np.array(source_pil.convert("RGB"))
    target_np = np.array(target_pil.convert("RGB"))
    mask_np = np.array(mask_pil.convert("L"))
    
    # Garante máscara binária limpa
    mask_bin = (mask_np > 127).astype(np.uint8) * 255
    
    # Erosão de segurança — remove ruído de borda da segmentação
    if erode_px > 0:
        kernel = np.ones((erode_px, erode_px), np.uint8)
        mask_bin = cv2.erode(mask_bin, kernel, iterations=1)
    
    # Encontra centro do objeto para seamlessClone
    coords = cv2.findNonZero(mask_bin)
    if coords is None:
        return source_pil
    
    x, y, w, h = cv2.boundingRect(coords)
    center = (x + w // 2, y + h // 2)
    
    # Segurança: centro dentro dos limites da imagem
    H, W = target_np.shape[:2]
    center = (np.clip(center[0], w//2 + 1, W - w//2 - 1),
              np.clip(center[1], h//2 + 1, H - h//2 - 1))
    
    clone_flag = cv2.MIXED_CLONE if mode == 'mixed' else cv2.NORMAL_CLONE
    
    try:
        result = cv2.seamlessClone(source_np, target_np, mask_bin, center, clone_flag)
        return Image.fromarray(result)
    except cv2.error:
        # Fallback se seamlessClone falhar (objeto muito perto da borda)
        return alpha_blend_smooth(source_pil, target_pil, mask_pil)


def alpha_blend_smooth(fg_pil, bg_pil, mask_pil, fade_pixels=None, profile='cosine'):
    """
    Alpha blending com gradiente suave via distance transform.
    A instância funde-se progressivamente no fundo sem seam.
    
    Args:
        fg_pil: Foreground (objeto harmonizado).
        bg_pil: Background (fundo original).
        mask_pil: Máscara do objeto.
        fade_pixels: Largura do gradiente em pixels (auto se None).
        profile: Perfil da transição:
          - 'cosine': (1-cos(π·t))/2 — derivada zero nas extremidades,
            transição C1-contínua. Recomendado para fusão natural.
          - 'linear': rampa linear (comportamento original).
    """
    mask_np = np.array(mask_pil.convert("L"))
    mask_bin = (mask_np > 127).astype(np.uint8)
    
    # Distance transform dentro e fora da máscara
    dist_inside = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5).astype(np.float32)
    dist_outside = cv2.distanceTransform(1 - mask_bin, cv2.DIST_L2, 5).astype(np.float32)
    
    # Auto-calcular largura do fade
    if fade_pixels is None:
        coords = cv2.findNonZero(mask_bin * 255)
        if coords is not None:
            _, _, w, h = cv2.boundingRect(coords)
            fade_pixels = max(8, int(min(w, h) * 0.15))
        else:
            fade_pixels = 15
    
    # Alpha: 1.0 no centro → decai suavemente → 0.0 no fundo
    alpha = np.zeros_like(dist_inside, dtype=np.float32)
    
    # Interior: de 0.5 (borda) a 1.0 (centro profundo)
    interior = dist_inside > 0
    alpha[interior] = 0.5 + 0.5 * np.clip(dist_inside[interior] / fade_pixels, 0, 1)
    
    # Exterior: de 0.5 (borda) a 0.0 (longe)
    exterior = dist_outside > 0
    alpha[exterior] = 0.5 * np.clip(1.0 - dist_outside[exterior] / fade_pixels, 0, 1)
    
    # Aplicar perfil de transição
    if profile == 'cosine':
        # f(t) = (1 - cos(π·t)) / 2
        # Derivada = 0 em t=0 e t=1 → sem descontinuidade de gradiente
        alpha = 0.5 * (1.0 - np.cos(np.pi * alpha))
    
    # Blur final para suavizar artefactos de quantização
    k = max(5, fade_pixels // 2)
    if k % 2 == 0: k += 1
    alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    
    # Blend
    fg_np = np.array(fg_pil.convert("RGB")).astype(np.float32)
    bg_np = np.array(bg_pil.convert("RGB")).astype(np.float32)
    alpha_3ch = alpha[:, :, np.newaxis]
    
    result = (fg_np * alpha_3ch + bg_np * (1 - alpha_3ch)).astype(np.uint8)
    return Image.fromarray(result)


class Harmonizer:
    """
    Pipeline de harmonização em 3 etapas — centrado no DoveNet:
    
      DoveNet (cor/tom) → Alpha Blend (junção suave) → SD Inpainting (fusão da fronteira)
    
    Princípios:
      - DoveNet é CENTRAL: é ele que harmoniza cor/luminosidade da instância.
      - Alpha Blend funde o output do DoveNet no fundo preservando a harmonização.
      - SD Inpainting funde a FRONTEIRA (ring) — não é só sombra, é para tornar
        a transição entre instância e fundo visualmente imperceptível.
      - Instância PRESERVADA ao máximo: o miolo nunca é tocado pelo SD.
    """
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting", lora_path=None, device="cuda"):
        self.device = device
        # print(f"Loading Harmonizer Pipeline: {model_id}...")
        
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            safety_checker=None
        ).to(device)
        
        if lora_path:
            print(f"Loading LoRA weights from {lora_path}...")
            try:
                # Tenta load_lora_weights (suporta safetensors/PEFT e .bin)
                self.pipe.load_lora_weights(lora_path)
                print("  LoRA carregado com sucesso (load_lora_weights).")
            except Exception as e1:
                try:
                    # Fallback: load_attn_procs (formato antigo .bin)
                    self.pipe.unet.load_attn_procs(lora_path)
                    print("  LoRA carregado com sucesso (load_attn_procs).")
                except Exception as e2:
                    print(f"  Aviso: LoRA não carregado. Usando modelo base.")
                    print(f"    Erro 1: {e1}")
                    print(f"    Erro 2: {e2}")
            
    def predict(self, dovenet_output, composite_input, mask_image, 
                prompt="seamless transition, realistic tissue texture, natural boundary, smooth blending",
                use_diffusion=True,
                blend_method='hybrid',
                ring_ratio=0.10,
                min_sd_area=2500):
        """
        Pipeline completo centrado no DoveNet:
          DoveNet output → Alpha/Poisson Blend (junção) → SD Inpainting (fusão fronteira) → Output
        
        Args:
            dovenet_output (PIL.Image): Saída do DoveNet (cores harmonizadas).
            composite_input (PIL.Image): Input original (fundo + objeto colado).
            mask_image (PIL.Image): Máscara binária do objeto.
            prompt (str): Prompt para SD — focado em fusão natural da fronteira.
            use_diffusion (bool): Se True, aplica SD na fronteira. Se False, só blend.
            blend_method (str): Método de junção na Etapa 1:
              - 'alpha': Alpha blend com perfil coseno (preserva cores DoveNet). Default.
              - 'poisson': Poisson mixed gradients (bom se DoveNet corrigiu bem a cor
                e queremos transição de textura mais natural). Nota: pode alterar
                ligeiramente as cores do DoveNet.
              - 'hybrid': Alpha blend + Poisson mixed. Poisson funde gradientes/textura,
                Alpha preserva cores harmonizadas do DoveNet no miolo.
            ring_ratio (float): Largura do anel para SD como fração da dimensão mínima
                do objeto. Default 0.10 (10%). Mais largo = SD tem mais contexto para
                fundir, mas arrisca alterar mais da instância.
            min_sd_area (int): Área mínima (em pixels) do objeto na imagem 512×512 para
                aplicar SD. Objectos com área inferior usam apenas blend (sem SD).
                O SD destrói objectos pequenos porque a ring mask ocupa quase toda a
                instância no espaço latente 64×64. Default 2500 (~50×50px).
                Set 0 para desactivar este limite.
            
        Returns:
            final_output (PIL.Image), ring_mask_pil (PIL.Image)
        """
        w_orig, h_orig = dovenet_output.size
        dove_resized = dovenet_output.resize((512, 512)).convert("RGB")
        comp_resized = composite_input.resize((512, 512)).convert("RGB")
        mask_resized = mask_image.resize((512, 512), Image.NEAREST).convert("L")
        
        # === ETAPA 1: Fusão DoveNet output → Background ===
        # O DoveNet já harmonizou cor/tom. Agora fundimos essa saída no fundo
        # de forma suave, preservando ao máximo a harmonização.
        if blend_method == 'poisson':
            # Poisson mixed: funde gradientes — bom para continuidade de textura
            # mas pode alterar ligeiramente a cor harmonizada pelo DoveNet.
            blended = poisson_blend(dove_resized, comp_resized, mask_resized,
                                    mode='mixed', erode_px=2)
        elif blend_method == 'hybrid':
            # Híbrido: Alpha coseno (preserva cores no miolo) + Poisson (gradientes na borda)
            alpha_result = alpha_blend_smooth(dove_resized, comp_resized, mask_resized, profile='cosine')
            poisson_result = poisson_blend(dove_resized, comp_resized, mask_resized,
                                           mode='mixed', erode_px=2)
            # Usar alpha no interior (preserva DoveNet), Poisson na fronteira (textura)
            mask_np_tmp = np.array(mask_resized)
            mask_bin_tmp = (mask_np_tmp > 127).astype(np.uint8)
            dist_inside = cv2.distanceTransform(mask_bin_tmp, cv2.DIST_L2, 5)
            coords_tmp = cv2.findNonZero(mask_bin_tmp * 255)
            if coords_tmp is not None:
                _, _, bw_tmp, bh_tmp = cv2.boundingRect(coords_tmp)
                fade_tmp = max(8, int(min(bw_tmp, bh_tmp) * 0.20))
            else:
                fade_tmp = 15
            # weight: 1.0 no centro (alpha), 0.0 na borda (poisson)
            w_alpha = np.clip(dist_inside / fade_tmp, 0, 1)
            w_alpha = cv2.GaussianBlur(w_alpha.astype(np.float32), (7, 7), 0)
            w_alpha_3 = w_alpha[:, :, np.newaxis]
            alpha_np = np.array(alpha_result).astype(np.float32)
            poisson_np = np.array(poisson_result).astype(np.float32)
            hybrid_np = (alpha_np * w_alpha_3 + poisson_np * (1 - w_alpha_3)).astype(np.uint8)
            blended = Image.fromarray(hybrid_np)
        else:  # 'alpha' (default)
            # Alpha coseno: preserva fielmente as cores do DoveNet, transição suave.
            blended = alpha_blend_smooth(dove_resized, comp_resized, mask_resized, profile='cosine')
        
        if not use_diffusion:
            blend_mask = calculate_boundary_mask(np.array(mask_resized), mode='blend')
            blend_mask_pil = Image.fromarray((blend_mask * 255).astype(np.uint8))
            return blended, blend_mask_pil
        
        # === GATE: Objectos pequenos — skip SD ===
        # O SD destrói objectos pequenos porque a ring mask, ao ser comprimida
        # 8× para o espaço latente (64×64), cobre quase todo o objecto.
        # O blend determinístico é suficiente para instâncias pequenas.
        if min_sd_area > 0:
            mask_area = int(np.sum(np.array(mask_resized) > 127))
            if mask_area < min_sd_area:
                print(f"  ⚡ Objecto pequeno ({mask_area}px² < {min_sd_area}px²) — skip SD, só blend.")
                blend_mask = calculate_boundary_mask(np.array(mask_resized), mode='blend')
                blend_mask_pil = Image.fromarray((blend_mask * 255).astype(np.uint8))
                return blended, blend_mask_pil
        
        # === ETAPA 2: SD Inpainting — fusão da fronteira ===
        # O SD actua APENAS no anel de transição (ring) entre instância e fundo.
        # Objectivo: tornar a fronteira imperceptível — fundir textura, iluminação,
        # sombras de contacto. A instância (miolo) fica INTOCADA.
        mask_np = np.array(mask_resized)
        mask_uint8 = (mask_np > 127).astype(np.uint8) * 255
        
        coords = cv2.findNonZero(mask_uint8)
        if coords is not None:
            _, _, bw, bh = cv2.boundingRect(coords)
            min_dim = min(bw, bh)
        else:
            min_dim = 100
        
        # Ring alargado (10%, era 5%) — dá ao SD contexto suficiente para
        # gerar uma transição natural entre instância e fundo.
        k_dil = max(5, int(min_dim * ring_ratio))
        dilated = cv2.dilate(mask_uint8, np.ones((k_dil, k_dil), np.uint8), iterations=1)
        k_ero = max(3, int(min_dim * 0.04))
        eroded = cv2.erode(mask_uint8, np.ones((k_ero, k_ero), np.uint8), iterations=1)
        ring = cv2.subtract(dilated, eroded)
        
        # Suavizar ring: gradiente em vez de binário → SD funde melhor
        k_blur = max(7, int(min_dim * 0.04))
        if k_blur % 2 == 0: k_blur += 1
        ring = cv2.GaussianBlur(ring, (k_blur, k_blur), 0)
        ring_pil = Image.fromarray(ring)
        
        negative = (
            "cartoon, drawing, artificial lines, hard cuts, glitches, "
            "extra objects, artifacts, gray border, unnatural texture"
        )
        
        with torch.no_grad():
            sd_output = self.pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=blended,               # Input = DoveNet + blend (já harmonizado)
                mask_image=ring_pil,         # SD só toca no anel de transição
                num_inference_steps=28,      # Ligeiramente mais que 25 → melhor fusão
                strength=0.55,               # Moderado: instância já está boa, SD refina
                guidance_scale=5.5
            ).images[0]
        
        # === ETAPA 3: Proteção — instância e fundo intocados ===
        # O SD só contribui na zona de transição (ring).
        # Interior profundo do objeto: preservado (= output do blend da Etapa 1)
        # Fundo longe da máscara: preservado (= original)
        protection = np.zeros_like(mask_np, dtype=np.uint8)
        k_protect = max(3, int(min_dim * 0.06))
        deep_eroded = cv2.erode(mask_uint8, np.ones((k_protect, k_protect), np.uint8), iterations=1)
        protection[deep_eroded > 0] = 255          # Interior do objeto: protegido
        protection[dilated == 0] = 255              # Fundo distante: protegido
        
        # Suavizar bordo da proteção proporcionalmente ao ring
        blur_r = max(3, k_dil // 3)
        protection_pil = Image.fromarray(protection).filter(ImageFilter.GaussianBlur(radius=blur_r))
        
        # Onde proteção=255 → usa blended (Etapa 1), onde proteção=0 → usa SD
        final = Image.composite(blended, sd_output, protection_pil)
        
        return final, ring_pil

    def harmonize(self, dovenet_output, composite_input, object_mask, 
                  prompt="seamless transition, realistic tissue texture, natural boundary, smooth blending",
                  use_diffusion=True, **kwargs):
        """Wrapper para o pipeline completo. Aceita kwargs extra (blend_method, ring_ratio)."""
        return self.predict(dovenet_output, composite_input, object_mask, prompt, use_diffusion, **kwargs)
