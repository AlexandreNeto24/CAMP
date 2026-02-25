"""
CIMPipeline — Copy-Insert-Merge Pipeline Completo.

Pipeline unificado para data augmentation médica:
  1. Segmentação  → Extrair instâncias (patches + máscaras) da imagem source
  2. Recomendação → Encontrar a melhor localização no background (target)
  3. Paste        → Colar o patch na localização recomendada
  4. DoveNet      → Harmonizar cor/tom da instância colada
  5. Blend + SD   → Fundir fronteira (Hybrid Blend + SD Inpainting)

Uso:
    from src.pipeline import CIMPipeline
    
    pipe = CIMPipeline.from_pretrained('metaplasia', experiments_dir='experiments/')
    result = pipe.generate(target_image, source_image)
    # ou batch:
    results = pipe.generate_batch(target_images, source_images, n=100)
"""

import gc
import cv2
import torch
import random
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Optional, List, Dict, Tuple, Union


class CIMPipeline:

    def generate(
        self,
        target_image: Image.Image,
        source_image: Image.Image,
        source_mask: Optional[Image.Image] = None,
        patch_index: int = 0,
        recommendation_index: int = 0,
        top_k: int = 5,
        use_tto: bool = True,
        blend_method: str = 'hybrid',
        use_diffusion: Optional[bool] = None,
        return_intermediates: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, Dict]]:
        """
        Gera UMA imagem aumentada (pipeline completo).
        Args:
            target_image: Imagem de fundo (onde colar)
            source_image: Imagem source (de onde extrair o patch)
            source_mask: Máscara da source (opcional, senão usa segmentador)
            patch_index: Qual patch usar (0=maior)
            recommendation_index: Qual recomendação usar (0=melhor)
            top_k: Número de candidatos
            use_tto: Test-Time Optimization
            blend_method: 'alpha', 'poisson', 'hybrid'
            use_diffusion: None=auto, True/False=forçar
            return_intermediates: Se True, retorna dict com todas as etapas
        Returns:
            PIL.Image (resultado final) ou (PIL.Image, dict) se return_intermediates
        """
        if self.verbose:
            print("\n─── CIMPipeline.generate() ───")
        # 1. Extrair patches
        if self.verbose: print("[1/4] Extraindo patches da source...")
        patches = self.extract_patches(source_image, source_mask)
        if not patches:
            raise ValueError("Nenhum patch extraído da source image.")
        patch_info = patches[min(patch_index, len(patches) - 1)]
        # 2. Recomendar localização
        if self.verbose: print("[2/4] Recomendando localização...")
        recommendations = self.recommend(target_image, patch_info, top_k=top_k, use_tto=use_tto)
        if not recommendations:
            raise ValueError("Nenhuma recomendação gerada.")
        rec = recommendations[min(recommendation_index, len(recommendations) - 1)]
        # 3. Paste
        if self.verbose: print("[3/4] Colando patch...")
        composite, mask, boundary = self.paste(target_image, patch_info, rec)
        # 4. Harmonizar
        if self.verbose: print("[4/4] Harmonizando...")
        result, harm_info = self.harmonize(
            composite, mask, boundary,
            blend_method=blend_method,
            use_diffusion=use_diffusion
        )
        if self.verbose: print("─── Pipeline concluído ───\n")
        if return_intermediates:
            intermediates = {
                'source_image': source_image,
                'target_image': target_image,
                'patches': patches,
                'patch_used': patch_info,
                'recommendations': recommendations,
                'recommendation_used': rec,
                'composite': composite,
                'mask': mask,
                'boundary': boundary,
                **harm_info
            }
            return result, intermediates
        return result
    
    def __init__(
        self,
        segmentor=None,
        recommender=None,
        dovenet=None,
        harmonizer=None,
        device: str = 'cuda',
        image_size: Tuple[int, int] = (512, 512),
        verbose: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.verbose = verbose
        
        # Componentes (podem ser None se não carregados)
        self.segmentor = segmentor
        self.recommender = recommender
        self.dovenet = dovenet
        self.harmonizer = harmonizer
        
        # Transforms standard
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        
        if self.verbose:
            loaded = []
            if segmentor:  loaded.append('Segmentor')
            if recommender: loaded.append('Recommender')
            if dovenet:    loaded.append('DoveNet')
            if harmonizer: loaded.append('Harmonizer')
            print(f"CIMPipeline inicializado ({self.device}) — Componentes: {', '.join(loaded) or 'nenhum'}")
    
    @classmethod
    def from_pretrained(
        cls,
        class_name: str,
        experiments_dir: str = 'experiments',
        project_root: str = '.',
        device: str = 'cuda',
        load_sd: bool = True,
        segmentor_encoder: str = 'efficientnet-b3',
        verbose: bool = True
    ) -> 'CIMPipeline':
        """
        Carrega todos os componentes automaticamente a partir de pesos existentes.
        
        Args:
            class_name: Nome da classe (ex: 'metaplasia', 'polyps', 'airplane')
            experiments_dir: Pasta com os pesos treinados
            project_root: Raiz do projecto
            device: 'cuda' ou 'cpu'
            load_sd: Se True, carrega SD Inpainting (+ ~4GB VRAM). Se False, só blend.
            segmentor_encoder: Encoder do segmentador ('efficientnet-b3', 'resnet34', etc.)
            verbose: Mostrar progresso
        
        Returns:
            CIMPipeline com todos os componentes carregados
        """
        exp_dir = Path(project_root) / experiments_dir
        
        if verbose:
            print(f"{'='*60}")
            print(f"  CIMPipeline — Carregando modelos para '{class_name}'")
            print(f"{'='*60}")
        
        # ─── 1. SEGMENTOR ───────────────────────────────────────────
        if verbose: print("\n[1/4] Segmentor (PretrainedUNet)...")
        segmentor = cls._load_segmentor(
            class_name, exp_dir, device, segmentor_encoder, verbose
        )
        
        # ─── 2. RECOMMENDER ─────────────────────────────────────────
        if verbose: print("\n[2/4] Recommender (AttentionRecommender)...")
        recommender = cls._load_recommender(
            class_name, exp_dir, device, segmentor, verbose
        )
        
        # ─── 3. DOVENET ─────────────────────────────────────────────
        if verbose: print("\n[3/4] DoveNet (Harmonização de cor)...")
        dovenet = cls._load_dovenet(exp_dir, device, verbose)
        
        # ─── 4. HARMONIZER (SD + LoRA) ──────────────────────────────
        harmonizer = None
        if load_sd:
            if verbose: print("\n[4/4] Harmonizer (SD Inpainting + LoRA)...")
            harmonizer = cls._load_harmonizer(exp_dir, device, verbose)
        else:
            if verbose: print("\n[4/4] Harmonizer — SKIP (load_sd=False)")
        
        if verbose:
            print(f"\n{'='*60}")
            vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            print(f"  Pipeline pronto — VRAM: {vram:.1f} GB")
            print(f"{'='*60}")
        
        return cls(
            segmentor=segmentor,
            recommender=recommender,
            dovenet=dovenet,
            harmonizer=harmonizer,
            device=device,
            verbose=verbose
        )
    
    # ─── LOADING HELPERS ────────────────────────────────────────────
    
    @staticmethod
    def _find_latest_weights(exp_dir: Path, pattern: str) -> Optional[Path]:
        """Encontra os pesos mais recentes que correspondem ao padrão."""
        candidates = sorted(exp_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None
    
    @classmethod
    def _load_segmentor(cls, class_name, exp_dir, device, encoder_name, verbose):
        from src.paste.segmentation_model import PretrainedUNet
        
        # 1. Nova estrutura: weights/<class>/segmentor/best.pth
        weights_path = None
        weights_dir = exp_dir.parent / 'weights' / class_name / 'segmentor'
        for name in ['best.pth', 'best_segmentor.pth']:
            if (weights_dir / name).exists():
                weights_path = weights_dir / name
                break
        
        # 2. Fallback: experiments/ (estrutura antiga)
        if weights_path is None:
            weights_path = cls._find_latest_weights(
                exp_dir, f"{class_name}_segmentation_*/best_segmentor.pth"
            )
        
        # Construir em CPU — AttentionRecommender.__init__ faz forward com dummy CPU
        model = PretrainedUNet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
        
        if weights_path:
            try:
                state = torch.load(weights_path, map_location='cpu')
                if isinstance(state, dict) and 'model_state_dict' in state:
                    state = state['model_state_dict']
                model.load_state_dict(state, strict=False)
            except Exception:
                model.load_adapted_state_dict(str(weights_path), strict=False)
            if verbose: print(f"  ✓ Pesos: {weights_path.relative_to(exp_dir.parent)}")
        else:
            model = PretrainedUNet(
                encoder_name=encoder_name,
                encoder_weights='imagenet',
                in_channels=3,
                classes=1
            )
            if verbose: print(f"  ⚠ Sem pesos para '{class_name}' — usando ImageNet features")
        
        model.eval()
        return model  # Fica em CPU; será movido para device após construir o recommender
    
    @classmethod
    def _load_recommender(cls, class_name, exp_dir, device, segmentor, verbose):
        from src.paste.models.attention_recommender import AttentionRecommender
        
        # 1. Nova estrutura: weights/<class>/recommender/best.pth
        weights_dir = exp_dir.parent / 'weights' / class_name / 'recommender'
        weights_path = None
        for name in ['best.pth', 'best_model.pt']:
            if (weights_dir / name).exists():
                weights_path = weights_dir / name
                break
        
        # 2. Fallback: experiments/ (estrutura antiga)
        if weights_path is None:
            weights_path = cls._find_latest_weights(
                exp_dir, f"*{class_name}*attention_recommender_*/best_model.pt"
            )
        if weights_path is None:
            weights_path = cls._find_latest_weights(
                exp_dir, f"{class_name}_attention_recommender_*/best_model.pt"
            )
        if weights_path is None:
            weights_path = cls._find_latest_weights(
                exp_dir, "attention_recommender_*/best_model.pt"
            )
        
        # Construir em CPU (segmentor já está em CPU)
        model = AttentionRecommender(
            backbone_model=segmentor,
            embedding_dim=2048,
            num_heads=16,
            dropout_rate=0.3
        )
        
        if weights_path:
            model.load_model(str(weights_path))
            if verbose: print(f"  ✓ Pesos: {weights_path.relative_to(exp_dir.parent)}")
        else:
            if verbose: print(f"  ⚠ Sem pesos de recommender — inferência sem treino")
        
        # Agora sim, mover tudo (segmentor + recommender) para device
        model.to(device)
        model.eval()
        return model
    
    @classmethod
    def _load_dovenet(cls, exp_dir, device, verbose):
        from src.blend.models.dovenet import DoveNetGenerator
        
        model = DoveNetGenerator().to(device)
        
        # Prioridade: finetuned > base > epoch checkpoints
        for name in ['best_harmonizer_dovenet_finetuned.pth',
                     'best_harmonizer_dovenet.pth',
                     'baseline_harmonizer.pth']:
            path = exp_dir / name
            if path.exists():
                model.load_state_dict(torch.load(path, map_location=device))
                if verbose: print(f"  ✓ Pesos: {name}")
                break
        else:
            # Fallback: último checkpoint de epoch
            epoch_path = cls._find_latest_weights(exp_dir, "dovenet_epoch_*.pth")
            if epoch_path:
                model.load_state_dict(torch.load(epoch_path, map_location=device))
                if verbose: print(f"  ✓ Pesos: {epoch_path.name}")
            else:
                if verbose: print(f"  ⚠ Sem pesos DoveNet — modelo não treinado")
        
        model.eval()
        return model
    
    @classmethod
    def _load_harmonizer(cls, exp_dir, device, verbose):
        from src.blend.inference.predictor import Harmonizer
        
        # Procura LoRA weights
        lora_path = None
        for candidate in [exp_dir / "diffusion_conservative" / "lora_final_conservative",
                          exp_dir / "diffusion_lora" / "checkpoint_final"]:
            if candidate.exists():
                lora_path = str(candidate)
                if verbose: print(f"  LoRA: {candidate.relative_to(exp_dir.parent)}")
                break
        
        harmonizer = Harmonizer(lora_path=lora_path, device=str(device))
        if verbose: print(f"  ✓ SD Inpainting carregado")
        return harmonizer
    
    # ─── CORE PIPELINE ──────────────────────────────────────────────
    
    def extract_patches(
        self,
        source_image: Image.Image,
        source_mask: Optional[Image.Image] = None,
        min_area: int = 200,
        max_patches: int = 10
    ) -> List[Dict]:
        """
        Extrai patches (instâncias) de uma imagem source.
        
        Se source_mask for fornecida, usa-a directamente.
        Se não, usa o segmentador para gerar a máscara.
        
        Returns:
            Lista de dicts, cada um com:
              - 'patch': PIL.Image (recorte do objecto)
              - 'mask': PIL.Image (máscara binária do recorte)
              - 'patch_tensor': Tensor [3, h, w]
              - 'mask_tensor': Tensor [1, h, w]
              - 'bbox': (x1, y1, x2, y2)
              - 'area': int
              - 'patch_with_context': PIL.Image (patch com contexto à volta)
              - 'context_mask': Tensor [1, h, w]
        """
        img_resized = source_image.resize(self.image_size, Image.BILINEAR)
        
        if source_mask is not None:
            mask_resized = source_mask.resize(self.image_size, Image.NEAREST)
            mask_np = (np.array(mask_resized.convert('L')) > 127).astype(np.uint8)
        elif self.segmentor is not None:
            # Segmentar automaticamente
            img_t = self.to_tensor(img_resized).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.segmentor(img_t)
                mask_np = (torch.sigmoid(logits).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        else:
            raise ValueError("Sem máscara fornecida e sem segmentador carregado.")
        
        # Connected components → instâncias individuais
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_np.astype(np.uint8), connectivity=8
        )
        
        img_np = np.array(img_resized)
        patches = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Máscara isolada desta instância
            instance_mask = (labels == i).astype(np.uint8) * 255
            
            # Crop do objecto
            patch_crop = img_np[y:y+h, x:x+w]
            mask_crop = instance_mask[y:y+h, x:x+w]
            
            # Patch com contexto (margem de 30%)
            margin = int(max(w, h) * 0.3)
            cx1 = max(0, x - margin)
            cy1 = max(0, y - margin)
            cx2 = min(self.image_size[1], x + w + margin)
            cy2 = min(self.image_size[0], y + h + margin)
            
            context_crop = img_np[cy1:cy2, cx1:cx2]
            context_mask_np = np.zeros_like(context_crop[:, :, 0], dtype=np.float32)
            # Marcar onde está o objecto no crop de contexto
            obj_y1 = y - cy1
            obj_x1 = x - cx1
            context_mask_np[obj_y1:obj_y1+h, obj_x1:obj_x1+w] = mask_crop.astype(np.float32) / 255.0
            
            patch_pil = Image.fromarray(patch_crop)
            mask_pil = Image.fromarray(mask_crop)
            context_pil = Image.fromarray(context_crop)
            
            # DEBUG: Mostra sumário da máscara e patch
            if self.verbose:
                print(f"  Patch {i}: area={area}, bbox=({x},{y},{x+w},{y+h}), mask sum={mask_crop.sum()}, shape={mask_crop.shape}")
            patches.append({
                'patch': patch_pil,
                'mask': mask_pil,  # Só a instância!
                'patch_tensor': self.to_tensor(patch_crop),
                'mask_tensor': torch.from_numpy(mask_crop).unsqueeze(0).float() / 255.0,
                'bbox': (x, y, x + w, y + h),
                'area': area,
                'patch_with_context': context_pil,
                'context_mask': torch.from_numpy(context_mask_np).unsqueeze(0),
            })
        
        # Ordenar por área (maior primeiro) e limitar
        patches.sort(key=lambda p: p['area'], reverse=True)
        patches = patches[:max_patches]
        
        if self.verbose:
            print(f"  Extraídos {len(patches)} patches (áreas: {[p['area'] for p in patches]})")
        
        return patches
    
    def recommend(
        self,
        target_image: Image.Image,
        patch_info: Dict,
        top_k: int = 5,
        use_tto: bool = True
    ) -> List[Dict]:
        """
        Recomenda localizações para colar o patch no target.
        
        Args:
            target_image: Imagem de fundo (background)
            patch_info: Dict retornado por extract_patches()
            top_k: Número de recomendações
            use_tto: Test-Time Optimization (multi-escala/rotação)
        
        Returns:
            Lista de recomendações: [{"box", "score", "rank", "scale", "angle", "flip"}]
        """
        if self.recommender is None:
            raise ValueError("Recommender não carregado.")
        
        from src.paste.inference.recommender import RecommendationEngine
        engine = RecommendationEngine(self.recommender)
        
        target_resized = target_image.resize(self.image_size, Image.BILINEAR)
        target_t = self.to_tensor(target_resized).unsqueeze(0).to(self.device)
        
        # Resize patch para 128×128 (tamanho de inferência do recommender)
        patch_t = patch_info['patch_tensor']
        patch_t = torch.nn.functional.interpolate(
            patch_t.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
        ).to(self.device)
        
        # Context patch
        ctx_t = None
        if 'patch_with_context' in patch_info:
            ctx_pil = patch_info['patch_with_context']
            ctx_t = self.to_tensor(ctx_pil.resize((128, 128), Image.BILINEAR))
            ctx_t = ctx_t.unsqueeze(0).to(self.device)
        
        recs = engine.recommend_placements(
            target_t, patch_t,
            patch_with_context=ctx_t,
            patch_box=list(patch_info['bbox']),
            top_k=top_k,
            use_tto=use_tto
        )
        
        if self.verbose and recs:
            print(f"  Top-{len(recs)} recomendações (melhor score: {recs[0]['score']:.4f})")
        
        return recs
    
    def paste(
        self,
        target_image: Image.Image,
        patch_info: Dict,
        recommendation: Dict
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """
        Cola o patch na localização recomendada.
        
        Returns:
            (composite_pil, mask_pil, boundary_pil)
            - composite: Imagem com o patch colado (colagem crua)
            - mask: Máscara binária do objecto na posição final
            - boundary: Boundary mask para o DoveNet (soft gradient)
        """
        target_resized = target_image.resize(self.image_size, Image.BILINEAR)
        
        box = recommendation['box']
        x1, y1, x2, y2 = [int(c) for c in box]
        target_w = x2 - x1
        target_h = y2 - y1
        
        if target_w <= 0 or target_h <= 0:
            raise ValueError(f"Box inválida: {box}")
        
        target_np = np.array(target_resized)
        # Garante que só o patch puro da instância é usado (NUNCA patch_with_context)
        patch_np = np.array(patch_info['patch'])
        mask_np = np.array(patch_info['mask'])
        # Adiciona assert para garantir que não é contexto
        if 'patch_with_context' in patch_info:
            ctx_np = np.array(patch_info['patch_with_context'])
            assert not np.array_equal(patch_np, ctx_np), "ERRO: patch está igual ao contexto, revise extração de patch."
        if self.verbose:
            print(f"[DEBUG paste] patch shape: {patch_np.shape}, mask shape: {mask_np.shape}, mask sum: {mask_np.sum()}")
        
        # Resize patch e máscara para o tamanho da box
        patch_resized = cv2.resize(patch_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_resized > 127
        
        # Clipping (box pode sair dos limites)
        H, W = target_np.shape[:2]
        img_x1, img_y1 = max(0, x1), max(0, y1)
        img_x2, img_y2 = min(W, x2), min(H, y2)
        patch_x1, patch_y1 = img_x1 - x1, img_y1 - y1
        patch_x2 = patch_x1 + (img_x2 - img_x1)
        patch_y2 = patch_y1 + (img_y2 - img_y1)
        
        # Composite (hard paste)
        composite = target_np.copy()
        region = composite[img_y1:img_y2, img_x1:img_x2]
        patch_crop = patch_resized[patch_y1:patch_y2, patch_x1:patch_x2]
        mask_crop = mask_bool[patch_y1:patch_y2, patch_x1:patch_x2]
        
        if len(region.shape) == 3:
            mask_3ch = np.stack([mask_crop] * 3, axis=-1)
        else:
            mask_3ch = mask_crop
        
        np.copyto(region, patch_crop, where=mask_3ch)
        composite[img_y1:img_y2, img_x1:img_x2] = region
        
        # Máscara completa na imagem
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[img_y1:img_y2, img_x1:img_x2] = mask_resized[patch_y1:patch_y2, patch_x1:patch_x2]
        
        # Boundary mask (soft gradient para DoveNet)
        from src.blend.data.synthesis import calculate_boundary_mask
        boundary = calculate_boundary_mask(full_mask, mode='wide')
        boundary_uint8 = (boundary * 255).astype(np.uint8)
        
        return (
            Image.fromarray(composite),
            Image.fromarray(full_mask),
            Image.fromarray(boundary_uint8)
        )
    
    def harmonize(
        self,
        composite: Image.Image,
        mask: Image.Image,
        boundary: Image.Image,
        blend_method: str = 'hybrid',
        use_diffusion: Optional[bool] = None,
        ring_ratio: float = 0.10,
        min_sd_area: int = 200
    ) -> Tuple[Image.Image, Dict]:
        """
        Harmoniza a colagem: DoveNet (cor) → Blend (junção) → SD (fronteira).
        
        Args:
            composite: Imagem com patch colado
            mask: Máscara binária do objecto
            boundary: Boundary mask para DoveNet
            blend_method: 'alpha', 'poisson', 'hybrid'
            use_diffusion: None=auto (True se harmonizer carregado), True/False=forçar
            ring_ratio: Largura do anel SD
            min_sd_area: Threshold de área para skip SD
        
        Returns:
            (result_pil, info_dict)
        """
        if self.dovenet is None:
            raise ValueError("DoveNet não carregado.")
        
        # Decide se usa SD
        if use_diffusion is None:
            use_diffusion = self.harmonizer is not None
        
        # ── Etapa 1: DoveNet (cor/tom) ──
        comp_t = self.to_tensor(composite.resize(self.image_size)).unsqueeze(0).to(self.device)
        mask_t = self.to_tensor(mask.resize(self.image_size, Image.NEAREST)).unsqueeze(0).to(self.device)
        bound_t = self.to_tensor(boundary.resize(self.image_size)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            dovenet_out = self.dovenet(comp_t, mask_t, bound_t)
        dovenet_pil = self.to_pil(dovenet_out.squeeze().cpu())
        
        mask_512 = mask.resize(self.image_size, Image.NEAREST)

        # ── Etapa 2+3: Blend + SD ──
        if use_diffusion and self.harmonizer is not None:
            result, ring_mask = self.harmonizer.predict(
                dovenet_pil, dovenet_pil, mask_512,
                blend_method=blend_method,
                ring_ratio=ring_ratio,
                min_sd_area=min_sd_area
            )
        else:
            # Só blend (sem SD) — sempre sobre a saída do DoveNet
            from src.blend.inference.predictor import alpha_blend_smooth, poisson_blend
            if blend_method == 'poisson':
                result = poisson_blend(dovenet_pil, dovenet_pil, mask_512)
            elif blend_method == 'hybrid':
                alpha_r = alpha_blend_smooth(dovenet_pil, dovenet_pil, mask_512, profile='cosine')
                poisson_r = poisson_blend(dovenet_pil, dovenet_pil, mask_512, mode='mixed')
                # Blend híbrido (simplificado)
                mask_np = np.array(mask_512.convert('L'))
                mask_bin = (mask_np > 127).astype(np.uint8)
                dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
                w_alpha = np.clip(dist / max(dist.max() * 0.3, 1), 0, 1)[:, :, np.newaxis]
                hybrid = (np.array(alpha_r).astype(float) * w_alpha + 
                         np.array(poisson_r).astype(float) * (1 - w_alpha))
                result = Image.fromarray(hybrid.astype(np.uint8))
            else:
                result = alpha_blend_smooth(dovenet_pil, dovenet_pil, mask_512, profile='cosine')
            ring_mask = mask_512
        
        info = {
            'dovenet_output': dovenet_pil,
            'blend_method': blend_method,
            'use_diffusion': use_diffusion and self.harmonizer is not None,
            'ring_mask': ring_mask,
        }
        return result, info

    def generate_batch(
        self,
        target_images: List[Image.Image],
        source_images: List[Image.Image],
        source_masks: Optional[List[Image.Image]] = None,
        n: int = 10,
        output_dir: Optional[str] = None,
        blend_method: str = 'hybrid',
        use_diffusion: Optional[bool] = None,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Gera N imagens aumentadas a partir de listas de targets e sources.
        
        Args:
            target_images: Lista de backgrounds
            source_images: Lista de sources (de onde extrair patches)
            source_masks: Lista de máscaras (opcional)
            n: Número de imagens a gerar
            output_dir: Se especificado, salva as imagens + máscaras
            blend_method: Método de fusão
            use_diffusion: Usar SD?
            seed: Seed para reproduzibilidade
        
        Returns:
            Lista de dicts: [{"image": PIL, "mask": PIL, "source_idx": int, "target_idx": int}]
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if output_dir:
            out_path = Path(output_dir)
            (out_path / 'images').mkdir(parents=True, exist_ok=True)
            (out_path / 'masks').mkdir(parents=True, exist_ok=True)
        
        # Pré-extrair patches de todas as sources
        if self.verbose: print(f"Pré-extraindo patches de {len(source_images)} sources...")
        all_patches = []
        for i, src in enumerate(source_images):
            src_mask = source_masks[i] if source_masks else None
            try:
                patches = self.extract_patches(src, src_mask)
                for p in patches:
                    p['source_idx'] = i
                all_patches.extend(patches)
            except Exception as e:
                if self.verbose: print(f"  Aviso: source {i} falhou — {e}")
        
        if not all_patches:
            raise ValueError("Nenhum patch extraído de nenhuma source.")
        
        if self.verbose:
            print(f"Total: {len(all_patches)} patches disponíveis")
        
        results = []
        old_verbose = self.verbose
        self.verbose = False  # Silenciar durante batch
        
        for i in range(n):
            try:
                # Escolher target e patch aleatórios
                target_idx = random.randint(0, len(target_images) - 1)
                patch_info = random.choice(all_patches)
                
                target = target_images[target_idx]
                
                # Recomendar + Paste + Harmonizar
                recommendations = self.recommend(target, patch_info, top_k=3, use_tto=False)
                if not recommendations:
                    continue
                
                rec = recommendations[0]
                composite, mask, boundary = self.paste(target, patch_info, rec)
                result, _ = self.harmonize(
                    composite, mask, boundary,
                    blend_method=blend_method,
                    use_diffusion=use_diffusion
                )
                
                entry = {
                    'image': result,
                    'mask': mask,
                    'source_idx': patch_info['source_idx'],
                    'target_idx': target_idx,
                    'score': rec['score']
                }
                results.append(entry)
                
                # Salvar se output_dir
                if output_dir:
                    result.save(out_path / 'images' / f'aug_{i:04d}.png')
                    mask.save(out_path / 'masks' / f'aug_{i:04d}.png')
                
                if old_verbose and (i + 1) % 10 == 0:
                    print(f"  Geradas {i+1}/{n} imagens...")
                    
            except Exception as e:
                if old_verbose: print(f"  Erro na imagem {i}: {e}")
                continue
        
        self.verbose = old_verbose
        if self.verbose:
            print(f"\nBatch concluído: {len(results)}/{n} imagens geradas com sucesso")
        
        return results
    
    # ─── UTILITÁRIOS ────────────────────────────────────────────────
    
    def load_dataset_images(
        self,
        data_root: str,
        max_images: int = 100,
        class_name: Optional[str] = None
    ) -> Tuple[List[Image.Image], List[Image.Image]]:
        """
        Carrega imagens e máscaras de um dataset.
        
        Returns:
            (images, masks) — listas de PIL.Image
        """
        root = Path(data_root)
        
        # Detecta estrutura: data/class/class/images ou data/class/images
        if class_name:
            candidates = [root / class_name / class_name, root / class_name, root]
        else:
            candidates = [root]
        
        img_dir = None
        for c in candidates:
            if (c / 'images').is_dir():
                img_dir = c / 'images'
                mask_dir = c / 'masks'
                break
        
        if img_dir is None:
            raise ValueError(f"Estrutura de dataset não encontrada em {data_root}")
        
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        img_files = []
        for ext in exts:
            img_files.extend(sorted(img_dir.glob(ext)))
        
        img_files = img_files[:max_images]
        
        images, masks = [], []
        for img_path in img_files:
            # Procura máscara correspondente
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                mask_path = mask_dir / (img_path.stem + '.png')
            
            if mask_path.exists():
                images.append(Image.open(img_path).convert('RGB'))
                masks.append(Image.open(mask_path).convert('L'))
        
        if self.verbose:
            print(f"Carregadas {len(images)} imagens de {img_dir}")
        
        return images, masks
    
    def memory_status(self) -> Dict:
        """Retorna estado da memória GPU."""
        if not torch.cuda.is_available():
            return {'device': 'cpu'}
        
        return {
            'device': str(self.device),
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        }
