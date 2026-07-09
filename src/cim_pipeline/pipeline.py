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
from PIL import Image
import torch
import random
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import cv2
from PIL import Image

from src.paste.segmentation_model import PretrainedUNet
from src.paste.models.attention_recommender import AttentionRecommender
from src.paste.inference.recommender import RecommendationEngine
from src.blend import SeamlessBlendingPipeline
from .loaders import (
    load_segmentor,
    load_recommender
)   


class CIMPipeline:
    def __init__(
        self,
        segmentor=None,
        recommender=None,
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
        
        # Transforms standard
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        
        if self.verbose:
            loaded = []
            if segmentor:  loaded.append('Segmentor')
            if recommender: loaded.append('Recommender')
            print(f"CIMPipeline inicializado ({self.device}) — Componentes: {', '.join(loaded) or 'nenhum'}")

    
    @classmethod
    def from_pretrained(
        cls,
        class_name: str,
        experiments_dir: str = 'experiments',
        project_root: str = '.',
        device: str = 'cuda',
        segmentor_encoder: str = 'efficientnet-b3',
        verbose: bool = False
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
        segmentor = load_segmentor(
            class_name, exp_dir, device, segmentor_encoder, verbose
        )
        
        # ─── 2. RECOMMENDER ─────────────────────────────────────────
        if verbose: print("\n[2/4] Recommender (AttentionRecommender)...")
        recommender = load_recommender(
            class_name, exp_dir, device, segmentor, verbose
        )

        if verbose:
            print(f"\n{'='*60}")
            vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            print(f"  Pipeline pronto — VRAM: {vram:.1f} GB")
            print(f"{'='*60}")
        
        return cls(
            segmentor=segmentor,
            recommender=recommender,
            device=device,
            verbose=verbose
        )
        
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
    
    # TODO: Otimizar o loader do Paste and Blend (SeamlessBlendingPipeline) 
     
    def blend(
        self, 
        target_image: Image.Image, 
        patch_info: Dict, 
        recommendation: Dict
    ) -> Tuple[Image.Image, Image.Image]:
        
        target_resized = target_image.resize(self.image_size, Image.BILINEAR)
        target_np = np.array(target_resized)
        
        self.seamless_blender = SeamlessBlendingPipeline(
            core_strength=0.58,
            edge_strength=0.88,
            l_weight=0.72
        )
        
        # 1. ENCONTRAR O CENTRO RECOMENDADO NO TARGET
        box = recommendation['box']
        box_center_x = int((box[0] + box[2]) / 2)
        box_center_y = int((box[1] + box[3]) / 2)
        
        # 2. USAR O PATCH COM CONTEXTO (A chave do sucesso do Notebook)
        # O contexto traz 30% de margem com tecido biológico real. Zero pixels pretos artificiais!
        ctx_pil = patch_info['patch_with_context']
        ctx_mask_t = patch_info['context_mask']
        
        ctx_np = np.array(ctx_pil)
        # O context_mask_t vem normalizado [0, 1], convertemos para [0, 255]
        ctx_mask_np = (ctx_mask_t.squeeze(0).numpy() * 255).astype(np.uint8)
        
        ctx_h, ctx_w = ctx_np.shape[:2]
        
        # 3. ENCONTRAR O CENTRO DA LESÃO DENTRO DO CONTEXTO
        # Como o contexto é maior, precisamos de saber onde a lesão está lá dentro
        coords = cv2.findNonZero(ctx_mask_np)
        if coords is not None:
            mx, my, mw, mh = cv2.boundingRect(coords)
            cx_internal = mx + (mw // 2)
            cy_internal = my + (mh // 2)
        else:
            cx_internal = ctx_w // 2
            cy_internal = ctx_h // 2
            
        # 4. ALINHAR GEOMETRICAMENTE NO TARGET
        # Posicionamos o contexto de forma a que a lesão caia exatamente no alvo do Recommender
        target_ctx_x1 = int(box_center_x - cx_internal)
        target_ctx_y1 = int(box_center_y - cy_internal)
        target_ctx_x2 = target_ctx_x1 + ctx_w
        target_ctx_y2 = target_ctx_y1 + ctx_h
        
        H, W = target_np.shape[:2]
        
        # Tratar os limites (clipping) caso a colagem saia fora da imagem
        img_x1 = max(0, target_ctx_x1)
        img_y1 = max(0, target_ctx_y1)
        img_x2 = min(W, target_ctx_x2)
        img_y2 = min(H, target_ctx_y2)
        
        if img_x2 <= img_x1 or img_y2 <= img_y1:
            # Proteção de segurança: se a recomendação for inválida/fora do ecrã
            return Image.fromarray(target_np), Image.fromarray(np.zeros((H, W), dtype=np.uint8))
            
        # Calcular as coordenadas de recorte para o contexto (caso tenha sofrido clipping)
        crop_x1 = img_x1 - target_ctx_x1
        crop_y1 = img_y1 - target_ctx_y1
        crop_x2 = crop_x1 + (img_x2 - img_x1)
        crop_y2 = crop_y1 + (img_y2 - img_y1)
        
        # Isolar os arrays finais para o blender
        ctx_np_cropped = ctx_np[crop_y1:crop_y2, crop_x1:crop_x2]
        ctx_mask_cropped = ctx_mask_np[crop_y1:crop_y2, crop_x1:crop_x2]
        bg_crop = target_np[img_y1:img_y2, img_x1:img_x2]
        
        # 5. O SEAMLESS BLENDING PURO
        # Agora o blender recebe tecido orgânico real nas margens. 
        # O Reinhard vai atuar perfeitamente e o Pyramid vai fundir sem deixar costuras.
        blend_results = self.seamless_blender.run(
            lesion_image=ctx_np_cropped,
            lesion_mask=ctx_mask_cropped,
            background=bg_crop,
            return_intermediate=False
        )
        
        blended_pixels = (blend_results['final'] * 255).astype(np.uint8)
        
        # 6. COLAGEM NA IMAGEM FINAL
        final_composite = target_np.copy()
        final_composite[img_y1:img_y2, img_x1:img_x2] = blended_pixels
        
        # Atualizar a máscara global
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[img_y1:img_y2, img_x1:img_x2] = (ctx_mask_cropped > 127).astype(np.uint8) * 255
        
        return Image.fromarray(final_composite), Image.fromarray(full_mask)
    
    ###############################################################
    # ─── Image generation ────────────────────────────────────────
    ###############################################################
    
    def generate(
        self,
        target_image: Image.Image,
        source_image: Image.Image,
        source_mask: Optional[Image.Image] = None,
        patch_index: int = 0,
        recommendation_index: int = 0,
        top_k: int = 5,
        use_tto: bool = True,
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
        if self.verbose: print("[1/3] Extracting patches...")
        patches = self.extract_patches(source_image, source_mask)
        if not patches:
            raise ValueError("Nenhum patch extraído da source image.")
        patch_info = patches[min(patch_index, len(patches) - 1)]

        # 2. Recomendar localização
        if self.verbose: print("[2/3] Recommending location...")
        recommendations = self.recommend(target_image, patch_info, top_k=top_k, use_tto=use_tto)
        if not recommendations:
            raise ValueError("Nenhuma recomendação gerada.")
        rec = recommendations[min(recommendation_index, len(recommendations) - 1)]
        # TODO: Adicionar o Paste and Blend ao pipeline principal 
        # 3. Paste and Blend (SeamlessBlendingPipeline)
        if self.verbose: print("[3/3] Paste and Blend...")
        img_final, mask_final = self.blend(target_image, patch_info, rec)
        
        
        results = {
            'image': img_final,
            'mask': mask_final
        }
        if self.verbose: print("─── Pipeline concluído ───\n")
        # if return_intermediates:
        #     intermediates = {
        #         'source_image': source_image,
        #         'target_image': target_image,
        #         'patches': patches,
        #         'patch_used': patch_info,
        #         'recommendations': recommendations,
        #         'recommendation_used': rec,
        #         'composite': composite,
        #         'mask': mask,
        #         'boundary': boundary,
        #     }
        #     return results, intermediates
        return results
    
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
                img_final, mask_final = self.blend(target, patch_info, rec)

                entry = {
                    'image': img_final,
                    'mask': mask_final,
                    'source_idx': patch_info['source_idx'],
                    'target_idx': target_idx,
                    'score': rec['score']
                }
                results.append(entry)
                
                # Salvar se output_dir
                if output_dir:
                    img_final.save(out_path / 'images' / f'aug_{i:04d}.png')
                    mask_final.save(out_path / 'masks' / f'aug_{i:04d}.png')
                
                if old_verbose and (i + 1) % 10 == 0:
                    print(f"  Geradas {i+1}/{n} imagens...")
                    
            except Exception as e:
                if old_verbose: print(f"  Erro na imagem {i}: {e}")
                continue
        
        self.verbose = old_verbose
        if self.verbose:
            print(f"\nBatch concluído: {len(results)}/{n} imagens geradas com sucesso")
        
        return results
    
    ##################################################################
    # ─── Utils ────────────────────────────────────────────────
    ##################################################################
    
    def load_dataset_images(
        self,
        data_root: str,
        max_images: int = 100,
        class_name: Optional[str] = None
    ) -> Tuple[List[Image.Image], List[Image.Image]]:
        """
        Carrega imagens e máscaras de um dataset.
        s
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