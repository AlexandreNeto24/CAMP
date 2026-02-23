#!/usr/bin/env python
"""
CIM Data Augmentation Generator — CLI.

Gera imagens aumentadas usando o pipeline completo:
  Segmentor → Recommender → Paste → DoveNet → Hybrid Blend + SD

Uso:
    python scripts/generate.py --class metaplasia --num 100 --output data/augmented/metaplasia/
    python scripts/generate.py --class polyps --num 50 --no-sd --seed 42
    python scripts/generate.py --class metaplasia --num 10 --blend alpha
"""

import argparse
import sys
import time
from pathlib import Path

# Garantir que o projecto está no path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description='CIM Pipeline — Geração de imagens aumentadas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python scripts/generate.py --class metaplasia --num 100
  python scripts/generate.py --class polyps --num 50 --no-sd --seed 42
  python scripts/generate.py --class metaplasia --blend hybrid --output ./augmented/
        """
    )
    
    parser.add_argument('--class-name', '--class', dest='class_name', type=str, required=True,
                        help='Classe alvo (ex: metaplasia, polyps, airplane)')
    parser.add_argument('--num', '-n', type=int, default=10,
                        help='Número de imagens a gerar (default: 10)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Directório de output (default: data/augmented/<class>/)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directório base dos datasets (default: data)')
    parser.add_argument('--target-dir', type=str, default=None,
                        help='Diretório de imagens alvo (normal). Se não definido, usa data/<class>/<class>/')
    parser.add_argument('--source-dir', type=str, default=None,
                        help='Diretório de imagens source (patches). Se não definido, usa data/<class>/<class>/')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                        help='Directório com pesos treinados (default: experiments)')
    
    # Pipeline options
    parser.add_argument('--blend', type=str, default='hybrid',
                        choices=['alpha', 'poisson', 'hybrid'],
                        help='Método de blending (default: hybrid)')
    parser.add_argument('--no-sd', action='store_true',
                        help='Desactivar SD Inpainting (mais rápido, menos VRAM)')
    parser.add_argument('--no-tto', action='store_true',
                        help='Desactivar Test-Time Optimization (mais rápido)')
    parser.add_argument('--encoder', type=str, default='efficientnet-b3',
                        help='Encoder do segmentador (default: efficientnet-b3)')
    
    # Reproduzibilidade
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed para reproduzibilidade')
    parser.add_argument('--max-source', type=int, default=50,
                        help='Máximo de imagens source a usar (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')

    # Debug / Visualização
    parser.add_argument('--save-intermediates', type=int, default=0, metavar='N',
                        help='Guardar N imagens com passos intermédios do pipeline (default: 0)')
    parser.add_argument('--debug', action='store_true',
                        help='Activar debug (equivale a --save-intermediates 5)')

    args = parser.parse_args()
    if args.debug and args.save_intermediates == 0:
        args.save_intermediates = 5
    return args


def main():
    args = parse_args()
    
    from src.pipeline import CIMPipeline
    
    # ── Output dir ──
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / 'data' / 'augmented' / args.class_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}")
    
    # ── Carregar pipeline ──
    pipe = CIMPipeline.from_pretrained(
        class_name=args.class_name,
        experiments_dir=args.experiments_dir,
        project_root=str(project_root),
        device=args.device,
        load_sd=not args.no_sd,
        segmentor_encoder=args.encoder
    )
    
    # ── Carregar datasets separados para target e source ──
    print(f"\nCarregando datasets...")
    data_root = str(project_root / args.data_dir)

    # Diretório de targets (imagens onde colar)
    if args.target_dir:
        target_dir = Path(args.target_dir)
    else:
        target_dir = project_root / args.data_dir / args.class_name / 'normal'
        if not target_dir.exists():
            target_dir = project_root / args.data_dir / args.class_name / args.class_name

    # Diretório de sources (imagens de onde extrair patches)
    if args.source_dir:
        source_dir = Path(args.source_dir)
    else:
        source_dir = project_root / args.data_dir / args.class_name / args.class_name

    print(f"  Target dir: {target_dir}")
    print(f"  Source dir: {source_dir}")

    # Carregar imagens alvo (sem máscaras)
    def load_images_from_dir(img_dir, max_images=None):
        from PIL import Image
        img_dir = Path(img_dir)
        images = []
        exts = ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.PNG']
        img_files = []
        for ext in exts:
            img_files.extend(sorted(img_dir.glob(ext)))
        img_files = sorted(set(img_files))
        for img_path in img_files:
            try:
                images.append(Image.open(img_path).convert('RGB'))
            except Exception:
                continue
            if max_images and len(images) >= max_images:
                break
        return images

    # Carregar imagens source e máscaras
    def load_images_and_masks_from_dir(img_dir, mask_dir=None, max_images=None):
        from PIL import Image
        img_dir = Path(img_dir)
        # Se não for especificado mask_dir, tenta ../masks se estamos em .../images
        if mask_dir is None:
            if img_dir.name == 'images' and (img_dir.parent / 'masks').exists():
                mask_dir = img_dir.parent / 'masks'
            else:
                mask_dir = img_dir / 'masks'
        images, masks = [], []
        exts = ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.PNG']
        img_files = []
        for ext in exts:
            img_files.extend(sorted(img_dir.glob(ext)))
        img_files = sorted(set(img_files))
        for img_path in img_files:
            try:
                images.append(Image.open(img_path).convert('RGB'))
                # Busca máscara pelo stem (prefixo), aceita qualquer extensão válida
                mask_candidate = None
                if mask_dir.exists():
                    for ext in exts:
                        candidates = list(mask_dir.glob(f"{img_path.stem}*{ext[1:]}") )
                        if candidates:
                            mask_candidate = candidates[0]
                            break
                if mask_candidate and mask_candidate.exists():
                    masks.append(Image.open(mask_candidate).convert('L'))
                else:
                    masks.append(None)
            except Exception:
                continue
            if max_images and len(images) >= max_images:
                break
        return images, masks

    # Carregar targets (tenta também <dir>/images/ se necessário)
    target_images = load_images_from_dir(target_dir, max_images=args.max_source)
    if len(target_images) < 1:
        images_subdir = target_dir / 'images'
        if images_subdir.exists():
            target_images = load_images_from_dir(images_subdir, max_images=args.max_source)
    if len(target_images) < 1:
        print(f"Erro: precisa de pelo menos 1 imagem alvo em {target_dir} ou {target_dir}/images/")
        sys.exit(1)

    # Carregar sources e máscaras (tenta também <dir>/images/ se necessário)
    source_images, source_masks = load_images_and_masks_from_dir(source_dir, max_images=args.max_source)
    if len(source_images) < 1:
        images_subdir = source_dir / 'images'
        if images_subdir.exists():
            source_images, source_masks = load_images_and_masks_from_dir(images_subdir, max_images=args.max_source)
    if len(source_images) < 1:
        print(f"Erro: precisa de pelo menos 1 imagem source em {source_dir} ou {source_dir}/images/")
        sys.exit(1)
    # Garante que só imagens de instância e máscaras de instância são usadas
    # Nunca usar contexto, patch_with_context ou anel em nenhuma chamada do pipeline
    
    # ── Gerar ──
    print(f"\nGerando {args.num} imagens (blend={args.blend}, sd={'on' if not args.no_sd else 'off'})...\n")
    
    t0 = time.time()

    # ── Geração com intermédios (primeiras N) + batch (resto) ──
    n_inter = min(args.save_intermediates, args.num)
    results = []
    intermediates_list = []

    if n_inter > 0:
        import random
        rng = random.Random(args.seed)
        viz_dir = output_dir / 'intermediates'
        viz_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Guardando {n_inter} imagens com passos intermédios em {viz_dir}")

        for idx in range(n_inter):
            src_idx = rng.randint(0, len(source_images) - 1)
            tgt_idx = rng.randint(0, len(target_images) - 1)
            result_img, inter = pipe.generate(
                target_image=target_images[tgt_idx],
                source_image=source_images[src_idx],
                source_mask=source_masks[src_idx] if source_masks else None,
                blend_method=args.blend,
                use_diffusion=not args.no_sd,
                return_intermediates=True
            )
            results.append({'image': result_img, 'mask': inter.get('mask')})
            intermediates_list.append(inter)

            # Salvar imagem final
            img_dir = output_dir / 'images'
            msk_dir = output_dir / 'masks'
            img_dir.mkdir(parents=True, exist_ok=True)
            msk_dir.mkdir(parents=True, exist_ok=True)
            result_img.save(img_dir / f'aug_{idx:04d}.png')
            if inter.get('mask'):
                inter['mask'].save(msk_dir / f'aug_{idx:04d}.png')

        # Gerar visualizações dos intermédios
        _save_intermediates_viz(intermediates_list, results, viz_dir)

    # Batch para o resto
    remaining = args.num - n_inter
    if remaining > 0:
        batch_results = pipe.generate_batch(
            target_images=target_images,
            source_images=source_images,
            source_masks=source_masks,
            n=remaining,
            output_dir=str(output_dir),
            blend_method=args.blend,
            use_diffusion=not args.no_sd,
            seed=args.seed
        )
        results.extend(batch_results)
    
    elapsed = time.time() - t0
    
    # ── Sumário ──
    print(f"\n{'='*50}")
    print(f"  Concluído: {len(results)}/{args.num} imagens")
    print(f"  Tempo: {elapsed:.1f}s ({elapsed/max(len(results),1):.1f}s/imagem)")
    print(f"  Output: {output_dir}")
    print(f"  Imagens: {output_dir / 'images'}")
    print(f"  Máscaras: {output_dir / 'masks'}")
    
    mem = pipe.memory_status()
    if 'allocated_gb' in mem:
        print(f"  VRAM pico: {mem['max_allocated_gb']:.1f} GB")
    if args.save_intermediates > 0:
        print(f"  Intermédios: {output_dir / 'intermediates'}")
    print(f"{'='*50}")


def _save_intermediates_viz(intermediates_list, results, viz_dir):
    """Gera visualizações dos passos intermédios do pipeline."""
    from src.blend.viz import plot_pipeline_steps, plot_difference_maps, plot_pipeline_zoom
    from src.viz_utils import save_figure

    for idx, (inter, res) in enumerate(zip(intermediates_list, results)):
        composite = inter.get('composite')
        dovenet_out = inter.get('dovenet_output')
        final = res.get('image') if isinstance(res, dict) else res
        mask = inter.get('mask')
        ring_mask = inter.get('ring_mask')

        # Pipeline steps (5-column)
        if composite and dovenet_out and final:
            try:
                fig = plot_pipeline_steps(
                    composite=composite,
                    dovenet_out=dovenet_out,
                    blend_only=composite,  # pre-SD = composite
                    ring_mask=ring_mask or mask,
                    final_result=final
                )
                save_figure(fig, viz_dir / f'pipeline_{idx:02d}.png')
            except Exception as e:
                print(f"    ✗ pipeline_{idx:02d}.png falhou: {e}")

        # Difference maps
        if composite and dovenet_out and final:
            try:
                fig, stats = plot_difference_maps(
                    composite=composite,
                    dovenet_out=dovenet_out,
                    blend_only=composite,
                    final_result=final
                )
                save_figure(fig, viz_dir / f'diffmap_{idx:02d}.png')
            except Exception as e:
                print(f"    ✗ diffmap_{idx:02d}.png falhou: {e}")

        # Zoom crops
        if composite and dovenet_out and final and mask:
            try:
                fig = plot_pipeline_zoom(
                    composite=composite,
                    dovenet_out=dovenet_out,
                    blend_only=composite,
                    final_result=final,
                    mask=mask
                )
                save_figure(fig, viz_dir / f'zoom_{idx:02d}.png')
            except Exception as e:
                print(f"    ✗ zoom_{idx:02d}.png falhou: {e}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
