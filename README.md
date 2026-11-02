# CAMP

CAMP is an image-augmentation framework. It copies an instance from a source image, chooses where that instance belongs on a target image, and blends it in.

CAMP runs in three phases:

1. **Extract.** A UNet++ segmentor (`efficientnet-b3` by default) finds the instance. Connected components become patches.
2. **Recommend.** An attention model scores where the patch should be pasted and returns a heatmap and boxes.
3. **Blend.** Seamless blending mattes the patch, transfers color locally (Reinhard), and composites it with a Laplacian pyramid. The four blend steps live in [`src/blend/README.md`](src/blend/README.md).

CAMP is used for natural images and for medical images. Clinical datasets are not included in this repository. You bring your own images and masks.

The Python entry point is `CIMPipeline` in `src/cim_pipeline`.

## Setup

Run every command from the repository root (the directory that contains `src/` and `scripts/`). The scripts import `src` before they change `sys.path`, so the root has to be on `PYTHONPATH`.

```bash
conda activate cim_yolo
export PYTHONPATH="$PWD"
```

```powershell
conda activate cim_yolo
$env:PYTHONPATH = (Get-Location).Path
```

[`environment.yml`](environment.yml) is a Linux lock of that environment (Python 3.9, prefix `/home/win11/miniconda3/envs/cim_yolo`). Creating it with `conda env create` is reliable on Linux. On Windows, use an existing `cim_yolo` environment with the same libraries: PyTorch, torchvision, segmentation-models-pytorch, albumentations, OpenCV, Pillow, and PyYAML.

Training and generation default to `--device cuda`.

## Data layout

Training reads pairs of RGB images and grayscale masks. A mask shares the image stem, with `.jpg` or `.jpeg` replaced by `.png`. Pairs whose mask is missing are skipped.

```
data/<class>/<class>/images/          # source images (jpg or png)
data/<class>/<class>/masks/           # matching png masks
data/<class>/normal/images/           # optional backgrounds with no instance
weights/<class>/segmentor/best.pth
weights/<class>/recommender/best.pth
data/augmented/<class>/images/        # generated images
data/augmented/<class>/masks/         # generated masks
```

`data/` and `weights/` are gitignored.

The training scripts resolve the class folder in this order:

1. `data/<class>/<class>/images`
2. `data/<class>/images`

Generation uses two folders:

- **Source** (where the instance is copied from): `data/<class>/<class>/`, with `images/` and `masks/`.
- **Target** (where it is pasted): `data/<class>/normal/` when that directory exists, otherwise the same folder as the source.

`--source-dir` and `--target-dir` override those defaults. If a directory has no images at the top level, the loader also looks in `<dir>/images`. Masks are read from a sibling `masks/` folder.

Default training resolution is 512×512. Patches are 128×128.

The training commands below load every paired image into both the training loader and the validation loader. `AttentionDataset` can hold out 20% (`split='train'` / `split='val'`, seed 42) or read `train.txt` and `val.txt`, but the current CLIs do not pass those split settings. `--split-ratio` is recorded in `config.yaml` and does not change which files are used.

## Prepare a dataset

Place your own image/mask pairs in the tree above. This repository does not download metaplasia, polyps, brain, or breast data.

For a trial on natural images, [`notebooks/download_coco_subset.py`](notebooks/download_coco_subset.py) builds one class folder from COCO. The example below writes `data/airplane/airplane/images` and `data/airplane/airplane/masks`, which is the layout the trainers expect. Point `--output` at a directory inside this repo. The script's built-in default path is a machine-specific home directory.

```bash
python notebooks/download_coco_subset.py --output data/airplane --classes airplane --max-per-class 100
```

COCO annotations and the image zip are large. The script downloads them into the output directory.

## Train the segmentor

Train the segmentor before the recommender. The recommender freezes it and uses it as the backbone.

Phase 1 trains the decoder with the encoder frozen (`--freeze-epochs`, default 30). Phase 2 unfreezes the encoder for the remaining epochs (`--epochs`, default 80). Checkpoints go to `weights/<class>/segmentor/`. The best weights are copied to `best.pth`, and the run writes `config.yaml` next to them.

```bash
python scripts/train_segmentor.py --class metaplasia
python scripts/train_segmentor.py --class polyps --epochs 100 --batch-size 4 --freeze-epochs 30
```

Defaults that matter: encoder `efficientnet-b3`, image size 512, patch size 128, batch size 8, phase-1 learning rate `3e-4`, phase-2 learning rate `1e-4`. `--save-viz N` writes N debug figures at the end of training. Run `python scripts/train_segmentor.py --help` for the rest.

## Train the recommender

The recommender needs the segmentor weights. If you omit `--segmentor-weights`, the script looks for `weights/<class>/segmentor/best.pth` (then `best_segmentor.pth`). Without those files it falls back to ImageNet features.

Phase 1 runs for `--epochs` (default 60). Phase 2 fine-tunes for `--finetune-epochs` (default 20). Checkpoints go to `weights/<class>/recommender/best.pth`.

```bash
python scripts/train_recommender.py --class metaplasia
python scripts/train_recommender.py --class metaplasia --epochs 60 --finetune-epochs 20 --segmentor-weights weights/metaplasia/segmentor/best.pth
```

Defaults that matter: embedding size 2048, 16 attention heads, batch size 4, phase-1 learning rate `3e-4`, fine-tune learning rate `5e-5`. `--save-viz N` writes debug figures. Run `python scripts/train_recommender.py --help` for the rest.

## Generate images

Generation loads both checkpoints:

- `weights/<class>/segmentor/best.pth`
- `weights/<class>/recommender/best.pth`

If either file is missing, CAMP looks under `experiments/` for an older run directory. A missing segmentor falls back to ImageNet features. A missing recommender still runs, with an untrained head.

```bash
python scripts/generate.py --class metaplasia --num 100
python scripts/generate.py --class metaplasia --num 50 --source-dir data/metaplasia/metaplasia --target-dir data/metaplasia/normal --seed 42
```

Images and masks are written to `data/augmented/<class>/images` and `data/augmented/<class>/masks` as `aug_0000.png`, `aug_0001.png`, and so on. `--output` changes that directory. `--max-source` (default 50) caps how many source and target images are loaded. `--seed` fixes the batch sampling order.

The command-line generator always uses seamless blending, and its batch path calls the recommender with test-time search turned off. `--blend`, `--no-sd`, and `--no-tto` are accepted and do not change that path.

## Python API

`CIMPipeline.from_pretrained` loads the segmentor and the recommender for one class. `generate` returns a dict with the blended image and its mask. `use_tto=True` turns on test-time search over scale, rotation, and flip. That search is available on this call. The batch helper below leaves it off, matching the CLI.

```python
from PIL import Image
from src.cim_pipeline.pipeline import CIMPipeline

pipe = CIMPipeline.from_pretrained(
    "metaplasia",
    project_root=".",
    device="cuda",
    verbose=True,
)
target_image = Image.open("data/metaplasia/normal/images/target.png").convert("RGB")
source_image = Image.open("data/metaplasia/metaplasia/images/source.png").convert("RGB")
source_mask = Image.open("data/metaplasia/metaplasia/masks/source.png").convert("L")

result = pipe.generate(
    target_image,
    source_image,
    source_mask=source_mask,
    use_tto=True,
)
result["image"].save("aug.png")
result["mask"].save("aug_mask.png")
```

`generate_batch` draws `n` random source/target pairs, writes `images/` and `masks/` when `output_dir` is set, and returns a list of dicts:

```python
pipe.generate_batch(
    target_images,
    source_images,
    source_masks,
    n=10,
    output_dir="data/augmented/metaplasia",
    seed=42,
)
```

If `source_mask` is omitted, the segmentor builds the mask.

## Code map

| Path | Role |
| --- | --- |
| `scripts/train_segmentor.py` | Train the segmentor |
| `scripts/train_recommender.py` | Train the recommender on a frozen segmentor |
| `scripts/generate.py` | Run CAMP and write augmented images |
| `src/paste/` | Segmentor, recommender, dataset, and training loop |
| `src/cim_pipeline/` | `CIMPipeline`: extract, recommend, blend |
| `src/blend/` | Matting, color transfer, and pyramid blend |

## License

CAMP is released under the MIT license. See [`LICENSE`](LICENSE).
