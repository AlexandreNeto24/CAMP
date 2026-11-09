# CAMP

CAMP is an image-augmentation framework. It copies an instance from a source image, chooses where that instance belongs on a target image, and blends it in.

CAMP runs in three phases:

1. **Extract.** A UNet++ segmentor (`efficientnet-b3` by default) finds the instance. Connected components become patches.
2. **Recommend.** An attention model scores where the patch should be pasted and returns a heatmap and boxes.
3. **Blend.** Seamless blending mattes the patch, transfers color locally (Reinhard), and composites it with a Laplacian pyramid. The four blend steps live in [`src/blend/README.md`](src/blend/README.md).

CAMP is used for natural images and for medical images. Clinical datasets are not included in this repository. You bring your own images and masks.

The public entry point is `CAMP` in the `camp` package.

## Setup

From the repository root, create the virtual environment once with [`environment.yml`](environment.yml). This is the only install file. It is the same on Windows, Linux, and macOS, and the environment is named `CAMP`. Conda installs the PyTorch build for that machine: CUDA 12.4 on Windows and on 64-bit Linux, and the CPU or MPS build on macOS and on Linux ARM.

```bash
conda env create -f environment.yml
conda activate CAMP
pip install -e .
```

`pip install -e .` is run inside that conda environment. It does not replace the PyTorch build conda installed. After that, `import camp` and the scripts below work from the repository root.

Training and generation use the GPU when you pass nothing (`--device cuda`). If CUDA was requested and this machine has no CUDA device, CAMP uses the CPU. On a machine without an NVIDIA GPU you can pass `--device cpu`. On Apple silicon you can pass `--device mps`.

Albumentations stays on 1.3.1 because the dataset calls `GaussNoise(var_limit=...)`, which 2.x removed. NumPy stays below 2, and OpenCV stays below 4.10, so the same file resolves on a new machine.

### Check the install

These commands train each stage for one or two epochs and write two augmented images. They check that the environment and the data layout work. They do not replace the training runs below. One epoch walks every paired image: there is no flag to take a subset.

```bash
python scripts/train_segmentor.py --class metaplasia --epochs 2 --freeze-epochs 1 --batch-size 2 --accumulation-steps 1 --image-size 256 --patience 5
python scripts/train_recommender.py --class metaplasia --epochs 1 --finetune-epochs 1 --batch-size 1 --accumulation-steps 1 --image-size 256 --patience 5
python scripts/generate.py --class metaplasia --num 2 --max-source 4 --seed 42
```

## Data layout

Training reads pairs of RGB images and grayscale masks. A mask shares the image stem, with `.jpg` or `.jpeg` replaced by `.png`. Pairs whose mask is missing are skipped.

```
data/<class>/<class>/images/          # source images (jpg or png)
data/<class>/<class>/masks/           # matching png masks
data/<class>/normal/                  # optional backgrounds, images in this folder
data/<class>/normal/images/           # or here, if normal/ itself has no images
weights/<class>/segmentor/best.pth
weights/<class>/recommender/best.pth
data/augmented/<class>/images/        # generated images
data/augmented/<class>/masks/         # generated masks
```

`data/` and `weights/` are gitignored.

The trainers resolve the class folder in this order:

1. `data/<class>/<class>/images`
2. `data/<class>/images`

Train and val are disjoint. `PairedMaskDataset` shuffles paired files with seed 42, cuts at `split_ratio` (default 0.2), and gives the head to train and the tail to val. Together the two splits cover every paired file. The same cut is used for the segmentor and the recommender.

Generation uses two folders:

- **Source** (where the instance is copied from): `data/<class>/<class>/` when that folder has `images/`, otherwise `data/<class>/`.
- **Target** (where it is pasted): `data/<class>/normal/` when that directory exists, otherwise the same folder as the source.

`--source-dir` and `--target-dir` override those defaults. If a directory has no images at the top level, the loader also looks in `<dir>/images`. Masks are read from a sibling `masks/` folder.

Default training resolution is 512×512. Patches are 128×128.

## Prepare a dataset

Place your own image/mask pairs in the tree above. This repository does not download metaplasia, polyps, brain, or breast data.

For a trial on natural images, [`notebooks/download_coco_subset.py`](notebooks/download_coco_subset.py) builds one class folder from COCO. The example below writes `data/airplane/airplane/images` and `data/airplane/airplane/masks`, which is the layout the trainers expect. Point `--output` at a directory inside this repo. The script's built-in default path is a machine-specific home directory.

```bash
python notebooks/download_coco_subset.py --output data/airplane --classes airplane --max-per-class 100
```

COCO annotations and the image zip are large. The script downloads them into the output directory. That script also needs `pycocotools`, which is not part of `environment.yml`.

## Train the segmentor

Train the segmentor before the recommender. The recommender freezes it and uses it as the backbone.

Phase 1 trains the decoder with the encoder frozen (`--freeze-epochs`, default 30). Phase 2 unfreezes the encoder for the remaining epochs (`--epochs`, default 80). Checkpoints go to `weights/<class>/segmentor/`. The best weights are copied to `best.pth`, and the run writes `config.yaml` next to them.

```bash
python scripts/train_segmentor.py --class metaplasia
python scripts/train_segmentor.py --class polyps --epochs 100 --batch-size 4 --freeze-epochs 30
```

Defaults that matter: encoder `efficientnet-b3`, image size 512, patch size 128, batch size 8, phase-1 learning rate `3e-4`, phase-2 learning rate `1e-4`. `--save-viz N` writes N debug figures at the end of training. Run `python scripts/train_segmentor.py --help` for the rest.

## Train the recommender

The recommender needs the segmentor checkpoint. If you omit `--segmentor-weights`, CAMP looks for `weights/<class>/segmentor/best.pth`, then `best_segmentor.pth`. A missing checkpoint raises `FileNotFoundError`.

Phase 1 runs for `--epochs` (default 60). Phase 2 fine-tunes for `--finetune-epochs` (default 20). Checkpoints go to `weights/<class>/recommender/best.pth`. `best_model.pt` from the trainer is copied to that name.

```bash
python scripts/train_recommender.py --class metaplasia
python scripts/train_recommender.py --class metaplasia --epochs 60 --finetune-epochs 20 --segmentor-weights weights/metaplasia/segmentor/best.pth
```

Defaults that matter: embedding size 2048, 16 attention heads, dropout 0.10, batch size 4, phase-1 learning rate `3e-4`, fine-tune learning rate `5e-5`. `--save-viz N` writes debug figures. Run `python scripts/train_recommender.py --help` for the rest.

## Generate images

Generation loads both checkpoints:

- `weights/<class>/segmentor/best.pth` (or `best_segmentor.pth`)
- `weights/<class>/recommender/best.pth` (or `best_model.pt`)

If either file is missing, `load` raises `FileNotFoundError`. There is no ImageNet fallback.

```bash
python scripts/generate.py --class metaplasia --num 100
python scripts/generate.py --class metaplasia --num 50 --no-tto --source-dir data/metaplasia/metaplasia --target-dir data/metaplasia/normal --seed 42
```

Images and masks are written to `data/augmented/<class>/images` and `data/augmented/<class>/masks` as `aug_0000.png`, `aug_0001.png`, and so on. `--output` changes that directory. `--max-source` (default 50) caps how many source and target images are loaded. `--seed` fixes the batch sampling order.

Test-time search over scale, rotation, and flip is on unless you pass `--no-tto`. That flag turns it off for the whole run, including the batch. Blending is always the seamless pipeline built with the `CAMP` strengths.

## Python API

`CAMP.load` reads the segmentor and recommender for one class and keeps one recommendation engine and one blending pipeline. `generate` returns a `Sample`. `Sample.image` is the blended RGB image and `Sample.mask` is the instance mask. `score` is the recommendation score. `use_tto=True` turns on test-time search. `generate_batch` leaves that search off unless you pass `use_tto=True`.

```python
from PIL import Image
from camp import CAMP

camp = CAMP(root=".", device="cuda", verbose=True)
camp.load("metaplasia")

target_image = Image.open("data/metaplasia/normal/images/target.png").convert("RGB")
source_image = Image.open("data/metaplasia/metaplasia/images/source.png").convert("RGB")
source_mask = Image.open("data/metaplasia/metaplasia/masks/source.png").convert("L")

sample = camp.generate(
    target_image,
    source_image,
    source_mask=source_mask,
    use_tto=True,
)
sample.image.save("aug.png")
sample.mask.save("aug_mask.png")
```

`generate_batch` draws `n` random source/target pairs. When `out` is set it writes `images/` and `masks/` as `aug_XXXX.png` and returns a list of `Sample`:

```python
samples = camp.generate_batch(
    target_images,
    source_images,
    masks=source_masks,
    n=10,
    out="data/augmented/metaplasia",
    use_tto=False,
    seed=42,
)
samples[0].image.save("one.png")
samples[0].mask.save("one_mask.png")
```

If `source_mask` is omitted, the segmentor builds the mask. Call `load` before `extract`, `recommend`, `blend`, `generate`, or `generate_batch`.

## Code map

| Path | Role |
| --- | --- |
| `camp/api.py` | `CAMP`: train, load, extract, recommend, blend, generate |
| `camp/sample.py` | `Sample` returned by `generate` and `generate_batch` |
| `scripts/train_segmentor.py` | Train the segmentor |
| `scripts/train_recommender.py` | Train the recommender on a frozen segmentor |
| `scripts/generate.py` | Run CAMP and write augmented images |
| `camp/models/` | Segmentor and recommender |
| `camp/blend/` | Matting, color transfer, and pyramid blend |
| `src/blend/README.md` | Notes on the four blend steps |
| `src/paste/viz.py` | Debug figures used by `--save-viz` |

## License

CAMP is released under the MIT license. See [`LICENSE`](LICENSE).
