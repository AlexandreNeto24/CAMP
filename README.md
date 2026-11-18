# CAMP: Context-Aware Mixing and Pasting

CAMP is an image-augmentation framework. It copies an instance from a source image, chooses where that instance belongs on a target image, and blends it in. The pipeline has three steps, top to bottom in the figure.

1. **Segmentor.** Context extraction and feature encoding. The ring around the instance and the target image go through one shared feature extractor.
2. **Recommender.** Cross-attention spatial recommendation. Multi-head cross-attention reads that context against the target, and a decoder turns the result into a placement heatmap.
3. **Generate.** Paste and blend. The instance is inserted at the chosen box, and the boundary is harmonised.

![CAMP pipeline. 1 Segmentor: context extraction and feature encoding. 2 Recommender: cross-attention spatial recommendation. 3 Generate: paste and blend.](docs/camp-pipeline.png)

Step 3 builds a closed-form alpha matte, restores the unmultiplied foreground, transfers colour locally in CIELAB (Reinhard, luminance weight 0.72), and composites the frequencies with a 4-level Laplacian pyramid. The blend notes are in [`camp/blend/README.md`](camp/blend/README.md).

CAMP is used for natural images and for medical images. Clinical datasets are not included in this repository. You bring your own images and masks.

The public entry point is `CAMP` in the `camp` package.

## Setup

From the repository root, create the virtual environment once with [`environment.yml`](environment.yml). This is the only install file. It is the same on Windows, Linux, and macOS, and the environment is named `CAMP`. Conda installs the PyTorch build for that machine: CUDA 12.4 on Windows and on 64-bit Linux, and the CPU or MPS build on macOS and on Linux ARM.

**Shell**

```bash
conda env create -f environment.yml
conda activate CAMP
pip install -e .
```

`pip install -e .` is run inside that conda environment. It does not replace the PyTorch build conda installed. After `pip install -e .`, `python -m camp` and `camp` are the commands. `python -m camp` is the form to copy. `camp` works when the CAMP environment is active. If `pyproject.toml` just changed, run `pip install -e .` again so the `camp` command appears.

Training and generation use the GPU when you pass nothing (`--device cuda`). If CUDA was requested and this machine has no CUDA device, CAMP uses the CPU. On a machine without an NVIDIA GPU you can pass `--device cpu`. On Apple silicon you can pass `--device mps`.

Albumentations stays on 1.3.1 because the dataset calls `GaussNoise(var_limit=...)`, which 2.x removed. NumPy stays below 2, and OpenCV stays below 4.10, so the same file resolves on a new machine.

## Data layout

Bring one class under `data/`. The folder name is the value you pass to `--class`. `--data-dir` defaults to `data` and has to be a folder named `data`. CAMP's project root is the parent of that folder. A relative `--data-dir` is resolved from the repository root. Training reads pairs of RGB images and grayscale masks from that class. A mask shares the image stem, with `.jpg` or `.jpeg` replaced by `.png`. Pairs whose mask is missing are skipped. `data/` and `weights/` are gitignored.

The tree below is the shape the trainers and `generate` look for. `<class>` is the folder name you pass to `--class`. `frame_001.jpg` is only there to show that it pairs with `frame_001.png`.

```
.
├── data/
│   ├── <class>/
│   │   ├── <class>/
│   │   │   ├── images/
│   │   │   │   ├── frame_001.jpg
│   │   │   │   └── frame_002.png
│   │   │   └── masks/
│   │   │       ├── frame_001.png
│   │   │       └── frame_002.png
│   │   └── normal/
│   │       └── images/
│   │           └── bg_001.jpg
│   └── augmented/
│       └── <class>/
│           ├── images/
│           │   └── aug_0000.png
│           └── masks/
│               └── aug_0000.png
└── weights/
    └── <class>/
        ├── segmentor/
        │   └── best.pth
        └── recommender/
            └── best.pth
```

`<class>/<class>` is the source: the images the instance is copied from, with masks beside them. `<class>/normal` is optional: the backgrounds the instance is pasted onto. Training writes `weights/`. Generation writes `data/augmented/`.

The same class can sit one level higher, with no nested folder of the same name. Backgrounds can sit in `normal/` itself when that folder has no `images/` inside it.

```
data/
└── <class>/
    ├── images/
    ├── masks/
    └── normal/
        └── images/
```

The trainers resolve the paired folder in this order:

1. `data/<class>/<class>/images`, with masks in the sibling `masks/`
2. `data/<class>/images`, with masks in the sibling `masks/`

Train and val are disjoint. `PairedMaskDataset` shuffles paired files with seed 42, cuts at `split_ratio` (default 0.2), and gives the head to train and the tail to val. Together the two splits cover every paired file. The same cut is used for the segmentor and the recommender.

Generation uses two folders:

- **Source** (where the instance is copied from): `data/<class>/<class>/` when that folder has `images/`, otherwise `data/<class>/`.
- **Target** (where it is pasted): `data/<class>/normal/` when that directory exists, otherwise the same folder as the source.

`--source-dir` and `--target-dir` override those defaults. If a directory has no images at the top level, the loader also looks in `<dir>/images`. Masks are read from a sibling `masks/` folder.

Default training resolution is 512×512. Patches are 128×128.

On the command line the flag uses hyphens (`--batch-size`). In Python the same setting is the keyword with underscores (`batch_size`). `--class` and `--class-name` are both `class_name`.

## CLI config file

Long flag lists are optional. [`configs/default.yaml`](configs/default.yaml) holds the defaults for `train segmentor`, `train recommender`, and `generate`. Edit that file once, then run short commands.

**Precedence:** a flag on the command line wins over the YAML. The YAML wins over the built-in fallbacks in the code.

### 1. Set the class (and anything else you care about)

Open `configs/default.yaml`. Under `common`, set the folder name that sits under `data/`:

```yaml
common:
  class_name: brain        # example only: the tutorial folder under data/
  data_dir: data
  device: cuda
  encoder: efficientnet-b3
```

Tune the three blocks below it as needed (`segmentor`, `recommender`, `generate`). Examples: `segmentor.epochs`, `recommender.batch_size`, `generate.num`, `generate.use_tto`.

### 2. Run without a long CLI

With `common.class_name` set, these are enough:

**CLI**

```bash
python -m camp train segmentor
python -m camp train recommender
python -m camp generate
```

You can still override one value for a single run:

**CLI**

```bash
python -m camp train segmentor --epochs 100 --batch-size 4
python -m camp generate --num 100 --seed 42
python -m camp generate --no-tto
```

`--class` still works if you prefer not to put the class in the YAML. `brain` here is only the tutorial name. Use the folder you placed under `data/`:

**CLI**

```bash
python -m camp train segmentor --class brain
```

If both the YAML and the CLI omit `class_name`, CAMP exits and points you at the config file.

### 3. Use another YAML (optional)

Copy `configs/default.yaml` (for a paper run, a CPU box, a quick smoke test) and point at it:

**CLI**

```bash
python -m camp train segmentor --config configs/smoke.yaml
python -m camp generate --config configs/paper.yaml --num 50
```

`--config` is resolved from the repository root when the path is relative. The Python API (`CAMP.train_segmentor`, `generate_batch`, …) does not read this file; it keeps its own keyword defaults. The YAML is only for the CLI.

## 1. Segmentor

This step is the context extraction and feature encoding in the figure. Train it before the recommender. The recommender freezes it and uses the encoder as its backbone.

The network is a UNet++. The encoder starts from ImageNet weights and is trained as a binary segmentation of the instance, with a Tversky focal loss that penalises missed boundary pixels. The first training stage keeps the encoder frozen and trains the decoder for `freeze_epochs`, at `lr`. The second stage unfreezes the encoder until `epochs` is reached, at `finetune_lr`. The checkpoint is `weights/<class>/segmentor/best.pth`, with `config.yaml` beside it.

**CLI**

```bash
python -m camp train segmentor
python -m camp train segmentor --class <class>
python -m camp train segmentor --class <class> --epochs 100 --batch-size 4 --freeze-epochs 30
```

**Python**

```python
from camp import CAMP

camp = CAMP(root=".", device="cuda")
camp.train_segmentor("<class>")
camp.train_segmentor("<class>", epochs=100, batch_size=4, freeze_epochs=30)
```

| Argument | Default | Role |
| --- | --- | --- |
| `class_name` | YAML `common.class_name` or `--class` | folder name under `data/` |
| `epochs` | `80` | total length; the second stage is what remains after the first |
| `freeze_epochs` | `30` | first stage, encoder frozen |
| `lr` | `3e-4` | first stage |
| `finetune_lr` | `1e-4` | second stage |

Defaults below also live in `configs/default.yaml`. Other `train_segmentor` arguments: `encoder` `efficientnet-b3`, `decoder_dropout` `0.4`, `image_size` `512`, `patch_size` `128`, `batch_size` `8`, `accumulation_steps` `2`, `patience` `25`, `split_ratio` `0.2`, `output` `weights/<class>/segmentor/`, `device` `cuda`, `debug` `false`. `--debug` and `debug=True` write the same 8 validation figures at the end to `debug/` beside `best.pth`: an overview, then per image the probability, the thresholded prediction, target and prediction contours, and the error map, with `index.txt` listing the metrics. `--data-dir` is the `data` folder; `CAMP(root=)` is its parent.

## 2. Recommender

This step is the cross-attention spatial recommendation in the figure. It needs the segmentor checkpoint. If `segmentor_weights` is omitted, CAMP looks for `weights/<class>/segmentor/best.pth`, then `best_segmentor.pth`. A missing file raises `FileNotFoundError`.

It attends from the source context, the tissue around the instance rather than the instance interior, onto the target image. The output is a placement heatmap. Peaks on tissue that already holds an instance are suppressed, and the remaining peaks become the boxes used in step 3. The first training stage runs for `epochs` at `lr`. The second runs for `finetune_epochs` at `finetune_lr`. `finetune_epochs=0` skips the second stage. The result is `weights/<class>/recommender/best.pth`.

**CLI**

```bash
python -m camp train recommender
python -m camp train recommender --class <class>
python -m camp train recommender --class <class> --epochs 60 --finetune-epochs 20 --segmentor-weights weights/<class>/segmentor/best.pth
```

**Python**

```python
from camp import CAMP

camp = CAMP(root=".", device="cuda")
camp.train_recommender("<class>")
camp.train_recommender(
    "<class>",
    epochs=60,
    finetune_epochs=20,
    segmentor_weights="weights/<class>/segmentor/best.pth",
)
```

| Argument | Default | Role |
| --- | --- | --- |
| `class_name` | YAML `common.class_name` or `--class` | same folder as the segmentor |
| `epochs` | `60` | first stage |
| `lr` | `3e-4` | first stage |
| `finetune_epochs` | `20` | second stage; `0` skips it |
| `finetune_lr` | `5e-5` | second stage |
| `segmentor_weights` | auto | checkpoint from the previous step |

Defaults below also live in `configs/default.yaml`. Other `train_recommender` arguments: `encoder` `efficientnet-b3`, `embedding_dim` `2048`, `num_heads` `16`, `dropout` `0.10`, `image_size` `512`, `patch_size` `128`, `batch_size` `4`, `accumulation_steps` `4`, `patience` `30`, `split_ratio` `0.2`, `output` `weights/<class>/recommender/`, `device` `cuda`, `debug` `false`. `--debug` and `debug=True` draw the same kind of spread for the recommender into its own `debug/` beside `best.pth`: instance box, training target, context crop, target heatmap against the prediction, and the placement boxes. `--data-dir` works as it does for the segmentor.

## 3. Generate

This step is the paste and blend at the bottom of the figure. It loads both checkpoints:

- `weights/<class>/segmentor/best.pth` (or `best_segmentor.pth`)
- `weights/<class>/recommender/best.pth` (or `best_model.pt`)

A missing file makes `load` raise `FileNotFoundError`. For each sample, CAMP extracts the instance from the source mask, or from the segmentor when no mask is given. It reads a box from the recommender heatmap and can search scale, rotation, and flip before blending. Images and masks are written to `data/augmented/<class>/images` and `data/augmented/<class>/masks` as `aug_0000.png`, `aug_0001.png`, and so on.

The CLI draws a batch. `use_tto` searches scale, rotation, and flip. It is on for the command unless you pass `--no-tto`. In `generate_batch` it is off unless you pass `use_tto=True`.

**CLI**

```bash
python -m camp generate
python -m camp generate --class <class> --num 100
python -m camp generate --class <class> --num 50 --no-tto --source-dir data/<class>/<class> --target-dir data/<class>/normal --seed 42
```

**Python**

```python
from PIL import Image
from camp import CAMP

camp = CAMP(root=".", device="cuda")
camp.load("<class>")

target_images = [Image.open("data/<class>/normal/images/bg_001.jpg").convert("RGB")]
source_images = [Image.open("data/<class>/<class>/images/frame_001.jpg").convert("RGB")]
source_masks = [Image.open("data/<class>/<class>/masks/frame_001.png").convert("L")]

samples = camp.generate_batch(
    target_images,
    source_images,
    masks=source_masks,
    n=100,
    out="data/augmented/<class>",
    use_tto=True,
)
samples = camp.generate_batch(
    target_images,
    source_images,
    masks=source_masks,
    n=50,
    out="data/augmented/<class>",
    use_tto=False,
    seed=42,
)
```

| Argument | Default | Role |
| --- | --- | --- |
| `class_name` | YAML `common.class_name` or `--class` | weights and folders for this class |
| `n` | `10` | how many images (`--num` / YAML `generate.num`) |
| `use_tto` | on in the CLI (`generate.use_tto`), off in `generate_batch` | scale, rotation, and flip search |
| `seed` | none | sampling order |
| `out` | `data/augmented/<class>/` | output directory (`--output`) |

Defaults below also live in `configs/default.yaml`. Other generate settings: `--encoder` `efficientnet-b3` (passed to `load`), `--source-dir` and `--target-dir` (the data-layout defaults), `--max-source` `50`, `--debug` (5 full-pipeline strips for the first samples in the same batch, written to `<output>/debug/aug_XXXX.png`: segmentor prediction, the mask that was cut, recommender heatmap and boxes, extraction, alpha, harmonization, and the blend), `--device` `cuda`, `--data-dir` `data`. The command leaves `image_size` at `(512, 512)`.

## Python API

`generate` is the one-image call. The batch above is what the CLI runs. `generate` returns a `Sample`: `image` is the blended RGB image, `mask` is the instance mask, `score` is the recommendation score. `source_mask=None` makes the segmentor build the mask. Call `load` before `extract`, `recommend`, `blend`, `generate`, or `generate_batch`.

**Python**

```python
sample = camp.generate(
    target_images[0],
    source_images[0],
    source_mask=source_masks[0],
    use_tto=True,
)
sample.image.save("aug.png")
sample.mask.save("aug_mask.png")
```

`generate(..., patch_index=0, recommendation_index=0, top_k=5, use_tto=True)`. Inside `generate_batch`, `top_k` is fixed at `3`. `extract` accepts `min_area=200` and `max_patches=10`. `recommend` accepts `top_k` and `use_tto`. None of these have a CLI flag.

`load(class_name, encoder="efficientnet-b3")` rebuilds the recommender with embedding 2048, 16 heads, and dropout 0.3 before reading the checkpoint. Those three are not arguments of `load`. Training defaults are embedding 2048, 16 heads, and dropout 0.10. A run that changed `embedding_dim` or `num_heads` cannot pass those into `load`.

`CAMP(root=".", device="cuda", verbose=True, image_size=(512, 512), core_strength=0.58, edge_strength=0.88, l_weight=0.72)`. `core_strength`, `edge_strength`, and `l_weight` set the blend and have no CLI flag. Training also passes `image_size` into `train_segmentor` and `train_recommender`. Generation leaves it at `(512, 512)`.

## Example

[examples/brain/tutorial.ipynb](examples/brain/tutorial.ipynb) is a worked run on a public brain MRI set. It imports `CAMP`, trains the segmentor and the recommender, and generates four images. The notebook is already executed. The figures from that run are in [examples/brain/outputs](examples/brain/outputs). The dataset, the checkpoints, and the generated images stay on the machine that runs the training, under `data/brain/`, `weights/brain/`, and `data/augmented/brain/`.

![One generated brain slice: segmentation, placement heatmap, and the paste and blend.](examples/brain/outputs/generate_00.png)

![A second generated slice, with the same three stages.](examples/brain/outputs/generate_01.png)

The example uses 40 segmentor epochs, the first 15 with the encoder frozen, and 30 recommender epochs plus 10 of fine-tuning. Those numbers are in [`configs/brain.yaml`](configs/brain.yaml). The longer defaults are in [`configs/default.yaml`](configs/default.yaml).

**CLI**

```bash
python -m camp train segmentor --config configs/brain.yaml
python -m camp train recommender --config configs/brain.yaml
python -m camp generate --config configs/brain.yaml
```

Re-running the notebook needs `jupyter` and `ipykernel` in the same environment as CAMP. `environment.yml` does not install them.

## Code map

| Path | Role |
| --- | --- |
| `camp/api.py` | `CAMP`: train, load, extract, recommend, blend, generate |
| `camp/sample.py` | `Sample` returned by `generate` and `generate_batch` |
| `docs/camp-pipeline.png` | Pipeline figure: context extraction, cross-attention, paste and blend |
| `camp/cli/__init__.py` | Command line: train segmentor, train recommender, generate |
| `camp/cli/config.py` | Load and merge `configs/*.yaml` defaults for the CLI |
| `configs/default.yaml` | Editable defaults for segmentor, recommender, and generate |
| `configs/brain.yaml` | Brain example: 40 segmentor epochs and 30+10 recommender epochs |
| `examples/brain/tutorial.ipynb` | Executed brain tutorial |
| `camp/__main__.py` | `python -m camp` |
| `camp/cli/viz.py` | Debug figures for `--debug` (train and generate) |
| `camp/models/` | Segmentor and recommender |
| `camp/blend/` | Matting, color transfer, and pyramid blend |
| `camp/blend/README.md` | Notes on the four blend steps |
| `camp/figures.py` | Stage strips for the segmentor, the recommender, and generate |
| `camp/attention_dataset.py` | Pairs each image with its mask for training |

## License

CAMP is released under the MIT license. See [`LICENSE`](LICENSE).
