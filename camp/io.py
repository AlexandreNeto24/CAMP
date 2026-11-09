"""Dataset folders and weight checkpoints for one class."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image

StrPath = Union[str, Path]

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")
_WEIGHT_FILES = {
    "segmentor": ("best.pth", "best_segmentor.pth"),
    "recommender": ("best.pth", "best_model.pt"),
}


def class_dirs(root: StrPath, class_name: str) -> Tuple[Path, Path]:
    """Return ``(source_dir, target_dir)`` for one class under ``root``.

    Source is ``root/data/<class>/<class>`` when ``images`` exists there,
    otherwise ``root/data/<class>``. Target is ``root/data/<class>/normal``
    when that directory exists, otherwise the source directory.
    """
    root = Path(root)
    nested = root / "data" / class_name / class_name
    flat = root / "data" / class_name
    source = nested if (nested / "images").is_dir() else flat
    normal = flat / "normal"
    target = normal if normal.is_dir() else source
    return source, target


def load_images(directory: StrPath, max_images: Optional[int] = None) -> List[Image.Image]:
    """Load RGB images from ``directory``.

    If that directory has no images, load from ``directory/images`` instead.
    """
    directory = Path(directory)
    images = _load_rgb(directory, max_images)
    if not images:
        nested = directory / "images"
        if nested.is_dir():
            images = _load_rgb(nested, max_images)
    return images


def load_image_mask_pairs(
    directory: StrPath,
    max_images: Optional[int] = None,
) -> Tuple[List[Image.Image], List[Optional[Image.Image]]]:
    """Load RGB images and grayscale masks.

    Uses the same ``directory/images`` fallback as ``load_images``. Masks are
    read from a sibling ``masks`` folder and matched by stem. A missing mask
    is ``None``.
    """
    directory = Path(directory)
    images, masks = _load_pairs(directory, max_images)
    if not images:
        nested = directory / "images"
        if nested.is_dir():
            images, masks = _load_pairs(nested, max_images)
    return images, masks


def find_weights(root: StrPath, class_name: str, kind: str) -> Path:
    """Return the checkpoint under ``root/weights/<class>/<kind>/``.

    ``kind`` is ``"segmentor"`` or ``"recommender"``. Segmentor files are
    ``best.pth`` then ``best_segmentor.pth``. Recommender files are
    ``best.pth`` then ``best_model.pt``.
    """
    try:
        names = _WEIGHT_FILES[kind]
    except KeyError:
        raise ValueError(
            f"kind must be 'segmentor' or 'recommender', got {kind!r}"
        ) from None
    weights_dir = Path(root) / "weights" / class_name / kind
    for name in names:
        path = weights_dir / name
        if path.is_file():
            return path
    tried = ", ".join(names)
    raise FileNotFoundError(
        f"No {kind} weights for {class_name!r} under {weights_dir} "
        f"(looked for {tried})"
    )


def _image_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    found = []
    for suffix in _IMAGE_SUFFIXES:
        found.extend(directory.glob(f"*{suffix}"))
    return sorted({path for path in found if path.is_file()})


def _mask_directory(image_dir: Path) -> Path:
    sibling = image_dir.parent / "masks"
    if image_dir.name == "images" and sibling.is_dir():
        return sibling
    return image_dir / "masks"


def _find_mask(mask_dir: Path, stem: str) -> Optional[Path]:
    if not mask_dir.is_dir():
        return None
    for suffix in _IMAGE_SUFFIXES:
        candidate = mask_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _read_mask(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L")


def _load_rgb(directory: Path, max_images: Optional[int]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in _image_files(directory):
        if max_images is not None and len(images) >= max_images:
            break
        try:
            images.append(_read_rgb(path))
        except Exception:
            continue
    return images


def _load_pairs(
    directory: Path,
    max_images: Optional[int],
) -> Tuple[List[Image.Image], List[Optional[Image.Image]]]:
    images: List[Image.Image] = []
    masks: List[Optional[Image.Image]] = []
    mask_dir = _mask_directory(directory)
    for path in _image_files(directory):
        if max_images is not None and len(images) >= max_images:
            break
        try:
            image = _read_rgb(path)
        except Exception:
            continue
        mask = None
        mask_path = _find_mask(mask_dir, path.stem)
        if mask_path is not None:
            try:
                mask = _read_mask(mask_path)
            except Exception:
                mask = None
        images.append(image)
        masks.append(mask)
    return images, masks
