"""Paired image/mask dataset with a real train/val split.

Samples match ``AttentionDataset``, so a trainer can batch them with
``attention_collate``.
"""

import random

from src.paste.attention_dataset import AttentionDataset, attention_collate

__all__ = ["PairedMaskDataset", "attention_collate"]


class PairedMaskDataset(AttentionDataset):
    """Image/mask pairs rooted at ``images/`` and ``masks/``.

    ``split`` is ``"train"``, ``"val"``, or ``"all"``. Train and val shuffle
    indices with ``random.seed(seed)``, then cut at
    ``split_idx = int(n * (1 - split_ratio))``. Train takes the head and val
    the tail, so the two splits are disjoint and together cover every paired
    file. ``"all"`` keeps every paired file.
    """

    def __init__(
        self,
        root,
        split="train",
        split_ratio=0.2,
        seed=42,
        image_size=(512, 512),
        patch_size=(128, 128),
        apply_aug=False,
        p_remove=0.0,
    ):
        if split not in ("train", "val", "all"):
            raise ValueError(
                f"split must be 'train', 'val', or 'all', got {split!r}"
            )
        self.seed = seed
        self.split_ratio = split_ratio
        # split_mode "none" skips AttentionDataset._split_by_ratio, which
        # keeps every image when split == "all" and ignores the caller's seed.
        super().__init__(
            root_path=root,
            image_size=image_size,
            patch_size=patch_size,
            apply_aug=apply_aug,
            split_ratio=split_ratio,
            split_mode="none",
            split=split,
            p_remove=p_remove,
        )
        self._assign_split()

    def _assign_split(self):
        if self.split == "all":
            return
        n = len(self.image_paths)
        indices = list(range(n))
        random.seed(self.seed)
        random.shuffle(indices)
        split_idx = int(n * (1 - self.split_ratio))
        if self.split == "train":
            chosen = indices[:split_idx]
        else:
            chosen = indices[split_idx:]
        self.image_paths = [self.image_paths[i] for i in chosen]
        self.mask_paths = [self.mask_paths[i] for i in chosen]
