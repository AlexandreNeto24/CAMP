"""Train/val split for PairedMaskDataset, plus weight lookup."""

import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from camp.dataset import PairedMaskDataset, attention_collate
from camp.io import class_dirs, find_weights, load_image_mask_pairs, load_images


def _write_image(path, mode="RGB", color=(255, 0, 0), size=(4, 4)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "L" and isinstance(color, tuple):
        color = color[0]
    Image.new(mode, size, color).save(path)


def _paired_stems(dataset):
    image_stems = [Path(path).stem for path in dataset.image_paths]
    mask_stems = [Path(path).stem for path in dataset.mask_paths]
    if image_stems != mask_stems:
        raise AssertionError("image and mask stems diverged")
    return image_stems


def _partition(stems, split_ratio, seed):
    indices = list(range(len(stems)))
    random.seed(seed)
    random.shuffle(indices)
    split_idx = int(len(stems) * (1 - split_ratio))
    train = {stems[i] for i in indices[:split_idx]}
    val = {stems[i] for i in indices[split_idx:]}
    return train, val


class PairedMaskDatasetSplitTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        images = self.root / "images"
        masks = self.root / "masks"
        # Five pairs so a 0.4 holdout leaves both sides non-empty (3 and 2).
        for stem in ("a", "b", "c", "d", "e"):
            _write_image(images / f"{stem}.png", color=(10, 20, 30))
            _write_image(masks / f"{stem}.png", mode="L", color=255)

    def test_train_and_val_are_disjoint_and_cover_every_stem(self):
        self.assertTrue(callable(attention_collate))
        seed = 123
        split_ratio = 0.4
        created = {"a", "b", "c", "d", "e"}

        all_ds = PairedMaskDataset(
            self.root, split="all", split_ratio=split_ratio, seed=seed
        )
        order = _paired_stems(all_ds)
        self.assertEqual(set(order), created)

        train_ds = PairedMaskDataset(
            self.root, split="train", split_ratio=split_ratio, seed=seed
        )
        val_ds = PairedMaskDataset(
            self.root, split="val", split_ratio=split_ratio, seed=seed
        )
        train_stems = set(_paired_stems(train_ds))
        val_stems = set(_paired_stems(val_ds))
        expected_train, expected_val = _partition(order, split_ratio, seed)

        self.assertTrue(train_stems.isdisjoint(val_stems))
        self.assertEqual(train_stems | val_stems, created)
        self.assertEqual(train_stems, expected_train)
        self.assertEqual(val_stems, expected_val)
        self.assertTrue(train_stems)
        self.assertTrue(val_stems)
        self.assertEqual(len(train_ds) + len(val_ds), len(all_ds))

        train_again = PairedMaskDataset(
            self.root, split="train", split_ratio=split_ratio, seed=seed
        )
        self.assertEqual(set(_paired_stems(train_again)), train_stems)

        # Default split is train, and the default holdout is 0.2.
        default_ds = PairedMaskDataset(self.root, seed=seed)
        default_train, default_val = _partition(order, 0.2, seed)
        self.assertEqual(set(_paired_stems(default_ds)), default_train)
        self.assertTrue(default_train.isdisjoint(default_val))
        self.assertEqual(default_train | default_val, created)

    def test_unknown_split_raises(self):
        with self.assertRaises(ValueError):
            PairedMaskDataset(self.root, split="test")


class FindWeightsTest(unittest.TestCase):
    def test_find_weights_raises_on_empty_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                find_weights(root, "metaplasia", "segmentor")
            with self.assertRaises(FileNotFoundError):
                find_weights(root, "metaplasia", "recommender")

    def test_find_weights_uses_only_class_kind_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiments = root / "experiments" / "cell_segmentation_001"
            experiments.mkdir(parents=True)
            (experiments / "best_segmentor.pth").write_bytes(b"old")
            (experiments / "best_model.pt").write_bytes(b"old")

            segmentor = root / "weights" / "cell" / "segmentor"
            recommender = root / "weights" / "cell" / "recommender"
            segmentor.mkdir(parents=True)
            recommender.mkdir(parents=True)
            (segmentor / "best_model.pt").write_bytes(b"wrong-kind")
            (recommender / "best_segmentor.pth").write_bytes(b"wrong-kind")

            with self.assertRaises(FileNotFoundError):
                find_weights(root, "cell", "segmentor")
            with self.assertRaises(FileNotFoundError):
                find_weights(root, "cell", "recommender")

            (segmentor / "best_segmentor.pth").write_bytes(b"seg")
            (recommender / "best_model.pt").write_bytes(b"rec")
            self.assertEqual(
                find_weights(root, "cell", "segmentor"),
                segmentor / "best_segmentor.pth",
            )
            self.assertEqual(
                find_weights(root, "cell", "recommender"),
                recommender / "best_model.pt",
            )

            (segmentor / "best.pth").write_bytes(b"best-seg")
            (recommender / "best.pth").write_bytes(b"best-rec")
            self.assertEqual(
                find_weights(root, "cell", "segmentor"),
                segmentor / "best.pth",
            )
            self.assertEqual(
                find_weights(root, "cell", "recommender"),
                recommender / "best.pth",
            )


class IoHelpersTest(unittest.TestCase):
    def test_class_dirs_prefers_nested_source_and_normal_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "data" / "cell" / "cell"
            (nested / "images").mkdir(parents=True)
            (root / "data" / "cell" / "images").mkdir(parents=True)
            normal = root / "data" / "cell" / "normal"
            normal.mkdir()
            source, target = class_dirs(root, "cell")
            self.assertEqual(source, nested)
            self.assertEqual(target, normal)

    def test_class_dirs_falls_back_when_nested_or_normal_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat = root / "data" / "cell"
            (flat / "images").mkdir(parents=True)
            source, target = class_dirs(root, "cell")
            self.assertEqual(source, flat)
            self.assertEqual(target, flat)

            normal = flat / "normal"
            normal.mkdir()
            source, target = class_dirs(root, "cell")
            self.assertEqual(source, flat)
            self.assertEqual(target, normal)

    def test_load_images_and_mask_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_image(root / "top.png", color=(1, 2, 3))
            _write_image(root / "images" / "nested.png", color=(9, 9, 9))
            loaded = load_images(root)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].mode, "RGB")
            self.assertEqual(loaded[0].getpixel((0, 0)), (1, 2, 3))

            empty = root / "empty_top"
            empty.mkdir()
            (empty / "notes.txt").write_text("not an image", encoding="utf-8")
            _write_image(empty / "images" / "s1.png", size=(4, 4))
            _write_image(empty / "images" / "s2.jpg", size=(5, 5))
            _write_image(empty / "masks" / "s1_extra.png", mode="L", color=1)
            _write_image(empty / "masks" / "s2.png", mode="L", color=200)
            _write_image(empty / "images" / "gray.png", mode="L", color=40, size=(6, 6))

            fallback = load_images(empty)
            self.assertEqual(len(fallback), 3)
            self.assertTrue(all(image.mode == "RGB" for image in fallback))
            self.assertEqual(len(load_images(empty, max_images=1)), 1)

            images, masks = load_image_mask_pairs(empty)
            self.assertEqual(len(images), len(masks))
            by_size = {image.size: mask for image, mask in zip(images, masks)}
            # s1_extra.png does not share a stem with s1.png.
            self.assertIsNone(by_size[(4, 4)])
            self.assertIsNone(by_size[(6, 6)])
            self.assertIsNotNone(by_size[(5, 5)])
            self.assertEqual(by_size[(5, 5)].mode, "L")


if __name__ == "__main__":
    unittest.main()
