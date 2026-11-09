"""Public CAMP facade. Does not download weights or train."""

import tempfile

import pytest
from PIL import Image

from camp import CAMP, __version__


def test_import_camp():
    assert isinstance(CAMP, type)
    assert __version__ == "0.1.0"


def test_generate_before_load_raises():
    camp = CAMP(root=".", device="cpu", verbose=False)
    image = Image.new("RGB", (16, 16), (1, 2, 3))
    mask = Image.new("L", (16, 16), 255)
    with pytest.raises(RuntimeError):
        camp.generate(image, image, source_mask=mask)


def test_load_missing_weights_raises_file_not_found():
    with tempfile.TemporaryDirectory() as tmp:
        camp = CAMP(root=tmp, device="cpu", verbose=False)
        with pytest.raises(FileNotFoundError):
            camp.load("noclass")
