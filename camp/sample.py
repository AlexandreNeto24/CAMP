"""Public result types.

``CAMP.generate`` and ``CAMP.generate_batch`` return ``Sample``.
``Patch`` and ``Placement`` describe crops and boxes inside the pipeline.
They are not what ``generate`` returns.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image


@dataclass
class Patch:
    image: Image.Image
    mask: Image.Image
    bbox: Tuple[int, int, int, int]
    area: int


@dataclass
class Placement:
    box: Tuple[float, float, float, float]
    score: float
    rank: int = 0
    scale: float = 1.0
    angle: float = 0.0
    flip: bool = False


@dataclass
class Sample:
    image: Image.Image
    mask: Image.Image
    score: Optional[float] = None
