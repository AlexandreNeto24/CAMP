"""Batch progress bars.

The training log is one line per epoch. The bar stays off when
``TQDM_DISABLE`` is ``1``, ``true``, or ``yes``, which is how the tutorials
keep a notebook to those epoch lines.
"""

import os

from tqdm.auto import tqdm


def batch_bar(iterable, desc):
    flag = os.environ.get("TQDM_DISABLE", "").strip().lower()
    disabled = flag in {"1", "true", "yes"}
    return tqdm(iterable, desc=desc, leave=False, disable=disabled)
