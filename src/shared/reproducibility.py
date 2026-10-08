"""One seed for the whole project.

Every estimator, splitter, sampler and bootstrap in both tasks takes its seed from `SEED`
(task 2's train/validation/test split keeps its own frozen seed, 20260907, recorded in
`src/task2_mortality/config.json`). `set_global_seed` additionally seeds Python's and NumPy's
global generators, which some libraries (e.g. SHAP's plot jitter) draw from implicitly.
"""
from __future__ import annotations

import os
import random

import numpy as np

SEED = 42


def set_global_seed(seed: int = SEED) -> int:
    """Seed `random`, NumPy's legacy global RNG, and PYTHONHASHSEED for child processes.

    PYTHONHASHSEED only affects interpreters started after this call; nothing in this project
    depends on string-hash order (all set/dict iteration that matters is sorted), so the
    running kernel's hash seed does not change any result.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
