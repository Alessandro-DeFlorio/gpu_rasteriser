"""Type variables shared by the tetvox modules."""

from typing import TypeVar

import numpy as np

# The real dtype of positions, values and real-valued op outputs (ADR 0008: float32 or float64,
# chosen at run time by the caller's arrays).
F = TypeVar("F", np.float32, np.float64)
