from __future__ import annotations

import numpy as np
import pytest


def make_alternating_tensor(k: int):
    """Return a vectorized callable T(x0, x1, ..., x_{k-1}) = x0 - x1 + x2 - ...

    Coordinates are in [0, 1] (matching the library convention).
    The Tucker rank of this tensor is exactly k in every mode.
    """
    signs = [(-1) ** i for i in range(k)]

    def tensor(*xs):
        return sum(s * x for s, x in zip(signs, xs))

    return tensor


def numpy_dense_tensor(fn, shape: tuple[int, ...]) -> np.ndarray:
    """Materialize the full tensor as a dense numpy array.

    Iterates all index combinations, maps them to [0, 1] coordinates via
    _unit_lerp, and evaluates fn at each point.
    """
    coords_1d = [(np.arange(n) / (n - 1) if n > 1 else np.zeros(1)) for n in shape]
    grids = np.meshgrid(*coords_1d, indexing="ij")
    return fn(*grids)


def pytest_configure(config):
    config.addinivalue_line("markers", "jax: tests requiring JAX")
    config.addinivalue_line("markers", "cupy: tests requiring CuPy")


def jax_available() -> bool:
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


def cupy_available() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


skip_no_jax = pytest.mark.skipif(not jax_available(), reason="JAX not installed")
skip_no_cupy = pytest.mark.skipif(not cupy_available(), reason="CuPy not installed")
