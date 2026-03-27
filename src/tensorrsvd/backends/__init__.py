"""Backend-dispatch helpers for NumPy, JAX, and CuPy.

Each ``get_*`` function accepts a backend string and returns the
corresponding callable from the selected array library, allowing the
rest of the library to remain backend-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from pylops.utils.backend import deps

__all__ = [
    "get_qr",
    "get_svd",
    "get_normal",
    "get_arange",
    "get_meshgrid",
    "get_empty",
    "get_zeros",
    "get_conj",
    "get_ravel",
    "is_complex",
    "real_dtype",
]

if deps.cupy_enabled:
    import cupy as cp

if deps.jax_enabled:
    import jax
    import jax.numpy as jnp


def _dispatch(
    backend: str,
    *,
    numpy_fn: Callable,
    jax_fn: Callable | None,
    cupy_fn: Callable | None,
) -> Callable:
    if backend == "jax":
        if not deps.jax_enabled:
            raise ImportError("JAX is not available. Please install JAX.")
        return jax_fn  # type: ignore[return-value]
    if backend == "cupy":
        if not deps.cupy_enabled:
            raise ImportError("CuPy is not available. Please install CuPy.")
        return cupy_fn  # type: ignore[return-value]
    if backend == "numpy":
        return numpy_fn
    raise ValueError(f"Unsupported backend: {backend!r}. Supported: 'numpy', 'jax', 'cupy'.")


def get_qr(backend: str) -> Callable:
    """Return the QR decomposition function for the given backend."""
    return _dispatch(
        backend,
        numpy_fn=np.linalg.qr,
        jax_fn=jnp.linalg.qr if deps.jax_enabled else None,
        cupy_fn=cp.linalg.qr if deps.cupy_enabled else None,
    )


def get_svd(backend: str) -> Callable:
    """Return a full_matrices=False SVD function for the given backend."""
    return _dispatch(
        backend,
        numpy_fn=lambda mat: np.linalg.svd(mat, full_matrices=False),
        jax_fn=(lambda mat: jnp.linalg.svd(mat, full_matrices=False)) if deps.jax_enabled else None,
        cupy_fn=(lambda mat: cp.linalg.svd(mat, full_matrices=False))
        if deps.cupy_enabled
        else None,
    )


def get_normal(backend: str, seed: int = 0) -> Callable:
    """Return a standard-normal sampling function for the given backend."""
    return _dispatch(
        backend,
        numpy_fn=lambda shape, dtype: np.random.default_rng(seed).standard_normal(
            size=shape, dtype=dtype
        ),
        jax_fn=(
            lambda shape, dtype: jax.random.normal(jax.random.key(seed), shape=shape, dtype=dtype)
        )
        if deps.jax_enabled
        else None,
        cupy_fn=(
            lambda shape, dtype: cp.random.default_rng(seed).standard_normal(
                size=shape, dtype=dtype
            )
        )
        if deps.cupy_enabled
        else None,
    )


def get_arange(backend: str) -> Callable:
    """Return the arange function for the given backend."""
    return _dispatch(
        backend,
        numpy_fn=np.arange,
        jax_fn=jnp.arange if deps.jax_enabled else None,
        cupy_fn=cp.arange if deps.cupy_enabled else None,
    )


def get_meshgrid(backend: str) -> Callable:
    """Return the meshgrid function for the given backend."""
    return _dispatch(
        backend,
        numpy_fn=np.meshgrid,
        jax_fn=jnp.meshgrid if deps.jax_enabled else None,
        cupy_fn=cp.meshgrid if deps.cupy_enabled else None,
    )


def get_empty(backend: str) -> Callable:
    """Return the empty-array constructor for the given backend."""
    return _dispatch(
        backend,
        numpy_fn=np.empty,
        jax_fn=jnp.empty if deps.jax_enabled else None,
        cupy_fn=cp.empty if deps.cupy_enabled else None,
    )


def get_zeros(backend: str) -> Callable:
    """Return the zeros-array constructor for the given backend."""
    return _dispatch(
        backend,
        numpy_fn=np.zeros,
        jax_fn=jnp.zeros if deps.jax_enabled else None,
        cupy_fn=cp.zeros if deps.cupy_enabled else None,
    )


def get_conj(backend: str) -> Callable:
    """Return the complex-conjugate function for the given backend."""
    return _dispatch(
        backend,
        numpy_fn=np.conj,
        jax_fn=jnp.conj if deps.jax_enabled else None,
        cupy_fn=cp.conj if deps.cupy_enabled else None,
    )


def get_ravel(backend: str) -> Callable:
    """Return the ravel function for the given backend."""
    return _dispatch(
        backend,
        numpy_fn=np.ravel,
        jax_fn=jnp.ravel if deps.jax_enabled else None,
        cupy_fn=cp.ravel if deps.cupy_enabled else None,
    )


def is_complex(dtype_like: npt.DTypeLike) -> bool:
    """Return True if dtype_like is a complex dtype."""
    return np.issubdtype(np.dtype(dtype_like), np.complexfloating)


def real_dtype(dtype_like: npt.DTypeLike) -> np.dtype:
    """Return the real counterpart of dtype_like (unchanged if already real)."""
    dt = np.dtype(dtype_like)
    if np.issubdtype(dt, np.complexfloating):
        return np.finfo(dt).dtype
    return dt
