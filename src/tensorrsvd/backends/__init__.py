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


def get_qr(backend: str) -> Callable:
    """Returns correct qr module for the given backend.

    Parameters
    ----------
    backend : str
        Backend to use. Supported backends are 'numpy', 'jax', and 'cupy'.

    Returns
    -------
    f : :obj:`callable`
        Function that performs QR decomposition either from
        :obj:`numpy.linalg.qr` or :obj:`jax.numpy.linalg.qr` or
        :obj:`cupy.linalg.qr`
    """
    if backend == "jax":
        if not deps.jax_enabled:
            raise ImportError("JAX is not available. Please install JAX to use this backend.")
        return jnp.linalg.qr
    elif backend == "cupy":
        if not deps.cupy_enabled:
            raise ImportError("CuPy is not available. Please install CuPy to use this backend.")
        return cp.linalg.qr
    elif backend == "numpy":
        return np.linalg.qr
    raise ValueError(
        f"Unsupported backend: {backend}. Supported backends are 'numpy', 'jax', and 'cupy'."
    )


def get_svd(backend: str) -> Callable:
    """Returns correct svd module for the given backend.

    Parameters
    ----------
    backend : str
        Backend to use. Supported backends are 'numpy', 'jax', and 'cupy'.

    Returns
    -------
    f : :obj:`callable`
        Function that performs SVD either from
        :obj:`numpy.linalg.svd` or :obj:`jax.numpy.linalg.svd` or
        :obj:`cupy.linalg.svd`
    """
    if backend == "jax":
        if not deps.jax_enabled:
            raise ImportError("JAX is not available. Please install JAX to use this backend.")
        return lambda mat: jnp.linalg.svd(mat, full_matrices=False)
    elif backend == "cupy":
        if not deps.cupy_enabled:
            raise ImportError("CuPy is not available. Please install CuPy to use this backend.")
        return lambda mat: cp.linalg.svd(mat, full_matrices=False)
    elif backend == "numpy":
        return lambda mat: np.linalg.svd(mat, full_matrices=False)
    raise ValueError(
        f"Unsupported backend: {backend}. Supported backends are 'numpy', 'jax', and 'cupy'."
    )


def get_normal(backend: str, seed: int = 0) -> Callable:
    """Returns correct normal module for the given backend.

    Parameters
    ----------
    backend : str
        Backend to use. Supported backends are 'numpy', 'jax', and 'cupy'.

    seed : int
        Random seed to use for normal distribution sampling.

    Returns
    -------
    f : :obj:`callable`
        Function that performs normal distribution sampling either from
        :obj:`numpy.random.normal` or :obj:`jax.random.normal` or
        :obj:`cupy.random.normal`
    """
    if backend == "jax":
        if not deps.jax_enabled:
            raise ImportError("JAX is not available. Please install JAX to use this backend.")
        return lambda shape, dtype: jax.random.normal(
            jax.random.key(seed), shape=shape, dtype=dtype
        )
    elif backend == "cupy":
        if not deps.cupy_enabled:
            raise ImportError("CuPy is not available. Please install CuPy to use this backend.")
        return lambda shape, dtype: cp.random.default_rng(seed).standard_normal(
            size=shape, dtype=dtype
        )
    elif backend == "numpy":
        return lambda shape, dtype: np.random.default_rng(seed).standard_normal(
            size=shape, dtype=dtype
        )
    raise ValueError(
        f"Unsupported backend: {backend}. Supported backends are 'numpy', 'jax', and 'cupy'."
    )


def get_arange(backend: str) -> Callable:
    """Returns correct arange module for the given backend.

    Parameters
    ----------
    backend : str
        Backend to use. Supported backends are 'numpy', 'jax', and 'cupy'.

    Returns
    -------
    f : :obj:`callable`
        Function that performs arange either from
        :obj:`numpy.arange` or :obj:`jax.numpy.arange` or
        :obj:`cupy.arange`
    """
    if backend == "jax":
        if not deps.jax_enabled:
            raise ImportError("JAX is not available. Please install JAX to use this backend.")
        return jnp.arange
    elif backend == "cupy":
        if not deps.cupy_enabled:
            raise ImportError("CuPy is not available. Please install CuPy to use this backend.")
        return cp.arange
    elif backend == "numpy":
        return np.arange
    raise ValueError(
        f"Unsupported backend: {backend}. Supported backends are 'numpy', 'jax', and 'cupy'."
    )


def get_meshgrid(backend: str) -> Callable:
    """Returns correct meshgrid module for the given backend.

    Parameters
    ----------
    backend : str
        Backend to use. Supported backends are 'numpy', 'jax', and 'cupy'.

    Returns
    -------
    f : :obj:`callable`
        Function that performs meshgrid either from
        :obj:`numpy.meshgrid` or :obj:`jax.numpy.meshgrid` or
        :obj:`cupy.meshgrid`
    """
    if backend == "jax":
        if not deps.jax_enabled:
            raise ImportError("JAX is not available. Please install JAX to use this backend.")
        return jnp.meshgrid
    elif backend == "cupy":
        if not deps.cupy_enabled:
            raise ImportError("CuPy is not available. Please install CuPy to use this backend.")
        return cp.meshgrid
    elif backend == "numpy":
        return np.meshgrid
    raise ValueError(
        f"Unsupported backend: {backend}. Supported backends are 'numpy', 'jax', and 'cupy'."
    )


def get_empty(backend: str) -> Callable:
    """Returns correct empty module for the given backend.

    Parameters
    ----------
    backend : str
        Backend to use. Supported backends are 'numpy', 'jax', and 'cupy'.

    Returns
    -------
    f : :obj:`callable`
        Function that performs empty either from
        :obj:`numpy.empty` or :obj:`jax.numpy.empty` or
        :obj:`cupy.empty`
    """
    if backend == "jax":
        if not deps.jax_enabled:
            raise ImportError("JAX is not available. Please install JAX to use this backend.")
        return jnp.empty
    elif backend == "cupy":
        if not deps.cupy_enabled:
            raise ImportError("CuPy is not available. Please install CuPy to use this backend.")
        return cp.empty
    elif backend == "numpy":
        return np.empty
    raise ValueError(
        f"Unsupported backend: {backend}. Supported backends are 'numpy', 'jax', and 'cupy'."
    )


def get_zeros(backend: str) -> Callable:
    """Returns correct zeros module for the given backend.

    Parameters
    ----------
    backend : str
        Backend to use. Supported backends are 'numpy', 'jax', and 'cupy'.

    Returns
    -------
    f : :obj:`callable`
        Function that performs zeros either from
        :obj:`numpy.zeros` or :obj:`jax.numpy.zeros` or
        :obj:`cupy.zeros`
    """
    if backend == "jax":
        if not deps.jax_enabled:
            raise ImportError("JAX is not available. Please install JAX to use this backend.")
        return jnp.zeros
    elif backend == "cupy":
        if not deps.cupy_enabled:
            raise ImportError("CuPy is not available. Please install CuPy to use this backend.")
        return cp.zeros
    elif backend == "numpy":
        return np.zeros
    raise ValueError(
        f"Unsupported backend: {backend}. Supported backends are 'numpy', 'jax', and 'cupy'."
    )


def get_conj(backend: str) -> Callable:
    """Returns correct conj module for the given backend.

    Parameters
    ----------
    backend : str
        Backend to use. Supported backends are 'numpy', 'jax', and 'cupy'.

    Returns
    -------
    f : :obj:`callable`
        Function that performs conjugation either from
        :obj:`numpy.conj` or :obj:`jax.numpy.conj` or
        :obj:`cupy.conj`
    """
    if backend == "jax":
        if not deps.jax_enabled:
            raise ImportError("JAX is not available. Please install JAX to use this backend.")
        return jnp.conj
    elif backend == "cupy":
        if not deps.cupy_enabled:
            raise ImportError("CuPy is not available. Please install CuPy to use this backend.")
        return cp.conj
    elif backend == "numpy":
        return np.conj
    raise ValueError(
        f"Unsupported backend: {backend}. Supported backends are 'numpy', 'jax', and 'cupy'."
    )


def get_ravel(backend: str) -> Callable:
    """Returns correct ravel module for the given backend.

    Parameters
    ----------
    backend : str
        Backend to use. Supported backends are 'numpy', 'jax', and 'cupy'.

    Returns
    -------
    f : :obj:`callable`
        Function that performs ravel either from
        :obj:`numpy.ravel` or :obj:`jax.numpy.ravel` or
        :obj:`cupy.ravel`
    """
    if backend == "jax":
        if not deps.jax_enabled:
            raise ImportError("JAX is not available. Please install JAX to use this backend.")
        return jnp.ravel
    elif backend == "cupy":
        if not deps.cupy_enabled:
            raise ImportError("CuPy is not available. Please install CuPy to use this backend.")
        return cp.ravel
    elif backend == "numpy":
        return np.ravel
    raise ValueError(
        f"Unsupported backend: {backend}. Supported backends are 'numpy', 'jax', and 'cupy'."
    )


def is_complex(dtype_like: npt.DTypeLike) -> bool:
    """Returns True if the given dtype is a complex dtype, False otherwise.

    Parameters
    ----------
    dtype_like : npt.DTypeLike
        The dtype to check.

    Returns
    -------
    is_complex : bool
        True if the given dtype is a complex dtype, False otherwise.
    """
    return np.issubdtype(np.dtype(dtype_like), np.complexfloating)


def real_dtype(dtype_like: npt.DTypeLike) -> np.dtype:
    """Returns the real dtype corresponding to the given dtype.

    If the given dtype is a complex dtype, returns the corresponding real dtype. Otherwise, returns the given dtype.

    Parameters
    ----------
    dtype_like : npt.DTypeLike
        The dtype to convert.

    Returns
    -------
    real_dtype : np.dtype
        The real dtype corresponding to the given dtype.
    """
    dt = np.dtype(dtype_like)
    if np.issubdtype(dt, np.complexfloating):
        return np.finfo(dt).dtype
    return dt
