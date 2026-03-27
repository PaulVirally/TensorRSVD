from __future__ import annotations

from collections.abc import Callable

from numpy.typing import ArrayLike, DTypeLike

from .core import MatricizedTensorOperator, rsvd_left


def _broadcast_params(rank, num_oversamples, num_power_iterations, num_idxs):
    """Infer num_idxs and broadcast scalar params to lists."""
    if num_idxs is None:
        for p in (rank, num_oversamples, num_power_iterations):
            if not isinstance(p, int):
                num_idxs = len(p)
                break
        else:
            raise ValueError("num_idxs is required when all parameters are scalars.")
    rank = [rank] * num_idxs if isinstance(rank, int) else list(rank)
    num_oversamples = (
        [num_oversamples] * num_idxs if isinstance(num_oversamples, int) else list(num_oversamples)
    )
    num_power_iterations = (
        [num_power_iterations] * num_idxs
        if isinstance(num_power_iterations, int)
        else list(num_power_iterations)
    )
    if not len(rank) == len(num_oversamples) == len(num_power_iterations) == num_idxs:
        raise ValueError(
            "rank, num_oversamples, and num_power_iterations must all have length num_idxs."
        )
    return rank, num_oversamples, num_power_iterations, num_idxs


def ho_rsvd(
    tensor: Callable,
    tensor_shape: tuple[int, ...],
    dtype: DTypeLike,
    rank: int | ArrayLike,
    num_oversamples: int | ArrayLike = 10,
    num_power_iterations: int | ArrayLike = 0,
    num_idxs: int | None = None,
    backend: str = "numpy",
) -> tuple[list[ArrayLike], list[ArrayLike]]:
    """Compute the randomized higher-order SVD (HOSVD) of a tensor represented by a callable function.

    Parameters
    ----------
    tensor : Callable
        Vectorized callable accepting k array arguments (coordinates in [0, 1])
        and returning the tensor values at those coordinates. Must be
        JAX-traceable if backend == 'jax'.
    tensor_shape : tuple of ints
        Shape of the tensor (n_0, ..., n_{k-1}).
    dtype : DTypeLike
        Numeric dtype for computations.
    rank : int or ArrayLike
        Target rank(s) for the approximation. Broadcast to all modes if scalar.
    num_oversamples : int or ArrayLike, optional
        Extra random vectors beyond rank for improved accuracy. Default is 10.
    num_power_iterations : int or ArrayLike, optional
        Number of power iterations for improved accuracy. Default is 0.
    num_idxs : int, optional
        Number of modes; inferred from array-like params if omitted.
    backend : str, optional
        Backend to use for computations: 'numpy', 'jax', or 'cupy'. Default is 'numpy'.

    Returns
    -------
    U_list : list of ArrayLike
        List of orthonormal matrices for each mode, where each matrix has shape
        (n_i, rank_i) and n_i is the size of the tensor along mode i.
    S_list : list of ArrayLike
        List of singular values for each mode, where each array has shape
        (rank_i,) and rank_i is the target rank for mode i.

    Examples
    --------
    Decompose a simple 3-D linear tensor into Tucker factors:

    >>> import numpy as np
    >>> from tensorrsvd import ho_rsvd
    >>> def my_tensor(x0, x1, x2):
    ...     return x0 - x1 + x2
    >>> U_list, S_list = ho_rsvd(
    ...     tensor=my_tensor,
    ...     tensor_shape=(16, 16, 16),
    ...     dtype=np.float64,
    ...     rank=3,
    ...     num_oversamples=5,
    ...     num_idxs=3,
    ... )
    >>> len(U_list)
    3
    >>> U_list[0].shape
    (16, 3)

    The factor matrices are orthonormal:

    >>> np.allclose(U_list[0].T @ U_list[0], np.eye(3), atol=1e-10)
    True

    See Also
    --------
    tensorrsvd.core.randomized_range_finder :
        Low-level range-finder used internally.
    tensorrsvd.core.rsvd_left :
        Low-level randomized SVD used internally.
    """
    if backend not in ("numpy", "jax", "cupy"):
        raise ValueError(f"Unsupported backend: {backend!r}. Supported: 'numpy', 'jax', 'cupy'.")
    rank, num_oversamples, num_power_iterations, num_idxs = _broadcast_params(
        rank, num_oversamples, num_power_iterations, num_idxs
    )

    U_list = []
    S_list = []

    for mode in range(num_idxs):
        matricized = MatricizedTensorOperator(tensor, tensor_shape, mode, dtype, backend)
        Um, Sm = rsvd_left(
            matricized, rank[mode], num_oversamples[mode], num_power_iterations[mode], backend
        )
        U_list.append(Um)
        S_list.append(Sm)

    return U_list, S_list
