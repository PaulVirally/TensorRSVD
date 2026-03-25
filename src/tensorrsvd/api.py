from __future__ import annotations

from collections.abc import Callable

from numpy.typing import ArrayLike, DTypeLike

from .core import MatricizedTensorOperator, rsvd_left


def ho_svd_r(
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
        Shape of the tensor to decompose. This is required because the tensor is
        represented as a callable function that takes in coordinates and returns
        the tensor value at those coordinates, so we need to know the shape of
        the tensor to perform the decomposition.
    dtype : DTypeLike
        The data type of the tensor values. This is used to determine the data
        type of the random vectors sampled for the randomized SVD computations.
    rank : int or ArrayLike
        Target rank(s) for the approximation. If an integer is provided, the
        same rank will be used for all modes. If an array-like of integers is
        provided, it should have the same length as num_idxs and specify the
        rank for each mode.
    num_oversamples : int or ArrayLike, optional
        Number of additional random vectors to sample (beyond the target rank)
        to improve accuracy. If an integer is provided, the same number of
        oversamples will be used for all modes. If an array-like of integers is
        provided, it should have the same length as num_idxs and specify the
        number of oversamples for each mode. Default is 10.
    num_power_iterations : int or ArrayLike, optional
        Number of power iterations to perform to improve the approximation of
        the range. Each iteration involves multiplying by the tensor and its
        adjoint, which can help capture the dominant singular vectors more
        accurately, especially when the singular values decay slowly. If an
        integer is provided, the same number of power iterations will be used
        for all modes. If an array-like of integers is provided, it should have
        the same length as num_idxs and specify the number of power iterations
        for each mode. Default is 0.
    num_idxs : int, optional
        Number of modes to decompose. This is required if rank, num_oversamples,
        or num_power_iterations are provided as integers, because the function
        needs to know how many modes to decompose. If rank, num_oversamples, and
        num_power_iterations are all provided as array-like with the same
        length, then num_idxs can be inferred and does not need to be provided
        explicitly.
    backend : str, optional
        Backend to use for computations. Supported backends are 'numpy', 'jax',
        and 'cupy'. Default is 'numpy'.

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
    >>> from tensorrsvd import ho_svd_r
    >>> def my_tensor(x0, x1, x2):
    ...     return x0 - x1 + x2
    >>> U_list, S_list = ho_svd_r(
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
    if backend not in ["numpy", "jax", "cupy"]:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends are 'numpy', 'jax', and 'cupy'."
        )
    if num_idxs is None:
        if not isinstance(rank, int):
            num_idxs = len(rank)
        elif not isinstance(num_oversamples, int):
            num_idxs = len(num_oversamples)
        elif not isinstance(num_power_iterations, int):
            num_idxs = len(num_power_iterations)
        else:
            raise ValueError(
                "num_idxs must be provided if rank, num_oversamples, or num_power_iterations are integers. This is because the function needs to know how many modes to decompose."
            )
    if num_idxs is not None:
        if isinstance(rank, int):
            rank = [rank] * num_idxs
        if isinstance(num_oversamples, int):
            num_oversamples = [num_oversamples] * num_idxs
        if isinstance(num_power_iterations, int):
            num_power_iterations = [num_power_iterations] * num_idxs
    if (
        len(rank) != num_idxs
        or len(num_oversamples) != num_idxs
        or len(num_power_iterations) != num_idxs
    ):
        raise ValueError(
            "num_idxs must be the same as the length of rank, num_oversamples, and num_power_iterations if they are provided as lists."
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
