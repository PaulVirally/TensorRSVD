from __future__ import annotations

import math

from numpy.typing import ArrayLike
from pylops import LinearOperator

from ..backends import get_normal, get_qr, get_svd, is_complex, real_dtype


def randomized_range_finder(
    op: LinearOperator, rank: int, num_oversamples: int, num_power_iterations: int, backend: str
) -> ArrayLike:
    r"""Compute an orthonormal matrix Q whose range approximates the range of op, i.e., :math:`op \approx Q Q^\dagger op`

    Parameters
    ----------
    op : LinearOperator
        The linear operator for which to find the range.
    rank : int
        Target rank for the approximation.
    num_oversamples : int
        Number of additional random vectors to sample (beyond the target rank) to improve accuracy.
    num_power_iterations : int
        Number of power iterations for improved accuracy.
    backend : str
        Backend to use: 'numpy', 'jax', or 'cupy'.

    Returns
    -------
    Q : ArrayLike
        Orthonormal matrix of shape (m, rank + num_oversamples) whose range approximates the range of op, where m is the number of rows of op.

    References
    ----------
        [HMT2011]_
    """
    m, n = op.shape
    num_components = rank + num_oversamples
    if num_components > n:
        raise ValueError(
            f"rank + num_oversamples must be <= number of columns of op (got {num_components} > {n})"
        )

    normal0 = get_normal(backend, seed=0)
    normal1 = get_normal(backend, seed=1)
    qr = get_qr(backend)

    if is_complex(op.dtype):
        real_type = real_dtype(op.dtype)
        omega = math.sqrt(0.5) * (
            normal0((n, num_components), real_type) + 1j * normal1((n, num_components), real_type)
        )
    else:
        omega = normal0((n, num_components), op.dtype)

    q, _ = qr(op._matmat(omega))

    # power iteration with re-orthogonalization
    for _ in range(num_power_iterations):
        q, _ = qr(op._rmatmat(q))
        q, _ = qr(op._matmat(q))

    return q


def rsvd_left(
    op: LinearOperator, rank: int, num_oversamples: int, num_power_iterations: int, backend: str
) -> tuple[ArrayLike, ArrayLike]:
    """Compute the left singular vectors and values of op using randomized SVD.

    Returns (U, S) such that op ≈ U @ diag(S) @ V^H (where V^H is not returned)

    Parameters
    ----------
    op : LinearOperator
        The linear operator to decompose.
    rank : int
        Target rank for the approximation.
    num_oversamples : int
        Number of additional random vectors to sample (beyond the target rank) to improve accuracy.
    num_power_iterations : int
        Number of power iterations for improved accuracy.
    backend : str
        Backend to use: 'numpy', 'jax', or 'cupy'.

    Returns
    -------
    U : ArrayLike
        Approximate left singular vectors of op, shape (m, rank).
    S : ArrayLike
        Approximate singular values of op, shape (rank,).

    References
    ----------
    Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011).
    Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions.
    *SIAM Review*, 53(2), 217–288.
    https://doi.org/10.1137/090771806
    """
    qr = get_qr(backend)
    svd = get_svd(backend)
    m, n = op.shape
    k = min(rank, m, n)

    q = randomized_range_finder(op, rank, num_oversamples, num_power_iterations, backend)

    bdag = op._rmatmat(q)  # B^H = Q^H @ op = (op^H @ Q)^H
    q2, r2 = qr(bdag)  # B^H = q2 @ r2

    _, s, vH = svd(r2)  # r2 = Ub @ diag(S) @ Vb^H
    s = s[:k]
    vH = vH[:k, :].T.conj()
    u = q @ vH  # U = Q * Vb^H = (Vb @ Q^H)^H
    return u, s
