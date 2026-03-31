"""tests/test_gaussian_tensor.py

Tests verifying that ho_rsvd recovers the analytically known HOSVD of a
d-dimensional multivariate Gaussian with structured covariance.

Given Σ(r) = σ²((1-r)I + r|1⟩⟨1|), the left singular vectors of the
mode-k unfolding are Hermite functions ψ_n(ν(x - µ_k)), and the singular
values decay geometrically as ρⁿ. Both ν and ρ are computable in closed
form from r and d. Crucially, ρ is scale-independent (independent of σ),
so the ratios S[n]/S[0] = ρⁿ are a clean numerical test target.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorrsvd import ho_rsvd


def _hermite_poly(n: int, x: np.ndarray) -> np.ndarray:
    """Physicist's Hermite polynomial H_n(x) via three-term recurrence.

    H_0(x) = 1
    H_1(x) = 2x
    H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    """
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h_prev2 = np.ones_like(x)
    h_prev1 = 2.0 * x
    for k in range(2, n + 1):
        h_curr = 2.0 * x * h_prev1 - 2.0 * (k - 1) * h_prev2
        h_prev2, h_prev1 = h_prev1, h_curr
    return h_prev1


def _hermite_fn(n: int, x: np.ndarray) -> np.ndarray:
    """Normalized physicist's Hermite function ψ_n(x).

    ψ_n(x) = (2ⁿ n! √π)^{-1/2} exp(-x²/2) H_n(x)

    These form an orthonormal basis of L²(ℝ).
    """
    norm = 1.0 / math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi))
    return norm * np.exp(-(x**2) / 2.0) * _hermite_poly(n, x)


def _gaussian_params(r: float, d: int, sigma: float) -> dict:
    """Compute all analytical theory constants for the Gaussian HOSVD.

    Parameters
    ----------
    r : float
        Off-diagonal correlation, in (-1/(d-1), 1).
    d : int
        Number of dimensions (modes).
    sigma : float
        Standard deviation scale (Σ = σ²((1-r)I + r|1⟩⟨1|)).

    Returns
    -------
    dict with keys: a, b, N, C, U, V, nu2, nu, rho
    """
    a = 1.0 / ((1.0 - r) * sigma**2)
    b = r / (1.0 + r * (d - 1))

    det_factor = (1.0 + r * (d - 1)) * (1.0 - r) ** (d - 1)
    N = 1.0 / (math.sqrt((2.0 * math.pi) ** d * det_factor) * sigma**d)

    C = N**2 * math.sqrt((math.pi / a) ** (d - 1) / (1.0 - b * (d - 1)))

    U = a * (b**2 * (d - 1) - 2.0 * (b * d - 1.0)) / (4.0 * (1.0 - b * (d - 1)))
    V = a * b**2 * (d - 1) / (2.0 * (1.0 - b * (d - 1)))

    # ν² in the theory
    nu2 = math.sqrt(4.0 * U**2 - V**2)
    nu = math.sqrt(nu2)

    # Choose root of ρ²V - 4Uρ + V = 0 with |ρ| < 1
    rho_plus = (2.0 * U + nu2) / V
    rho_minus = (2.0 * U - nu2) / V
    rho = rho_minus if abs(rho_minus) < 1.0 else rho_plus

    return {"a": a, "b": b, "N": N, "C": C, "U": U, "V": V, "nu2": nu2, "nu": nu, "rho": rho}


def _make_gaussian_tensor(r: float, d: int, params: dict, mu: float = 0.5):
    """Return a callable fn(*xs) evaluating the d-dim Gaussian on [0,1]^d.

    The Gaussian has mean µ·ones and covariance Σ = σ²((1-r)I + r|1⟩⟨1|).
    The quadratic form uses the precomputed precision parameters a and b:

        Q(x) = a · Σ_k (x_k - µ)² - a·b · (Σ_k (x_k - µ))²
        f(*xs) = N · exp(-Q/2)
    """
    a = params["a"]
    b = params["b"]
    N = params["N"]

    def fn(*xs):
        deltas = [x - mu for x in xs]
        sum_sq = sum(d_k**2 for d_k in deltas)
        sum_lin = sum(deltas)
        Q = a * sum_sq - a * b * sum_lin**2
        return N * np.exp(-0.5 * Q)

    return fn


def _analytical_sv_matrix(rank: int, nu: float, mu: float, n_grid: int) -> np.ndarray:
    """Build the (n_grid, rank) matrix of analytical singular vectors.

    Column n is ψ_n(ν·(x - µ)) evaluated on the [0,1] grid with n_grid points.
    Returned matrix is QR-orthonormalized so the projector formula in
    _subspace_distance is valid on the discrete grid.
    """
    xs = np.arange(n_grid) / (n_grid - 1)
    cols = [_hermite_fn(n, nu * (xs - mu)) for n in range(rank)]
    U_raw = np.column_stack(cols)
    U_ortho, _ = np.linalg.qr(U_raw)
    return U_ortho


def _subspace_distance(U1: np.ndarray, U2: np.ndarray) -> float:
    """Frobenius norm of the difference of orthogonal projectors.

    Both U1 and U2 must have orthonormal columns.
    """
    return float(np.linalg.norm(U1 @ U1.T - U2 @ U2.T, "fro"))


@pytest.mark.parametrize(
    "r,d",
    [
        (0.5, 2),
        (0.8, 2),
        (0.5, 3),
        (0.3, 4),
    ],
)
def test_gaussian_singular_value_ratios(r, d):
    """Singular value ratios S[n]/S[0] should match the analytical ρⁿ.

    ρ is the geometric decay rate derived from the Mehler expansion of the
    kernel κ_k(t, s). It is scale-independent (independent of σ), so this
    test isolates the structural prediction of the theory.
    """
    sigma = 0.1
    mu = 0.5
    n_grid = 64
    rank = 6
    shape = (n_grid,) * d

    params = _gaussian_params(r, d, sigma)
    rho = params["rho"]

    fn = _make_gaussian_tensor(r, d, params, mu)

    _, S_list = ho_rsvd(
        tensor=fn,
        tensor_shape=shape,
        dtype=np.float64,
        rank=rank,
        num_oversamples=10,
        num_power_iterations=2,
        num_idxs=d,
        backend="numpy",
    )

    # σ_k^(n) = sqrt(K·ρⁿ), so S[n]/S[0] = sqrt(ρⁿ) = ρ^(n/2)
    # Equivalently, (S[n]/S[0])² = ρⁿ — eigenvalue decay ratio is ρ
    S = S_list[0]
    for n in range(1, 5):
        ratio_numerical = float(S[n] / S[0])
        ratio_analytical = rho ** (n / 2)
        np.testing.assert_allclose(
            ratio_numerical,
            ratio_analytical,
            rtol=0.05,
            err_msg=(
                f"r={r}, d={d}: S[{n}]/S[0]={ratio_numerical:.4f}, "
                f"expected rho^(n/2)={ratio_analytical:.4f} (rho={rho:.4f})"
            ),
        )


@pytest.mark.parametrize(
    "r,d",
    [
        (0.5, 2),
        (0.8, 2),
    ],
)
def test_gaussian_singular_vectors_subspace(r, d):
    """RSVD singular vectors should span the same subspace as Hermite functions.

    By the Mehler expansion, the eigenfunctions of the mode-k operator C_k are
    ψ_n(ν(x - µ)). The RSVD factor matrix U should therefore lie close to the
    subspace spanned by the first `rank` Hermite functions on the discrete grid.
    """
    sigma = 0.1
    mu = 0.5
    n_grid = 64
    rank = 4
    shape = (n_grid,) * d

    params = _gaussian_params(r, d, sigma)
    nu = params["nu"]

    fn = _make_gaussian_tensor(r, d, params, mu)

    U_list, _ = ho_rsvd(
        tensor=fn,
        tensor_shape=shape,
        dtype=np.float64,
        rank=rank,
        num_oversamples=10,
        num_power_iterations=2,
        num_idxs=d,
        backend="numpy",
    )

    U_analytical = _analytical_sv_matrix(rank, nu, mu, n_grid)

    for mode in range(d):
        dist = _subspace_distance(U_list[mode], U_analytical)
        assert dist < 0.15, f"r={r}, d={d}, mode={mode}: subspace distance={dist:.4f} >= 0.15"


def test_gaussian_mode_symmetry():
    """All modes should have identical singular values for a symmetric Gaussian.

    The covariance Σ(r) treats all dimensions equally, so the mode-k operator
    C_k is the same for every k. The RSVD singular values must therefore
    agree across modes up to numerical noise (tests are deterministic: the
    library uses fixed seeds in get_normal).
    """
    d = 3
    r = 0.5
    sigma = 0.1
    mu = 0.5
    n_grid = 64
    rank = 6
    shape = (n_grid,) * d

    params = _gaussian_params(r, d, sigma)
    fn = _make_gaussian_tensor(r, d, params, mu)

    _, S_list = ho_rsvd(
        tensor=fn,
        tensor_shape=shape,
        dtype=np.float64,
        rank=rank,
        num_oversamples=10,
        num_power_iterations=2,
        num_idxs=d,
        backend="numpy",
    )

    for i in range(1, d):
        np.testing.assert_allclose(
            S_list[i],
            S_list[0],
            rtol=0.02,
            err_msg=f"Mode {i} singular values differ from mode 0",
        )


def test_gaussian_factor_matrices_are_orthonormal():
    """Factor matrices returned by ho_rsvd on the Gaussian tensor must be orthonormal."""
    d = 3
    r = 0.5
    sigma = 0.1
    mu = 0.5
    n_grid = 32
    rank = 4
    shape = (n_grid,) * d

    params = _gaussian_params(r, d, sigma)
    fn = _make_gaussian_tensor(r, d, params, mu)

    U_list, _ = ho_rsvd(
        tensor=fn,
        tensor_shape=shape,
        dtype=np.float64,
        rank=rank,
        num_oversamples=10,
        num_idxs=d,
        backend="numpy",
    )

    for mode, U in enumerate(U_list):
        UtU = U.T @ U
        np.testing.assert_allclose(
            UtU,
            np.eye(rank),
            atol=1e-10,
            err_msg=f"U_list[{mode}] is not orthonormal",
        )


def test_gaussian_near_separable():
    """For r ≈ 0 the Gaussian is nearly separable, so S[1]/S[0] ≈ ρ ≈ 0.

    When r → 0 the covariance becomes σ²I (independent modes), the tensor is
    rank-1 in each mode, and ρ → 0. We verify both the analytical prediction
    (|ρ| < 0.05) and the numerical outcome (S[1]/S[0] < 0.05).

    Note: r = 0 exactly makes V = 0 (division by zero in the ρ formula), so
    we use r = 0.02 as a safe near-separable value.
    """
    d = 3
    r = 0.02
    sigma = 0.1
    mu = 0.5
    n_grid = 64
    rank = 6
    shape = (n_grid,) * d

    params = _gaussian_params(r, d, sigma)
    rho = params["rho"]

    # Analytical assertion: theory predicts near-zero ρ for r = 0.02
    assert abs(rho) < 0.05, f"Expected near-zero rho for r=0.02, got rho={rho:.6f}"

    fn = _make_gaussian_tensor(r, d, params, mu)

    _, S_list = ho_rsvd(
        tensor=fn,
        tensor_shape=shape,
        dtype=np.float64,
        rank=rank,
        num_oversamples=10,
        num_power_iterations=2,
        num_idxs=d,
        backend="numpy",
    )

    ratio = float(S_list[0][1] / S_list[0][0])
    assert ratio < 0.05, f"Expected near-separable S[1]/S[0] < 0.05, got {ratio:.4f}"
