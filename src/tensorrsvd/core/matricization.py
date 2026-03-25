from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pylops
from pylops.utils.backend import deps

from ..backends import get_arange, get_conj, get_meshgrid, get_ravel, get_zeros

if deps.jax_enabled:
    import jax
    import jax.numpy as jnp


def _unit_lerp(idx, n):
    """Map integer indices in {0, ..., n-1} to the interval [0, 1].

    For n == 1, returns 0 (single grid point at the origin). Works on both
    scalars and arrays. Equivalent to np.linspace(0, 1, n)[idx] but without
    materializing the full array.
    """
    if n <= 1:
        return idx * 0  # preserves dtype and backend array type
    return idx / (n - 1)


def _insert_mode_coord(other_coords, mode_val, mode, num_idxs):
    """Build the full coordinate list by inserting mode_val at position mode."""
    coords = [None] * num_idxs
    j = 0
    for d in range(num_idxs):
        if d == mode:
            coords[d] = mode_val
        else:
            coords[d] = other_coords[j]
            j += 1
    return coords


class MatricizedTensorOperator(pylops.LinearOperator):
    """Linear operator for the mode-m unfolding of a tensor defined by a callable.

    Parameters
    ----------
    tensor_callable : Callable
        Vectorized callable accepting k array arguments (coordinates in [0, 1])
        and returning the tensor values at those coordinates. Must be
        JAX-traceable if backend == 'jax'.
    tensor_shape : tuple[int, ...]
        Shape of the tensor (n_0, n_1, ..., n_{k-1}).
    mode : int
        Mode along which to unfold.
    dtype : npt.DTypeLike
        Data type of the operator.
    backend : str
        Array backend: 'numpy', 'cupy', or 'jax'.
    """

    def __init__(
        self,
        tensor_callable: Callable,
        tensor_shape: tuple[int, ...],
        mode: int,
        dtype: npt.DTypeLike,
        backend: str,
    ):
        num_idxs = len(tensor_shape)
        num_rows = tensor_shape[mode]
        other_dims = [i for i in range(num_idxs) if i != mode]
        num_cols = math.prod(tensor_shape[i] for i in other_dims)

        self._callable = tensor_callable
        self._tensor_shape = tuple(tensor_shape)
        self._mode = mode
        self._num_idxs = num_idxs
        self._num_rows = num_rows
        self._other_dims = other_dims
        self._backend = backend

        arange = get_arange(backend)
        meshgrid = get_meshgrid(backend)
        ravel = get_ravel(backend)

        other_1d = [  # precompute the 1D coordinate arrays for the non-mode dimensions
            _unit_lerp(arange(tensor_shape[i], dtype=dtype), tensor_shape[i]) for i in other_dims
        ]
        mesh = meshgrid(*other_1d, indexing="ij")
        self._other_coords = tuple(ravel(m) for m in mesh)
        del mesh, other_1d

        self._mode_vals = _unit_lerp(arange(num_rows, dtype=dtype), num_rows)

        # build jitted JAX kernels once to avoid retracing
        if backend == "jax" and deps.jax_enabled:
            self._jax_matmat_fn = self._build_jax_matmat()
            self._jax_rmatmat_fn = self._build_jax_rmatmat()

        super().__init__(dtype=np.dtype(dtype), shape=(num_rows, num_cols))

    def _eval_row(self, i_m: int):
        """Evaluate all tensor entries with mode index fixed to i_m."""
        mode_val = self._mode_vals[i_m]
        coords = _insert_mode_coord(self._other_coords, mode_val, self._mode, self._num_idxs)
        return self._callable(*coords)

    def _matvec(self, x):
        if self._backend == "jax":
            return self._jax_matmat_fn(x[:, None]).ravel()
        zeros = get_zeros(self._backend)
        y = zeros(self._num_rows, dtype=self.dtype)
        for i in range(self._num_rows):
            y[i] = self._eval_row(i) @ x
        return y

    def _rmatvec(self, y):
        if self._backend == "jax":
            return self._jax_rmatmat_fn(y[:, None]).ravel()
        conj = get_conj(self._backend)
        z = get_zeros(self._backend)(self.shape[1], dtype=self.dtype)
        for i in range(self._num_rows):
            z += y[i] * conj(self._eval_row(i))
        return z

    def _matmat(self, X):
        """Compute A @ X where X is (num_cols, l)."""
        if self._backend == "jax":
            return self._jax_matmat_fn(X)
        X_cols = X.shape[1]
        Y = get_zeros(self._backend)((self._num_rows, X_cols), dtype=self.dtype)
        for i in range(self._num_rows):
            row = self._eval_row(i)
            Y[i, :] = row @ X
        return Y

    def _rmatmat(self, Y):
        """Compute A^H @ Y where Y is (num_rows, l).

        Uses a column-wise rank-1 accumulation to avoid allocating an
        O(num_cols * l) broadcast temporary per row. Peak working memory
        beyond the output buffer is O(num_cols) for a single conjugated row.
        """
        if self._backend == "jax":
            return self._jax_rmatmat_fn(Y)
        conj = get_conj(self._backend)
        Y_cols = Y.shape[1]
        num_cols = self.shape[1]
        Z = get_zeros(self._backend)((num_cols, Y_cols), dtype=self.dtype)
        for i in range(self._num_rows):
            crow = conj(self._eval_row(i))
            for j in range(Y_cols):
                Z[:, j] += Y[i, j] * crow
        return Z

    def _build_jax_matmat(self):
        """Return a jitted function computing A @ X via lax.fori_loop."""
        fn = self._callable
        other_coords = self._other_coords
        mode_vals = self._mode_vals
        mode = self._mode
        num_idxs = self._num_idxs
        num_rows = self._num_rows
        dtype = self.dtype

        @jax.jit
        def matmat(X):
            X_cols = X.shape[1]

            def body(i, Y):
                coords = _insert_mode_coord(other_coords, mode_vals[i], mode, num_idxs)
                row = fn(*coords)
                return Y.at[i].set(row @ X)

            return jax.lax.fori_loop(0, num_rows, body, jnp.zeros((num_rows, X_cols), dtype=dtype))

        return matmat

    def _build_jax_rmatmat(self):
        """Return a jitted function computing A^H @ Y via lax.fori_loop."""
        fn = self._callable
        other_coords = self._other_coords
        mode_vals = self._mode_vals
        mode = self._mode
        num_idxs = self._num_idxs
        num_rows = self._num_rows
        num_cols_val = self.shape[1]
        dtype = self.dtype

        @jax.jit
        def rmatmat(Y):
            Y_cols = Y.shape[1]

            def body(i, Z):
                coords = _insert_mode_coord(other_coords, mode_vals[i], mode, num_idxs)
                row = fn(*coords)
                return Z + jnp.conj(row)[:, None] * Y[i, :]

            return jax.lax.fori_loop(
                0, num_rows, body, jnp.zeros((num_cols_val, Y_cols), dtype=dtype)
            )

        return rmatmat
