Mathematical Background
=======================

This page provides the mathematical foundations underlying TensorRSVD. Readers
familiar with tensor decompositions can jump directly to :ref:`randomized-svd`
or :ref:`tensor-free-approach`.

.. contents:: Contents
   :local:
   :depth: 2

.. _tensors-and-notation:

Tensors and Notation
--------------------

For the purposes of this library, a tensor of order :math:`k` is a
multi-dimensional array

.. math::

   \mathcal{T} \in \mathbb{C}^{n_1 \times n_2 \times \cdots \times n_k},

indexed by :math:`\mathcal{T}_{i_1, i_2, \ldots, i_k}` where
:math:`0 \le i_m < n_m` for each mode :math:`m`. Ordinary matrices are
order-2 tensors and vectors are order-1 tensors.

**Callable representation.** TensorRSVD never stores :math:`\mathcal{T}` as a
dense array. Instead it represents the tensor as a Python callable (i.e.,
function)

.. math::

   f : [0,1]^k \to \mathbb{C},

where each coordinate :math:`x_m = i_m / (n_m - 1)` is the *normalized* grid
position of index :math:`i_m` in mode :math:`m`. This means the grid spans
:math:`[0, 1]` uniformly in every mode.

The callable must be fully vectorized: each argument :math:`x_m` is a NumPy
(or JAX / CuPy) array, and the function returns an array of the same shape with
the corresponding tensor values.

.. code-block:: python

   import numpy as np

   # T(x0, x1, x2) = exp(-(x0^2 + x1^2 + x2^2))  (3-D Gaussian)
   def gaussian(x0, x1, x2):
       return np.exp(-(x0**2 + x1**2 + x2**2))

.. _matricization:

Mode-m Unfolding (Matricization)
---------------------------------

The mode-:math:`m` unfolding (or matricization) of :math:`\mathcal{T}`
is the matrix

.. math::

   T_{(m)} \in \mathbb{C}^{n_m \times N_m},
   \qquad N_m = \prod_{j \ne m} n_j,

obtained by making the mode-:math:`m` index the row index and collapsing all
remaining indices into a single column index in C (row-major) order.

Concretely, the element in row :math:`i_m` and the column that corresponds to
multi-index :math:`(i_0, \ldots, i_{m-1}, i_{m+1}, \ldots, i_{k-1})` is

.. math::

   \bigl[T_{(m)}\bigr]_{i_m,\, \mathrm{col}} = \mathcal{T}_{i_0, \ldots, i_{k-1}}.

Each row of :math:`T_{(m)}` is a *mode-m fiber*: the 1-D section of the
tensor obtained by varying index :math:`i_m` while holding all other indices
fixed.

Mode-m unfoldings are the key primitive behind HOSVD: the SVD of :math:`T_{(m)}`
gives the most important "directions" of :math:`\mathcal{T}` along mode
:math:`m`.

.. _tucker-hosvd:

Tucker Decomposition and HOSVD
--------------------------------

The Tucker decomposition expresses a tensor approximately as

.. math::

   \mathcal{T} \approx \mathcal{G} \times_1 U_1 \times_2 U_2 \cdots \times_k U_k,

where

- :math:`\mathcal{G} \in \mathbb{C}^{r_1 \times r_2 \times \cdots \times r_k}` is the core tensor,
- :math:`U_m \in \mathbb{C}^{n_m \times r_m}` is the factor matrix for mode :math:`m` (orthonormal columns),
- :math:`\times_m` denotes the mode-m product: contracting the :math:`m`-th index of :math:`\mathcal{G}` with :math:`U_m`.

The integer tuple :math:`(r_1, \ldots, r_k)` is called the Tucker rank.

**HOSVD construction.**
The Higher-Order SVD (HOSVD) [DLV2000]_ computes each factor matrix :math:`U_m`
as the leading :math:`r_m` left singular vectors of the mode-m unfolding:

.. math::

   T_{(m)} = U_m \Sigma_m V_m^\dagger, \qquad U_m \in \mathbb{C}^{n_m \times r_m}.

The mode-m singular values :math:`\sigma_1^{(m)} \ge \sigma_2^{(m)} \ge
\cdots \ge 0` (the diagonal of :math:`\Sigma_m`) measure how much energy
:math:`\mathcal{T}` has in each direction along mode :math:`m`. Truncating to
rank :math:`r_m` discards the directions that carry least energy.

.. note::

   TensorRSVD returns the factor matrices :math:`U_m` and the mode-m singular
   values :math:`\Sigma_m` but not the core tensor :math:`\mathcal{G}`.
   See :ref:`user-guide-reconstruction` in the User Guide for how to recover
   :math:`\mathcal{G}` if needed.

.. _randomized-svd:

Randomized SVD
--------------

Computing the exact SVD of :math:`T_{(m)}` requires forming the matrix
explicitly, which costs :math:`\mathcal{O}(n_m \cdot N_m)` memory and
:math:`\mathcal{O}(n_m^2 \cdot N_m)` time.

TensorRSVD uses the randomized SVD algorithm of Halko, Martinsson, and Tropp
[HMT2011]_.

**Stage 1 — Range finder.**
Find an orthonormal matrix :math:`Q \in \mathbb{C}^{n_m \times \ell}`,
:math:`\ell = r_m + p`, whose column space approximates the range of
:math:`T_{(m)}`:

1. Draw a Gaussian random matrix
   :math:`\Omega \in \mathbb{C}^{N_m \times \ell}`.
2. Form the sketch :math:`Y = T_{(m)} \Omega \in \mathbb{C}^{n_m \times \ell}`.
3. Orthonormalize: :math:`Q, \_ = \mathrm{QR}(Y)`.
4. Power iterations (optional, controlled by ``num_power_iterations``):
   repeat :math:`q` times

   .. math::

      Q \leftarrow \mathrm{QR}\!\left(T_{(m)}^\dagger Q\right), \qquad
      Q \leftarrow \mathrm{QR}\!\left(T_{(m)} Q\right).

   Each iteration sharpens the range estimate, at the cost of two additional
   passes through the operator.

The oversampling parameter :math:`p` (``num_oversamples``) provides a cushion: a
value of 5–10 is usually sufficient for near-exact results.

**Stage 2 — Compression and small SVD.**
Project the operator into the :math:`\ell`-dimensional subspace spanned by
:math:`Q` and perform a cheap dense SVD:

1. Compute :math:`B^\dagger = T_{(m)}^\dagger Q \in \mathbb{C}^{N_m \times \ell}`.
2. Factorize: :math:`B^\dagger = \hat{Q} R` (QR decomposition).
3. Small SVD: :math:`R = \hat{U} \Sigma \hat{V}^\dagger`.
4. Recover left singular vectors: :math:`U_m = Q \hat{V}[:, :r_m]`.
5. Singular values: :math:`S_m = \Sigma[:r_m]`.

The dominant cost is Step 1 (two matrix-vector products with :math:`T_{(m)}` per
power iteration), each of which costs :math:`O(\ell \cdot n_m \cdot N_m)`. Since
:math:`\ell \ll \min(n_m, N_m)`, this is far cheaper than exact SVD.

.. _tensor-free-approach:

Tensor-Free Approach
---------------------

The critical insight is that Stage 1 and Stage 2 only require matrix–matrix
products :math:`T_{(m)} X` and :math:`T_{(m)}^\dagger Y` — never the explicit
matrix :math:`T_{(m)}`.

:class:`~tensorrsvd.core.MatricizedTensorOperator` implements these products
on the fly using the callable :math:`f`:

- **Forward product** :math:`T_{(m)} X`: iterate over rows :math:`i_m` of
  :math:`T_{(m)}`. Row :math:`i_m` equals :math:`f(x_0, \ldots, x_{m-1},
  x_m^{(i_m)}, x_{m+1}, \ldots, x_{k-1})` evaluated over all combinations of the
  remaining indices — one vectorized call per row. Contract with the
  corresponding rows of :math:`X` and accumulate.

- **Adjoint product** :math:`T_{(m)}^\dagger Y`: the same row evaluations, but now
  each row value is an outer product contribution to :math:`T_{(m)}^\dagger Y`.

The tensor :math:`\mathcal{T}` is never materialized as a dense array. Only
:math:`\mathcal{O}(\ell \cdot n_m)` intermediates are required at any one time.

.. rubric:: References

.. [HMT2011] Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011).
   Finding structure with randomness: Probabilistic algorithms for
   constructing approximate matrix decompositions.
   *SIAM Review*, 53(2), 217–288.
   https://doi.org/10.1137/090771806

.. [DLV2000] De Lathauwer, L., De Moor, B., & Vandewalle, J. (2000).
   A multilinear singular value decomposition.
   *SIAM Journal on Matrix Analysis and Applications*, 21(4), 1253–1278.
   https://doi.org/10.1137/S0895479896305696
