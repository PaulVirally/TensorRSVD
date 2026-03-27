User Guide
==========

This page walks through everything you need to use TensorRSVD in practice: how
to define your tensor, run the decomposition, read the output, choose good
parameters, reconstruct the approximation, and switch backends.

.. contents:: Contents
   :local:
   :depth: 2

.. _user-guide-callable:

Defining a Tensor as a Callable
---------------------------------

TensorRSVD represents tensors as Python callables rather than dense arrays.
The callable must accept :math:`k` positional arguments (one per mode) and
return the tensor values at those coordinates.

**Signature convention**

.. code-block:: python

   def my_tensor(x0, x1, ..., x_{k-1}):
       ...
       return values  # same shape as x0

Each argument :math:`x_m` is a NumPy array of normalized coordinates in
:math:`[0, 1]`. Index :math:`i_m` in a dimension of size :math:`n_m` maps to:

.. math::

   x_m = \frac{i_m}{n_m - 1}.

The function must be fully vectorized: it must operate element-wise on
arrays without Python-level loops, and it must return an array of the same shape
as its inputs.

**Examples**

.. code-block:: python

   import numpy as np

   # Alternating-sign linear tensor (exact Tucker rank = k)
   def alternating(x0, x1, x2):
       return x0 - x1 + x2

   # 3-D Gaussian bump
   def gaussian(x0, x1, x2):
       return np.exp(-(x0**2 + x1**2 + x2**2))

   # Product of 1-D functions (rank-1 in every mode)
   def rank1(x0, x1, x2):
       return np.sin(np.pi * x0) * np.cos(np.pi * x1) * (1 + x2)

.. note::

   **JAX backend**: if you use ``backend="jax"``, the callable must be
   *JAX-traceable*. Replace ``np`` with ``jnp`` (``import jax.numpy as jnp``)
   and avoid Python control flow that depends on array values.

.. _user-guide-running:

Running ``ho_rsvd``
---------------------

The single public entry point is :func:`tensorrsvd.ho_rsvd`. A minimal call
looks like this:

.. code-block:: python

   import numpy as np
   from tensorrsvd import ho_rsvd

   def my_tensor(x0, x1, x2):
       return x0 - x1 + x2

   U_list, S_list = ho_rsvd(
       tensor=my_tensor,
       tensor_shape=(32, 32, 32),
       dtype=np.float64,
       rank=3,
       num_oversamples=10,
       num_power_iterations=2,
       num_idxs=3,
   )

**Parameter guide**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Guidance
   * - ``tensor_shape``
     - Grid dimensions :math:`(n_1, \ldots, n_k)`. This determines the
       coordinate grid and the shape of the linear operators.
   * - ``rank``
     - Tucker rank per mode. Pass a single ``int`` to use the same rank for
       all modes, or a list of ``int`` to specify per-mode ranks. Should be
       much smaller than :math:`\min(n_m, N_m)` for memory savings to apply.
   * - ``num_oversamples``
     - Extra random vectors beyond ``rank`` (default: 10). A value of 5–10
       almost always suffices. Higher values improve accuracy at the cost of
       more passes through the operator.
   * - ``num_power_iterations``
     - Number of power iterations (default: 0). Use 0 for speed, 1–2 for
       better accuracy when singular values decay slowly. Rarely worth going
       above 3.
   * - ``num_idxs``
     - Number of modes to decompose. Inferred automatically when ``rank`` is
       a list; required when ``rank`` is a scalar.
   * - ``backend``
     - ``"numpy"`` (default), ``"jax"``, or ``"cupy"``. See
       :ref:`user-guide-backends`.

.. _user-guide-output:

Interpreting the Output
------------------------

:func:`~tensorrsvd.ho_rsvd` returns a pair ``(U_list, S_list)``:

``U_list``
   A list of :math:`k` factor matrices. ``U_list[m]`` has shape
   ``(n_m, rank_m)`` and contains the leading left singular vectors of the
   mode-:math:`m` unfolding. The columns are orthonormal:
   ``U_list[m].T @ U_list[m] == I``.

``S_list``
   A list of :math:`k` singular value arrays. ``S_list[m]`` has shape
   ``(rank_m,)`` and contains the mode-:math:`m` singular values in
   non-increasing order.

**Reading the singular values**

The mode-:math:`m` singular values quantify how much energy :math:`\mathcal{T}`
has along each direction captured by :math:`U_m`. A sharp decay indicates that
only a few directions are needed; a flat spectrum suggests the tensor is not
well approximated at the chosen rank.

.. code-block:: python

   import matplotlib.pyplot as plt

   for m, S in enumerate(S_list):
       plt.semilogy(S, label=f"mode {m}")
   plt.xlabel("index")
   plt.ylabel("singular value")
   plt.legend()
   plt.title("Mode singular value spectra")
   plt.show()

.. _user-guide-reconstruction:

Reconstructing the Tensor
--------------------------

:func:`~tensorrsvd.ho_rsvd` returns the factor matrices but not the
core tensor :math:`\mathcal{G}`. To reconstruct a dense approximation,
you need to:

1. Materialize the tensor from the callable.
2. Project onto the factor matrices to obtain the core.
3. Expand the core back to the original size.

.. code-block:: python

   import numpy as np

   def materialize(tensor, shape):
       grids = [np.arange(n) / max(n - 1, 1) for n in shape]
       coords = np.meshgrid(*grids, indexing="ij")
       return tensor(*coords)

   T_dense = materialize(my_tensor, tensor_shape)

   # Core tensor G = T x_0 U0^T x_1 U1^T x_2 U2^T ...
   G = T_dense
   for mode, U in enumerate(U_list):
       G = np.tensordot(U.T, G, axes=([1], [mode]))
       G = np.moveaxis(G, 0, mode)

   # T_hat = G x_0 U0 x_1 U1 x_2 U2 ...
   T_hat = G
   for mode, U in enumerate(U_list):
       T_hat = np.tensordot(U, T_hat, axes=([1], [mode]))
       T_hat = np.moveaxis(T_hat, 0, mode)

   # Relative reconstruction error
   rel_err = np.linalg.norm(T_dense - T_hat) / np.linalg.norm(T_dense)
   print(f"Relative error: {rel_err:.2e}")

.. note::

   This reconstruction requires materializing :math:`\mathcal{T}` as a dense
   array, which defeats the memory savings for large tensors. In practice,
   the factor matrices and singular values are often sufficient for downstream
   tasks (dimensionality reduction, feature extraction, compression).

.. _user-guide-backends:

Switching Backends
-------------------

Pass ``backend="numpy"`` (default), ``"jax"``, or ``"cupy"`` to
:func:`~tensorrsvd.ho_rsvd`:

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Backend
     - Required package
     - Notes
   * - ``"numpy"``
     - (always available)
     - CPU, single-threaded. Default choice.
   * - ``"jax"``
     - ``pip install ".[jax]"``
     - CPU / GPU / TPU. Operators are JIT-compiled on first call; subsequent
       calls are fast. Tensors must be JAX-traceable.
   * - ``"cupy"``
     - ``pip install ".[cupy]"``
     - NVIDIA GPU only. Requires CUDA.

**JAX example**

.. code-block:: python

   import jax.numpy as jnp
   from tensorrsvd import ho_rsvd

   def gaussian_jax(x0, x1, x2):
       return jnp.exp(-(x0**2 + x1**2 + x2**2))

   U_list, S_list = ho_rsvd(
       tensor=gaussian_jax,
       tensor_shape=(64, 64, 64),
       dtype=jnp.float32,
       rank=8,
       num_oversamples=10,
       num_power_iterations=2,
       num_idxs=3,
       backend="jax",
   )

.. warning::

   JAX returns its own array type (``jaxlib.xla_extension.ArrayImpl``).
   Convert to NumPy with ``numpy.array(U_list[0])`` if you need standard
   NumPy arrays downstream.

Choosing Good Parameters
-------------------------

.. rubric:: Rank

Set ``rank`` to the expected Tucker rank of your tensor, or to the largest
rank you can afford computationally. For unknown tensors, start with a
conservative estimate and check the singular value decay.

.. rubric:: Oversampling

The default ``num_oversamples=10`` works well for most problems. Increase to
20–30 if you observe large errors or if the singular values decay slowly.

.. rubric:: Power iterations

For tensors with slowly decaying singular values (flat spectra), increase
``num_power_iterations`` to 1 or 2. Each additional iteration adds two more
passes through the operator but typically gives a significant accuracy boost.
Beyond 3 iterations, improvements are usually marginal.

.. rubric:: Grid size

Larger grids (bigger ``tensor_shape``) increase the cost of each matrix–vector
product. The total cost scales roughly as :math:`\mathcal{O}(k \cdot (r + p)
\cdot q \cdot n_{\max} \cdot N_{\max})`, where :math:`q` is
``num_power_iterations`` and :math:`N_{\max} = \prod_{j \ne m} n_j`.
