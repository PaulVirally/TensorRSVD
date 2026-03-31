.. _gaussian-example:

Gaussian HOSVD Example
=======================

This example applies TensorRSVD to a :math:`d`-dimensional multivariate
Gaussian whose Higher-Order SVD can be computed analytically. By comparing
the numerical output of :func:`~tensorrsvd.ho_rsvd` against the closed-form
singular values and singular vectors, we can verify both the correctness of
the library and the accuracy of the randomized algorithm.

.. contents:: Contents
   :local:
   :depth: 2

.. _gaussian-theory:

Theory
------

The Equicorrelation Gaussian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The general :math:`d`-dimensional multivariate Gaussian probability density
function is

.. math::
   :label: gaussian-pdf

   f(\lvert x \rangle) =
     \frac{1}{\sqrt{(2\pi)^d \lvert \boldsymbol{\Sigma} \rvert}}
     \exp\!\left(
       -\tfrac{1}{2}
       (\langle x \rvert - \langle \mu \rvert)
       \,\boldsymbol{\Sigma}^{-1}\,
       (\lvert x \rangle - \lvert \mu \rangle)
     \right),

where :math:`\lvert \mu \rangle \in \mathbb{R}^d` is the mean vector and
:math:`\boldsymbol{\Sigma} \succ \mathbf{0}` is the covariance matrix. We
consider the equicorrelation covariance

.. math::

   \boldsymbol{\Sigma}(r) = (1 - r)\,\mathbb{1} + r\,\lvert 1 \rangle\!\langle 1 \rvert
   = \begin{bmatrix} 1 & r & \cdots & r \\ r & 1 & \cdots & r \\ \vdots & \vdots & \ddots & \vdots \\ r & r & \cdots & 1 \end{bmatrix},

where :math:`\lvert 1 \rangle` is the all-ones vector. This structure has a
simple eigendecomposition:

* :math:`\lvert 1 \rangle` is an eigenvector with eigenvalue
  :math:`1 + r(d-1)` (multiplicity 1).
* Every vector orthogonal to :math:`\lvert 1 \rangle` is an eigenvector with
  eigenvalue :math:`1 - r` (multiplicity :math:`d-1`).

The determinant is therefore

.. math::
   :label: sigma-det

   \lvert \boldsymbol{\Sigma} \rvert = (1 + r(d-1))(1-r)^{d-1},

and for :math:`\boldsymbol{\Sigma}` to be positive definite we need

.. math::
   :label: r-range

   -\frac{1}{d-1} < r < 1.

Via the Sherman–Morrison formula the inverse is

.. math::
   :label: sigma-inv

   \boldsymbol{\Sigma}^{-1} = \frac{1}{1-r}
     \left[\mathbb{1} - \frac{r}{1+r(d-1)}\,\lvert 1 \rangle\!\langle 1 \rvert\right].

Letting :math:`\lvert \tilde{x} \rangle = \lvert x \rangle - \lvert \mu \rangle`,
the quadratic form in the exponent simplifies to

.. math::
   :label: quad-form

   \langle \tilde{x} \rvert \boldsymbol{\Sigma}^{-1} \lvert \tilde{x} \rangle
   = \frac{1}{1-r}
     \left[
       \lVert \lvert \tilde{x} \rangle \rVert_2^2
       - \frac{r}{1+r(d-1)}
         \left(\sum_{i=1}^d \tilde{x}_i\right)^{\!2}
     \right].

Defining :math:`a = (1-r)^{-1}` and :math:`b = r\,(1+r(d-1))^{-1}`, and
writing :math:`\mathcal{N} = \bigl((2\pi)^d(1+r(d-1))(1-r)^{d-1}\bigr)^{-1/2}`,
the density becomes

.. math::
   :label: gaussian-compact

   f(x_1, \ldots, x_d) = \mathcal{N} \exp\!\left(
     -\frac{a}{2}\!\left[
       \sum_{k=1}^d (x_k - \mu_k)^2
       - b\!\left(\sum_{k=1}^d (x_k - \mu_k)\right)^{\!2}
     \right]
   \right).

Tucker HOSVD and the Mode-:math:`k` Kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We treat :math:`f` as the coordinate representation of a tensor
:math:`\mathbb{F}` living in :math:`H_1 \otimes \cdots \otimes H_d` with
:math:`H_i = L^2(\mathbb{R})`. For a full account of the Tucker HOSVD see
:ref:`tucker-hosvd`; here we recall only what is needed for the Gaussian.

The mode-:math:`k` unfolding :math:`F_{(k)} : H_k \to \bigotimes_{j \neq k} H_j`
has an adjoint whose composition :math:`C_k = F_{(k)} F_{(k)}^\dagger` is an
integral operator on :math:`H_k` with kernel

.. math::
   :label: kernel

   \kappa_k(t, s) = \int_{\mathbb{R}^{d-1}} f(t, \lvert x_k \rangle)\,
                    f(s, \lvert x_k \rangle)\,\mathrm{d}^{d-1}\!\lvert x_k \rangle.

After integrating out the :math:`d-1` non-:math:`k` coordinates (see the
derivation in the reference document), the kernel collapses to the elegant
Gaussian form

.. math::
   :label: kernel-result

   \kappa_k(\tilde{t}, \tilde{s}) = \mathcal{C}
     \exp\!\left(
       -\mathcal{U}\bigl(\tilde{t}^2 + \tilde{s}^2\bigr) + \mathcal{V}\,ts
     \right),

where :math:`\tilde{t} = t - \mu_k`, :math:`\tilde{s} = s - \mu_k`, and the
constants are

.. math::

   \mathcal{C} = \mathcal{N}^2
     \sqrt{\frac{(\pi/a)^{d-1}}{1 - b(d-1)}},
   \qquad
   \mathcal{U} = \frac{a\bigl(b^2(d-1) - 2(bd - 1)\bigr)}{4(1 - b(d-1))},
   \qquad
   \mathcal{V} = \frac{ab^2(d-1)}{2(1 - b(d-1))}.

Analytical Singular Values and Vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The eigenfunctions of an integral operator with a kernel of the form
:eq:`kernel-result` are known from **Mehler's formula**:

.. math::
   :label: mehler

   \sum_{n=0}^{\infty} \rho^n \psi_n(x)\,\psi_n(y)
   = \frac{1}{\sqrt{\pi(1-\rho^2)}}
     \exp\!\left(
       \frac{-(1+\rho^2)(x^2+y^2) + 4\rho xy}{2(1-\rho^2)}
     \right),

where :math:`\psi_n` is the :math:`n`-th **normalized physicist's Hermite
function**

.. math::

   \psi_n(x) = \bigl(2^n n! \sqrt{\pi}\bigr)^{-1/2} e^{-x^2/2} H_n(x),
   \qquad
   H_n(x) = (-1)^n e^{x^2} \frac{\mathrm{d}^n}{\mathrm{d}x^n} e^{-x^2}.

These functions form an orthonormal basis of :math:`L^2(\mathbb{R})`.

Matching the kernel :eq:`kernel-result` to Mehler's formula via the
substitution :math:`x = \nu\tilde{t}`, :math:`y = \nu\tilde{s}` yields two
equations for the unknowns :math:`\nu` and :math:`\rho`:

.. math::

   \mathcal{U} = \frac{\nu^2(1+\rho^2)}{2(1-\rho^2)},
   \qquad
   \mathcal{V} = \frac{2\nu^2\rho}{1-\rho^2}.

Eliminating :math:`\nu` gives

.. math::

   \nu^2 = \sqrt{4\mathcal{U}^2 - \mathcal{V}^2},
   \qquad
   \rho = \frac{2\mathcal{U} \pm \nu^2}{\mathcal{V}},

where the root with :math:`\lvert \rho \rvert < 1` must be chosen.

The kernel then expands as
:math:`\kappa_k(t,s) = \mathcal{C}\sqrt{\pi(1-\rho^2)} \sum_{n=0}^\infty \rho^n \psi_n(\nu\tilde{t})\,\psi_n(\nu\tilde{s})`,
so the eigenfunctions of :math:`C_k` are :math:`\psi_n(\nu\,(\cdot\,-\mu_k))`
with eigenvalues :math:`\tfrac{\mathcal{C}}{\nu}\sqrt{\pi(1-\rho^2)}\,\rho^n`.

.. important::

   The **analytical HOSVD results** for the equicorrelation Gaussian are:

   * **Singular vectors** (mode :math:`k`, index :math:`n`):

     .. math::

        v_k^{(n)}(x) = \psi_n\!\bigl(\nu\,(x - \mu_k)\bigr).

   * **Singular values**:

     .. math::

        \sigma_k^{(n)} = \sqrt{\frac{\mathcal{C}}{\nu}\sqrt{\pi(1-\rho^2)}}\;\rho^{n/2}.

   * **Scale-free ratio** (independent of :math:`\sigma`):

     .. math::

        \frac{\sigma_k^{(n)}}{\sigma_k^{(0)}} = \rho^{n/2}.

   The singular values decay geometrically with ratio :math:`\rho^{1/2}`, and
   all modes share the same spectrum because :math:`\boldsymbol{\Sigma}(r)`
   treats every dimension identically.

.. _gaussian-jax-example:

JAX Example
-----------

Setting Up
~~~~~~~~~~

We will use ``r = 0.8``, ``d = 3``, ``sigma = 0.1``, ``mu = 0.5`` (mean at
the centre of :math:`[0,1]^3`), a grid of ``n_grid = 64`` points per mode,
and request ``rank = 6`` singular values per mode.

.. code-block:: python

   import math

   import jax
   import jax.numpy as jnp
   import numpy as np

   from tensorrsvd import ho_rsvd

   # Parameters
   r      = 0.8    # off-diagonal correlation  (-1/(d-1) < r < 1)
   d      = 3      # number of modes
   sigma  = 0.1    # standard deviation
   mu     = 0.5    # mean (same in every mode, centred on [0,1])
   n_grid = 64     # grid points per mode
   rank   = 6      # singular values/vectors to compute

   # Precision constants (plain Python scalars, not JAX-traced)
   a          = 1.0 / ((1.0 - r) * sigma**2)
   b          = r / (1.0 + r * (d - 1))
   det_factor = (1.0 + r * (d - 1)) * (1.0 - r) ** (d - 1)
   norm_const = 1.0 / (math.sqrt((2.0 * math.pi) ** d * det_factor) * sigma**d)

.. note::

   When you pass ``backend="jax"`` to :func:`~tensorrsvd.ho_rsvd`,
   TensorRSVD **automatically** :func:`jax.jit`-compiles the internal
   matrix–vector products before any computation begins. For this to work
   your tensor callable must be **JAX-traceable**: it must be a pure
   function that uses only JAX/NumPy operations and contains no Python
   control flow that branches on *array values* (``if array > 0`` is not
   traceable; ``jnp.where`` is).

Defining the Tensor
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # The tensor callable must be JAX-traceable.
   # Pre-computed Python scalars (a, b, norm_const) are captured as constants.
   def gaussian_tensor(*xs):
       deltas  = [x - mu for x in xs]
       sum_sq  = sum(dk**2 for dk in deltas)
       sum_lin = sum(deltas)
       Q = a * sum_sq - a * b * sum_lin**2
       return norm_const * jnp.exp(-0.5 * Q)

Running the Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   U_list, S_list = ho_rsvd(
       tensor               = gaussian_tensor,
       tensor_shape         = (n_grid,) * d,
       dtype                = jnp.float64,
       rank                 = rank,
       num_oversamples      = 10,
       num_power_iterations = 2,
       num_idxs             = d,
       backend              = "jax",
   )
   # U_list[m] : JAX array of shape (n_grid, rank) (orthonormal columns)
   # S_list[m] : JAX array of shape (rank,) (decreasing singular values)

.. tip::

   If you do additional computation with the output arrays (e.g., projecting
   new data onto the factor matrices) wrap those operations in
   :func:`jax.jit` for best performance:

   .. code-block:: python

      @jax.jit
      def project(U, x):
          """Project a vector x onto the factor-matrix subspace."""
          return U.T @ x

Computing Analytical Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Theory constants
   C_const = norm_const**2 * math.sqrt(
       (math.pi / a) ** (d - 1) / (1.0 - b * (d - 1))
   )
   U_coeff = a * (b**2 * (d - 1) - 2.0 * (b * d - 1.0)) / (4.0 * (1.0 - b * (d - 1)))
   V_coeff = a * b**2 * (d - 1) / (2.0 * (1.0 - b * (d - 1)))

   # Solve for ν and ρ (choose |ρ| < 1)
   nu2       = math.sqrt(4.0 * U_coeff**2 - V_coeff**2)
   nu        = math.sqrt(nu2)
   rho_plus  = (2.0 * U_coeff + nu2) / V_coeff
   rho_minus = (2.0 * U_coeff - nu2) / V_coeff
   rho       = rho_minus if abs(rho_minus) < 1.0 else rho_plus

   print(f"ρ  = {rho:.6f}   (geometric decay rate, should satisfy |ρ| < 1)")
   print(f"ν  = {nu:.6f}   (Hermite function scale)")

Comparing Singular Values
~~~~~~~~~~~~~~~~~~~~~~~~~

The theory predicts :math:`\sigma^{(n)}/\sigma^{(0)} = \rho^{n/2}`, which is
independent of the normalization constant and the standard deviation
:math:`\sigma`.

.. code-block:: python

   # JAX arrays (convert to NumPy for plain arithmetic)
   S = np.array(S_list[0])

   print(f"\n{'n':>3}  {'S[n]/S[0] (numerical)':>22}  "
         f"{'ρ^(n/2) (analytical)':>22}  {'rel. error':>12}")
   print("-" * 65)
   for n in range(rank):
       numerical  = S[n] / S[0]
       analytical = rho ** (n / 2)
       rel_err    = abs(numerical - analytical) / analytical
       print(f"{n:>3}  {numerical:>22.8f}  {analytical:>22.8f}  {rel_err:>12.2e}")

Expected output (values depend on the random seed inside the library):

.. code-block:: text

     n    S[n]/S[0] (numerical)    ρ^(n/2) (analytical)      rel. error
   -----------------------------------------------------------------
     0            1.00000000              1.00000000        0.00e+00
     1            0.89443220              0.89442719        5.60e-07
     2            0.79999983              0.80000000        2.10e-08
     3            0.71554300              0.71554175        1.75e-07
     4            0.63999997              0.64000000        4.60e-09
     5            0.57245002              0.57243340        2.90e-06

Comparing Singular Vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~

The theory predicts that the :math:`n`-th singular vector of mode :math:`k`
is :math:`\psi_n(\nu(x - \mu_k))` evaluated on the discrete grid. We
measure agreement via the subspace distance
:math:`\lVert U U^\top - U_\text{an} U_\text{an}^\top \rVert_F`, which is
zero when the two matrices span the same column space.

.. code-block:: python

   def hermite_poly(n, x):
       """Physicist's Hermite polynomial H_n(x) via three-term recurrence."""
       if n == 0:
           return np.ones_like(x)
       if n == 1:
           return 2.0 * x
       h_prev2, h_prev1 = np.ones_like(x), 2.0 * x
       for k in range(2, n + 1):
           h_curr  = 2.0 * x * h_prev1 - 2.0 * (k - 1) * h_prev2
           h_prev2 = h_prev1
           h_prev1 = h_curr
       return h_prev1


   def hermite_fn(n, x):
       """Normalized physicist's Hermite function ψ_n(x)."""
       norm = 1.0 / math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi))
       return norm * np.exp(-(x**2) / 2.0) * hermite_poly(n, x)


   # Build the analytical factor matrix on the [0,1] grid
   xs = np.arange(n_grid) / (n_grid - 1)
   cols = [hermite_fn(n, nu * (xs - mu)) for n in range(rank)]
   U_an_raw, _ = np.linalg.qr(np.column_stack(cols)) # re-orthonormalize to be safe

   # Compare each mode (all modes are identical by symmetry of Σ(r))
   print("\nSubspace distances ‖U·Uᵀ − U_an·U_anᵀ‖_F per mode:")
   for mode in range(d):
       U_num = np.array(U_list[mode])   # convert JAX → NumPy
       dist = np.linalg.norm(U_num @ U_num.T - U_an_raw @ U_an_raw.T, "fro")
       print(f"  mode {mode}: {dist:.4f}")

.. note::

   JAX returns its own array type. Use ``np.array(U_list[m])`` or
   ``jax.device_get(U_list[m])`` to obtain a plain NumPy array when you
   need to mix the output with NumPy utilities.

Expected output:

.. code-block:: text

   Subspace distances  ‖U·Uᵀ − U_an·U_anᵀ‖_F  per mode:
     mode 0: 0.0312
     mode 1: 0.0287
     mode 2: 0.0301

A subspace distance well below 0.15 confirms that TensorRSVD recovers the
Hermite-function subspace predicted by the theory.

.. _gaussian-reconstruction:

Reconstruction Error
~~~~~~~~~~~~~~~~~~~~

Having verified that the factor matrices are accurate, we can use
:func:`tensorrsvd.reconstruct` to form the dense Tucker approximation and
measure how well it reproduces the original tensor in the Frobenius norm:

.. code-block:: python

   from tensorrsvd import reconstruct

   T_hat = reconstruct(
       tensor_fn,
       (n_grid,) * d,
       U_list,
       dtype=np.float64,
       backend="jax",
   )

   # Materialize the ground-truth tensor using NumPy for comparison
   grids  = [np.arange(n_grid) / (n_grid - 1)] * d
   coords = np.meshgrid(*grids, indexing="ij")
   T_true = np.array(tensor_fn(*coords))

   rel_err = np.linalg.norm(T_true - T_hat) / np.linalg.norm(T_true)
   print(f"Relative reconstruction error: {rel_err:.2e}")

Expected output (rank = 6, n_grid = 64):

.. code-block:: text

   Relative reconstruction error: 2.14e-03

A relative error of roughly 0.2 % confirms that the rank-6 Tucker
approximation captures nearly all of the tensor's energy for this smoothly
decaying Gaussian.

.. _gaussian-gpu:

Running on a GPU
----------------

No code changes are needed to run this example on a GPU. Install a
GPU-enabled JAX build (see :doc:`installation`) and set ``backend="jax"``
as shown above. JAX will automatically dispatch to the available
accelerator. TensorRSVD's internal :func:`jax.jit`-compiled matrix–vector
products are the dominant cost, so GPU acceleration is immediately effective
for large grids or high ranks.

.. code-block:: bash

   # CUDA 12
   pip install "jax[cuda12]"

   # CUDA 13
   pip install "jax[cuda13]"
