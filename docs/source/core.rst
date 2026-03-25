Core Internals
==============

These modules implement the two algorithmic building blocks that power
:func:`tensorrsvd.ho_svd_r`. They are not part of the public API but are
documented here for contributors and anyone who wants to use the primitives
directly.

For the mathematical description of both components see
:ref:`tensor-free-approach` and :ref:`randomized-svd` in the :doc:`Theory
<theory>` page.

Matricization operator
----------------------

The key idea behind TensorRSVD is that a mode-*m* unfolding of the tensor can be
exposed as a :class:`pylops.LinearOperator` without ever forming the dense
matrix. The operator evaluates the tensor callable on-the-fly whenever a
matrix–vector (or matrix–matrix) product is requested.

.. currentmodule:: tensorrsvd.core.matricization

.. autoclass:: MatricizedTensorOperator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Randomized SVD
--------------

The randomized SVD is implemented in two stages:

1. :func:`randomized_range_finder` builds an orthonormal basis :math:`Q` for the
   approximate column space of the operator using a Gaussian sketch and optional
   power iterations.
2. :func:`rsvd_left` calls the range finder and then extracts the leading
   singular values and left singular vectors via a small dense SVD.

.. currentmodule:: tensorrsvd.core.rsvd

.. autofunction:: randomized_range_finder

.. autofunction:: rsvd_left
