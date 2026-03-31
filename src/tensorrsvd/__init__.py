"""TensorRSVD: tensor-free randomized Higher-Order SVD.

Represents tensors as Python callables and computes their Tucker decomposition
using randomized linear algebra, without ever forming the dense tensor.
Public entry points: :func:`tensorrsvd.ho_rsvd` and
:func:`tensorrsvd.reconstruct`.
"""

from importlib.metadata import version

from .api import ho_rsvd, reconstruct

__version__ = version("tensorrsvd")
__all__ = ["ho_rsvd", "reconstruct", "__version__"]
