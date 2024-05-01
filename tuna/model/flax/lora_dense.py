

import dataclasses
from typing import (
  Any,
  Iterable,
  List,
  Optional,
  Sequence,
  Tuple,
  Union,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax import eval_shape, lax
from jax.core import ShapedArray

from flax.core import meta
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.module import Module, compact
from flax.typing import (
  Array,
  PRNGKey as PRNGKey,
  Dtype,
  Shape as Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
  ConvGeneralDilatedT,
  PaddingLike,
  LaxPadding,
)

from flax.linen import Module, initializers, compact
from flax.linen.linear import default_kernel_init
from fjformer.xrapture import LoraWeight

class LoraCompatDense(Module):
  """A linear transformation applied over the last dimension of the input.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> layer = nn.Dense(features=4)
    >>> params = layer.init(jax.random.key(0), jnp.ones((1, 3)))
    >>> jax.tree_map(jnp.shape, params)
    {'params': {'bias': (4,), 'kernel': (3, 4)}}

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """

  features: int
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = initializers.zeros_init()
  # Deprecated. Will be removed.
  dot_general: Optional[DotGeneralT] = None
  dot_general_cls: Any = None

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    kernel = self.param(
      'kernel',
      self.kernel_init,
      (jnp.shape(inputs)[-1], self.features),
      self.param_dtype,
    )
    if self.use_bias:
      bias = self.param(
        'bias', self.bias_init, (self.features,), self.param_dtype
      )
    else:
      bias = None
    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

    if self.dot_general_cls is not None:
      dot_general = self.dot_general_cls()
    elif self.dot_general is not None:
      dot_general = self.dot_general
    else:
      dot_general = lax.dot_general
    y = dot_general(
      inputs,
      kernel,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision,
    )
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y
