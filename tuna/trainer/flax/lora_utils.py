import jax
import jax.numpy as jnp
import re

from jax.experimental.pjit import pjit
from jax.lax import with_sharding_constraint as _with_sharding_constraint
import numpy as np
from jax.sharding import PartitionSpec as PS
from jax.experimental import mesh_utils
from jax.interpreters import pxla
import flax
from jax.sharding import Mesh
from typing import Sequence



def make_shard_and_gather_fns_lora_compatible(partition_specs, dtype_specs=None):
    """
    The make_shard_and_gather_fns function takes in a partition_specs and dtype_specs,
    and returns two functions: shard_fns and gather_fns. The shard function is used to
    shard the input tensor into the specified partitions. The gather function is used to
    gather all the shards back together into one tensor.

    :param partition_specs: Specify the sharding of the input tensor
    :param dtype_specs: Specify the dtype of the tensor
    :return: A tuple of functions
    
    """
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)

    def make_to_dtype_fn(dtype_spec):
        def to_dtype(tensor):
            if dtype_specs in float_dtypes and getattr(tensor, 'dtype', None) in float_dtypes:
                # force np array to jax numpy array
                return jnp.asarray(tensor).astype(dtype_specs)
            elif hasattr(dtype_spec, 'dtype') and hasattr(tensor, 'dtype'):
                return jnp.asarray(tensor).astype(dtype_spec.dtype)
            return jnp.asarray(tensor)

        return to_dtype

    def make_shard_fn(partition_spec, dtype_spec=None):
        jax_shard_function = pjit(
            make_to_dtype_fn(dtype_spec),
            in_shardings=None,
            out_shardings=partition_spec
        )

        def shard_fn(tensor):
            return jax_shard_function(tensor).block_until_ready()

        return shard_fn

    def make_gather_fn(partition_spec, dtype_spec=None):
        jax_gather_fn = pjit(
            make_to_dtype_fn(dtype_spec),
            in_shardings=partition_spec,
            out_shardings=None
        )

        def gather_fn(tensor):
            return jax.device_get(jax_gather_fn(tensor))

        return gather_fn

    if dtype_specs is None or dtype_specs in float_dtypes:
        shard_fns = jax.tree_util.tree_map(make_shard_fn, partition_specs)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, partition_specs)
    else:
        shard_fns = jax.tree_util.tree_map(
            make_shard_fn, partition_specs, dtype_specs
        )
        gather_fns = jax.tree_util.tree_map(
            make_gather_fn, partition_specs, dtype_specs
        )
    return shard_fns, gather_fns
