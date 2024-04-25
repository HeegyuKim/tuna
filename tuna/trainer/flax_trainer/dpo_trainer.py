import copy
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental.pjit import pjit
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import PartitionSpec
import flax
import optax
from flax.training.train_state import TrainState
import fjformer
from fjformer import (
    match_partition_rules,
    make_shard_and_gather_fns,
    with_sharding_constraint
)

from ..flax_base import FlaxBaseTrainer, trainers
from ..args import BaseTrainingArguments
from ...task.flax.flax_base import FlaxTask
from ..flax.partition_rules import get_partition_rules
from tqdm.auto import tqdm


class DPOTrainState(TrainState):
    ref_params: flax.core.FrozenDict[str, Any] = flax.struct.field(pytree_node=True)



@trainers.register("dpo")
class FlaxDPOTrainer(FlaxBaseTrainer):
    
    def shard_params(self):
        self.init_mesh()
        self.task.init_model(self.dtype)
        with jax.default_device(jax.devices("cpu")[0]):
            params=self.task.params
            ref_params = copy.deepcopy(params)

        def init_train_state():
            state = DPOTrainState.create(
                apply_fn=None,
                params=params,
                ref_params=ref_params,
                tx=self.optimizer
            )
            return state
        
        def create_train_state(params, ref_params):
            state = DPOTrainState.create(
                apply_fn=None,
                params=params,
                ref_params=ref_params,
                tx=self.optimizer
            )
            return state
        
        state_shape = jax.eval_shape(init_train_state)

        with self.mesh:
            print("matching partition rules")
            partition_specs = match_partition_rules(params=state_shape, rules=get_partition_rules(self.task.model.config, self.args.fully_sharded))
            shard_fns, gather_fns = make_shard_and_gather_fns(partition_specs.params, self.dtype)
            print(
                "sharding parameters across all of the chosen backend(tpu/gpu/cpu)s"
            )
            def shard_params(params, shard_fns, desc="Sharding Params"):
                params = flax.traverse_util.flatten_dict(params)
                shard_fns = flax.traverse_util.flatten_dict(shard_fns)
                pbar = tqdm(params.keys(), desc=desc)
                for key in pbar:
                    key = tuple(key)
                    params[key] = shard_fns[key](params[key])
                params = flax.traverse_util.unflatten_dict(params)
                params = flax.core.freeze(params)
                return params

            params = shard_params(params, shard_fns)
            ref_params = shard_params(ref_params, shard_fns, "Sharding Reference Params")

            self.create_sharded_train_state = pjit(
                create_train_state,
                in_shardings=(partition_specs.params, partition_specs.params),
                out_shardings=partition_specs,
                donate_argnums=(0,)
            )

            self.sharded_train_step = self.task.create_train_step(
                pjit,
                partition_specs,
                PartitionSpec
            )
            self.sharded_eval_step = self.task.create_eval_step(
                pjit,
                partition_specs,
                PartitionSpec
            )

            sharded_state = self.create_sharded_train_state(params, ref_params)

        self.state_partition_specs = partition_specs
        self.sharded_state = sharded_state
        self.shard_fns = shard_fns
        self.gather_fns = gather_fns
        self.task.params = None