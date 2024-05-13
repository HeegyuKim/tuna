import copy
from typing import Any

import flax.jax_utils
import flax.traverse_util
import flax.traverse_util
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
from fjformer.xrapture import use_implicit_args, LoraWeight
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


def unwrap_lora(params):
    # return jax.tree_util.tree_map(
    #     lambda x: x.w if hasattr(x, "w") else x,
    #     params,
    # )
    params = flax.core.frozen_dict.unfreeze(params)
    params = flax.traverse_util.flatten_dict(params)
    params = {k: v.w if isinstance(v, LoraWeight) else v for k, v in params.items()}
    return flax.core.frozen_dict.freeze(flax.traverse_util.unflatten_dict(params))

@trainers.register("dpo")
class FlaxDPOTrainer(FlaxBaseTrainer):
    
    def shard_params(self):
        self.init_mesh()
        self.task.init_model(self.dtype)
        with jax.default_device(jax.devices("cpu")[0]):
            params=self.task.params
            
            if self.args.use_lora:
                self.apply_lora_params(self.task.model, self.optimizer, params)
                params = self.task.params
                self.optimizer = self.lora_modules.lora_tx
                ref_params = None
            else:
                ref_params = copy.deepcopy(params)

        def init_train_state():
            if self.args.use_lora:
                ref_params = unwrap_lora(params)

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

            # sharding
            params = jax.tree_util.tree_map(
                lambda f, x: f(x),
                shard_fns,
                params
            )


            if self.args.use_lora:
                ref_params = unwrap_lora(params)
            else:
                ref_shard_fns, ref_gather_fns = make_shard_and_gather_fns(partition_specs.ref_params, self.dtype)
                ref_params = jax.tree_util.tree_map(
                    lambda f, x: f(x),
                    ref_shard_fns,
                    ref_params
                )

        
            if self.args.use_lora:
                def create_train_state(params):
                    return DPOTrainState(
                        step=0,
                        apply_fn=None,
                        params=params,
                        ref_params=ref_params,
                        tx=self.optimizer,
                        opt_state=self.lora_modules.lora_opt_state
                    )
                self.create_sharded_train_state = pjit(
                    create_train_state,
                    in_shardings=(partition_specs.params,),
                    out_shardings=partition_specs,
                )
                sharded_state = self.create_sharded_train_state(params)
                # print("params")
                # print(sharded_state.params)
                # print("ref_params")
                # print(sharded_state.ref_params)
            else:
                def create_train_state(params, ref_params):
                    return DPOTrainState.create(
                        apply_fn=None,
                        params=params,
                        ref_params=ref_params,
                        tx=self.optimizer
                    )
                self.create_sharded_train_state = pjit(
                    create_train_state,
                    in_shardings=(partition_specs.params, partition_specs.ref_params),
                    out_shardings=partition_specs,
                    donate_argnums=(0,1)
                )
                sharded_state = self.create_sharded_train_state(params, ref_params)
            

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


        self.state_partition_specs = partition_specs
        self.sharded_state = sharded_state
        self.shard_fns = shard_fns
        self.gather_fns = gather_fns
        self.task.params = None