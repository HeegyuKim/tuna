import optax
import jax.numpy as jnp


def get_optimizer(
    init_lr: float,
    lr: float,
    lr_warmup_steps: int,
    lr_decay_steps: int,
    end_lr: float,
    clip_gradient: float,
    gradient_accumulation_steps: int,
    beta1: float,
    beta2: float,
):