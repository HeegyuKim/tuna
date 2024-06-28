from typing import Optional
import chex
import optax
import jax.numpy as jnp


def get_scheduler(
    steps: int,
    learning_rate_start: float = 5e-5,
    learning_rate_end: float = 1e-5,
    warmup_steps: int = 0,
    scheduler: str = 'linear'
):
    if warmup_steps > 0:
        scheduler_warmup = optax.linear_schedule(init_value=5e-8, end_value=learning_rate_start,
                                                transition_steps=warmup_steps)
    else:
        scheduler_warmup = None

    if scheduler is None:
        scheduler_decay = None
    elif scheduler == 'linear':
        scheduler_decay = optax.linear_schedule(init_value=learning_rate_start, end_value=learning_rate_end,
                                                transition_steps=steps - warmup_steps)
    elif scheduler == 'cosine':
        scheduler_decay = optax.cosine_decay_schedule(
            init_value=learning_rate_start,
            alpha=learning_rate_end / learning_rate_start,
            decay_steps=steps,
        )
    else:
        raise ValueError(f"unknown scheduler: {scheduler}")
    
    if scheduler_warmup and scheduler_decay:
        return optax.join_schedules(schedules=[scheduler_warmup, scheduler_decay], boundaries=[warmup_steps])
    elif scheduler_warmup:
        return scheduler_warmup
    elif scheduler_decay:
        return scheduler_decay
    else:
        return learning_rate_start


def get_adamw(
        steps: int,
        learning_rate_start: float = 5e-5,
        learning_rate_end: float = 1e-5,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        weight_decay: float = 1e-4,
        gradient_accumulation_steps: int = 1,
        mu_dtype: Optional[chex.ArrayDType] = None,
        gradient_clipping: float = 1.0,
        warmup_steps: int = 0,
        scheduler: str = "linear"
):
    scheduler = get_scheduler(steps, learning_rate_start, learning_rate_end, warmup_steps, scheduler)

    tx = optax.chain(
        optax.clip_by_global_norm(gradient_clipping),
        optax.adamw(
            learning_rate=scheduler,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            weight_decay=weight_decay
        ),
    )

    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler


def get_adafactor(
        steps: int,
        learning_rate_start: float = 5e-5,
        learning_rate_end: float = 1e-5,
        min_dim_size_to_factor: int = 128,
        decay_rate: float = 0.8,
        decay_offset: int = 0,
        multiply_by_parameter_scale: float = True,
        clipping_threshold: Optional[float] = 1.0,
        momentum: Optional[float] = None,
        dtype_momentum: chex.ArrayDType = jnp.float32,
        weight_decay_rate: Optional[float] = None,
        eps: float = 1e-30,
        factored: bool = True,
        gradient_accumulation_steps: int = 1,
        gradient_clipping: float = 1.0,
        warmup_steps: int = 0,
        scheduler: str = "linear"
):
    scheduler = get_scheduler(steps, learning_rate_start, learning_rate_end, warmup_steps, scheduler)

    tx = optax.chain(
        optax.clip_by_global_norm(gradient_clipping),
        optax.adafactor(
            learning_rate=scheduler,
            min_dim_size_to_factor=min_dim_size_to_factor,
            decay_rate=decay_rate,
            decay_offset=decay_offset,
            multiply_by_parameter_scale=multiply_by_parameter_scale,
            clipping_threshold=clipping_threshold,
            eps=eps,
            momentum=momentum,
            weight_decay_rate=weight_decay_rate,
            dtype_momentum=dtype_momentum,
            factored=factored
        ),
        optax.scale(-1)
    )
    
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler


def get_adafactor_with_warmup_linear_scheduler(
        steps: int,
        min_dim_size_to_factor: int = 128,
        decay_rate: float = 0.8,
        decay_offset: int = 0,
        multiply_by_parameter_scale: float = True,
        clipping_threshold: Optional[float] = 1.0,
        momentum: Optional[float] = None,
        dtype_momentum: chex.ArrayDType = jnp.float32,
        weight_decay_rate: Optional[float] = None,
        eps: float = 1e-30,
        factored: bool = True,
        gradient_accumulation_steps: int = 1,
        learning_rate_start: float = 5e-5,
        learning_rate_end: float = 1e-5,
        warmup_steps: int = 500
):
    """
    :param min_dim_size_to_factor:
    :param decay_rate:
    :param decay_offset:
    :param multiply_by_parameter_scale:
    :param clipping_threshold:
    :param momentum:
    :param dtype_momentum:
    :param weight_decay_rate:
    :param factored:
    :param warmup_steps:
    :param gradient_accumulation_steps:
    :param steps:
    :param learning_rate_start:
    :param learning_rate_end:
    :param eps:
    :param weight_decay:

     # New parameter for warmup
     @warmup_steps (int): Number of steps for the warmup phase

     # return Optimizer and Scheduler with WarmUp feature
   """

    tx = optax.chain(
        optax.adafactor(
            learning_rate=scheduler_combined,
            min_dim_size_to_factor=min_dim_size_to_factor,
            decay_rate=decay_rate,
            decay_offset=decay_offset,
            multiply_by_parameter_scale=multiply_by_parameter_scale,
            clipping_threshold=clipping_threshold,
            eps=eps,
            momentum=momentum,
            weight_decay_rate=weight_decay_rate,
            dtype_momentum=dtype_momentum,
            factored=factored
        )
    )
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler_combined

