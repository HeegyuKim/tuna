import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable, NamedTuple, Optional, Union

class AdamMiniState(NamedTuple):
    count: jnp.ndarray
    m: Any
    v: Any
    vmean: Any

def adam_mini(
    learning_rate: float = 1.0,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    mask: Optional[Union[Any, Callable[[optax.Params], Any]]] = None,
    n_embd: int = 2048,
    n_head: int = 32,
    n_query_groups: Optional[int] = None
):
    n_query_groups = n_query_groups or n_head
    assert n_head % n_query_groups == 0

    def init_fn(params):
        m = jax.tree_map(jnp.zeros_like, params)
        v = jax.tree_map(jnp.zeros_like, params)
        vmean = jax.tree_map(lambda p: jnp.zeros((), dtype=p.dtype), params)
        return AdamMiniState(count=jnp.zeros([], jnp.int32), m=m, v=v, vmean=vmean)

    def update_fn(updates, state, params):
        count = state.count + 1
        lr = learning_rate
        beta1 = b1
        beta2 = b2

        def update_param(g, p, m, v, vmean, name):
            if 'embed_tokens' in name or 'wte' in name or 'lm_head' in name:
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * jnp.square(g)
                vmean = v

            elif 'self_attn.k_proj.weight' in name or 'self_attn.q_proj.weight' in name or 'attn.wq.weight' in name or 'attn.wk.weight' in name:
                dim = n_embd * n_embd // n_head
                head = g.shape[0] // dim
                g = g.reshape(head, dim)
                m = m.reshape(head, dim)
                tmp_lr = jnp.mean(jnp.square(g), axis=1)
                vmean = beta2 * vmean + (1 - beta2) * tmp_lr
                v = vmean.reshape(-1, 1)
                m = beta1 * m + (1 - beta1) * g
                m = m.reshape(-1)

            elif 'attn.attn.weight' in name or 'attn.qkv.weight' in name:
                g = g.reshape(n_head, n_head // n_query_groups + 2, -1)
                m = m.reshape(n_head, n_head // n_query_groups + 2, -1)
                tmp_lr = jnp.mean(jnp.square(g), axis=2)
                vmean = beta2 * vmean + (1 - beta2) * tmp_lr
                v = vmean.reshape(-1, 1)
                m = beta1 * m + (1 - beta1) * g
                m = m.reshape(-1)

            else:
                m = beta1 * m + (1 - beta1) * g
                tmp_lr = jnp.sum(jnp.square(g)) / g.size
                vmean = beta2 * vmean + (1 - beta2) * tmp_lr
                v = vmean

            bias_correction1 = 1 - beta1 ** count
            bias_correction2 = 1 - beta2 ** count
            step_size = lr / bias_correction1
            bias_correction2_sqrt = jnp.sqrt(bias_correction2)

            denom = jnp.sqrt(v) / bias_correction2_sqrt + eps
            update = step_size * m / denom

            if weight_decay != 0:
                update += weight_decay * p

            new_p = p - update
            return new_p, m, v, vmean

        updates, new_m, new_v, new_vmean = jax.tree_map(
            lambda g, p, m, v, vmean, name: update_param(g, p, m, v, vmean, name),
            updates, params, state.m, state.v, state.vmean, jax.tree_map(lambda _: '', params)
        )

        new_state = AdamMiniState(count=count, m=new_m, v=new_v, vmean=new_vmean)
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)

