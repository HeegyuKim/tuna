import jax
import jax.numpy as jnp
from fjformer.pallas_operations.flash_attention.tpu import flash_attention
from flax.linen.attention import dot_product_attention_weights
from lucidrains_flash_attention import causal_flash_attention

def causal_mask(shape):
    mask = jnp.full(shape, -jnp.inf)
    mask = jnp.triu(mask, k=1)
    return mask

# [batch_size, num_heads, q_seq_len, d_model]
shape = (4, 2048, 8, 64)
key = jax.random.PRNGKey(1701)
Q = jax.random.normal(jax.random.PRNGKey(1701), shape)
K = jax.random.normal(jax.random.PRNGKey(1702), shape)
V = jax.random.normal(jax.random.PRNGKey(1703), shape)

attn_bias = causal_mask((4, 8, 2048, 2048))
print(attn_bias.shape)

attn_weights = dot_product_attention_weights(
    Q,
    K,
    bias=attn_bias
)
attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, V)
print("default attn_output", attn_output.shape)

# flash_output = flash_attention(
flash_output = causal_flash_attention(
    jnp.transpose(Q, (0, 2, 1, 3)),
    jnp.transpose(K, (0, 2, 1, 3)),
    jnp.transpose(V, (0, 2, 1, 3)),
    # causal=True
)
# print(flash_output.shape)
print("flash attn_output", flash_output.shape)
flash_output = jnp.transpose(flash_output, (0, 2, 1, 3))
mse = jnp.pow(flash_output - attn_output, 2).mean()
print("MSE", mse)