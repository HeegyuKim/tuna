from functools import partial
from typing import Optional, Tuple, Union

import chex
import fjformer
import flax.linen.partitioning
import jax
import jax.numpy as jnp
from fjformer import linen as nn
from fjformer import with_sharding_constraint
from fjformer.linen import Dense
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.linen.dtypes import promote_dtype
from flax.typing import (
  Array,
  PRNGKey,
  Dtype,
  Shape as Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
)

from jax import lax
from jax.sharding import PartitionSpec

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput, FlaxSequenceClassifierOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from transformers import Gemma2Config


logger = logging.get_logger(__name__)


def add_positional_embedding(
    input_embedding: jax.Array,
    position: int,
    theta: int = 10_000,
) -> jax.Array:
    """Adds positional embeddings to input embeddings. From DeepMind Gemma"""
    embed_dim = input_embedding.shape[-1]
    num_timescales = embed_dim // 2
    log_timescale_increment = jnp.log(float(theta)) / jnp.maximum(
        jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1
    )
    inv_timescales = jnp.exp(
        jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
    )
    scaled_time = position * inv_timescales
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)])
    signal = jnp.pad(signal, [[0, jnp.mod(embed_dim, 2)]])
    position_embedding = signal.astype(jnp.float32)

    return input_embedding + position_embedding


def apply_rope(
    inputs: jax.Array,  # [B, L]
    positions: jax.Array,  # [B, L]
    head_dim: int,
    theta: int = 10_000,
) -> jax.Array:
    """Applies RoPE. From DeepMind Gemma"""
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = theta**fraction

    sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


class FlaxGemma2RMSNorm(nn.Module):
    config: Gemma2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.epsilon = self.config.rms_norm_eps
        self.weight_kernel = self.param(
            "kernel", lambda _, shape: jnp.ones(shape), self.config.hidden_size
        )

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)
        w = 1 + nn.linen.control_quantization(self.weight_kernel, self.dtype)
        return (w * jnp.asarray(hidden_states, dtype=self.dtype)).astype(
            hidden_states.dtype
        )


def rotate_half(x):
    """The rotate_half function takes a complex-valued array and rotates the
    phase of its second half by 180 degrees. This is equivalent to multiplying
    the second half by -i, or equivalently rotating it 90 degrees counterclockwise.

    Args:
        x: Specify the input array

    Returns:
        A new array that is the same as the input
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jax.numpy.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb(tensor, sin_, cos_):
    """The apply_rotary_pos_emb function applies a rotary positional embedding to the input tensor.
    b,h,s,d or pytorch style

    Args:
        tensor: Store the tensor that is passed into the function
        sin_: Rotate the tensor by pi/2
        cos_: Apply the cosine function to the tensor

    Returns:
        A tensor with the same shape as the input tensor
    """
    b, h, s, d = tensor.shape
    return (tensor * cos_[:, :, :s, :]) + (rotate_half(tensor) * sin_[:, :, :s, :])

class FlaxGemma2RotaryEmbedding(nn.Module):
    config: Gemma2Config
    dtype: jnp.dtype = jnp.float32

    def __call__(self, freq_cis, key_states, query_states, position_ids):
        sin, cos = freq_cis

        sin = sin[position_ids][:, None, :, :]
        cos = cos[position_ids][:, None, :, :]

        key = apply_rotary_pos_emb(key_states, sin, cos)
        query = apply_rotary_pos_emb(query_states, sin, cos)

        return query.astype(self.dtype), key.astype(self.dtype)


def dot_product_attention_weights_softcapping(
    query: Array,
    key: Array,
    softcap: float = None,
    scaling: float = 1.0,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    module: Optional[nn.Module] = None,
    ):
    """Computes dot-product attention weights given query and key.

    Used by :func:`dot_product_attention`, which is what you'll most likely use.
    But if you want access to the attention weights for introspection, then
    you can directly call this function and call einsum yourself.

    Args:
    query: queries for calculating attention with shape of ``[batch..., q_length,
        num_heads, qk_depth_per_head]``.
    key: keys for calculating attention with shape of ``[batch..., kv_length,
        num_heads, qk_depth_per_head]``.
    bias: bias for the attention weights. This should be broadcastable to the
        shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
        incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
        shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
        incorporating causal masks. Attention weights are masked out if their
        corresponding mask value is ``False``.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs and params)
    precision: numerical precision of the computation see ``jax.lax.Precision``
        for details.
    module: the Module that will sow the attention weights into the
        'intermediates' collection. Remember to mark 'intermediates' as mutable via
        ``mutable=['intermediates']`` in order to have that collection returned.
        If ``module`` is None, the attention weights will not be sowed.

    Returns:
    Output of shape ``[batch..., num_heads, q_length, kv_length]``.
    """
    query, key = promote_dtype(query, key, dtype=dtype)
    dtype = query.dtype

    assert query.ndim == key.ndim, 'q, k must have same rank.'
    assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
    assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

    # calculate attention matrix
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = jnp.einsum(
    '...qhd,...khd->...hqk', query, key, precision=precision
    )
    attn_weights = attn_weights * scaling

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias
    # apply attention mask
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    if softcap:
        attn_weights = attn_weights / softcap
        attn_weights = jnp.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    # normalize the attention weights
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    if module:
        module.sow('intermediates', 'attention_weights', attn_weights)

    # apply attention dropout
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            # dropout is broadcast across the batch + head dimensions
            dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
            keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
        else:
            keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attn_weights = attn_weights * multiplier

    return attn_weights


class FlaxGemma2Attention(nn.Module):
    config: Gemma2Config
    layer_idx: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")
    causal: bool = True
    is_cross_attention: bool = False

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        kernel = jax.nn.initializers.normal(self.config.initializer_range)

        dense_class = partial(
            Dense,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=kernel,
        )
        self.q_proj = dense_class(self.num_heads * self.head_dim)
        self.k_proj = dense_class(self.num_key_value_heads * self.head_dim)
        self.v_proj = dense_class(self.num_key_value_heads * self.head_dim)
        self.o_proj = dense_class(self.embed_dim)
        self.sliding_window = (
            config.sliding_window if (self.layer_idx % 2 == 0) else None
        )

        self.rotary_emb = FlaxGemma2RotaryEmbedding(config, dtype=self.dtype)

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.num_heads * self.head_dim,)
        )

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (num_heads, self.head_dim)
        )

    def apply_rotary(
        self, batch_size, sequence_length, query, key, value, freq_cis, position_ids
    ):
        """The apply_rotary function is a modified version of the apply_attention function in the BertModel class.
        The main difference is that it takes in an additional argument, freq_cis, which are used to calculate
        the rotary attention weights. The other differences are minor and mostly related to reshaping tensors.

        Args:
            self: Access variables that belong to the class
            batch_size: Reshape the query, key and value tensors
            sequence_length: Reshape the query, key and value tensors
            query: Calculate the attention weights
            key: Calculate the attention
            value: Compute the attention weights
            freq_cis: Calculate the frequency of each word in the
                vocabulary
            position_ids: Identify the position of each token in the
                sequence

        Returns:
            A tuple of 3 tensors: query, key and value
        """
        query, key, value = self._transpose_sequence_head(query, key, value)
        query, key = self.rotary_emb(
            position_ids=position_ids,
            query_states=query,
            key_states=key,
            freq_cis=freq_cis,
        )
        return self._transpose_sequence_head(query, key, value)

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        freq_cis: Tuple[chex.Array, chex.Array],
        causal_mask: chex.Array,
        segment_ids: Optional[chex.Array] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        batch_size, sequence_length = hidden_states.shape[:2]
        (query_states, key_states, value_states) = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        query_states = query_states.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        query_states, key_states, value_states = self.apply_rotary(
            query_states.shape[0],
            query_states.shape[1],
            query_states,
            key_states,
            value_states,
            freq_cis,
            position_ids,
        )

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                causal_mask,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length),
            )
        else:
            causal_mask = causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:]
        )

        attention_mask = jnp.broadcast_to(
            jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
        )
        attention_mask = combine_masks(attention_mask, causal_mask)

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        if self.has_variable("cache", "cached_key") or init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        key_states, value_states = self.repeat_key_value(
            key_states, value_states, self.num_key_value_groups
        )

        if bool((self.layer_idx % 2) == 0):
            sliding_window_mask = jnp.tril(
                jnp.ones_like(attention_mask, dtype=jnp.bool),
                k=-self.sliding_window,
            )
            window_mask = jnp.where(sliding_window_mask, 0, 1)
            attention_mask = jnp.logical_and(window_mask, attention_mask)
        
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                self.dtype
            ),
        )
        
        if bool((self.layer_idx % 2) == 0):
            if attention_bias.shape[-1] <= 1:  # when decoding
                attention_bias = attention_bias[:, :, :, -self.sliding_window :]

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        attentions = self.attention_performer(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            bias=attention_bias,
            attention_mask=attention_mask,
            causal=True,
            dropout_rng=dropout_rng,
            deterministic=deterministic,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            uses_cache=self.has_variable("cache", "cached_key") or init_cache,
            segment_ids=segment_ids,
            causal_mask=causal_mask,
        )

        attn_output = self._merge_heads(attentions.attention_outputs)
        attn_output = self.o_proj(attn_output)

        # jax.debug.print("{}", attn_output[0, 0, :5])
        return (
            (attn_output, attentions.attention_weights)
            if output_attentions
            else (attn_output, None)
        )


class FlaxGemma2MLP(nn.Module):
    config: Gemma2Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self):

        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)

        self.act = ACT2FN[self.config.hidden_activation]
        dense_class = partial(
            Dense,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=kernel_init,
        )
        self.gate_proj = dense_class(self.config.intermediate_size)
        self.up_proj = dense_class(self.config.intermediate_size)
        self.down_proj = dense_class(self.config.hidden_size)

    def __call__(self, hidden_states, deterministic=False):
        return self.down_proj(
            self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class FlaxGemma2DecoderLayer(nn.Module):
    config: Gemma2Config
    layer_idx: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self):
        mlp_block = FlaxGemma2MLP
        attn_block = FlaxGemma2Attention

        self.is_sliding = bool(self.layer_idx % 2)
        self.self_attn = attn_block(
            self.config,
            layer_idx=self.layer_idx,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.mlp = mlp_block(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )

        self.input_layernorm = FlaxGemma2RMSNorm(
            self.config,
            dtype=self.dtype,
        )
        self.post_attention_layernorm = FlaxGemma2RMSNorm(
            self.config,
            dtype=self.dtype,
        )
        self.pre_feedforward_layernorm = FlaxGemma2RMSNorm(
            self.config,
            dtype=self.dtype,
        )
        self.post_feedforward_layernorm = FlaxGemma2RMSNorm(
            self.config,
            dtype=self.dtype,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        freq_cis: Tuple[chex.Array, chex.Array],
        causal_mask: chex.Array,
        segment_ids: Optional[chex.Array] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weight = self.self_attn(
            hidden_states,
            attention_mask,
            position_ids,
            freq_cis,
            causal_mask,
            segment_ids,
            deterministic,
            init_cache,
            output_attentions,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        
        hidden_states = self.mlp(hidden_states, deterministic)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weight


class FlaxGemma2PreTrainedModel(EDPretrainedModel):
    """An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Gemma2Config
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: Gemma2Config,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision(
            "fastest"
        ),
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(
            config=config,
            dtype=dtype,
            precision=precision,
            param_dtype=param_dtype,
            **kwargs,
        )
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_cache(self, batch_size, max_length):
        """The init_cache function is used to initialize the cache for a given batch size and sequence length.
        The cache is a dictionary that contains all the intermediate states from each layer in the model.
        This allows us to run inference on multiple batches without having to re-run forward passes through every layer in
        the model, which would be very slow.

        Args:
            self: Access the module
            batch_size: Define the batch size of the input tensors
            max_length: Set the length of the input sequence

        Returns:
            A dictionary with the following keys:
        """
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
        )

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            input_ids,
            attention_mask,
            position_ids,
            return_dict=False,
            init_cache=True,
        )
        return init_variables["cache"]

    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None
    ) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape
        )
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs, input_ids, attention_mask, position_ids, return_dict=False
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array = None,
        position_ids: chex.Array = None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        extra_embedding: Optional[Union[jnp.ndarray, None]] = None,
        add_params_field: bool = False,
        **kwargs,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `position_ids` when passing `past_key_values`."
                )

            position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = (
            {"params": params or self.params}
            if add_params_field
            else params or self.params
        )

        if past_key_values is not None:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxGemma2LayerCollection(nn.Module):
    config: Gemma2Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self):
        self.blocks = [
            FlaxGemma2DecoderLayer(
                self.config,
                layer_idx=i,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(i),
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        freq_cis: Tuple[chex.Array, chex.Array],
        causal_mask: chex.Array,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                freq_cis=freq_cis,
                causal_mask=causal_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxGemma2Module(nn.Module):
    config: Gemma2Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.layers = FlaxGemma2LayerCollection(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.norm = FlaxGemma2RMSNorm(
            self.config,
            dtype=self.dtype,
        )
        self.freq_cis = precompute_freq_cis(
            max_position_embeddings=self.config.max_position_embeddings,
            dim=self.config.head_dim,
            base=self.config.rope_theta,
        )
        self.causal_mask = make_causal_mask(
            jnp.ones((1, self.config.max_position_embeddings), dtype="bool"),
            dtype="bool",
        )

    # Ignore copy
    def __call__(
        self,
        input_ids,
        attention_mask: Optional[chex.Array] = None,
        position_ids: chex.Array = None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_embeds = self.embed_tokens(input_ids.astype("i4"))

        input_embeds = input_embeds * jnp.asarray(
            self.config.hidden_size**0.5, dtype=input_embeds.dtype
        )

        outputs = self.layers(
            input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            freq_cis=self.freq_cis,
            causal_mask=self.causal_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


class FlaxGemma2Model(FlaxGemma2PreTrainedModel):
    module_class = FlaxGemma2Module


class FlaxGemma2ForCausalLMModule(nn.Module):
    config: Gemma2Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self):
        self.model = FlaxGemma2Module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.lm_head = Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
        )

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[chex.Array] = None,
        position_ids: chex.Array = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_kernel = self.model.variables["params"]["embed_tokens"]["embedding"]
            shared_kernel = fjformer.linen.control_quantization(
                shared_kernel, self.param_dtype
            ).T
            lm_logits = self.lm_head.apply(
                {"params": {"kernel": shared_kernel}}, hidden_states
            )
        else:
            lm_logits = self.lm_head(hidden_states)

        if self.config.final_logit_softcapping is not None:
            lm_logits = lm_logits / self.config.final_logit_softcapping
            lm_logits = jax.nn.tanh(lm_logits)
            lm_logits = lm_logits * self.config.final_logit_softcapping
        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxGemma2ForCausalLM(FlaxGemma2PreTrainedModel):
    module_class = FlaxGemma2ForCausalLMModule

    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: Optional[jax.Array] = None
    ):

        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, attention_mask, (0, 0)
            )
        else:
            position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
            )

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs

    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None
    ) -> FrozenDict:

        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape
        )
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs, input_ids, attention_mask, position_ids, return_dict=False
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params