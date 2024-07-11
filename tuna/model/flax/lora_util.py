from typing import Any, List
from transformers import GPTNeoXConfig, T5Config, LlamaConfig, MistralConfig, LlavaConfig, GemmaConfig
# import jax.numpy as jnp
# from flax.linen.dtypes import canonicalize_dtype
# from fjformer.xrapture import LoraWeight


GPTNEOX_TARGETS = [
    "query_key_value",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
]

T5_TARGETS = [
    "q", "k", "v", "o"
    "w", "wi_0", "wi_1", "wo",
]

LLAMA_TARGETS = [
    "q_proj", "v_proj", "o_proj", "k_proj",
    "gate_proj", "up_proj", "down_proj"
]

MISTRAL_TARGETS = [
    "q_proj", "v_proj", "o_proj", "k_proj", 
    "gate_proj", "up_proj", "down_proj"
]

PHI3_TARGETS = [
    "qkv_proj", "o_proj",
    "gate_up_proj", "down_proj"
]

LLAVA_LM_TARGETS = [
    f"language_model.+{x}"
    for x in [
        "q_proj", "v_proj", "o_proj", "k_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
]

LORA_TARGETS = [
    (GPTNeoXConfig, GPTNEOX_TARGETS),
    (T5Config, T5_TARGETS),
    (LlamaConfig, LLAMA_TARGETS),
    (MistralConfig, MISTRAL_TARGETS),
    ("Qwen2Config", MISTRAL_TARGETS),
    (GemmaConfig, LLAMA_TARGETS),
    ("Phi3Config", PHI3_TARGETS)
]

def find_lora_targets(model):
    for config, rule in LORA_TARGETS:
        if isinstance(config, str) and model.config.__class__.__name__ == config:
            return rule
        elif model.config.__class__ == config:
            return rule
    raise Exception("unsupported model to lora targeting")

def find_lora_targets_from_config(model_config):
    for config, rule in LORA_TARGETS:
        if isinstance(config, str) and config.__class__.__name__ == config:
            return rule
        if model_config.__class__ == config:
            return rule
    raise Exception("unsupported model to lora targeting")



# def promote_dtype_lora_compat(*args, dtype=None, inexact=True) -> List[Any]:
#     """ "Promotes input arguments to a specified or inferred dtype.

#     All args are cast to the same dtype. See ``canonicalize_dtype`` for how
#     this dtype is determined.

#     The behavior of promote_dtype is mostly a convinience wrapper around
#     ``jax.numpy.promote_types``. The differences being that it automatically casts
#     all input to the inferred dtypes, allows inference to be overridden by a
#     forced dtype, and has an optional check to garantuee the resulting dtype is
#     inexact.

#     Args:
#     *args: JAX array compatible values. None values are returned as is.
#     dtype: Optional dtype override. If specified the arguments are cast to the
#         specified dtype instead and dtype inference is disabled.
#     inexact: When True, the output dtype must be a subdtype of `jnp.inexact`.
#         Inexact dtypes are real or complex floating points. This is useful when
#         you want to apply operations that don't work directly on integers like
#         taking a mean for example.

#     Returns:
#     The arguments cast to arrays of the same dtype.
#     """
#     dtype = jnp.canonicalize_dtype(*args, dtype=dtype, inexact=inexact)
#     outs = []
#     for x in args:
#         if isinstance(x, LoraWeight):
#             outs.append(
#                 LoraWeight(
#                     w=x.w.astype(dtype),
#                     a=x.a.astype(dtype),
#                     b=x.b.astype(dtype),
#                 )
#             )
#         else:
#             outs.append(jnp.asarray(x, dtype) if x is not None else None)
#     return outs
