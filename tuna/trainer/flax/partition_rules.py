from jax.sharding import PartitionSpec
import transformers as htf

def get_gemma(fully_sharded_data_parallel: bool = True):
    """
    The get_partition_rules function is used to define the partitioning scheme for a model.
    It returns a list of tuples, where each tuple contains two elements:
        1) A regex string that matches the name of one or more parameters in the model.
        2) A PartitionScheme object that defines how those parameters should be partitioned across devices.

    :param fully_sharded_data_parallel: bool: Determine whether to partition the model fully or not
    :return: A list of tuples

    """
    return (

        ("model/embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),

        ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
        ("self_attn/o_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),

        ("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
        ("mlp/down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
        ("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),

        ("input_layernorm/kernel", PartitionSpec(None)),
        ("post_attention_layernorm/kernel", PartitionSpec(None)),

        ("model/norm/kernel", PartitionSpec(None)),
        ("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
        (".*", PartitionSpec(None)),
    ) if not fully_sharded_data_parallel else (

        ("model/embed_tokens/embedding", PartitionSpec(("fsdp", "sp"))),

        ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"))),
        ("self_attn/o_proj/kernel", PartitionSpec(("fsdp", "sp"))),

        ("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"))),
        ("mlp/down_proj/kernel", PartitionSpec(("fsdp", "sp"))),
        ("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"))),

        ("input_layernorm/kernel", PartitionSpec(None)),
        ("post_attention_layernorm/kernel", PartitionSpec(None)),

        ("model/norm/kernel", PartitionSpec(None)),
        ("lm_head/kernel", PartitionSpec(("fsdp", "sp"))),
        (".*", PartitionSpec(("fsdp", "sp"))),
    )

def get_llama(fully_sharded_data_parallel: bool = True):
    """
    The get_partition_rules function is used to define the partitioning scheme for a model.
    It returns a list of tuples, where each tuple contains two elements:
        1) A regex string that matches the name of one or more parameters in the model.
        2) A PartitionScheme object that defines how those parameters should be partitioned across devices.

    :param fully_sharded_data_parallel: bool: Determine whether to partition the model fully or not
    :return: A list of tuples

    """
    return (

        ("model/embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),

        ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
        ("self_attn/o_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),

        ("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
        ("mlp/down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
        ("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),

        ("input_layernorm/kernel", PartitionSpec(None)),
        ("post_attention_layernorm/kernel", PartitionSpec(None)),

        ("model/norm/kernel", PartitionSpec(None)),
        ("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
        (".*", PartitionSpec(None)),
    ) if not fully_sharded_data_parallel else (

        ("model/embed_tokens/embedding", PartitionSpec(("fsdp", "sp"))),

        ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"))),
        ("self_attn/o_proj/kernel", PartitionSpec(("fsdp", "sp"))),

        ("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"))),
        ("mlp/down_proj/kernel", PartitionSpec(("fsdp", "sp"))),
        ("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"))),

        ("input_layernorm/kernel", PartitionSpec(None)),
        ("post_attention_layernorm/kernel", PartitionSpec(None)),

        ("model/norm/kernel", PartitionSpec(None)),
        ("lm_head/kernel", PartitionSpec(("fsdp", "sp"))),
        (".*", PartitionSpec(("fsdp", "sp"))),
    )

def get_mistral(fully_sharded_data_parallel: bool = True):
    """
    The get_partition_rules function is used to define the partitioning scheme for a model.
    It returns a list of tuples, where each tuple contains two elements:
        1) A regex string that matches the name of one or more parameters in the model.
        2) A PartitionScheme object that defines how those parameters should be partitioned.

    :param fully_sharded_data_parallel: bool: Determine whether to use the fully_sharded_data_parallel partitioning scheme or not
    :return: A list of tuples

    """
    return (

        ("model/embed_tokens/embedding", PartitionSpec("dp", "fsdp")),

        ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec("fsdp", "dp")),
        ("self_attn/o_proj/kernel", PartitionSpec("dp", "fsdp")),

        ("mlp/gate_proj/kernel", PartitionSpec("fsdp", "dp")),
        ("mlp/down_proj/kernel", PartitionSpec("dp", "fsdp")),
        ("mlp/up_proj/kernel", PartitionSpec("fsdp", "dp")),

        ("input_layernorm/kernel", PartitionSpec(None)),
        ("post_attention_layernorm/kernel", PartitionSpec(None)),

        ("model/norm/kernel", PartitionSpec(None)),
        ("lm_head/kernel", PartitionSpec("fsdp", "dp")),
        (".*", PartitionSpec(None)),
    ) if not fully_sharded_data_parallel else (
        ("model/embed_tokens/embedding", PartitionSpec(("fsdp", "sp"))),

        ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"))),
        ("self_attn/o_proj/kernel", PartitionSpec(("fsdp", "sp"))),

        ("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"))),
        ("mlp/down_proj/kernel", PartitionSpec(("fsdp", "sp"))),
        ("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"))),

        ("input_layernorm/kernel", PartitionSpec(None)),
        ("post_attention_layernorm/kernel", PartitionSpec(None)),

        ("model/norm/kernel", PartitionSpec(None)),
        ("lm_head/kernel", PartitionSpec(("fsdp", "sp"))),
        (".*", PartitionSpec(("fsdp", "sp"))),
    )

RULES = {
    (htf.GemmaConfig, get_gemma),
    (htf.LlamaConfig, get_llama),
    (htf.MistralConfig, get_mistral),
}

def get_partition_rules(config, fully_sharded_data_parallel):
    for k, v in RULES:
        if isinstance(config, k):
            return v(fully_sharded_data_parallel=fully_sharded_data_parallel)
    
    raise ValueError(f"Unsupported config type: {type(config)}")