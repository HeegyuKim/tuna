import torch
import torch.nn as nn
import re
import torch_xla.distributed.spmd.xla_sharding as xs
import torch_xla.core.xla_model as xm
import transformers as tf
from transformers import (
    GPTNeoXConfig, T5Config, LlamaConfig, RobertaConfig, MistralConfig, LlavaConfig, CLIPConfig, CLIPVisionConfig, GemmaConfig, Gemma2Config,
    XLMRobertaConfig
)

GPTNEOX_RULES = (
    # embeddings
    ("gpt_neox\\.embed_in", ("mp", ("fsdp", "sp"))),
    # atention
    ("attention\\.query_key_value", (("fsdp", "sp"), "mp")),
    ("attention\\.dense", ("mp", ("fsdp", "sp"))),
    # mlp
    ("mlp\\.dense_h_to_4h", (("fsdp", "sp"), "mp")),
    ("mlp\\.dense_4h_to_h", ("mp", ("fsdp", "sp"))),
    # output
    ("embed_out", (("fsdp", "sp"), "mp")),
    # GTA2
    ("heads.\d+", (("fsdp", "sp"), "mp")),
    ("v_heads.\d+", (("fsdp", "sp"), "mp")),
    # PKU score model
    ("score_head", (("fsdp", "sp"), "mp")),
)

T5_RULES = (
    # embeddings
    ("shared", ("mp", ("fsdp", "sp"))),
    ("embed_tokens", ("mp", ("fsdp", "sp"))),
    
    # attention
    ("q", (("fsdp", "sp"), "mp")),
    ("k", (("fsdp", "sp"), "mp")),
    ("v", (("fsdp", "sp"), "mp")),
    ("o", ("mp", ("fsdp", "sp"))),

    # mlp
    ("w", (("fsdp", "sp"), "mp")),
    ("wi_0", (("fsdp", "sp"), "mp")),
    ("wi_1", (("fsdp", "sp"), "mp")),
    ("wo", ("mp", ("fsdp", "sp"))),

    # seq2seq lm head
    ("lm_head", (("fsdp", "sp"), "mp")),
)

LLAMA_RULES = (
    ("model\\.embed_tokens", ("mp", ("fsdp", "sp"))),
    ("self_attn\\.(q_proj|k_proj|v_proj)", (("fsdp", "sp"), "mp")),
    ("self_attn\\.o_proj", ("mp", ("fsdp", "sp"))),
    ("mlp\\.gate_proj", (("fsdp", "sp"), "mp")),
    ("mlp\\.down_proj", ("mp", ("fsdp", "sp"))),
    ("mlp\\.up_proj", (("fsdp", "sp"), "mp")),
    ("lm_head", (("fsdp", "sp"), "mp")),
    ("score", (("fsdp", "sp"), "mp")),
    # GTA2
    ("heads.\d+", (("fsdp", "sp"), "mp")),
    ("v_heads.\d+", (("fsdp", "sp"), "mp")),
    # PKU score model
    ("score_head", (("fsdp", "sp"), "mp")),
    )

MISTRAL_RULES = (
    ("model\\.embed_tokens", ("mp", ("fsdp", "sp"))),
    ("self_attn\\.(q_proj|k_proj|v_proj)", (("fsdp", "sp"), "mp")),
    ("self_attn\\.o_proj", ("mp", ("fsdp", "sp"))),
    ("mlp\\.gate_proj", (("fsdp", "sp"), "mp")),
    ("mlp\\.down_proj", ("mp", ("fsdp", "sp"))),
    ("mlp\\.up_proj", (("fsdp", "sp"), "mp")),
    ("lm_head", (("fsdp", "sp"), "mp")),
    ("score", (("fsdp", "sp"), "mp")),
    # GTA2
    ("heads.\d+", (("fsdp", "sp"), "mp")),
    ("v_heads.\d+", (("fsdp", "sp"), "mp")),
    # PKU score model
    ("score_head", (("fsdp", "sp"), "mp")),
    )
    
PHI3_RULES = (
    ("model\\.embed_tokens", ("mp", ("fsdp", "sp"))),
    ("self_attn\\.qkv_proj", (("fsdp", "sp"), "mp")),
    ("self_attn\\.o_proj", ("mp", ("fsdp", "sp"))),
    ("mlp\\.gate_up_proj", (("fsdp", "sp"), "mp")),
    ("mlp\\.down_proj", ("mp", ("fsdp", "sp"))),
    ("lm_head", (("fsdp", "sp"), "mp")),
    ("score", (("fsdp", "sp"), "mp")),
    )
    
ROBERTA_RULES = (
    ("embeddings", ("mp", ("fsdp", "sp"))),
    ("attention\\.self\\.(query|key|value)", (("fsdp", "sp"), "mp")),
    ("attention\\.output\\.dense", ("mp", ("fsdp", "sp"))),
    ("intermediate\\.dense", (("fsdp", "sp"), "mp")),
    ("output\\.dense", ("mp", ("fsdp", "sp"))),
    ("classifier\\.dense", (("fsdp", "sp"), "mp")),
    ("classifier\\.out_proj", ("mp", ("fsdp", "sp"))),
    # PKU score model
    ("score_head", (("fsdp", "sp"), "mp")),
    )

    
CLIP_RULES = (
    ("patch_embedding", (("fsdp", "sp"), "mp", None, None)),
    ("position_embedding", (("fsdp", "sp"), "mp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", (("fsdp", "sp"), "mp")),
    ("self_attn\\.out_proj", ("mp", ("fsdp", "sp"))),
    ("mlp\\.fc1", (("fsdp", "sp"), "mp")),
    ("mlp\\.fc2", ("mp", ("fsdp", "sp"))),
    ("visual_projection", (("fsdp", "sp"), "mp")),
    ("text_projection", (("fsdp", "sp"), "mp")),
    )

LLAVA_RULES = (
    ("multi_modal_projector\\.linear_1", (("fsdp", "sp"), "mp")),
    ("multi_modal_projector\\.linear_2", ("mp", ("fsdp", "sp"))),
    *LLAMA_RULES,
    *CLIP_RULES,
)


ALL_RULES = [
    (GPTNeoXConfig, GPTNEOX_RULES),
    (T5Config, T5_RULES),
    (LlamaConfig, LLAMA_RULES),
    (MistralConfig, MISTRAL_RULES),
    ("Qwen2Config", MISTRAL_RULES),
    (RobertaConfig, ROBERTA_RULES),
    (XLMRobertaConfig, ROBERTA_RULES),
    (CLIPConfig, CLIP_RULES),
    (CLIPVisionConfig, CLIP_RULES),
    (LlavaConfig, LLAVA_RULES),
    (GemmaConfig, LLAMA_RULES),
    (Gemma2Config, LLAMA_RULES),
    ("Phi3Config", PHI3_RULES),
    ("ExaoneConfig", LLAMA_RULES),
]

def find_rule(model):
    for config, rule in ALL_RULES:
        if isinstance(config, str) and model.config.__class__.__name__ == config:
            return rule
        elif model.config.__class__ == config:
            return rule
    raise Exception("unsupported model to partitioning")

strkey2id = {
    "dp": 0,
    "fsdp": 1,
    "sp": 2,
    "mp": 3,
}

def partition_module(model, mesh, device=xm.xla_device(), verbose=False):
    partition_specs = find_rule(model)
    # rule = [(k, tuple([strkey2id[x] for x in v])) for k, v in partition_specs]
        
    # print(rule)

    for name, module in model.named_modules():
        module.to(device)
        # print(name, module.__class__.__name__)
        if isinstance(module, (nn.Embedding, nn.Linear, nn.Conv1d, nn.Conv2d)):
            find = False
            for rule_pattern, spec in partition_specs:
                if re.findall(rule_pattern, name):
                    if verbose:
                        print("match", rule_pattern, name)
                    
                    xs.mark_sharding(module.weight, mesh, spec)
                    if hasattr(module, "bias") and module.bias is not None:
                        xs.mark_sharding(module.bias, mesh, spec[:1])
                        
                    find = True
                    break
                    
            
            if not find and verbose:
                print(f"{name} not found in partition_specs")

            
        
def partition_module_dp(model, mesh, device=xm.xla_device(), verbose=False):
    spec = (1, 2)

    for name, module in model.named_modules():
        module.to(device)
        if isinstance(module, (nn.Embedding, nn.Linear, nn.Conv1d, nn.Conv2d)):
            xs.mark_sharding(module.weight, mesh, spec)