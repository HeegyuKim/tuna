import torch
import torch.nn as nn
import re
import torch_xla.experimental.xla_sharding as xs
import torch_xla.core.xla_model as xm
from transformers import (
    GPTNeoXConfig, T5Config, LlamaConfig, RobertaConfig, MistralConfig, LlavaConfig, CLIPConfig, CLIPVisionConfig, GemmaConfig,
    XLMRobertaConfig
)

GPTNEOX_RULES = (
    # embeddings
    ("gpt_neox\\.embed_in", ("mp", "fsdp")),
    # atention
    ("attention\\.query_key_value", ("fsdp", "mp")),
    ("attention\\.dense", ("mp", "fsdp")),
    # mlp
    ("mlp\\.dense_h_to_4h", ("fsdp", "mp")),
    ("mlp\\.dense_4h_to_h", ("mp", "fsdp")),
    # output
    ("embed_out", ("fsdp", "mp")),
    # GTA2
    ("heads.\d+", ("fsdp", "mp")),
    ("v_heads.\d+", ("fsdp", "mp")),
    # PKU score model
    ("score_head", ("fsdp", "mp")),
)

T5_RULES = (
    # embeddings
    ("shared", ("mp", "fsdp")),
    ("embed_tokens", ("mp", "fsdp")),
    
    # attention
    ("q", ("fsdp", "mp")),
    ("k", ("fsdp", "mp")),
    ("v", ("fsdp", "mp")),
    ("o", ("mp", "fsdp")),

    # mlp
    ("w", ("fsdp", "mp")),
    ("wi_0", ("fsdp", "mp")),
    ("wi_1", ("fsdp", "mp")),
    ("wo", ("mp", "fsdp")),

    # seq2seq lm head
    ("lm_head", ("fsdp", "mp")),
)

LLAMA_RULES = (
    ("model\\.embed_tokens", ("mp", "fsdp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.o_proj", ("mp", "fsdp")),
    ("mlp\\.gate_proj", ("fsdp", "mp")),
    ("mlp\\.down_proj", ("mp", "fsdp")),
    ("mlp\\.up_proj", ("fsdp", "mp")),
    ("lm_head", ("fsdp", "mp")),
    ("score", ("fsdp", "mp")),
    # GTA2
    ("heads.\d+", ("fsdp", "mp")),
    ("v_heads.\d+", ("fsdp", "mp")),
    # PKU score model
    ("score_head", ("fsdp", "mp")),
    )

MISTRAL_RULES = (
    ("model\\.embed_tokens", ("mp", "fsdp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.o_proj", ("mp", "fsdp")),
    ("mlp\\.gate_proj", ("fsdp", "mp")),
    ("mlp\\.down_proj", ("mp", "fsdp")),
    ("mlp\\.up_proj", ("fsdp", "mp")),
    ("lm_head", ("fsdp", "mp")),
    ("score", ("fsdp", "mp")),
    # GTA2
    ("heads.\d+", ("fsdp", "mp")),
    ("v_heads.\d+", ("fsdp", "mp")),
    # PKU score model
    ("score_head", ("fsdp", "mp")),
    )
    
ROBERTA_RULES = (
    ("embeddings", ("mp", "fsdp")),
    ("attention\\.self\\.(query|key|value)", ("fsdp", "mp")),
    ("attention\\.output\\.dense", ("mp", "fsdp")),
    ("intermediate\\.dense", ("fsdp", "mp")),
    ("output\\.dense", ("mp", "fsdp")),
    ("classifier\\.dense", ("fsdp", "mp")),
    ("classifier\\.out_proj", ("mp", "fsdp")),
    # PKU score model
    ("score_head", ("fsdp", "mp")),
    )

    
CLIP_RULES = (
    ("patch_embedding", ("fsdp", "mp", None, None)),
    ("position_embedding", ("fsdp", "mp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.out_proj", ("mp", "fsdp")),
    ("mlp\\.fc1", ("fsdp", "mp")),
    ("mlp\\.fc2", ("mp", "fsdp")),
    ("visual_projection", ("fsdp", "mp")),
    ("text_projection", ("fsdp", "mp")),
    )

LLAVA_RULES = (
    ("multi_modal_projector\\.linear_1", ("fsdp", "mp")),
    ("multi_modal_projector\\.linear_2", ("mp", "fsdp")),
    *LLAMA_RULES,
    *CLIP_RULES,
)


ALL_RULES = [
    (GPTNeoXConfig, GPTNEOX_RULES),
    (T5Config, T5_RULES),
    (LlamaConfig, LLAMA_RULES),
    (MistralConfig, MISTRAL_RULES),
    (RobertaConfig, ROBERTA_RULES),
    (XLMRobertaConfig, ROBERTA_RULES),
    (CLIPConfig, CLIP_RULES),
    (CLIPVisionConfig, CLIP_RULES),
    (LlavaConfig, LLAVA_RULES),
    (GemmaConfig, LLAMA_RULES)
]

def find_rule(model):
    for config, rule in ALL_RULES:
        if model.config.__class__ == config:
            return rule
    raise Exception("unsupported model to partitioning")

strkey2id = {
    "dp": 0,
    "fsdp": 1,
    "mp": 2
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
                    find = True
                    break
            
            if not find:
                print(f"{name} not found in partition_specs")

            
        
def partition_module_dp(model, mesh, device=xm.xla_device(), verbose=False):
    spec = (1, 2)

    for name, module in model.named_modules():
        module.to(device)
        if isinstance(module, (nn.Embedding, nn.Linear, nn.Conv1d, nn.Conv2d)):
            xs.mark_sharding(module.weight, mesh, spec)