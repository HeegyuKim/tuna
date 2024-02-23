from transformers import GPTNeoXConfig, T5Config, LlamaConfig, MistralConfig, LlavaConfig, GemmaConfig


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
    (GemmaConfig, LLAMA_TARGETS)
]

def find_lora_targets(model):
    for config, rule in LORA_TARGETS:
        if model.config.__class__ == config:
            return rule
    raise Exception("unsupported model to lora targeting")

def find_lora_targets_from_config(model_config):
    for config, rule in LORA_TARGETS:
        if model_config.__class__ == config:
            return rule
    raise Exception("unsupported model to lora targeting")
