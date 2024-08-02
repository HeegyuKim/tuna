import tuna.flax_model
from tuna.flax_model.gemma2 import FlaxGemma2DecoderLayer, FlaxGemma2Attention, FlaxGemma2MLP, FlaxGemma2RMSNorm
from transformers import AutoModelForCausalLM, AutoTokenizer, FlaxAutoModelForCausalLM, AutoConfig
import jax
import jax.numpy as jnp
import numpy as np
import torch
import transformers

# def flax_cross_entropy_loss(logits, labels):
#     logits = logits[:, :-1, :]
#     labels = labels[:, 1:]
    
#     valid = (labels != -100).astype(jnp.float32)
#     num_valid = jnp.sum(valid)
    
#     log_probs = jax.nn.log_softmax(logits, axis=-1)
#     label_log_probs = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)
#     label_log_probs = jnp.squeeze(label_log_probs, axis=-1)
    
#     loss = -jnp.sum(label_log_probs * valid) / num_valid
#     return loss

def flax_cross_entropy_loss(logits, labels):
    # compute the negative log-likelihood
    mask = (labels != -100).astype(jnp.float32)
    logits = logits[:, :-1]
    mask = mask[:, 1:]
    labels = labels[:, 1:]

    log_probs = jax.nn.log_softmax(logits)
    log_probs = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)
    log_probs = jnp.squeeze(log_probs, axis=-1)
    loss = -jnp.sum(log_probs * mask) / (jnp.sum(mask) + 1e-8)
    return loss

def convert_to_bf16(params):
    def to_bf16(x):
        if isinstance(x, jnp.ndarray) and x.dtype == jnp.float32:
            return x.astype(jnp.bfloat16)
        return x
    
    return jax.tree_map(to_bf16, params)

with jax.default_device(jax.devices("cpu")[0]), torch.no_grad():

    model_name = "mistralai/Mistral-Nemo-Base-2407"
    # model_name = "Locutusque/TinyMistral-248M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.config.freq_max_position_embeddings = 1024
    # config = AutoConfig.from_pretrained(model_name)
    # config.num_hidden_layers = 1
    # model = AutoModelForCausalLM.from_config(config)
    
    flax_model = transformers.FlaxAutoModelForCausalLM.from_config(
        model.config,
        _do_init=True,
        dtype=jnp.bfloat16,
        input_shape=(1, 1024)
        )
    # print(flax_model.params)

    pt_state_dict = model.state_dict()

    print("converting to flax")
    params = transformers.modeling_flax_pytorch_utils.convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model)

    # 모델의 파라미터를 bf16으로 변환
    print("converting to bf16")
    params = convert_to_bf16(params)

    text_input = "<|im_start|>user\nIf I am delivered mail not intended for me but misdelivered to me and it has cash in it, am I liable at all for keeping the money? <|im_end|>\n<|im_start|>assistant\nGenerally no, unless the mail contains something illegal or a court order which mandates that you must return it."
    max_length = 64
    tokenizer.padding_side = "right"
    inputs = tokenizer(text_input, return_tensors="pt", max_length=max_length, padding="max_length", return_token_type_ids=False)
    inputs["labels"] = (inputs["input_ids"] * inputs["attention_mask"]) + (-100 * (1 - inputs["attention_mask"]))
    inputs_flax = tokenizer(text_input, return_tensors="np", max_length=max_length, padding="max_length", return_token_type_ids=False)
    flax_labels = (inputs_flax["input_ids"] * inputs_flax["attention_mask"]) + (-100 * (1 - inputs_flax["attention_mask"]))
    
    print(inputs["input_ids"])
    print(inputs["attention_mask"])
    print(inputs["labels"])


    print("running model")
    output = model(**inputs)
    output_flax = flax_model(params=params, **inputs_flax)
    flax_loss = flax_cross_entropy_loss(output_flax.logits, flax_labels)
    
    print("PyTorch logits:", output.logits)
    print("Flax logits:", output_flax.logits)

    print("PyTorch softmax:", output.logits.softmax(-1))
    print("Flax softmax:", jax.nn.softmax(output_flax.logits, -1))

    # compare mse loss
    print(np.mean((output.logits.softmax(-1).numpy() - jnp.asarray(jax.nn.softmax(output_flax.logits, -1))) ** 2))
    # print("last_hidden", np.mean((pt_hidden.numpy() - jnp.asarray(fx_hidden)) ** 2))
    
    print(output.loss, flax_loss)