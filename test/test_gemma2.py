import tuna.flax_model
from tuna.flax_model.gemma2 import FlaxGemma2DecoderLayer, FlaxGemma2Attention, FlaxGemma2MLP, FlaxGemma2RMSNorm
from transformers import AutoModelForCausalLM, AutoTokenizer, FlaxAutoModelForCausalLM, AutoConfig
import jax
import jax.numpy as jnp
import numpy as np
import torch
import transformers

def flax_nll_loss(logits, labels, mask):
    # compute the negative log-likelihood
    logits = logits[:, :-1]
    mask = mask[:, 1:]
    labels = labels[:, 1:]

    log_probs = jax.nn.log_softmax(logits)
    log_probs = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)
    log_probs = jnp.squeeze(log_probs, axis=-1)
    loss = -jnp.sum(log_probs * mask) / (jnp.sum(mask) + 1e-8)
    return loss


with jax.default_device(jax.devices("cpu")[0]), torch.no_grad():

    model_name = "google/gemma-2-9b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # config = AutoConfig.from_pretrained(model_name)
    # config.num_hidden_layers = 1
    # model = AutoModelForCausalLM.from_config(config)
    
    flax_model = transformers.FlaxAutoModelForCausalLM.from_config(
        model.config,
        _do_init=True,
        input_shape=(1, 1024)
        )
    # print(flax_model.params)

    pt_state_dict = model.state_dict()

    print("converting to flax")
    params = transformers.modeling_flax_pytorch_utils.convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model)

    text_input = "<|im_start|>user\nIf I am delivered mail not intended for me but misdelivered to me and it has cash in it, am I liable at all for keeping the money? <|im_end|>\n<|im_start|>assistant\nGenerally no, unless the mail contains something illegal or a court order which mandates that you must return it."
    max_length = 128
    tokenizer.padding_side = "right"
    inputs = tokenizer(text_input, return_tensors="pt", max_length=max_length, padding="max_length")
    inputs["labels"] = (inputs["input_ids"] * inputs["attention_mask"]) + (-100 * (1 - inputs["attention_mask"]))
    inputs_flax = tokenizer(text_input, return_tensors="np", max_length=max_length, padding="max_length")
    # inputs_flax["labels"] = inputs_flax["input_ids"]
    
    print(inputs["input_ids"])
    print(inputs["attention_mask"])
    print(inputs["labels"])

    # print("compare one-by-one")
    # print("DecoderLayer")
    # hidden_states = np.random.random((1, 128, config.hidden_size)).astype(jnp.float32)
    # position_ids = np.arange(128, dtype=jnp.int32).reshape((1, 128))
    # model.model.layers[0].is_sliding = False
    # pt_layer_out = model.model.layers[0](
    #     torch.tensor(hidden_states), 
    #     position_ids=torch.tensor(position_ids)
    #     )

    # layer = FlaxGemma2DecoderLayer(model.config)
    # # flax_layer_out = flax_model.module.model.blocks.layers[0](
    # flax_layer_out = layer(
    #     jnp.array(hidden_states), 
    #     inputs_flax["attention_mask"], 
    #     position_ids=jnp.array(position_ids),
    #     params={"params": params["model"]["layers"]["0"]}
    #     )
    # print(np.mean((pt_layer_out.numpy() - jnp.asarray(flax_layer_out.logits)) ** 2))



    print("running model")
    output = model(**inputs)
    output_flax = flax_model(params=params, **inputs_flax)
    flax_loss = flax_nll_loss(output_flax.logits, inputs_flax["input_ids"], inputs_flax["attention_mask"])

    # compare mse loss
    print(np.mean((output.logits.softmax(-1).numpy() - jnp.asarray(jax.nn.softmax(output_flax.logits, -1))) ** 2))
    # print("last_hidden", np.mean((pt_hidden.numpy() - jnp.asarray(fx_hidden)) ** 2))
    
    print(output.loss, flax_loss)