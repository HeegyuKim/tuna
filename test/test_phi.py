import tuna.flax_model
from transformers import AutoModelForCausalLM, AutoTokenizer, FlaxAutoModelForCausalLM, Phi3Config
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


with jax.default_device(jax.devices("cpu")[0]):

    model_name = "microsoft/phi-1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    # config = Phi3Config(num_hidden_layers=2, sliding_window=2048)
    # model = AutoModelForCausalLM.from_config(config)
    flax_model = transformers.FlaxAutoModelForCausalLM.from_config(
        model.config,
        _do_init=True,
        input_shape=(1, 1024)
        )

    pt_state_dict = model.state_dict()

    print("converting to flax")
    params = transformers.modeling_flax_pytorch_utils.convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model)

    text_input = "If I am delivered mail not intended for me but misdelivered to me and it has cash in it, am I liable at all for keeping the money?"
    max_length = 128
    
    # tokenizer.padding_side = "left"
    inputs = tokenizer(text_input, return_tensors="pt", max_length=max_length, padding="max_length")
    inputs["labels"] = (inputs["input_ids"] * inputs["attention_mask"]) + (-100 * (1 - inputs["attention_mask"]))
    inputs_flax = tokenizer(text_input, return_tensors="np", max_length=max_length, padding="max_length")
    # inputs_flax["labels"] = inputs_flax["input_ids"]
    
    print(inputs["input_ids"])
    print(inputs["attention_mask"])
    print(inputs["labels"])

    print("running model")
    with torch.no_grad():
        output = model(**inputs)
        output_flax = flax_model(params=params, **inputs_flax)
        flax_loss = flax_nll_loss(output_flax.logits, inputs_flax["input_ids"], inputs_flax["attention_mask"])

        # compare mse loss
        print(np.mean((output.logits.numpy() - jnp.asarray(output_flax.logits)) ** 2))
        print(output.loss, flax_loss)