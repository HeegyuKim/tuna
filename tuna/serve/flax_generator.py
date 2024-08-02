import logging
import warnings
from typing import Any, Dict, Optional, List, Union, Tuple

import functools
import copy
from tqdm.auto import tqdm
import requests
import transformers
from transformers import GenerationConfig
from .. import flax_model

import jax, flax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec
from fjformer import make_shard_and_gather_fns, match_partition_rules, with_sharding_constraint

def get_dtype(dtype):
    if isinstance(dtype, str):
        dtype = {
            "bf16": jnp.bfloat16,
            "bfloat16": jnp.bfloat16,
            "fp16": jnp.float16,
            "float16": jnp.float16,
            "fp32": jnp.float32,
            "float32": jnp.float32,
            "fp64": jnp.float64,
            "float64": jnp.float64,
        }[dtype]
    return dtype

import torch
from .prompt_templates import PROMPT_TEMPLATES
from ..trainer.flax.partition_rules import get_partition_rules
from copy import deepcopy


def load_huggingface_model_tokenizer(model_name: str, dtype: torch.dtype = torch.bfloat16, device = "auto", trust_remote_code=False, merge_peft=False):
    print("load model", model_name)
    if "@" not in model_name:
        repo_id, revision = model_name, None
    else:
        repo_id, revision = model_name.split("@")

    if repo_id.startswith("peft:"):
        import peft

        repo_id = repo_id[len("peft:"):]

        config = peft.PeftConfig.from_pretrained(repo_id, revision=revision, trust_remote_code=trust_remote_code)
        base_model_name_or_path, base_revision = config.base_model_name_or_path, config.revision

        model = transformers.AutoModelForCausalLM.from_pretrained(base_model_name_or_path, revision=base_revision, torch_dtype=dtype, device_map=device, trust_remote_code=trust_remote_code)  
        tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name_or_path, revision=base_revision, trust_remote_code=trust_remote_code)

        model = peft.PeftModel.from_pretrained(model, repo_id, revision=revision)
        if merge_peft:
            model = model.merge_and_unload()
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, torch_dtype=dtype, device_map=device, trust_remote_code=trust_remote_code)
        tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id, revision=revision, trust_remote_code=trust_remote_code)
    
    return model, tokenizer

MESH_SHAPES = {
    "fsdp": (1, -1, 1, 1),
    "tp": (1, 1, -1, 1),
    "sp": (1, 1, 1, -1),
}

class RNG:
    def __init__(self, seed):
        self.rng = jax.random.PRNGKey(seed)

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}


USUAL_EOS_TOKENS = [
    '<|im_start|>', '<|im_end|>', '<|endoftext|>', 
    '<start_of_turn>', '<end_of_turn>', 
    '<|user|>', '<|assistant|>', '</s>'
    ]

class FlaxHuggingfaceModel:

    def __init__(
        self,
        model_name_or_path: str, 
        prompt_length: int,
        max_length: int,
        batch_size: int = 1, 
        fully_sharded_data_parallel = True,
        chat_template: Optional[str] = None,
        eos_token_id: Optional[int] = None,
        eos_token: Optional[str] = None,
        gen_args: Optional[Dict[str, Any]] = None,
        mesh_axes_names = ("dp", "fsdp", "tp", "sp"),
        mesh_axes_shape: Union[Tuple, str] = (1, -1, 1, 1),
        dtype: str = "bf16",
    ):
        gen_args = gen_args or {}
        if "max_new_tokens" not in gen_args:
            gen_args["max_new_tokens"] = max_length - prompt_length
            
        pt_model, tokenizer = load_huggingface_model_tokenizer(model_name_or_path, device="cpu", merge_peft=True)

        if chat_template:
            tokenizer.chat_template = PROMPT_TEMPLATES[chat_template]
        if tokenizer.chat_template is None:
            tokenizer.chat_template = PROMPT_TEMPLATES.get(model_name_or_path)

        if eos_token_id is not None:
            tokenizer.eos_token_id = eos_token_id
            print("Setting eos token to", tokenizer.eos_token)
        elif eos_token is not None:
            tokenizer.eos_token = eos_token
            print("Setting eos token to", eos_token)

        self.eos_token_ids = [tokenizer.eos_token_id]

        for token in USUAL_EOS_TOKENS:
            if token in tokenizer.special_tokens_map.values():
                self.eos_token_ids.append(tokenizer.convert_tokens_to_ids(token))
            if token in tokenizer.additional_special_tokens:
                self.eos_token_ids.append(tokenizer.convert_tokens_to_ids(token))
        
        self.eos_token_ids = list(set(self.eos_token_ids))
        print("EOS Tokens:", self.eos_token_ids)


        with jax.default_device(jax.devices("cpu")[0]):
            print("create flax model")
            config = pt_model.config
            config.freq_max_position_embeddings = max_length
            if isinstance(config, transformers.MistralConfig):
                config.sliding_window=4096

            flax_model = transformers.FlaxAutoModelForCausalLM.from_config(
                config,
                _do_init=True,
                dtype=get_dtype(dtype),
                input_shape=(1, max_length)
                )

            pt_state_dict = pt_model.state_dict()

            print("converting to flax")
            params = transformers.modeling_flax_pytorch_utils.convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model)

            del pt_state_dict
            del pt_model
            import gc
            gc.collect()

            self.model = flax_model

        if isinstance(mesh_axes_shape, str):
            mesh_axes_shape = MESH_SHAPES[mesh_axes_shape]

        array = jnp.ones((len(jax.devices()), 1)).reshape(mesh_axes_shape)
        self.mesh = Mesh(mesh_utils.create_device_mesh(array.shape), mesh_axes_names)
        with self.mesh:
            logging.info(
                "matching partition rules"
            )
            partition_specs = match_partition_rules(params=params, rules=get_partition_rules(flax_model.config, fully_sharded_data_parallel=fully_sharded_data_parallel))
            shard_fns, _ = make_shard_and_gather_fns(partition_specs, self.mesh)
            logging.info(
                "sharding parameters across all of the chosen backend(tpu/gpu/cpu)s"
            )
            params = flax.traverse_util.flatten_dict(params)
            shard_fns = flax.traverse_util.flatten_dict(shard_fns)
            pbar = tqdm(params.keys(), desc="Sharding Params")
            for key in pbar:
                key = tuple(key)
                params[key] = shard_fns[key](params[key])
            self.params = flax.traverse_util.unflatten_dict(params)
        self.partition_specs = partition_specs

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("Setting pad token to eos token")
            
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        self.prefix_tokenizer = copy.deepcopy(tokenizer)
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        self.tokenizer = copy.deepcopy(tokenizer)
        self.max_sequence_length = max_length
        self.prompt_length = prompt_length
        self.rng_generator = RNG(42)

        generation_ps = PartitionSpec("dp", "fsdp")

        @functools.partial(
            pjit,
            in_shardings=(self.partition_specs, PartitionSpec(), PartitionSpec()),
            out_shardings=(PartitionSpec())
        )
        def greedy_generate(parameters, input_ids, attention_mask):
            input_ids = with_sharding_constraint(input_ids, generation_ps)
            attention_mask = with_sharding_constraint(attention_mask, generation_ps)
            predict = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                params=parameters,
                generation_config=GenerationConfig(
                    max_length=self.max_sequence_length,
                    max_new_tokens=gen_args.get("max_new_tokens"),
                    min_length=None,

                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,

                    temperature=gen_args.get("temperature", 1.0),
                    early_stopping=True,
                    do_sample=False,
                    num_beams=1,
                    top_p=gen_args.get("top_p", 1.0),
                    top_k=gen_args.get("top_k"),
                    repetition_penalty=gen_args.get("repetition_penalty", 1.0)
                )
            ).sequences[:, input_ids.shape[1]:]
            return predict

        @functools.partial(
            pjit,
            in_shardings=(self.partition_specs, PartitionSpec(), PartitionSpec(), PartitionSpec()),
            out_shardings=(PartitionSpec())
        )
        def generate(parameters, rng, input_ids, attention_mask):
            input_ids = with_sharding_constraint(input_ids, generation_ps)
            attention_mask = with_sharding_constraint(attention_mask, generation_ps)
            predict = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                params=parameters,
                prng_key=rng,
                generation_config=GenerationConfig(
                    max_length=self.max_sequence_length,
                    max_new_tokens=gen_args.get("max_new_tokens"),

                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,

                    temperature=gen_args.get("temperature", 1.0),
                    early_stopping=True,
                    do_sample=True,
                    top_p=gen_args.get("top_p", 1.0),
                    top_k=gen_args.get("top_k"),
                    repetition_penalty=gen_args.get("repetition_penalty", 1.0)
                )
            ).sequences[:, input_ids.shape[1]:]
            return predict

        self.generate_function = generate
        self.greedy_generate_function = greedy_generate
        self._funcs_generated = True
        self.system_prompt = None
        self.batch_size = batch_size

    def set_system_message(self, prompt):
        self.system_prompt = prompt
    
    def generate_from_prompts(self,
               prompts: list[str],
               gen_args: dict,
               ):
        fixed_pad = self.prompt_length
        tokens = self.prefix_tokenizer(
            prompts,
            max_length=fixed_pad,
            padding="max_length",
            truncation=True,
            return_tensors="jax",
            add_special_tokens=False
        )

        greedy = not gen_args.get("do_sample", False)

        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask

        
        with self.mesh:
            outputs = self.greedy_generate_function(
                self.params,
                input_ids,
                attention_mask,
            ) if greedy else self.generate_function(
                self.params,
                self.rng_generator(),
                input_ids,
                attention_mask,
            )

        # output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print(self.tokenizer.decode(input_ids[0], skip_special_tokens=False).replace(self.tokenizer.pad_token, ""))
        # print(output)
        return outputs

    def generate_batch(self, prompts, histories = None, generation_prefix: str = None, gen_args = {}):
        assert len(prompts) <= self.batch_size

        if histories is None:
            histories = [[] for _ in prompts]
        else:
            histories = deepcopy(histories)

        final_prompts = []
        for prompt, history in zip(prompts, histories):
            history.append({
                'role': 'user',
                'content': prompt,
            })

            inputs = self.tokenizer.apply_chat_template(history, add_special_tokens=True, tokenize=False, add_generation_prompt=True)
            if generation_prefix is not None:
                inputs = inputs + generation_prefix
            
            final_prompts.append(inputs)

        if len(final_prompts) < self.batch_size:
            final_prompts += [final_prompts[0]] * (self.batch_size - len(final_prompts))
        
        outputs = self.generate_from_prompts(final_prompts, gen_args)

        if gen_args.get("verbose", False):
            special_token_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[:len(prompts)]
            for inputs, output in zip(final_prompts, special_token_outputs):
                print("Prompt:", inputs)
                print("Response:", output.replace(self.tokenizer.pad_token, ""))

        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[:len(prompts)]
        cleaned_outputs = []

        for output in outputs:
            for eos_token in USUAL_EOS_TOKENS:
                if eos_token in output:
                    output = output[:output.index(eos_token)]
            cleaned_outputs.append(output)
            
        return cleaned_outputs
    
    def generate(self, prompt, history = None, generation_prefix: str = None, gen_args = {}):
        if history is None:
            history = []
        else:
            history = deepcopy(history)
            
        history.append({
            'role': 'user',
            'content': prompt,
        })

        inputs = self.tokenizer.apply_chat_template(history, add_special_tokens=True, tokenize=False, add_generation_prompt=True)
        if generation_prefix is not None:
            inputs = inputs + generation_prefix
        
        outputs = self.generate_from_prompts([inputs], gen_args)
        output = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
            )


        for eos_token in USUAL_EOS_TOKENS:
            if eos_token in output:
                output = output[:output.index(eos_token)]
        
        if gen_args.get("verbose", False):
            print("Prompt:", inputs)
            print("Response:", self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=False
            ).replace(self.tokenizer.pad_token, ""))

        return output

    def compile(self, test_prompt="Hi, who are you?", targets=["greedy", "sample"], gen_args: dict = {}):
        gen_args = deepcopy(gen_args)

        print("Compiling functions")
        if "greedy" in targets:
            gen_args["do_sample"] = False
            print("Greedy:", self.generate_batch([test_prompt] * self.batch_size, gen_args=gen_args))
        
        if "sample" in targets:
            gen_args["do_sample"] = True
            print("Sample:", self.generate_batch([test_prompt] * self.batch_size, gen_args=gen_args))


class FlaxAPI:

    def __init__(self, server: str):
        if server[-1] != '/':
            server += "/"
            
        self.server = server
        self.system_prompt = None

    def set_system_message(self, prompt):
        self.system_prompt = prompt
    
    def chat(self, conversations, greedy = False, response_prefix: str = ""):
        """
            prompt: str,
            system: Optional[str] = "",
            history: Union[List[str], None] = [],
            temperature: Optional[float] = 1.0,
            greedy: Optional[bool] = False,
        """
        history = []
        for i in range(0, len(conversations) - 1, 2):
            user = conversations[i]['content']
            assistant = conversations[i+1]['content']
            history.append([user, assistant])
            
        if conversations[0]['role'] != 'system' and self.system_prompt is not None:
            history.insert(0, {'role': 'user', 'content': self.system_prompt})

        body = {
            "conversations": conversations,
            "response_prefix": response_prefix,
            "greedy": greedy,
        }
        resp = requests.post(f"{self.server}chat", json=body)
        if resp.ok:
            output = resp.json()
            return output
        else:
            print(resp.text)
            raise ValueError(f"Failed to get response from server: {self.server}")

    def generate(self, messages, clear_old_history=True, **kwargs):
        """
        Generates a response based on messages that include conversation history.
        :param list[str]|str messages: A list of messages or a single message string.
                                       User and assistant messages should alternate.
        :param bool clear_old_history: If True, clears the old conversation history before adding new messages.
        :return str: The response generated by the OpenAI model based on the conversation history.
        """
        if clear_old_history:
            self.conversation = []

        if isinstance(messages, str):
            messages = [messages]

        for index, message in enumerate(messages):
            self.conversation.append({
                'role': 'user' if index % 2 == 0 else 'assistant', 
                'content': message
                })
            
        response = self.chat(self.conversation)
        return response


    def batch_generate(self, conversations, **kwargs):
        """
        Generates responses for multiple conversations in a batch.
        :param list[list[str]]|list[str] conversations: A list of conversations, each as a list of messages.
        :return list[str]: A list of responses for each conversation.
        """
        responses = []
        for conversation in conversations:
            if isinstance(conversation, str):
                warnings.warn('For batch generation based on several conversations, provide a list[str] for each conversation. '
                              'Using list[list[str]] will avoid this warning.')
            responses.append(self.generate(conversation, **kwargs))
        return responses