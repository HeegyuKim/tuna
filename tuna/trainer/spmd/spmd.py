import os, tempfile
os.environ["PJRT_DEVICE"] = "TPU"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

xr.use_spmd()
# try:
#     from jax_smi import initialise_tracking
#     initialise_tracking()
# except:
#     print("failed to execute jax-smi")
    
    
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
from torch_xla.experimental.xla_sharding import Mesh
from .partition_rules import partition_module, partition_module_dp
from ...task.collator import GenerativeLanguageModelCollator
from transformers.modeling_outputs import BaseModelOutput

from functools import partial
from dataclasses import dataclass
from copy import deepcopy
from typing import Iterable, Optional
from tqdm import trange, tqdm
from ...task import Task, TensorWrapper
from ..base import BaseTrainer, BaseTrainingArguments, trainers
# from task.gta2.model_value_head import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead


def build_mesh_shape(mesh_str: str, device_count: int):
    if mesh_str == "fsdp":
        mesh = (1, device_count, 1, 1)
    elif mesh_str == "dp":
        mesh = (device_count, 1, 1, 1)
    elif mesh_str == "mp":
        mesh = (1, 1, device_count, 1,)
    else:
        raise ValueError(f"Unknown mesh string: {mesh_str}")

    return tuple(mesh)


@dataclass
class SPMDArguments(BaseTrainingArguments):
    mesh: str = "fsdp"
    

PARTITION_INPUTS = ["input_ids", "decoder_input_ids", "labels", "attention_mask", "decoder_attention_mask"]

class SPMDTensorWrapper(TensorWrapper):
    
    def __init__(self, mesh, device) -> None:
        super().__init__(device)
        self.mesh = mesh

    def shard_tensor(self, v):
        xs.mark_sharding(v, self.mesh, tuple(range(v.dim())))
    
    def shard_pixel_values(self, v):
        xs.mark_sharding(v, self.mesh, (0, 1, None, None))

    def __call__(self, tensor):
        if torch.is_tensor(tensor):
            v = tensor.to(self.device)
            self.shard_tensor(v)
            return v
        else:
            for k, v in tensor.items():
                if torch.is_tensor(v):
                    tensor[k] = v.to(self.device)
                    if k == "pixel_values":
                        self.shard_pixel_values(tensor[k])
                        # pass
                    else:
                        self.shard_tensor(tensor[k])
            return tensor

@trainers.register("spmd")
class SPMDTrainer(BaseTrainer):
    ARG_CLASS = SPMDArguments

    def setup(self):
        self.device = xm.xla_device()
        self.partition_model()

        super().setup()

        self.task.wrapper = SPMDTensorWrapper(self.mesh, xm.xla_device())

    def partition_model(self):
        self.model_cpu_copy = deepcopy(self.task.model)

        device_count = xr.global_runtime_device_count()
        mesh_shape = build_mesh_shape(self.args.mesh, device_count)
        device_ids = np.array(range(device_count))
        print(f"XLA Device count {device_count}, shape: {mesh_shape}")
        self.mesh = Mesh(device_ids, mesh_shape, ('dp', 'fsdp', 'mp', 'sp'))

        has_ref = hasattr(self.task, "ref_model")

        if mesh_shape[1] == 1 and mesh_shape[2] == 1:
            partition_module_dp(self.task.model, self.mesh, self.device)
            if has_ref and self.task.ref_model:
                partition_module_dp(self.task.ref_model, self.mesh, self.device)
        else:
            partition_module(self.task.model, self.mesh, self.device)
            if has_ref and self.task.ref_model:
                partition_module(self.task.ref_model, self.mesh, self.device)

    def backward_loss(self, loss):
        loss.backward()
        xm.mark_step()

    def evaluate(self, epoch, global_step):
        super().evaluate(epoch, global_step)
        xm.mark_step()
    
    def _make_cpu_copy(self, model):
        # if isinstance(model, (AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead)):
        #     xla_states = model.state_dict(keep_vars=True, for_xla_copy=True)
        # else:
        xla_states = model.state_dict(keep_vars=True)

        cpu_states = {}
        for k, v in xla_states.items():
            cpu_states[k] = v.to("cpu")
        self.model_cpu_copy.load_state_dict(cpu_states)
        return self.model_cpu_copy

    def save_model(self, name, is_last=False):
        print("save", name)
        run_name = self.args.run_name.replace("/", "__")
        model = self._make_cpu_copy(self.task.model)

        if self.args.save_format == "fp16":
            model.half()
        elif self.args.save_format == "bf16":
            model.bfloat16()

        repo_id = run_name.replace("/", "__").replace(",", "_")

        if self.args.output_dir:
            path = f"{self.args.output_dir}/{run_name}/{name}"
            self.task.save_artifacts(model, path)

            if self.args.push_to_hub:
                self.push_to_hub_revision(repo_id, path, "main" if is_last else name)

        elif self.args.push_to_hub:
            with tempfile.TemporaryDirectory() as path:
                self.task.save_artifacts(model, path)
                self.push_to_hub_revision(repo_id, path, "main" if is_last else name)


# see https://github.com/huggingface/transformers/issues/18661
class SPMDInference():
    def __init__(self, 
                 model, 
                 tokenizer, 
                 mesh="fsdp",
                 max_seq_length: int = 1024, 
                 max_decoder_seq_length: int = 256,
                 device: str = "auto"
                 ):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length

        self.device = xm.xla_device() if device == "auto" else device

        if mesh:
            device_count = xr.global_runtime_device_count()
            mesh_shape = build_mesh_shape(mesh, device_count)
            device_ids = np.array(range(device_count))
            self.mesh = Mesh(device_ids, mesh_shape, ('dp', 'fsdp', 'mp'))

            if mesh_shape[1] == 1 and mesh_shape[2] == 1:
                partition_module_dp(self.model, self.mesh, self.device)
            else:
                partition_module(self.model, self.mesh, self.device)
            
            if not model.config.is_encoder_decoder:
                # import inspect
                self.has_position_ids = True # "position_ids" in inspect.getargspec(model.forward).args
            else:
                self.has_position_ids = False

            self.wrapper = SPMDTensorWrapper(self.mesh, xm.xla_device())

            # original_forward = self.model.forward
            # def spmd_forward(self, **kwargs):
            #     for k, v in kwargs.items():
            #         if "input_ids" in k or "attention_mask" in k or "position_ids" in k:
            #             print("shard", k)
            #             wrapper(v)

            #     return original_forward(**kwargs)

            # import types
            # self.model.forward = types.MethodType(spmd_forward, self.model)

        else:
            self.model = self.model.to(self.device)
            self.wrapper = TensorWrapper(self.device)
            self.has_position_ids = False
            self.mesh = None
        
        self.collator = GenerativeLanguageModelCollator(
            tokenizer,
            max_length=max_seq_length,
            decoder_max_length=max_decoder_seq_length,
            return_tensors="pt"
        )

        self._generate_one_token_fn = torch.compile(
            self._generate_one_token_fn,
            backend="openxla", # openxla
            # fullgraph=True
        )


        
    def _prepare_batch_clm(self, inputs, outputs):
        batch = []
        encode = partial(
            self.tokenizer.encode,
            truncation=True, 
            max_length=self.max_seq_length
            )
        
        for prefix, out in zip(inputs, outputs):
            prefix, out = encode(prefix), encode(out)
            input_ids = prefix + out
            labels = [-100] * len(prefix) + out

            batch.append({
                "input_ids": input_ids,
                "attention_mask": [1 for _ in range(len(input_ids))],
                "labels": labels
            })

        batch = self.wrapper(self.collator(batch))
        return batch

    def _prepare_batch_seq2seq(self, inputs, outputs):
        batch = []
        encode = partial(
            self.tokenizer.encode,
            truncation=True, 
            max_length=self.max_seq_length
            )
        
        for prefix, out in zip(inputs, outputs):
            prefix, out = encode(prefix), encode(out)
            input_ids = prefix
            labels = out

            batch.append({
                "input_ids": input_ids,
                "attention_mask": [1 for _ in range(len(input_ids))],
                "decoder_input_ids": labels[:-1],
                "decoder_attention_mask": [1 for _ in range(len(input_ids) - 1)],
                "labels": labels[1:]
            })

        batch = self.wrapper(self.collator(batch))
        return batch

    @torch.no_grad()
    def loglikelihood(self, inputs, outputs):
        batch = []

        is_encoder_decoder = self.model.config.is_encoder_decoder

        if is_encoder_decoder:
            batch = self._prepare_batch_seq2seq(inputs, outputs)
        else:
            batch = self._prepare_batch_clm(inputs, outputs)

        xm.mark_step()
        model_output = self.model(**batch)

        if is_encoder_decoder:
            input_ids = batch["decoder_input_ids"]
            labels = batch["labels"]
            logits = model_output.logits
        else:
            input_ids = batch["input_ids"][:, :-1]
            labels = batch["labels"][:, 1:]
            logits = model_output.logits[:, :-1]

        bs, seq_len, vocab = logits.shape
        logits = logits.log_softmax(-1)

        label_mask = (labels != -100).float()
        # print("labels", labels)
        # print("label_mask", label_mask)

        labels = labels.masked_fill(labels < 0, 0)
        target_logits = torch.gather(
            logits, -1, labels.unsqueeze(-1)
        ).squeeze(-1)

        # print(entropy.dtype, entropy.shape, label_mask.dtype, label_mask.shape)
        # print("bf target_logits", target_logits)
        target_logits = (target_logits * label_mask)
        # print("af target_logits", target_logits)
        target_logits = target_logits.sum(-1)

        greedy_labels = logits.argmax(-1)
        label_sum = label_mask.sum(-1)
        is_greedy = (greedy_labels == labels).sum(-1) == label_sum

        return target_logits.cpu().numpy(), is_greedy.cpu().numpy()

            

    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        xm.mark_step()
        return self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            **kwargs, 
            use_cache=False,
            wrapper=self.wrapper
            )

    @torch.no_grad()
    def generate_causal_lm(self, 
                           input_ids, 
                           attention_mask, 
                           max_length: int, 
                           max_new_tokens: Optional[int] = None,
                           do_sample=True, 
                           early_stopping=True,
                           top_k=0,
                           top_p=1.0,
                           temperature: float = 1.0,
                           verbose=False
                           ):
        batch_size = input_ids.shape[0]
        prompt_length = input_ids.shape[1]
        generation_length = max_length - prompt_length
        assert generation_length > 0

        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        if attention_mask.device != self.device:
            attention_mask = attention_mask.to(self.device)

        inputs = {
            "input_ids": torch.cat([input_ids, torch.zeros((batch_size, generation_length), dtype=input_ids.dtype, device=self.device)], -1),
            "attention_mask": torch.cat([attention_mask, torch.ones((batch_size, generation_length), dtype=attention_mask.dtype, device=self.device)], -1),
        }
        eos_mask = torch.zeros((batch_size, 1), dtype=torch.int64, device=self.device).bool()
        eos_token_id = torch.tensor(self.tokenizer.eos_token_id, dtype=torch.int64, device=self.device)
        filter_value = torch.tensor(-99999, dtype=torch.float32, device=self.device)

        # if self.has_position_ids:
        inputs["position_ids"] = torch.cumsum(inputs["attention_mask"], -1) - 1
        
        # update_ids = torch.arange(0, max_length, device=self.device, dtype=torch.int32).repeat(batch_size, 1)

        select_index = torch.tensor(prompt_length - 1, device=self.device, dtype=torch.int64)
        input_ids = inputs["input_ids"]

        if self.wrapper:
            inputs = self.wrapper(inputs)

        if max_new_tokens:
            gen_length = min(prompt_length + max_new_tokens, max_length) - 1
        else:
            gen_length = max_length - 1

        for seq_length in trange(prompt_length, gen_length, disable=not verbose):
            
            next_tokens = self._generate_one_token_fn(
                inputs,
                select_index,
                eos_token_id.int(),
                do_sample=do_sample,
                early_stopping=early_stopping,
                top_k=top_k,
                top_p=top_p,
                top_k_filter_value=filter_value
            )

            if early_stopping:
                next_tokens = (~eos_mask) * next_tokens + eos_mask * eos_token_id
                eos_mask = eos_mask | (next_tokens == self.tokenizer.eos_token_id)

                if eos_mask.sum() == eos_mask.shape[0]:
                    break
            
            select_index += 1
            inputs["input_ids"].index_copy_(1, select_index, next_tokens)

            xm.mark_step()

        return inputs["input_ids"].cpu()[:, :seq_length]

    def _generate_one_token_fn(self, 
                    inputs, 
                    select_index,
                    eos_token_id,
                    do_sample=True, 
                    early_stopping=True,
                    top_k=0,
                    top_p=0,
                    top_k_filter_value=0,
                    ):

        outputs = self.model(**inputs)
        next_token_logits = outputs.logits.index_select(1, select_index).squeeze(1)
        
        if not early_stopping:
            next_token_logits.index_fill_(1, eos_token_id, -99999)

        if do_sample:
            next_tokens = top_k_top_p_filtering(next_token_logits, top_k, top_p, filter_value=top_k_filter_value)
        else: # greedy
            next_tokens = next_token_logits.argmax(-1)
    
        return next_tokens
    


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = (probs_sum - probs_sort) > p
    probs_sort = torch.where(mask, 0.0, probs_sort)
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k and top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = (logits < torch.topk(logits, top_k)[0][..., -1, None]).int()
        # print(indices_to_remove.shape, (1 - indices_to_remove).sum(), filter_value)
        logits = logits + (indices_to_remove * filter_value)

    if top_p < 1.0:
        next_tokens = sample_top_p(logits.softmax(-1), top_p)
    else:
        next_tokens = torch.multinomial(logits.softmax(-1), 1)
    return next_tokens