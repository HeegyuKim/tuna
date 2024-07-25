from dataclasses import dataclass
from .flax_base import FlaxLMTask, flax_tasks, FlaxLMTaskArguments, global_norm
from ..chat.train_templates import find_template
from ..dpo.collator import DPOCollator
from typing import Optional, Union

import jax, flax
import jax.numpy as jnp

from fjformer import with_sharding_constraint
from fjformer.xrapture import use_implicit_args
from fjformer.functions.loss_func import (
    cross_entropy_loss_and_accuracy,
)
import chex
import transformers as tf


def get_batch_logps(
    logits: chex.Array,
    labels: chex.Array,
    loss_mask: chex.Array,
    average_log_prob: bool = True
) -> chex.Array:
    per_token_logps = jnp.take_along_axis(
        jax.nn.log_softmax(logits), 
        labels[..., None], 
        axis=-1
    ).squeeze(-1)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
    
def get_model_batch_logps(model, params, chosen_input, rejected_input, chosen_labels, rejected_labels):

    chosen_loss_mask = chosen_labels >= 0 
    rejected_loss_mask = rejected_labels >= 0
    
    chosen_labels = jnp.where(chosen_loss_mask, chosen_labels, 0)
    rejected_labels = jnp.where(rejected_loss_mask, rejected_labels, 0)

    chosen_output = model(params=params, **chosen_input)
    chosen = chosen_output.logits[:, :-1]

    rejected_output = model(params=params, **rejected_input)
    rejected = rejected_output.logits[:, :-1]
    
    chosen_logprobs = get_batch_logps(chosen, chosen_labels, chosen_loss_mask)
    rejected_logprobs = get_batch_logps(rejected, rejected_labels, rejected_loss_mask)

    return chosen, chosen_logprobs, rejected_logprobs

def masked_mean(arr, mask):
    return (arr * mask).sum(-1) / mask.sum(-1)

def cross_entropy_loss_and_accuracy(logits, labels):
    valid = (labels >= 0).astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    logits = logits.astype(jnp.float32) # for numerical stability
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(labels, -1),
            axis=-1,
        ),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
    loss = -jnp.mean(jnp.sum(token_log_prob, axis=-1) / valid_text_length)
    correct = jnp.where(
        valid > 0.0,
        jnp.argmax(logits, axis=-1) == labels,
        jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy

def compute_simpo_loss(
    policy_chosen_logps: chex.Array,
    policy_rejected_logps: chex.Array,
    beta: float,
    gamma_beta_ratio: float,
    label_smoothing: float = None
):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    logits = pi_logratios - gamma_beta_ratio
    if label_smoothing:
        losses = (
            - jax.nn.sigmoid(beta * logits) * (1 - label_smoothing) \
            - jax.nn.sigmoid(-beta * logits) * label_smoothing
        )
    else:
        losses = - jax.nn.log_sigmoid(beta * logits)
    
    chosen_rewards = policy_chosen_logps
    rejected_rewards = policy_rejected_logps

    return losses, chosen_rewards, rejected_rewards
    

@dataclass
class SimPOTaskArguments(FlaxLMTaskArguments):
    train_template: Optional[str] = None
    simpo_beta: float = 0.1
    simpo_gamma_beta_ratio: float = 0.5
    simpo_label_smoothing: Optional[float] = None
    simpo_prompt_length: int = 1024
    simpo_response_length: int = 1024
    filter_no_bos: bool = False

@flax_tasks.register("simpo")
class SimPOTask(FlaxLMTask):
    ARG_CLASS = SimPOTaskArguments

    def init_tokenizer_collator(self):
        super().init_tokenizer_collator()
        self.train_template = find_template(self.args.train_template or self.args.model_name_or_path)(self.tokenizer)
    
    def __init__(self, args) -> None:
        super().__init__(args)
        self.beta = args.simpo_beta
        self.gamma_beta_ratio = args.simpo_gamma_beta_ratio
        self.label_pad_token_id = -100
    
    def _init_collator(self):
        self.collator = DPOCollator(
            self.tokenizer, 
            # padding=self.args.padding,
            padding_side=self.args.padding_side,
            max_length=self.args.max_length,
            # decoder_max_length=self.args.decoder_max_length,
            return_tensors="np")
        
    def encode_item(self, item):
        conversation = item["conversations"]
        chosen = self._encode_prompt_response(conversation, item["chosen"])
        rejected = self._encode_prompt_response(conversation, item["rejected"])

        return dict(
            chosen=chosen,
            rejected=rejected
        )

    def filter_item(self, item):
        trainables = sum(x >= 0 for x in item["chosen"]["labels"])
        trainables += sum(x >= 0 for x in item["rejected"]["labels"])

        if self.args.filter_no_bos:
            bos_count = sum(x == self.tokenizer.bos_token_id for x in item["chosen"]["input_ids"])
            bos_count += sum(x == self.tokenizer.bos_token_id for x in item["rejected"]["input_ids"])
            return trainables > 0 and bos_count >= 2
        else:
            return trainables > 0
    
    def _left_pad(self, seq, max_length, pad_value=0):
        if len(seq) < max_length:
            seq = [pad_value] * (max_length - len(seq)) + seq
        return seq

    def _encode_prompt_response(self, conversation, response):
        concat_inputs, concat_labels, concat_mask = [], [], []
        
        for i, uttr in enumerate(conversation):
            content, _ = self.train_template.handle_utterance(uttr, i)

            input_ids = self.tokenizer.encode(content, add_special_tokens=False)
            labels = [-100] * len(input_ids)

            concat_inputs.extend(input_ids)
            concat_labels.extend(labels)
            concat_mask.extend([1] * len(input_ids))

        if len(concat_inputs) > self.args.simpo_prompt_length:
            concat_inputs = concat_inputs[-self.args.simpo_prompt_length:]
            concat_labels = concat_labels[-self.args.simpo_prompt_length:]
            concat_mask = concat_mask[-self.args.simpo_prompt_length:]
        else:
            concat_inputs = self._left_pad(concat_inputs, self.args.simpo_prompt_length, pad_value=self.tokenizer.pad_token_id)
            concat_labels = self._left_pad(concat_labels, self.args.simpo_prompt_length, pad_value=-100)
            concat_mask = self._left_pad(concat_mask, self.args.simpo_prompt_length, pad_value=0)

        response_id = self.tokenizer.encode(response + self.tokenizer.eos_token, add_special_tokens=False)
        if len(response_id) > self.args.simpo_response_length:
            response_id = response_id[:self.args.simpo_response_length]

        concat_inputs.extend(response_id)
        concat_labels.extend(response_id)
        concat_mask.extend([1] * len(response_id))

        return self.truncate_dict({
            "input_ids": concat_inputs,
            "attention_mask": concat_mask,
            "labels": concat_labels
        })
        
    def filter_item(self, item):
        trainables = sum(x >= 0 for x in item["chosen"]["labels"])
        trainables += sum(x >= 0 for x in item["rejected"]["labels"])

        if self.args.filter_no_bos:
            bos_count = sum(x == self.tokenizer.bos_token_id for x in item["chosen"]["input_ids"])
            bos_count += sum(x == self.tokenizer.bos_token_id for x in item["rejected"]["input_ids"])
            return trainables > 0 and bos_count >= 2
        else:
            return trainables > 0
    
    def collate_step_outputs(self, outputs):
        keys = list(outputs[0].keys())
        return {k: jnp.stack([x[k] for x in outputs]).mean().tolist() for k in keys}

    @property
    def eval_metric_definitions(self):
        return {"loss": "min", "sft_accuracy": "max", "simpo_accuracy": "max", "chosen_rewards": "max", "rejected_rewards": "min"}
    
    def create_train_step(self, pjit_func, state_ps, PS):
        partition_spec = PS(("dp", "fsdp"), "sp")
        beta = self.args.simpo_beta
        gamma_beta_ratio = self.args.simpo_gamma_beta_ratio

        model_func = use_implicit_args(self.model)

        def train_step(state, batch):
            batch = with_sharding_constraint(batch, partition_spec)

            def calculate_loss(params, batch, beta):
                chosen, rejected = batch["chosen"], batch["rejected"]
                chosen_labels, rejected_labels = chosen.pop("labels")[:, 1:], rejected.pop("labels")[:, 1:]

                chosen_logits, policy_chosen_logps, policy_rejected_logps = get_model_batch_logps(
                    model_func, params, chosen, rejected, chosen_labels, rejected_labels,
                )
                simpo_losses, chosen_rewards, rejected_rewards = compute_simpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    beta,
                    gamma_beta_ratio
                )
                simpo_loss = simpo_losses.mean()
                simpo_accuracy = (chosen_rewards > rejected_rewards).mean()

                return simpo_loss, dict(
                    loss=simpo_loss, 
                    simpo_accuracy=simpo_accuracy,
                    chosen_rewards=chosen_rewards,
                    rejected_rewards=rejected_rewards,
                    )
            
            grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
            (loss, aux_output), grad = grad_fn(state.params, batch, beta)
            state = state.apply_gradients(grads=grad)
            aux_output["gradient_norm"] = global_norm(grad)
            return state, aux_output

        return pjit_func(
            train_step,
            in_shardings=(state_ps, PS()),
            out_shardings=(state_ps, PS()),
            donate_argnums=(0, 0),
        )

    def create_eval_step(self, pjit_func, state_ps, PS):
        partition_spec = PS(("dp", "fsdp"), "sp")
        beta = self.args.simpo_beta
        gamma_beta_ratio = self.args.simpo_gamma_beta_ratio
        model_func = use_implicit_args(self.model)

        def eval_step(state, batch):
            batch = with_sharding_constraint(batch, partition_spec)

            chosen, rejected = batch["chosen"], batch["rejected"]
            chosen_labels, rejected_labels = chosen.pop("labels")[:, 1:], rejected.pop("labels")[:, 1:]

            chosen_logits, policy_chosen_logps, policy_rejected_logps = get_model_batch_logps(
                model_func, state.params, chosen, rejected, chosen_labels, rejected_labels,
            )
            simpo_losses, chosen_rewards, rejected_rewards = compute_simpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                beta,
                gamma_beta_ratio
            )
            simpo_loss = simpo_losses.mean()
            simpo_accuracy = (chosen_rewards > rejected_rewards).mean()


            return dict(
                loss=simpo_loss, 
                simpo_accuracy=simpo_accuracy,
                chosen_rewards=chosen_rewards,
                rejected_rewards=rejected_rewards,
                )

        return pjit_func(
            eval_step,
            in_shardings=(state_ps, PS()),
            out_shardings=(PS()),
            donate_argnums=(0, 0),
        )
    

def test_simpo_loss():

    # Test simpoTask
    # [1, 3, 3]
    chosen = jnp.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.4, 0.5, 0.6]]])
    rejected = jnp.array([[[0.4, 0.2, 0.1], [0.4, 0.3, 0.6], [0.4, 0.2, 0.6]]])
    attention_mask = jnp.array([[1, 1, 1]])
    # [1, 3]
    labels = jnp.array([[-100, 1, 2]])

    chosen, rejected = chosen[:, :-1], rejected[:, :-1]
    labels= labels[:, 1:]

    chosen_loss_mask = labels >= 0 
    rejected_loss_mask = labels >= 0

    chosen_labels = jnp.where(chosen_loss_mask, labels, 0)
    rejected_labels = jnp.where(rejected_loss_mask, labels, 0)

    print(chosen_labels)

    chosen_logprobs = get_batch_logps(chosen, chosen_labels, chosen_loss_mask)
    rejected_logprobs = get_batch_logps(rejected, rejected_labels, rejected_loss_mask)

    simpo_losses, chosen_rewards, rejected_rewards = compute_simpo_loss(
        chosen_logprobs,
        rejected_logprobs,
        0.1,
        0.5,
        0.0
    )
    simpo_loss = simpo_losses.mean()
    simpo_accuracy = (chosen_rewards > rejected_rewards).mean()

    print(simpo_loss, simpo_accuracy)

if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        test_simpo_loss()