from dataclasses import dataclass
from .flax_base import FlaxLMTask, flax_tasks, FlaxLMTaskArguments
from ..chat.train_templates import find_template
from ..dpo.collator import DPOCollator
from typing import Optional, Union

import jax, flax
import jax.numpy as jnp

from fjformer import with_sharding_constraint
from fjformer.func.loss_func import (
    cross_entropy_loss_and_accuracy,
    SpecialLossNormalizingFactor,
    get_loss_normalizing_factor_and_weights,
    compute_weighted_cross_entropy_and_accuracy,
)
import chex


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

def compute_sft_loss(batch, logits, labels, label_smoothing_factor, z_loss):
    loss_normalizing_factor = (
        SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS
    )
    # loss_weights is 1 unless the label is <= 0 or the attention mask is 0
    loss_weights = jnp.where(
        (batch["attention_mask"][:, 1:] != 0) & (labels >= 0), 1, 0
    )
    lnf, weights = get_loss_normalizing_factor_and_weights(
        loss_normalizing_factor,
        {
            "decoder_target_tokens": labels,
            "decoder_loss_weights": loss_weights,
        },
    )
    (
        loss,
        z_loss_computed,
        weight_sum,
        accuracy,
    ) = compute_weighted_cross_entropy_and_accuracy(
        logits=logits,#[:, :-1, :],
        targets=labels,
        weights=weights,
        label_smoothing=label_smoothing_factor,
        z_loss=z_loss,
        loss_normalizing_factor=lnf,
    )
    return loss, accuracy

def compute_orpo_loss(
    policy_chosen_logps: chex.Array,
    policy_rejected_logps: chex.Array,
    beta: float,
):
    log_odds = (policy_chosen_logps - policy_rejected_logps) - (
        jnp.log1p(-jnp.exp(policy_chosen_logps)) - jnp.log1p(-jnp.exp(policy_rejected_logps))
    )
    sig_ratio = jax.nn.sigmoid(log_odds)
    ratio = jnp.log(sig_ratio)
    losses = beta * ratio
    
    chosen_rewards = beta * policy_chosen_logps
    rejected_rewards = beta * policy_rejected_logps

    return losses, chosen_rewards, rejected_rewards, ratio.mean(), log_odds.mean()
    

@dataclass
class ORPOTaskArguments(FlaxLMTaskArguments):
    train_template: Optional[str] = None
    orpo_beta: float = 0.1

@flax_tasks.register("orpo")
class ORPOTask(FlaxLMTask):
    ARG_CLASS = ORPOTaskArguments

    def init_tokenizer_collator(self):
        super().init_tokenizer_collator()
        self.train_template = find_template(self.args.train_template or self.args.model_name_or_path)(self.tokenizer)
    
    def __init__(self, args) -> None:
        super().__init__(args)
        self.beta = args.orpo_beta
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


    def _encode_prompt_response(self, conversation, response):
        concat_inputs, concat_labels = [], []
        
        for i, uttr in enumerate(conversation):
            content, _ = self.train_template.handle_utterance(uttr, i)

            input_ids = self.tokenizer.encode(content, add_special_tokens=False)
            labels = [-100] * len(input_ids)

            concat_inputs.extend(input_ids)
            concat_labels.extend(labels)

        response_id = self.tokenizer.encode(response + self.tokenizer.eos_token, add_special_tokens=False)
        concat_inputs.extend(response_id)
        concat_labels.extend(response_id)

        return self.truncate_dict({
            "input_ids": concat_inputs,
            "attention_mask": [1] * len(concat_inputs),
            "labels": concat_labels
        })
        
    def filter_item(self, item):
        trainables = sum(x >= 0 for x in item["chosen"]["labels"])
        trainables += sum(x >= 0 for x in item["rejected"]["labels"])
        return trainables > 0
    
    def collate_step_outputs(self, outputs):
        keys = list(outputs[0].keys())
        return {k: jnp.stack([x[k] for x in outputs]).mean().tolist() for k in keys}

    @property
    def eval_metric_definitions(self):
        return {"loss": "min", "sft_accuracy": "max", "orpo_accuracy": "max", "chosen_rewards": "max", "rejected_rewards": "min"}
    
    def create_train_step(self, pjit_func, state_ps, PS):
        partition_spec = PS(("dp", "fsdp"), "sp")
        beta = self.args.orpo_beta
        label_smoothing_factor = self.args.label_smoothing_factor
        z_loss = self.args.z_loss

        def train_step(state, batch):
            batch = with_sharding_constraint(batch, partition_spec)

            def calculate_loss(params, batch, beta):
                chosen, rejected = batch["chosen"], batch["rejected"]
                chosen_labels, rejected_labels = chosen.pop("labels")[:, 1:], rejected.pop("labels")[:, 1:]

                chosen_logits, policy_chosen_logps, policy_rejected_logps = get_model_batch_logps(
                    self.model, params, chosen, rejected, chosen_labels, rejected_labels,
                )
                sft_loss, sft_accuracy = compute_sft_loss(
                    chosen, chosen_logits, chosen_labels, label_smoothing_factor, z_loss
                )
                orpo_losses, chosen_rewards, rejected_rewards, odds_ratio, log_odds = compute_orpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    beta
                )
                orpo_loss = -orpo_losses.mean()
                orpo_accuracy = (chosen_rewards > rejected_rewards).mean()

                loss = sft_loss + orpo_loss

                return loss, dict(
                    loss=loss, 
                    orpo_loss=orpo_loss,
                    sft_loss=sft_loss,
                    sft_accuracy=sft_accuracy,
                    orpo_accuracy=orpo_accuracy,
                    chosen_rewards=chosen_rewards,
                    rejected_rewards=rejected_rewards,
                    odds_ratio=odds_ratio,
                    log_odds=log_odds
                    )
            
            grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
            (loss, aux_output), grad = grad_fn(state.params, batch, beta)
            state = state.apply_gradients(grads=grad)
            return state, aux_output

        return pjit_func(
            train_step,
            in_shardings=(state_ps, PS()),
            out_shardings=(state_ps, PS()),
            donate_argnums=(0, 0),
        )

    def create_eval_step(self, pjit_func, state_ps, PS):
        partition_spec = PS(("dp", "fsdp"), "sp")
        beta = self.args.orpo_beta
        label_smoothing_factor = self.args.label_smoothing_factor
        z_loss = self.args.z_loss

        def eval_step(state, batch):
            batch = with_sharding_constraint(batch, partition_spec)

            chosen, rejected = batch["chosen"], batch["rejected"]
            chosen_labels, rejected_labels = chosen.pop("labels")[:, 1:], rejected.pop("labels")[:, 1:]

            chosen_logits, policy_chosen_logps, policy_rejected_logps = get_model_batch_logps(
                self.model, state.params, chosen, rejected, chosen_labels, rejected_labels,
            )
            sft_loss, sft_accuracy = compute_sft_loss(
                chosen, chosen_logits, chosen_labels, label_smoothing_factor, z_loss
            )
            orpo_losses, chosen_rewards, rejected_rewards, odds_ratio, log_odds = compute_orpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                beta
            )
            orpo_loss = -orpo_losses.mean()
            orpo_accuracy = (chosen_rewards > rejected_rewards).mean()

            loss = sft_loss + orpo_loss

            return dict(
                loss=loss, 
                orpo_loss=orpo_loss,
                sft_loss=sft_loss,
                sft_accuracy=sft_accuracy,
                orpo_accuracy=orpo_accuracy,
                chosen_rewards=chosen_rewards,
                rejected_rewards=rejected_rewards,
                odds_ratio=odds_ratio,
                log_odds=log_odds
                )

        return pjit_func(
            eval_step,
            in_shardings=(state_ps, PS()),
            out_shardings=(PS()),
            donate_argnums=(0, 0),
        )
    

def test_orpo_loss():

    # Test ORPOTask
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

    orpo_losses, chosen_rewards, rejected_rewards = compute_orpo_loss(
        chosen_logprobs,
        rejected_logprobs,
        0.1
    )
    orpo_loss = orpo_losses.mean()
    orpo_accuracy = (chosen_rewards > rejected_rewards).mean()

    print(orpo_loss, orpo_accuracy)

if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        test_orpo_loss()