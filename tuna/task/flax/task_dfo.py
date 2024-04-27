from dataclasses import dataclass
from .flax_base import FlaxLMTask, flax_tasks, FlaxLMTaskArguments
from ..chat.train_templates import find_template
from ..dpo.collator import DDFOCollator
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


def masked_mean(arr, mask):
    return (arr * mask).sum(-1) / mask.sum(-1)

def get_logits(model, params, batch):
    logits = model(**batch, params=params, return_dict=True).logits
    return logits

def distil_loss(
    teacher_logits: chex.Array,
    student_logits: chex.Array,
    response_length: int,
    loss_mask: chex.Array,
):
    loss_mask = loss_mask[:, -response_length:]
    teacher_logits, student_logits = teacher_logits[:, -response_length-1:-1], student_logits[:, -response_length-1:-1]
    target = jax.nn.softmax(teacher_logits)
    preds = jax.nn.log_softmax(student_logits)
    
    kl_div = (target * (jnp.log(target) - preds)).sum(-1)
    loss = masked_mean(kl_div, loss_mask)

    return loss.mean()
    

@dataclass
class DFOTaskArguments(FlaxLMTaskArguments):
    train_template: Optional[str] = None
    distil_alpha: float = 0.1
    prompt_length: Optional[int] = None


@flax_tasks.register("dfo")
class DFOTask(FlaxLMTask):
    ARG_CLASS = DFOTaskArguments

    def init_tokenizer_collator(self):
        super().init_tokenizer_collator()
        self.train_template = find_template(self.args.train_template or self.args.model_name_or_path)(self.tokenizer)
    
    def __init__(self, args) -> None:
        super().__init__(args)
        self.label_pad_token_id = -100
    
    def _init_collator(self):
        self.collator = DDFOCollator(
            self.tokenizer, 
            # padding=self.args.padding,
            padding_side=self.args.padding_side,
            max_length=self.args.max_length,
            # decoder_max_length=self.args.decoder_max_length,
            return_tensors="np")
    
    def check_dataset(self, split, dataset):
        return dataset
        
    def encode_item(self, item):
        conversation = item["conversations"]
        full_chosen = self._encode_prompt_response(conversation, item["chosen"])
        full_rejected = self._encode_prompt_response(conversation, item["rejected"])
        chosen = self._encode_prompt_response(conversation, item["chosen"], True)
        rejected = self._encode_prompt_response(conversation, item["rejected"], False)

        return dict(
            full_chosen=full_chosen,
            full_rejected=full_rejected,
            chosen=chosen,
            rejected=rejected
        )


    def _encode_prompt_response(self, conversation, response, singleturn=False):
        concat_inputs, concat_labels = [], []
        prompt_length = self.args.prompt_length
        response_length = self.args.max_length - prompt_length
        
        if singleturn:
            if conversation[0]['role'] == 'system':
                conversation = conversation[:2]
            else:
                conversation = conversation[:1]

        for i, uttr in enumerate(conversation):
            content, _ = self.train_template.handle_utterance(uttr, i)

            input_ids = self.tokenizer.encode(content, add_special_tokens=False)
            labels = [-100] * len(input_ids)

            concat_inputs.extend(input_ids)
            concat_labels.extend(labels)

        if len(concat_inputs) > prompt_length:
            concat_inputs = concat_inputs[-prompt_length:]
            concat_labels = concat_labels[-prompt_length:]

        response_id = self.tokenizer.encode(response + self.tokenizer.eos_token, add_special_tokens=False)
        if len(response_id) > response_length:
            response_id = response_id[:response_length]
        concat_inputs.extend(response_id)
        concat_labels.extend(response_id)

        return self.truncate_dict({
            "input_ids": concat_inputs,
            "attention_mask": [1] * len(concat_inputs),
            # "labels": concat_labels
        })
        
    
    def collate_step_outputs(self, outputs):
        loss = jnp.stack([x["loss"] for x in outputs]).mean()
        chosen_loss = jnp.stack([x["chosen_loss"] for x in outputs]).mean().tolist()
        rejected_loss = jnp.stack([x["rejected_loss"] for x in outputs]).mean().tolist()
        return {"loss": loss, "chosen_loss": chosen_loss, "rejected_loss": rejected_loss}

    @property
    def eval_metric_definitions(self):
        return {"loss": "min"}
    
    def create_train_step(self, pjit_func, state_ps, PS):
        partition_spec = PS(("dp", "fsdp"), "sp")
        alpha = self.args.distil_alpha
        prompt_length = self.args.prompt_length
        response_length = self.args.max_length - prompt_length

        def train_step(state, batch):
            batch = with_sharding_constraint(batch, partition_spec)

            teacher_chosen_logits = get_logits(self.model, state.ref_params, batch["full_chosen"])
            # teacher_rejected_logits = get_logits(self.model, state.ref_params, batch["full_chosen"])

            def calculate_loss(params):
                student_chosen_logits = get_logits(self.model, params, batch["chosen"])
                # student_rejected_logits = get_logits(self.model, params, batch["rejected"])
                
                chosen_loss = distil_loss(teacher_chosen_logits, student_chosen_logits, response_length, batch["chosen"]["attention_mask"])
                # rejected_loss = distil_loss(teacher_rejected_logits, student_rejected_logits, response_length, batch["rejected"]["attention_mask"])
                
                # loss = (chosen_loss + rejected_loss) / 2
                loss = chosen_loss
                # return loss, dict(loss=loss, chosen_loss=chosen_loss, rejected_loss=rejected_loss)
                return loss, dict(loss=loss, chosen_loss=chosen_loss, rejected_loss=0)
            
            grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
            (loss, aux_output), grad = grad_fn(state.params)
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
        alpha = self.args.distil_alpha
        prompt_length = self.args.prompt_length
        response_length = self.args.max_length - prompt_length

        def eval_step(state, batch):
            batch = with_sharding_constraint(batch, partition_spec)

            chosen, rejected = batch["chosen"], batch["rejected"]
            chosen_labels, rejected_labels = chosen.pop("labels")[:, 1:], rejected.pop("labels")[:, 1:]

            chosen_loss_mask = chosen_labels >= 0 
            rejected_loss_mask = rejected_labels >= 0

            chosen_labels = jnp.where(chosen_loss_mask, 0, chosen_labels)
            rejected_labels = jnp.where(rejected_loss_mask, 0, rejected_labels)

            ref_chosen_logits, ref_rejected_logits = get_logits(self.model, state.ref_params, chosen), get_logits(self.model, state.ref_params, rejected)

            chosen_logits, rejected_logits = get_logits(self.model, state.params, chosen), get_logits(self.model, state.params, rejected)
            
            chosen_loss = distil_loss(ref_chosen_logits, chosen_logits, response_length, chosen_loss_mask)
            rejected_loss = distil_loss(ref_rejected_logits, rejected_logits, response_length, rejected_loss_mask)
            
            loss = (chosen_loss + rejected_loss) / 2
            return dict(loss=loss, chosen_loss=chosen_loss, rejected_loss=rejected_loss)

        return pjit_func(
            eval_step,
            in_shardings=(state_ps, PS()),
            out_shardings=(PS()),
            donate_argnums=(0, 0),
        )
    