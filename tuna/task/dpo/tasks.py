from dataclasses import dataclass, field
from datasets import DatasetDict
from ..base import LMTask, tasks, TaskArguments, TensorWrapper
from ..chat.train_templates import find_template
from ..collator import GenerativeVLMCollator
from typing import Optional, Union, List, Dict, Any, Tuple
from copy import deepcopy
from collections import defaultdict
from .collator import DPOCollator, DCOCollator

import torch
import torch.nn.functional as F


@dataclass
class DPOTaskArguments(TaskArguments):
    train_template: Optional[str] = None
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"
    dpo_prompt_length: int = 1024
    dpo_response_length: int = 1024

@tasks.register("dpo")
class DPOTask(LMTask):
    ARG_CLASS = DPOTaskArguments

    def __init__(self, args, artifacts, wrapper: Union[TensorWrapper, str]) -> None:
        super().__init__(args, artifacts, wrapper)
        self.train_template = find_template(args.train_template or artifacts.get("model_name_or_path"))(self.tokenizer)
        self.beta = args.dpo_beta
        self.loss_type = args.dpo_loss_type
        self.label_pad_token_id = -100

    def set_model(self, model):
        super().set_model(model)

        self.is_encoder_decoder = getattr(self.model.config, "is_encoder_decoder", False)
        
        from peft import PeftModel
        if not isinstance(self.model, PeftModel):
            self.ref_model = deepcopy(self.model)
        else:
            self.ref_model = None
    
    def _init_collator(self):
        self.collator = DPOCollator(
            self.tokenizer, 
            # padding=self.args.padding,
            padding_side=self.args.padding_side,
            max_length=self.args.max_length,
            # decoder_max_length=self.args.decoder_max_length,
            return_tensors="pt")

    def filter_item(self, item):
        trainables = sum(x >= 0 for x in item["chosen"]["labels"])
        trainables += sum(x >= 0 for x in item["rejected"]["labels"])
        return trainables > 0
    
    def encode_item(self, item):
        conversation = item["conversations"]
        chosen = self._encode_prompt_response(conversation, item["chosen"])
        rejected = self._encode_prompt_response(conversation, item["rejected"])

        return dict(
            chosen=chosen,
            rejected=rejected
        )

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

        if len(concat_inputs) > self.args.dpo_prompt_length:
            concat_inputs = concat_inputs[-self.args.dpo_prompt_length:]
            concat_labels = concat_labels[-self.args.dpo_prompt_length:]
            concat_mask = concat_mask[-self.args.dpo_prompt_length:]
        else:
            concat_inputs = self._left_pad(concat_inputs, self.args.dpo_prompt_length, self.tokenizer.pad_token_id)
            concat_labels = self._left_pad(concat_labels, self.args.dpo_prompt_length, pad_value=-100)
            concat_mask = self._left_pad(concat_mask, self.args.dpo_prompt_length, 0)

        response_id = self.tokenizer.encode(response + self.tokenizer.eos_token, add_special_tokens=False)
        if len(response_id) > self.args.dpo_response_length:
            response_id = response_id[:self.args.dpo_response_length]

        concat_inputs.extend(response_id)
        concat_labels.extend(response_id)

        return self.truncate_dict({
            "input_ids": concat_inputs,
            "attention_mask": [1] * len(concat_inputs),
            "labels": concat_labels
        })
        
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not self.is_encoder_decoder:
            prompt_length = self.args.dpo_prompt_length
            labels = labels[:, prompt_length:].clone()
            logits = logits[:, prompt_length-1:-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels = labels.masked_fill(labels < 0, 0)

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)


    def _batch_logps(self, model, chosen_input, rejected_input, chosen_labels, rejected_labels):

        chosen_output = model(**chosen_input)
        chosen = chosen_output.logits

        # if self.is_encoder_decoder:
        #     rejected_input["encoder_outputs"] = (chosen_output["encoder_last_hidden_state"],)
        
        rejected_output = model(**rejected_input)
        rejected = rejected_output.logits
        
        # return (rejected - chosen).mean()
        chosen_logprobs = self._get_batch_logps(chosen, chosen_labels)
        rejected_logprobs = self._get_batch_logps(rejected, rejected_labels)
        return chosen_logprobs, rejected_logprobs

    def step(self, batch, step):
        chosen_input, rejected_input = self.wrapper(batch["chosen"]), self.wrapper(batch["rejected"])

        chosen_labels = chosen_input.pop("labels")
        rejected_labels = rejected_input.pop("labels")


        policy_chosen_logps, policy_rejected_logps = self._batch_logps(self.model, chosen_input, rejected_input, chosen_labels, rejected_labels)
        if self.ref_model:
            reference_chosen_logps, reference_rejected_logps = self._batch_logps(self.ref_model, chosen_input, rejected_input, chosen_labels, rejected_labels)
        else:
            with self.model.disable_adapter():
                reference_chosen_logps, reference_rejected_logps = self._batch_logps(self.model, chosen_input, rejected_input, chosen_labels, rejected_labels)
        
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        return dict(
            loss=losses,
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            accuracy=(chosen_rewards > rejected_rewards).float()
        )

    def train_step(self, batch, step):
        step_output = self.step(batch, step)
        step_output["loss"] = step_output["loss"].mean()
        return step_output

    def collate_step_outputs(self, outputs):
        keys = outputs[0].keys()# ["loss", "chosen_rewards", "rejected_rewards", "accuracy"]
        return {
            k: torch.stack([x[k] for x in outputs]).mean() for k in keys
        }

    @property
    def eval_metric_definitions(self):
        return {
            "accuracy": "max",
            "loss": "min",
            }
    

@dataclass
class DCOTaskArguments(DPOTaskArguments):
    dco_gamma: float = 0.1
    dco_detach: bool = False

@tasks.register("dco")
class DCOTask(DPOTask):
    ARG_CLASS = DCOTaskArguments

    def _init_collator(self):
        self.collator = DCOCollator(
            self.tokenizer, 
            # padding=self.args.padding,
            padding_side=self.args.padding_side,
            max_length=self.args.max_length,
            # decoder_max_length=self.args.decoder_max_length,
            return_tensors="pt")
        
    def encode_item(self, item):
        conversation = deepcopy(item["conversations"])
        chosen = self._encode_prompt_response(conversation, item["chosen"])
        rejected = self._encode_prompt_response(conversation, item["rejected"])

        chosen_declined = self._encode_critique(conversation, item["chosen"], item["chosen_critique"], item["rejected"])
        rejected_improved = self._encode_critique(conversation, item["rejected"], item["rejected_critique"], item["chosen"])

        return dict(
            chosen=chosen,
            rejected=rejected,
            chosen_declined=chosen_declined,
            rejected_improved=rejected_improved
        )

    def filter_item(self, item):
        trainables = sum(x >= 0 for x in item["chosen"]["labels"])
        trainables += sum(x >= 0 for x in item["rejected"]["labels"])
        trainables += sum(x >= 0 for x in item["chosen_declined"]["labels"])
        trainables += sum(x >= 0 for x in item["rejected_improved"]["labels"])
        return trainables > 0

    def _encode_critique(self, conversation, response, critique, revision):
        return self._encode_prompt_response(
            conversation + [
                {
                    "role": "assistant",
                    "content": response
                },
                {
                    "role": "user",
                    "content": critique
                }
            ],
            revision
        )
    
    def step(self, batch, step):
        chosen_input, rejected_input = self.wrapper(batch["chosen"]), self.wrapper(batch["rejected"])

        chosen_labels = chosen_input.pop("labels")
        rejected_labels = rejected_input.pop("labels")

        policy_chosen_logps, policy_rejected_logps = self._batch_logps(self.model, chosen_input, rejected_input, chosen_labels, rejected_labels)
        if self.ref_model:
            reference_chosen_logps, reference_rejected_logps = self._batch_logps(self.ref_model, chosen_input, rejected_input, chosen_labels, rejected_labels)
        else:
            with self.model.disable_adapter():
                reference_chosen_logps, reference_rejected_logps = self._batch_logps(self.model, chosen_input, rejected_input, chosen_labels, rejected_labels)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        chosen_declined_input, rejected_improved_input = self.wrapper(batch["chosen_declined"]), self.wrapper(batch["rejected_improved"])
        chosen_declined_labels = chosen_declined_input.pop("labels")
        rejected_improved_labels = rejected_improved_input.pop("labels")

        policy_chosen_declined_logps, policy_rejected_improved_logps = self._batch_logps(self.model, chosen_declined_input, rejected_improved_input, chosen_declined_labels, rejected_improved_labels)
        if self.ref_model:
            reference_chosen_declined_logps, reference_rejected_improved_logps = self._batch_logps(self.ref_model, chosen_declined_input, rejected_improved_input, chosen_declined_labels, rejected_improved_labels)
        else:
            with self.model.disable_adapter():
                reference_chosen_declined_logps, reference_rejected_improved_logps = self._batch_logps(self.model, chosen_declined_input, rejected_improved_input, chosen_declined_labels, rejected_improved_labels)

        if self.args.dco_detach:
            policy_chosen_declined_logps = policy_chosen_declined_logps.detach()

        losses_declined, rejected_rewards_improved, chosen_rewards_declined = self.dpo_loss(
            policy_rejected_improved_logps,
            policy_chosen_declined_logps,
            reference_rejected_improved_logps,
            reference_chosen_declined_logps,
        )

        dpo_loss, dco_loss = losses.mean(), losses_declined.mean()
        loss = dpo_loss + self.args.dco_gamma * dco_loss

        return dict(
            loss=loss,
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            accuracy=(chosen_rewards > rejected_rewards).float(),
            dpo_loss=dpo_loss,
            dco_loss=dco_loss,
            dco_accuracy=(chosen_rewards_declined > rejected_rewards_improved).float(),
            chosen_rewards_declined=chosen_rewards_declined,
            rejected_rewards_improved=rejected_rewards_improved
        )


@tasks.register("dco-d")
class DCO_D_Task(DCOTask):
    def __init__(self, args, artifacts, wrapper: TensorWrapper | str) -> None:
        super().__init__(args, artifacts, wrapper)
        self.args.dco_detach = True


@tasks.register("dco-v4")
class DCOTaskV4(DCOTask):
        
    def encode_item(self, item):
        conversation = deepcopy(item["conversations"])
        chosen = self._encode_prompt_response(conversation, item["chosen"])
        rejected = self._encode_prompt_response(conversation, item["rejected"])

        chosen_declined = self._encode_critique(conversation, item["rejected"], item["chosen_critique"], item["chosen"])
        rejected_improved = self._encode_critique(conversation, item["rejected"], item["rejected_critique"], item["chosen"])

        return dict(
            chosen=chosen,
            rejected=rejected,
            chosen_declined=chosen_declined,
            rejected_improved=rejected_improved
        )
    
@tasks.register("dco-v4d")
class DCOTaskV4D(DCOTaskV4):
    def __init__(self, args, artifacts, wrapper: TensorWrapper | str) -> None:
        super().__init__(args, artifacts, wrapper)
        self.args.dco_detach = True
