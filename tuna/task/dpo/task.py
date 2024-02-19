from typing import Union, Tuple
from task.base import TensorWrapper

from dataclasses import dataclass
import torch.nn as nn
from ..base import Task, collate_dictlist
from trainer.utils import convert_dict_tensor_devices
from itertools import chain

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm


@dataclass
class DPOArguments():
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"
    
class DPOTask(Task):

    def __init__(self, args: DPOArguments, model: nn.Module, wrapper, ref_model = None) -> None:
        super().__init__(model, wrapper)

        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.label_pad_token_id = -100
        self.beta = args.dpo_beta
        self.loss_type = args.dpo_loss_type
        self.ref_model = ref_model

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

        # if not self.is_encoder_decoder:
        #     labels = labels[:, 1:].clone()
        #     logits = logits[:, :-1, :]
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

        if self.is_encoder_decoder:
            rejected_input["encoder_outputs"] = (chosen_output["encoder_last_hidden_state"],)
        
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
