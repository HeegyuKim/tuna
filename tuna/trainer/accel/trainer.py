
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from accelerate import Accelerator

from ..base import BaseTrainer, BaseTrainingArguments, trainers


@trainers.register("accelerator")
class AcceleratorTrainer(BaseTrainer):

    def setup(self):
        super().setup()

        self.accelerator = Accelerator()

        self.model, self.optimizer, self.lr_scheduler, self.train_loader, self.eval_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler,
            self.train_loader, self.eval_loader
        )

    def backward_loss(self, loss):
        self.accelerator.backward(loss)
