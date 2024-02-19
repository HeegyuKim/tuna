import random
import warnings
from collections.abc import Mapping
from collections import defaultdict
from dataclasses import dataclass, field
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np
import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from ..common import Registry


collators = Registry("collators")

@collators("default")
class DefaultCollator:
    __init__ = lambda *args, **kwargs: None
    def collate_dictlist(dl):

        out = defaultdict(list)

        for d in dl:
            for k, v in d.items():
                out[k].append(v)

        return out

def pad_sequence(seq, side: str, max_length: int, pad_value: int = 0):
    if len(seq) < max_length:
        remainders = [pad_value] * (max_length - len(seq))
        if side == "left":
            seq = remainders + seq
        else:
            seq = seq + remainders
    return seq

def pad_sequences(seqs, side: str, max_length: int, pad_value: int = 0, return_tensors="pt"):
    seqs = [pad_sequence(s, side, max_length, pad_value) for s in seqs]

    if return_tensors == "pt":
        return torch.tensor(seqs)
    if return_tensors == "np":
        return np.array(seqs)
    
    return seqs



@collators("sequence-classification")
@dataclass
class SequenceClassificationCollator(object):
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    padding_feature_keys: List[str] = field(
        default_factory=lambda: [
            "input_ids",
            "decoder_input_ids",
            "token_type_ids",
            "attention_mask",
            "decoder_attention_mask",
        ]
    )

    def __collate(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = defaultdict(list)
        for item in features :
            for k, v in item.items():
                out[k].append(v)
        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = self.__collate(features)
        labels = features.pop("labels")
        padding_features = {}

        for k in self.padding_feature_keys:
            if k in features:
                padding_features[k] = features.get(k)

        batch = self.tokenizer.pad(
            padding_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if self.return_tensors == "pt":
            labels = torch.tensor(labels, dtype=torch.long)

        batch["labels"] = labels

        for k in features.keys():
            if k not in self.padding_feature_keys:
                batch[k] = features[k]

        return batch

@collators("lm")
@dataclass
class LanguageModelCollator(object):
    tokenizer: PreTrainedTokenizerBase
    padding_side: str = "right"
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    is_encoder_decoder: bool = False
    padding_feature_keys: List[str] = field(
        default_factory=lambda: [
            "input_ids",
            "decoder_input_ids",
            "token_type_ids",
            "labels",
            "attention_mask",
            "decoder_attention_mask",
        ]
    )

    def __collate(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = defaultdict(list)
        for item in features :
            for k, v in item.items():
                out[k].append(v)
        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = self.__collate(features)
        padding_features = {}

        for k in self.padding_feature_keys:
            if k in features:
                padding_features[k] = features.get(k)

        self.tokenizer.padding_side = self.padding_side
        batch = self.tokenizer.pad(
            padding_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        for k in features.keys():
            if k not in self.padding_feature_keys:
                batch[k] = features[k]

        return batch


@collators("gen-lm")
@dataclass
class GenerativeLanguageModelCollator(object):
    tokenizer: PreTrainedTokenizerBase
    padding_side: str = "right"
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    decoder_max_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    padding_keys: List[str] = field(
        default_factory=lambda: [
            "input_ids",
            "decoder_input_ids",
            "token_type_ids",
            "attention_mask",
            "decoder_attention_mask",
            "labels"
        ]
    )

    def __collate(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = defaultdict(list)
        for item in features:
            for k, v in item.items():
                out[k].append(v)
        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.__collate(features)

        is_encoder_decoder = "decoder_input_ids" in features

        if self.padding == True or self.padding == "longest":
            if is_encoder_decoder:
                decoder_max_length = max(len(x) for x in batch["decoder_input_ids"])
            max_length = max(len(x) for x in batch["input_ids"])
        else:
            decoder_max_length = self.decoder_max_length
            max_length = self.max_length

        new_batch = {}
        for k in self.padding_keys:
            if k not in batch:
                continue
            
            if "labels" in k:
                pad_token_id = -100
            else:
                pad_token_id = self.tokenizer.pad_token_id

            if is_encoder_decoder and (k == "labels" or "decoder" in k):
                pad_max_length = decoder_max_length
            else:
                pad_max_length = max_length

            new_batch[k] = pad_sequences(
                batch[k],
                self.padding_side,
                pad_max_length,
                pad_token_id
            )

        return new_batch



@collators("dpo")
@dataclass
class DPOCollator(object):
    tokenizer: PreTrainedTokenizerBase
    padding_side: str = "right"
    # padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    decoder_max_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    padding_keys: List[str] = field(
        default_factory=lambda: [
            "input_ids",
            "decoder_input_ids",
            "token_type_ids",
            "attention_mask",
            "decoder_attention_mask",
            "labels"
        ]
    )

    def __collate(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = defaultdict(list)
        for item in features:
            for k, v in item.items():
                out[k].append(v)
        return out

    def __pad(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.__collate(features)

        is_encoder_decoder = "decoder_input_ids" in features

        for k in self.padding_keys:
            if k not in batch:
                continue
            
            if "labels" in k:
                pad_token_id = -100
            else:
                pad_token_id = self.tokenizer.pad_token_id

            if is_encoder_decoder and (k == "labels" or "decoder" in k):
                max_length = self.decoder_max_length
            else:
                max_length = self.max_length

            batch[k] = pad_sequences(
                batch[k],
                self.padding_side,
                max_length,
                pad_token_id
            )

        return batch
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        chosens = [x["chosen"] for x in features]
        rejections = [x["rejected"] for x in features]

        chosens = self.__pad(chosens)
        rejections = self.__pad(rejections)

        return {
            "chosen": chosens,
            "rejected": rejections
        }