from dataclasses import dataclass, field
from ..chat.train_templates import find_template
from typing import Optional, Union, List, Dict, Any, Tuple
from collections import defaultdict
import torch
import numpy as np

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


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

    def pad(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
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
                pad_token_id,
                return_tensors=self.return_tensors
            )

        return batch
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        chosens = [x["chosen"] for x in features]
        rejections = [x["rejected"] for x in features]

        chosens = self.pad(chosens)
        rejections = self.pad(rejections)

        return {
            "chosen": chosens,
            "rejected": rejections
        }


DDFO_KEYS = ["chosen", "rejected", "full_chosen", "full_rejected"]
@dataclass
class DDFOCollator(DPOCollator):
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_output = {}
        for k in DDFO_KEYS:
            batch = [x[k] for x in features]
            batch = self.pad(batch)
            final_output[k] = batch

        return final_output



class DCOCollator(DPOCollator):

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        chosens = [x["chosen"] for x in features]
        rejections = [x["rejected"] for x in features]
        chosen_declined = [x["chosen_declined"] for x in features]
        rejected_improved = [x["rejected_improved"] for x in features]

        chosens, rejections = self.pad(chosens), self.pad(rejections)
        chosen_declined, rejected_improved = self.pad(chosen_declined), self.pad(rejected_improved)

        return {
            "chosen": chosens,
            "rejected": rejections,
            "chosen_declined": chosen_declined,
            "rejected_improved": rejected_improved
        }