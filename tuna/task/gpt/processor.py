from dataclasses import dataclass
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Dict
from tqdm import tqdm
import random
from functools import partial

from task.base import DatasetArguments
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset as HDataset, IterableDataset, interleave_datasets
from ..base import GPTDatasetArguments


class BaseProcessor():
    def __init__(self, args: DatasetArguments) -> None:
        self.args = args

    def load_dataset_from_path(self, path, args):
        return load_dataset(
            path, 
            use_auth_token=True,
            streaming=args.dataset_streaming
            )
    
    def load_dataset(self, args: GPTDatasetArguments):
        dataset_names = args.dataset.split("&")
        print(dataset_names)

        datasets = [self.load_dataset_from_path(path, args) for path in dataset_names]

        if len(datasets) > 1:
            combined_dict = DatasetDict()
            for split in ["train", "test"]:
                split_sets = [x[split] for x in datasets if split in x]
                if not split_sets:
                    continue

                if args.combine_strategy == "interleave":
                    combined_dict[split] = interleave_datasets(
                        split_sets,
                        seed=42,
                        stopping_strategy="all_exhausted"
                        )
                else:
                    combined_dict[split] = concatenate_datasets(split_sets)
                    
            return combined_dict
        else:
            return datasets[0]

    def process(self, item, tokenizer):
        pass


class MachineTranslationProcessor(BaseProcessor):

    def process(self, item, tokenizer):
        if random.random() > 0.5:
            src, tgt = item["ko"], item["en"]
        else:
            src, tgt = item["en"], item["ko"]
        
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        text = f"{bos} {src} {eos} {tgt} {eos}"

        return [
            {"from": "gpt", "value": text}
        ]

class VicunaProcessor(BaseProcessor):

    def process(self, item, tokenizer):
        convs = item["conversations"]
        out_convs = []
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        for conv in convs:
            who = conv["from"]
            text = conv["value"]

            if who == "human":
                text = f"### Human:\n{text}\n\n### Assistant:\n"
            else:
                text = f"{text} {eos}\n\n"

            out_convs.append({
                "from": who, "value": text
            })

        return out_convs
    
class HHRLHFProcessor(BaseProcessor):
    
    def load_dataset_from_path(self, path, args):
        ds = load_dataset(
            "heegyu/hh-rlhf-vicuna-format", 
            use_auth_token=True,
            streaming=args.dataset_streaming
            )

        if path == "hh-rlhf":
            return ds
        elif path == "helpful-rlhf":
            return ds.filter(lambda x: "helpful" in x["source"])
        elif path == "harmless-rlhf":
            return ds.filter(lambda x: "harmless" in x["source"])
        else:
            raise NotImplementedError(f"{path} is unknown!!")
    

    def process(self, item, tokenizer):
        convs = item["context"] + [item["instruction"], item["chosen"]]
        out_convs = []
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        for conv in convs:
            who = conv["from"]
            text = conv["value"]

            if who == "human":
                text = f"### Human:\n{text}\n\n### Assistant:\n"
            else:
                text = f"{text} {eos}\n\n"

            out_convs.append({
                "from": who, "value": text
            })

        return out_convs

class AlpacaProcessor(BaseProcessor):

    def process(self, item, tokenizer):
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        input, instruction, output = item["input"].strip(), item["instruction"].strip(), item["output"].strip()

        if input:
            prefix = f"### Input:\n{input}\n\n### Human:\n{instruction}\n\n### Assistant:\n"
        else:
            prefix = f"### Human:\n{instruction}\n\n### Assistant:\n"

        output = f"{output} {eos}"

        return [
            {"from": "human", "value": prefix},
            {"from": "bot", "value": output}
        ]
    
class AlpacaZephyrProcessor(BaseProcessor):

    def process(self, item, tokenizer):
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        input, instruction, output = item["input"].strip(), item["instruction"].strip(), item["output"].strip()

        if input:
            prefix = f"<|system|>\n{input}</s><|user|>\n{instruction}</s>\n<|assistant|>\n"
        else:
            prefix = f"<|user|>\n{instruction}</s>\n<|assistant|>\n"

        output = f"{output} {eos}"

        return [
            {"from": "human", "value": prefix},
            {"from": "bot", "value": output}
        ]
    
class BeaverZephyrProcessor(BaseProcessor):

    def load_dataset_from_path(self, path, args):
        ds = load_dataset(
            "PKU-Alignment/BeaverTails", 
            use_auth_token=True,
            streaming=args.dataset_streaming
            )
        dd = DatasetDict()
        # dd["train"] = ds["30k_train"]
        # dd["test"] = ds["30k_test"]
        dd["train"] = ds["330k_train"]
        dd["test"] = ds["330k_test"]
        return dd

    def process(self, item, tokenizer):
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        instruction, output = item["prompt"].strip(), item["response"].strip()

        item["source"] = "beavertails-safe" if item["is_safe"] else "beavertails-harmless"

        prefix = f"<|user|>\n{instruction}</s>\n<|assistant|>\n"

        output = f"{output} {eos}"

        return [
            {"from": "human", "value": prefix},
            {"from": "bot", "value": output}
        ]
    
class MarxProcessor(HHRLHFProcessor):

    def process(self, item, tokenizer):
        convs = item["context"] + [item["instruction"], item["chosen"]]
        out_convs = []
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        for conv in convs:
            who = conv["from"]
            text = conv["value"]

            if who == "human":
                text = f"### HUMAN: {text}\n\n### RESPONSE: "
            else:
                text = f"{text} {eos}\n\n"

            out_convs.append({
                "from": who, "value": text
            })

        return out_convs

    
class SimpleProcessor(HHRLHFProcessor):

    def process(self, item, tokenizer):
        convs = item["context"] + [item["instruction"], item["chosen"]]
        out_convs = []
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        for conv in convs:
            who = conv["from"]
            text = conv["value"]

            if who == "human":
                text = f"Human: {text}\n\nAssistant: "
            else:
                text = f"{text} {eos}\n\n"

            out_convs.append({
                "from": who, "value": text
            })

        return out_convs
    
class ZephyrProcessor(HHRLHFProcessor):

    def process(self, item, tokenizer):
        convs = item["context"] + [item["instruction"], item["chosen"]]
        out_convs = []
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        for conv in convs:
            who = conv["from"]
            text = conv["value"]

            if who == "human":
                text = f"<|user|>\n{text} {eos}\n<|assistant|>\n"
            elif who == "system":
                text = f"<|system|>{text} {eos}\n"
            else:
                text = f"{text} {eos}\n"

            out_convs.append({
                "from": who, "value": text
            })

        return out_convs
    
class SimpleCoTProcessor(HHRLHFProcessor):

    def process(self, item, tokenizer):
        convs = item["context"] + [item["instruction"], item["chosen"]]
        rejected = item["rejected"]["value"]
        out_convs = []
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        for i, conv in enumerate(convs):
            who = conv["from"]
            text = conv["value"]

            if who == "human":
                if i == len(convs) - 2:
                    text = f"Human: {text}\n\nHarmful Assistant: {rejected}\n\nSafe Assistant:"
                else:
                    text = f"Human: {text}\n\nAssistant:"
            else:
                text = f"{text} {eos}\n\n"

            out_convs.append({
                "from": who, "value": text
            })

        return out_convs

class SafeRLHFSafer(BaseProcessor):
    # PKU-Alignment/PKU-SafeRLHF
    def load_dataset(self, args: GPTDatasetArguments):
        dataset = super().load_dataset(args)

        final_ds = DatasetDict()

        for k, ds in dataset.items():
            items = []
            for item in ds:
                if item["is_response_0_safe"] and item["safer_response_id"] == 0:
                    items.append(dict(
                        instruction={
                            "from": "human",
                            "value": item["prompt"],
                        },
                        chosen={
                            "from": "gpt",
                            "value": item["response_0"],
                        }
                    ))

                if item["is_response_1_safe"] and item["safer_response_id"] == 1:
                    items.append(dict(
                        instruction={
                            "from": "human",
                            "value": item["prompt"],
                        },
                        chosen={
                            "from": "gpt",
                            "value": item["response_1"],
                        }
                    ))

            final_ds[k] = HDataset(items)
        
        return final_ds

    def process(self, item, tokenizer):
        convs = [item["instruction"], item["chosen"]]
        out_convs = []
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        for conv in convs:
            who = conv["from"]
            text = conv["value"]

            if who == "human":
                text = f"Human: {text}\n\nAssistant: "
            else:
                text = f"{text} {eos}\n\n"

            out_convs.append({
                "from": who, "value": text
            })

        return out_convs
    
class SafeRLHFJH(BaseProcessor):
    # PKU-Alignment/PKU-SafeRLHF
    def load_dataset(self, args: GPTDatasetArguments):
        dataset = super().load_dataset(args)

        final_ds = DatasetDict()

        for k, ds in dataset.items():
            items = []
            for item in ds:
                items.append(dict(
                    instruction={
                        "from": "human",
                        "value": item["prompt"],
                    },
                    chosen={
                        "from": "gpt",
                        "value": item["response_0"],
                        "safe": item["is_response_0_safe"]
                    }
                ))

                items.append(dict(
                    instruction={
                        "from": "human",
                        "value": item["prompt"],
                    },
                    chosen={
                        "from": "gpt",
                        "value": item["response_1"],
                        "safe": item["is_response_1_safe"]
                    }
                ))
            final_ds[k] = HDataset.from_list(items)
        
        return final_ds

    def process(self, item, tokenizer):
        convs = [item["instruction"], item["chosen"]]
        out_convs = []
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        for conv in convs:
            who = conv["from"]
            text = conv["value"]

            if who == "human":
                text = f"Human: {text}\n\nAssistant: "
            elif conv["safe"]:
                text = f"safe: {text} {eos}\n\n"
            else:
                text = f"harmful: {text} {eos}\n\n"

            out_convs.append({
                "from": who, "value": text
            })

        return out_convs

class SafeRLHFJHForGen(BaseProcessor):

    def process(self, item, tokenizer):
        prompt = item["prompt"]

        return [{
            "from": "human",
            "value": f"Human: {prompt}\n\nAssistant: safe:"
        }]


class SafeRLHFSafer(BaseProcessor):
    # PKU-Alignment/PKU-SafeRLHF
    def load_dataset(self, args: GPTDatasetArguments):
        dataset = super().load_dataset(args)

        final_ds = DatasetDict()

        for k, ds in dataset.items():
            items = []
            for item in ds:
                if not (item["is_response_0_safe"] and item["is_response_1_safe"]):
                    chosen = item["safer_response_id"]
                    rejected = 1 - chosen

                    items.append(dict(
                        instruction=item["prompt"],
                        chosen=item[f"response_{chosen}"],
                        rejected=item[f"response_{rejected}"],
                    ))

            final_ds[k] = HDataset.from_list(items)
        
        return final_ds

    def process(self, item, tokenizer):
        instruction, chosen, rejected = item["instruction"], item["chosen"], item["rejected"]
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        return [
            {
                "from": "human",
                "value": f"Human: {instruction}\n\nAssistant:"
            },
            {
                "from": "gpt",
                "value": f"{chosen} {eos}\n\n"
            }
            ]

# class SafeRLHFSaferForGen(SafeRLHFSafer):

#     def process(self, item, tokenizer):
#         instruction, chosen, rejected = item["instruction"], item["chosen"], item["rejected"]
#         bos, eos = tokenizer.bos_token, tokenizer.eos_token
#         return [{
#                 "from": "human",
#             "value": f"Human: {instruction}\n\nAssistant:"
#         }]
    
class SafeRLHFSaferCoT(SafeRLHFSafer):

    def process(self, item, tokenizer):
        instruction, chosen, rejected = item["instruction"], item["chosen"], item["rejected"]
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        return [
            {
                "from": "human",
                "value": f"Human: {instruction}\n\nHarmful Assistant: {rejected}\n\nSafe Assistant:"
            },
            {
                "from": "gpt",
                "value": f"{chosen} {eos}\n\n"
            }
            ]

# class SafeRLHFSaferCoTForGen(SafeRLHFSaferCoT):

#     def process(self, item, tokenizer):
#         instruction, chosen, rejected = item["instruction"], item["chosen"], item["rejected"]
#         bos, eos = tokenizer.bos_token, tokenizer.eos_token
#         return [{
#             "from": "human",
#             "value": f"Human: {instruction}\n\nHarmful Assistant: {rejected}\n\nSafe Assistant:"
#         }]

PROCESSOR_LIST = {
    "mt_ko_en": MachineTranslationProcessor,
    "vicuna": VicunaProcessor,
    "alpaca": AlpacaProcessor,
    "alpaca-zephyr": AlpacaZephyrProcessor,
    "hh-rlhf": HHRLHFProcessor,
    "marx": MarxProcessor,
    "simple": SimpleProcessor,
    "zephyr": ZephyrProcessor,
    "zephyr-beaver": BeaverZephyrProcessor,
    "simple-cot": SimpleCoTProcessor,
    "pku-saferlhf-safer": SafeRLHFSafer,
    "pku-saferlhf-safer-cot": SafeRLHFSaferCoT,
    # "pku-saferlhf-safer-gen": SafeRLHFSaferForGen,
    # "pku-saferlhf-safer-cot-gen": SafeRLHFSaferCoTForGen,
    "pku-saferlhf-jh": SafeRLHFJH,
    "pku-saferlhf-jh-gen": SafeRLHFJHForGen,
}
