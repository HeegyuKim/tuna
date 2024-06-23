import json

from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments, VicunaChatDataSource, convert_vicuna2openai
from datasets import load_dataset, Dataset, interleave_datasets
from copy import deepcopy


@datasources("infiniinstruct+qarv-100k")
class InifiInstruct(ChatDataSource):
    def _map_qarv(self, item):
        return {
            "conversations": [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["response"]}
            ]
        }
    
    def _map_infiniinstruct(self, item):
        convs = convert_vicuna2openai(item["conversations"])
        if convs[0]['role'] == 'assistant' and convs[0]['content'].startswith("You are"):
            convs[0]['role'] = 'system'
        return {
            "conversations": convs
        }
    
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        
        ds1 = load_dataset("BAAI/Infinity-Instruct", split=split, streaming=True) \
            .select_columns(["conversations"]) \
            .map(self._map_infiniinstruct)
        ds2 = load_dataset("HAERAE-HUB/qarv-instruct-100k", split=split).map(self._map_qarv).select_columns(["conversations"])

        return interleave_datasets([ds1, ds2.to_iterable_dataset()], probabilities=[0.9, 0.1], stopping_strategy="first_exhausted")