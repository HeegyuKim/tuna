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


@datasources("infiniinstruct500k+qarv-100k+ultrachat_200k")
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
        
        ds1 = load_dataset("arcee-ai/infini-instruct-top-500k", split=split) \
            .to_iterable_dataset() \
            .select_columns(["conversations"]) \
            .map(self._map_infiniinstruct)
        ds2 = load_dataset("HAERAE-HUB/qarv-instruct-100k", split=split).map(self._map_qarv).select_columns(["conversations"])
        ds3 = load_dataset("HuggingFaceH4/ultrachat_200k", split=split + "_sft").select_columns(["messages"]).rename_column("messages", "conversations")

        sizes = [50.0, 10.0, 20.0]
        return interleave_datasets([ds1, ds2.to_iterable_dataset(), ds3.to_iterable_dataset()], probabilities=[s/sum(sizes) for s in sizes], stopping_strategy="first_exhausted")


@datasources("0701-koen-3M")
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
        
        ds1 = load_dataset("arcee-ai/infini-instruct-top-500k", split=split) \
            .to_iterable_dataset() \
            .select_columns(["conversations"]) \
            .map(self._map_infiniinstruct)
        ds2 = load_dataset("HAERAE-HUB/qarv-instruct-100k", split=split).map(self._map_qarv).select_columns(["conversations"])
        ds3 = load_dataset("HuggingFaceH4/ultrachat_200k", split=split + "_sft").select_columns(["messages"]).rename_column("messages", "conversations")

        # 1.82M
        ds4 = load_dataset("heegyu/KoCommercial-Dataset", split=split) \
            .shuffle(seed=42) \
            .to_iterable_dataset() \
            .map(lambda x: {"conversations": [{"role": "user", "content": x["instruction"]}, {"role": "assistant", "content": x["instruction"]}]}).select_columns(["conversations"])

        datasets = [ds1, ds2.to_iterable_dataset(), ds3.to_iterable_dataset(), ds4]
        sizes = [50.0, 10.0, 20.0, 182.0]
        
        final_ds = interleave_datasets(datasets, probabilities=[s/sum(sizes) for s in sizes], stopping_strategy="first_exhausted")
        return final_ds.shuffle(buffer_size=10_000, seed=42)


        