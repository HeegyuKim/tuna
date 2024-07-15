import json

from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments, VicunaChatDataSource
from datasets import load_dataset, Dataset, concatenate_datasets
from copy import deepcopy


@datasources.register("iknow-lab/dialogsum_cluster10_0613:server")
class Dialogsum(ChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split == "test":
            split = "validation"
        if split == "train":
            split = "server"

        ds = load_dataset("iknow-lab/dialogsum_cluster10_0613", split=split)
        return ds

    def map_conversations(self, item):
        convs = [
            {
                "role": "user",
                "content": f"Summarize a conversation:\n{item['dialogue']}"
            },
            {
                "role": "assistant",
                "content": item["summary"]
            }
        ]

        return {
            "conversations": convs
        }


@datasources.register("iknow-lab/dialogsum_cluster10_0613:full")
class DialogsumMajorCluster(Dialogsum):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split == "test":
            split = "validation"
            ds = load_dataset("iknow-lab/dialogsum_cluster10_0613", split=split)

        if split == "train":
            ds1 = load_dataset("iknow-lab/dialogsum_cluster10_0613", split="server")
            ds2 = load_dataset("iknow-lab/dialogsum_cluster10_0613", split="client_train")
            ds = concatenate_datasets([ds1, ds2])
        
        return ds


@datasources.register("iknow-lab/glaive-function-calling-v2-single-cluster10-0614:server")
class GlaiveFunctionCallingServer(BaseAlpacaDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split == "test":
            split = "client_test"
        if split == "train":
            split = "server"

        ds = load_dataset("iknow-lab/glaive-function-calling-v2-single-cluster10-0614", split=split)
        return ds

@datasources.register("iknow-lab/glaive-function-calling-v2-single-cluster10-0614:full")
class GlaiveFunctionCallingFull(BaseAlpacaDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split == "test":
            split = "client_test"
            ds = load_dataset("iknow-lab/glaive-function-calling-v2-single-cluster10-0614", split=split)

        if split == "train":
            ds1 = load_dataset("iknow-lab/glaive-function-calling-v2-single-cluster10-0614", split="server")
            ds2 = load_dataset("iknow-lab/glaive-function-calling-v2-single-cluster10-0614", split="client_train")
            ds = concatenate_datasets([ds1, ds2])

        return ds
