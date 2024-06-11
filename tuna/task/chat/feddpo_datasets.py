import json

from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments, VicunaChatDataSource
from datasets import load_dataset, Dataset
from copy import deepcopy


@datasources.register("iknow-lab/dialogsum_cluster10")
class Dialogsum(ChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split == "test":
            split = "validation"

        ds = load_dataset("iknow-lab/dialogsum_cluster10", split=split)
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


@datasources.register("iknow-lab/dialogsum_cluster10_major")
class DialogsumMajorCluster(ChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split == "test":
            split = "validation"

        ds = load_dataset("iknow-lab/dialogsum_cluster10", split=split)
        minor_clusters = [0,1,2,4,5]
        ds = ds.filter(lambda x: x["cluster_id"] not in minor_clusters)
        return ds