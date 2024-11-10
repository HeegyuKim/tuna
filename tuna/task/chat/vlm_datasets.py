from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments, VicunaChatDataSource
from datasets import load_dataset, Dataset
from copy import deepcopy


@datasources("heegyu/KoLLaVA-Instruct-313k-tokenized-trl")
class KoLLaVAInstruct(ChatDataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split == "test":
            split = "validation"

        ds = load_dataset("heegyu/KoLLaVA-Instruct-313k-tokenized-trl", streaming=args.dataset_streaming, split=split)
        ds = ds.rename_column("messages", self.CONVERSATION_KEY)
        return ds