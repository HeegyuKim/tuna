
from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments
from datasets import load_dataset, Dataset


@datasources.register("tatsu-lab/alpaca")
class AlpacaChat(BaseAlpacaDataSource):
    dataset_path = "tatsu-lab/alpaca"

@datasources("nvidia/OpenMathInstruct-1")
class OpenMathInstruct(BaseAlpacaDataSource):
    instruction_key = "question"
    output_key = "generated_solution"
    dataset_path = "nvidia/OpenMathInstruct-1"

    def num_items(self, split: str) -> int:
        if split == "train":
            return 5700000
        elif split == "test":
            return 1130000
        

@datasources.register("nampdn-ai/tiny-codes")
class TinyCodes(BaseAlpacaDataSource):
    instruction_key = "prompt"
    output_key = "response"
    dataset_path = "nampdn-ai/tiny-codes"
    

@datasources.register("HuggingFaceH4/ultrachat_200k")
class UltraChat(ChatDataSource):
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"{split}_sft")
        ds = ds.rename_column("messages", "conversations")
        return ds
