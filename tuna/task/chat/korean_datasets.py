
from .datasets import ChatDataset, BaseAlpacaDataset, datasources, DatasetArguments
from datasets import load_dataset, Dataset


@datasources("kyujinpy/KOR-OpenOrca-Platypus-v3")
class KOROpenOrcaPlatypusV3(BaseAlpacaDataset):
    
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("kyujinpy/KOR-OpenOrca-Platypus-v3", split=split)
        return ds
    

@datasources("heegyu/glaive-function-calling-v2-ko")
class GlaiveFunctionCallingV2Ko(ChatDataset):
    
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/glaive-function-calling-v2-ko", split=split)
        ds = ds.select_columns(["function_description", "conversations_ko"])
        ds.rename_column("conversations_ko", "conversations")
        return ds