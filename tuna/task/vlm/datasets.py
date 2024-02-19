
from datasets import Dataset, load_dataset
from ..dataset import DataSource, datasources, DatasetArguments

class VisionChatDataset(DataSource):
    
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        pass

@datasources("kollava-pretrain")
class KoLlavaPretrainDataset(VisionChatDataset):

    def load(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/KoLLaVA-CC3M-Pretrain-595K", split=split)
        
        return ds


@datasources("textvqa")
class TextVQA(VisionChatDataset):

    def load(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("textvqa", split=split)
        if ds is not None:
            ds = ds.map(self.map_conversations, num_proc=8, load_from_cache_file=False)
        return ds
    
    def map_conversations(self, item):
        convs = []
    
        convs.append({
            "role": "user",
            "content": "<image>\n" + item["question"]
        })

        convs.append({
            "role": "assistant",
            "content": item["answers"][0]
        })
        
        return {
            "conversations": convs
        }
    

@datasources("BilbaoQA")
class TextVQA(VisionChatDataset):
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("TheMrguiller/BilbaoQA", split=split)
        if ds is not None:
            ds = ds.map(self.map_conversations, num_proc=8, load_from_cache_file=False)
        return ds
    
    def map_conversations(self, item):
        convs = [
            {
                "role": "user",
                "content": "<image>\n" + item["question"] + "\n" + item["choices"]
            },
            {
                "role": "assistant",
                "content": item["answer"]
            }
        ]
        
        return {
            "conversations": convs
        }
    