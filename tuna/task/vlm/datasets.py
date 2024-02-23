
from datasets import Dataset, load_dataset
from ..dataset import DataSource, datasources, DatasetArguments

class VisionChatDataset(DataSource):
    
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        pass

ROLE_MAPPER = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "system": "system"
}
@datasources("kollava-pretrain")
class KoLlavaPretrainingDataset(VisionChatDataset):

    def load(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/KoLLaVA-CC3M-Pretrain-595K", split=split)
        if args.limit:
            ds = ds.select(range(args.limit))
        ds = ds.map(self.map_conversations, num_proc=8, load_from_cache_file=False)
        return ds


    def map_conversations(self, item):
        convs = []
        
        for uttr in item["conversations"]:
            convs.append({
                "role": ROLE_MAPPER[uttr["from"]],
                "content": uttr["value"]
            })
        
        return {
            "conversations": convs
        }
    
@datasources("kollava-finetune")
class KoLlavaFinetuningDataset(KoLlavaPretrainingDataset):

    def load(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/kollava_v1_5_mix581k", split=split)
        if args.limit:
            ds = ds.select(range(args.limit))
        ds = ds.map(self.map_conversations, num_proc=8, load_from_cache_file=False)
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
            "content": item["question"]
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
                "content": item["question"] + "\n" + item["choices"]
            },
            {
                "role": "assistant",
                "content": item["answer"]
            }
        ]
        
        return {
            "conversations": convs
        }
    