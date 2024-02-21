from datasets import Dataset, load_dataset
from ..dataset import DataSource, datasources, DatasetArguments

class ChatDataset(DataSource):
    
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        pass

class BaseAlpacaDataset(ChatDataset):
    
    system_key = "system"
    instruction_key = "instruction"
    input_key = "input"
    output_key = "output"

    instruction_input_format = "{instruction}\ninput: {input}"

    def load(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = self.load_dataset(args, split=split)
        if ds is not None:
            print(ds)
            ds = ds.map(self.map_conversations).select_columns(["conversations"])
        return ds
    
    def map_conversations(self, item):
        convs = []
        
        if self.system_key in item:
            convs.append({
                "role": "system",
                "content": item[self.system_key]
            })
        
        if item.get(self.input_key):
            convs.append({
                "role": "user",
                "content": self.instruction_input_format.format(
                    instruction=item[self.instruction_key],
                    input=item[self.input_key]
                )
            })
        else:
            convs.append({
                "role": "user",
                "content": item[self.instruction_key]
            })

        convs.append({
            "role": "assistant",
            "content": item[self.output_key]
        })
        
        return {
            "conversations": convs
        }


@datasources.register("tatsu-lab/alpaca")
class AlpacaChat(BaseAlpacaDataset):
        
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("tatsu-lab/alpaca", split=split)

@datasources("nvidia/OpenMathInstruct-1")
class OpenMathInstruct(BaseAlpacaDataset):
    instruction_key = "question"
    output_key = "generated_solution"
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("nvidia/OpenMathInstruct-1", split=split, streaming=args.dataset_streaming)
    
    def num_items(self, split: str) -> int:
        if split == "train":
            return 5700000
        elif split == "test":
            return 1130000
        

@datasources.register("nampdn-ai/tiny-codes")
class TinyCodes(BaseAlpacaDataset):
    instruction_key = "prompt"
    output_key = "response"
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("nampdn-ai/tiny-codes", split=split)
        return ds
    

@datasources.register("HuggingFaceH4/ultrachat_200k")
class UltraChat(ChatDataset):
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"{split}_sft")
        ds = ds.rename_column("messages", "conversations")
        return ds
