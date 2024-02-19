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

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        pass

@datasources.register("tatsu-lab/alpaca")
class AlpacaChat(BaseAlpacaDataset):
        
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("tatsu-lab/alpaca", split=split)
