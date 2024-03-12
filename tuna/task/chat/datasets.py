from datasets import Dataset, load_dataset
from ..dataset import DataSource, datasources, DatasetArguments, NUM_PROC

ROLE_MAPPER = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "system": "system"
}

class ChatDataSource(DataSource):
    
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = self.load_dataset(args, split=split)
        if ds is not None and hasattr(self, "map_conversations"):
            ds = ds.map(self.map_conversations, num_proc=NUM_PROC, load_from_cache_file=False, desc="Converting to conversational format").select_columns(["conversations"])
        return ds
    
    # def map_conversations(self, item):
    #     return item

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        pass

class VicunaChatDataSource(ChatDataSource):
    CONVSERATION_KEY = "conversations"

    def map_conversations(self, item):
        convs = []
        
        for uttr in item[self.CONVSERATION_KEY]:
            convs.append({
                "role": ROLE_MAPPER[uttr["from"]],
                "content": uttr["value"]
            })
        
        return {
            "conversations": convs
        }

class BaseAlpacaDataSource(ChatDataSource):
    
    system_key = "system"
    instruction_key = "instruction"
    input_key = "input"
    output_key = "output"
    dataset_path = None

    instruction_input_format = "{instruction}\ninput: {input}"
    
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset(self.dataset_path, split=split)
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

