from datasets import IterableDataset, Dataset, load_dataset
from ..dataset import DataSource, datasources, DatasetArguments, NUM_PROC

ROLE_MAPPER = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
    "system": "system",
}


@datasources("hf-chat:")
class ChatDataSource(DataSource):
    CONVERSATION_KEY = "conversations"

    def __init__(self, dataset_name: str = None) -> None:
        super().__init__()
        self.dataset_name = dataset_name
    
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        ds = self.load_dataset(args, split=split)
        if ds is not None and hasattr(self, "map_conversations"):
            if isinstance(ds, IterableDataset):
                ds = ds.map(self.map_conversations)
            else:
                ds = ds.map(self.map_conversations, num_proc=NUM_PROC, load_from_cache_file=args.load_from_cache_file, desc="Converting to conversational format").select_columns([self.CONVERSATION_KEY])
        return ds
    
    # def map_conversations(self, item):
    #     return item

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        assert self.dataset_name is not None, "Please specify the dataset name"
        ds = load_dataset(self.dataset_name, streaming=args.dataset_streaming).get(split)
        if ds:
            ds = ds.select_columns(self.CONVERSATION_KEY)
        return ds

def convert_vicuna2openai(convs):
    new_convs = []
    for uttr in convs:
        new_convs.append({
            "role": ROLE_MAPPER[uttr["from"]],
            "content": uttr["value"]
        })
    return new_convs

class VicunaChatDataSource(ChatDataSource):
    CONVSERATION_KEY = "conversations"

    def map_conversations(self, item):
        return {
            "conversations": convert_vicuna2openai(item[self.CONVSERATION_KEY])
        }

class BaseAlpacaDataSource(ChatDataSource):
    
    system_key = "system"
    instruction_key = "instruction"
    input_key = "input"
    output_key = "output"
    dataset_path = None

    instruction_input_format = "{instruction}\ninput: {input}"
    has_testset: bool = False

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if not self.has_testset and split == 'test':
            return None
        
        ds = load_dataset(self.dataset_path, split=split, streaming=args.dataset_streaming)
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

