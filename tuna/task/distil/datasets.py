from copy import deepcopy
from datasets import Dataset, load_dataset
from ..dataset import DataSource, datasources, DatasetArguments, NUM_PROC
from ..chat.datasets import ROLE_MAPPER
from ..dpo.datasets import DPODataSource



@datasources.register("distil:openbmb/UltraInteract_pair")
class UltraInteract_pair(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("openbmb/UltraInteract_pair", split=split)
        ds = ds.filter(lambda x: len(x["trajectory"]) > 1)
        return ds
    
    def map_conversations(self, item):
        convs = []
        for uttr in item["trajectory"]:
            convs.append({
                "role": ROLE_MAPPER[uttr["from"]],
                "content": uttr["value"]
            })
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "rejected": rejected
        }