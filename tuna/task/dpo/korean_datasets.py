from .datasets import datasources, DPODataSource, DatasetArguments
from datasets import Dataset, load_dataset



@datasources.register("dpo:SJ-Donald/orca-dpo-pairs-ko")
class UltraFeedbackDataSource(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("SJ-Donald/orca-dpo-pairs-ko", split=split)
    
    def map_conversations(self, item):
        convs = []

        if item["system"]:
            convs.append({
                'role': 'system',
                'content': item['system']
            })
        convs.append({
            'role': 'user',
            'content': item['question']
        })
        
        return {
            "conversations": convs,
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        }

@datasources("dpo:heegyu/orca_dpo_pairs_ko")
class UltraFeedbackDataSource(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("heegyu/orca_dpo_pairs_ko", split=split)
    
    def map_conversations(self, item):
        convs = []

        if item["system"]:
            convs.append({
                'role': 'system',
                'content': item['system']
            })

        convs.append({
            'role': 'user',
            'content': item['question']
        })
        
        return {
            "conversations": convs,
            "chosen": item["gpt3_response"],
            "rejected": item["eeve_response"]
        }

# iknow-lab/ultrafeedback_ko_20k
@datasources("dpo:iknow-lab/ultrafeedback_ko_20k")
class UltraFeedbackDataSource(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("iknow-lab/ultrafeedback_ko_20k", split=split)
    
    def map_conversations(self, item):
        convs = []

        convs.append({
            'role': 'user',
            'content': item['instruction_ko']
        })
        
        return {
            "conversations": convs,
            "chosen": item["gpt3_response"],
            "rejected": item["eeve_response"]
        }