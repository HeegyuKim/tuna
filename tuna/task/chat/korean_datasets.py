
from .datasets import ChatDataSource, VicunaChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments
from datasets import load_dataset, Dataset


@datasources("kyujinpy/KOR-OpenOrca-Platypus-v3")
class KOROpenOrcaPlatypusV3(BaseAlpacaDataSource):
    dataset_path = "kyujinpy/KOR-OpenOrca-Platypus-v3"
    

@datasources("squarelike/OpenOrca-gugugo-ko")
class OpenOrcaGugugoKo(BaseAlpacaDataSource):
    system_key = "system_prompt"
    instruction_key = "question"
    input_key = "input"
    output_key = "response"
    dataset_path = "squarelike/OpenOrca-gugugo-ko"

@datasources("heegyu/PKU-SafeRLHF-ko:safer")
class PKUSafeRLHFKoSafer(ChatDataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("heegyu/PKU-SafeRLHF-ko", split=split)
        # remove unsafe responses
        ds = ds.filter(lambda x: not (x["is_response_1_safe"] or x["is_response_0_safe"]))
        return ds
    
    def map_conversations(self, item):
        convs = []
        convs.append({
            "role": "user",
            "content": item["prompt_k"]
        })
        
        safer = item["safer_response_id"]
        response = item[f"response_{safer}_ko"]
        convs.append({
            "role": "assistant",
            "content": response
        })
        
        return {
            "conversations": convs
        }
    
@datasources("heegyu/PKU-SafeRLHF-ko:better")
class PKUSafeRLHFKoBetter(ChatDataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("heegyu/PKU-SafeRLHF-ko", split=split)
        return ds
    
    def map_conversations(self, item):
        convs = []
        convs.append({
            "role": "user",
            "content": item["prompt_k"]
        })
        
        safer = item["better_response_id"]
        response = item[f"response_{safer}_ko"]
        convs.append({
            "role": "assistant",
            "content": response
        })
        
        return {
            "conversations": convs
        }


@datasources("changpt/ko-lima-vicuna")
class KoLimaVicuna(ChatDataSource):
    
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("changpt/ko-lima-vicuna", split=split)
        return ds
    
@datasources("FreedomIntelligence/evol-instruct-korean")
class EvolInstructKorean(VicunaChatDataSource):
    
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("FreedomIntelligence/evol-instruct-korean", split=split)
        return ds

@datasources("heegyu/glaive-function-calling-v2-ko")
class GlaiveFunctionCallingV2Ko(ChatDataSource):
    
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/glaive-function-calling-v2-ko", split=split)
        ds = ds.select_columns(["function_description", "conversations_ko"])
        ds.rename_column("conversations_ko", "conversations")
        return ds
    
    def map_conversations(self, item):
        item = super().map_conversations(item)

        if item["function_description"]:
            item["conversations"].insert(0, {
                "role": "system",
                "content": item["system_message"] + "\n" + item["function_description"]
            })
        else:
            item["conversations"].insert(0, {
                "role": "system",
                "content": item["system_message"]
            })
        return item
    