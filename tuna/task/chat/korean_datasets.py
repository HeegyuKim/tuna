
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

@datasources("heegyu/OpenOrca-gugugo-ko-len500")
class OpenOrcaGugugoKoLen500(OpenOrcaGugugoKo):
    dataset_path = "heegyu/OpenOrca-gugugo-ko-len500"

@datasources("heegyu/CoT-collection-ko")
class Cot_Collection_Ko(ChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != 'train':
            return None
        
        ds = load_dataset("heegyu/CoT-collection-ko", split=split)
        return ds
    
    def map_conversations(self, item):
        convs = []
        convs.append({
            "role": "user",
            "content": item["source"]
        })
        response = item[f"rationale"] + "\n#### " + item["target"]
        convs.append({
            "role": "assistant",
            "content": response
        })
        return {
            "conversations": convs
        }

@datasources("dbdu/ShareGPT-74k-ko")
class KoVast(VicunaChatDataSource):
    
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("dbdu/ShareGPT-74k-ko", split=split, streaming=args.dataset_streaming)
        return ds
    
@datasources("MarkrAI/KoCommercial-Dataset")
class KoCommercialDataset(BaseAlpacaDataSource):
    dataset_path = "MarkrAI/KoCommercial-Dataset"

@datasources("heegyu/KoCommercial-Dataset")
class KoCommercialDatasetHeegyu(BaseAlpacaDataSource):
    dataset_path = "heegyu/KoCommercial-Dataset"

@datasources('kuotient/gsm8k-ko')
class GSM8k_Ko(BaseAlpacaDataSource):

    instruction_key = "question"
    output_key = "answer"
    dataset_path = "kuotient/gsm8k-ko"

    
@datasources("heegyu/HRC")
class HRC(BaseAlpacaDataSource):
    dataset_path = "heegyu/HRC"
    instruction_key = "질의"
    output_key = "상담례, 결정례 기반 답변"

@datasources("MrBananaHuman/kor_ethical_question_answer")
class kor_ethical_question_answer(BaseAlpacaDataSource):
    instruction_key = "question"
    output_key = "answer"
    dataset_path = "MrBananaHuman/kor_ethical_question_answer"


@datasources("heegyu/kor_counselgpt_multiturn")
class KorCounselGPTMultiturn(VicunaChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/kor_counselgpt_multiturn", split=split, streaming=args.dataset_streaming)
        return ds


@datasources("maywell/koVast")
class KoVast(VicunaChatDataSource):
    
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("maywell/koVast", split=split, streaming=args.dataset_streaming)
        return ds
    
    
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
class KoLimaVicuna(VicunaChatDataSource):
    
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
        # ds = ds.filter(lambda x: x[''])
        return ds

    # def filter_item(self, item):
    #     # find gpt output
    #     has_response = False
    #     for uttr in item['conversations']:
    #         if uttr['from'] == 'gpt' and len(uttr['value'].strip()):
    #             has_response

@datasources("heegyu/glaive-function-calling-v2-ko")
class GlaiveFunctionCallingV2Ko(ChatDataSource):
    
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/glaive-function-calling-v2-ko", split=split)
        ds = ds.select_columns(["function_description", "conversations_ko"])
        ds = ds.rename_column("conversations_ko", "conversations")
        return ds
    
    def map_conversations(self, item):
        if item["function_description"]:
            item["conversations"].insert(0, {
                "role": "system",
                "content": "당신은 사용자에게 도움이 되는 AI 어시스턴트입니다. 필요하다면 다음 함수를 사용하세요\n" + item["function_description"]
            })
        else:
            item["conversations"].insert(0, {
                "role": "system",
                "content": "당신은 사용자에게 도움이 되는 AI 어시스턴트입니다. 사용할 수 있는 함수는 없습니다."
            })
        return item

@datasources("HAERAE-HUB/qarv-instruct-100k")
class QarvInstructKo(BaseAlpacaDataSource):
    instruction_key = "instruction"
    output_key = "response"
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("HAERAE-HUB/qarv-instruct-100k", split=split)
        return ds

@datasources("squarelike/sharegpt_deepl_ko_translation")
class ShareGPTDeepLKoTranslation(ChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        train_files = [
            "translation_data_sharegpt_long_v7_newlineClean.json",
            "translation_data_sharegpt_v19_newlineClean.json"   
        ]
        ds = load_dataset("squarelike/sharegpt_deepl_ko_translation", data_files={"train": train_files})['train']
        return ds

    def map_conversations(self, item):
        convs = []
        convs.append({
            "role": "system",
            "content": "You are a Korean translator. Translate given English text into Korean."
        })
        convs.append({
            "role": "user",
            "content": item["english"]
        })
        convs.append({
            "role": "assistant",
            "content": item["korean"]
        })
        return {
            "conversations": convs
        }