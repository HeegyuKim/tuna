
from .datasets import ChatDataSource, VicunaChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments
from datasets import load_dataset, Dataset
import json
from tqdm.auto import tqdm


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

@datasources("HAERAE-HUB/qarv-instruct-ko")
class QarvInstructKo(BaseAlpacaDataSource):
    instruction_key = "instruction"
    output_key = "answer"
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("HAERAE-HUB/qarv-instruct-ko", split=split)
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

# beomi/KoAlpaca-v1.1a
@datasources("beomi/KoAlpaca-v1.1a")
class KoAlpacaV1_1a(BaseAlpacaDataSource):
    dataset_path = "beomi/KoAlpaca-v1.1a"


PROMPT_QA = """당신은 시험문제 출제위원입니다. 다음 자료에 기반하여 전문가 수준의 시험문제를 출제할 것입니다. 자료를 바탕으로 지시사항에 맞는 결과물을 json 형식으로 반환해주세요.

1. 생성한 문제는 실생활에서 사용하는 질문의 말투를 사용해야 합니다(~무엇인가요? ~작성해주세요. ~ 어떻게 해야하죠?)
2. 먼저 고등학교 수준의 문제를 생성하고, 이를 전문가 수준으로 고난이도 문제로 향상해주세요. 각 문제는 반드시 제시된 자료를 바탕으로 만들어져야 합니다. 연관성이 적더라도, 창의적인 아이디어로 해당 자료를 활용하세요.
3. 문제에는 답안 작성에 필요한 내용을 주어진 자료에서 추출해서 함께 제공해야합니다.
4. 출제할 문제의 과목 후보는 다음과 같습니다: 글쓰기, 한국어, 영어, 수학, 사회과학, 과학, 역사 문화예술, 법, 도덕, 정치, 종교, 외국어, 경제, 경영, 의료, 공학, 인문학 등 - 후보에 없어도, 적절한 과목을 자유롭게 말할 수 있다.

# 제목: {title}
# 자료:
{text}
"""

PROMPT_WRITING = """당신은 글쓰기 시험문제 출제위원입니다. 다음 자료에 기반하여 전문가 수준의 시험문제를 출제할 것입니다. 자료를 바탕으로 지시사항에 맞는 결과물을 json 형식으로 반환해주세요.

1. 생성한 문제는 실생활에서 사용하는 질문의 말투를 사용해야 합니다(~무엇인가요? ~작성해주세요. ~ 어떻게 해야하죠?)
2. 먼저 고등학교 수준의 문제를 생성하고, 이를 전문가 수준으로 고난이도 문제로 향상해주세요. 각 문제는 반드시 제시된 자료를 바탕으로 만들어져야 합니다. 연관성이 적더라도, 창의적인 아이디어로 해당 자료를 활용하세요.
3. 문제에는 글쓰기 작성에 필요한 내용을 주어진 자료에서 추출해서 함께 제공해야합니다.
4. 출제할 문제의 주제 후보는 다음과 같습니다. 이 중에서 적절한 주제를 3가지 선택하세요: 이력서, 노래가사, 시 혹은 소설, 에세이, 극본, 시나리오, 여행일기, 여행계획서, 요리레시피, 해설, 자기소개서, 편지, 이메일, 리뷰 및 평가, 소셜 미디어 포스트, 일기, 청원서, 항의서, 쇼핑 리스트, 메모, 연구 논문 및 계획서, 비즈니스 보고서 및 게획서, 기술 문서, 발표자료, 계약서 혹은 법률 문서, 편집 및 출판 문서, 광고 카피라이트, 웹 콘텐츠, 뉴스레터, 연설문, 자기계발서, 분석보고서, 기획안, 제안서

# 제목: {title}
# 자료:
{text}
"""
@datasources("iknow-lab/ko-genstruct-v1")
class KoGenstructV1(ChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        
        ds = load_dataset("iknow-lab/ko-genstruct-v1", split="train")
        items = []

        for item in tqdm(ds, "Processing genstruct items"):
            prompt = PROMPT_WRITING if "namuwiki" in item["source"] else PROMPT_QA
            text = item["text"]
            if len(text) >= 5000:
                text = text[:5000]

            prompt = prompt.format(title=item["title"], text=text)
            questions = json.loads(item["questions"])

            for question in questions:
                output = f"```json\n{json.dumps(question, ensure_ascii=False, indent=2)}\n```"

                items.append({
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": output}
                    ]
                })

        return Dataset.from_list(items)

    
@datasources("sft:kuotient/orca-math-korean-preference")
class OrcaMathKoreanPreference(BaseAlpacaDataSource):
    dataset_path = "kuotient/orca-math-korean-preference"
    instruction_key = "question"
    output_key = "answer"

@datasources("sft:kuotient/orca-math-korean-preference:hard")
class OrcaMathKoreanPreferenceHard(OrcaMathKoreanPreference):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = super().load_dataset(args, split)
        if ds is not None:
            ds = ds.filter(lambda x: x["label"] == False)
        return ds

@datasources("iknow-lab/ko-genstruct-v1-output:simple")
class KoGenstructV1Output(BaseAlpacaDataSource):
    dataset_path = "iknow-lab/ko-genstruct-v1-output"
    instruction_key = "question"
    output_key = "llm_response"

@datasources("CarrotAI/ko-instruction-dataset")
class KoInstructionDataset(BaseAlpacaDataSource):
    dataset_path = "CarrotAI/ko-instruction-dataset"

@datasources("maywell/kiqu_samples")
class KiquSamples(BaseAlpacaDataSource):
    dataset_path = "maywell/kiqu_samples"

@datasources("HAERAE-HUB/K2-Feedback:score5")
class K2FeedbackScore5(BaseAlpacaDataSource):
    dataset_path = "HAERAE-HUB/K2-Feedback"
    output_key = "response"

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = super().load_dataset(args, split)
        if ds:
            ds = ds.filter(lambda x: x["score"] == 5)
        return ds

@datasources("jojo0217/korean_safe_conversation")
class KoreanSafeConversation(BaseAlpacaDataSource):
    dataset_path = "jojo0217/korean_safe_conversation"

@datasources("iknow-lab/qarv-instruct-ko-mt-deduped")
class QarvInstructKoMt(ChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("iknow-lab/qarv-instruct-ko-mt-deduped", split=split).rename_column("conversations", "messages")