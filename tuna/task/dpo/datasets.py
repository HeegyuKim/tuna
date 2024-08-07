from copy import deepcopy
from datasets import Dataset, load_dataset
from ..dataset import DataSource, datasources, DatasetArguments, NUM_PROC
from ..chat.datasets import convert_vicuna2openai


ROLE_MAPPER = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "system": "system"
}

class DPODataSource(DataSource):
    
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        ds = self.load_dataset(args, split=split)
        if ds is not None and hasattr(self, "map_conversations"):
            ds = ds.map(self.map_conversations, num_proc=NUM_PROC, load_from_cache_file=False, desc="Converting to conversational format").select_columns(["conversations", "chosen", "rejected"])
        return ds
    
    # def map_conversations(self, item):
    #     return item

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        pass


@datasources.register("dpo:ultrafeedback")
class UltraFeedbackDataSource(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split=split)
    
    def map_conversations(self, item):
        convs = item["chosen"][:-1]
        chosen = item["chosen"][-1]["content"]
        rejected = item["rejected"][-1]["content"]
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "rejected": rejected
        }

@datasources("dpo:openbmb/UltraInteract_pair")
class UltraInteractPair(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("openbmb/UltraInteract_pair", split=split)
    
    def map_conversations(self, item):
        convs = item["trajectory"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        return {
            "conversations": convert_vicuna2openai(convs),
            "chosen": chosen,
            "rejected": rejected
        }


@datasources("dpo:heegyu/ultrafeedback_binarized_feedback:user-feedback")
class UltraFeedbackUserFeedback(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/ultrafeedback_binarized_feedback", split=split)
        return ds

    def map_conversations(self, item):
        convs = deepcopy(item["rejected"])
        for conv in convs:
            conv["trainable"] = False

        feedback = item["feedback"]
        instruction = item["chosen"][-2]["content"]
        convs.append({
            "role": "user",
            "content": f"Your feedback and score for your response are as follows.\n[Feedback]\n{feedback}\n[Instruction]\nFollow the feedback and provide your response again:{instruction}"
        })

        return {
            "conversations": convs,
            "chosen": item["chosen"][-1]["content"],
            "rejected": item["chosen_second"][-1]["content"]
        }
    
@datasources("dpo:heegyu/ultrafeedback_binarized_feedback:self-feedback")
class UltraFeedbackSelfFeedback(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/ultrafeedback_binarized_feedback", split=split)
        return ds

    def map_conversations(self, item):
        convs = deepcopy(item["rejected"])
        for conv in convs:
            conv["trainable"] = False

        feedback = item["feedback"]
        instruction = item["chosen"][-2]["content"]
        convs.append({
            "role": "user",
            "content": "Score your previous response [1-10] and give feedback"
        })
        convs.append({
            "role": "assistant",
            "content": feedback
        })
        convs.append({
            "role": "user",
            "content": f"Follow the feedback and provide your response again:{instruction}"
        })

        return {
            "conversations": convs,
            "chosen": item["chosen"][-1]["content"],
            "rejected": item["chosen_second"][-1]["content"]
        }


@datasources.register("dpo:heegyu/UltraFeedback-feedback-tree-3")
class UltraFeedbackDataSource(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("heegyu/UltraFeedback-feedback-tree-3", split=split)


@datasources.register("dfo:heegyu/UltraFeedback-feedback-tree-3")
class UltraFeedbackDataSourceForDistil(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/UltraFeedback-feedback-tree-3", split=split)
        ds = ds.filter(lambda x: len(x['conversations']) > 1)
        return ds


@datasources("dpo:heegyu/Ultrafeedback-split-dpo-max-margin")
class UltraFeedback0430MaxMargin(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        return load_dataset("heegyu/Ultrafeedback-split-dpo-max-margin", split=split)
    
    def map_conversations(self, item):
        convs = [{'role': 'user', 'content': item['instruction']}]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "rejected": rejected
        }

@datasources("dpo:heegyu/Ultrafeedback-max-margin")
class UltraFeedbackMaxMargin(DPODataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/Ultrafeedback-max-margin-critique", split=split)
        ds = ds.filter(lambda x: x["chosen_score"] > 7 and x["rejected_score"] < 5)
        return ds
    
    def map_conversations(self, item):
        convs = [{'role': 'user', 'content': item['instruction']}]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "rejected": rejected,
        }
    

@datasources("dpo:heegyu/Ultrafeedback-max-margin-critique:cot")
class UltraFeedbackMaxMarginCritique(DPODataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/Ultrafeedback-max-margin-critique", split=split)
        ds = ds.filter(lambda x: x["chosen_score"] > 7 and x["rejected_score"] < 5)
        return ds
    
    def map_conversations(self, item):
        convs = [{'role': 'user', 'content': item['instruction']}]
        chosen = "Critique: " + item["chosen_critique"] + "\nResponse: " + item["chosen"]
        rejected = "Critique: " + item["rejected_critique"] + "\nResponse: " + item["rejected"]
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "rejected": rejected,
        }
    
@datasources("dpo:droussis/UltraSafety_binarized-orpo-dpo")
class UltraSafetyBinarizedOrpoDpo(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("droussis/UltraSafety_binarized-orpo-dpo", split=split)
    
    def map_conversations(self, item):
        convs = item["chosen"][:-1]
        chosen = item["chosen"][-1]['content']
        rejected = item["rejected"][-1]['content']
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "rejected": rejected
        }

@datasources("dpo:PKU-Alignment/PKU-SafeRLHF-30K")
class PKUSafeRLHF30kDPOSafer(DPODataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split=split)
        ds = ds.filter(lambda x: x["is_response_0_safe"] or x["is_response_1_safe"])
        return ds

    def map_conversations(self, item):
        return {
            "conversations": [
                {
                    "role": "user",
                    "content": item["prompt"]
                },
            ],
            "chosen": item["response_{0}".format(item["safer_response_id"])],
            "rejected": item["response_{0}".format(1 - item["safer_response_id"])],
        }

    
class DCODataSource(DPODataSource):

    def load(self, args: DatasetArguments, split: str) -> Dataset:
        ds = self.load_dataset(args, split=split)
        if ds is not None and hasattr(self, "map_conversations"):
            ds = ds.map(self.map_conversations, num_proc=NUM_PROC, load_from_cache_file=False, desc="Converting to conversational format").select_columns(["conversations", "chosen", "rejected", "chosen_critique", "rejected_critique"])
        return ds
    
@datasources("dco:heegyu/Ultrafeedback-max-margin-critique")
class UltraFeedbackMaxMarginDCO(DCODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("heegyu/Ultrafeedback-max-margin-critique", split=split)
    
    def map_conversations(self, item):
        convs = [{'role': 'user', 'content': item['instruction']}]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "chosen_critique": item["chosen_critique"],
            "rejected": rejected,
            "rejected_critique": item["rejected_critique"]
        }

@datasources("dco:heegyu/Ultrafeedback-split-dpo-max-margin")
class UltraFeedback0430MaxMarginDCO(DCODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        return load_dataset("heegyu/Ultrafeedback-split-dpo-max-margin", split=split)
    
    def map_conversations(self, item):
        convs = [{'role': 'user', 'content': item['instruction']}]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "chosen_critique": item["chosen_critique"],
            "rejected": rejected,
            "rejected_critique": item["rejected_critique"]
        }

@datasources("openbmb/UltraInteract_pair")
class UltraInteractPairDCOMax3(DCODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("heegyu/Ultrafeedback-split-dpo-max-margin", split=split)
        
        return ds
    
    def map_conversations(self, item):
        convs = [{'role': 'user', 'content': item['instruction']}]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "rejected": rejected,
        }

@datasources("dpo:Magpie-Align/Magpie-Pro-DPO-200K")
class MagpieProDPO200K(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        return load_dataset("Magpie-Align/Magpie-Pro-DPO-200K", split=split)
    
    def map_conversations(self, item):
        convs = item["chosen"][:-1]
        chosen = item["chosen"][-1]['content']
        rejected = item["rejected"][-1]['content']
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "rejected": rejected
        }

@datasources("dpo:heegyu/Magpie-Pro-DPO-200K-Filtered")
class MagpieProDPO200KFiltered(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        return load_dataset("heegyu/Magpie-Pro-DPO-200K-Filtered", split=split)
    
    def map_conversations(self, item):
        convs = item["chosen"][:-1]
        chosen = item["chosen"][-1]['content']
        rejected = item["rejected"][-1]['content']
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "rejected": rejected
        }

@datasources("dpo:argilla/distilabel-capybara-dpo-7k-binarized")
class DistilabelCapybaraDPO7kBinarized(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("argilla/distilabel-capybara-dpo-7k-binarized", split=split)
    
    def map_conversations(self, item):
        convs = item["chosen"][:-1]
        chosen = item["chosen"][-1]['content']
        rejected = item["rejected"][-1]['content']
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "rejected": rejected
        }

@datasources("dpo:argilla/distilabel-math-preference-dpo")
class DistilabelMathPreferenceDPO(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("argilla/distilabel-math-preference-dpo", split=split)
    
    def map_conversations(self, item):
        convs = [{"role": "user", "content": item["instruction"]}]
        chosen = item["chosen_response"]
        rejected = item["rejected_response"]
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "rejected": rejected
        }

@datasources("dpo:argilla/distilabel-intel-orca-dpo-pairs")
class DistilabelIntelOrcaDPOPairs(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split=split)
    
    def map_conversations(self, item):
        convs = [
            {"role": "user", "content": item["input"]}
            ]
        if item["system"] is not None:
            convs.insert(0, {"role": "system", "content": item["system"]})

        chosen = item["chosen"]
        rejected = item["rejected"]
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "rejected": rejected
        }

@datasources("dpo:openbmb/UltraInteract_pair")
class UltraInteractPair(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("openbmb/UltraInteract_pair", split=split)
    
    def map_conversations(self, item):
        convs = item["trajectory"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        return {
            "conversations": convert_vicuna2openai(convs),
            "chosen": chosen,
            "rejected": rejected
        }

@datasources("dpo:princeton-nlp/gemma2-ultrafeedback-armorm")
class Gemma2UltraFeedbackArmorm(DPODataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        return load_dataset("princeton-nlp/gemma2-ultrafeedback-armorm", split=split)
    
    def map_conversations(self, item):
        convs = item["chosen"][:-1]
        chosen = item["chosen"][-1]["content"]
        rejected = item["rejected"][-1]["content"]
        
        return {
            "conversations": convs,
            "chosen": chosen,
            "rejected": rejected
        }