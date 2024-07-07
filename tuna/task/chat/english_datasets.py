import json

from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments, VicunaChatDataSource
from datasets import load_dataset, Dataset
from copy import deepcopy


@datasources.register("tatsu-lab/alpaca")
class AlpacaChat(BaseAlpacaDataSource):
    dataset_path = "tatsu-lab/alpaca"

@datasources.register("GAIR/lima")
class Lima(ChatDataSource):
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("GAIR/lima", split=split, streaming=args.dataset_streaming)
        ds = ds.map(self._map_conv, load_from_cache_file=False, desc="Converting a GAIR/lima dataset")
        
        return ds

    def _map_conv(self, item):
        convs = []

        for i, conv in enumerate(item["conversations"]):
            convs.append(dict(
                role="user" if i % 2 == 0 else "assistant",
                content=conv
            ))

        return {
            "conversations": convs
        }

@datasources("nvidia/OpenMathInstruct-1")
class OpenMathInstruct(BaseAlpacaDataSource):
    instruction_key = "question"
    output_key = "generated_solution"
    dataset_path = "nvidia/OpenMathInstruct-1"

    def num_items(self, split: str) -> int:
        if split == "train":
            return 5700000
        elif split == "test":
            return 1130000
        

@datasources.register("nampdn-ai/tiny-codes")
class TinyCodes(BaseAlpacaDataSource):
    instruction_key = "prompt"
    output_key = "response"
    dataset_path = "nampdn-ai/tiny-codes"

@datasources("glaiveai/glaive-code-assistant-v2")
class GlaiveCodeAssistantV2(BaseAlpacaDataSource):
    instruction_key = "question"
    output_key = "answer"
    dataset_path = "glaiveai/glaive-code-assistant-v2"
    

@datasources.register("HuggingFaceH4/ultrachat_200k")
class UltraChat(ChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"{split}_sft")
        ds = ds.rename_column("messages", "conversations").select_columns(["conversations"])
        return ds

@datasources("Open-Orca/SlimOrca-Dedup")
class SlimOrcaDedup(VicunaChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("Open-Orca/SlimOrca-Dedup", split=split)
        return ds


@datasources("heegyu/ultrafeedback_binarized_feedback:user-feedback")
class UltraFeedbackUserFeedback(ChatDataSource):

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
        convs.append(item["chosen"][-1])

        return {
            "conversations": convs
        }
    
@datasources("heegyu/ultrafeedback_binarized_feedback:self-feedback")
class UltraFeedbackSelfFeedback(ChatDataSource):

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
        convs.append(item["chosen"][-1])

        return {
            "conversations": convs
        }

@datasources.register("thu-coai/esconv")
class ESConv(ChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("thu-coai/esconv", split=f"{split}")
        return ds

    def map_conversations(self, item):
        item = json.loads(item["text"])
        dialog = item["dialog"]
        convs = []
        speaker2role = {
            'sys': "assistant",
            'usr': 'user'
        }

        for i in range(1, len(dialog) - 1):
            uttr = dialog[i]
            speaker = uttr["speaker"]

            if speaker == "sys":
                text = "[{strategy}] {text}".format(**uttr)
            else:
                text = uttr["text"]

            convs.append({
                "role": speaker2role[speaker],
                "content": text,
            })

        return {
            "conversations": convs
        }


@datasources("sft:droussis/UltraSafety_binarized-orpo-dpo")
class UltraSafetyBinarizedSFT(ChatDataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        return load_dataset("droussis/UltraSafety_binarized-orpo-dpo", split=split)
    
    def map_conversations(self, item):
        return {
            "conversations": item["chosen"]
        }

@datasources("sft:iknow-lab/PKU-SafeRLHF-30K-safe-safer")
class PKUSafeRLHF30kSFTSafer(ChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("iknow-lab/PKU-SafeRLHF-30K-safe-safer", split=split)
        return ds

    def map_conversations(self, item):
        return {
            "conversations": [
                {
                    "role": "user",
                    "content": item["instruction"]
                },
                {
                    "role": "assistant",
                    "content": item["output"]
                }
            ]
        }


@datasources("sft:heegyu/Ultrafeedback-max-margin-critique:cot")
class UltraFeedbackMaxMarginCritique(ChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/Ultrafeedback-max-margin-critique", split=split)
        ds = ds.filter(lambda x: x["chosen_score"] > 7 and x["rejected_score"] < 5)
        return ds
    
    def map_conversations(self, item):
        chosen = "Response Guide: " + item["chosen_critique"] + "\n\nAssistant Response:" + item["chosen"]
        convs = [
            {'role': 'user', 'content': item['instruction']},
            {'role': 'assistant', 'content': chosen}
            ]
        return {
            "conversations": convs,
        }
    
@datasources("sft:heegyu/Ultrafeedback-max-margin-critique:chosen")
class UltraFeedbackMaxMarginChosen(ChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/Ultrafeedback-max-margin-critique", split=split)
        ds = ds.filter(lambda x: x["chosen_score"] > 7 and x["rejected_score"] < 5)
        return ds
    
    def map_conversations(self, item):
        chosen = item["chosen"]
        convs = [
            {'role': 'user', 'content': item['instruction']},
            {'role': 'assistant', 'content': chosen}
            ]
        return {
            "conversations": convs,
        }
    
@datasources("Magpie-Align/Magpie-Air-300K-Filtered")
class MagpieAir300KFiltered(VicunaChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return
        ds = load_dataset("Magpie-Align/Magpie-Air-300K-Filtered", split=split)
        return ds

@datasources("Magpie-Align/Magpie-Pro-MT-300K-v0.1")
class MagpieProMT300K(VicunaChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return
        ds = load_dataset("Magpie-Align/Magpie-Pro-MT-300K-v0.1", split=split)
        return ds