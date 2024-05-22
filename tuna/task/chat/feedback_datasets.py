import json

from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments, VicunaChatDataSource
from datasets import load_dataset, Dataset
from copy import deepcopy


DFO_FEEDBACK_INST_FORMAT = """
{instruction}

Answer me to get the following feedback: {feedback}
""".strip()

@datasources("dfo:kaist-ai/Feedback-Collection")
class UltraFeedbackUserFeedback(ChatDataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("prometheus-eval/Feedback-Collection", split=split)
        return ds

    def map_conversations(self, item):
        convs = []

        feedback = item["orig_feedback"]
        if "So the overall score is" in feedback:
            feedback = feedback.split("So the overall score is", 1)[0].strip()

        instruction = item["orig_response"]

        convs.append({
            "role": "user",
            "content": DFO_FEEDBACK_INST_FORMAT.format(instruction=instruction, feedback=feedback)
        })
        convs.append({
            "role": "assistant",
            "content": item["output"]
        })

        return {
            "conversations": convs
        }


FEEDBACK_REF_MODEL_FORMAT = "{instruction}\n\nYou should provide an answer that matches the given critique: {critique}"
@datasources("ccft:heegyu/Ultrafeedback-split-dpo-max-margin")
class UltraFeedbackAllCritiquesFeedback(ChatDataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("heegyu/Ultrafeedback-split-dpo-max-margin", split=split)
        items = []
        for item in ds:
            for key in ["chosen", "rejected"]:
                items.append({
                    "instruction": item["instruction"],
                    "critique": item[f"{key}_critique"],
                    "output": item[key],
                })
        return Dataset.from_list(items)

    def map_conversations(self, item):
        convs = []
        convs.append({
            "role": "user",
            "content": FEEDBACK_REF_MODEL_FORMAT.format(**item)
        })
        convs.append({
            "role": "assistant",
            "content": item["output"]
        })

        return {
            "conversations": convs
        }