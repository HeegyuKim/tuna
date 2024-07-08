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


PROMETHEUS_FORMAT = """
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
[{criteria}]
Score 1: {score1}
Score 2: {score2}
Score 3: {score3}
Score 4: {score4}
Score 5: {score5}

###Feedback: 
""".strip()
PROMETHEUS_FORMAT_OUTPUT = "{feedback} [RESULT] {score}"

@datasources("heegyu/K2-Feedback-splited")
class K2FeedbackSplited(ChatDataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("heegyu/K2-Feedback-splited", split=split)
        return ds

    def map_conversations(self, item):
        convs = []
        convs.append({
            "role": "user",
            "content": PROMETHEUS_FORMAT.format(**item)
        })
        convs.append({
            "role": "assistant",
            "content": PROMETHEUS_FORMAT_OUTPUT.format(feedback=item["feedback"], score=item["score"])
        })

        return {
            "conversations": convs
        }