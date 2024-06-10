import json

from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments, VicunaChatDataSource
from datasets import load_dataset, Dataset
from copy import deepcopy


# m-a-p/Code-Feedback
@datasources("m-a-p/Code-Feedback")
class CodeFeedback(ChatDataSource):
    CONVERSATION_KEY = "messages"
    def __init__(self) -> None:
        super().__init__("m-a-p/Code-Feedback")

# migtissera/Tess-Coder-v1.0
@datasources("migtissera/Tess-Coder-v1.0")
class TessCoder(ChatDataSource):
    def __init__(self) -> None:
        super().__init__("migtissera/Tess-Coder-v1.0")

    def map_conversations(self, item):
        return {
            "conversations": [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["original_answer"]},
                {"role": "user", "content": item["follow_up_question"]},
                {"role": "assistant", "content": item["claude_response"]},
                ]
        }


# glaiveai/glaive-code-assistant-v2
@datasources("glaiveai/glaive-code-assistant-v2")
class GlaiveCodeAssistant(BaseAlpacaDataSource):
    dataset_path = "glaiveai/glaive-code-assistant-v2"
    instruction_key = "question"
    output_key = "answer"

# openbmb/UltraInteract_sft
@datasources("openbmb/UltraInteract_sft")
class UltraInteractSft(BaseAlpacaDataSource):
    dataset_path = "openbmb/UltraInteract_sft"
    instruction_key = "instruction"
    output_key = "response"

    