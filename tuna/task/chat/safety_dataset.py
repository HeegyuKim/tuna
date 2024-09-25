from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments, VicunaChatDataSource
from datasets import load_dataset, Dataset
from copy import deepcopy

INSTRUCTION_FORMAT_REASON = """
Your task is to classify whether the following prompt includes a safe or unsafe intention.
Provide step-by-step reasoning between <reasoning> and </reasoning> tags and the final label as safe or unsafe.
The prompt is as follows:
{prompt}
""".strip()

INSTRUCTION_FORMAT_DIRECT = """
Your task is to classify whether the following prompt includes a safe or unsafe intention.
The output label should be safe or unsafe.
The prompt is as follows:
{prompt}
""".strip()

REASON_OUTPUT_FORMAT = """
<reasoning>
{prompt_label_reasoning}
</reasoning>
{prompt_label}
""".strip()

@datasources("heegyu/wildjailbreak-train-reason10k-0917:direct")
class WildjailbreakDirect(ChatDataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("heegyu/wildjailbreak-train-reason10k-0917", streaming=args.dataset_streaming).get(split)
        return ds

    def map_conversations(self, item):
        return dict(
            conversations=[
                {
                    "role": "user",
                    "content": INSTRUCTION_FORMAT_DIRECT.format(prompt=item['prompt'])
                },
                {
                    "role": "assistant",
                    "content": item['prompt_label']
                }
            ]
        )

@datasources("heegyu/wildjailbreak-train-reason10k-0917:reason")
class WildjailbreakReason(WildjailbreakDirect):
    def map_conversations(self, item):
        return dict(
            conversations=[
                {
                    "role": "user",
                    "content": INSTRUCTION_FORMAT_REASON.format(prompt=item['prompt'])
                },
                {
                    "role": "assistant",
                    "content": REASON_OUTPUT_FORMAT.format(
                        prompt_label_reasoning=item['prompt_label_reasoning'],
                        prompt_label=item['prompt_label']
                    )
                }
            ]
        )


if __name__ == "__main__":
    pass