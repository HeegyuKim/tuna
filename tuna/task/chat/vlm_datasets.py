from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments, VicunaChatDataSource, convert_vicuna2openai
from datasets import load_dataset, Dataset
from copy import deepcopy
import json


BOI_TOKEN, EOI_TOKEN = "<begin_of_image>", "<end_of_image>"
# [f"<image_{i}>" for i in range(codebook_size - 2)]

@datasources("heegyu/KoLLaVA-Instruct-313k-tokenized-trl")
class KoLLaVAInstruct(ChatDataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split == "test":
            split = "validation"

        ds = load_dataset("heegyu/KoLLaVA-Instruct-313k-tokenized-trl", streaming=args.dataset_streaming, split=split)
        ds = ds.rename_column("messages", self.CONVERSATION_KEY)
        return ds


@datasources("heegyu/kollava-instruct-cosmos-di16-256")
class KoLLaVAInstructCosmosDi16(ChatDataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split == "test":
            split = "train[-100:]"
        else:
            split = "train[:-100]"

        ds = load_dataset("heegyu/kollava-instruct-cosmos-di16-256", streaming=args.dataset_streaming, split=split)
        return ds

    def map_conversations(self, item):
        img_tokens = "\n".join(["".join([f"<image_{i}>" for i in row]) for row in item["images"]])
        img_tokens = f"{BOI_TOKEN}{img_tokens}{EOI_TOKEN}"

        if "<image>" in item["questions"]:
            instructions = item["questions"].replace("<image>", img_tokens)
        else:
            instructions = img_tokens + "\n" + item["questions"]
        return {
            "conversations": [
                {"role": "user", "content": instructions},
                {"role": "assistant", "content": item["answers"]},
            ]
        }

@datasources("heegyu/llava-pretrain-cosmos-di16-256")
class LlavaPretrainCosmosDi16_256px(ChatDataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/llava-pretrain-cosmos-di16-256", streaming=args.dataset_streaming, split=split)
        return ds
        
    def map_conversations(self, item):
        img_tokens = "\n".join(["".join([f"<image_{i}>" for i in row]) for row in item["image"]])
        img_tokens = f"{BOI_TOKEN}{img_tokens}{EOI_TOKEN}"

        conversations = convert_vicuna2openai(json.loads(item["conversations"]))

        instruction = conversations[0]["content"]
        if "<image>" in instruction:
            instruction = instruction.replace("<image>", img_tokens)
        else:
            instruction = img_tokens + "\n" + instruction

        conversations[0]["content"] = instruction

        return {
            "conversations": conversations,
        }


@datasources("heegyu/llava-pretrain-titok-256px")
class LlavaPretrainTitok256px(ChatDataSource):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("heegyu/llava-pretrain-titok-256px", streaming=args.dataset_streaming, split=split)
        return ds

    def map_conversations(self, item):
        img_tokens = "".join(f"<image_{i}>" for i in item["image"])
        img_tokens = f"{BOI_TOKEN}{img_tokens}{EOI_TOKEN}"

        conversations = convert_vicuna2openai(json.loads(item["conversations"]))

        instruction = conversations[0]["content"]
        if "<image>" in instruction:
            instruction = instruction.replace("<image>", img_tokens)
        else:
            instruction = img_tokens + "\n" + instruction

        conversations[0]["content"] = instruction

        return {
            "conversations": conversations,
        }

@datasources("heegyu/clean-llama-instruct-mix-titok-256px")
class CleanLlamaInstructMixTitok256px(LlavaPretrainTitok256px):

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            split = "train[-100:]"
        else:
            split = "train[:-100]"
            
        ds = load_dataset("heegyu/clean-llama-instruct-mix-titok-256px", streaming=args.dataset_streaming, split=split)
        return ds
