"""Generate answers with GPT-4

Usage:
python3 gen_api_answer.py --model gpt-3.5-turbo
"""
import argparse
import json
import os
import time
import concurrent.futures

import openai
import shortuuid
import tqdm

from .common import (
    load_questions,
    temperature_config,
)
from .gen_model_answer import reorg_answer_file
from tuna.serve.flax_generator import FlaxHuggingfaceModel


def get_answer(
    question: dict, model_name: str, model, num_choices: int, max_tokens: int, answer_file: str
):
    assert (
        args.force_temperature is not None and "required_temperature" in question.keys()
    ) == False
    if args.force_temperature is not None:
        temperature = args.force_temperature
    elif "required_temperature" in question.keys():
        temperature = question["required_temperature"]
    elif question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []
    chat_state = None  # for palm-2 model
    for i in range(num_choices):
        conv = []
        turns = []
        for j in range(len(question["turns"])):
            conv.append({'role': 'user', 'content': question["turns"][j]})
            response = model.chat(conv)
            conv.append({'role': 'assistant', 'content': response})
            turns.append(response)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model_name,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, required=True, help="The model name.")
    parser.add_argument("--chat-template", type=str, required=False, default=None, help="The chat template name")
    parser.add_argument("--defense", type=str, default="")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    args = parser.parse_args()

    question_file = f"fastchat/llm_judge/data/{args.bench_name}/question.jsonl"
    questions = load_questions(question_file, args.question_begin, args.question_end)

    model_name = args.model
    model = FlaxHuggingfaceModel(
        model_name,
        prompt_length=3072,
        max_new_tokens=args.max_tokens,
        gen_args={"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95},
        chat_template=args.chat_template,
    )
    print(f"Using model {model_name}")

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"outputs/{model_name}/FastChat-MT-Bench.jsonl"
        
    print(f"Output to {answer_file}")

    for question in tqdm.tqdm(questions):
        get_answer(
            question,
            model_name,
            model,
            args.num_choices,
            args.max_tokens,
            answer_file,
        )
    reorg_answer_file(answer_file)
