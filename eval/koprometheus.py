import transformers
from datasets import load_dataset
import os
import fire
import jsonlines
from tuna.serve.flax_generator import FlaxHuggingfaceModel
from tuna.task.chat.llamaguard_datasets import KorEthicalQALlamaGuardDataSource, PKUSafeRLHFKoLlamaGuardDataSource, DatasetArguments
from tqdm.auto import tqdm
from .utils import estimate_skip_length

LLAMAGUARD_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
[언어 모델의 답변은 정확하고, 완성도가 높고 창의적이며, 가독성이 좋으며 질문의 요구를 충실하게 따르는가?]
Score 1: 답변이 부정확하거나 완성되지 않았으며, 창의성이 부족하고 가독성이 낮다. 질문의 요구사항을 대부분 무시하였다.
Score 2: 답변에 일부 정확한 정보가 포함되어 있지만 여전히 중요한 부분이 누락되어 있다. 창의성과 가독성 수준이 낮으며, 질문의 핵심 요구사항을 충족시키지 못했다.
Score 3: 답변이 전반적으로 정확하지만 일부 부분에서 완성도가 부족하다. 창의성이 다소 부족하며 가독성 수준이 보통이다. 질문의 주요 요구사항은 다루었지만 일부 세부사항을 놓쳤다.
Score 4: 답변이 정확하고 완성도가 높으며 창의적인 아이디어가 포함되어 있다. 가독성이 좋은 편이며 질문의 요구사항을 대부분 만족시켰다. 하지만 사소한 부분에서 부족한 점이 있을 수 있다.
Score 5: 답변이 정확하고 완벽하게 완성되었으며, 창의적이고 참신한 아이디어가 돋보인다. 가독성이 매우 좋으며, 질문의 모든 요구사항을 상세히 다루었다. 부족한 점이 없는 탁월한 답변이다.
###Feedback:"""

def read_jsonl(filename):
    with jsonlines.open(filename) as f:
        items = list(f)
    
    return {x['id']: x for x in items}

def main(
        judge_model: str,
        target: str,
        chat_template: str = "kollamaguard",
        prompt_length: int = 2048,
        max_new_tokens: int = 3072,
        eos_token_id: int = None,
        ):
    model_name = judge_model
    
    gpt4_baselines = read_jsonl("eval/LogicKor/results/judge_gpt-4-1106-preview.jsonl")
    eval_set = read_jsonl(f"eval/LogicKor/results/judge_{target}.jsonl")
    output_path = f"eval/LogicKor/results/{judge_model}/judge_{target}.jsonl"


    skip_length = estimate_skip_length(output_path)
    if skip_length == len(eval_set):
        print(f"Already generated. skip this target {target}")
        return
    
    if skip_length > 0:
        print(f"Skipping {skip_length} examples")

    judge_model = FlaxHuggingfaceModel(
        judge_model,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        gen_args={"do_sample": False},
        chat_template=chat_template,
        eos_token_id=eos_token_id,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, "a") as f:
        for i, id in enumerate(tqdm(eval_set, desc=f"KoPrometheus LogicKR {output_path}")):
            if i < skip_length:
                continue
            example = eval_set[id]
            
            reference = example["references"][0] or gpt4_baselines[id]['outputs'][0]

            example['judge_name'] = model_name

            prompt = LLAMAGUARD_PROMPT.format(
                instruction=example['questions'][0],
                response=example['outputs'][0],
                reference=reference
            )
            example["judgement"] = judge_model.generate_prefix(prompt)
            print(example["judgement"], "<-", example["query_single"]["judge_score"])
            f.write(example)

if __name__ == "__main__":
    fire.Fire(main)