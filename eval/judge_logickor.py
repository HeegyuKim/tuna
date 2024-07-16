from typing import Dict, Union
import argparse
import re
import json
import time
import os
from datetime import datetime
from threading import Lock

import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor


time_start = datetime.now().strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--model-output', help=' : Model Output File Location', default=None)
parser.add_argument('-k', '--openai-api-key', help=' : Model', default=os.environ.get("OPENAI_API_KEY"))
parser.add_argument('-j', '--judge-model', help=' : Judge Model', default='gpt-4-1106-preview')
parser.add_argument('-t', '--threads', help=' : Thread count', default=10, type=int)
args = parser.parse_args()

if args.model_output is None:
    raise ValueError('Model Output File Location is required')
if args.openai_api_key is None:
    raise ValueError('OpenAI API Key is required')

client = OpenAI(
  api_key=args.openai_api_key
)

df_model_outputs = pd.read_json(args.model_output, lines=True)
df_judge_template = pd.read_json('eval/LogicKor/judge_template.jsonl', lines=True)
judge_file = os.path.join(os.path.splitext(args.model_output)[0], f"{args.judge_model}.jsonl")

if os.path.exists(judge_file):
    print(f"File {judge_file} already exists. Please move or delete it before running this script.")
    exit(1)
else:
    os.makedirs(os.path.dirname(judge_file), exist_ok=True)

lock = Lock()


def create_answers(
    model_output, is_multi_turn: bool = False
) -> Dict[str, Union[str, float]]:
    # Construct prompt from model output
    model_questions = model_output['questions']
    model_outputs = model_output['outputs']
    model_references = model_output['references']
    
    prompt = f"**질문**\n{model_questions[0]}\n\n**모델 답변**\n{model_outputs[0]}"

    if model_references and model_references[0]:
        prompt += f"\n\n**Ground Truth**\n{model_references[0]}"

    if is_multi_turn:
        prompt += f"\n\n**이어지는 질문**\n{model_questions[1]}\n\n**모델 답변**\n{model_outputs[1]}"
        if model_references and model_references[1]:
            prompt += f"\n\n**Ground Truth**\n{model_references[1]}"
    
    prompt += "\n\n[[대화 종료. 평가 시작.]]"

    try:
        response = client.chat.completions.create(
            model=args.judge_model,
            temperature=0,
            n=1,
            messages=[
                {"role": "system", "content": df_judge_template.iloc[1 if is_multi_turn else 0]['system_prompt']},
                {"role": "user", "content": prompt}
            ]
        )
        # Extract judge message and score using regular expressions
        content = response.choices[0].message.content
        judge_message_match = re.search(r"평가:(.*?)점수:", content, re.DOTALL)
        judge_message = judge_message_match.group(1).strip() if judge_message_match else "No judge message found"
        
        judge_score_match = re.search(r"점수:\s*(\d+(\.\d+)?)", content)
        if judge_score_match:
            judge_score = float(judge_score_match.group(1))
        else:
            raise ValueError("No score found in response")
        
        return {
            'judge_message': judge_message,
            'judge_score': judge_score
        }

    except Exception as e:
        print("Error. Retrying after 10 sec", e)
        time.sleep(10)
        return create_answers(model_output, is_multi_turn)

def process_item(_, row):
    row = row[1]
    
    query_single = create_answers(row)
    query_multi = create_answers(row, is_multi_turn=True)

    row['query_single'] = query_single
    row['query_multi'] = query_multi
    row = row.to_dict()

    
    with lock:
        with open(judge_file, 'a', encoding='utf-8-sig') as f:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write('\n')

with ThreadPoolExecutor(max_workers=int(args.threads)) as executor:
    list(executor.map(
        process_item,
        df_model_outputs.index,
        df_model_outputs.iterrows()
    ))


# 카테고리별 점수 집계를 위한 딕셔너리
category_scores = {}

# 전체 싱글 점수와 멀티 점수의 리스트
total_single_scores = []
total_multi_scores = []

with open(judge_file, 'r', encoding='utf-8-sig') as file:  # 'utf-8-sig'로 인코딩 변경
    for line in file:
        item = json.loads(line)
        category = item['category']
        single_score = item['query_single']['judge_score']
        multi_score = item['query_multi']['judge_score']

        if category not in category_scores:
            category_scores[category] = {'single_scores': [], 'multi_scores': []}

        category_scores[category]['single_scores'].append(single_score)
        category_scores[category]['multi_scores'].append(multi_score)

        # 전체 점수 리스트에 추가
        total_single_scores.append(single_score)
        total_multi_scores.append(multi_score)

# 카테고리별 평균 점수 계산
for category, scores in category_scores.items():
    avg_single = sum(scores['single_scores']) / len(scores['single_scores'])
    avg_multi = sum(scores['multi_scores']) / len(scores['multi_scores'])
    print(f"카테고리: {category}, 싱글 점수 평균: {avg_single:.2f}, 멀티 점수 평균: {avg_multi:.2f}")

# 전체 점수의 평균 계산 및 출력
avg_total_single = sum(total_single_scores) / len(total_single_scores)
avg_total_multi = sum(total_multi_scores) / len(total_multi_scores)
avg_total = (avg_total_single + avg_total_multi) / 2
print(f"전체 싱글 점수 평균: {avg_total_single:.2f}")
print(f"전체 멀티 점수 평균: {avg_total_multi:.2f}")
print(f"전체 점수: {avg_total:.2f}")
