


eval(){
    model=$1
    model_name=$2
    cot=$3

    python -m eval.nlgbench_gen \
        --model $model \
        --model_name $model_name \
        --dataset alpaca-eval \
        --eos_token "<end_of_turn>" \
        --chat_template gemma \
        --batch_size 4 \
        --cot $cot

    # python -m eval.instruction_following_eval.evaluation_main \
    #     --input_data=eval/instruction_following_eval/data/input_data.jsonl \
    #     --input_response_data=outputs/$model/ifeval.json \
    #     --output_dir=outputs/$model/ifeval/

    alpaca_eval --model_outputs "outputs/$model_name/alpaca-eval.json" --annotators_config chatgpt
}

# Phi3 LC 38.04  WR 45.82

# eval "google/gemma-1.1-2b-it" "google/gemma-1.1-2b-it" # 41/29 / 42.86

# eval "/data/checkpoint/0625-gemma2b-sft-chosen/epoch-3/" "heegyu-local/0625-gemma2b-sft-chosen@epoch-3" # 31.52 / 27.39
# eval "/data/checkpoint/0625-gemma2b-sft-cot/epoch-3/" "heegyu-local/0625-gemma2b-sft-cot@epoch-3" # 31.76 / 27.40
eval "/data/checkpoint/0625-gemma2b-orpo/epoch-3/" "heegyu-local/0625-gemma2b-orpo@epoch-3" false # 21.69 / 21.37
# eval "/data/checkpoint/0625-gemma2b-orpo-cot/epoch-3/" "heegyu-local/0625-gemma2b-orpo-cot@epoch-3" true # 35.85     33.56