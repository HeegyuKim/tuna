


eval(){
    model=$1
    model_name=$2

    python -m eval.nlgbench_gen \
        --model $model \
        --model_name $model_name \
        --dataset alpaca-eval \
        --eos_token "<end_of_turn>" \
        --chat_template gemma \
        --batch_size 4 \
        --cot

    # python -m eval.instruction_following_eval.evaluation_main \
    #     --input_data=eval/instruction_following_eval/data/input_data.jsonl \
    #     --input_response_data=outputs/$model/ifeval.json \
    #     --output_dir=outputs/$model/ifeval/

    # alpaca_eval --model_outputs "outputs/$model/alpaca-eval.json" --annotators_config chatgpt
}


 eval "/data/checkpoint/0625-phi-3b-sft-cot/" "heegyu-local/0625-phi-3b-sft-cot"