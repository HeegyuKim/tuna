


eval(){
    model=$1
    cot=$2

    python -m eval.nlgbench_gen \
        --model $model \
        --model_name $model \
        --dataset "alpaca-eval,ifeval" \
        --batch_size 4 \
        --cot $cot

    # python -m eval.instruction_following_eval.evaluation_main \
    #     --input_data=eval/instruction_following_eval/data/input_data.jsonl \
    #     --input_response_data=outputs/$model/ifeval.json \
    #     --output_dir=outputs/$model/ifeval/

    # alpaca_eval --model_outputs "outputs/$model_name/alpaca-eval.json" --annotators_config chatgpt
}


# eval "heegyu/0625-zephyr7b-orpo-cot@lr5e-6-beta0.1-epoch-1" true
# eval "heegyu/0625-zephyr7b-orpo-cot@lr5e-6-beta0.1-epoch-2" true
# eval "heegyu/0625-zephyr7b-orpo-cot@lr5e-6-beta0.1-epoch-3" true


eval "heegyu/0625-zephyr7b-orpo@lr5e-6-beta0.1-epoch-1" false
eval "heegyu/0625-zephyr7b-orpo@lr5e-6-beta0.1-epoch-2" false
eval "heegyu/0625-zephyr7b-orpo@lr5e-6-beta0.1-epoch-3" false