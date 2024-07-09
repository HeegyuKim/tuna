export CUDA_VISIBLE_DEVICES=1


eval(){
    model=$1
    model_name=$2
    cot=$3

    python -m eval.nlgbench_gen \
        --model $model \
        --model_name $model_name \
        --dataset alpaca-eval,ifeval \
        --batch_size 4 \
        --cot $cot

    python -m eval.instruction_following_eval.evaluation_main \
        --input_data=eval/instruction_following_eval/data/input_data.jsonl \
        --input_response_data=outputs/$model/ifeval.json \
        --output_dir=outputs/$model/ifeval/

    alpaca_eval --model_outputs "outputs/$model_name/alpaca-eval.json" --annotators_config chatgpt
}

eval "HuggingFaceH4/zephyr-7b-beta" "HuggingFaceH4/zephyr-7b-beta" false
eval "HuggingFaceH4/mistral-7b-sft-beta" "HuggingFaceH4/mistral-7b-sft-beta" false