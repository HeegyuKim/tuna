template="zephyr"

eval(){
    model=$1
    python -m eval.nlgbench_gen \
        --model $model \
        --chat_template $template
}

# eval "alignment-handbook/zephyr-7b-sft-full"
# eval "alignment-handbook/zephyr-7b-dpo-full"
# eval "HuggingFaceH4/zephyr-7b-beta"
# eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.01@epoch-1"
# eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.1@epoch-1"
# eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.5@epoch-1"

python -m eval.nlgbench_eval_prometheus --input_files "outputs/*/*/alpaca-eval.json" --dataset alpaca-eval
python -m eval.nlgbench_eval_prometheus --input_files "outputs/*/*/mt-bench.json" --dataset mt-bench