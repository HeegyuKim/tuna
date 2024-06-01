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

# eval "heegyu/0510-dpo-7b@lr2e-4-beta0.01-epoch-1"
# eval "heegyu/0510-dpo-7b@lr3e-4-beta0.01-epoch-1"
# eval "heegyu/0510-dco-v2-7b@lr2e-4-beta0.01-epoch-1"
# eval "heegyu/0510-dco-v4-7b@lr2e-4-beta0.01-epoch-1"

# eval "heegyu/0510-orpo-7b@lr1e-5-beta0.1-epoch-3"
# eval "heegyu/0510-orpo-7b@lr5e-6-beta0.1-epoch-3"

# eval "heegyu/0513-dpo-7b@lr1e-4-beta0.01-epoch-1"
# eval "heegyu/0513-dpo-7b@lr2e-4-beta0.01-epoch-1"
# eval "heegyu/0513-dpo-7b@lr3e-4-beta0.01-epoch-1"
# eval "heegyu/0513-dco-7b@lr1e-4-beta0.01-epoch-1"
# eval "heegyu/0513-dco-v1d-7b@lr1e-4-beta0.01-epoch-1"
# eval "heegyu/0513-dco-v4-7b@lr1e-4-beta0.01-epoch-1"
# eval "heegyu/0513-dco-v4d-7b@lr1e-4-beta0.01-epoch-1"

eval "heegyu/0513-dco-7b@lr2e-4-beta0.01-epoch-1"
eval "heegyu/0513-dco-v1d-7b@lr2e-4-beta0.01-epoch-1"
eval "heegyu/0513-dco-v4-7b@lr2e-4-beta0.01-epoch-1"
eval "heegyu/0513-dco-v4d-7b@lr2e-4-beta0.01-epoch-1"


python -m eval.nlgbench_eval_prometheus --input_files "outputs/*/*/alpaca-eval.json" --dataset alpaca-eval
# python -m eval.nlgbench_eval_prometheus --input_files "outputs/*/*/mt-bench.json" --dataset mt-bench