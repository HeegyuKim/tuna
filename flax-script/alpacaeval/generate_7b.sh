

template="zephyr"

eval(){
    model=$1
    python -m eval.alpacaeval_generate \
        --model $model \
        --output_path outputs/$model/alpacaeval.json \
        --chat_template $template

    python -m eval.alpacaeval_judge \
        --input_path outputs/$model/alpacaeval.json
}


eval "heegyu/0509-1.1b-sft-dpo-beta0.01-max-margin@epoch-1"
eval "heegyu/0509-1.1b-sft-dpo-beta0.1-max-margin@epoch-1"
eval "heegyu/0509-1.1b-sft-dpo-beta0.25-max-margin@epoch-1"

# eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.01@epoch-1"
# eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.1@epoch-1"
eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.5@epoch-1"

# eval "heegyu/0507-DCO-v4-1.1b-max-margin-gamma0.1-5e-5@epoch-1"
# eval "heegyu/0509-DCO-v2-1.1b-max-margin-gamma0.1-5e-5@epoch-1"

# eval "alignment-handbook/zephyr-7b-sft-full"
# eval "alignment-handbook/zephyr-7b-dpo-full"