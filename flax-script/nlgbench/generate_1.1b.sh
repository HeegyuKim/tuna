

template="zephyr"

eval(){
    model=$1
    python -m eval.nlgbench_gen \
        --model $model \
        --chat_template $template
}

eval heegyu/TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T-tinyllama-1.1b-sft@steps-51966
eval heegyu/TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T-tinyllama-1.1b-sft@steps-103932
eval heegyu/TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T-tinyllama-1.1b-sft@steps-155897 

eval "heegyu/0509-1.1b-sft-dpo-beta0.01-max-margin@epoch-1"
eval "heegyu/0509-1.1b-sft-dpo-beta0.1-max-margin@epoch-1"
eval "heegyu/0509-1.1b-sft-dpo-beta0.25-max-margin@epoch-1"
eval "heegyu/0509-1.1b-sft-dpo-beta0.5-max-margin@epoch-1"

eval "heegyu/0507-DCO-v4-1.1b-max-margin-gamma0.1-5e-5@epoch-1"
eval "heegyu/0509-DCO-v2-1.1b-max-margin-gamma0.1-5e-5@epoch-1"
