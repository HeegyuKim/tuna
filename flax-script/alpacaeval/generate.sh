# export HF_HOME=/data-plm/hf-home
# export LD_LIBRARY_PATH=~/anaconda3/envs/easydel/lib/:$LD_LIBRARY_PATH

eval(){
    model=$1
    template=$2
    python -m eval.alpacaeval_generate \
        --model $model \
        --output_path outputs/$model/alpacaeval.json \
        --chat_template $template

    python -m eval.alpacaeval_judge \
        --input_path outputs/$model/alpacaeval.json
}

## BASELINES (처음에 망함))
# eval heegyu/TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T-tinyllama-1.1b-sft@steps-51966 zephyr
# eval heegyu/TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T-tinyllama-1.1b-sft@steps-103932 zephyr
eval heegyu/TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T-tinyllama-1.1b-sft@steps-155897 zephyr

# eval "heegyu/TinyLlama-1.1b-max-margin@epoch-1" "zephyr"
# eval "heegyu/TinyLlama-1.1b-max-margin@epoch-2" "zephyr"
# eval "heegyu/TinyLlama-1.1b-max-margin@epoch-3" "zephyr"

# eval "heegyu/TinyLlama-1.1b-feedback-tree-3-0422@epoch-1" "zephyr"
# eval "heegyu/TinyLlama-1.1b-feedback-tree-3-0422@epoch-2" "zephyr"
# eval "heegyu/TinyLlama-1.1b-feedback-tree-3-0422@epoch-3" "zephyr"

# eval "heegyu/TinyLlama-1.1b-feedback-all@epoch-1" zephyr
# eval "heegyu/TinyLlama-1.1b-feedback-all@epoch-2" zephyr
# eval "heegyu/TinyLlama-1.1b-feedback-all@epoch-3" zephyr

# 얘네 망함
# eval "heegyu/TinyLlama-1.1b-max-margin-0424-orpo@epoch-1" zephyr
# eval "heegyu/TinyLlama-1.1b-max-margin-0424-orpo@epoch-2" zephyr
# eval "heegyu/TinyLlama-1.1b-max-margin-0424-orpo@epoch-3" zephyr
# eval "heegyu/TinyLlama-1.1b-feedback-tree-3-epoch3-distil@epoch-1" zephyr
# eval "heegyu/TinyLlama-1.1b-feedback-tree-3-epoch3-distil@epoch-2" zephyr
# eval "heegyu/TinyLlama-1.1b-feedback-tree-3-epoch3-distil@epoch-3" zephyr


# 0429 learning rate 조절실험
# eval "heegyu/TinyLlama-1.1b-max-margin-0429-2e-5@epoch-1" zephyr
# eval "heegyu/TinyLlama-1.1b-max-margin-0429-2e-5@epoch-2" zephyr
# eval "heegyu/TinyLlama-1.1b-max-margin-0429-2e-5@epoch-3" zephyr

# eval "heegyu/TinyLlama-1.1b-max-margin-0429-1e-5@epoch-1" zephyr
# eval "heegyu/TinyLlama-1.1b-max-margin-0429-1e-5@epoch-2" zephyr
# eval "heegyu/TinyLlama-1.1b-max-margin-0429-1e-5@epoch-3" zephyr

# eval "heegyu/TinyLlama-1.1b-max-margin-0429-5e-6@epoch-1" zephyr
# eval "heegyu/TinyLlama-1.1b-max-margin-0429-5e-6@epoch-2" zephyr
# eval "heegyu/TinyLlama-1.1b-max-margin-0429-5e-6@epoch-3" zephyr

# eval "heegyu/TinyLlama-1.1b-max-margin-0427-orpo@epoch-1" zephyr
# eval "heegyu/TinyLlama-1.1b-max-margin-0427-orpo@epoch-2" zephyr
# eval "heegyu/TinyLlama-1.1b-max-margin-0427-orpo@epoch-3" zephyr