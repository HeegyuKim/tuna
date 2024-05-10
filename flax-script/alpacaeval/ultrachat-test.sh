# export HF_HOME=/data-plm/hf-home
# export LD_LIBRARY_PATH=~/anaconda3/envs/easydel/lib/:$LD_LIBRARY_PATH

eval(){
    model=$1
    template=$2

    if [ -z "$template" ]; then
        template="zephyr"
        echo "Using default template: $template"
    fi

    python -m eval.ccft_generate \
        --model $model \
        --output_path outputs/$model/ultrachat-test.json \
        --ccft False \
        --chat_template $template

    python -m eval.ccft_judge \
        --input_path outputs/$model/ultrachat-test.json
}

# eval "heegyu/0430-DFO-1.1b-DPO@epoch-1"
eval "heegyu/TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T-tinyllama-1.1b-sft@steps-155897"