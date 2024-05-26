
batch_size="auto:4"

# NLU: MMLU, ARC-C, TruthfulQA-MC2
# NLG: IFEval,
# Math: GSM8K, hendrycks_math(not yet)?
# Code
eval() {
    model=$1
    revision=$2

    if [ -z $revision ]; then
        revision="main"
    fi

    accelerate launch -m `lm-evaluation-harness`.lm_eval --model hf \
        --model_args pretrained=$model,revision=$revision,dtype="bfloat16" \
        --tasks openllm \
        --batch_size $batch_size \
        --device cuda:0 \
        --output_path results
        
    # lm_eval --model hf \
    #     --model_args pretrained=$model,revision=$revision,dtype="bfloat16" \
    #     --tasks arc_challenge \
    #     --device cuda:0 \
    #     --batch_size $batch_size \
    #     --output_path results \
    #     --num_fewshot 25 

    # lm_eval --model hf \
    #     --model_args pretrained=$model,revision=$revision,dtype="bfloat16" \
    #     --tasks mmlu,gsm8k \
    #     --device cuda:0 \
    #     --batch_size $batch_size \
    #     --output_path results \
    #     --num_fewshot 5 

    # lm_eval --model hf \
    #     --model_args pretrained=$model,revision=$revision,dtype="bfloat16" \
    #     --tasks truthfulqa_mc2,ifeval \
    #     --device cuda:0 \
    #     --batch_size $batch_size \
    #     --output_path results \
    #     --num_fewshot 0
}

eval "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

eval "alignment-handbook/zephyr-7b-sft-full"
eval "alignment-handbook/zephyr-7b-dpo-full"
eval "HuggingFaceH4/zephyr-7b-beta"
eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.01" "epoch-1"
eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.1" "epoch-1"
eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.5" "epoch-1"