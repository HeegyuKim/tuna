

template="phi-3"

eval(){
    model=$1
    python -m eval.nlgbench_gen \
        --model $model \
        --dataset ifeval,alpaca-eval \
        --eos_token "<|eos|>" \
        --batch_size 4 

    python -m eval.instruction_following_eval.evaluation_main \
        --input_data=eval/instruction_following_eval/data/input_data.jsonl \
        --input_response_data=outputs/$model/ifeval.json \
        --output_dir=outputs/$model/ifeval/

    alpaca_eval --model_outputs "outputs/$model/alpaca-eval.json" --annotators_config chatgpt
}


eval "microsoft/Phi-3-mini-4k-instruct" # 1.90 (LC WR) / 5.82 (WR)
# eval "heegyu/0601-phi-3b-dpo@lr5e-6-beta0.01-epoch-1"
# eval "heegyu/0601-phi-3b-dpo@lr5e-6-beta0.1-epoch-1" # 2.18 / 5.91
# eval "heegyu/0601-phi-3b-dpo@lr5e-6-beta0.25-epoch-1" # 1.99 / 5.55
# eval "heegyu/0601-phi-3b-dpo@lr5e-6-beta0.5-epoch-1" # 2.11 / 5.53

# for lr in 1e-5 5e-6; do
#     for beta in 0.1 0.25 0.5 0.75 1.0; do
#         eval "heegyu/0601-phi-3b-dpo@lr$lr-beta$beta-epoch-1"
#         eval "heegyu/0601-phi-3b-dpo-feedback-tree@lr$lr-beta$beta-epoch-1"
#         eval "heegyu/0601-phi-3b-dco@lr$lr-beta$beta-epoch-1"
#     done
# done

for lr in 1e-5 5e-6; do
    for beta in 0.1 0.5 1.0; do
        eval "heegyu/0620-phi-3b-dpo-cot@lr$lr-beta$beta-epoch-1"
        eval "heegyu/0620-phi-3b-dpo@lr$lr-beta$beta-epoch-1"
    done
done

