
eval() {
    lr=$1
    beta=$2

    # python -m eval.nlgbench_gen \
    #     "/data/checkpoint/0710-qwen2-magpie-qarv-komath-dpo-lr$lr-beta$beta/epoch-1" \
    #     --model_name "heegyu-local/0710-qwen2-magpie-qarv-komath-dpo-lr$lr-beta$beta" \
    #     --batch_size 1 \
    #     --chat_template chatml \
    #     --dataset logickor \
    #     --prompt_length 3072 \
    #     --eos_token "<|endoftext|>" 

    python -m eval.judge_logickor \
        -o "outputs/heegyu-local/0710-qwen2-magpie-qarv-komath-dpo-lr$lr-beta$beta/logickor.jsonl"
}
for lr in 5e-6; do
    eval $lr 0.25
    eval $lr 0.1
    eval $lr 0.5
    eval $lr 1.0
done