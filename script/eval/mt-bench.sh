cd eval/FastChat/

eval() {
    model_path=$1
    revision=$2
    
    python -m fastchat.llm_judge.gen_model_answer \
        --model-path $model_path \
        --model-id "$model_path@$revision" \
        --revision $revision \
        --chat-template zephyr
}
eval "HuggingFaceH4/zephyr-7b-beta" "main"
# eval "heegyu/0510-dpo-7b" "lr2e-4-beta0.01-epoch-1"