

wandb online
model="HuggingFaceH4/mistral-7b-sft-beta"

train() {
    dataset=$1
    name=$2
    python -m tuna.launcher.train \
        --mesh fsdp \
        --do_train \
        --task dpo \
        --truncation \
        --truncation_side left \
        --padding max_length \
        --model_arch causal-lm \
        --project "feedback-tree" \
        --run_name "$name" \
        --dataset="$dataset" \
        --use_lora \
        --max_length=2048 \
        --model_name_or_path $model \
        --logging_steps 1 \
        --total_epochs 3 \
        --learning_rate 1e-5 \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --output_dir ""
}

# train "dpo:heegyu/UltraFeedback-max-margin" "mistral-7b-sft-max-margin-0418"
train "dpo:heegyu/UltraFeedback-feedback-tree-3" "mistral-7b-sft-max-margin-0418"
