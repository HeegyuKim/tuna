wandb online

train() {
    model=$1

    python -m tuna.launcher.train \
        --mesh fsdp \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --truncation \
        --truncation_side left \
        --model_arch causal-lm \
        --project "dfo-feedback" \
        --run_name "$model-self-feedback-0327" \
        --dataset="heegyu/ultrafeedback_binarized_feedback:self-feedback" \
        --max_length=2048 \
        --model_name_or_path $model \
        --logging_steps 1 \
        --total_epochs 3 \
        --learning_rate 2e-5 \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --use_lora \
        --output_dir ""
}

train $1 # "HuggingFaceH4/mistral-7b-sft-beta"

# train "openchat/openchat-3.5-0106"