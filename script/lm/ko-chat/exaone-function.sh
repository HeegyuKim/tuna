
wandb online

dataset="hf-chat:iknow-lab/hermes-function-calling-v1-ko-cleaned"
model="heegyu/EXAONE-3.0-7.8B-Instruct-bf16"
run_name="EXAONE-7.8B-function-1025"

train() {
    lr=$1
    python -m tuna.launcher.train \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --model_arch causal-lm \
        --project "KoChat-SFT" \
        --train_template exaone \
        --trust_remote_code \
        --run_name "$run_name-lr$lr" \
        --dataset="$dataset" \
        --packing False \
        --amp False \
        --use_lora \
        --max_length=2048 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 4 \
        --eval_batch_size_per_device 4 \
        --save_strategy epoch \
        --push_to_hub \
        --revision_prefix "lr$lr-bs128" \
        --output_dir ""
}

train 1e-4
train 2e-4
