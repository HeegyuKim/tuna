wandb online

dataset="hf-chat:iknow-lab/wildguardmix-train-ko-trl"
model="Qwen/Qwen2.5-0.5B-Instruct"
run_name="Qwen2.5-0.5B-wildguard-ko"

train() {
    lr=$1
    python -m tuna.launcher.train \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --model_arch causal-lm \
        --project "KoChat-SFT" \
        --train_template chatml \
        --run_name "$run_name" \
        --dataset="$dataset" \
        --packing False \
        --amp \
        --lr_warmup_steps 1024 \
        --logging_steps 1024 \
        --max_length=2048 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 2 \
        --learning_rate $lr \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 4 \
        --eval_batch_size_per_device 4 \
        --save_strategy epoch \
        --push_to_hub \
        --revision_prefix "lr$lr-bs128" \
        --output_dir ""
}

train 5e-5
train 2e-5
# train 1e-5
# train 5e-6
