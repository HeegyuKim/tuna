wandb online

dataset="hf-chat:iknow-lab/wildguardmix-train-ko-trl"
model="Bllossom/llama-3.2-Korean-Bllossom-3B"
run_name="Bllossom-3B-WildguardMix-Ko-1024"

train() {
    lr=$1
    python -m tuna.launcher.train \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --model_arch causal-lm \
        --project "KoChat-SFT" \
        --train_template llama3 \
        --run_name "$run_name-lr$lr" \
        --dataset="$dataset" \
        --packing False \
        --amp \
        --logging_steps 1024 \
        --max_length=2048 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --revision_prefix "lr$lr-bs128" \
        --output_dir ""
}

train 1e-5
train 5e-6
train 1e-6
train 5e-7
