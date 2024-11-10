
dataset="heegyu/KoLLaVA-Instruct-313k-tokenized-trl"

wandb online

model="heegyu/llama-3.2-Korean-Bllossom-3B-vision-expanded"
run_name="llama-3.2-3B-chitto-1109"

train() {
    lr=$1
    python -m tuna.launcher.train \
        --do_train --do_eval \
        --task chat-lm \
        --padding max_length \
        --model_arch causal-lm \
        --project "KoChat-SFT" \
        --train_template llama3 \
        --run_name "$run_name" \
        --dataset="$dataset" \
        --packing False \
        --amp False \
        --use_lora \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_modules_to_save "embed_tokens,lm_head" \
        --lr_warmup_ratio 0.01 \
        --logging_steps 1024 \
        --max_length=1024 \
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

# train 1e-4
train 2e-4
train 3e-4
train 6e-4
