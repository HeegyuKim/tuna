
dataset="heegyu/llava-pretrain-titok-256px"
model="heegyu/Llama-3.2-1B-vis4k"

wandb online

run_name="llama-3.2-1B-llava-titok-pretrain-1115"

train() {
    lr=$1
    python -m tuna.launcher.train \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --model_arch causal-lm \
        --project "Any2Any" \
        --train_template llama3 \
        --run_name "$run_name" \
        --dataset="$dataset" \
        --check_dataset False \
        --load_from_cache_file \
        --packing False \
        --train_only_response True \
        --max_length=512 \
        --amp False \
        --use_lora True \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_modules_to_save "embed_tokens,lm_head" \
        --lr_warmup_ratio 0.01 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 16 \
        --eval_batch_size_per_device 16 \
        --save_strategy epoch \
        --push_to_hub \
        --revision_prefix "lr$lr-bs128" \
        --output_dir ""
}

train 5e-5
train 1e-4
train 2e-4
train 3e-4