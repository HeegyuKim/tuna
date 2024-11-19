
dataset="heegyu/llava-pretrain-titok-256px"
model="heegyu/Llama-3.2-1B-vis4k"
peft_model="heegyu/llama-3.2-1B-llava-titok-pretrain-1115"
peft_revision="lr5e-5-bs128-epoch-3"
wandb online

run_name="llama-3.2-1B-llava-titok-pretrain-full-1115"

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
        --amp \
        --lr_warmup_ratio 0.01 \
        --truncation \
        --model_name_or_path $model \
        --peft_model_id $peft_model \
        --peft_revision $peft_revision \
        --total_epochs 3 \
        --learning_rate $lr \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 2 \
        --eval_batch_size_per_device 2 \
        --save_strategy epoch \
        --push_to_hub \
        --revision_prefix "lr$lr-bs128" \
        --output_dir ""
}

train 5e-5
# train 1e-5
# train 5e-6