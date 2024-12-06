
dataset="heegyu/infinity-mm-stage1-cosmosdi16-256px-rev"
model="Qwen/Qwen2.5-0.5B-Instruct"
run_name="Qwen2.5-0.5B-Instruct-cosmosdi16-stage1"

wandb online
export DATASET_NUM_PROC=32

train() {
    lr=$1
    python -m tuna.launcher.train \
        --do_train \
        --mesh dp \
        --task chat-lm \
        --padding max_length \
        --model_arch causal-lm-vision-expansion-stage1 \
        --project "Any2Any" \
        --train_template chatml \
        --num_visual_tokens 65539 \
        --run_name "$run_name" \
        --revision_prefix "lr$lr-bs128-1206" \
        --dataset="$dataset" \
        --dataset_streaming \
        --check_dataset False \
        --load_from_cache_file \
        --packing False \
        --train_only_response True \
        --max_length=512 \
        --amp True \
        --lr_warmup_ratio 0.005 \
        --lr_scheduler linear \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 1 \
        --total_steps 625000 \
        --learning_rate $lr \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 16 \
        --eval_batch_size_per_device 16 \
        --save_strategy steps \
        --save_steps 156250 \
        --push_to_hub \
        --output_dir ""
}

train 5e-5
# train 1e-5
# train 5e-6