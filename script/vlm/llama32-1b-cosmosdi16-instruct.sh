
dataset="heegyu/clean-llava-instruct-mix-cosmos-di16-256"
model="heegyu/llama-3.2-1B-Instruct-cosmosdi16-stage1-1126"
revision="lr5e-5-bs128-main"

wandb online

run_name="llama-3.2-1B-cosmosdi16-instruct-1126"

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
        --max_length=1024 \
        --amp \
        --lr_warmup_ratio 0.01 \
        --truncation \
        --model_name_or_path $model \
        --revision $revision \
        --total_epochs 1 \
        --learning_rate $lr \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 2 \
        --eval_batch_size_per_device 2 \
        --save_strategy last \
        --push_to_hub \
        --output_dir ""
}

train 5e-5
# train 1e-5
# train 5e-6