wandb online
model="google/gemma-2b"

train() {
    lr=$1
    datasets=$2
    run_name="0620-gemma-2B-$3"

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --project "KoChat-SFT" \
        --run_name "$run_name-lr$lr" \
        --dataset="$datasets" \
        --dataset_streaming \
        --truncation \
        --max_length=2048 \
        --model_name_or_path $model \
        --learning_rate $lr \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_steps 1000 \
        --lr_decay_steps 100000 \
        --train_template gemma \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 2 \
        --eval_batch_size_per_device 2 \
        --save_strategy steps \
        --save_steps 100000 \
        --total_steps 500000 \
        --push_to_hub \
        --push_to_hub_id $run_name \
        --revision_prefix "lr$lr-" \
        --output_dir ""
}

# train 2e-5 "HAERAE-HUB/qarv-instruct-100k" "qarv"
# train 2e-5 "kyujinpy/KOR-OpenOrca-Platypus-v3" "OOP-v3"
train 2e-5 "infiniinstruct+qarv-100k" "infini-qarv"