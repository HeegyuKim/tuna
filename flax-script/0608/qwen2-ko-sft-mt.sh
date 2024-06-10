wandb online
model="Qwen/Qwen2-7B"
run_name="0608-Qwen2-7B-sft-en2ko"
datasets="squarelike/sharegpt_deepl_ko_translation"
# datasets="HAERAE-HUB/qarv-instruct-100k,heegyu/OpenOrca-gugugo-ko-len500"

train() {
    lr=$1

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --project "KoChat-SFT" \
        --run_name "$run_name-lr$lr" \
        --dataset="$datasets" \
        --packing False \
        --truncation \
        --max_length=2048 \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.01 \
        --train_template chatml \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy last \
        --push_to_hub \
        --push_to_hub_id $run_name \
        --revision_prefix "lr$lr-" \
        --output_dir ""
}

train 2e-5