wandb online
model="google/gemma-2b"


train() {
    lr=$1
    datasets=$2
    run_name=$3

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --project "DDFO-SFT" \
        --run_name "$run_name-lr$lr" \
        --dataset="$datasets" \
        --packing False \
        --truncation \
        --max_length=2048 \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.1 \
        --train_template gemma \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 4 \
        --eval_batch_size_per_device 4 \
        --save_strategy epoch \
        --revision_prefix "lr$lr-" \
        --output_dir "/data/checkpoint/$run_name"
}

train 2e-5 "HuggingFaceH4/ultrachat_200k" "0625-gemma2b-sft-chat"