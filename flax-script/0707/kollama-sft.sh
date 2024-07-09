wandb online
model="beomi/Llama-3-Open-Ko-8B"

train() {
    lr=$1
    datasets=$2
    run_name=$3

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
        --train_template llama3 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --push_to_hub \
        --push_to_hub_id $run_name \
        --save_strategy epoch \
        --revision_prefix "lr$lr-" \
        --output_dir ""
}

train 2e-5 "Magpie-Align/Magpie-Pro-MT-300K-v0.1,HAERAE-HUB/qarv-instruct-100k" "0708-kollama3-magpie-qarv"