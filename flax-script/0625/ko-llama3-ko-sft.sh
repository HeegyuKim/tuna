wandb online
model="beomi/Llama-3-Open-Ko-8B"

train() {
    lr=$1
    datasets=$2
    run_name="llama3-7B-$3"

    step_batch=4
    total_batch=512
    epoch_steps=$4
    save_steps=$((epoch_steps / step_batch / 2)) # 4 times per epoch
    total_steps=$((epoch_steps * 3 / step_batch))
    lr_warmup_steps=$((total_batch * 40 / step_batch))

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
        --max_length=1024 \
        --model_name_or_path $model \
        --adam_beta2 0.95 \
        --logging_steps 4096 \
        --lr_scheduler cosine \
        --learning_rate $lr \
        --last_learning_rate_ratio 0.0 \
        --lr_warmup_steps $lr_warmup_steps \
        --train_template llama3 \
        --train_total_batch_size $total_batch \
        --train_batch_size_per_device $step_batch \
        --eval_batch_size_per_device $step_batch \
        --save_strategy steps \
        --save_steps $save_steps \
        --total_steps $total_steps \
        --revision_prefix "lr$lr-" \
        --output_dir "gs://heegyu-v4/$run_name"
        
        # --push_to_hub \
        # --push_to_hub_id $run_name \
}

# train 5e-6 "infiniinstruct+qarv-100k" "infini-qarv" 1000000
train 5e-6 "0705-koen-1M" "0705-koen-1M" 975000