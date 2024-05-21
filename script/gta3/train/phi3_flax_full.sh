wandb online

model="microsoft/Phi-3-mini-4k-instruct"
template="phi-3"

train() {
    dataset="$1:droussis/UltraSafety_binarized-orpo-dpo,$1:PKU-Alignment/PKU-SafeRLHF-30K"
    task=$2
    trainer=$3
    run_name="0521-Phi-3-mini-4k-instruct-safety-$task-full-2e-5"

    if [ "$task" == "sft" ]; then
        packing="true"
    else
        packing="false"
    fi

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --task $task \
        --trainer $trainer \
        --do_train \
        --padding max_length \
        --project "GTA3-SAFE-$1" \
        --train_template $template \
        --run_name "$run_name" \
        --dataset="$dataset" \
        --packing $packing \
        --max_length=2048 \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate 2e-5 \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.1 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_per_epoch 2 \
        --save_strategy epoch \
        --push_to_hub \
        --output_dir ""

}


# train "sft" "chat-lm"
train "dpo" "dpo" "dpo"
