wandb online

model="microsoft/Phi-3-mini-4k-instruct"
template="phi-3"

train() {
    dataset="$1:droussis/UltraSafety_binarized-orpo-dpo,$1:PKU-Alignment/PKU-SafeRLHF-30K"
    task=$2
    trainer=$3
    run_name="0521-Phi-3-mini-4k-instruct-safety-$task-full-flax"

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --task $task \
        --trainer $trainer \
        --do_train \
        --padding max_length \
        --project "GTA3-SAFE-$2" \
        --train_template $template \
        --run_name "$run_name" \
        --dataset="$dataset" \
        --packing false \
        --max_length=1024 \
        --truncation \
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
        --revision_prefix "lr2e-5-beta0.1" \
        --push_to_hub \
        --output_dir ""

}


train "dpo" "dpo" "dpo"
