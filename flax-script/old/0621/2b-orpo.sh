wandb online
model="google/gemma-2b"
run_name="0621-gemma-2b-dpo"

train() {
    lr=$1
    beta=$2
    
    rm -rf ~/.cache/huggingface/datasets

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task orpo \
        --padding max_length \
        --project "DDFO-ORPO" \
        --run_name "$run_name-lr$lr-beta$beta" \
        --dataset="dpo:heegyu/Ultrafeedback-max-margin-critique" \
        --packing False \
        --truncation \
        --max_length=2048 \
        --orpo_prompt_length=1024 \
        --orpo_response_length=1024 \
        --orpo_beta $beta \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.1 \
        --train_template gemma \
        --filter_no_bos \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 2 \
        --eval_batch_size_per_device 2 \
        --save_strategy last \
        --output_dir "/data/checkpoint/$run_name/lr$lr-beta$beta"
}

train 2e-5 0.1
# for lr in 1e-5 5e-6; do
#     train $lr 0.5
#     # train $lr 0.25
#     train $lr 0.1
#     # train $lr 0.75
#     train $lr 1.0
# done