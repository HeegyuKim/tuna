wandb online
model="/data/checkpoint/0716-gemma2-koenzh/epoch-2/"
run_name="0716-gemma2-koenzh-simpo-margpie"

# dataset="
# dpo:argilla/distilabel-math-preference-dpo
# dpo:argilla/distilabel-capybara-dpo-7k-binarized
# dpo:argilla/distilabel-intel-orca-dpo-pairs
# "
dataset="dpo:heegyu/Magpie-Pro-DPO-200K-Filtered"

train() {
    lr=$1
    beta=$2
    gamma_beta_ratio=$3
    
    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task simpo \
        --padding max_length \
        --project "KoChat-SimPO" \
        --run_name "$run_name-lr$lr-beta$beta" \
        --dataset="$dataset" \
        --packing False \
        --truncation \
        --max_length=1536 \
        --simpo_prompt_length=512 \
        --simpo_response_length=1024 \
        --model_name_or_path $model \
        --total_epochs 1 \
        --learning_rate $lr \
        --filter_no_bos \
        --simpo_beta $beta \
        --simpo_gamma_beta_ratio $gamma_beta_ratio \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.03 \
        --train_template gemma \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --revision_prefix "lr$lr-beta$beta-" \
        --output_dir "/data/checkpoint/$run_name-lr$lr-beta$beta" \

        # --push_to_hub \
        # --push_to_hub_id $run_name \
}

gamma_beta_ratio=0.5

for lr in 5e-6; do
    # train $lr 0.25
    train $lr 0.1 $gamma_beta_ratio
    train $lr 0.01 $gamma_beta_ratio
    # train $lr 0.5
    # train $lr 1.0
done