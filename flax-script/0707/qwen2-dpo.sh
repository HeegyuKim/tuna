wandb offline
model="heegyu/0710-qwen2-magpie-qarv-komath"
run_name="0710-qwen2-magpie-qarv-komath-dpo-ultrainteract"

# dataset="
# dpo:argilla/distilabel-math-preference-dpo
# dpo:argilla/distilabel-capybara-dpo-7k-binarized
# dpo:argilla/distilabel-intel-orca-dpo-pairs
# "
#,dpo:heegyu/Magpie-Pro-DPO-200K-Filtered

dataset="dpo:openbmb/UltraInteract_pair"

train() {
    lr=$1
    beta=$2
    
    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task dpo \
        --trainer dpo \
        --padding max_length \
        --project "KoChat-DPO" \
        --run_name "$run_name-lr$lr-beta$beta" \
        --dataset="$dataset" \
        --packing False \
        --truncation \
        --max_length=2048 \
        --dpo_prompt_length=1024 \
        --dpo_response_length=1024 \
        --model_name_or_path $model \
        --total_epochs 1 \
        --learning_rate $lr \
        --dpo_beta $beta \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.03 \
        --train_template chatml \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --revision_prefix "lr$lr-beta$beta-" \
        --output_dir "/data/checkpoint/$run_name-lr$lr-beta$beta" \

        # --push_to_hub \
        # --push_to_hub_id $run_name \

    python -m eval.nlgbench_gen \
        "/data/checkpoint/$run_name-lr$lr-beta$beta/epoch-1" \
        --model_name "heegyu-local/$run_name-lr$lr-beta$beta" \
        --prompt_length 3072 \
        --dataset logickor \
        --batch_size 1 \
        --eos_token "<|endoftext|>"

    # python -m eval.judge_logickor \
    #     -o "outputs/heegyu-local/$run_name-lr$lr-beta$beta/logickor.jsonl"
}


for lr in 5e-6; do
    # train $lr 0.25
    train $lr 0.1
    train $lr 0.01
    # train $lr 0.5
    # train $lr 1.0
done