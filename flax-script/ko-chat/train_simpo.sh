wandb online
model="/data/checkpoint/0716-gemma2-koenzh/epoch-2/"
run_name="0716-gemma2-koenzh-simpo"

dataset="
dpo:argilla/distilabel-math-preference-dpo
dpo:argilla/distilabel-capybara-dpo-7k-binarized
dpo:argilla/distilabel-intel-orca-dpo-pairs
"
# dataset="dpo:heegyu/Magpie-Pro-DPO-200K-Filtered"

# python -m eval.nlgbench_gen \
#     "/data/checkpoint/0716-gemma2-koenzh/epoch-2/" \
#     --model_name "0716-gemma2-koenzh@epoch-2" \
#     --dataset "alpaca-eval,ifeval,logickor" \
#     --chat_template gemma \
#     --batch_size 4 \
#     --prompt_length 2048 \
#     --max_new_tokens 2048 

train() {
    lr=$1
    beta=$2
    gamma_beta_ratio=$3

    output_name="$run_name-lr$lr-beta$beta-gbr$gamma_beta_ratio"

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task simpo \
        --padding max_length \
        --project "KoChat-SimPO" \
        --run_name "$output_name" \
        --dataset="$dataset" \
        --packing False \
        --truncation \
        --max_length=1536 \
        --simpo_prompt_length=768 \
        --simpo_response_length=768 \
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
        --load_from_cache_file \
        --save_strategy epoch \
        --revision_prefix "lr$lr-beta$beta-" \
        --output_dir "/data/checkpoint/$output_name" \

        # --push_to_hub \
        # --push_to_hub_id $run_name \

    python -m eval.nlgbench_gen \
        "/data/checkpoint/$output_name/epoch-1" \
        --model_name "$output_name" \
        --dataset "alpaca-eval,ifeval,logickor" \
        --chat_template gemma \
        --batch_size 4 \
        --prompt_length 2048 \
        --max_new_tokens 2048 

    # python -m eval.judge_logickor \
    #     -o "outputs/$run_name-lr$lr-beta$beta/logickor.jsonl"
}

train 5e-7 10 0.5
train 5e-7 2 0.5
train 1e-6 10 0.5
train 1e-6 2 0.5