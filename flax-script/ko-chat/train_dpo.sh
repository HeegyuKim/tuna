wandb online
# model="HuggingFaceM4/tiny-random-LlamaForCausalLM"
# model="Felladrin/TinyMistral-248M-Chat-v2"
model="/data/checkpoint/0718-llama3-ko/epoch-2/"
name="0718-llama3-ko-dpo-magpie"

train() {
    lr=$1
    beta=$2

    # python -m tuna.launcher.train_flax \
    #     --mesh sp \
    #     --do_train \
    #     --task dpo \
    #     --trainer dpo \
    #     --padding max_length \
    #     --project "KoChat-DPO" \
    #     --run_name "$name-lr$lr-beta$beta" \
    #     --dataset="dpo:heegyu/Magpie-Pro-DPO-200K-Filtered" \
    #     --packing False \
    #     --truncation \
    #     --max_length=1536 \
    #     --model_name_or_path $model \
    #     --total_epochs 1 \
    #     --logging_steps 128 \
    #     --learning_rate 5e-6 \
    #     --filter_no_bos \
    #     --last_learning_rate_ratio 0.1 \
    #     --lr_warmup_ratio 0.03 \
    #     --train_template llama3 \
    #     --train_total_batch_size 32 \
    #     --train_batch_size_per_device 1 \
    #     --eval_batch_size_per_device 1 \
    #     --save_strategy epoch \
    #     --revision_prefix "lr$lr-beta$beta-" \
    #     --output_dir "/data/checkpoint/$name-lr$lr-beta$beta"

    # python -m eval.nlgbench_gen \
    #     "/data/checkpoint/$name-lr$lr-beta$beta/epoch-1" \
    #     --model_name "$name-lr$lr-beta$beta" \
    #     --dataset logickor \
    #     --batch_size 1 \
    #     --prompt_length 3072 \
    #     --max_new_tokens 2048 

    python -m eval.judge_logickor \
        -o "outputs/$name-lr$lr-beta$beta/logickor.jsonl"
}

train 5e-6 0.1
train 5e-6 0.01
