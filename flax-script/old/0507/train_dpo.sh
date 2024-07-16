wandb online
# model="HuggingFaceM4/tiny-random-LlamaForCausalLM"
# model="Felladrin/TinyMistral-248M-Chat-v2"
model="heegyu/TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T-tinyllama-1.1b-sft@steps-155897"

# model="HuggingFaceH4/mistral-7b-sft-beta"
# name="test-mistral-7b-sft-beta-lora-max-margin"

train() {
    beta=$1
    name="0509-1.1b-sft-dpo-beta$beta-max-margin"

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task dpo \
        --trainer dpo \
        --padding max_length \
        --project "DDFO-DPO" \
        --run_name "$name" \
        --dataset="dpo:heegyu/UltraFeedback-max-margin" \
        --packing False \
        --truncation \
        --max_length=2048 \
        --model_name_or_path $model \
        --total_epochs 1 \
        --logging_steps 128 \
        --learning_rate 5e-5 \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.1 \
        --train_template zephyr \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --output_dir ""
}

for beta in 0.01 0.1 0.25 0.5; do
    train $beta
done