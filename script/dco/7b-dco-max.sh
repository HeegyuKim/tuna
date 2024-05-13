wandb online
model="alignment-handbook/zephyr-7b-sft-full"
# model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
# model="HuggingFaceM4/tiny-random-LlamaForCausalLM"


train() {
    lr=$1
    beta=$2
    task=$3

    run_name="0513-$task-7b"
    
    python -m tuna.launcher.train \
        --mesh fsdp \
        --do_train \
        --task $task \
        --padding max_length \
        --model_arch causal-lm \
        --project "DDFO-DCO" \
        --dpo_beta $beta \
        --run_name "$run_name-$lr-b$beta" \
        --dataset="dco:heegyu/Ultrafeedback-split-dpo-max-margin" \
        --packing False \
        --truncation \
        --truncation_side left \
        --max_length 2048 \
        --use_lora \
        --model_name_or_path $model \
        --total_epochs 1 \
        --learning_rate $lr \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.1 \
        --train_template zephyr \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --push_to_hub_id $run_name \
        --revision_prefix "lr$lr-beta$beta" \
        --output_dir ""
}

train 2e-4 0.01 dco
train 2e-4 0.01 dco-v1d
train 2e-4 0.01 dco-v4
train 2e-4 0.01 dco-v4d
# train 2e-4 0.01 dco-v2