wandb online
# model="HuggingFaceM4/tiny-random-LlamaForCausalLM"
# model="Felladrin/TinyMistral-248M-Chat-v2"
model="maywell/Llama-3-Ko-8B-Instruct"
name="Llama-3-Ko-8B-Instruct-orca-dpo"

# model="HuggingFaceH4/mistral-7b-sft-beta"
# name="test-mistral-7b-sft-beta-lora-max-margin"

python -m tuna.launcher.train_flax \
    --mesh sp \
    --do_train \
    --task dpo \
    --trainer dpo \
    --padding max_length \
    --project "KoChat-DPO" \
    --run_name "$name" \
    --dataset="dpo:heegyu/orca_dpo_pairs_ko" \
    --packing False \
    --truncation \
    --max_length=1024 \
    --model_name_or_path $model \
    --total_epochs 1 \
    --logging_steps 128 \
    --learning_rate 1e-5 \
    --last_learning_rate_ratio 0.1 \
    --lr_warmup_ratio 0.1 \
    --train_template llama-3 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 1 \
    --eval_batch_size_per_device 1 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""