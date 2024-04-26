wandb online
model="beomi/Llama-3-Open-Ko-8B"

python -m tuna.launcher.train_flax \
    --mesh fsdp \
    --do_train \
    --task orpo \
    --padding max_length \
    --project "KoChat-ORPO" \
    --run_name "Llama-3-Open-Ko-8B-orca-dpo-pairs-0425" \
    --dataset="dpo:SJ-Donald/orca-dpo-pairs-ko" \
    --packing False \
    --truncation \
    --truncation_side left \
    --max_length=2048 \
    --model_name_or_path $model \
    --total_epochs 3 \
    --logging_steps 1 \
    --learning_rate 2e-5 \
    --last_learning_rate_ratio 0.1 \
    --lr_warmup_ratio 0.1 \
    --train_template llama3 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 1 \
    --eval_batch_size_per_device 1 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""