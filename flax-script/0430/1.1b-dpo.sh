wandb online
model="heegyu/TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T-tinyllama-1.1b-sft@steps-155897"
dataset="dpo:heegyu/Ultrafeedback-split-dpo-max-margin"
name="0430-DFO-1.1b-DPO"

python -m tuna.launcher.train_flax \
    --mesh fsdp \
    --do_train --do_eval \
    --task dpo \
    --trainer dpo \
    --padding max_length \
    --project "DDFO-DPO" \
    --run_name "$name" \
    --dataset="$dataset" \
    --packing False \
    --truncation \
    --max_length=2048 \
    --model_name_or_path $model \
    --total_epochs 3 \
    --logging_steps 128 \
    --learning_rate 5e-6 \
    --last_learning_rate_ratio 0.1 \
    --lr_warmup_ratio 0.1 \
    --train_template zephyr \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 2 \
    --eval_batch_size_per_device 2 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""