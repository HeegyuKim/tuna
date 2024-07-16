wandb online
# model="Felladrin/TinyMistral-248M-Chat-v2"
# model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model="heegyu/TinyLlama-1.1b-feedback-tree-3-0422@epoch-3"

python -m tuna.launcher.train_flax \
    --mesh fsdp \
    --do_train \
    --task dfo \
    --trainer dpo \
    --padding max_length \
    --project "DDFO" \
    --run_name "TinyLlama-1.1b-feedback-tree-3-epoch3-distil" \
    --dataset="dfo:heegyu/UltraFeedback-feedback-tree-3" \
    --packing False \
    --truncation \
    --truncation_side left \
    --max_length=2048 \
    --prompt_length 1024 \
    --model_name_or_path $model \
    --total_epochs 3 \
    --logging_steps 128 \
    --learning_rate 1e-5 \
    --last_learning_rate_ratio 0.1 \
    --lr_warmup_ratio 0.1 \
    --train_template zephyr \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 4 \
    --eval_batch_size_per_device 4 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""