wandb online
model="Locutusque/TinyMistral-248M-v2.5-Instruct"

python -m tuna.launcher.train_flax \
    --mesh fsdp \
    --do_train \
    --task orpo \
    --padding max_length \
    --project "DDFO-ORPO" \
    --run_name "TinyMistral-248M-v2.5-Instruct-orpo" \
    --dataset="dpo:heegyu/UltraFeedback-max-margin" \
    --packing False \
    --truncation \
    --max_length=2048 \
    --model_name_or_path $model \
    --total_epochs 3 \
    --learning_rate 1e-5 \
    --last_learning_rate_ratio 0.1 \
    --train_template chatml \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 4 \
    --eval_batch_size_per_device 4 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""