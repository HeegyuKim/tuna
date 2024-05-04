wandb online
model="HuggingFaceH4/mistral-7b-sft-beta"
dataset="ccft:heegyu/Ultrafeedback-split-dpo-max-margin"
name="0430-DFO-7b-CCFT"

python -m tuna.launcher.train_flax \
    --mesh sp \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --project "DDFO-CCFT" \
    --run_name "$name" \
    --dataset="$dataset" \
    --packing False \
    --truncation \
    --max_length=2048 \
    --model_name_or_path $model \
    --total_epochs 3 \
    --logging_steps 128 \
    --learning_rate 1e-5 \
    --last_learning_rate_ratio 0.1 \
    --lr_warmup_ratio 0.1 \
    --train_template zephyr \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 2 \
    --eval_batch_size_per_device 2 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""