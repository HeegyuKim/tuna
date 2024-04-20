wandb online
model="Felladrin/TinyMistral-248M-Chat-v2"
# model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

python -m tuna.launcher.train_flax \
    --mesh fsdp \
    --do_train \
    --task dpo \
    --trainer dpo \
    --padding max_length \
    --project "flax-dpo" \
    --run_name "TinyMistral-248M-Chat-v2-ultrafeedback" \
    --dataset="dpo:ultrafeedback" \
    --packing False \
    --truncation \
    --truncation_side left \
    --max_length=1024 \
    --model_name_or_path $model \
    --total_epochs 3 \
    --learning_rate 5e-5 \
    --train_template chatml \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 8 \
    --eval_batch_size_per_device 8 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""