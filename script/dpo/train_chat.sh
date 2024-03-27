export CUDA_VISIBLE_DEVICES=0

wandb offline
model="Locutusque/TinyMistral-248M-v2.5"

# accelerate launch -m tuna.launcher.train \
python -m tuna.launcher.train \
    --mesh fsdp \
    --do_train \
    --task dpo \
    --truncation \
    --truncation_side left \
    --padding max_length \
    --model_arch causal-lm \
    --project "test" \
    --run_name "test" \
    --dataset="dpo:ultrafeedback" \
    --use_lora \
    --max_length=1024 \
    --model_name_or_path $model \
    --logging_steps 1 \
    --total_epochs 3 \
    --learning_rate 1e-5 \
    --train_total_batch_size 128 \
    --train_batch_size_per_device 8 \
    --eval_batch_size_per_device 8 \
    --save_strategy no \
    --output_dir ""