export CUDA_VISIBLE_DEVICES=0

wandb online
model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# accelerate launch -m tuna.launcher.train \
python -m tuna.launcher.train \
    --mesh fsdp \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --model_arch causal-lm \
    --project "esconv" \
    --run_name "esconv-tinyllama-plm" \
    --dataset="thu-coai/esconv" \
    --train_template tinyllama \
    --truncation \
    --max_length=1024 \
    --model_name_or_path $model \
    --total_epochs 3 \
    --learning_rate 2e-5 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 2 \
    --eval_batch_size_per_device 2 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""