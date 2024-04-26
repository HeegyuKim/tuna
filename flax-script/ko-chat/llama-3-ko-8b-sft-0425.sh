wandb online
model="beomi/Llama-3-Open-Ko-8B"
dataset="FreedomIntelligence/evol-instruct-korean"
# dataset="changpt/ko-lima-vicuna"

python -m tuna.launcher.train_flax \
    --mesh fsdp \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --project "KoChat-SFT" \
    --run_name "Llama-3-Open-Ko-8B-sft-0425" \
    --dataset="$dataset" \
    --packing False \
    --max_length=1024 \
    --truncation \
    --model_name_or_path $model \
    --logging_steps 1 \
    --total_epochs 3 \
    --learning_rate 1e-5 \
    --last_learning_rate_ratio 0.1 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 1 \
    --eval_batch_size_per_device 1 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""