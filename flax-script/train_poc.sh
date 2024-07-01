wandb offline
# model="HuggingFaceM4/tiny-random-LlamaForCausalLM"
model="Locutusque/TinyMistral-248M"

python -m tuna.launcher.train_flax \
    --mesh fsdp \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --project "test" \
    --run_name "test" \
    --dataset="GAIR/lima" \
    --packing \
    --max_length=512 \
    --model_name_or_path $model \
    --logging_steps 1 \
    --total_epochs 3 \
    --learning_rate 1e-4 \
    --last_learning_rate_ratio 0.1 \
    --lr_warmup_ratio 0.1 \
    --train_total_batch_size 16 \
    --train_batch_size_per_device 8 \
    --eval_batch_size_per_device 8 \
    --save_strategy epoch \
    --output_dir "gs://heegyu-v4/flax-models/lima"