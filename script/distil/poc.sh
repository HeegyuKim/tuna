export CUDA_VISIBLE_DEVICES=0

wandb online
model="Felladrin/TinyMistral-248M-Chat-v2"

# accelerate launch -m tuna.launcher.train \
python -m tuna.launcher.train \
    --mesh fsdp \
    --do_train \
    --task self-distillation \
    --truncation \
    --truncation_side right \
    --padding max_length \
    --model_arch causal-lm \
    --project "test" \
    --run_name "test" \
    --dataset="distil:openbmb/UltraInteract_pair" \
    --use_lora \
    --max_prompt_length=1536 \
    --max_response_length=512 \
    --distil_gamma 10 \
    --model_name_or_path $model \
    --total_epochs 3 \
    --learning_rate 2e-5 \
    --train_total_batch_size 128 \
    --train_batch_size_per_device 4 \
    --eval_batch_size_per_device 4 \
    --save_strategy no \
    --output_dir ""