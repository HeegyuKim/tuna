wandb online
model="openbmb/Eurus-7b-kto"


python -m tuna.launcher.train \
    --mesh fsdp \
    --do_train \
    --task self-distillation \
    --truncation \
    --truncation_side right \
    --padding max_length \
    --model_arch causal-lm \
    --project "dfo-distil" \
    --run_name "Eurus-7b-kto-gamma10" \
    --dataset="distil:openbmb/UltraInteract_pair" \
    --use_lora \
    --max_prompt_length=1536 \
    --max_response_length=512 \
    --distil_gamma 10 \
    --model_name_or_path $model \
    --total_epochs 3 \
    --learning_rate 2e-5 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 1 \
    --eval_batch_size_per_device 1 \
    --save_strategy epoch \
    --output_dir ""