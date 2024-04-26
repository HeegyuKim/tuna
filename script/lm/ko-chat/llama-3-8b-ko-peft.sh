wandb online

dataset="FreedomIntelligence/evol-instruct-korean"
run_name="Llama-3-Open-Ko-8B-SFT-0426"
model="beomi/Llama-3-Open-Ko-8B"

python -m tuna.launcher.train \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --model_arch causal-lm \
    --project "KoChat-SFT" \
    --train_template llama3 \
    --run_name "$run_name" \
    --dataset="$dataset" \
    --packing False \
    --amp \
    --use_lora \
    --max_length=1024 \
    --truncation \
    --model_name_or_path $model \
    --total_epochs 3 \
    --learning_rate 1e-5 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 2 \
    --eval_batch_size_per_device 2 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""