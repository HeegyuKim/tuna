wandb online
# model="beomi/Llama-3-Open-Ko-8B"
model="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
dataset="prometheus:heegyu/K2-Feedback-splited"
# model="maywell/Llama-3-Ko-8B-Instruct"

model="maywell/Llama-3-Ko-8B-Instruct"

python -m tuna.launcher.train_flax \
    --mesh sp \
    --do_train --do_eval \
    --task chat-lm \
    --padding max_length \
    --project "KoChat-SFT" \
    --run_name "KoPrometheus-8B-0508" \
    --dataset="$dataset" \
    --packing False \
    --max_length=1536 \
    --truncation \
    --model_name_or_path $model \
    --logging_steps 1 \
    --total_epochs 3 \
    --learning_rate 1e-5 \
    --train_template no \
    --last_learning_rate_ratio 0.1 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 1 \
    --eval_batch_size_per_device 1 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""