wandb online
model="KETI-AIR/ke-t5-base-ko"

batch=4
total_steps=$(( 10000000 / $batch * 3 ))
save_steps=$(( $total_steps / 30 ))

echo "Total steps: $total_steps"

python -m tuna.launcher.train \
    --mesh fsdp \
    --do_train --do_eval \
    --task seq2seq-lm \
    --padding max_length \
    --model_arch seq2seq-lm \
    --project "kamo" \
    --run_name "aihub_mt_$model" \
    --dataset="aihub-mt-koen" \
    --dataset_streaming \
    --truncation \
    --max_length=128 \
    --model_name_or_path $model \
    --total_steps $total_steps \
    --learning_rate 3e-4 \
    --train_total_batch_size 128 \
    --train_batch_size_per_device $batch \
    --eval_batch_size_per_device $batch \
    --save_strategy steps \
    --save_steps $save_steps \
    --eval_strategy steps \
    --eval_steps $save_steps \
    --eval_batch_limit 10000 \
    --push_to_hub \
    --output_dir ""