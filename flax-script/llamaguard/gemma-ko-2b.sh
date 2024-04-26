wandb online
model="beomi/gemma-ko-2b"

python -m tuna.launcher.train_flax \
    --mesh fsdp \
    --do_train --do_eval \
    --task chat-lm \
    --padding max_length \
    --truncation \
    --project "KoSafeGuard" \
    --train_template llamaguard \
    --run_name "KoSafeGuard-gemma-ko-2b-0426" \
    --dataset="llamaguard:heegyu/PKU-SafeRLHF-ko" \
    --packing False \
    --max_length=2048 \
    --model_name_or_path $model \
    --total_epochs 2 \
    --learning_rate 5e-5 \
    --eval_limit 1000 \
    --last_learning_rate_ratio 0.1 \
    --train_total_batch_size 128 \
    --train_batch_size_per_device 2 \
    --eval_batch_size_per_device 2 \
    --save_strategy epoch \
    --save_per_epoch 2 \
    --push_to_hub \
    --output_dir ""