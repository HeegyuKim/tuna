wandb online
model="beomi/gemma-ko-2b"

python -m tuna.launcher.train_flax \
    --mesh fsdp \
    --do_train --do_eval \
    --task chat-lm \
    --padding max_length \
    --truncation \
    --project "LlamaGuard" \
    --train_template llamaguard \
    --run_name "gemma-ko-2b-0426" \
    --dataset="llamaguard:heegyu/PKU-SafeRLHF-ko" \
    --packing False \
    --max_length=2048 \
    --model_name_or_path $model \
    --total_epochs 2 \
    --learning_rate 5e-5 \
    --limit 256 \
    --last_learning_rate_ratio 0.1 \
    --train_total_batch_size 128 \
    --train_batch_size_per_device 2 \
    --eval_batch_size_per_device 2 \
    --save_strategy no \
    --push_to_hub \
    --output_dir ""