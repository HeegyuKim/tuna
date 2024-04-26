wandb online
model="klue/roberta-base"

python -m tuna.launcher.train \
    --mesh dp \
    --do_train --do_eval \
    --padding max_length \
    --num_labels 1 \
    --truncation \
    --task sequence-classification \
    --model_arch sequence-classification \
    --project "KoSafeGuard-BERT" \
    --run_name "KoSafeGuard-klue-roberta-base-0426" \
    --dataset="llamaguard-classification:heegyu/PKU-SafeRLHF-ko" \
    --max_length=512 \
    --model_name_or_path $model \
    --total_epochs 3 \
    --learning_rate 1e-4 \
    --train_total_batch_size 128 \
    --train_batch_size_per_device 32 \
    --eval_batch_size_per_device 32 \
    --eval_limit 1024 \
    --save_strategy epoch \
    --save_epochs 1 \
    --push_to_hub \
    --output_dir ""