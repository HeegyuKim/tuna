# example
# ./script/seq2seq/mt.sh aihub-mt-koen
# ./script/seq2seq/mt.sh aihub-mt-koen-report

wandb online
model="heegyu/aihub_mt_ket5-base-ko_aihub-mt-koen-report"
dataset="kamo-mt-koen"

batch=4

echo "Model: $model"
echo "Dataset: $dataset"

python -m tuna.launcher.train \
    --mesh fsdp \
    --do_train --do_eval \
    --task seq2seq-lm \
    --model_arch seq2seq-lm \
    --model_name_or_path $model \
    --revision "steps-2000000"\
    --project "kamo" \
    --run_name "aihub-mt-koen-report-2M-kamo-0405-en-only-b32" \
    --dataset=$dataset \
    --padding max_length \
    --truncation \
    --max_length=128 \
    --learning_rate 1e-4 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device $batch \
    --eval_batch_size_per_device $batch \
    --total_epochs 3 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --push_to_hub \
    --output_dir ""