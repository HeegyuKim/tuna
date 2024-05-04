wandb online
model="heegyu/TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T-tinyllama-1.1b-sft@steps-155897"
dataset="dpo:heegyu/Ultrafeedback-split-dpo-max-margin"



train() {
    lr=$1
    name="0504-DCO-1.1b-$dataset-$lr"
    
    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task dco \
        --trainer dpo \
        --padding max_length \
        --project "DDFO-DCO" \
        --run_name "$name" \
        --dataset="dco:heegyu/Ultrafeedback-split-dpo-max-margin" \
        --packing False \
        --truncation \
        --max_length=2048 \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.1 \
        --train_template zephyr \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --output_dir ""
}

train 5e-5