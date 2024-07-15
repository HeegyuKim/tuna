wandb online

template="chatml"
lr=5e-5
task="chat-lm"
model="Qwen/Qwen2-1.5B-Instruct"

train() {
    dataset=$1
    run_name=$2
    
    hub_id="iknow-lab/0613-glaive_cluster10_$run_name"

    python -m tuna.launcher.train \
        --mesh sp \
        --do_train --do_eval \
        --task $task \
        --padding max_length \
        --model_arch causal-lm \
        --project "FedDPO-DialogSum" \
        --train_template $template \
        --run_name "dialogsum_cluster10_$run_name" \
        --dataset="$dataset" \
        --packing False \
        --trust_remote_code \
        --amp True \
        --max_length=512 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --lr_warmup_ratio 0.1 \
        --lr_decay_ratio 0.1 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --push_to_hub_id $hub_id \
        --output_dir ""
}

# train "iknow-lab/dialogsum_cluster10_0613:full" "full"
# train "iknow-lab/dialogsum_cluster10_0613:server" "server"
train "iknow-lab/glaive-function-calling-v2-single-cluster10-0614:full" "full"
train "iknow-lab/glaive-function-calling-v2-single-cluster10-0614:server" "server"