wandb online
model="Qwen/Qwen2-7B"

train() {
    lr=$1
    datasets=$2
    run_name="0610-Qwen2-7B-$3"

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --project "Qwen2-SFT" \
        --run_name "$run_name-lr$lr" \
        --dataset="$datasets" \
        --packing \
        --truncation \
        --max_length=2048 \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.01 \
        --train_template chatml \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 2 \
        --eval_batch_size_per_device 2 \
        --save_strategy epoch \
        --load_from_cache_file \
        --push_to_hub \
        --push_to_hub_id $run_name \
        --output_dir ""
}

train 2e-5 "openbmb/UltraInteract_sft,microsoft/orca-math-word-problems-200k,cognitivecomputations/WizardLM_evol_instruct_V2_196k_unfiltered_merged_split" "chito"
# train 2e-5 "TIGER-Lab/MathInstruct,microsoft/orca-math-word-problems-200k,migtissera/Tess-Coder-v1.0,m-a-p/Code-Feedback" 