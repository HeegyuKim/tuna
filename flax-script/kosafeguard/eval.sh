# export HF_HOME=/data-plm/hf-home
# export LD_LIBRARY_PATH=~/anaconda3/envs/easydel/lib/:$LD_LIBRARY_PATH

eval(){
    model=$1
    # python -m eval.kosafeguard \
    #     --model $model \
    #     --dataset pku-safe-rlhf \
    #     --output_path outputs/$model/pku-safe-rlhf.json
    
    python -m eval.kosafeguard \
        --model $model \
        --dataset kor-ethical-qa \
        --output_path outputs/$model/kor-ethical-qa.json
}

eval "heegyu/KoSafeGuard-gemma-ko-2b-0426@steps-143504"