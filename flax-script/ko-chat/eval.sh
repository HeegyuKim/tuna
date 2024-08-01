model="/data/checkpoint/0716-gemma2-koenzh/epoch-2/"

train() {
    lr=$1
    beta=$2
    gamma_beta_ratio=$3
    dataset=$4
    name=$5

    run_name="0716-gemma2-koenzh-simpo-$name"
    output_name="$run_name-lr$lr-beta$beta-gbr$gamma_beta_ratio"


    # alpaca_eval --model_outputs "outputs/$model_name/alpaca-eval.json"

    # ifeval
    # python -m eval.instruction_following_eval.evaluation_main \
    #     --input_response_data=outputs/$output_name/ifeval.jsonl

    python -m eval.judge_logickor -o "outputs/$output_name/logickor.jsonl"
}

train 1e-6 10 0.5 "dpo:princeton-nlp/gemma2-ultrafeedback-armorm" "armorm"
train 2e-6 10 0.5 "dpo:princeton-nlp/gemma2-ultrafeedback-armorm" "armorm"

# train 1e-6 10 0.5 "dpo:princeton-nlp/gemma2-ultrafeedback-armorm" "armorm"
# train 1e-6 10 0.5 "dpo:heegyu/Magpie-Pro-DPO-200K-Filtered" "magpie"

# train 2e-6 10 0.5 "dpo:princeton-nlp/gemma2-ultrafeedback-armorm" "armorm"
# train 2e-6 10 0.5 "dpo:heegyu/Magpie-Pro-DPO-200K-Filtered" "magpie"
