datasets="HAERAE-HUB/K2-Feedback"
model="google/gemma-2-9b-it"
# model="google/gemma-2-2b-it"

python -m eval.dpo_gen $model --dataset $datasets --eos_token "<end_of_turn>" --batch_size 8