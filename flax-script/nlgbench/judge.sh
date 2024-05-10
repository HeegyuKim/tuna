# v3-C
judge="http://34.147.19.153:35020/prometheus-7b-v2.0"

python -m eval.nlgbench_eval_prometheus --input_files outputs/*/*/alpaca-eval.json --dataset alpaca-eval --judge $judge