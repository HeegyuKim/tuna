judge="http://34.90.80.144:35020/prometheus-7b-v2.0"

python -m eval.nlgbench_eval_prometheus --input_files outputs/*/*/mt-bench.json --dataset mt-bench --judge $judge