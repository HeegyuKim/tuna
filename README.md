<div align="center">
  <div>&nbsp;</div>
  <img src="image/tuna.webp" width="400"/> 

</div>

# TUNA

## Supported Features

- [x] Fine-tuning LM (chatbot)
- [ ] Pre-training LM
- 

## Setup
<details>
<summary>TPU</summary>
<div markdown="1">

```
# install torch, torch_xla
pip install torch~=2.2.0 torch_xla[tpu]~=2.2.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# or use docker
sudo docker run -it --name tuna \
    -d --privileged \
    --net host \
    --shm-size=16G \
    -e VM_NAME="TPUv4-A" \
    -v $HOME:/workspace \
    -v /data/hf-home:/root/.cache/huggingface/ \
    -v /data/checkpoint:/data/checkpoint/ \
    tuna \
    /bin/bash

# us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.2.0_3.10_tpuvm
# us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_tpuvm
# us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.0_3.10_tpuvm


```

If you see a error like a below while using conda:
```
RuntimeError: Failed to import transformers.training_args because of the following error (look up to see its traceback): 
libpython3.11.so.1.0: cannot open shared object file: No such file or director```
```
export USE_TORCH=True
export LD_LIBRARY_PATH=$HOME/miniconda/lib/
# or
export LD_LIBRARY_PATH=$HOME/miniconda/envs/?/lib
export LD_LIBRARY_PATH=$HOME/miniconda/envs/qax/lib:$LD_LIBRARY_PATH
```

</div>
</details>

```
pip install -r requirements.txt
```


# Discord Bot
```
python -m tuna.serve.flax_discord Qwen/Qwen2-7B-Instruct
```

## Evaluation

### Generations
```
python -m eval.nlgbench_gen MODEL_NAME --batch_size 4 --use_vllm --dataset ifeval,alpaca-eval,mt-bench,logickor
```

### Evaluation
```bash
# Logickor
python eval.judge_logickor -o outputs/heegyu/0713-qwen2-magpie-qarv@lr2e-5-epoch-1/logickor.json

# alpacaeval
alpaca_eval --model_outputs "outputs/$model_name/alpaca-eval.json" --annotators_config chatgpt

# ifeval
python -m eval.instruction_following_eval.evaluation_main \
    --input_response_data=outputs/$model/ifeval.json

# mt-bench
```