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
    -p 7860:7860 \
    -e VM_NAME="TPUv3-A" \
    -v $HOME:/workspace \
    heegyukim/tuna:0.0.1 \
    /bin/bash

# us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.2.0_3.10_tpuvm
# us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_tpuvm

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
export LD_LIBRARY_PATH=$HOME/miniconda/envs/qax/lib
```

</div>
</details>

```
pip install -r requirements.txt
```