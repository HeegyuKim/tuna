
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install fjformer==0.0.69
pip install -r requirements.txt
pip install -U flax transformers gradio