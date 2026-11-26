# MADAO-1
A very stupid model. I wrote a naive implementation of the forward pass in SimpleLLaMA.py using numpy instead of pytorch, and then (rightfully so) got too lazy to write anything else. MADAO.py and train_madao.py were all vibe-coded.

On accident, I only added 1 block instead of 8 blocks, so the embedding matrix represents 80% of the total parameters. I believe the size is around 48 M parameters.

Training was not with AdamW.

## Setup

These commands use Python 3.12 and a CUDA-enabled PyTorch build for NVIDIA GPUs. Run them from the project directory. PyTorch installation options

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install numpy tiktoken

Check that PyTorch can access the GPU:

python -c 'import torch; print(torch.cuda.get_device_name(0))'

## Run

Start the original one-block model:

python MADAO.py --device cuda

Type a prompt at >> and enter 'quit' to exit. Set --temperature between 0 and 1.

## Train

Provide a UTF-8 text file. For example:

python train_madao.py data/corpus.txt \
  --device cuda --steps 2000 --context 128 --eval-every 200

The models save separate checkpoints in weights/ after a completed run. On first use, a model creates its checkpoint. The eight-block model starts with fresh weights. To continue an existing model, place its checkpoint in weights/ before running it.
