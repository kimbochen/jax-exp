# Experimenting with JAX

This repository aims to be a non-trivial example of implementing a deep learning model using JAX.

It contains the training code of a small GPT model, which is implemented using a minimal neural network library
written in JAX.

## Setup

Create a virtual environment using Conda:
```bash=
conda env create -f environment.yml
```

## Usage

### Training

```bash=
python train.py
```

### Evaluating

```bash=
python evaluate.py --prompt <PROMPT>
```

`--prompt`: The input prompt given to the model.

## The Model

The model resides in `model.py`.

The model implementation is largely based on Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT),
including:
- `CausalSelfAttention`: A self-attention block that masks future words for the model to predict (causal).
- `Block`: A Transformer block.
- `GPT`: The whole model.
- `GPTConfig`: A class that collects the configurations of the model.

## The Training Pipeline


