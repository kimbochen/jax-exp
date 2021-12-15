# Experimenting with JAX

This repository aims to be a non-trivial example of implementing a deep learning model using JAX.  
It contains the training code of a small GPT model, which is implemented using a minimal neural network library
written in JAX.

- [Setup and Usage](#setup-and-usage)
- [Model](#model)
- [Data Pipeline](#data)
- [Training Pipeline](#training-pipeline)
  - [`train`](#train)
  - [`step`](#step)
  - [`adam_i`](#adam_i)

## Setup and Usage

Replicate the virtual environment using Conda:
```bash
conda env create -f environment.yml
```

To train the model:
```bash
python train.py
```

To test out the model:
```bash
python evaluate.py --prompt <INPUT_PROMPT>
```

## Model

The model is defined in `model.py`.  
The model implementation is based on Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT), including:
- High-level modules: `GPT`, `Block`(Transformer), and `CausalSelfAttention`(self-attention).
- Helper modules: Modules originally provided by PyTorch, here implemented in the Python module `nn`.
  See `nn` for the full details.
- `GPTConfig`: A class that collects the configurations of the model.

## Data Pipeline

The data pipeline is based on Shawn Presser's [jax-exp](https://github.com/shawwn/jax-exp).  
My future plan is to implement a dataloader object based on
[JaxNeRF](https://github.com/google-research/google-research/blob/master/jaxnerf/nerf/datasets.py).

## Training Pipeline

The training pipeline is defined in `train.py`.  
The script trains a model and saves it in the pickle format.

### `train`

Trains the given model.

Function signature: `train(model, dataloader, tconf)`

- `model`: The model to be trained.
- `dataloader`: An iterator that feeds data.
- `tconf`: An object of class `TrainConfig`, containing the training configurations.

Output: A trained model.

### `step`

Performs one optimization step, including:

- A forward pass.
- Calculating the loss and the gradients.
- Update the optimization state.
- Optimize the model using the optimization state and the gradients.

Function signature: `step(model, xb, yb, opt_state)`

- `model`, `xb`, `yb`: The model and the data.
- `opt_state`: The optimization state.

Output:

- `model`
- `opt_state`

### `adam_i`

Optimizes the model using according to the Adaptive Momentum (Adam) optimizer.

Function signature: `adam_i(param, mu, var, i)`

- `param`: A single parameter of the model.
- `mu`: The running mean of gradients.
- `var`: The running variance of gradients.
- `i`: The epoch count.

Usage:

- This function will be wrapped by `partial`, specifying keyword `i` and becoming `adam`.
- `adam` is used with `jax.tree_map` to update `model`.
  ```python
  model = jax.tree_map(adam, model, mu, var)
  ```
- This function will be defined in `train`, eliminating the need to pass `tconf` in the arguments.
