# Experimenting with JAX

This repository aims to be a non-trivial example of implementing a deep learning model using JAX.  
It contains the training code of a small GPT model, which is implemented using a minimal neural network library
written in JAX.

## Setup and Usage

Replicate the virtual environment using pip:
```bash
pip install -r requirements.txt
```

To train the model:
```bash
python train.py
```

To test out the model:
```bash
python evaluate.py -c <INPUT_PROMPT>
# -c / --context: The input prompt / the context.
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

### Optimizer

We implement a simple Adam optimizer, following a functional programming pattern.

#### `init_adam`

Creates optimizer functions specified by the training configuration.

Function signature: `init_adam(tconf)`

- `tconf`: The training configuration

Returns:
- `adam`: The adam optimizer.
- `init_opt_state`: Initialize the optimizer state, including the mean (`mu`) and variance (`var`).
- `split_tree`: A helper function for splitting the optimizer tree into model and optimizer state.

#### `adam`

Optimizes the model using according to the Adaptive Momentum (Adam) optimizer.

Function signature: `adam(param, grad, mu, var, i)`

- `param`: A single parameter of the model.
- `grad`: A single gradient.
- `mu`: The running mean of gradients.
- `var`: The running variance of gradients.
- `i`: The epoch count.

Usage:

- This function will be wrapped by `partial`, specifying keyword `i`.
- The wrapped function is then used with `jax.tree_map` to update `model`.
  ```python
  model = jax.tree_map(adam, model, grads, mu, var)
  ```
- This function will be defined in `train`, eliminating the need to pass `tconf` in the arguments.

#### `init_opt_state`

Initializes the optimizer state according to the model.  
Specifically, 2 PyTrees and an epoch-counting variable is initialized.

Function signature: `init_opt_state(model)`

- `model`: The model.

#### `split_tree`

Splits an optimizer tree into subtrees.  
The output of `adam` is the optimizer tree, containing a PyTree with the same structure as the model,
but **each leaf is a tuple containing a parameter, a running mean, and a running variance**.  
To make the training pipeline clear, I split the optimizer tree into the model tree, the mean tree (`mu`),
and the variance tree (`var`).

Function signature: `split_tree(opt_tree, idx)`

- `opt_tree`: The optimizer tree.
- `idx`: The index of the leaf tuple.
