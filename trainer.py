from dataclasses import dataclass
from functools import partial

import jax
import tqdm
import numpy as onp
import jax.numpy as jnp
import jax.random as rand

from dataset_util import iterbatches, process_dataset, train_test_split
from model import GPT, GPTConfig


@dataclass
class TrainerConfig:
    max_epoch: int
    batch_size: int
    lr: float
    b1: float = 0.9
    b2: float = 0.99
    eps: float = 1e-8


def init_model(model, seed):
    params, treedef = jax.tree_flatten(model)
    keys = rand.split(rand.PRNGKey(seed), len(params))
    init_param = lambda p, k: p.init(k)
    return treedef.unflatten(init_param(*xs) for xs in zip(params, keys))


def Adam(tconf):
    def init_fn(model):
        create_state = lambda x: (x, jnp.zeros_like(x), jnp.zeros_like(x))
        state = jax.tree_map(create_state, model)
        return state

    def get_param(state, idx):
        get_elem = lambda x: x[idx]
        is_tuple = lambda x: isinstance(x, tuple)
        return jax.tree_map(get_elem, state, is_leaf=is_tuple)

    def update_fn(idx, state, grad):
        def adam_i(model, grad, mu, var, i):
            mu = (1.0 - tconf.b1) * grad + tconf.b1 * mu
            m_hat = mu / (1.0 - jnp.asarray(tconf.b1, mu.dtype) ** i)

            var = (1.0 - tconf.b2) * jnp.square(grad) + tconf.b2 * var
            v_hat = var / (1.0 - jnp.asarray(tconf.b2, var.dtype) ** i)

            model = model - tconf.lr * m_hat / (jnp.sqrt(v_hat) + tconf.eps)

            return model, mu, var

        adam = partial(adam_i, i=idx)
        model, mu, var = [get_param(state, i) for i in range(3)]
        state = jax.tree_map(adam, model, grad, mu, var)

        return state

    return init_fn, update_fn, get_param

def cross_entropy(model, x, y):
    logit = model(x)
    logprob = -jax.nn.log_softmax(logit, axis=-1)
    logprob = logprob.reshape(-1, logprob.shape[-1])  # (B*T, n_vocab)
    loss = logprob[jnp.arange(logprob.shape[0]), y.reshape([-1, ])].mean()
    return loss


def train(model, train_batch, tconf, mconf):
    opt_init, opt_update, get_param = Adam(tconf)
    state = opt_init(model)

    iterbatch = partial(iterbatches, batch_size=tconf.batch_size)
    pbar = tqdm.trange(tconf.max_epoch)

    @jax.jit
    def step(idx, state, xb, yb):
        model = get_param(state, 0)
        loss, grad = jax.value_and_grad(cross_entropy)(model, xb, yb)
        state = opt_update(idx, state, grad)
        return state, loss

    for epoch in pbar:
        losses = []
        for i, (batch,) in enumerate(iterbatch(train_batch), start=1):
            xb, yb = batch[:, :-1], batch[:, 1:]
            state, loss = step(i, state, xb, yb)
            losses.append(loss)
        pbar.set_description(f'Train loss {onp.mean(losses):.5f}')

    model = get_param(state, 0)

    return model

def evaluate(model, ctx, mconf, codebook):
    x = jnp.asarray(codebook.encode(ctx)).reshape(1, -1)

    for _ in tqdm.trange(300):
        x_cond = x if x.shape[1] <= mconf.block_size else x[:, -mconf.block_size:]
        logit = model(x_cond)
        prob = jax.nn.softmax(logit[:, -1, :], axis=-1)
        x = jnp.append(x, jnp.argmax(prob)).reshape(1, -1)

    response = ''.join([codebook.idx2token(idx) for idx in x.flatten()])
    print(response)


def main():
    text, codebook = process_dataset('input.txt', print_stats=False)

    mconf = GPTConfig(
        n_head=8, d_embd=256, n_layer=8,
        block_size=128, n_vocab=codebook.size
    )
    model = GPT(mconf)
    model = init_model(model, 39)
    train_batch, _ = train_test_split(codebook, text, mconf.block_size)

    tconf = TrainerConfig(max_epoch=500, batch_size=512, lr=1e-3)
    model = train(model, train_batch, tconf, mconf)

    ctx = "Thou shalt not fear"
    evaluate(model, ctx, mconf, codebook)

if __name__ == '__main__':
    main()
