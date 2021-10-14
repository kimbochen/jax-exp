from dataclasses import dataclass
from functools import partial

import jax
import optax
import tqdm
import numpy as onp
import jax.numpy as jnp
import jax.experimental.optimizers as opt

from dataset_util import iterbatches, process_dataset, train_test_split
from model import GPT, GPTConfig

@dataclass
class TrainerConfig:
    max_epoch: int
    batch_size: int
    lr: float


@jax.value_and_grad
def cross_entropy(model, x, y):
    logit = model(x)
    logprob = -jax.nn.log_softmax(logit, axis=-1)
    logprob = logprob.reshape(-1, logprob.shape[-1])  # (B*T, n_vocab)
    loss = logprob[jnp.arange(logprob.shape[0]), y.reshape([-1, ])].mean()
    return loss

# @jax.jit
# def step(model, xb, yb, lr):
#     loss, grad = cross_entropy(model, xb, yb)
#     model = jax.tree_multimap(lambda m, g: m - lr * g, model, grad)
#     return model, loss

# def train(model, train_batch, tconf, mconf):
#     iterbatch = partial(
#         iterbatches, batch_size=tconf.batch_size, shuffle=False,
#         include_final_partial_batch=False
#     )
#     pbar = tqdm.trange(tconf.max_epoch)
# 
#     for epoch in pbar:
#         losses = []
#         for batch, in iterbatch(train_batch):
#             xb, yb = batch[:, :-1], batch[:, 1:]
#             model, loss = step(model, xb, yb, tconf.lr)
#             losses.append(loss)
#         loss = onp.mean(losses)
#         pbar.set_description(f'Train loss {loss:.5f}')
# 
#     return model

@partial(jax.jit, static_argnames=['get_params', 'opt_update'])
def step(i, opt_state, get_params, opt_update, xb, yb):
    model = get_params(opt_state)
    loss, grad = cross_entropy(model, xb, yb)
    opt_state = opt_update(i, grad, opt_state)
    return loss, opt_state


def train(model, train_batch, tconf, mconf):
    opt_init, opt_update, get_params = opt.adam(tconf.lr)
    opt_state = opt_init(model)

    iterbatch = partial(
        iterbatches, batch_size=tconf.batch_size, shuffle=False,
        include_final_partial_batch=False
    )
    pbar = tqdm.trange(tconf.max_epoch)

    for epoch in pbar:
        losses = []
        for idx, (batch,) in enumerate(iterbatch(train_batch)):
            xb, yb = batch[:, :-1], batch[:, 1:]
            opt_state, loss = step(idx, opt_state, get_params, opt_update, xb, yb)
            losses.append(loss)
        loss = onp.mean(losses)
        pbar.set_description(f'Train loss {loss:.5f}')

    return get_params(opt_state)

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
    model, _ = GPT(mconf).init(39)
    train_batch, _ = train_test_split(codebook, text, mconf.block_size)

    tconf = TrainerConfig(max_epoch=350, batch_size=256, lr=3e-3)
    model = train(model, train_batch, tconf, mconf)

    ctx = "Thou shalt not fear"
    evaluate(model, ctx, mconf, codebook)

if __name__ == '__main__':
    main()
