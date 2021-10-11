import math
from dataclasses import dataclass
from functools import partial
from typing import List

import jax
import optax
import tqdm
import equinox as eqx
import numpy as onp
import jax.numpy as jnp
import jax.random as rnd

from dataset_util import iterbatches, process_dataset, train_test_split


def get_key():
    try:
        get_key.key = rnd.split(get_key.key)[0]
    except AttributeError:
        get_key.key = rnd.PRNGKey(39)
    return get_key.key


@dataclass
class GPTConfig:
    n_head: int
    d_embd: int
    n_layer: int
    block_size: int
    n_vocab: int
    embd_pdrop: float = 0.1
    res_pdrop: float = 0.1
    attn_pdrop: float = 0.1

@dataclass
class TrainerConfig:
    max_epoch: int
    batch_size: int
    lr: float


class Sequential(eqx.Module):
    layers: List[eqx.Module]

    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Linear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = rnd.normal(get_key(), [d_in, d_out])
        self.bias = jnp.zeros([d_out, ])

    def __call__(self, x):
        y = x @ self.weight + self.bias
        return y

class Embedding(eqx.Module):
    embd: jnp.ndarray

    def __init__(self, n_embd, d_embd):
        super().__init__()
        self.embd = rnd.normal(get_key(), [n_embd, d_embd])

    def __call__(self, x):
        x = self.embd[x, :]
        return x

class LayerNorm(eqx.Module):
    gamma: jnp.ndarray
    beta: jnp.ndarray
    eps: float = 1e-5

    def __init__(self, norm_shape):
        super().__init__()
        self.gamma = jnp.ones(norm_shape, 'f')
        self.beta = jnp.zeros(norm_shape, 'f')

    def __call__(self, x):
        u = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)

        x = (x - u) / jnp.sqrt(var + self.eps)
        x = x * self.gamma + self.beta

        return x

class Dropout(eqx.Module):
    keep: float

    def __init__(self, keep):
        super().__init__()
        self.keep = keep

    def __call__(self, x):
        return x


class CausalSelfAttention(eqx.Module):
    query: Linear
    key: Linear
    value: Linear
    attn_drop: Dropout
    res_drop: Dropout
    mask: jnp.ndarray
    project: Linear
    n_head: int

    def __init__(self, cfg):
        assert cfg.d_embd % cfg.n_head == 0
        super().__init__()

        self.query = Linear(cfg.d_embd, cfg.d_embd)
        self.key = Linear(cfg.d_embd, cfg.d_embd)
        self.value = Linear(cfg.d_embd, cfg.d_embd)

        self.attn_drop = Dropout(cfg.attn_pdrop)
        self.res_drop = Dropout(cfg.res_pdrop)

        mask = jnp.ones([cfg.block_size, cfg.block_size]) * float('-inf')
        self.mask = jnp.triu(mask, 1).reshape(1, 1, cfg.block_size, cfg.block_size)

        self.project = Linear(cfg.d_embd, cfg.d_embd)
        self.n_head = cfg.n_head

    def __call__(self, x):
        B, T, C = x.shape
        n_token = C // self.n_head

        # Q, K, V: (B, n_head, T, n_token)
        Q = self.query(x).reshape(B, T, self.n_head, n_token).transpose(0, 2, 1, 3)
        K = self.key(x).reshape(B, T, self.n_head, n_token).transpose(0, 2, 1, 3)
        V = self.value(x).reshape(B, T, self.n_head, n_token).transpose(0, 2, 1, 3)

        # attn: (B, n_head, T, T)
        attn = (Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(K.shape[-1])
        attn = attn * (self.mask[..., :T, :T] == 0) + self.mask[..., :T, :T]
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        y = attn @ V  # (B, n_head, T, n_token)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        y = self.res_drop(self.project(y))

        return y

class Block(eqx.Module):
    pre_ln: LayerNorm
    attn: CausalSelfAttention
    post_ln: LayerNorm
    mlp: Sequential

    def __init__(self, cfg):
        super().__init__()

        self.pre_ln = LayerNorm(cfg.d_embd)
        self.attn = CausalSelfAttention(cfg)
        self.post_ln = LayerNorm(cfg.d_embd)

        self.mlp = Sequential(
            Linear(cfg.d_embd, 4 * cfg.d_embd),
            jax.nn.gelu,
            Linear(4 * cfg.d_embd, cfg.d_embd),
            Dropout(cfg.res_pdrop)
        )

    def __call__(self, x):
        x = x + self.attn(self.pre_ln(x))
        x = x + self.mlp(self.post_ln(x))
        return x


class GPT(eqx.Module):
    tok_embd: Embedding
    pos_embd: jnp.ndarray
    drop: Dropout
    blocks: Sequential
    norm: LayerNorm
    head: Linear
    block_size: int

    def __init__(self, cfg):
        super().__init__()

        self.tok_embd = Embedding(cfg.n_vocab, cfg.d_embd)
        self.pos_embd = jnp.zeros([1, cfg.block_size, cfg.d_embd])
        self.drop = Dropout(cfg.embd_pdrop)

        self.blocks = Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = LayerNorm(cfg.d_embd)
        self.head = Linear(cfg.d_embd, cfg.n_vocab)

        self.block_size = cfg.block_size

    def __call__(self, idx):
        T = idx.shape[-1]
        assert T <= self.block_size

        tok_embd = self.tok_embd(idx)    # (T, d_embd)
        pos_embd = self.pos_embd[:, :T, :]  # (T, d_embd)
        x = self.drop(tok_embd + pos_embd)
        x = self.blocks(x)
        x = self.norm(x)
        logit = self.head(x)  # (T, n_vocab)

        return logit


@partial(jax.jit, static_argnums=1)
@jax.value_and_grad
def loss_fn(param, static, x, y):
    model = eqx.combine(param, static)
    logit = model(x)  # (B, T, n_vocab)

    logprob = -jax.nn.log_softmax(logit, axis=-1)
    logprob = logprob.reshape(-1, logprob.shape[-1])  # (B*T, n_vocab)
    loss = logprob[jnp.arange(logprob.shape[0]), y.reshape([-1, ])].mean()

    return loss

@partial(jax.jit, static_argnums=(1, 5))
def update(param, static, xb, yb, state, optim):
    loss, grad = loss_fn(param, static, xb, yb)
    updates, state = optim.update(grad, state)
    param = eqx.apply_updates(param, updates)
    return state, param, loss


def main():
    text, codebook = process_dataset('alice.txt', print_stats=False)
    tconf = TrainerConfig(max_epoch=500, batch_size=64, lr=1e-3)
    mconf = GPTConfig(
        n_head=8, d_embd=256, n_layer=8,
        block_size=128, n_vocab=codebook.size
    )

    train_batch, _ = train_test_split(codebook, text, mconf.block_size)
    iterbatch = partial(
        iterbatches, batch_size=tconf.batch_size, shuffle=False,
        include_final_partial_batch=False
    )

    param, static = eqx.partition(GPT(mconf), eqx.is_array)
    optim = optax.adam(tconf.lr)
    state = optim.init(param)
    pbar = tqdm.trange(tconf.max_epoch)

    for epoch in pbar:
        losses = []
        for batch, in iterbatch(train_batch):
            xb, yb = batch[:, :-1], batch[:, 1:]
            state, param, loss = update(param, static, xb, yb, state, optim)
            losses.append(loss)
        pbar.set_description(f'Train loss {onp.mean(losses):.5f}')


    model = eqx.combine(param, static)
    ctx = "Alice freezed as she heard"
    x = jnp.asarray(codebook.encode(ctx)).reshape(1, -1)

    for _ in tqdm.trange(100):
        x_cond = x if x.shape[1] <= mconf.block_size else x[:, -mconf.block_size:]
        logit = model(x_cond)
        prob = jax.nn.softmax(logit[:, -1, :], axis=-1)
        x = jnp.append(x, jnp.argmax(prob)).reshape(1, -1)

    response = ''.join([codebook.idx2token(idx) for idx in x.flatten()])
    print(response)

if __name__ == '__main__':
    main()
