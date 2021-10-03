import math
from dataclasses import dataclass
from functools import partial
from typing import Any, List

import jax
import optax
import tqdm

import equinox as eqx
import numpy as onp
import jax.numpy as jnp
import jax.random as rnd

from data import get_data, get_dataloader


@dataclass
class GPTConfig:
    key: jnp.ndarray
    n_head: int
    d_embd: int
    block_size: int
    n_vocab: int
    n_layer: int
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1


class Sequential(eqx.Module):
    layers: List[Any]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Embedding(eqx.Module):
    embd: jnp.ndarray

    def __init__(self, n_embd, d_embd, key):
        super().__init__()
        self.embd = rnd.normal(key, [n_embd, d_embd])

    def __call__(self, x):
        x = self.embd[x, :]
        return x

class Linear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, d_in, d_out, rng):
        super().__init__()
        rngs = rnd.split(rng)
        self.weight = rnd.normal(rngs[0], [d_in, d_out])
        self.bias = jnp.zeros([d_out, ])

    def __call__(self, x):
        y = x @ self.weight+ self.bias
        return y

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


class MaskedSelfAttention(eqx.Module):
    query: Linear
    key: Linear
    value: Linear
    attn_drop: eqx.Module
    resid_drop: eqx.Module
    mask: jnp.ndarray
    proj: Linear
    n_head: int

    def __init__(self, cfg, key):
        assert cfg.d_embd % cfg.n_head == 0
        super().__init__()
        keys = rnd.split(key, 4)

        self.query = Linear(cfg.d_embd, cfg.d_embd, keys[0])
        self.key = Linear(cfg.d_embd, cfg.d_embd, keys[1])
        self.value = Linear(cfg.d_embd, cfg.d_embd, keys[2])

        self.attn_drop = eqx.nn.Identity()
        self.resid_drop = eqx.nn.Identity()

        mask = jnp.ones([cfg.block_size, cfg.block_size]) * float('-inf')
        self.mask = jnp.triu(mask, 1)

        self.proj = Linear(cfg.d_embd, cfg.d_embd, keys[3])
        self.n_head = cfg.n_head

    def __call__(self, x):
        B, T, C = x.shape
        n_token = C // self.n_head

        # Q, K, V: (B, n_head, T, n_token)
        Q = self.query(x).reshape(B, T, self.n_head, n_token).transpose(0, 2, 1, 3)
        K = self.key(x).reshape(B, T, self.n_head, n_token).transpose(0, 2, 1, 3)
        V = self.value(x).reshape(B, T, self.n_head, n_token).transpose(0, 2, 1, 3)

        attn = (Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(K.shape[-1])
        attn = attn * (self.mask[..., :T, :T] == 0) + self.mask[..., :T, :T]
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)  # (B, n_head, T, T)

        y = attn @ V  # (B, n_head, T, n_token)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_drop(self.proj(y))

        return y

class Block(eqx.Module):
    pre_ln: LayerNorm
    post_ln: LayerNorm
    attn: MaskedSelfAttention
    mlp: eqx.nn.Sequential

    def __init__(self, cfg, key):
        super().__init__()
        keys = rnd.split(key, 3)

        self.pre_ln = LayerNorm(cfg.d_embd)
        self.post_ln = LayerNorm(cfg.d_embd)
        self.attn = MaskedSelfAttention(cfg, keys[0])
        self.mlp = Sequential([
            Linear(cfg.d_embd, 4 * cfg.d_embd, keys[1]),
            jax.nn.gelu,
            Linear(4 * cfg.d_embd, cfg.d_embd, keys[2]),
            eqx.nn.Identity()
        ])

    def __call__(self, x):
        x = x + self.attn(self.pre_ln(x))
        x = x + self.mlp(self.post_ln(x))
        return x

class GPT(eqx.Module):
    tok_embd: Embedding
    pos_embd: jnp.ndarray
    drop: eqx.Module
    blocks: eqx.nn.Sequential
    norm: LayerNorm
    head: Linear
    block_size: int

    def __init__(self, cfg):
        super().__init__()
        keys = rnd.split(cfg.key, cfg.n_layer + 2)

        self.tok_embd = Embedding(cfg.n_vocab, cfg.d_embd, keys[-2])
        self.pos_embd = jnp.zeros([1, cfg.block_size, cfg.d_embd])
        self.drop = eqx.nn.Identity()

        self.blocks = Sequential([
            Block(cfg, keys[i]) for i in range(cfg.n_layer)
        ])

        self.norm = LayerNorm(cfg.d_embd)
        self.head = Linear(cfg.d_embd, cfg.n_vocab, keys[-1])

        self.block_size = cfg.block_size

    def __call__(self, idx):
        T = idx.shape[1]
        assert T <= self.block_size

        tok_embd = self.tok_embd(idx)
        pos_embd = self.pos_embd[:, :T, :]
        x = self.drop(tok_embd + pos_embd)
        x = self.blocks(x)
        x = self.norm(x)
        logit = self.head(x)

        return logit


@partial(jax.jit, static_argnums=1)
@jax.value_and_grad
def loss_fn(param, static, x, y):
    model = eqx.combine(param, static)
    logit = model(x)
    B, T, V = logit.shape

    logit = logit.reshape(B*T, V)
    y = y.reshape([-1, ])

    logprob = -jax.nn.log_softmax(logit)
    loss = logprob[jnp.arange(B*T), y].mean()

    return loss

    return model, state, loss

def main():
    text, chars = get_data('alice.txt')
    print(f'Data has {len(text)} characters, {len(chars)} unique.')

    cfg = GPTConfig(
        key=rnd.PRNGKey(39),
        n_vocab=len(chars), block_size=128,
        n_layer=8, n_head=8, d_embd=128
    )

    model = GPT(cfg)
    param, static = eqx.partition(model, eqx.is_array)
    optim = optax.adam(3e-4)
    state = optim.init(param)

    max_epoch, batch_size = 100, 128
    pbar = tqdm.trange(max_epoch)

    for epoch in pbar:
        dataloader = get_dataloader(text, chars, batch_size, cfg.block_size)
        losses = []
        for xb, yb in dataloader:
            loss, grad = loss_fn(param, static, xb, yb)
            updates, state = optim.update(grad, state)
            param = eqx.apply_updates(param, updates)
            losses.append(loss)
        pbar.set_description(f'Epoch {epoch} loss: {onp.mean(losses):.5f}')


    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    model = eqx.combine(param, static)

    ctx = "Alice stopped in front of a "
    x = jnp.asarray([stoi[ch] for ch in ctx]).reshape(1, -1)

    for _ in tqdm.trange(50):
        x_cond = x if x.shape[1] <= cfg.block_size else x[:, -cfg.block_size:]
        logits = model(x_cond)
        logits = logits[:, -1, :]
        logprob = -jax.nn.log_softmax(logits)
        ix = jnp.argmax(logprob)
        x = jnp.append(x, ix).reshape(1, -1)

    response = ''.join([itos[idx] for idx in x.flatten()])
    print(response)


if __name__ == '__main__':
    main()
