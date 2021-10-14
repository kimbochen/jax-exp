import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as rand

from framework import Module, ModuleList, Param, RandParam


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


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = ModuleList(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Linear(Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = RandParam([d_in, d_out])
        self.bias = Param(jnp.zeros([d_out, ]))

    def __call__(self, x):
        y = x @ self.weight + self.bias
        return y

class Embedding(Module):
    def __init__(self, n_embd, d_embd):
        super().__init__()
        self.embd = RandParam([n_embd, d_embd])

    def __call__(self, x):
        x = self.embd[x, :]
        return x

class LayerNorm(Module):
    def __init__(self, norm_shape):
        super().__init__()
        self.gamma = Param(jnp.ones(norm_shape, 'f'))
        self.beta = Param(jnp.zeros(norm_shape, 'f'))
        self.eps = 1e-5

    def __call__(self, x):
        u = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)

        x = (x - u) / jnp.sqrt(var + self.eps)
        x = x * self.gamma + self.beta

        return x

class Dropout(Module):
    def __init__(self, keep):
        super().__init__()
        self.keep = keep

    def __call__(self, x):
        return x


class CausalSelfAttention(Module):
    def __init__(self, cfg):
        assert cfg.d_embd % cfg.n_head == 0
        super().__init__()

        self.query = Linear(cfg.d_embd, cfg.d_embd)
        self.key = Linear(cfg.d_embd, cfg.d_embd)
        self.value = Linear(cfg.d_embd, cfg.d_embd)

        self.attn_drop = Dropout(cfg.attn_pdrop)
        self.res_drop = Dropout(cfg.res_pdrop)

        mask = jnp.ones([cfg.block_size, cfg.block_size]) * float('-inf')
        self.mask = jnp.triu(mask, k=1).reshape(1, 1, cfg.block_size, cfg.block_size)

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

class Block(Module):
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

class GPT(Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_embd = Embedding(cfg.n_vocab, cfg.d_embd)
        self.pos_embd = Param(jnp.zeros([1, cfg.block_size, cfg.d_embd]))
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