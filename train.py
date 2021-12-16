import pickle
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as rand

from dataset_util import Dataloader, process_dataset, train_test_split
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

def adam_init(tconf):
    def adam(param, grad, mu, var, i):
        mu = (1.0 - tconf.b1) * grad + tconf.b1 * mu
        var = (1.0 - tconf.b2) * jnp.square(grad) + tconf.b2 * var

        m_hat = mu / (1.0 - tconf.b1 ** i)
        v_hat = var / (1.0 - tconf.b2 ** i)
        param = param - tconf.lr * m_hat / (jnp.sqrt(v_hat) + tconf.eps)

        return param, (mu, var)

    def init_opt_state(model):
        init_stats = lambda p: (jnp.zeros_like(p), jnp.zeros_like(p))
        stats = jax.tree_map(init_stats, model)
        return stats, 1

    return adam, init_opt_state

def cross_entropy(model, x, y):
    logit = model(x)
    logprob = -jax.nn.log_softmax(logit, axis=-1)
    logprob = logprob.reshape(-1, logprob.shape[-1])  # (B*T, n_vocab)
    loss = logprob[jnp.arange(logprob.shape[0]), y.reshape([-1, ])].mean()
    return loss


def train(model, train_dl, tconf):
    adam, init_opt_state = adam_init(tconf)
    opt_state = init_opt_state(model)

    @jax.jit
    def step(model, xb, yb, opt_state):
        loss, grads = jax.value_and_grad(cross_entropy)(model, xb, yb)

        is_tuple = lambda x: isinstance(x, tuple)
        stats, opt_treedef = jax.tree_flatten(opt_state[0], is_leaf=is_tuple)
        idx = opt_state[1]

        params, md_treedef = jax.tree_flatten(model)
        grads, _ = jax.tree_flatten(grads)

        leaves = [adam(p, g, *s, i=idx) for (p, g, s) in zip(params, grads, stats)]
        params, stats = zip(*leaves)

        model = md_treedef.unflatten(params)
        opt_state = (opt_treedef.unflatten(stats), idx + 1)

        return loss, model, opt_state

    xb, yb = next(train_dl)
    loss, model, opt_state = step(model, xb, yb, opt_state)
    print(loss)

    return model


def main():
    tconf = TrainerConfig(max_epoch=500, batch_size=512, lr=1e-3)
    text, codebook = process_dataset('data/input.txt', print_stats=False)

    mconf = GPTConfig(
        n_head=8, d_embd=256, n_layer=8,
        block_size=128, n_vocab=codebook.size
    )
    model = init_model(GPT(mconf), 39)

    train_ds, _ = train_test_split(codebook, text, mconf.block_size)
    train_dl = Dataloader(train_ds, tconf.batch_size)

    model = train(model, train_dl, tconf)

    # with open('ckpt_model.pkl', 'wb') as pkl_file:
    #     pickle.dump(model, pkl_file)

if __name__ == '__main__':
    main()
