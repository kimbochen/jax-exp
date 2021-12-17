import pickle
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as rand
import tqdm

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


def init_adam(tconf):
    def adam(param, grad, mu, var, i):
        mu = (1.0 - tconf.b1) * grad + tconf.b1 * mu
        var = (1.0 - tconf.b2) * jnp.square(grad) + tconf.b2 * var

        m_hat = mu / (1.0 - tconf.b1 ** i)
        v_hat = var / (1.0 - tconf.b2 ** i)
        param = param - tconf.lr * m_hat / (jnp.sqrt(v_hat) + tconf.eps)

        return param, mu, var

    def init_opt_state(model):
        mu = jax.tree_map(lambda p: jnp.zeros_like(p), model)
        var = jax.tree_map(lambda p: jnp.zeros_like(p), model)
        return mu, var, 1

    def split_tree(opt_tree, idx):
        is_tuple = lambda x: isinstance(x, tuple)
        tree = jax.tree_map(lambda t: t[idx], opt_tree, is_leaf=is_tuple)
        return tree

    return adam, init_opt_state, split_tree

def cross_entropy(model, x, y):
    logit = model(x)
    logprob = -jax.nn.log_softmax(logit, axis=-1)
    logprob = logprob.reshape(-1, logprob.shape[-1])  # (B*T, n_vocab)
    loss = logprob[jnp.arange(logprob.shape[0]), y.reshape([-1, ])].mean()
    return loss


def train(model, train_dl, tconf):
    adam_i, init_opt_state, split_tree = init_adam(tconf)
    opt_state = init_opt_state(model)

    @jax.jit
    def step(model, xb, yb, opt_state):
        loss, grads = jax.value_and_grad(cross_entropy)(model, xb, yb)
        mu, var, idx = opt_state
        adam = partial(adam_i, i=idx)
        opt_tree = jax.tree_map(adam, model, grads, mu, var)
        model, mu, var = (split_tree(opt_tree, i) for i in range(3))
        return loss, model, (mu, var, idx + 1)

    pbar = tqdm.trange(tconf.max_epoch)
    for epoch in pbar:
        losses = []
        for xb, yb in train_dl:
            loss, model, opt_state = step(model, xb, yb, opt_state)
            losses.append(loss)
        loss = jnp.asarray(losses).mean()
        pbar.set_description(f'Epoch {epoch} loss: {loss:.4f}')

    return model


def main():
    tconf = TrainerConfig(max_epoch=1000, batch_size=512, lr=3e-4)
    text, codebook = process_dataset('data/input.txt', print_stats=False)

    mconf = GPTConfig(
        n_head=8, d_embd=256, n_layer=8,
        block_size=128, n_vocab=codebook.size
    )
    model = GPT(mconf).init(rand.PRNGKey(39))

    train_ds, _ = train_test_split(codebook, text, mconf.block_size)
    train_dl = Dataloader(train_ds, tconf.batch_size)

    model = train(model, train_dl, tconf)

    with open('ckpt_model.pkl', 'wb') as pkl_file:
        pickle.dump(model, pkl_file)

if __name__ == '__main__':
    main()
