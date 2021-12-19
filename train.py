import pickle
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as rand
import tqdm

from dataset import CharDataset
from model import GPT, GPTConfig


@dataclass
class TrainConfig:
    max_epoch: int
    batch_size: int
    lr: float
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8


def init_layer(layer, seed):
    params, treedef = jax.tree_flatten(layer)
    keys = rand.split(seed, len(params))
    return treedef.unflatten(p(k) for p, k in zip(params, keys))


def init_adam(tconf):
    def update_model(model, opt_state):
        mu, var, t = opt_state
        def adam_update(param, mu, var):
            m_hat = mu / (1.0 - tconf.b1 ** t)
            v_hat = var / (1.0 - tconf.b2 ** t)
            param = param - tconf.lr * m_hat / (jnp.sqrt(v_hat) + tconf.eps)
            return param
        return jax.tree_map(adam_update, model, mu, var)

    def update_opt_state(grads, opt_state):
        mu, var, t = opt_state

        update_mu = lambda mu, grad: (1.0 - tconf.b1) * grad + tconf.b1 * mu
        mu = jax.tree_map(update_mu, mu, grads)

        update_var = lambda var, grad: (1.0 - tconf.b2) * jnp.square(grad) + tconf.b2 * var
        var = jax.tree_map(update_var, var, grads)

        return mu, var, t + 1

    def init_opt_state(model):
        mu = jax.tree_map(lambda p: jnp.zeros_like(p), model)
        var = jax.tree_map(lambda p: jnp.zeros_like(p), model)
        return mu, var, 0

    return update_model, update_opt_state, init_opt_state


def cross_entropy(model, x, y):
    logit = model(x)
    logprob = -jax.nn.log_softmax(logit, axis=-1)
    logprob = logprob.reshape(-1, logprob.shape[-1])  # (B*T, n_vocab)
    loss = logprob[jnp.arange(logprob.shape[0]), y.reshape([-1, ])].mean()
    return loss


def train(model, train_dl, tconf):
    update_model, update_opt_state, init_opt_state = init_adam(tconf)
    opt_state = init_opt_state(model)

    @jax.jit
    def step(model, xb, yb, opt_state):
        loss, grads = jax.value_and_grad(cross_entropy)(model, xb, yb)
        opt_state = update_opt_state(grads, opt_state)
        model = update_model(model, opt_state)
        return loss, model, opt_state

    pbar = tqdm.trange(tconf.max_epoch)
    for epoch in pbar:
        losses = []
        for xb, yb in train_dl():
            loss, model, opt_state = step(model, xb, yb, opt_state)
            losses.append(loss)
        loss = jnp.asarray(losses).mean()
        pbar.set_description(f'Epoch {epoch} loss: {loss:.4f}')

    return model


def main():
    char_ds = CharDataset('data/input.txt')
    mconf = GPTConfig(
        d_embd=128, n_head=4, n_layer=4, block_size=64,
        n_vocab=char_ds.vocab_size
    )
    tconf = TrainConfig(max_epoch=1000, batch_size=256, lr=1e-3)

    model = init_layer(GPT(mconf), rand.PRNGKey(39))
    train_dl = char_ds.dataloader(tconf.batch_size, mconf.block_size)
    model = train(model, train_dl, tconf)

    with open('ckpt_model.pkl', 'wb') as file:
        pickle.dump(model, file)

if __name__ == '__main__':
    main()
