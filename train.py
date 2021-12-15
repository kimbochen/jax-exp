import pickle
from dataclasses import dataclass

import jax
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


def cross_entropy(model, x, y):
    logit = model(x)
    logprob = -jax.nn.log_softmax(logit, axis=-1)
    logprob = logprob.reshape(-1, logprob.shape[-1])  # (B*T, n_vocab)
    loss = logprob[jnp.arange(logprob.shape[0]), y.reshape([-1, ])].mean()
    return loss


def train(model, train_dl, tconf):
    pass





def test():
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

    with open('ckpt_model.pkl', 'wb') as pkl_file:
        pickle.dump(model, pkl_file)

if __name__ == '__main__':
    test()
