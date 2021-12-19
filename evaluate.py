import pickle

import jax
import jax.numpy as jnp
import tqdm

from dataset import CharDataset


def evaluate(model, char_ds, ctx):
    block_size = model.block_size
    x = jnp.asarray([char_ds.encode(char) for char in ctx]).reshape(1, -1)

    for _ in tqdm.trange(300):
        x_cond = x if x.shape[1] <= block_size else x[:, -block_size:]
        logit = model(x_cond)
        prob = jax.nn.softmax(logit[:, -1, :], axis=-1)
        x = jnp.append(x, jnp.argmax(prob)).reshape(1, -1)

    response = ''.join([char_ds.decode(idx) for idx in x.flatten()])
    print(response)


def main(args):
    with open('ckpt_model.pkl', 'rb') as f:
        model = pickle.load(f)
    char_ds = CharDataset('data/input.txt')
    evaluate(model, char_ds, args.context)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--context', type=str, default='O Lord have mercy')
    args = parser.parse_args()

    main(args)
