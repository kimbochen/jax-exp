import pickle

import jax
import jax.numpy as jnp
import tqdm

from dataset_util import process_dataset


def evaluate(model, ctx, block_size, codebook):
    x = jnp.asarray(codebook.encode(ctx)).reshape(1, -1)

    for _ in tqdm.trange(300):
        x_cond = x if x.shape[1] <= block_size else x[:, -block_size:]
        logit = model(x_cond)
        prob = jax.nn.softmax(logit[:, -1, :], axis=-1)
        x = jnp.append(x, jnp.argmax(prob)).reshape(1, -1)

    response = ''.join([codebook.idx2token(idx) for idx in x.flatten()])
    print(response)


def main(args):
    with open('ckpt_model.pkl', 'rb') as f:
        model = pickle.load(f)
    _, codebook = process_dataset('data/input.txt', print_stats=False)
    evaluate(model, args.prompt, model.block_size, codebook)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--prompt', type=str, default="O'Lord have mercy")
    args = parser.parse_args()

    main(args)
