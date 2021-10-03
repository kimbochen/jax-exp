import re
import joblib
import numpy as np


mem = joblib.Memory('/tmp/dataset_util')

@mem.cache
def get_data(filename):
    with open(filename) as file:
        text = file.read()
    chars = sorted(list(set(text)))
    return text, chars


# def get_dataloader(data, chars, batch_size, block_size):
#     stoi = {ch: i for i, ch in enumerate(chars)}
#     batch_char = batch_size * (block_size + 1)
# 
#     for idx in range(0, len(data)-batch_char, batch_char):
#         chunk = data[idx:idx + batch_char]
#         idx = [stoi[ch] for ch in chunk]
#         xyb = np.asarray(idx).reshape([batch_size, block_size + 1])
#         yield xyb[:, :-1], xyb[:, 1:]

def get_dataloader(data, chars, batch_size, block_size):
    stoi = {ch: i for i, ch in enumerate(chars)}
    block_size += 1

    init_idx = [m.end() for m in re.finditer(r'\n\n|\. |; |: ', data)]
    init_idx = np.concatenate([[0], init_idx])
    block_idx = init_idx[init_idx + block_size < len(data)]
    batch_idx = block_idx[:-(block_idx.size % batch_size)]

    def create_xy():
        for idx in batch_idx:
            chunk = data[idx:idx + block_size]
            xy = np.asarray([stoi[ch] for ch in chunk]).reshape(1, -1)
            yield xy
    xy = np.concatenate([xy for xy in create_xy()])
    xy_batch = xy.reshape(-1, batch_size, block_size)

    for xyb in xy_batch:
        yield xyb[:, :-1], xyb[:, 1:]


if __name__ == '__main__':
    text, chars = get_data('alice.txt')
    dl = get_dataloader(text, chars, 64, 128)
    itos = {i: ch for i, ch in enumerate(chars)}
    to_str = lambda z: ''.join([itos[ch] for ch in z.flatten()])
    x, y = next(dl)
    print(x.shape, y.shape)

    print([to_str(x[i]) for i in range(64)])
