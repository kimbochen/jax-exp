import collections
import io
import re
from functools import partial

import joblib
import numpy as np


class TrainDataloader:
    def __init__(self, filename, block_size, batch_size):
        text, self._codebook = process_dataset(filename, print_stats=False)
        train_batch, _ = train_test_split(self.codebook, text, block_size)
        self.iterbatch = iterbatches(train_batch, batch_size=batch_size)

    def __iter__(self):
        for (batch,) in self.iterbatch:
            xb, yb = batch[:, :-1], batch[:, 1:]
            yield xb, yb

    def __next__(self):
        (batch,) = next(self.iterbatch)
        xb, yb = batch[:, :-1], batch[:, 1:]
        return xb, yb

    @property
    def codebook(self):
        return self._codebook


class Codebook(object):
    def __init__(self, tokens):
        self.tokens = tokens

    def token2idx(self, token):
        return self.tokens.index(token)

    def idx2token(self, idx):
        return self.tokens[idx]

    def encode(self, text):
        return [self.token2idx(token) for token in text]

    @property
    def size(self):
        return len(self.tokens)


mem = joblib.Memory('/tmp/dataset_util')


@mem.cache
def make_codebook(text):
    all_chars = list(sorted(set(text)))
    codebook = Codebook(all_chars)
    return codebook


@mem.cache
def get_zip_ratio(text):
    import zlib
    text = text.encode()
    smalltext = zlib.compress(text, level=-1)
    ratio = len(smalltext) / len(text)
    return ratio


def process_dataset(text_file, print_stats=True):
    with io.open(text_file, encoding='utf-8') as f:
        text = f.read().strip()
    codebook = make_codebook(text)
    if print_stats:
        token2count = collections.Counter(text)
        counts = np.array([token2count[c] for c in codebook.tokens])
        probs = counts / counts.sum()
        print(tabulate(zip(map(repr, codebook.tokens), probs, map(int, counts)),
                       headers=['tokens', 'probs', 'counts'], floatfmt='.3e'))
        zipratio = get_zip_ratio(text)
        print(tabulate([
            ('Marg ent', (probs * np.log(1 / probs)).sum()),
            ('Zip', zipratio * np.log(256))
        ]))
    return text, codebook

def iterbatches(*arrays, num_batches=None, batch_size=None, shuffle=False, include_final_partial_batch=False):
    assert (num_batches is None) != (batch_size is None), 'Provide num_batches or batch_size, but not both'
    arrays = tuple(map(np.asarray, arrays))
    n = arrays[0].shape[0]
    assert all(a.shape[0] == n for a in arrays[1:])
    inds = np.arange(n)
    if shuffle: np.random.shuffle(inds)
    sections = np.arange(0, n, batch_size)[1:] if num_batches is None else num_batches
    for batch_inds in np.array_split(inds, sections):
        if include_final_partial_batch or len(batch_inds) == batch_size:
            yield tuple(a[batch_inds] for a in arrays)


def train_test_split(codebook, text, n_ctx):
    # TODO start at every character
    flatdata = np.array([codebook.token2idx(token) for token in text])
    splits = [mo.end() for mo in re.finditer(r'\n\n|\. |; |: ',text)]
    starts = np.concatenate([[0], splits])
    teststart = starts[int(len(starts) * 0.75)]
    chunksize = n_ctx + 1
    starts_train = starts[starts + chunksize <= teststart]
    starts_test = starts[starts + chunksize <= len(flatdata)]
    return (np.array([flatdata[s : s+chunksize] for s in starts_train]),
        np.array([flatdata[s : s+chunksize] for s in starts_test]))


if __name__ == '__main__':
    train_dl = TrainDataloader('data/input.txt', block_size=128, batch_size=2)
    for xb, yb in train_dl:
        print(xb, yb, sep='\n\n')
        break

    print('-' * 39)

    train_dl = TrainDataloader('data/input.txt', block_size=128, batch_size=2)
    xb, by = next(train_dl)
    print(xb, yb, sep='\n\n')
