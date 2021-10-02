import re
import numpy as onp
import dataset_util as ds


def train_test_split(codebook, text, n_ctx):
    flatdata = onp.array([codebook.token2idx(token) for token in text])
    splits = [mo.end() for mo in re.finditer(r'\n\n|\. |; |: ',text)]
    starts = onp.concatenate([[0], splits])
    teststart = starts[int(len(starts) * 0.75)]
    chunksize = n_ctx + 1
    starts_train = starts[starts + chunksize <= teststart]
    starts_test = starts[starts + chunksize <= len(flatdata)]
    return (onp.array([flatdata[s : s+chunksize] for s in starts_train]),
        onp.array([flatdata[s : s+chunksize] for s in starts_test]))

def get_iterbatch(n_ctx, batch_size):
    text, codebook = ds.process_dataset('alice.txt')
    Xtr_bt, Xte_bt = train_test_split(codebook, text, n_ctx)
    iterbatch = ds.iterbatches(
        Xtr_bt, batch_size=batch_size, include_final_partial_batch=False
    )

    return iterbatch, codebook.size, Xtr_bt.shape[0]
