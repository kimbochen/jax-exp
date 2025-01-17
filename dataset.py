import jax.numpy as jnp
from joblib import Memory


memory = Memory('/tmp/dataset_cache')


def make_collate_fn(text, batch_size, block_size, encode):
    batch_chunk_size = batch_size * (block_size + 1)
    def collate_fn(pos):
        chunk = text[pos:pos + batch_chunk_size]
        idxs = jnp.asarray([encode(char) for char in chunk])
        idxs = idxs.reshape(batch_size, block_size + 1)
        return idxs
    return collate_fn


class CharDataset:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.text = f.read()
        self.codebook = sorted(list(set(self.text)))
        self.vocab_size = len(self.codebook)
        self.encoder = {char: idx for idx, char in enumerate(self.codebook)}

    def dataloader(self, batch_size, block_size):
        batch_chunk_size = batch_size * (block_size + 1)
        n_trailing_char = len(self.text) % batch_chunk_size
        if n_trailing_char != 0:
            self.text = self.text[:-n_trailing_char]

        collate_fn = make_collate_fn(self.text, batch_size, block_size, self.encode)
        collate_fn = memory.cache(collate_fn)

        def load_data():
            for pos in range(0, len(self.text), batch_chunk_size):
                idxs = collate_fn(pos)
                yield idxs[:, :-1], idxs[:, 1:]

        return load_data

    def encode(self, char):
        return self.encoder[char]

    def decode(self, idx):
        return self.codebook[idx]
