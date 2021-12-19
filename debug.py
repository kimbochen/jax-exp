import jax
import jax.numpy as jnp
import jax.random as rand
import tqdm

from dataset import CharDataset
from model import GPT, GPTConfig
from train import TrainConfig, init_layer, init_adam, cross_entropy


char_ds = CharDataset('data/input.txt')
mconf = GPTConfig(d_embd=128, n_head=4, n_layer=4, block_size=64, n_vocab=char_ds.vocab_size)
tconf = TrainConfig(max_epoch=300, batch_size=1, lr=1e-3)

model = init_layer(GPT(mconf), rand.PRNGKey(39))
train_dl = char_ds.dataloader(tconf.batch_size, mconf.block_size)

xb, yb = next(train_dl())
update_model, update_opt_state, init_opt_state = init_adam(tconf)
opt_state = init_opt_state(model)

@jax.jit
def step(model, xb, yb, opt_state):
    loss, grads = jax.value_and_grad(cross_entropy)(model, xb, yb)
    opt_state = update_opt_state(grads, opt_state)
    model = update_model(model, opt_state)
    return loss, model, opt_state

decode_str = lambda z, idx: ''.join([char_ds.decode(char) for char in z[idx, :]])

for n_iter in range(tconf.max_epoch):
    loss, model, opt_state = step(model, xb, yb, opt_state)
    if (n_iter + 1) % (tconf.max_epoch // 5) == 0:
        logits = model(xb)
        probs = jax.nn.softmax(logits, axis=-1)
        preds = jnp.argmax(probs, axis=-1)
        print(f'n_iter {n_iter} loss: {loss:.4f} pred:\n--------------\n{decode_str(preds, 0)}\n==============\n')
