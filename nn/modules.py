import jax
import jax.numpy as jnp

from .core import Module, ModuleList, Parameter


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = ModuleList(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Linear(Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = Parameter([d_in, d_out])
        self.bias = Parameter([d_out,], jnp.zeros)

    def __call__(self, x):
        y = x @ self.weight + self.bias
        return y

class Embedding(Module):
    def __init__(self, n_embd, d_embd):
        super().__init__()
        self.embd = Parameter([n_embd, d_embd])

    def __call__(self, x):
        x = self.embd[x, :]
        return x

class LayerNorm(Module):
    def __init__(self, norm_shape):
        super().__init__()
        self.gamma = Parameter(norm_shape, jnp.ones)
        self.beta = Parameter(norm_shape, jnp.zeros)
        self.eps = 1e-5

    def __call__(self, x):
        u = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)

        x = (x - u) / jnp.sqrt(var + self.eps)
        x = x * self.gamma + self.beta

        return x

class Dropout(Module):
    def __init__(self, keep_rate):
        '''
        Dropout explained: https://stats.stackexchange.com/questions/205932
        '''
        super().__init__()
        self.p = keep_rate

    def __call__(self, x, state=None):
        # mask = rand.bernoulli(state, self.p, x.shape)
        # x = np.where(mask, x / self.p, 0.0)
        return x

class GeLU(Module):
    def __call__(self, x):
        return jax.nn.gelu(x)
