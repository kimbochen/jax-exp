import jax
import jax.numpy as jnp

from .core import Module, Parameter


class Sequential(Module):
    def __init__(self, *modules):
        assert all([isinstance(m, Module) for m in modules])
        self.modules = modules
        super().__init__()

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def __repr__(self):
        reprs = '\n  '.join([repr(m) for m in self.modules])
        return f'Sequential(modules= \n  {reprs}\n)'


class Linear(Module):
    def __init__(self, d_in, d_out):
        self.weight = Parameter([d_in, d_out])
        self.bias = Parameter([d_out,], jnp.zeros)
        super().__init__()

    def __call__(self, x):
        y = x @ self.weight + self.bias
        return y

    def __repr__(self):
        return f'Linear(weight:{self.weight.shape}, bias:{self.bias.shape})'


class Embedding(Module):
    def __init__(self, n_embd, d_embd):
        self.embd = Parameter([n_embd, d_embd])
        super().__init__()

    def __call__(self, x):
        x = self.embd[x, :]
        return x

    def __repr__(self):
        return f'Embedding(embd:{self.embd.shape})'


class LayerNorm(Module):
    def __init__(self, norm_shape):
        self.gamma = Parameter(norm_shape, jnp.ones)
        self.beta = Parameter(norm_shape, jnp.zeros)
        self.eps = 1e-5
        super().__init__()

    def __call__(self, x):
        u = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)

        x = (x - u) / jnp.sqrt(var + self.eps)
        x = x * self.gamma + self.beta

        return x

    def __repr__(self):
        return f'LayerNorm(gamma:{self.gamma.shape}, beta:{self.beta.shape})'


class Dropout(Module):
    def __init__(self, keep_rate):
        '''
        Dropout explained: https://stats.stackexchange.com/questions/205932
        '''
        self.p = keep_rate
        super().__init__()

    def __call__(self, x, state=None):
        # mask = rand.bernoulli(state, self.p, x.shape)
        # x = np.where(mask, x / self.p, 0.0)
        return x

    def __repr__(self):
        return f'Dropout(keep_rate:{self.p})'


class GeLU(Module):
    def __call__(self, x):
        return jax.nn.gelu(x)

    def __repr__(self):
        return 'GeLU'
