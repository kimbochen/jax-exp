import jax
import jax.numpy as jnp
import jax.random as rand
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Module:
    def __init_subclass__(cls):
        register_pytree_node_class(cls)

    def __init__(self):
        _leaf_names, _static_names = [], []
        for name, value in self.__dict__.items():
            if isinstance(value, (Module, Sequential, Parameter)):
                _leaf_names.append(name)
            else:
                _static_names.append(name)
        self.leaf_names = _leaf_names
        self.static_names = _static_names

    def tree_flatten(self):
        static_fields = [self.__dict__[name] for name in self.static_names]
        leaves = [self.__dict__[name] for name in self.leaf_names]
        return leaves, (static_fields, self.static_names, self.leaf_names)

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        obj = cls.__new__(cls)
        static_fields, obj.static_names, obj.leaf_names = treedef

        for name, value in zip(obj.static_names, static_fields):
            object.__setattr__(obj, name, value)
        for name, value in zip(obj.leaf_names, leaves):
            object.__setattr__(obj, name, value)

        return obj


@register_pytree_node_class
class Sequential:
    def __init__(self, *modules):
        self.modules = modules
        for module in self.modules:
            assert isinstance(module, Module)

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def __repr__(self):
        module_reprs = '\n'.join([repr(m) for m in self.modules])
        return f'Sequential(\n{module_reprs}\n)'

    def tree_flatten(self):
        return self.modules, ()

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        return cls(*leaves)


class Parameter:
    def __init__(self, shape, method=None):
        self.shape = shape
        self.method = method

    def __call__(self, key):
        if self.method is None:
            return rand.normal(key, self.shape)
        else:
            return self.method(self.shape)

    def __repr__(self):
        return f'Parameter(shape={self.shape})'
