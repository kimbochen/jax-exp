import jax
import jax.numpy as jnp
import jax.random as rand
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Module:
    def __init_subclass__(cls):
        register_pytree_node_class(cls)

    def __init__(self):
        is_leaf = lambda m: isinstance(m, (Module, Parameter))
        get_elem = lambda m: jax.tree_flatten(m, is_leaf=is_leaf)[0][0]

        _leaf_names, _static_names = [], []
        for name, value in self.__dict__.items():
            if is_leaf(value) or is_leaf(get_elem(value)):
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


class Parameter:
    def __init__(self, shape, method=None):
        self.shape = shape
        self.method = method

    def __call__(self, key):
        if self.method is None:
            return rand.normal(key, self.shape)
        else:
            return self.method(self.shape)
