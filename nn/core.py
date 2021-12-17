import typing as tp
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as rand
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Module:
    def __init_subclass__(cls):
        register_pytree_node_class(cls)

    def init(self, seed):
        is_leaf = lambda x: isinstance(x, (Module, Parameter))
        _leaf_names, _static_names = [], []

        for name, value in self.__dict__.items():
            (obj, *_), _ = jax.tree_flatten(value, is_leaf=is_leaf)
            if is_leaf(obj):
                _leaf_names.append(name)
            else:
                _static_names.append(name)
        self._leaf_names = _leaf_names
        self._static_names = _static_names

        keys = rand.split(seed, len(self.leaf_names))
        for name, key in zip(self.leaf_names, keys):
            value = self.__dict__[name]
            object.__setattr__(self, name, value.init(key))

        return self

    def tree_flatten(self):
        leaves = [self.__dict__[name] for name in self.leaf_names]
        static_fields = [self.__dict__[name] for name in self.static_names]
        return leaves, [self.leaf_names, self.static_names, static_fields]

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        obj = cls.__new__(cls)
        obj._leaf_names, obj._static_names, static_fields = treedef

        def set_obj_attrs(names, values):
            for name, value in zip(names, values):
                object.__setattr__(obj, name, value)
        set_obj_attrs(obj.leaf_names, leaves)
        set_obj_attrs(obj.static_names, static_fields)

        return obj

    @property
    def leaf_names(self):
        return self._leaf_names

    @property
    def static_names(self):
        return self._static_names


@register_pytree_node_class
class ModuleList:
    def __init__(self, *modules):
        self.modules = list(modules)

    def __iter__(self):
        for module in self.modules:
            yield module

    def init(self, seed):
        keys = rand.split(seed, len(self.modules))
        for module, key in zip(self.modules, keys):
            module.init(key)
        return self

    def tree_flatten(self):
        return self.modules, []

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        return cls(*leaves)


@dataclass
class Parameter:
    shape: tp.List[int]
    method: tp.Optional[tp.Callable] = None

    def __repr__(self):
        return f'Parameter(shape={self.shape})'

    def init(self, key):
        if self.method is None:
            return rand.normal(key, self.shape)
        else:
            return self.method(self.shape, dtype=jnp.float32)
