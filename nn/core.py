import typing as tp
from abc import ABC
from dataclasses import dataclass
from typing import Callable, List

import jax
import jax.numpy as jnp
import jax.random as rand
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Module:
    def __init_subclass__(cls):
        register_pytree_node_class(cls)

    def init(self, seed):
        is_leaf = lambda x: isinstance(x, (Module, ModuleList, Parameter))
        _leaf_names, _static_names = [], []

        for name, value in self.__dict__.items():
            (obj, *_), _ = jax.tree_flatten(value, is_leaf=is_leaf)
            if is_leaf(obj):
                _leaf_names.append(name)
            else:
                _static_names.append(name)
        self.leaf_names = _leaf_names
        self.static_names = _static_names

        keys = rand.split(seed, len(self.leaf_names))
        for name, key in zip(self.leaf_names, keys):
            value = self.__dict__[name]
            object.__setattr__(self, name, value.init(key))

        return self

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
class ModuleList:
    def __init__(self, modules):
        self.modules = modules

    def __iter__(self):
        for module in self.modules:
            yield module

    def init(self, seed):
        self.activation_idx = []
        keys = rand.split(seed, len(self.modules))
        for module, key in zip(self.modules, keys):
            module.init(key)
        return self

    def tree_flatten(self):
        static_fields = [self.modules[idx] for idx in self.activation_idx]
        leaves = [
            module for idx, module in enumerate(self.modules)
            if idx not in self.activation_idx
        ]
        return leaves, (static_fields, self.activation_idx)

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        obj = cls.__new__(cls)
        static_fields, obj.activation_idx = treedef
        obj.modules = [
            act if idx in obj.activation_idx else mod
            for idx, (act, mod) in enumerate(zip(static_fields, leaves))
        ]
        return obj


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
