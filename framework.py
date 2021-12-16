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

    def init(self):
        is_leaf = lambda x: isinstance(x, (Module, Parameter))
        _leaf_names, _static_names = [], []
        for name, value in self.__dict__.items():
            (obj, *_), _ = jax.tree_flatten(value, is_leaf=is_leaf)
            if is_leaf(obj):
                _leaf_names.append(name)
            else:
                _static_names.append(name)
        self.leaf_name = _leaf_names
        self.static_name = _static_names

    def tree_flatten(self):
        static_field = [self.__dict__[name] for name in self.static_name]
        dynamic_field = [self.__dict__[name] for name in self.leaf_name]
        return dynamic_field, (static_field, self.static_name, self.leaf_name)

    @classmethod
    def tree_unflatten(cls, aux, dynamic_field):
        obj = cls.__new__(cls)
        static_field, obj.static_name, obj.leaf_name = aux

        for name, value in zip(obj.static_name, static_field):
            object.__setattr__(obj, name, value)
        for name, value in zip(obj.leaf_name, dynamic_field):
            object.__setattr__(obj, name, value)

        return obj


@register_pytree_node_class
@dataclass
class ModuleList:
    modules: List[Module]

    def __iter__(self):
        for mod in self.modules:
            yield mod

    def init(self, seed):
        keys = rand.split(seed, len(self.modules))
        self.activation_idx = []

        for idx, (module, key) in enumerate(zip(self.modules, keys)):
            if isinstance(module, Module):
                module.init(key)
            elif isinstance(module, Callable):
                self.activation_idx.append(idx)
            else:
                raise ValueError(f'Unexpected data type {type(module)} in ModuleList.')

        return self

    def tree_flatten(self):
        static_field = [self.modules[idx] for idx in self.activation_idx]
        dynamic_field = [
            module for idx, module in enumerate(self.modules)
            if idx not in self.activation_idx
        ]
        return dynamic_field, (static_field, self.activation_idx)

    @classmethod
    def tree_unflatten(cls, aux, dynamic_field):
        obj = cls.__new__(cls)
        static_field, obj.activation_idx = aux
        obj.modules = [
            act if idx in obj.activation_idx else mod
            for idx, (act, mod) in enumerate(zip(static_field, dynamic_field))
        ]
        return obj

class Parameter(ABC):
    pass

@dataclass
class Param(Parameter):
    data: jnp.ndarray

    def init(self, key):
        return self.data

@dataclass
class RandParam(Parameter):
    shape: tuple
    method: Callable = rand.normal

    def init(self, key):
        data = self.method(key, self.shape)
        return data
