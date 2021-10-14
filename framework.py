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

    def init(self, key):
        is_model = isinstance(key, int)
        key = rand.PRNGKey(key) if is_model else key
        self.static_name = []

        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                obj, key = value.init(key)
                object.__setattr__(self, name, obj)
            elif isinstance(value, (Module, ModuleList)):
                key = value.init(key)
            else:
                self.static_name.append(name)

        return (self, key) if is_model else key

    def tree_flatten(self):
        static_field = [self.__dict__[name] for name in self.static_name]
        is_dyn = lambda name: name not in self.static_name
        dynamic_name = list(filter(is_dyn, self.__dict__.keys()))
        dynamic_field = [self.__dict__[name] for name in dynamic_name]
        return dynamic_field, (static_field, self.static_name, dynamic_name)

    @classmethod
    def tree_unflatten(cls, aux, dynamic_field):
        obj = cls.__new__(cls)
        static_field, obj.static_name, dynamic_name = aux

        for name, value in zip(obj.static_name, static_field):
            object.__setattr__(obj, name, value)
        for name, value in zip(dynamic_name, dynamic_field):
            object.__setattr__(obj, name, value)

        return obj


@register_pytree_node_class
@dataclass
class ModuleList:
    modules: List[Module]

    def __iter__(self):
        for mod in self.modules:
            yield mod

    def init(self, key):
        self.activation_idx = []

        for idx, module in enumerate(self.modules):
            if isinstance(module, Module):
                key = module.init(key)
            elif isinstance(module, Callable):
                self.activation_idx.append(idx)
            else:
                raise ValueError(f'Unexpected data type {type(module)} in ModuleList.')

        return key

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
        key, _ = rand.split(key)
        return self.data, key

@dataclass
class RandParam(Parameter):
    shape: tuple
    method: Callable = rand.normal

    def init(self, key):
        data = self.method(key, self.shape)
        key, _ = rand.split(key)
        return data, key
