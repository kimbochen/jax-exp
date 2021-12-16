# Neural Network Library Design

This is the design documentation of the neural network library.  

## Concept

Because I am unfamiliar with the functional programming paradigm,
I decided to create a PyTorch-like API for my library.  
However, JAX follows the functional programming paradigm,
so I ran into the problem almost every JAX-based library faces:
**making an object-oriented API compatible with JAX**.

Inspired by [Equinox](https://github.com/patrick-kidger/equinox),
I converged on the solution that **represents neural networks as PyTrees**.  
By representing neural networks as PyTrees, we can use almost every JAX feature
without worrying about compatibility issues.

## PyTree

PyTree is a container class that is native to JAX.  
JAX requires a PyTree to support 2 operations: _flatten_ and _unflatten_.  

### Flatten

The _flatten_ operation flattens the PyTree into two elements:
- `leaves`: A list of leaf nodes.
- `treedef`: The definition of the PyTree.

To support JAX operations such as `jax.grad`, `leaves` contain data types native to JAX,
including the PyTree's subtrees.  
Everything else goes to `treedef`.

### Unflatten

The _unflatten_ operation reconstructs the PyTree based on information _flatten_ creates:
`leaves` and `treedef`.  
The invocation looks like this:
```python
treedef.unflatten(leaves)
```

## `Module` Class

Following the PyTorch API, we will design a `Module` class for neural networks and their layers to inherit.  
The `Module` class and its subclasses will be registered as PyTrees:
```python
@jax.tree_util.register_pytree_node_class
class Module:
    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)
```

### Defining a `Module` PyTree

I chose to make the leaves of a `Module` instance its **trainable parameters**
because it seems natural to do so and easier to manipulate the PyTree.  
Specifically, the leaves of a PyTree should be JAX arrays.  
On the other hand, the `treedef` contains:
- The names of the leaves
- The names of the non-leaf attributes (`static_names`)
- The values of the non-leaf attributes (`static_fields`)

As mentioned above, PyTrees need to support _flatten_ and _unflatten_ operations.  
_flatten_, defined in `tree_flatten`, filters out leaves and static fields
using a list of leaf names and a list of static names.  
_unflatten_, defined as a class method in `tree_unflatten`, reconstructs the object
by assigning the attributes according to the lists of names.

### Collecting the lists of attribute names

How do we collect the list of leaf names?  
After a module is instantiated, we iterate over its list of attributes (`self.__dict__`).  
If an attribute is of type `Module` or a trainable parameter, we put its name into the list of leaf names.
Otherwise, we put its name into the list of static names.

However, attributes are sometimes nested, e.g. a list of Transformer blocks.
Therefore, we need to break down the attribute object into a list of objects.
I implemented this with `jax.tree_flatten`,
assuming the list would contain elements of the same data type.
`jax.tree_flatten` breaks down PyTrees recursively, so we specify `is_leaf`,
telling the function to treat `Modules` and trainable parameter types as end leaves and stop.

## Initializing Random Parameters

Neural network parameters need to be initialized explicitly or randomly.  
In PyTorch, random states are implicitly defined,
so users do not need to worry about random states when calling functions that generate random numbers.  
However, random states are passed into functions explicitly in JAX.
Thus, we need a way to assign every parameter with a unique random state.  

### `Parameter` class

We provide a `Parameter` class to record all arguments for creating an array and defer the array creation.
Specifically, `Parameter` records 2 arguments:
- `shape`: The shape of ~~you~~ the array.
- `method`: The function that creates the array, which defaults to `jax.random.normal`.

### Initializing the parameters

Once a neural network is instantiated, we call `jax.treeflatten` to flatten_ the neural network.
By the design of the `Module` class, tree leaves would only contain `Parameter` objects.  
Given the list of _n_ `Parameter` objects, we can split a master key into _n_ subkeys with `jax.random.split`.
We can then pair up each `Parameter` object with one subkey to initialize an array.  
Finally, we reconstruct (unflatten) the neural network using `treedef` and the list of arrays.

## Stochastic Modules

Stochastic modules refer to modules whose outputs are dependent on a random state, e.g. Dropout.  
In JAX, functions have to be free of side effects in order to be JIT compiled.
This prohibits stochastic modules from keeping an attribute of a random state,
because **updating the state is considered modifying the module object, which violates the rule**.  

Unfortunately, I do not have time to find the solution to this problem, so here are some failed attempts:

### Create a new module with all the old attributes but a different random state

The JAX tracer still considers it as a side effect and raises an error.  
This makes me wonder why updating weights and reassinging variables are valid.  
**What is the criteria of being a tracer leak? How much change or what data types would be valid?**

### Making the random state a leaf node

`jax.grad` complains because random states are of type `uint32`.

### Updating the state outside of the JIT-compiled function

JAX raises XLA-related errors.

I ran out of ideas, so I chose the most unpleasant solution: passing random states into the model
with the input.

## Activation Modules

JAX provides common activation functions in `jax.nn`.  
By wrapping activation functions in `Module` classes, users can mix activation modules with other modules
in arbitrary containers (e.g. a list of layers).

Consider this example:
```python
class MLP(Module):
    def __init__(self):
        self.layers = [
            Linear(...),
            jax.nn.relu,
            Linear(...),
            jax.nn.relu
        ]
```
By the design of the `Module` class, `self.layers` would be broken down into a list of
objects with different types, violating our assumption (See [here](#collecting-the-list-of-attribute-names)).  
Thus, we wrap `jax.nn.relu` in a `Module` class:
```python
class ReLU(Module):
    def __call__(self, x):
        return jax.nn.relu(x)
```

## Helper Modules

Helper modules are refactored out of `model.py` to reduce the model complexity and mimic the PyTorch API.  
Here is the list of helper modules implemented:
- `Sequential`
- `Linear`
- `Embedding`
- `LayerNorm`
- `Dropout`
- `GeLU`
