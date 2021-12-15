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
If an attribute is of type `Module` or a trainable parameter, put its name into the list of leaf names.
Otherwise, put its name into the list of static names.

However, attributes are sometimes nested, e.g. a list of Transformer blocks.
Therefore, we need to break down the attribute object into a list of objects.
I implemented this with `jax.tree_flatten`,
assuming the list would contain elements of the same data type.
`jax.tree_flatten` breaks down PyTrees recursively, so we specify `is_leaf`,
telling the function to treat `Modules` and trainable parameter types as end leaves and stop.

## Initializing Parameters
