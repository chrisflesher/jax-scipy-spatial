# jax-scipy-spatial

This package implements `scipy.spatial` API for JAX.

Currently the following functions / classes are implemented:

- `scipy.spatial.transform.Rotation`
- `scipy.spatial.transform.Slerp`

Note that much of the code in this module may be too difficult to implement
properly in JAX, (e.g. nearest neighbor search). We request any submissions to
this repo be fully compatible with both `vmap` and `grad`.

## Install
```
pip install .
```

## Usage
```
import jax.numpy as jnp
import jax_scipy_spatial.transform as jtr

rotation = jtr.Rotation.from_euler('xyz', jnp.array([0., 0., 180.]), degrees=True)
print(rotation)
```

## Documentation
Please refer to scipy documentation.
