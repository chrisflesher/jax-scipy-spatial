# jax-scipy-spatial

This package implements `scipy.spatial` API for JAX.

Currently the following items are implemented:

- `scipy.spatial.distance`
- `scipy.spatial.transform.Rotation`
- `scipy.spatial.transform.Slerp`

## Install
```
pip install .
```

## Usage
```
import jax_scipy_spatial.transform as jtr

rotation = jtr.Rotation.from_euler('xyz', jnp.array([0., 0., 180.]), degrees=True)
print(rotation)
```

## Documentation
Please refer to scipy documentation.

## Contributing

Note that much of the code in this module may be difficult to implement
properly in JAX, (e.g. nearest neighbor search). We request any submissions to
this repo be fully compatible with both `vmap` and `grad`.

To run unit tests on your local machine:
```
tox
```
