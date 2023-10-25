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

To run unit tests on your local machine:
```
tox
```
