Scipy spatial API for JAX

Installation
```
pip install .
```

Example Usage
```
import jax.numpy as jnp
import jax_scipy_spatial.transform as jtr

rotation = jtr.Rotation.from_euler('xyz', jnp.array([0., 0., 180.]), degrees=True)
print(rotation)
```
