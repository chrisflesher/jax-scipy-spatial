# Copyright 2023 Chris Flesher.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import typing

import scipy.spatial.distance

import jax
import jax.numpy as jnp
from jax._src.numpy.util import _wraps


@_wraps(scipy.spatial.distance.chebyshev)
def chebyshev(u: jax.Array, v: jax.Array, w: typing.Optional[jax.Array] = None) -> jax.Array:
  """Compute the Chebyshev distance."""
  l1_diff = jnp.abs(u - v)
  if w is not None:
    l1_diff = jnp.where(w > 0, l1_diff, -jnp.inf)
  return jnp.max(l1_diff)


@_wraps(scipy.spatial.distance.cityblock)
def cityblock(u: jax.Array, v: jax.Array, w: typing.Optional[jax.Array] = None) -> jax.Array:
  """Compute the City Block (Manhattan) distance."""
  l1_diff = jnp.abs(u - v)
  if w is not None:
    l1_diff = w * l1_diff
  return jnp.sum(l1_diff)


@_wraps(scipy.spatial.distance.euclidean)
def euclidean(u: jax.Array, v: jax.Array, w: typing.Optional[jax.Array] = None) -> jax.Array:
  """Computes the Euclidean distance between two 1-D arrays."""
  return minkowski(u, v, p=2, w=w)


@_wraps(scipy.spatial.distance.hamming)
def hamming(u: jax.Array, v: jax.Array, w: typing.Optional[jax.Array] = None) -> jax.Array:
    """Compute the Hamming distance between two 1-D arrays."""
    return jnp.average(u != v, weights=w)


@_wraps(scipy.spatial.distance.euclidean)
@functools.partial(jax.jit, static_argnames=['p'])
def minkowski(u: jax.Array, v: jax.Array, p: int = 2, w: typing.Optional[jax.Array] = None) -> jax.Array:
  """Compute the Minkowski distance between two 1-D arrays."""
  u_v = u - v
  if w is not None:
    if p == 1:
      root_w = w
    elif p == 2:
      root_w = jnp.sqrt(w)
    elif p == jnp.inf:
      root_w = (w != 0)
    else:
      root_w = jnp.power(w, 1/p)
    u_v = root_w * u_v
  dist = jnp.linalg.norm(u_v, ord=p)
  return dist


@_wraps(scipy.spatial.distance.sqeuclidean)
def sqeuclidean(u: jax.Array, v: jax.Array, w: typing.Optional[jax.Array] = None) -> jax.Array:
  """Compute the squared Euclidean distance between two 1-D arrays."""
  u_v = u - v
  u_v_w = u_v
  if w is not None:
    u_v_w = w * u_v
  return jnp.dot(u_v, u_v_w)
