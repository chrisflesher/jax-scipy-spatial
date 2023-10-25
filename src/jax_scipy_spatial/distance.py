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


@_wraps(scipy.spatial.distance.braycurtis)
def braycurtis(u: jax.Array, v: jax.Array, w: typing.Optional[jax.Array] = None) -> jax.Array:
  """Compute the Bray-Curtis distance between two 1-D arrays."""
  l1_diff = jnp.abs(u - v)
  l1_sum = jnp.abs(u + v)
  if w is not None:
    l1_diff = w * l1_diff
    l1_sum = w * l1_sum
  return jnp.sum(l1_diff) / jnp.sum(l1_sum)


@_wraps(scipy.spatial.distance.canberra)
def canberra(u: jax.Array, v: jax.Array, w: typing.Optional[jax.Array] = None) -> jax.Array:
  """Compute the Canberra distance between two 1-D arrays."""
  l1_diff = jnp.abs(u - v)
  abs_u = jnp.abs(u)
  abs_v = jnp.abs(v)
  d = l1_diff / (abs_u + abs_v)
  if w is not None:
    d = w * d
  return jnp.nansum(d)


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


@_wraps(scipy.spatial.distance.correlation)
def correlation(u: jax.Array, v: jax.Array, w: typing.Optional[jax.Array] = None, centered: bool = True) -> jax.Array:
  """Compute the correlation distance between two 1-D arrays."""
  if centered:
    umu = jnp.average(u, weights=w)
    vmu = jnp.average(v, weights=w)
    u = u - umu
    v = v - vmu
  uv = jnp.average(u * v, weights=w)
  uu = jnp.average(jnp.square(u), weights=w)
  vv = jnp.average(jnp.square(v), weights=w)
  dist = 1.0 - uv / jnp.sqrt(uu * vv)
  return jnp.abs(dist)


@_wraps(scipy.spatial.distance.cosine)
def cosine(u: jax.Array, v: jax.Array, w: typing.Optional[jax.Array] = None) -> jax.Array:
  """Compute the Cosine distance between 1-D arrays."""
  return jnp.clip(correlation(u, v, w=w, centered=False), 0.0, 2.0)


@_wraps(scipy.spatial.distance.euclidean)
def euclidean(u: jax.Array, v: jax.Array, w: typing.Optional[jax.Array] = None) -> jax.Array:
  """Computes the Euclidean distance between two 1-D arrays."""
  return minkowski(u, v, p=2, w=w)


@_wraps(scipy.spatial.distance.hamming)
def hamming(u: jax.Array, v: jax.Array, w: typing.Optional[jax.Array] = None) -> jax.Array:
  """Compute the Hamming distance between two 1-D arrays."""
  u_ne_v = u != v
  if w is not None:
    u_ne_v = u_ne_v.astype(w)
  return jnp.average(u_ne_v, weights=w)


@_wraps(scipy.spatial.distance.jaccard)
def jaccard(u, v, w=None):
  """Compute the Jaccard-Needham dissimilarity between two boolean 1-D arrays."""
  nonzero = jnp.bitwise_or(u != 0, v != 0)
  unequal_nonzero = jnp.bitwise_and((u != v), nonzero)
  if w is not None:
    nonzero = jnp.where(nonzero, w, 0.)
    unequal_nonzero = jnp.where(unequal_nonzero, w, 0.)
  a = jnp.sum(unequal_nonzero)
  b = jnp.sum(nonzero)
  return jnp.where(b != 0, a / b, 0)


@_wraps(scipy.spatial.distance.minkowski)
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
      root_w = (w != 0).astype(w)
    else:
      root_w = jnp.power(w, 1/p)
    u_v = root_w * u_v
  dist = jnp.linalg.norm(u_v, ord=p)
  return dist


@_wraps(scipy.spatial.distance.russellrao)
def russellrao(u: jax.Array, v: jax.Array, w: typing.Optional[jax.Array] = None) -> jax.Array:
  """Compute the Russell-Rao dissimilarity between two boolean 1-D arrays."""
  if u.dtype == v.dtype == bool and w is None:
    ntt = jnp.sum(u & v)
    n = u.size
  elif w is None:
    ntt = jnp.sum(u * v)
    n = u.size
  else:
    ntt = jnp.sum(u * v * w)
    n = jnp.sum(w)
  return (n - ntt) / n


@_wraps(scipy.spatial.distance.sqeuclidean)
def sqeuclidean(u: jax.Array, v: jax.Array, w: typing.Optional[jax.Array] = None) -> jax.Array:
  """Compute the squared Euclidean distance between two 1-D arrays."""
  u_v = u - v
  u_v_w = u_v
  if w is not None:
    u_v_w = w * u_v
  return jnp.dot(u_v, u_v_w)
