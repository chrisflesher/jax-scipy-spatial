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

import jax
from jax_scipy_spatial.distance import minkowski


class KNN:
  """K-dimensional nearest-neighbor lookup.

  This class provides an index into a set of k-dimensional points which can
  be used to rapidly look up the nearest neighbors of any point.
  """

  def __init__(self, data: jax.Array):
    """Initialize."""
    self._data = data

  def query(self, x: jax.Array, k: int = 1, recall_target: float = 1., p: float = 2.,) -> typing.Tuple[jax.Array, jax.Array]:
    """Query for nearest neighbors."""
    return _knn(data=self._data, x=x, k=k, recall_target=recall_target, p=p)


@functools.partial(jax.jit, static_argnames=['k', 'recall_target', 'p'])
def _knn(data: jax.Array, x: jax.Array, k: int, recall_target: float, p: float) -> typing.Tuple[jax.Array, jax.Array]:
  distances = minkowski(data, x, p)  # brute force N^2...
  import pdb; pdb.set_trace()
  return jax.lax.approx_min_k(distances, k, recall_target=recall_target)
