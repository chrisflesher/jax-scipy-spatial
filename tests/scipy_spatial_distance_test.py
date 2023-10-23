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

from absl.testing import absltest

import jax

import scipy.version
from jax._src import test_util as jtu
import jax_scipy_spatial.distance as jsp_distance
import scipy.spatial.distance as osp_distance

import jax.numpy as jnp
import numpy as onp
from jax.config import config

config.parse_flags_with_absl()

scipy_version = tuple(map(int, scipy.version.version.split('.')[:3]))

float_dtypes = jtu.dtypes.floating
real_dtypes = float_dtypes + jtu.dtypes.integer + jtu.dtypes.boolean

num_samples = 2

class LaxBackedScipySpatialDistanceTests(jtu.JaxTestCase):
  """Tests for LAX-backed scipy.spatial.distance implementations."""

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(2,)],
  )
  def testEuclidean(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(shape, dtype))
    jnp_fn = lambda u, v: jsp_distance.euclidean(u, v)
    np_fn = lambda u, v: osp_distance.euclidean(u, v)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, tol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    order=[1, 2, 3, jnp.inf],
    shape=[(2,)],
  )
  def testMinkowski(self, order, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(shape, dtype))
    jnp_fn = lambda u, v: jsp_distance.minkowski(u, v, p=order)
    np_fn = lambda u, v: osp_distance.minkowski(u, v, p=order)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, tol=1e-4)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
