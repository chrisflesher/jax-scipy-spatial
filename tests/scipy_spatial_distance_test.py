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

num_samples = 2


class LaxBackedScipySpatialDistanceTests(jtu.JaxTestCase):
  """Tests for LAX-backed scipy.spatial.distance implementations."""

  @jtu.sample_product(
    method=[
      # 'dice',
      # 'directed_hausdorff',
      'hamming',
      # 'jensenshannon',
      # 'kulczynski1',
      # 'rogerstanimoto',
      # 'sokalmichener',
      # 'sokalsneath',
      # 'yule'
    ],
    dtype=jtu.dtypes.integer,
    shape=[(num_samples,)],
    use_weight=[False, True],
  )
  def testDistanceBoolean(self, method, shape, dtype, use_weight):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(shape, dtype), jnp.abs(rng(shape, jnp.float32)) if use_weight else None)
    jnp_fn = lambda u, v, w: getattr(jsp_distance, method)(u, v, w)
    np_fn = lambda u, v, w: getattr(osp_distance, method)(u, v, w)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, tol=1e-4)

  @jtu.sample_product(
    method=[
      'braycurtis',
      'canberra',
      'chebyshev',
      'cityblock',
      'correlation',
      'cosine',
      'euclidean',
      'hamming',
      'jaccard',
      'russellrao',
      'sqeuclidean',
    ],
    dtype=float_dtypes,
    shape=[(num_samples,)],
    use_weight=[False, True],
  )
  def testDistanceNumeric(self, method, shape, dtype, use_weight):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(shape, dtype), jnp.abs(rng(shape, dtype)) if use_weight else None)
    jnp_fn = lambda u, v, w: getattr(jsp_distance, method)(u, v, w)
    np_fn = lambda u, v, w: getattr(osp_distance, method)(u, v, w)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, tol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    order=[1, 2, 3, jnp.inf],
    shape=[(num_samples,)],
    use_weight=[False, True],
  )
  def testMinkowski(self, order, shape, dtype, use_weight):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(shape, dtype), jnp.abs(rng(shape, dtype)) if use_weight else None)
    jnp_fn = lambda u, v, w: jsp_distance.minkowski(u, v, p=order, w=w)
    np_fn = lambda u, v, w: osp_distance.minkowski(u, v, p=order, w=w)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, tol=1e-4)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
