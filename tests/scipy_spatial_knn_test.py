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

from jax._src import test_util as jtu
from jax_scipy_spatial.knn import KNN as jsp_KNN
from scipy.spatial import KDTree as osp_KNN

import jax.numpy as jnp
from jax.config import config

config.parse_flags_with_absl()

num_samples = 2


class LaxBackedScipySpatialKnnTests(jtu.JaxTestCase):
  """Tests for LAX-backed scipy.spatial.distance implementations."""

  @jtu.sample_product(
    k=[1],
    shape=[(128, 3)],
    dtype=jtu.dtypes.floating,
  )
  def testKnn(self, shape, k, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng((k, shape[-1]), dtype))
    jnp_fn = lambda data, x: jsp_KNN(data).query(x)
    np_fn = lambda data, x: osp_KNN(data).query(x)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, tol=1e-4)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
