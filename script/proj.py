import numpy as np
import random

import sys
sys.path.insert(0, '/home/aboulin/')

from CoPY.src.rng.evd import Logistic

def project_simplex(v, z=1.0, axis=-1):
    """
    Implements the algorithm in Figure 1 of
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra,
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008.
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    This algorithm project vectors v onto the simplex w >= 0, \sum w_i = z.
    :param v: A numpy array, will be interpreted as a collection of vectors.
    :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
    :param axis: Indicates the axis of v, which defines the vectors to be projected.
    :return: w: result of the projection
    """

    def _project_simplex_2d(v, z):
        """
        Helper function, assuming that all vectors are arranged in rows of v.
        :param v: NxD numpy array; Duchi et al. algorithm is applied to each row in vecotrized form
        :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
        :return: w: result of the projection
        """
        shape = v.shape
        if shape[1] == 1:
            w = np.array(v)
            w[:] = z
            return w

        mu = np.sort(v, axis=1)
        mu = np.flip(mu, axis=1)
        cum_sum = np.cumsum(mu, axis=1)
        j = np.expand_dims(np.arange(1, shape[1] + 1), 0)
        rho = np.sum(mu * j - cum_sum + z > 0.0, axis=1, keepdims=True) - 1
        max_nn = cum_sum[np.arange(shape[0]), rho[:, 0]]
        theta = (np.expand_dims(max_nn, -1) - z) / (rho + 1)
        w = (v - theta).clip(min=0.0)
        return w

    shape = v.shape

    if len(shape) == 0:
        return np.array(1.0, dtype=v.dtype)
    elif len(shape) == 1:
        return _project_simplex_2d(np.expand_dims(v, 0), z)[0, :]
    else:
        axis = axis % len(shape)
        t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
        tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
        v_t = np.transpose(v, t_shape)
        v_t_shape = v_t.shape
        v_t_unroll = np.reshape(v_t, (-1, v_t_shape[-1]))

        w_t = _project_simplex_2d(v_t_unroll, z)

        w_t_reroll = np.reshape(w_t, v_t_shape)
        return np.transpose(w_t_reroll, tt_shape)

copula = Logistic(theta = 1.0, d = 100, n_sample = 1)
v = copula.sample_unimargin()
print(v)
z = 1

w = project_simplex(v,z)

print(np.sum(w))