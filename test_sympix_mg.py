from __future__ import division

import numpy as np

from .. import sympix_mg
from ....sphere import sympix, beams, scatter_l_to_lm, sharp

def hammer(n, m, op):
    u = np.zeros(m)
    out = np.zeros((n, m))
    for j in range(m):
        u[j] = 1
        out[:, j] = op(u)
        u[j] = 0
    return out


def as_dense(shape, indptr, indices, indirection, blocks, mirror=False):
    out = np.zeros(shape)
    m, n = blocks.shape[0:2]
    idx = 0
    for j in range(indptr.shape[0] - 1):
        for i in indices[indptr[j]:indptr[j + 1]]:
            out[i * m:(i + 1) * m, j * n:(j + 1) * n] = blocks[:, :, indirection[idx]]
            if mirror:
                out[j * n:(j + 1) * n, i * m:(i + 1) * m] = blocks[:, :, indirection[idx]].T
            idx += 1
    return out


def test_compute_YDYt_block():
    lmax = 6
    bl = np.linspace(1, 0, lmax + 1)
    blm = scatter_l_to_lm(bl)

    tile_counts = [8, 16, 16, 20]
    grid_lo = sympix.SymPixGrid(tile_counts, tilesize=2)
    grid_hi = sympix.SymPixGrid(tile_counts, tilesize=4)
    plan_lo = sharp.SymPixGridPlan(grid_lo, lmax)
    plan_hi = sharp.SymPixGridPlan(grid_hi, lmax)

    # The truth via unit vector hammering
    def op(x):
        x = plan_lo.adjoint_synthesis(x)
        x *= blm
        x = plan_hi.synthesis(x)
        return x

    true_matrix = hammer(grid_hi.npix, grid_lo.npix, op)

    neighmat, (label_to_i, label_to_j) = sympix.sympix_csc_neighbours(grid_lo, lower_only=False)
    blocks = sympix_mg.compute_many_YDYt_blocks(grid_hi, grid_lo, bl.astype(np.float32),
                                                np.asarray(label_to_i, dtype=np.int32),
                                                np.asarray(label_to_j, dtype=np.int32))
    approx_matrix = as_dense((grid_hi.npix, grid_lo.npix), neighmat.indptr, neighmat.indices, neighmat.data, blocks)
    mask = as_dense((grid_hi.npix, grid_lo.npix), neighmat.indptr, neighmat.indices, neighmat.data, blocks * 0 + 1)

    rel_delta = np.abs((approx_matrix - true_matrix) * mask) / true_matrix
    assert np.max(rel_delta) < 1e-5
