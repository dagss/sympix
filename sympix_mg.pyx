from libc.stdint cimport int32_t, int8_t, int64_t
from libc.math cimport sqrt, log10, pow
import numpy as np
cimport numpy as cnp
cimport cython

cdef extern:
    void legendre_transform_recfac_s(float *recfac, int32_t lmax)
    void rescale_bl_s "sympix_mg_rescale_bl_s"(float *bl, float *rescaled_bl, int32_t lmax)

    void compute_YDYt_block_ "sympix_mg_compute_YDYt_block"(
        int32_t n1, int32_t n2, double dphi1, double dphi2,
        double *thetas1, double *thetas2, double phi0_2,
        int32_t lmax, float *rescaled_bl, float *recfac,
        float *out)

    void compute_many_YDYt_blocks_ "sympix_mg_compute_many_YDYt_blocks"(
       int32_t nblocks,
       int32_t tilesize1, int32_t bandcount1, double *thetas1, int32_t *tilecounts1, int32_t *tileindices1,
       int32_t tilesize2, int32_t bandcount2, double *thetas2, int32_t *tilecounts2, int32_t *tileindices2,
       int32_t lmax, float *bl, int32_t *ierr, float *out_blocks)


def compute_YDYt_block(
    cnp.ndarray[double, mode='c'] thetas1,
    cnp.ndarray[double, mode='c'] thetas2,
    float dphi1, float dphi2, float phi0_D2,
    cnp.ndarray[float, mode='c'] bl):

    cdef int32_t lmax = bl.shape[0] - 1, n1 = thetas1.shape[0], n2 = thetas2.shape[0]
    cdef cnp.ndarray[float, mode='c'] rescaled_bl = np.empty_like(bl)
    cdef cnp.ndarray[float, mode='c'] recfac = np.empty_like(bl)
    cdef cnp.ndarray[float, ndim=2, mode='fortran'] out = np.empty((n1 * n1, n2 * n2), np.float32,
                                                                   order='F')

    rescale_bl_s(&bl[0], &rescaled_bl[0], lmax)
    legendre_transform_recfac_s(&recfac[0], lmax)
    compute_YDYt_block_(n1, n2, dphi1, dphi2, &thetas1[0], &thetas2[0], phi0_D2,
                        lmax, &rescaled_bl[0], &recfac[0], &out[0, 0])
    return out

def compute_many_YDYt_blocks(grid1, grid2,
                             cnp.ndarray[float, mode='c'] bl,
                             cnp.ndarray[int32_t, mode='c'] indices1,
                             cnp.ndarray[int32_t, mode='c'] indices2):
    cdef int32_t lmax = bl.shape[0] - 1, nblocks = indices1.shape[0], ierr
    cdef cnp.ndarray[float, ndim=3, mode='fortran'] out = (
        np.empty((grid1.tilesize**2, grid2.tilesize**2, nblocks), np.float32, order='F'))
    cdef cnp.ndarray[double, ndim=1, mode='c'] thetas1 = grid1.thetas, thetas2 = grid2.thetas
    cdef cnp.ndarray[int32_t, ndim=1, mode='c'] tile_counts1 = grid1.tile_counts
    cdef cnp.ndarray[int32_t, ndim=1, mode='c'] tile_counts2 = grid2.tile_counts

    if indices1.shape[0] != indices2.shape[0]:
        raise ValueError()
    compute_many_YDYt_blocks_(
        nblocks,
        grid1.tilesize, grid1.band_pair_count, &thetas1[0], &tile_counts1[0], &indices1[0],
        grid2.tilesize, grid2.band_pair_count, &thetas2[0], &tile_counts2[0], &indices2[0],
        lmax, &bl[0], &ierr, &out[0, 0, 0])
    if ierr != 0:
        msg = 'unknown'
        if ierr == 1:
            msg = 'Illegal pixel index'
        raise Exception(msg)
    return out
