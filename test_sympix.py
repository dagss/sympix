import numpy as np
from .. import sympix

def test_roundup():
    x = [sympix.roundup(i, 2) for i in range(10)]
    assert np.all(np.asarray(x) == [0, 2, 2, 4, 4, 6, 6, 8, 8, 10])

def test_make_sympix_grid():
    k = 5
    nrings_min = 200
    g = sympix.make_sympix_grid(nrings_min, k)
    assert np.all(g.tile_counts == [10, 20, 20, 40, 40, 40, 48, 48, 64, 64, 64, 80, 80, 80, 80, 80, 80, 80, 80, 96])

    if 0:
        # debug plot showing that string of numbers below make sense
        sympix.plot_sympix_grid_efficiency(nrings_min, g)
        from matplotlib.pyplot import show
        show()

def manual_make_sympix_grid_undersample():
    k = 4
    nrings_min = 1000
    g = sympix.make_sympix_grid(nrings_min, k, undersample=True)

    # debug plot showing that string of numbers below make sense
    print g.tile_counts
    sympix.plot_sympix_grid_efficiency(nrings_min, g)
    from matplotlib.pyplot import show
    show()
    assert np.all(g.tile_counts == [10, 20, 20, 40, 40, 40, 48, 48, 64, 64, 64, 80, 80, 80, 80, 80, 80, 80, 80, 80])


def test_scatter_to_rings():
    band_lengths = [4, 8, 10]
    grid = sympix.SymPixGrid(band_lengths, tilesize=2)
    map = sympix.scatter_to_rings(grid, np.arange(2 * 2 * 3))
    assert np.all(map == [
0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8,
4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6])

def test_weight_map():
    from commander.sphere.sharp import SymPixGridPlan
    # Test that a grid's weights is what is needed for spherical harmonic analysis;
    # adjoint synthesis on the weight map should produce a monopole
    band_lengths = [4, 8, 10]
    grid = sympix.SymPixGrid(band_lengths, tilesize=2)
    wmap = grid.compute_weight_map()
    plan = SymPixGridPlan(grid, lmax=3)
    mono = plan.adjoint_synthesis(wmap)
    assert abs(mono[0] - 3.549) < 1e-2  # monopole coefficient
    assert np.all(mono[1:] < 1e-12)     # rest is zero

def test_get_strip():
    ring_lengths = [8, 8, 10, 10]
    grid = sympix.SymPixGrid(ring_lengths)
    iband = 1
    dphi = 2 * np.pi / 8

    def check(phi_start, phi_stop, expected):
        assert np.all(grid.get_strip(iband, phi_start, phi_stop) == np.asarray(expected))

    check(0, 2 * np.pi - 1e-10, np.arange(8))
    check(0, 3 * dphi, [0, 1, 2])
    check(-1 * dphi, 3 * dphi, [7, 0, 1, 2])
    check(-0.9 * dphi, 2.9 * dphi, [7, 0, 1, 2])
    check(-1.5 * dphi, 3 * dphi, [6, 7, 0, 1, 2])
    check(-2 * np.pi, 2 * np.pi - 1e-10, np.concatenate([np.arange(8), np.arange(8)]))

def manual_sht_synthesis():
    # quick test of plotting a dipole and it showing up the expected way
    from commander.sphere.sharp import SymPixGridPlan
    # synthesise a dipole
    alms = np.asarray([1, 0., 0., 1.])
    ring_lengths = [8, 10, 12]
    ring_lengths = np.repeat(ring_lengths, 2)
    grid = sympix.SymPixGrid(ring_lengths, tilesize=4)
    plan = SymPixGridPlan(grid, lmax=1)
    map = plan.synthesis(alms)
    from matplotlib.pyplot import clf, matshow, show, plot

    image = np.zeros((300, 300))
#    plot(map, '-o')
    sympix.sympix_plot(grid, map, image)
    matshow(image)
    show()
    1/0

def test_sht():
    # perfect roundtrip at lmax=5 with 6 rings
    from commander.sphere.sharp import SymPixGridPlan
    lmax = 5
    ring_lengths = np.repeat([16, 16 * 5 // 4, 16 * 5 * 4 // (4 * 3)], 2)
    grid = sympix.SymPixGrid(ring_lengths)
    plan = SymPixGridPlan(grid, lmax=lmax)

    alms0 = np.random.normal(size=(lmax + 1)**2)
    map = plan.synthesis(alms0)
    alms_rt = plan.analysis(map)
    assert np.linalg.norm(alms_rt - alms0) < 1e-14

#def test_sharp_sympix():
#    from commander.sphere.sharp import sympix_geom_info
#    sympix_geom_info(10, 4)
    
    
def manual_neighbours():
    n = 4
    ring_lengths = []#2, 3, 3, 4]
    for inc in sympix.POSSIBLE_INCREMENTS[::-1]:
        nn = n * inc
        assert int(nn) == nn
        n = int(nn)
        ring_lengths += [n, n]
    print ring_lengths
    grid = sympix.SymPixGrid(ring_lengths)
    mat, examples = sympix.sympix_csc_neighbours(grid, lower_only=False)
    print examples
    npix = 2 * np.sum(ring_lengths)
    from matplotlib.pyplot import matshow, show
    from scipy.sparse import csc_matrix
    M = mat.toarray()
    M = M + M.T - np.diagflat(np.diagonal(M))
    matshow(M)
    show()
    1/0
    for i in range(625, grid.npix):
        from matplotlib.pyplot import clf, imshow, show, savefig
        map = M[i,:]#np.zeros(grid.npix)
        nz = map != 0
        #map[nz] = np.arange(1, np.sum(nz) + 1).astype(np.int32)
        #if np.all(map == 0):
        #    continue
        map[i] = map.max()#np.max(M)
        image = np.zeros((300, 300))
        sympix.sympix_plot(grid, map, image)
        clf()
        imshow(image, interpolation='nearest')#, vmin=np.min(M), vmax=np.max(M))
        savefig('tmp/%d.png' % i)
        print i

def manual_test_plot():
    nrings_min = 10
    grid = sympix.SymPixGrid([8, 10, 15, 20])
    print grid.nrings_half, grid.ring_lengths
    map = np.zeros(grid.npix)
    for i in range(grid.nrings_half):
        J = grid.ring_lengths[i]
        map[grid.offsets[i]:grid.offsets[i] + J] = np.arange(J)
        map[grid.offsets[i] + J:grid.offsets[i] + 2 * J] = np.arange(J)
    image = np.zeros((100, 100))
    sympix.sympix_plot(grid, map, image)

    if 1:
        from matplotlib.pyplot import matshow, show
        matshow(image)
        show()
