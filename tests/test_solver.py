import unittest
import numpy as np
from heatdiffsim import Boundary, Grid, solve
from heatdiffsim.experiments import convergence


class SolverTests(unittest.TestCase):
    def test_convergence(self):
        rows = convergence()
        self.assertTrue(all(1.9 < r['order'] < 2.1 for r in rows[1:]))
        self.assertLess(rows[-1]['l2'], 1e-4)

    def test_constant_fields(self):
        for periodic, kind in [(False, 'dirichlet'), (False, 'neumann'), (True, None)]:
            g = Grid(13, 9, 2, 1, periodic)
            bc = None if periodic else {e: Boundary(kind, 3 if kind == 'dirichlet' else 0) for e in ['left','right','top','bottom']}
            r = solve(g, np.full((9,13), 3.), boundaries=bc)
            np.testing.assert_allclose(r.fields, 3, atol=1e-12)

    def test_neumann_conservation(self):
        g = Grid(17, 13)
        x, y = g.mesh
        r = solve(g, np.exp(-40*((x-.5)**2+(y-.5)**2)), boundaries={e: Boundary('neumann') for e in ['left','right','top','bottom']})
        weights = np.ones((13,17))
        weights[[0,-1], :] *= .5
        weights[:, [0,-1]] *= .5
        mass = np.sum(r.fields*weights, axis=(1,2))
        np.testing.assert_allclose(mass, mass[0], atol=1e-10)

    def test_nonzero_neumann_and_mixed_edges(self):
        g = Grid(11, 9, 2, 1)
        x, _ = g.mesh
        bcs = {'left': Boundary('neumann', -1), 'right': Boundary('neumann', 1),
               'bottom': Boundary('dirichlet', lambda x,y,t: x),
               'top': Boundary('dirichlet', lambda x,y,t: x)}
        r = solve(g, x, boundaries=bcs)
        np.testing.assert_allclose(r.fields[-1], x, atol=1e-12)

    def test_periodic_eigenmode(self):
        g = Grid(40, 32, periodic=True)
        x,y = g.mesh
        initial = np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
        r = solve(g, initial, duration=.02)
        exact = initial*np.exp(-.1*8*np.pi**2*.02)
        self.assertLess(np.max(np.abs(r.fields[-1]-exact)), .002)

    def test_time_boundary_and_snapshots(self):
        bcs = {e: Boundary('dirichlet', lambda x,y,t: t) for e in ['left','right','top','bottom']}
        r = solve(Grid(9,9), duration=.013, boundaries=bcs, snapshots=4)
        self.assertEqual(r.times[-1], .013)
        self.assertEqual(len(r.times), 2 if r.steps == 1 else min(4, r.steps+1))
        np.testing.assert_allclose(r.fields[-1, 0], .013)

    def test_reject_invalid(self):
        for kwargs in [dict(dt=1), dict(alpha=0), dict(duration=-1), dict(snapshots=1), dict(initial=np.zeros((2,2)))]:
            with self.assertRaises(ValueError):
                solve(**kwargs)
        with self.assertRaises(ValueError):
            solve(Grid(periodic=True), boundaries={'left': Boundary()})
        with self.assertRaises(ValueError):
            solve(boundaries={'left': Boundary(value=1)})

    def test_maximum_principle(self):
        rng = np.random.default_rng(42)
        r = solve(Grid(13,11), rng.random((11,13)))
        self.assertGreaterEqual(r.fields.min(), 0)
        self.assertLessEqual(r.fields.max(), 1)


if __name__ == '__main__':
    unittest.main()
