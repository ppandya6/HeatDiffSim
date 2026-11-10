"""Manufactured eigenmode benchmark for spatial and temporal convergence."""
import numpy as np
from .solver import Grid, solve


def convergence(sizes=(17, 33, 65), alpha=0.1, duration=0.1):
    rows = []
    previous = None
    for n in sizes:
        grid = Grid(nx=n, ny=n)
        x, y = grid.mesh
        initial = np.sin(np.pi*x) * np.sin(np.pi*y)
        result = solve(grid, initial, alpha=alpha, duration=duration, snapshots=2)
        exact = initial * np.exp(-2*np.pi**2*alpha*duration)
        error = result.fields[-1] - exact
        l2 = np.sqrt(np.sum(error**2)*grid.dx*grid.dy)
        order = np.nan if previous is None else np.log(previous[1]/l2)/np.log(previous[0]/grid.dx)
        rows.append(dict(n=n, h=grid.dx, dt=result.dt, l2=l2,
                         linf=np.max(np.abs(error)), order=order))
        previous = grid.dx, l2
    return rows
