"""Second-order spatial differences with explicit Euler time integration.

Arrays have shape (ny, nx). Neumann values are outward normal derivatives,
not heat fluxes. Periodic grids omit the duplicate endpoint.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.sparse import diags, eye, kron


@dataclass(frozen=True)
class Grid:
    nx: int = 51
    ny: int = 51
    lx: float = 1.0
    ly: float = 1.0
    periodic: bool = False

    def __post_init__(self):
        if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 3 for n in (self.nx, self.ny)):
            raise ValueError("nx and ny must be integers >= 3")
        if not all(np.isfinite(v) and v > 0 for v in (self.lx, self.ly)):
            raise ValueError("Domain lengths must be finite and positive")

    @property
    def dx(self):
        return self.lx / (self.nx if self.periodic else self.nx - 1)

    @property
    def dy(self):
        return self.ly / (self.ny if self.periodic else self.ny - 1)

    @property
    def mesh(self):
        return np.meshgrid(np.arange(self.nx) * self.dx, np.arange(self.ny) * self.dy)


@dataclass(frozen=True)
class Boundary:
    """A constant value or f(x, y, t) on an edge."""
    kind: str = "dirichlet"
    value: float | Callable = 0.0

    def __post_init__(self):
        if self.kind not in ("dirichlet", "neumann"):
            raise ValueError("Boundary kind must be dirichlet or neumann")
        if not callable(self.value) and not np.isfinite(self.value):
            raise ValueError("Boundary value must be finite")


@dataclass
class Result:
    grid: Grid
    times: np.ndarray
    fields: np.ndarray
    dt: float
    steps: int

    def save(self, path):
        """Save fields plus coordinates and integration metadata to an NPZ archive."""
        x, y = self.grid.mesh
        np.savez_compressed(path, times=self.times, fields=self.fields, x=x, y=y,
                            dt=self.dt, steps=self.steps, periodic=self.grid.periodic)


def _axis_operator(n, h, periodic):
    a = diags([np.ones(n-1), -2*np.ones(n), np.ones(n-1)], [-1, 0, 1], format="lil")
    if periodic:
        a[0, -1] = a[-1, 0] = 1
    else:
        # Mirrored ghost nodes for outward Neumann derivatives.
        a[0, 1] = a[-1, -2] = 2
    return a.tocsr() / h**2


def solve(grid=None, initial=None, *, alpha=0.1, duration=0.1, dt=None,
          boundaries=None, snapshots=21):
    """Integrate a field, returning selected states including t=0 and t=duration.

    boundaries maps left/right/bottom/top to Boundary objects; omitted edges are
    zero Dirichlet. At corners Dirichlet takes precedence over Neumann; two
    Dirichlet edges must agree. Periodicity applies to both axes and disallows
    edge conditions. dt is an upper bound, adjusted to hit the end time exactly.
    """
    grid = grid or Grid()
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("alpha and duration must be finite and positive")
    if isinstance(snapshots, bool) or not isinstance(snapshots, (int, np.integer)) or snapshots < 2:
        raise ValueError("snapshots must be an integer >= 2")
    limit = 1 / (2 * alpha * (grid.dx**-2 + grid.dy**-2))
    dt = 0.9 * limit if dt is None else dt
    if not np.isfinite(dt) or dt <= 0 or dt > limit * (1 + 1e-12):
        raise ValueError(f"Explicit Euler requires 0 < dt <= {limit:.8g}")
    steps = int(np.ceil(duration / dt))
    dt = duration / steps
    bcs = dict(boundaries or {})
    if set(bcs) - {"left", "right", "bottom", "top"}:
        raise ValueError("Unknown boundary edge")
    if grid.periodic and bcs:
        raise ValueError("Periodic grids cannot have edge conditions")
    if not grid.periodic:
        bcs = {edge: bcs.get(edge, Boundary()) for edge in ("left", "right", "bottom", "top")}
    if any(not isinstance(bc, Boundary) for bc in bcs.values()):
        raise ValueError("Each edge must be a Boundary")
    x, y = grid.mesh
    u = np.asarray(initial(x, y) if callable(initial) else (np.zeros_like(x) if initial is None else initial), dtype=float).copy()
    if u.shape != x.shape or not np.isfinite(u).all():
        raise ValueError(f"Initial field must be finite with shape {x.shape}")
    edges = {"left": (slice(None), 0), "right": (slice(None), -1),
             "bottom": (0, slice(None)), "top": (-1, slice(None))}

    def edge_values(bc, edge, t):
        idx = edges[edge]
        value = bc.value(x[idx], y[idx], t) if callable(bc.value) else bc.value
        values = np.broadcast_to(np.asarray(value, dtype=float), x[idx].shape)
        if not np.isfinite(values).all():
            raise ValueError("Boundary function returned nonfinite values")
        return values

    def impose(field, t):
        assigned = np.full_like(field, np.nan)
        for edge, bc in bcs.items():
            if bc.kind == "dirichlet":
                idx = edges[edge]
                values = edge_values(bc, edge, t)
                old = assigned[idx]
                if np.any(np.isfinite(old) & ~np.isclose(old, values, atol=1e-12, rtol=1e-10)):
                    raise ValueError("Dirichlet values disagree at a corner")
                assigned[idx] = values
                field[idx] = values

    lap = kron(eye(grid.ny), _axis_operator(grid.nx, grid.dx, grid.periodic)) + kron(_axis_operator(grid.ny, grid.dy, grid.periodic), eye(grid.nx))
    lap = lap.tocsr()
    impose(u, 0.0)
    selected = set(np.rint(np.linspace(0, steps, min(snapshots, steps+1))).astype(int))
    fields, times = [u.copy()], [0.0]
    for step in range(1, steps+1):
        t = (step-1) * dt
        rhs = (lap @ u.ravel()).reshape(u.shape)
        for edge, bc in bcs.items():
            if bc.kind == "neumann":
                h = grid.dx if edge in ("left", "right") else grid.dy
                rhs[edges[edge]] += 2 * edge_values(bc, edge, t) / h
        u += alpha * dt * rhs
        impose(u, step * dt)
        if not np.isfinite(u).all():
            raise FloatingPointError("Temperature became nonfinite")
        if step in selected:
            fields.append(u.copy())
            times.append(step * dt)
    return Result(grid, np.array(times), np.array(fields), dt, steps)
