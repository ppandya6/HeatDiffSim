# Numerical Heat Diffusion Simulator

A Python simulator for the two-dimensional heat equation

```text
∂T/∂t = α (∂²T/∂x² + ∂²T/∂y²)
```

Built with NumPy, SciPy sparse operators, and Matplotlib. Includes a finite-difference solver, configurable boundaries, temperature history export, convergence experiments, static plots, and GIF animations. No server, credentials, or GPU required.

## Install and run

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
# On machines with a read-only home directory:
export MPLCONFIGDIR=/tmp/heatdiffsim-mpl
export XDG_CACHE_HOME=/tmp/heatdiffsim-cache
.venv/bin/python -m heatdiffsim simulate --animate
.venv/bin/python -m heatdiffsim convergence
.venv/bin/python -m unittest discover -s tests -v
```

The installed `heatdiffsim` command also works. Use `--help` on either subcommand to see all options.

```bash
.venv/bin/heatdiffsim simulate --nx 81 --ny 61 --lx 2 --ly 1 \
  --alpha 0.2 --duration 0.5 --boundary neumann --animate
.venv/bin/heatdiffsim simulate --boundary periodic --output outputs/periodic
.venv/bin/heatdiffsim convergence --sizes 17 33 65 129
```

`simulate` starts from a Gaussian hot spot. It writes `temperature.npz` (coordinates, times, fields, step count, dt, periodic flag), `snapshots.png`, and optionally `diffusion.gif`. `convergence` writes `errors.csv` and `convergence.png`. Outputs are ignored by Git. The CLI boundary option applies one boundary type/value to every edge; use the Python API for mixed or time-dependent edges.

## Python API

```python
import numpy as np
from heatdiffsim import Boundary, Grid, solve

# Arrays use (ny, nx), and physical coordinates use x, y.
grid = Grid(nx=61, ny=41, lx=2.0, ly=1.0)
boundaries = {
    "left": Boundary("dirichlet", 0.0),
    "right": Boundary("dirichlet", 0.0),
    "bottom": Boundary("neumann", 0.0),
    "top": Boundary("neumann", 0.0),
}
result = solve(
    grid,
    initial=lambda x, y: np.sin(np.pi*x/2) * np.ones_like(y),
    alpha=0.1, duration=0.5, boundaries=boundaries, snapshots=31,
)
result.save("field.npz")
print(result.fields.shape, result.times[-1])
```

A boundary value may also be a function `f(x, y, t)`, evaluated on its edge. Neumann values mean **outward normal temperature derivatives**, not physical heat flux: left/bottom normals point in the negative coordinate direction. For conductivity k and outward heat flux q, specify derivative `-q/k`. Unspecified edges default to zero Dirichlet. At corners Dirichlet overrides Neumann; adjacent Dirichlet values must agree. Initial edge values are projected to the supplied Dirichlet conditions at t=0.

For periodic boundaries use `Grid(periodic=True)` without edge conditions. Both axes are periodic, and duplicate endpoints are omitted. Nonperiodic grids include both endpoints.

## Numerical method and accuracy

The solver uses centered second-order spatial differences and first-order forward Euler in time. SciPy constructs the sparse Kronecker-sum Laplacian. Neumann conditions use mirrored ghost nodes. Each step must satisfy

```text
dt <= 1 / [2 α (1/dx² + 1/dy²)]
```

The default is 90% of this limit. A supplied dt above the limit is rejected; accepted dt is reduced to reach the requested final time exactly. Snapshots include the initial and final states, are selected at integer steps, and never exceed the requested count. Reducing snapshot count reduces storage, not integration work. Explicit integration costs increase quickly with grid refinement; large simulations should use a small number of snapshots.

The convergence benchmark compares against the exact solution on the unit square with homogeneous Dirichlet conditions:

```text
T(x,y,t) = sin(πx) sin(πy) exp(-2 απ²t)
```

It reports discrete L2 error `sqrt(dx*dy*sum(error²))`, maximum error, and observed order between refinements. Because the default dt scales as h², the combined error should approach order two. This is a coupled space/time refinement experiment, not an independent measurement of temporal order. A validated run at 17, 33, and 65 nodes per axis gave L2 errors 4.09e-4, 1.09e-4, and 2.74e-5, with final observed order 1.99.

Tests cover convergence, periodic eigenmode decay, constant preservation, the discrete maximum principle, insulated-boundary mass conservation using trapezoidal weights, nonzero Neumann derivatives, mixed edges, time-dependent Dirichlet data, snapshot timing, and invalid inputs. The model assumes constant positive diffusivity, a rectangular uniform grid, and no source term.
