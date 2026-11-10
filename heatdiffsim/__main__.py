import argparse
import csv
from pathlib import Path
import numpy as np
from .solver import Boundary, Grid, solve


def main():
    parser = argparse.ArgumentParser(description="2D finite-difference heat diffusion")
    sub = parser.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate", help="Simulate a Gaussian hot spot")
    sim.add_argument("--nx", type=int, default=51)
    sim.add_argument("--ny", type=int, default=51)
    sim.add_argument("--lx", type=float, default=1.0)
    sim.add_argument("--ly", type=float, default=1.0)
    sim.add_argument("--alpha", type=float, default=0.1)
    sim.add_argument("--duration", type=float, default=0.1)
    sim.add_argument("--dt", type=float)
    sim.add_argument("--snapshots", type=int, default=41)
    sim.add_argument("--boundary", choices=["dirichlet", "neumann", "periodic"], default="dirichlet")
    sim.add_argument("--boundary-value", type=float, default=0.0)
    sim.add_argument("--animate", action="store_true")
    sim.add_argument("--output", type=Path, default=Path("outputs/simulation"))
    conv = sub.add_parser("convergence", help="Verify convergence against an exact eigenmode")
    conv.add_argument("--sizes", type=int, nargs="+", default=[17, 33, 65])
    conv.add_argument("--output", type=Path, default=Path("outputs/convergence"))
    args = parser.parse_args()
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        if args.command == "simulate":
            from .plotting import animate, plot_snapshots
            periodic = args.boundary == "periodic"
            if periodic and args.boundary_value != 0:
                raise ValueError("Periodic boundaries have no boundary value")
            grid = Grid(args.nx, args.ny, args.lx, args.ly, periodic)
            bcs = None if periodic else {e: Boundary(args.boundary, args.boundary_value) for e in ("left", "right", "bottom", "top")}
            initial = lambda x, y: np.exp(-80*((x/grid.lx-.5)**2+(y/grid.ly-.5)**2))
            result = solve(grid, initial, alpha=args.alpha, duration=args.duration,
                           dt=args.dt, boundaries=bcs, snapshots=args.snapshots)
            result.save(args.output / "temperature.npz")
            plot_snapshots(result, args.output / "snapshots.png")
            if args.animate:
                animate(result, args.output / "diffusion.gif")
            print(f"Completed {result.steps} steps; dt={result.dt:.6g}; final range=[{result.fields[-1].min():.6g}, {result.fields[-1].max():.6g}]")
        else:
            from .experiments import convergence
            from .plotting import plot_convergence
            if len(args.sizes) < 2 or any(b <= a for a, b in zip(args.sizes, args.sizes[1:])):
                raise ValueError("Provide at least two increasing grid sizes")
            rows = convergence(args.sizes)
            with (args.output / "errors.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            plot_convergence(rows, args.output / "convergence.png")
            for row in rows:
                print(f"n={row['n']:3d} L2={row['l2']:.6g} order={row['order']:.3f}")
        print(f"Results saved to {args.output}")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
