"""Headless figures and portable GIF animations."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def plot_snapshots(result, path):
    indices = np.unique(np.linspace(0, len(result.times)-1, 4).astype(int))
    fig, axes = plt.subplots(1, len(indices), figsize=(4*len(indices), 3.7), squeeze=False, constrained_layout=True)
    low, high = result.fields.min(), result.fields.max()
    for ax, i in zip(axes.flat, indices):
        image = ax.imshow(result.fields[i], origin="lower", extent=(0, result.grid.lx, 0, result.grid.ly),
                          cmap="inferno", vmin=low, vmax=high, aspect="equal")
        ax.set(title=f"t = {result.times[i]:.4g}", xlabel="x", ylabel="y")
    fig.colorbar(image, ax=list(axes.flat), label="Temperature")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def animate(result, path):
    fig, ax = plt.subplots()
    image = ax.imshow(result.fields[0], origin="lower", extent=(0, result.grid.lx, 0, result.grid.ly),
                      cmap="inferno", vmin=result.fields.min(), vmax=result.fields.max())
    ax.set(xlabel="x", ylabel="y")
    fig.colorbar(image, ax=ax, label="Temperature")
    def update(i):
        image.set_data(result.fields[i])
        ax.set_title(f"Heat diffusion · t = {result.times[i]:.4g}")
        return (image,)
    animation = FuncAnimation(fig, update, frames=len(result.times), interval=80)
    animation.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


def plot_convergence(rows, path):
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    h = np.array([r['h'] for r in rows])
    error = np.array([r['l2'] for r in rows])
    ax.loglog(h, error, 'o-', label='Measured L2 error')
    ax.loglog(h, error[0]*(h/h[0])**2, '--', label='Second-order reference')
    ax.set(xlabel='Grid spacing h', ylabel='Error', title='Heat equation convergence')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
