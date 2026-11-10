"""Finite-difference simulation of u_t = alpha (u_xx + u_yy)."""
from .solver import Boundary, Grid, Result, solve

__all__ = ["Boundary", "Grid", "Result", "solve"]
