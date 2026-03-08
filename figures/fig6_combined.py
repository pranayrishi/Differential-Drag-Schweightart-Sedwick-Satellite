"""
Figure 6 — Validation: combined radial + tangential force (Case D).

delta_fx = 1e-7, delta_fy = 1e-7 m/s^2.
Similar to Case C but both forces present.
"""

import numpy as np
import matplotlib.pyplot as plt

from dynamics.numerical_integrator import SS_numerical_integration
from dynamics.analytical_traub import SS_analytical_traub
from dynamics.analytical_shao import SS_analytical_shao


def plot_figure_6(n, c, T, y0, t_eval):
    """
    Compare numerical, Shao, and Traub for Case D (combined forces).

    Parameters
    ----------
    n : float
    c : float
    T : float
    y0 : list
    t_eval : ndarray

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    delta_fx = 1e-7
    delta_fy = 1e-7
    delta_fz = 0.0

    dx0, dxd0, dy0, dyd0 = y0[0], y0[1], y0[2], y0[3]

    # Numerical reference
    t_num, sol_num = SS_numerical_integration(
        (t_eval[0], t_eval[-1]), y0, delta_fx, delta_fy, delta_fz,
        n, c, t_eval=t_eval)

    # Traub
    dx_t, _, dy_t, _ = SS_analytical_traub(
        t_eval, dx0, dxd0, dy0, dyd0, delta_fx, delta_fy, n, c)

    # Shao
    dx_s, _, dy_s, _ = SS_analytical_shao(
        t_eval, dx0, dxd0, dy0, dyd0, delta_fx, delta_fy, n, c)

    t_per = t_eval / T

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 6.5), sharex=True)

    # --- delta_x ---
    ax = axes[0]
    ax.plot(t_per, sol_num[0], color='#1f77b4', lw=2.0, ls='-',
            label='Numerical (reference)', zorder=4)
    ax.plot(t_per, dx_s,       color='#d62728', lw=1.5, ls='--',
            label='Shao — incorrect', zorder=3)
    ax.plot(t_per, dx_t,       color='#2ca02c', lw=1.5, ls='-.',
            label='Traub — corrected', zorder=3)
    ax.set_ylabel(r'$\delta x$ [m]')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(
        r'Case D: Combined Forces ($\delta f_x = \delta f_y = 10^{-7}$ m/s$^2$)',
        fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- delta_y ---
    ax = axes[1]
    ax.plot(t_per, sol_num[2], color='#1f77b4', lw=2.0, ls='-',
            label='Numerical (reference)', zorder=4)
    ax.plot(t_per, dy_s,       color='#d62728', lw=1.5, ls='--',
            label='Shao — incorrect', zorder=3)
    ax.plot(t_per, dy_t,       color='#2ca02c', lw=1.5, ls='-.',
            label='Traub — corrected', zorder=3)
    ax.set_xlabel('Time [orbital periods]')
    ax.set_ylabel(r'$\delta y$ [m]')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
