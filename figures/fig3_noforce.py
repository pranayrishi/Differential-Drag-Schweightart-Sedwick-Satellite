"""
Figure 3 — Validation: no external force (Case A).

delta_fx = 0, delta_fy = 0.
All three solutions (Numerical, Shao, Traub) are identical in this case.
"""

import numpy as np
import matplotlib.pyplot as plt

from dynamics.numerical_integrator import SS_numerical_integration
from dynamics.analytical_traub import SS_analytical_traub
from dynamics.analytical_shao import SS_analytical_shao


def plot_figure_3(n, c, T, y0, t_eval):
    """
    Compare numerical, Shao, and Traub for Case A (no force).

    Parameters
    ----------
    n : float
        Mean motion [rad/s]
    c : float
        SS coefficient
    T : float
        Orbital period [s]
    y0 : list
        Initial state [dx, dxd, dy, dyd, dz, dzd]
    t_eval : ndarray
        Evaluation times [s]

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    delta_fx = 0.0
    delta_fy = 0.0
    delta_fz = 0.0

    dx0, dxd0, dy0, dyd0 = y0[0], y0[1], y0[2], y0[3]

    # Numerical
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
    ax.plot(t_per, sol_num[0],  color='#1f77b4', lw=2.0, ls='-',
            label='Numerical', zorder=3)
    ax.plot(t_per, dx_s,        color='#d62728', lw=1.5, ls='--',
            label='Shao (analytical)', zorder=2)
    ax.plot(t_per, dx_t,        color='#2ca02c', lw=1.5, ls='-.',
            label='Traub (analytical)', zorder=2)
    ax.set_ylabel(r'$\delta x$ [m]')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(r'Case A: No External Force ($\delta f_x = \delta f_y = 0$)',
                 fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- delta_y ---
    ax = axes[1]
    ax.plot(t_per, sol_num[2], color='#1f77b4', lw=2.0, ls='-',
            label='Numerical', zorder=3)
    ax.plot(t_per, dy_s,       color='#d62728', lw=1.5, ls='--',
            label='Shao (analytical)', zorder=2)
    ax.plot(t_per, dy_t,       color='#2ca02c', lw=1.5, ls='-.',
            label='Traub (analytical)', zorder=2)
    ax.set_xlabel('Time [orbital periods]')
    ax.set_ylabel(r'$\delta y$ [m]')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
