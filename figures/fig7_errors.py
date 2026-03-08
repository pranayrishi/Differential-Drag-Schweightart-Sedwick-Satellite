"""
Figure 7 — Error comparison on semi-log scale (Case C: tangential force).

Shows absolute error |analytical - numerical| for both Shao and Traub.
  - Shao error grows secularly (linearly with time on log scale)
  - Traub error remains near machine precision
"""

import numpy as np
import matplotlib.pyplot as plt

from dynamics.numerical_integrator import SS_numerical_integration
from dynamics.analytical_traub import SS_analytical_traub
from dynamics.analytical_shao import SS_analytical_shao


def plot_figure_7(n, c, T, y0, t_eval):
    """
    Semi-log error comparison for Case C.

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
    delta_fx = 0.0
    delta_fy = 1e-7
    delta_fz = 0.0

    dx0, dxd0, dy0, dyd0 = y0[0], y0[1], y0[2], y0[3]

    # Numerical reference
    _, sol_num = SS_numerical_integration(
        (t_eval[0], t_eval[-1]), y0, delta_fx, delta_fy, delta_fz,
        n, c, t_eval=t_eval)

    # Traub
    dx_t, _, dy_t, _ = SS_analytical_traub(
        t_eval, dx0, dxd0, dy0, dyd0, delta_fx, delta_fy, n, c)

    # Shao
    dx_s, _, dy_s, _ = SS_analytical_shao(
        t_eval, dx0, dxd0, dy0, dyd0, delta_fx, delta_fy, n, c)

    t_per = t_eval / T

    # Errors
    eps = 1e-16   # floor to avoid log(0)
    err_x_shao  = np.abs(dx_s - sol_num[0]) + eps
    err_x_traub = np.abs(dx_t - sol_num[0]) + eps
    err_y_shao  = np.abs(dy_s - sol_num[2]) + eps
    err_y_traub = np.abs(dy_t - sol_num[2]) + eps

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 6.5), sharex=True)

    # --- delta_x error ---
    ax = axes[0]
    ax.semilogy(t_per, err_x_shao,  color='#d62728', lw=1.5, ls='--',
                label='Shao error')
    ax.semilogy(t_per, err_x_traub, color='#2ca02c', lw=1.5, ls='-.',
                label='Traub error')
    ax.set_ylabel(r'$|\delta x_{\rm anal} - \delta x_{\rm num}|$ [m]')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(
        r'Absolute Error vs Numerical Reference — Case C ($\delta f_y = 10^{-7}$ m/s$^2$)',
        fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(bottom=1e-14)

    # --- delta_y error ---
    ax = axes[1]
    ax.semilogy(t_per, err_y_shao,  color='#d62728', lw=1.5, ls='--',
                label='Shao error')
    ax.semilogy(t_per, err_y_traub, color='#2ca02c', lw=1.5, ls='-.',
                label='Traub error')
    ax.set_xlabel('Time [orbital periods]')
    ax.set_ylabel(r'$|\delta y_{\rm anal} - \delta y_{\rm num}|$ [m]')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(bottom=1e-14)

    fig.tight_layout()
    return fig
