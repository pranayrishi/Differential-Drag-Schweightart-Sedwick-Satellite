"""
Figure 9 — Phase plane trajectories in (delta_x, delta_xdot/n) space.

Shows how a constant tangential force (positive = thrust, negative = drag)
displaces the phase-plane trajectory from the unforced ellipse.

Solid lines: +delta_fy (acceleration / thrust)
Dashed lines: -delta_fy (deceleration / differential drag)
"""

import numpy as np
import matplotlib.pyplot as plt

from dynamics.analytical_traub import SS_analytical_traub


def plot_figure_9(n, c, T):
    """
    Phase plane figure for various ICs under ±tangential force.

    Parameters
    ----------
    n : float
    c : float
    T : float

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    delta_fx     = 0.0
    delta_fy_mag = 1e-7
    delta_fz     = 0.0

    dx0_list     = [-150.0, -75.0, 0.0, 75.0, 150.0]   # m
    delta_xdot0  = 0.0

    t_eval = np.linspace(0, 5 * T, 5000)

    # Colour cycle
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']

    fig, ax = plt.subplots(figsize=(7.1, 5.5))

    for idx, dx0 in enumerate(dx0_list):
        # Bounded-orbit IC
        dyd0 = -2.0 * c * n * dx0
        dy0  = 0.0

        col = colours[idx % len(colours)]

        for sign, style in [(+1, '-'), (-1, '--')]:
            fy = sign * delta_fy_mag
            dx, dxd, dy, _ = SS_analytical_traub(
                t_eval, dx0, delta_xdot0, dy0, dyd0, delta_fx, fy, n, c)

            # Phase plane: (delta_x, delta_xdot / n)
            ax.plot(dx, dxd / n, color=col, ls=style, lw=1.2, alpha=0.8)

        # Mark initial point
        ax.plot(dx0, delta_xdot0 / n, 'o', color=col, markersize=5, zorder=5)

    # Legend proxy
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', ls='-',  lw=1.5,
               label=r'$+\delta f_y$ (thrust / acceleration)'),
        Line2D([0], [0], color='gray', ls='--', lw=1.5,
               label=r'$-\delta f_y$ (drag / deceleration)'),
        Line2D([0], [0], marker='o', color='gray', ls='None', markersize=5,
               label='Initial condition'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

    ax.set_xlabel(r'$\delta x$ [m]')
    ax.set_ylabel(r'$\dot{\delta x}\,/\,n$ [m]')
    ax.set_title(
        r'Phase Plane ($\delta x$, $\dot{\delta x}/n$) — Various ICs, '
        r'$|\delta f_y| = 10^{-7}$ m/s$^2$',
        fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.axvline(0, color='k', lw=0.8, ls=':')

    fig.tight_layout()
    return fig
