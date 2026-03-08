"""
Figure 2 — State decomposition into secular and periodic components.

Shows delta_x(t) and delta_y(t) decomposed into:
  - Total solution
  - Secular (drift) component
  - Periodic (oscillatory) component

Uses illustrative parameters (not tied to a specific force case).
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_figure_2(T):
    """
    Create state decomposition figure.

    Parameters
    ----------
    T : float
        Orbital period [s], used to define time axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Illustrative parameters (chosen to produce a clear figure)
    A     = 80.0          # oscillation amplitude [m]
    phi0  = np.radians(30.0)
    x_c   = 20.0          # secular offset in x [m]
    y_c   = 50.0          # secular offset in y [m]

    # Secular drift rate (illustrative small value)
    kx    = 5.0 / T       # m / s   (drift in x)
    ky    = -15.0 / T     # m / s   (drift in y — dominates for tang. force)

    t = np.linspace(0, 5 * T, 2000)
    t_per = t / T         # time in orbital periods

    # Periodic parts
    x_periodic = A * np.cos(2 * np.pi * t / T + phi0)
    y_periodic = -2.0 * A * np.sin(2 * np.pi * t / T + phi0)

    # Secular parts
    x_secular  = x_c + kx * t
    y_secular  = y_c + ky * t

    # Total
    x_total = x_secular + x_periodic
    y_total = y_secular + y_periodic

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 6.5), sharex=True)

    # --- delta_x panel ---
    ax = axes[0]
    ax.plot(t_per, x_total,    color='#1f77b4', lw=2.0, label='Total $\\delta x$')
    ax.plot(t_per, x_secular,  color='#d62728', lw=1.5, linestyle='--',
            label='Secular')
    ax.plot(t_per, x_periodic, color='#2ca02c', lw=1.5, linestyle='-.',
            label='Periodic')
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_ylabel(r'$\delta x$ [m]')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(r'State Decomposition: Secular + Periodic Components', fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- delta_y panel ---
    ax = axes[1]
    ax.plot(t_per, y_total,    color='#1f77b4', lw=2.0, label='Total $\\delta y$')
    ax.plot(t_per, y_secular,  color='#d62728', lw=1.5, linestyle='--',
            label='Secular')
    ax.plot(t_per, y_periodic, color='#2ca02c', lw=1.5, linestyle='-.',
            label='Periodic')
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('Time [orbital periods]')
    ax.set_ylabel(r'$\delta y$ [m]')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
