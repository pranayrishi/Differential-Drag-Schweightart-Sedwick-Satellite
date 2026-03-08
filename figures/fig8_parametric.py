"""
Figure 8 — Parametric study: Shao error vs (c - 1).

Varies inclination to sweep c values, then plots the final-time Shao error
in delta_y for Case C (tangential force only) against (c - 1) on a log-log
scale. The slope should be close to 1, confirming error ∝ (c - 1).
"""

import numpy as np
import matplotlib.pyplot as plt

from dynamics.constants import mu, R_E, J2
from dynamics.ss_model import compute_mean_motion, compute_SS_coefficient
from dynamics.numerical_integrator import SS_numerical_integration
from dynamics.analytical_shao import SS_analytical_shao


def plot_figure_8(a, e, n, T):
    """
    Parametric error study varying inclination.

    Parameters
    ----------
    a : float
        Semi-major axis [m]
    e : float
        Eccentricity [-]
    n : float
        Mean motion [rad/s]
    T : float
        Orbital period [s]

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Inclinations that give a range of c values
    # i = arcsin(sqrt(2/3)) ~ 54.74 deg gives c = 1 exactly
    inclinations_deg = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                                  60.0, 70.0, 80.0, 90.0, 98.0, 110.0, 130.0])

    delta_fx = 0.0
    delta_fy = 1e-7
    delta_fz = 0.0

    delta_x0    = 100.0
    delta_xdot0 = 0.0
    delta_y0    = 0.0

    t_end    = 5 * T
    N_points = 2000
    t_eval   = np.linspace(0, t_end, N_points)

    c_values     = []
    error_y_shao = []

    for i_deg in inclinations_deg:
        i_rad = np.radians(i_deg)
        c = compute_SS_coefficient(a, e, i_rad)

        omega_SS_sq = n**2 * (4.0 * c + 2.0 - 5.0 * c**2)
        if omega_SS_sq <= 0:
            continue  # Skip unstable parameter region

        # Bounded-orbit IC
        dyd0 = -2.0 * c * n * delta_x0
        y0   = [delta_x0, delta_xdot0, delta_y0, dyd0, 0.0, 0.0]

        # Numerical reference
        _, sol_num = SS_numerical_integration(
            (t_eval[0], t_eval[-1]), y0, delta_fx, delta_fy, delta_fz,
            n, c, t_eval=t_eval)

        # Shao analytical
        _, _, dy_s, _ = SS_analytical_shao(
            t_eval, delta_x0, delta_xdot0, delta_y0, dyd0,
            delta_fx, delta_fy, n, c)

        # Final-time error in delta_y
        err_y = np.abs(dy_s[-1] - sol_num[2, -1])

        if np.abs(c - 1.0) > 1e-8 and err_y > 1e-12:
            c_values.append(c)
            error_y_shao.append(err_y)

    c_arr   = np.array(c_values)
    err_arr = np.array(error_y_shao)
    cm1_arr = np.abs(c_arr - 1.0)

    # Log-log linear fit
    log_cm1  = np.log10(cm1_arr)
    log_err  = np.log10(err_arr)
    coeffs   = np.polyfit(log_cm1, log_err, 1)
    slope    = coeffs[0]
    intercept = coeffs[1]
    fit_cm1  = np.linspace(cm1_arr.min(), cm1_arr.max(), 200)
    fit_err  = 10**(intercept + slope * np.log10(fit_cm1))

    fig, ax = plt.subplots(figsize=(7.1, 5.0))

    ax.loglog(cm1_arr, err_arr, 'o', color='#d62728', markersize=7,
              label='Shao error at $t = 5T$')
    ax.loglog(fit_cm1, fit_err, '--', color='gray', lw=1.5,
              label=f'Linear fit: slope = {slope:.2f}')

    # Label inclination angles
    for idx, (cm1, err, i_deg) in enumerate(zip(cm1_arr, err_arr, inclinations_deg)):
        ax.annotate(f'{i_deg:.0f}°', xy=(cm1, err),
                    xytext=(5, 3), textcoords='offset points',
                    fontsize=7, color='#333333')

    ax.set_xlabel(r'$|c - 1|$')
    ax.set_ylabel(r'$|\delta y_{\rm Shao} - \delta y_{\rm num}|$ at $t=5T$ [m]')
    ax.set_title(r'Parametric Study: Shao $\delta y$ Error vs $|c-1|$ — Case C',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # Annotate slope
    ax.text(0.05, 0.92, f'Fitted slope $\\approx$ {slope:.2f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    fig.tight_layout()
    return fig
