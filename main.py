"""
Reproduction of:
    Traub, Neubert, Ingrillini (2025)
    "Corrected Closed-Form Solutions to the Schweighart-Sedwick Satellite
     Relative Motion Model Including Differential Drag"
    Acta Astronautica 234, 742-753.

Generates all 9 figures and runs 5 validation tests.
Outputs saved to ./outputs/ as both PDF and PNG.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamics.constants import mu, R_E, J2
from dynamics.ss_model import compute_mean_motion, compute_SS_coefficient
from dynamics.numerical_integrator import SS_numerical_integration
from dynamics.analytical_traub import SS_analytical_traub
from dynamics.analytical_shao import SS_analytical_shao

# ------------------------------------------------------------------ #
# Matplotlib style                                                      #
# ------------------------------------------------------------------ #
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   12,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'lines.linewidth':  1.5,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'text.usetex':      False,
    'mathtext.fontset': 'stix',
})

DOUBLE_COL      = (7.1, 5.0)
DOUBLE_COL_TALL = (7.1, 6.5)

# ------------------------------------------------------------------ #
# Orbital parameters                                                   #
# ------------------------------------------------------------------ #
a  = 6871.0e3          # m  (500 km altitude)
e  = 0.0
i  = np.radians(98.0)  # Sun-synchronous
n  = compute_mean_motion(a)
T  = 2.0 * np.pi / n
c  = compute_SS_coefficient(a, e, i)

print('=' * 60)
print('  Schweighart-Sedwick Reproduction — Traub et al. (2025)')
print('=' * 60)
print(f'\nOrbital parameters:')
print(f'  a   = {a/1e3:.1f} km')
print(f'  i   = {np.degrees(i):.1f} deg  (sun-synchronous)')
print(f'  n   = {n:.6e} rad/s')
print(f'  T   = {T:.1f} s = {T/60:.2f} min')
print(f'  c   = {c:.8f}')
print(f'  c-1 = {c-1:.4e}')
print(f'  omega_SS / n = {np.sqrt(n**2*(4*c+2-5*c**2))/n:.8f}')

# ------------------------------------------------------------------ #
# Initial conditions (bounded orbit)                                   #
# ------------------------------------------------------------------ #
delta_x0    = 100.0              # m
delta_xdot0 = 0.0                # m/s
delta_y0    = 0.0                # m
delta_ydot0 = -2.0 * c * n * delta_x0   # m/s  (bounded orbit condition)
delta_z0    = 0.0
delta_zdot0 = 0.0

y0 = [delta_x0, delta_xdot0, delta_y0, delta_ydot0, delta_z0, delta_zdot0]

print(f'\nInitial conditions:')
print(f'  delta_x0    = {delta_x0:.1f} m')
print(f'  delta_xdot0 = {delta_xdot0:.3f} m/s')
print(f'  delta_y0    = {delta_y0:.1f} m')
print(f'  delta_ydot0 = {delta_ydot0:.6f} m/s  (bounded orbit IC)')

# ------------------------------------------------------------------ #
# Time grid                                                            #
# ------------------------------------------------------------------ #
N_points = 5000
t_eval   = np.linspace(0, 5 * T, N_points)

# ------------------------------------------------------------------ #
# Output directory                                                     #
# ------------------------------------------------------------------ #
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

def save_fig(fig, name):
    """Save figure as PDF and PNG."""
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(path)
    print(f'  Saved {name}.pdf / .png')
    plt.close(fig)

# ------------------------------------------------------------------ #
# Generate all figures                                                 #
# ------------------------------------------------------------------ #
print('\n' + '-'*60)
print('  Generating figures …')
print('-'*60)

# Figure 1 — geometry schematic
from figures.fig1_geometry import plot_figure_1
print('Fig 1: Formation geometry schematic')
fig = plot_figure_1()
save_fig(fig, 'fig1_geometry')

# Figure 2 — state decomposition
from figures.fig2_decomposition import plot_figure_2
print('Fig 2: State decomposition')
fig = plot_figure_2(T)
save_fig(fig, 'fig2_decomposition')

# Figure 3 — no force
from figures.fig3_noforce import plot_figure_3
print('Fig 3: Case A — no force')
fig = plot_figure_3(n, c, T, y0, t_eval)
save_fig(fig, 'fig3_noforce')

# Figure 4 — radial force
from figures.fig4_radial import plot_figure_4
print('Fig 4: Case B — radial force only')
fig = plot_figure_4(n, c, T, y0, t_eval)
save_fig(fig, 'fig4_radial')

# Figure 5 — tangential force (KEY)
from figures.fig5_tangential import plot_figure_5
print('Fig 5: Case C — tangential force only  [KEY FIGURE]')
fig = plot_figure_5(n, c, T, y0, t_eval)
save_fig(fig, 'fig5_tangential')

# Figure 6 — combined force
from figures.fig6_combined import plot_figure_6
print('Fig 6: Case D — combined radial + tangential force')
fig = plot_figure_6(n, c, T, y0, t_eval)
save_fig(fig, 'fig6_combined')

# Figure 7 — errors
from figures.fig7_errors import plot_figure_7
print('Fig 7: Error comparison (semi-log)')
fig = plot_figure_7(n, c, T, y0, t_eval)
save_fig(fig, 'fig7_errors')

# Figure 8 — parametric study
from figures.fig8_parametric import plot_figure_8
print('Fig 8: Parametric study — error vs (c-1)')
fig = plot_figure_8(a, e, n, T)
save_fig(fig, 'fig8_parametric')

# Figure 9 — phase plane
from figures.fig9_phase_plane import plot_figure_9
print('Fig 9: Phase plane trajectories')
fig = plot_figure_9(n, c, T)
save_fig(fig, 'fig9_phase_plane')

# ------------------------------------------------------------------ #
# Validation tests                                                     #
# ------------------------------------------------------------------ #
from validation.verify_CW_reduction import run_all_tests
results = run_all_tests(n, c, t_eval, y0, a=a, e=e)

# ------------------------------------------------------------------ #
# Summary table of force cases                                         #
# ------------------------------------------------------------------ #
print('\n' + '='*60)
print('  Summary — Final-time values at t = 5T')
print('='*60)
print(f'{"Case":<10} {"fx [m/s²]":>12} {"fy [m/s²]":>12} '
      f'{"dx_num [m]":>12} {"dy_num [m]":>14} {"dy_Shao [m]":>14} {"dy_Traub [m]":>14}')
print('-'*90)

cases = [
    ('A (none)',      0.0,  0.0),
    ('B (radial)',  1e-7,   0.0),
    ('C (tang.)',    0.0,  1e-7),
    ('D (both)',   1e-7,  1e-7),
]

for label, dfx, dfy in cases:
    _, sol_n = SS_numerical_integration(
        (t_eval[0], t_eval[-1]), y0, dfx, dfy, 0.0,
        n, c, t_eval=t_eval)
    _, _, dy_t, _ = SS_analytical_traub(
        t_eval, delta_x0, delta_xdot0, delta_y0, delta_ydot0, dfx, dfy, n, c)
    _, _, dy_s, _ = SS_analytical_shao(
        t_eval, delta_x0, delta_xdot0, delta_y0, delta_ydot0, dfx, dfy, n, c)

    print(f'{label:<10} {dfx:>12.1e} {dfy:>12.1e} '
          f'{sol_n[0,-1]:>12.4f} {sol_n[2,-1]:>14.4f} '
          f'{dy_s[-1]:>14.4f} {dy_t[-1]:>14.4f}')

print('='*60)
print(f'\nAll outputs saved to: {OUT_DIR}')
print('Done.')
