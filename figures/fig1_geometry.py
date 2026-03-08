"""
Figure 1 — Formation flight geometry schematic.

Shows the LVLH (Local Vertical Local Horizontal) reference frame:
  x-hat: radial (outward from Earth centre)
  y-hat: along-track (direction of orbital motion)
  z-hat: cross-track (normal to orbital plane)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch


def plot_figure_1():
    """Create and return the formation geometry schematic (Fig. 1)."""
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.set_aspect('equal')
    ax.axis('off')

    # ----- Earth -----
    earth = plt.Circle((0, 0), 1.0, color='#4477AA', alpha=0.85, zorder=2)
    ax.add_patch(earth)
    # Continents hint (simple ellipses)
    for (cx, cy, w, h, ang) in [(0.0, 0.3, 0.6, 0.35, 30),
                                  (-0.3, -0.1, 0.4, 0.25, -20),
                                  (0.4, -0.3, 0.3, 0.2, 10)]:
        cont = patches.Ellipse((cx, cy), w, h, angle=ang,
                                color='#3A8A3A', alpha=0.7, zorder=3)
        ax.add_patch(cont)
    ax.text(0, 0, 'Earth', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=4)

    # ----- Chief orbit (dashed circle) -----
    theta = np.linspace(0, 2 * np.pi, 500)
    R_orbit = 1.65
    ax.plot(R_orbit * np.cos(theta), R_orbit * np.sin(theta),
            'k--', linewidth=1.2, alpha=0.6, zorder=1)

    # ----- Chief spacecraft -----
    chief_angle = np.radians(50)
    cx = R_orbit * np.cos(chief_angle)
    cy = R_orbit * np.sin(chief_angle)
    ax.plot(cx, cy, 's', markersize=10, color='#1f77b4', zorder=5,
            label='Chief spacecraft')
    ax.text(cx + 0.05, cy + 0.12, 'Chief\n(reference)',
            fontsize=9, ha='left', va='bottom', color='#1f77b4', zorder=6)

    # ----- LVLH axes at chief -----
    # x-hat: radial outward
    x_hat = np.array([np.cos(chief_angle), np.sin(chief_angle)])
    # y-hat: along-track (perpendicular to radial, in direction of motion)
    y_hat = np.array([-np.sin(chief_angle), np.cos(chief_angle)])

    arrow_len = 0.35
    ax.annotate('', xy=(cx + arrow_len * x_hat[0], cy + arrow_len * x_hat[1]),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.8),
                zorder=7)
    ax.text(cx + (arrow_len + 0.07) * x_hat[0],
            cy + (arrow_len + 0.07) * x_hat[1],
            r'$\hat{x}$ (radial)', fontsize=9, color='red',
            ha='center', va='center', zorder=7)

    ax.annotate('', xy=(cx + arrow_len * y_hat[0], cy + arrow_len * y_hat[1]),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.8),
                zorder=7)
    ax.text(cx + (arrow_len + 0.1) * y_hat[0],
            cy + (arrow_len + 0.1) * y_hat[1],
            r'$\hat{y}$ (along-track)', fontsize=9, color='darkgreen',
            ha='center', va='center', zorder=7)

    # ----- Deputy spacecraft -----
    # Place it offset in radial + along-track
    rho_x =  0.15   # radial offset (in LVLH x)
    rho_y =  0.30   # along-track offset (in LVLH y)
    dx = cx + rho_x * x_hat[0] + rho_y * y_hat[0]
    dy = cy + rho_x * x_hat[1] + rho_y * y_hat[1]
    ax.plot(dx, dy, '^', markersize=10, color='#d62728', zorder=5,
            label='Deputy spacecraft')
    ax.text(dx + 0.05, dy + 0.08, 'Deputy\n(follower)',
            fontsize=9, ha='left', va='bottom', color='#d62728', zorder=6)

    # ----- Relative position vector rho -----
    ax.annotate('', xy=(dx, dy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2.0,
                                linestyle='solid'),
                zorder=7)
    mid_x = 0.5 * (cx + dx)
    mid_y = 0.5 * (cy + dy)
    ax.text(mid_x - 0.12, mid_y + 0.05,
            r'$\boldsymbol{\rho}$', fontsize=13, color='purple',
            fontstyle='italic', zorder=7)

    # ----- Earth centre to chief (dashed) -----
    ax.plot([0, cx], [0, cy], 'gray', linestyle=':', linewidth=1.0, zorder=1)

    # ----- Velocity arrow (orbit direction) -----
    v_angle = chief_angle + np.radians(90)
    ax.annotate('', xy=(cx + 0.25 * np.cos(v_angle),
                         cy + 0.25 * np.sin(v_angle)),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color='navy', lw=1.5,
                                linestyle='dashed'),
                zorder=6)

    # ----- Legend proxies -----
    chief_patch  = mpatches.Patch(color='#1f77b4', label='Chief spacecraft')
    deputy_patch = mpatches.Patch(color='#d62728', label='Deputy spacecraft')
    ax.legend(handles=[chief_patch, deputy_patch],
              loc='lower right', fontsize=9)

    # ----- Labels -----
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_title('Formation Flight Geometry — LVLH Reference Frame',
                 fontsize=11, pad=8)

    fig.tight_layout()
    return fig
