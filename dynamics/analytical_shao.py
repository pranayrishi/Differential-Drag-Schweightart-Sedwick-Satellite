"""
Erroneous closed-form solution attributed to Shao et al.

This reproduces the INCORRECT solution that uses the Hill-Clohessy-Wiltshire
particular-solution coefficient for the secular (ramp) term under a
constant along-track force:

    B = 2 δfᵧ / n                       (SHAO — WRONG for c ≠ 1)

The correct value (Traub et al., 2025) is:

    B = 2n δfᵧ / ω²_SS = 2n δfᵧ / [n²(4c+2−5c²)]

These coincide only when c = 1 (Keplerian, ω_SS = n).

For c ≠ 1 the mismatch generates a spurious term

    (B_wrong − B_correct) × t

in δx which, via the y quadrature, produces a quadratically growing error
in δy proportional to (c − 1).
"""

import numpy as np


def SS_analytical_shao(t, delta_x0, delta_xdot0, delta_y0, delta_ydot0,
                        delta_fx, delta_fy, n, c):
    """
    Erroneous analytical solution of the in-plane SS equations (Shao et al.).

    Identical derivation structure to the Traub solution except the
    particular-solution ramp coefficient uses the HCW value 2*delta_fy/n
    instead of 2*n*delta_fy/omega_SS^2.

    Parameters
    ----------
    t : float or array_like
        Evaluation time(s) [s]
    delta_x0 : float
        Initial radial separation [m]
    delta_xdot0 : float
        Initial radial relative velocity [m/s]
    delta_y0 : float
        Initial along-track separation [m]
    delta_ydot0 : float
        Initial along-track relative velocity [m/s]
    delta_fx : float
        Constant radial differential acceleration [m/s^2]
    delta_fy : float
        Constant along-track differential acceleration [m/s^2]
    n : float
        Mean motion [rad/s]
    c : float
        SS coefficient [-]

    Returns
    -------
    delta_x : ndarray
        Radial relative position [m]
    delta_xdot : ndarray
        Radial relative velocity [m/s]
    delta_y : ndarray
        Along-track relative position [m]
    delta_ydot : ndarray
        Along-track relative velocity [m/s]
    """
    t = np.asarray(t, dtype=float)
    scalar_input = t.ndim == 0
    t = np.atleast_1d(t)

    omega_SS = np.sqrt(n**2 * (4.0 * c + 2.0 - 5.0 * c**2))

    # First integral of EOM_y
    K1 = delta_ydot0 + 2.0 * c * n * delta_x0

    # Particular solution for delta_x — WRONG ramp coefficient
    A        = (delta_fx + 2.0 * n * K1) / omega_SS**2
    B_wrong  = 2.0 * delta_fy / n          # HCW value — INCORRECT for c ≠ 1

    # Homogeneous (pure oscillator at omega_SS)
    C1 = delta_x0 - A
    C2 = (delta_xdot0 - B_wrong) / omega_SS

    cos_wt = np.cos(omega_SS * t)
    sin_wt = np.sin(omega_SS * t)

    delta_x    = C1 * cos_wt + C2 * sin_wt + A + B_wrong * t
    delta_xdot = -C1 * omega_SS * sin_wt + C2 * omega_SS * cos_wt + B_wrong

    # delta_y by quadrature (using the wrong x):
    # int_0^t delta_x_wrong ds
    int_x = (C1 * sin_wt / omega_SS
             + C2 * (1.0 - cos_wt) / omega_SS
             + A * t
             + 0.5 * B_wrong * t**2)

    delta_y    = delta_y0 + K1 * t + 0.5 * delta_fy * t**2 - 2.0 * c * n * int_x
    delta_ydot = delta_fy * t + K1 - 2.0 * c * n * delta_x

    if scalar_input:
        return (float(delta_x[0]), float(delta_xdot[0]),
                float(delta_y[0]), float(delta_ydot[0]))
    return delta_x, delta_xdot, delta_y, delta_ydot
