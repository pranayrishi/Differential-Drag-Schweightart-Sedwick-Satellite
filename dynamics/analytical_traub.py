"""
Corrected closed-form solution to the Schweighart-Sedwick equations.

Reference:
    Traub, Neubert, Ingrillini (2025)
    "Corrected Closed-Form Solutions to the Schweighart-Sedwick Satellite
     Relative Motion Model Including Differential Drag"
    Acta Astronautica 234, 742-753.

Derivation summary
------------------
The in-plane SS equations (EOM_x, EOM_y) with constant forcing are:

    EOM_x:  δẍ − 2n δẏ − (5c²−2)n² δx = δfₓ
    EOM_y:  δÿ + 2cn δẋ              = δfᵧ

Step 1 — integrate EOM_y once (exact first integral):
    δẏ + 2cn δx = δfᵧ t + K₁    where K₁ = δẏ₀ + 2cn δx₀   (constant)

Step 2 — substitute into EOM_x to decouple:
    δẍ + ω²_SS δx = δfₓ + 2n K₁ + 2n δfᵧ t
    where ω²_SS = n²(4c + 2 − 5c²)

Step 3 — particular solution (polynomial ansatz δxₚ = A + Bt):
    A = (δfₓ + 2n K₁) / ω²_SS
    B = 2n δfᵧ / ω²_SS          ← CORRECTED (Traub)
      = 2 δfᵧ / n only when c=1

    With this xₚ the homogeneous remainder is a pure oscillator:
    δẍₕ + ω²_SS δxₕ = 0

Step 4 — homogeneous solution:
    δxₕ = C₁ cos(ω_SS t) + C₂ sin(ω_SS t)
    C₁ = δx₀ − A,   C₂ = (δẋ₀ − B) / ω_SS

Step 5 — δy from the first integral (direct quadrature):
    δẏ = δfᵧ t + K₁ − 2cn δx
    δy(t) = δy₀ + K₁ t + ½ δfᵧ t² − 2cn ∫₀ᵗ δx(s) ds
"""

import numpy as np


def SS_analytical_traub(t, delta_x0, delta_xdot0, delta_y0, delta_ydot0,
                         delta_fx, delta_fy, n, c):
    """
    Corrected analytical solution of the in-plane SS equations with constant forcing.

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

    # ------------------------------------------------------------------ #
    # Step 1: First integral of EOM_y                                      #
    # K1 = delta_ydot0 + 2*c*n*delta_x0   (constant of integration)       #
    # ------------------------------------------------------------------ #
    K1 = delta_ydot0 + 2.0 * c * n * delta_x0

    # ------------------------------------------------------------------ #
    # Step 2–3: Particular solution for delta_x                           #
    # delta_xdd + omega_SS^2 * delta_x = delta_fx + 2n*K1 + 2n*delta_fy*t #
    # Ansatz: delta_x_p = A + B*t                                          #
    # ------------------------------------------------------------------ #
    A = (delta_fx + 2.0 * n * K1) / omega_SS**2   # constant offset
    B = 2.0 * n * delta_fy / omega_SS**2           # ramp coefficient (CORRECTED)

    # ------------------------------------------------------------------ #
    # Step 4: Homogeneous solution — pure oscillator at omega_SS           #
    # ------------------------------------------------------------------ #
    C1 = delta_x0 - A
    C2 = (delta_xdot0 - B) / omega_SS

    cos_wt = np.cos(omega_SS * t)
    sin_wt = np.sin(omega_SS * t)

    delta_x    = C1 * cos_wt + C2 * sin_wt + A + B * t
    delta_xdot = -C1 * omega_SS * sin_wt + C2 * omega_SS * cos_wt + B

    # ------------------------------------------------------------------ #
    # Step 5: delta_y by direct quadrature of the first integral           #
    # delta_ydot = delta_fy * t + K1 - 2*c*n * delta_x                    #
    # int_0^t delta_x ds = C1*sin(wt)/w + C2*(1-cos(wt))/w + A*t + B*t^2/2 #
    # ------------------------------------------------------------------ #
    int_x = (C1 * sin_wt / omega_SS
             + C2 * (1.0 - cos_wt) / omega_SS
             + A * t
             + 0.5 * B * t**2)

    delta_y    = delta_y0 + K1 * t + 0.5 * delta_fy * t**2 - 2.0 * c * n * int_x
    delta_ydot = delta_fy * t + K1 - 2.0 * c * n * delta_x

    if scalar_input:
        return (float(delta_x[0]), float(delta_xdot[0]),
                float(delta_y[0]), float(delta_ydot[0]))
    return delta_x, delta_xdot, delta_y, delta_ydot
