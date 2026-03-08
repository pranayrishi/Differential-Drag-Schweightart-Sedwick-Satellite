"""
High-accuracy numerical integration of the Schweighart-Sedwick equations of motion.

The SS equations (in-plane + out-of-plane):
    delta_xdd - 2n*delta_yd - (5c^2 - 2)*n^2*delta_x = delta_fx
    delta_ydd + 2*c*n*delta_xd                         = delta_fy
    delta_zdd + c^2*n^2*delta_z                        = delta_fz
"""

import numpy as np
from scipy.integrate import solve_ivp


def SS_numerical_integration(t_span, y0, delta_fx, delta_fy, delta_fz,
                              n, c, t_eval=None):
    """
    Numerically integrate the SS equations of motion using DOP853.

    Parameters
    ----------
    t_span : (float, float)
        Start and end times [s]
    y0 : array_like, shape (6,)
        Initial state [delta_x, delta_xdot, delta_y, delta_ydot, delta_z, delta_zdot] [m, m/s, ...]
    delta_fx : float
        Constant radial differential force/acceleration [m/s^2]
    delta_fy : float
        Constant along-track differential force/acceleration [m/s^2]
    delta_fz : float
        Constant out-of-plane differential force/acceleration [m/s^2]
    n : float
        Mean motion [rad/s]
    c : float
        SS coefficient [-]
    t_eval : array_like or None
        Times at which to store solution

    Returns
    -------
    t : ndarray
        Solution times [s]
    y : ndarray, shape (6, len(t))
        Solution state matrix
    """
    def EOM(t, y):
        dx, dxd, dy, dyd, dz, dzd = y
        dxdd = 2.0 * n * dyd + (5.0 * c**2 - 2.0) * n**2 * dx + delta_fx
        dydd = -2.0 * c * n * dxd + delta_fy
        dzdd = -c**2 * n**2 * dz + delta_fz
        return [dxd, dxdd, dyd, dydd, dzd, dzdd]

    sol = solve_ivp(
        EOM,
        t_span,
        y0,
        method='DOP853',
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-14,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"Numerical integration failed: {sol.message}")

    return sol.t, sol.y
