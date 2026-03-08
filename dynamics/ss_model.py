"""
Schweighart-Sedwick (SS) model coefficients and derived quantities.

References:
    Schweighart & Sedwick (2002), JGCD 25(6), 1073-1080.
    Traub, Neubert, Ingrillini (2025), Acta Astronautica 234, 742-753.
"""

import numpy as np
from .constants import mu, R_E, J2


def compute_SS_coefficient(a, e, i):
    """
    Compute the Schweighart-Sedwick J2-correction coefficient c.

    Parameters
    ----------
    a : float
        Semi-major axis [m]
    e : float
        Eccentricity [-]
    i : float
        Inclination [rad]

    Returns
    -------
    c : float
        SS coefficient (c >= 1 for typical LEO orbits)
    """
    p = a * (1.0 - e**2)
    c_squared = 1.0 + (3.0 * J2 * R_E**2) / (2.0 * p**2) * (1.0 - 1.5 * np.sin(i)**2)
    return np.sqrt(c_squared)


def compute_mean_motion(a):
    """
    Compute the Keplerian mean motion.

    Parameters
    ----------
    a : float
        Semi-major axis [m]

    Returns
    -------
    n : float
        Mean motion [rad/s]
    """
    return np.sqrt(mu / a**3)


def compute_SS_frequency(n, c):
    """
    Compute the in-plane oscillation frequency of the SS equations.

    For the Clohessy-Wiltshire (HCW) limit c=1, omega_SS = n.

    Parameters
    ----------
    n : float
        Mean motion [rad/s]
    c : float
        SS coefficient [-]

    Returns
    -------
    omega_SS : float
        SS oscillation frequency [rad/s]
    """
    omega_SS_sq = n**2 * (4.0 * c + 2.0 - 5.0 * c**2)
    return np.sqrt(omega_SS_sq)
