"""
for defect-based transport (elastic hopping, inelastic hopping, localization radius, etc.)
"""

import numpy as np
from const.constants import q, hbar, me


def localization_radius(E_D):
    """
    Compute localization radius:
        r_D = ħ / sqrt(2 m* E_D)
    Parameters
    ----------
    E_D : float
        Defect depth in Joules.
    """
    return hbar / np.sqrt(2 * me * E_D)


def elastic_hopping_rate(r_i, r_j, E_D=0.3*q, nu=1e13):
    """
    Mott elastic hopping rate between two defect sites.

    Parameters
    ----------
    r_i, r_j : ndarray
        3D coordinates (in meters) of the two defect sites.
    E_D : float
        Defect depth (J). Default: 0.3 eV.
    nu : float
        Attempt frequency (Hz). Default: 1e13.

    Returns
    -------
    R_ij : float
        Hopping rate in s⁻¹
    """
    r_ij = np.linalg.norm(r_i - r_j)
    r_D = localization_radius(E_D)
    return nu * np.exp(-2 * r_ij / r_D)


def inelastic_hopping_rate(r_i, r_j, E_i, E_j, E_D=0.3*q, nu=1e13, T=300):
    """
    Miller–Abrahams inelastic hopping rate
    Eq. (3.45): R_ij = ν exp(-2r_ij/r_D) * exp(-ΔE/kT) if ΔE>0 else 1
    """
    k_B = 1.381e-23
    r_ij = np.linalg.norm(r_i - r_j)
    r_D = localization_radius(E_D)

    base = nu * np.exp(-2 * r_ij / r_D)
    ΔE = (E_j - E_i) * q  # convert eV → J
    if ΔE > 0:
        return base * np.exp(-ΔE / (k_B * T))
    else:
        return base

