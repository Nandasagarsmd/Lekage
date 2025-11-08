"""
for defect-based transport (elastic hopping, localization radius, etc.)
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
