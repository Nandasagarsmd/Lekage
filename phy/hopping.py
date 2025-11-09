"""
hopping.py
===========

Implements defect–defect tunneling (hopping) models:

1. Elastic Mott hopping (Eq. 3.42)
   R_ij = ν₀ * exp(-2 * r_ij / r_D)

2. Inelastic Miller–Abrahams (Eq. 3.45)
   R_ij = ν₀ * exp(-2 * r_ij / r_D) * exp(-ΔE / k_BT) for ΔE > 0
         = ν₀ * exp(-2 * r_ij / r_D) for ΔE ≤ 0

These describe electron transitions between localized trap states in the band gap.
"""

import numpy as np
from const.constants import k_B, q, hbar, me

# --------------------------------------------------------------------------- #
# --- Localization radius (Eq. 3.44) --------------------------------------- #
# --------------------------------------------------------------------------- #
def localization_radius(E_D_eV):
    """
    Compute localization radius r_D = ħ / sqrt(2 m* E_D)
    Parameters
    ----------
    E_D_eV : float
        Defect depth [eV].
    Returns
    -------
    r_D : float
        Localization radius [m].
    """
    E_D = E_D_eV * q
    return hbar / np.sqrt(2 * me * E_D)


# --------------------------------------------------------------------------- #
# --- Elastic Mott hopping (Eq. 3.42) -------------------------------------- #
# --------------------------------------------------------------------------- #
def elastic_mott_rate(r_ij, E_D_eV, nu0=1e13):
    """
    Elastic defect–defect hopping rate (Mott model).

    Parameters
    ----------
    r_ij : float
        Distance between defects [m].
    E_D_eV : float
        Defect depth [eV].
    nu0 : float
        Attempt frequency [Hz].

    Returns
    -------
    R_ij : float
        Hopping rate [s⁻¹].
    """
    r_D = localization_radius(E_D_eV)
    return nu0 * np.exp(-2 * r_ij / r_D)


# --------------------------------------------------------------------------- #
# --- Inelastic Miller–Abrahams hopping (Eq. 3.45) ------------------------- #
# --------------------------------------------------------------------------- #
def inelastic_ma_rate(r_ij, ΔE_eV, E_D_eV, T_K, nu0=1e13):
    """
    Inelastic (phonon-assisted) hopping between two defects (Miller–Abrahams model).

    Parameters
    ----------
    r_ij : float
        Distance between defects [m].
    ΔE_eV : float
        Energy difference (E_final - E_initial) [eV].
    E_D_eV : float
        Average defect depth [eV].
    T_K : float
        Temperature [K].
    nu0 : float
        Attempt frequency [Hz].

    Returns
    -------
    R_ij : float
        Hopping rate [s⁻¹].
    """
    r_D = localization_radius(E_D_eV)
    base = nu0 * np.exp(-2 * r_ij / r_D)

    if ΔE_eV > 0:
        return base * np.exp(-ΔE_eV * q / (k_B * T_K))
    else:
        return base


# --------------------------------------------------------------------------- #
# --- Unified interface ----------------------------------------------------- #
# --------------------------------------------------------------------------- #
def hopping_rate(r_i, r_j, E_i_eV, E_j_eV, T_K=300, model="MA", nu0=1e13):
    """
    Compute hopping rate between two defect sites given their
    positions and energy levels.

    Parameters
    ----------
    r_i, r_j : ndarray
        3D coordinates [m] of defects i and j.
    E_i_eV, E_j_eV : float
        Defect energy levels [eV].
    T_K : float
        Temperature [K].
    model : str
        "Mott" for elastic or "MA" for inelastic Miller–Abrahams.
    nu0 : float
        Attempt frequency [Hz].

    Returns
    -------
    R_ij : float
        Hopping rate [s⁻¹].
    """
    r_ij = np.linalg.norm(np.array(r_i) - np.array(r_j))
    E_D_mean = 0.5 * (E_i_eV + E_j_eV)
    ΔE_eV = E_j_eV - E_i_eV

    if model.lower() == "mott":
        return elastic_mott_rate(r_ij, E_D_mean, nu0)
    elif model.lower() == "ma":
        return inelastic_ma_rate(r_ij, ΔE_eV, E_D_mean, T_K, nu0)
    else:
        raise ValueError("model must be 'Mott' or 'MA'")
