"""
poole_frenkel.py
================
Field-assisted (Poole–Frenkel) emission from a trap to the conduction band:

R_PF = ν0 * exp(-(E_T - ΔE_PF)/(kB T)),
ΔE_PF = sqrt(q^3 F / (π ε0 ε_r))   # using optical permittivity if desired
"""

import numpy as np
from const.constants import q, k_B, eps0, pi

def poole_frenkel_rate(E_T_eV, F_Vpm, T_K, eps_r=5.6, nu0=1e13):
    """
    Parameters
    ----------
    E_T_eV : float
        Trap depth to CB minimum [eV].
    F_Vpm : float
        Local electric field magnitude [V/m]. (Use bias/Lx or later from Poisson)
    T_K : float
        Temperature [K].
    eps_r : float
        Relative permittivity (use ε_opt for image-lowering if you wish).
    nu0 : float
        Attempt frequency [Hz].

    Returns
    -------
    R_pf : float
        PF emission rate [s^-1].
    """
    E_T = E_T_eV * q
    deltaE = np.sqrt(q**3 * F_Vpm / (pi * eps0 * eps_r))  # PF lowering
    barrier = max(E_T - deltaE, 0.0)
    return nu0 * np.exp(-barrier / (k_B * T_K))
