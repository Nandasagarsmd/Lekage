"""
direct_tunneling.py
===================

Implements direct tunneling, Schottky (SE), and Fowler–Nordheim (FN) tunneling
currents/rates through a dielectric barrier as described in the kMC model.

Equations implemented:
    - (3.12) WKB transmission coefficient
    - (3.20) Schottky (thermionic) emission current
    - (3.21) Fowler–Nordheim tunneling current

Returned quantity:
    Rate per site [s⁻¹]  (converted from current density J/q * A_cell)
"""

import numpy as np
from const.constants import q, k_B, hbar, eps0, me, pi

# --------------------------------------------------------------------------- #
# --- Helper: WKB transmission coefficient ---------------------------------- #
# --------------------------------------------------------------------------- #
def wkb_transmission(E, Vx, x, m_eff):
    """
    Compute transmission coefficient T(E) ≈ exp(-2 ∫ κ(x) dx)
    where κ(x) = sqrt(2 m_eff (V(x) - E)) / ħ.
    """
    V_interp = np.interp(x, np.linspace(x[0], x[-1], len(Vx)), Vx)
    mask = V_interp > E
    if not np.any(mask):
        return 1.0  # energy above barrier
    kappa = np.sqrt(2 * m_eff * np.maximum(V_interp - E, 0)) / hbar
    integral = np.trapezoid(kappa, x)
    return np.exp(-2 * integral)


# --------------------------------------------------------------------------- #
# --- Schottky emission (Eq. 3.20) ----------------------------------------- #
# --------------------------------------------------------------------------- #
def schottky_current_density(F, T, EB_eV, eps_r, m_eff=me):
    """
    Richardson-Schottky emission current density [A/m^2].
    Eq. (3.20): j_SE = (e m* kB^2 T^2)/(2π^2 ħ^3)
                  * exp[-(EB - ΔE_SE)/(kB T)]
    """
    EB = EB_eV * q
    deltaE = np.sqrt(q**3 * F / (4 * pi * eps0 * eps_r))
    prefactor = (q * m_eff * (k_B * T)**2) / (2 * (pi**2) * (hbar**3))
    exponent = - (EB - deltaE) / (k_B * T)
    return prefactor * np.exp(exponent)


# --------------------------------------------------------------------------- #
# --- Fowler-Nordheim tunneling (Eq. 3.21) --------------------------------- #
# --------------------------------------------------------------------------- #
def fowler_nordheim_current_density(F, EB_eV, m_eff=me):
    """
    FN tunneling current density [A/m^2].
    Eq. (3.21): j_FN = (e^3 F^2)/(8π h EB)
                * exp[- (4 sqrt(2 m*) EB^(3/2)) / (3 ħ e F)]
    """
    EB = EB_eV * q
    h = 2 * pi * hbar
    prefactor = (q**3 * F**2) / (8 * pi * h * EB)
    exponent = - (4 * np.sqrt(2 * m_eff) * (EB**1.5)) / (3 * hbar * q * F)
    return prefactor * np.exp(exponent)


# --------------------------------------------------------------------------- #
# --- Unified direct-tunneling rate ---------------------------------------- #
# --------------------------------------------------------------------------- #
def direct_tunneling_rate(F, T, EB_eV, eps_r=5.6, A_cell=1e-18, m_eff=0.5*me, mode='auto'):
    """
    Compute rate per site for electron transmission across the dielectric barrier.

    Parameters
    ----------
    F : float
        Electric field [V/m].
    T : float
        Temperature [K].
    EB_eV : float
        Barrier height (CBO) [eV].
    eps_r : float
        Relative permittivity of dielectric.
    A_cell : float
        Representative area per defect [m²].
    m_eff : float
        Effective mass for tunneling [kg].
    mode : str
        'auto' | 'FN' | 'SE' | 'WKB' to select computation mode.

    Returns
    -------
    R : float
        Transition rate [s⁻¹].
    """
    # --- decide regime ---
    if mode == 'auto':
        # Rough heuristic: high field (>2e6 V/cm) ⇒ FN; low field ⇒ SE
        if F > 2e8 and T < 350:
            mode = 'FN'
        else:
            mode = 'SE'

    # --- compute current density ---
    if mode == 'SE':
        J = schottky_current_density(F, T, EB_eV, eps_r, m_eff)
    elif mode == 'FN':
        J = fowler_nordheim_current_density(F, EB_eV, m_eff)
    else:
        raise ValueError("mode must be 'auto', 'SE', or 'FN'")

    # --- convert to rate per site ---
    R = (J * A_cell) / q  # [A = C/s] → [s⁻¹]
    return R
