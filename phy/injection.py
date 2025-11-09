"""
injection.py
=============

Implements elastic tunneling *into* and *out of* defect states from electrodes,
according to Eqs. (3.26)–(3.27):

    R_ED = C * f(Ee) * T(Ee)
    R_DE = C * [1 - f(Ee)] * T(Ee)

where:
    - f(Ee): Fermi–Dirac occupation in electrode
    - T(Ee): transmission coefficient through barrier (via WKB or analytic)
    - C: normalization constant depending on masses and energies

This module provides:
    - injection_rate(Ee, F, T, E_D)
    - emission_rate(Ee, F, T, E_D)
"""

import numpy as np
from const.constants import q, k_B, hbar, me
from phy.direct_tunneling import direct_tunneling_rate

# --------------------------------------------------------------------------- #
# --- Helper: Fermi-Dirac function ----------------------------------------- #
# --------------------------------------------------------------------------- #
def fermi_dirac(E, Ef, T):
    """Return Fermi–Dirac occupation probability at energy E [J]."""
    beta = 1 / (k_B * T)
    return 1.0 / (1.0 + np.exp((E - Ef) * beta))


# --------------------------------------------------------------------------- #
# --- Normalization constant (Eq. 3.26) ------------------------------------ #
# --------------------------------------------------------------------------- #
def prefactor_C(m_e=me, m_i=0.5*me, E_e_eV=1.0, E_D_eV=1.0):
    """
    C = (m_e/m_i)^(5/2) * [8 * E_e^(3/2)] / [3 ħ √E_D]
    Inputs in eV, output in [1/s] scaling.
    """
    E_e = E_e_eV * q
    E_D = E_D_eV * q
    C = ((m_e / m_i)**(2.5)) * (8 * (E_e**1.5)) / (3 * hbar * np.sqrt(E_D))
    return C


# --------------------------------------------------------------------------- #
# --- Injection rate: electrode → defect (Eq. 3.26) ------------------------ #
# --------------------------------------------------------------------------- #
def injection_rate(E_D_eV, F, T_K, Ef_eV=0.0, EB_eV=1.0, eps_r=5.6):
    """
    Compute injection rate [s⁻¹] from electrode into defect at depth E_D.
    Uses Eq. (3.26): R_ED = C * f(Ee) * T(Ee)
    """
    E_e_eV = EB_eV - E_D_eV  # electron energy relative to barrier top
    C = prefactor_C(E_e_eV=E_e_eV, E_D_eV=E_D_eV)
    f = fermi_dirac(E_e_eV*q, Ef_eV*q, T_K)
    T_E = direct_tunneling_rate(F, T_K, EB_eV, eps_r, mode='auto')
    return C * f * T_E


# --------------------------------------------------------------------------- #
# --- Emission rate: defect → electrode (Eq. 3.27) ------------------------- #
# --------------------------------------------------------------------------- #
def emission_rate(E_D_eV, F, T_K, Ef_eV=0.0, EB_eV=1.0, eps_r=5.6):
    """
    Compute emission rate [s⁻¹] from defect into electrode.
    Uses Eq. (3.27): R_DE = C * (1 - f(Ee)) * T(Ee)
    """
    E_e_eV = EB_eV - E_D_eV
    C = prefactor_C(E_e_eV=E_e_eV, E_D_eV=E_D_eV)
    f = fermi_dirac(E_e_eV*q, Ef_eV*q, T_K)
    T_E = direct_tunneling_rate(F, T_K, EB_eV, eps_r, mode='auto')
    return C * (1 - f) * T_E
