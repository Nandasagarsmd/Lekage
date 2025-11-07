"""
const.py — Fundamental and material constants
Used for leakage-current & KMC simulations in high-κ dielectrics
All values in SI units unless noted otherwise.
"""

# ---------- Fundamental physical constants ----------
q = 1.602176634e-19        # [C] elementary charge
k_B = 1.380649e-23         # [J/K] Boltzmann constant
h = 6.62607015e-34         # [J·s] Planck constant
hbar = h / (2 * 3.141592653589793)  # [J·s] reduced Planck constant
eps_vaccume = 8.8541878128e-12    # [F/m] vacuum permittivity
me = 9.1093837015e-31      # [kg] electron rest mass
c = 2.99792458e8           # [m/s] speed of light
NA = 6.02214076e23         # [1/mol] Avogadro constant

# ---------- Material / model defaults (tunable) ----------
# Typical for high-κ dielectrics like HfO2 / ZrO2
eps_opt = 4.0              # optical permittivity used in image potential (ε_opt)
eps_r = 25.0               # static permittivity (ε_r)
m_eff = 0.25 * me          # [kg] effective electron mass in oxide
E_B = 2.0                  # [eV] conduction-band offset (CBO)
nu0 = 1e13                 # [1/s] attempt-to-escape frequency
gamma = 1.0e10             # [1/m] tunneling decay constant (~1/nm)
lambda_scr = 1e-9          # [m] screening length for defect potentials (~1 nm)