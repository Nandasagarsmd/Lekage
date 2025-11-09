"""
phy.py – central rate manager for all physical transitions
"""

import numpy as np
from .hopping import hopping_rate          # Mott / MA
from .injection import injection_rate, emission_rate
from .poole_frenkel import poole_frenkel_rate
from .direct_tunneling import direct_tunneling_rate  # optional direct cross-barrier
from const.constants import q
from logger.log import Logger

class RateManager:
    """
    Aggregates all possible transitions and returns a flat list of candidate events.
    """

    def __init__(self, field_Vpm, temperature_K=300.0,
                 EB_eV=1.0, eps_r=5.6, area_per_site_m2=1e-18,
                 hopping_model="MA", nu0=1e13,
                 enable_pf=True, enable_direct=False):
        """
        Parameters
        ----------
        field_Vpm : float
            Approx. uniform field for now [V/m]. (Replace with local field later)
        temperature_K : float
            Temperature [K].
        EB_eV : float
            Conduction band offset (barrier height) [eV].
        eps_r : float
            Relative permittivity of dielectric.
        area_per_site_m2 : float
            Area used to map current density to rate for direct tunneling.
        hopping_model : str
            "Mott" or "MA" for defect–defect hopping.
        nu0 : float
            Attempt frequency for hopping/PF [Hz].
        enable_pf : bool
            Include PF emission events (trap → CB).
        enable_direct : bool
            Include global direct-tunneling (SE/FN) events.
        """
        self.F = field_Vpm
        self.T = temperature_K
        self.EB_eV = EB_eV
        self.eps_r = eps_r
        self.A_cell = area_per_site_m2
        self.hopping_model = hopping_model
        self.nu0 = nu0
        self.enable_pf = enable_pf
        self.enable_direct = enable_direct
        self.logger = Logger()

    def compute_events(self, defect_pos_m, defect_E_eV, occupancy, Ef_eV=0.0):
        """
        Build the list of all candidate transitions with rates.

        Parameters
        ----------
        defect_pos_m : (N,3) array
        defect_E_eV  : (N,) array
        occupancy    : (N,) array of {0,1}
        Ef_eV        : electrode Fermi level [eV] (reference 0 by default)

        Returns
        -------
        events : list of dict
            Each dict: { 'i': idx or 'electrode',
                         'j': idx or 'electrode' or 'CB',
                         'type': str,
                         'rate': float }
        """
        N = len(defect_pos_m)
        events = []

        # -------- defect -> defect hopping --------
        occ_idx = np.where(occupancy == 1)[0]
        emp_idx = np.where(occupancy == 0)[0]
        for i in occ_idx:
            ri = defect_pos_m[i]
            Ei = defect_E_eV[i]
            # hops only to empty traps
            for j in emp_idx:
                if j == i:
                    continue
                rj = defect_pos_m[j]
                Ej = defect_E_eV[j]
                R = hopping_rate(ri, rj, Ei, Ej, T_K=self.T,
                                 model=self.hopping_model, nu0=self.nu0)
                if R > 0.0:
                    events.append({'i': i, 'j': j, 'type': 'hop', 'rate': R})

        # -------- electrode -> defect (injection) --------
        for j in emp_idx:
            Ej = defect_E_eV[j]
            Rinj = injection_rate(E_D_eV=Ej, F=self.F, T_K=self.T,
                                  Ef_eV=Ef_eV, EB_eV=self.EB_eV, eps_r=self.eps_r)
            if Rinj > 0.0:
                events.append({'i': 'electrode', 'j': j, 'type': 'inject', 'rate': Rinj})

        # -------- defect -> electrode (emission) --------
        for i in occ_idx:
            Ei = defect_E_eV[i]
            Remit = emission_rate(E_D_eV=Ei, F=self.F, T_K=self.T,
                                  Ef_eV=Ef_eV, EB_eV=self.EB_eV, eps_r=self.eps_r)
            if Remit > 0.0:
                events.append({'i': i, 'j': 'electrode', 'type': 'emit', 'rate': Remit})

        # -------- PF emission (trap -> CB) --------
        if self.enable_pf:
            for i in occ_idx:
                Ei = defect_E_eV[i]  # treat as trap depth to CB
                Rpf = poole_frenkel_rate(E_T_eV=Ei, F_Vpm=self.F,
                                         T_K=self.T, eps_r=self.eps_r, nu0=self.nu0)
                if Rpf > 0.0:
                    events.append({'i': i, 'j': 'CB', 'type': 'pf', 'rate': Rpf})

        # -------- optional: direct tunneling channel (global) --------
        if self.enable_direct:
            Rdir = direct_tunneling_rate(self.F, self.T, self.EB_eV,
                                         eps_r=self.eps_r, A_cell=self.A_cell, mode='auto')
            if Rdir > 0.0:
                # Represent as a global event (not tied to a specific defect)
                events.append({'i': 'electrode', 'j': 'electrode*', 'type': 'direct', 'rate': Rdir})

        return events

    @staticmethod
    def draw_event(events, rng):
        """
        Gillespie selection: pick one event μ by cumulative rates.
        Returns (event, tau, Rtot).
        """
        if not events:
            return None, np.inf, 0.0

        rates = np.array([e['rate'] for e in events], dtype=float)
        Rtot = rates.sum()
        if Rtot <= 0.0:
            return None, np.inf, 0.0

        r1, r2 = rng.random(), rng.random()
        tau = -np.log(r1) / Rtot
        cum = np.cumsum(rates) / Rtot
        idx = int(np.searchsorted(cum, r2))
        return events[idx], tau, Rtot

    def compute_events(self, defect_positions, defect_energies, defect_occ):
        """
        Accumulate all possible transitions between defects
        and between defects ↔ electrodes, log neatly.
        """
        events = []
        Nd = len(defect_positions)
        self.logger.section("Rate computation cycle")
        self.logger.push()

        for i in range(Nd):
            if defect_occ[i] == 1:  # occupied → can hop to others
                for j in range(Nd):
                    if i != j and defect_occ[j] == 0:
                        hop_data = hopping.elastic_hopping_rate(
                            defect_positions[i],
                            defect_positions[j],
                            E_D=defect_energies[i] * q
                        )
                        events.append({"i": i, "j": j, **hop_data})

            # Poole–Frenkel emission from defect i
            pf_data = pf.poole_frenkel_rate(
                defect_energies[i] * q,
                self.field_Vpm,
                self.temperature_K
            )
            events.append({"i": i, **pf_data})

            # Injection into defect i (if unoccupied)
            if defect_occ[i] == 0:
                inj_data = injection.elastic_injection_rate(
                    E_D=defect_energies[i] * q,
                    EB=self.EB_eV * q,
                    T=self.temperature_K,
                    F=self.field_Vpm
                )
                events.append({"j": i, **inj_data})

        # summary log
        self.logger.write(f"Generated {len(events)} transitions.")
        self.logger.pop()

        # detailed logs for first few events
        for ev in events[:5]:
            self.logger.write(
                f"[{ev['model']}] rate={ev['rate']:.3e} s⁻¹ | "
                f"E_D={ev['E_D_eV']:.3f} eV | "
                f"r={ev.get('r', 0)*1e9:.2f} nm | type={ev['model']}"
            )

        return events