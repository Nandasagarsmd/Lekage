"""
phy.py – central rate manager for all physical transitions
"""

import numpy as np
from .hopping import hopping_rate          # Mott / MA
from .injection import injection_rate, emission_rate
from .poole_frenkel import poole_frenkel_rate
from .direct_tunneling import direct_tunneling_rate
from const.constants import q
from logger.log import Logger


class RateManager:
    """
    Aggregates all possible transitions and returns a flat list of candidate events.
    """

    def __init__(self, field_Vpm, temperature_K=300.0,
                 EB_eV=1.0, eps_r=5.6, area_per_site_m2=1e-18,
                 hopping_model="MA", nu0=1e13,
                 enable_pf=True, enable_direct=False,
                 logger=None, max_log_events=5, log_rates=True):
        """
        Parameters
        ----------
        field_Vpm : float
            Approx. uniform electric field [V/m].
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
            Attempt frequency [Hz].
        enable_pf : bool
            Include Poole–Frenkel emission events.
        enable_direct : bool
            Include direct tunneling events.
        logger : Logger or None
            Optional logger object shared with simulation.
        max_log_events : int
            Limit of events to print when logging.
        log_rates : bool
            Toggle to enable/disable low-level rate logging.
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
        self.logger = logger or Logger()
        self.max_log_events = max_log_events
        self.log_rates = log_rates  

    def compute_events(self, defect_positions, defect_energies, defect_occ, Ef_eV=0.0):
        """
        Build all possible charge transport transitions.
        """
        N = len(defect_positions)
        events = []

        # -------- defect → defect hopping --------
        occ_idx = np.where(defect_occ == 1)[0]
        emp_idx = np.where(defect_occ == 0)[0]

        for i in occ_idx:
            ri = defect_positions[i]
            Ei = defect_energies[i]
            for j in emp_idx:
                if j == i:
                    continue
                rj = defect_positions[j]
                Ej = defect_energies[j]
                R = hopping_rate(ri, rj, Ei, Ej, T_K=self.T,
                                model=self.hopping_model, nu0=self.nu0)
                if np.isfinite(R) and R > 0.0:
                    events.append({'i': i, 'j': j, 'type': 'hop', 'model': 'hopping', 'rate': R})

        # -------- electrode → defect (injection) --------
        for j in emp_idx:
            Ej = defect_energies[j]
            Rinj = injection_rate(E_D_eV=Ej, F=self.F, T_K=self.T,
                                Ef_eV=Ef_eV, EB_eV=self.EB_eV, eps_r=self.eps_r)
            if np.isfinite(Rinj) and Rinj > 0.0:
                events.append({'i': 'electrode', 'j': j, 'type': 'inject', 'model': 'injection', 'rate': Rinj})

        # -------- defect → electrode (emission) --------
        for i in occ_idx:
            Ei = defect_energies[i]
            Remit = emission_rate(E_D_eV=Ei, F=self.F, T_K=self.T,
                                Ef_eV=Ef_eV, EB_eV=self.EB_eV, eps_r=self.eps_r)
            if np.isfinite(Remit) and Remit > 0.0:
                events.append({'i': i, 'j': 'electrode', 'type': 'emit', 'model': 'emission', 'rate': Remit})

        # -------- Poole–Frenkel emission --------
        if self.enable_pf:
            for i in occ_idx:
                Ei = defect_energies[i]
                Rpf = poole_frenkel_rate(E_T_eV=Ei, F_Vpm=self.F,
                                        T_K=self.T, eps_r=self.eps_r, nu0=self.nu0)
                if np.isfinite(Rpf) and Rpf > 0.0:
                    events.append({'i': i, 'j': 'CB', 'type': 'pf', 'model': 'poole_frenkel', 'rate': Rpf})

        # -------- direct tunneling --------
        if self.enable_direct:
            Rdir = direct_tunneling_rate(self.F, self.T, self.EB_eV,
                                        eps_r=self.eps_r, A_cell=self.A_cell, mode='auto')
            if np.isfinite(Rdir) and Rdir > 0.0:
                events.append({'i': 'electrode', 'j': 'electrode*',
                            'type': 'direct', 'model': 'direct_tunneling', 'rate': Rdir})

        # --- Structured logging ---
        if self.log_rates and self.logger is not None:
            self.logger.section("RateManager Transition Summary")
            self.logger.write(Source="RateManager", Info=f"Computed {len(events)} transitions")
            limit = None if self.max_log_events is None else self.max_log_events
            for e in (events if limit is None else events[:limit]):
                self.logger.write(Source="RateManager",
                                Model=e['model'],
                                Type=e['type'],
                                i=e['i'],
                                j=e['j'],
                                Rate=e['rate'])

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
