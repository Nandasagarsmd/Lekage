"""
insulator.py
-------------
Defines the dielectric slab geometry and defect (trap) distribution.
Imports physical constants from const/constants.py.
Later will call the effective_barrier_potential() routine to compute V(x,y,z).
## Logging only 5 events for now
"""

import numpy as np
from const.constants import q, eps0, eps_opt, eps_r
#from phy.phy import inelastic_hopping_rate     
import os, platform
from logger.log import Logger
from phy.phy import RateManager
        

class DielectricSlab:
    """
    Represents a 3D dielectric slab used for leakage and trap simulations.
    """

    def __init__(self,
                 lx_nm: float = 10.0,
                 ly_nm: float = 5.0,
                 lz_nm: float = 5.0,
                 nx: int = 100,
                 ny: int = 50,
                 nz: int = 50,
                 Nd: int = 100,
                 seed: int = 42):
        """
        Initialize a 3D dielectric region.

        Parameters
        ----------
        lx_nm, ly_nm, lz_nm : float
            Dimensions of the dielectric box in nm.
        nx, ny, nz : int
            Number of grid points in each direction.
        Nd : int
            Number of defect sites to place in the box.
        seed : int
            Random seed for reproducibility.
        """
        # --- geometry ---
        self.lx = lx_nm * 1e-9
        self.ly = ly_nm * 1e-9
        self.lz = lz_nm * 1e-9
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Nd = Nd
        self.seed = seed

        # --- dielectric parameters ---
        self.eps_r = eps_r
        self.eps_opt = eps_opt

        # --- generate grids ---
        self._generate_grid()

        # --- seed defects ---
        self._seed_defects()

        # --- placeholder for potential ---
        self.Veff = None

    # ------------------------- internal methods -----------------------------

    def _generate_grid(self):
        """Create 3D coordinate grid arrays."""
        self.x = np.linspace(0, self.lx, self.nx)
        self.y = np.linspace(0, self.ly, self.ny)
        self.z = np.linspace(0, self.lz, self.nz)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

    def _seed_defects(self):
        """Randomly place Nd defects inside the slab (uniform distribution)."""
        rng = np.random.default_rng(self.seed)
        self.defect_positions = np.column_stack((
            rng.uniform(0, self.lx, self.Nd),
            rng.uniform(0, self.ly, self.Nd),
            rng.uniform(0, self.lz, self.Nd)
        ))
        # one electron per defect
        self.defect_occupancy = np.ones(self.Nd, dtype=int)
        self.defect_charge = -q * self.defect_occupancy

        # defect energy levels (around mean depth E_D ± Δ)
        self.E_D_mean_eV = 0.8   # average defect depth in eV
        self.E_D_spread_eV = 0.2 # variation range
        self.defect_energies = rng.uniform(
            self.E_D_mean_eV - self.E_D_spread_eV / 2,
            self.E_D_mean_eV + self.E_D_spread_eV / 2,
            self.Nd
        )


    # ------------------------- external / public methods --------------------

    def compute_effective_potential(self, bias_V: float = 1.0, EB_eV: float = 2.0):
        """
        Compute the effective barrier potential inside the dielectric slab.
        Combines: base barrier, field tilt, image potential, and defect contributions.

        Parameters
        ----------
        bias_V : float
            Applied bias across the oxide [V].
        EB_eV : float
            Conduction band offset (barrier height) [eV].

        Returns
        -------
        Veff : ndarray
            3D array [J] of total potential energy landscape.
        """
        import numpy as np
        from const.constants import q, eps0

        print(f"[INFO] Computing effective barrier potential ...")

        # --- constants ---
        EB = EB_eV * q        # convert to joules
        F = bias_V / self.lx   # uniform field [V/m]
        xgrid = self.x.copy()

        # --- base + field + image term (1D along x) ---
        x_safe = np.clip(xgrid, 1e-10, None)  # avoid x=0 singularity
        V0 = EB - q * F * x_safe - (q**2) / (16 * np.pi * eps0 * self.eps_opt * x_safe)

        # broadcast to 3D grid
        V_base = np.repeat(V0[:, None, None], self.ny, axis=1)
        V_base = np.repeat(V_base, self.nz, axis=2)

        # --- defect potential (screened Coulomb) ---
        V_def = np.zeros_like(V_base)
        kappa = 1 / (1e-9)  # screening length = 1 nm
        soft_a = 0.1e-9     # soft core radius [m]

        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        for (xd, yd, zd, qd) in zip(
            self.defect_positions[:, 0],
            self.defect_positions[:, 1],
            self.defect_positions[:, 2],
            self.defect_charge
        ):
            r = np.sqrt((X - xd)**2 + (Y - yd)**2 + (Z - zd)**2 + soft_a**2)
            V_def += (qd / (4 * np.pi * eps0 * self.eps_r)) * np.exp(-kappa * r) / r

        # --- total potential ---
        self.Veff = V_base + V_def
        print("[INFO] Effective potential computed.")

        return self.Veff


    def summary(self):
        """Print a summary of the system."""
        print("---- Dielectric Slab ----")
        print(f"Dimensions (nm): {self.lx*1e9:.1f} × {self.ly*1e9:.1f} × {self.lz*1e9:.1f}")
        print(f"Grid points: {self.nx} × {self.ny} × {self.nz}")
        print(f"Defects: {self.Nd}")
        print(f"Dielectric constants: eps_r={self.eps_r}, eps_opt={self.eps_opt}")
        print("-------------------------")

    # ------------------------- visualization ------------------------------

    def plot_defects(self, show_axes=True):
        """
        3D scatter of defect positions inside the dielectric box.

        Uses matplotlib for portability.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = self.defect_positions.T * 1e9  # convert to nm
        ax.scatter(x, y, z, c='orange', s=25, alpha=0.8, edgecolors='k')

        # draw box edges
        ax.set_xlim(0, self.lx * 1e9)
        ax.set_ylim(0, self.ly * 1e9)
        ax.set_zlim(0, self.lz * 1e9)

        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel('z (nm)')
        ax.set_title(f'Defect distribution (Nd = {self.Nd})')

        if not show_axes:
            ax.set_axis_off()

        plt.tight_layout()
        plt.show()


    def plot_potential_slice(self, axis: str = "z", index: int = None):
        """
        Plot a 2D slice of the effective potential [eV] using matplotlib.
        """
        import matplotlib.pyplot as plt

        if self.Veff is None:
            raise ValueError("Run compute_effective_potential() first.")

        if axis == "z":
            if index is None:
                index = self.nz // 2
            data = self.Veff[:, :, index] / q
            xlabel, ylabel = "x (nm)", "y (nm)"
            extent = [0, self.ly * 1e9, 0, self.lx * 1e9]
        elif axis == "y":
            if index is None:
                index = self.ny // 2
            data = self.Veff[:, index, :] / q
            xlabel, ylabel = "x (nm)", "z (nm)"
            extent = [0, self.lz * 1e9, 0, self.lx * 1e9]
        else:
            raise ValueError("axis must be 'y' or 'z'")

        plt.figure(figsize=(6, 5))
        plt.imshow(data, origin="lower", cmap="inferno", aspect="auto", extent=extent)
        plt.colorbar(label="Potential energy (eV)")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"Effective potential slice ({axis}={index})")
        plt.tight_layout()
        plt.show()

    def animate_hopping_with_potential(self, n_steps=20, localization_energy_eV=0.3, T=300):
        """
        Physically realistic 3D Plotly animation:
        - Transparent 3D potential landscape (dynamically updated)
        - Orange defects (neutral)
        - Blue = electron leaves, Red = arrives
        - Red line = hop path
        - Local potential field updates after each hop
        """

        import numpy as np
        import plotly.graph_objects as go
        from plotly.io import renderers
        from const.constants import q
        from phy.phy import inelastic_hopping_rate

        renderers.default = "browser"  # open in browser

        if self.Veff is None:
            raise ValueError("Run compute_effective_potential() first.")
        if not hasattr(self, "defect_energies"):
            raise ValueError("Defect energies not initialized — call _seed_defects() first.")

        rng = np.random.default_rng(self.seed)
        positions_nm = self.defect_positions * 1e9

        # --- Setup 3D mesh for the volume field ---
        downsample = 3
        xs = self.x[::downsample] * 1e9
        ys = self.y[::downsample] * 1e9
        zs = self.z[::downsample] * 1e9
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        # --- Initial potential volume ---
        V = self.Veff[::downsample, ::downsample, ::downsample] / q  # [eV]
        volume_trace = go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=V.flatten(),
            isomin=np.percentile(V, 5),
            isomax=np.percentile(V, 95),
            opacity=0.03,
            surface_count=8,
            colorscale="Viridis",
            showscale=False,
            name="Potential"
        )

        # --- Initial defect state ---
        colors = np.full(self.Nd, "orange", dtype=object)
        defects_trace = go.Scatter3d(
            x=positions_nm[:, 0], y=positions_nm[:, 1], z=positions_nm[:, 2],
            mode="markers",
            marker=dict(size=5, color=colors, opacity=0.9, line=dict(width=0.5, color="darkorange")),
            name="Defects"
        )

        # --- Occupancy array ---
        if not hasattr(self, "defect_occupancy"):
            self.defect_occupancy = np.ones(self.Nd, dtype=int)

        frames = []

        eps = self.eps_r * 8.854e-12  # dielectric permittivity

        # --- Main hopping loop ---
        for step in range(n_steps):
            occupied = np.where(self.defect_occupancy == 1)[0]
            if len(occupied) == 0:
                print("[INFO] All defects empty, stopping animation.")
                break

            # Select one occupied defect as source
            i = rng.choice(occupied)
            r_i = self.defect_positions[i]
            E_i = self.defect_energies[i]

            # Compute hopping probabilities
            rates = np.zeros(self.Nd)
            for j in range(self.Nd):
                if j == i:
                    continue
                r_j = self.defect_positions[j]
                E_j = self.defect_energies[j]
                rates[j] = inelastic_hopping_rate(r_i, r_j, E_i, E_j,
                                                E_D=self.E_D_mean_eV*q, nu=1e13, T=T)

            if np.all(rates == 0):
                continue

            # Pick destination j probabilistically
            P = rates / rates.sum()
            j = rng.choice(np.arange(self.Nd), p=P)
            r_j = self.defect_positions[j]

            # Update occupancies
            self.defect_occupancy[i] = 0
            self.defect_occupancy[j] = 1

            # --- Update potential field locally (Coulombic response) ---
            charge = -q
            for pos, sign in [(r_i, +1), (r_j, -1)]:
                dx = self.x[:, None, None] - pos[0]
                dy = self.y[None, :, None] - pos[1]
                dz = self.z[None, None, :] - pos[2]
                r = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-12
                deltaV = sign * charge / (4 * np.pi * eps * r)
                self.Veff += 0.05 * deltaV  # small perturbation factor

            # --- Update color states ---
            colors[:] = "orange"
            colors[i] = "blue"
            colors[j] = "red"

            defects_trace = go.Scatter3d(
                x=positions_nm[:, 0], y=positions_nm[:, 1], z=positions_nm[:, 2],
                mode="markers",
                marker=dict(size=5, color=colors, opacity=0.95, line=dict(width=0.5, color="darkorange")),
                name="Defects"
            )

            # --- Recalculate the potential volume for this frame ---
            V_new = self.Veff[::downsample, ::downsample, ::downsample] / q
            volume_trace = go.Volume(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=V_new.flatten(),
                isomin=np.percentile(V_new, 5),
                isomax=np.percentile(V_new, 95),
                opacity=0.04,
                surface_count=8,
                colorscale="Viridis",
                showscale=False,
                name="Potential"
            )

            # --- Add red line for the hop path ---
            hop_line = go.Scatter3d(
                x=[r_i[0]*1e9, r_j[0]*1e9],
                y=[r_i[1]*1e9, r_j[1]*1e9],
                z=[r_i[2]*1e9, r_j[2]*1e9],
                mode="lines",
                line=dict(color="rgba(255,0,0,0.9)", width=10),
                name=f"Hop {step+1}"
            )

            # --- Assemble this frame ---
            frames.append(go.Frame(data=[volume_trace, defects_trace, hop_line],
                                name=f"Hop{step+1}"))

            print(f"[DEBUG] Hop {i}->{j} | ΔE={(E_j - E_i):.3f} eV | "
                f"r_ij={np.linalg.norm(r_i - r_j)*1e9:.2f} nm")

        # --- Build figure and layout ---
        fig = go.Figure(data=[volume_trace, defects_trace], frames=frames)
        fig.update_layout(
            title=f"Inelastic Hopping with Dynamic Field (Eₗ={localization_energy_eV:.2f} eV, T={T} K)",
            scene=dict(
                xaxis_title='x (nm)',
                yaxis_title='y (nm)',
                zaxis_title='z (nm)',
                bgcolor='rgb(245,245,245)'
            ),
            width=950,
            height=800,
            margin=dict(l=0, r=0, t=60, b=0),
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "x": 0.1, "y": 0,
                "buttons": [
                    {"label": "▶ Play", "method": "animate",
                    "args": [None, {"frame": {"duration": 700, "redraw": True},
                                    "transition": {"duration": 150},
                                    "fromcurrent": True, "mode": "immediate"}]},
                    {"label": "⏸ Pause", "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]}
                ]
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top", "xanchor": "left",
                "currentvalue": {"font": {"size": 16}, "prefix": "Hop: ", "visible": True, "xanchor": "right"},
                "transition": {"duration": 100, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 60},
                "len": 0.9, "x": 0.05, "y": -0.05,
                "steps": [
                    {"args": [[f"Hop{k+1}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": str(k+1), "method": "animate"} for k in range(len(frames))
                ]
            }]
        )

        fig.show()

    def run_kmc(self, t_stop=1e-6, sample_interval=1e-9,
                temperature_K=300, bias_V=1.0):
        """
        Run kinetic Monte Carlo (kMC) simulation of leakage transport.

        Parameters
        ----------
        t_stop : float
            Simulation end time [s].
        sample_interval : float
            Sampling interval for current logging [s].
        temperature_K : float
            Temperature [K].
        bias_V : float
            Applied bias voltage [V].
        """
        print("[INFO] Starting kMC simulation...")

        # --- setup logger ---
        logger = Logger(filepath="output/log.txt")
        logger.section("Kinetic Monte Carlo Simulation Started")
        logger.write(f"[Setup] bias={bias_V:.2f} V, temperature={temperature_K} K, t_stop={t_stop:.2e}s")

        # --- electric field and rate manager ---
        F = bias_V / (self.lx if hasattr(self, "lx") else 2e-9)  # field [V/m]
        rm = RateManager(field_Vpm=F, temperature_K=temperature_K, EB_eV=2.0, logger=logger, max_log_events=5 )
        logger.write(f"[Setup] Electric field = {F:.3e} V/m")

        # --- initial defect occupancy (random 0/1) ---
        defect_occ = np.random.choice([0, 1], size=self.Nd, p=[0.5, 0.5])
        logger.write(f"[Setup] Initialized {defect_occ.sum()} / {self.Nd} occupied defects")

        # --- simulation state variables ---
        t = 0.0
        total_transfers = 0
        current_log = []  # to store time-current pairs

        logger.section("Simulation Loop")

        # --- main kinetic Monte Carlo loop ---
        while t < t_stop:
            events = rm.compute_events(
                defect_positions=self.defect_positions,
                defect_energies=self.defect_energies,
                defect_occ=defect_occ
            )

            rates = np.array([ev["rate"] for ev in events])
            total_rate = np.sum(rates)

            if total_rate <= 1e-50 or np.isnan(total_rate):
                logger.write("[WARN] No available transitions. Ending simulation.")
                break

            # generate random numbers for τ and event choice
            r1, r2 = np.random.random(2)
            tau = -np.log(r1) / total_rate
            t += tau

            # choose event by cumulative probability
            chosen_idx = np.searchsorted(np.cumsum(rates / total_rate), r2)
            chosen_event = events[chosen_idx]

            # --- process chosen event ---
            model = chosen_event["model"]
            rate_val = chosen_event["rate"]
            i = chosen_event.get("i")
            j = chosen_event.get("j")

            # Handle according to event type safely
            if isinstance(i, int) and isinstance(j, int):
                # defect → defect hop
                defect_occ[i], defect_occ[j] = 0, 1
                logger.write(f"[Event] {model:>16s}: hop {i}->{j}, rate={rate_val:.3e}")

            elif i == "electrode" and isinstance(j, int):
                # injection into defect
                defect_occ[j] = 1
                logger.write(f"[Event] {model:>16s}: inject electrode→{j}, rate={rate_val:.3e}")

            elif isinstance(i, int) and j == "electrode":
                # emission to electrode
                defect_occ[i] = 0
                logger.write(f"[Event] {model:>16s}: emit {i}→electrode, rate={rate_val:.3e}")

            elif isinstance(i, int) and j == "CB":
                # Poole–Frenkel emission
                defect_occ[i] = 0
                logger.write(f"[Event] {model:>16s}: PF {i}→CB, rate={rate_val:.3e}")

            elif i == "electrode" and j == "electrode*":
                # direct tunneling channel
                logger.write(f"[Event] {model:>16s}: direct tunneling event, rate={rate_val:.3e}")

            else:
                # Unknown / fallback
                logger.write(f"[WARN] Unhandled event type: {chosen_event}")


        # --- finalize ---
        logger.section("Simulation Ended")
        logger.write(f"Total simulated time: {t:.3e} s")
        logger.write(f"Total events executed: {total_transfers}")
        logger.write(f"Average current: {np.mean([c for _, c in current_log]):.3e} A")

        logger.close()
        print("[INFO] Simulation completed. Logs saved to output/log.txt.")