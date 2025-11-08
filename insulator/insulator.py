"""
insulator.py
-------------
Defines the dielectric slab geometry and defect (trap) distribution.
Imports physical constants from const/constants.py.
Later will call the effective_barrier_potential() routine to compute V(x,y,z).
"""

import numpy as np
from const.constants import q, eps0, eps_opt, eps_r
import pyvista as pv
from phy.phy import elastic_hopping_rate     
import os, platform
        

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
        """Uniform random placement of defect sites inside the slab."""
        rng = np.random.default_rng(self.seed)
        self.defect_positions = np.column_stack((
            rng.uniform(0, self.lx, self.Nd),
            rng.uniform(0, self.ly, self.Nd),
            rng.uniform(0, self.lz, self.Nd)
        ))
        # occupancy & charge (1 electron per defect)
        self.defect_occupancy = np.ones(self.Nd, dtype=int)
        self.defect_charge = -q * self.defect_occupancy

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

    # --------------------- 3D volume visualization -------------------------

    def plot_potential_3d(self, downsample: int = 2):
        """
        3D interactive visualization of the effective potential using PyVista.

        Parameters
        ----------
        downsample : int
            Factor to reduce grid density for faster rendering (e.g., 2 = use every 2nd point).
        """
        import pyvista as pv
        import numpy as np
        from const.constants import q

        if self.Veff is None:
            raise ValueError("Run compute_effective_potential() first.")

        print("[INFO] Launching 3D PyVista visualization ...")

        # downsample grid for speed
        xs = self.x[::downsample] * 1e9  # nm
        ys = self.y[::downsample] * 1e9
        zs = self.z[::downsample] * 1e9
        Vred = self.Veff[::downsample, ::downsample, ::downsample] / q  # eV

        # Create structured grid
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        grid = pv.StructuredGrid(X, Y, Z)
        grid["V (eV)"] = Vred.ravel(order="F")

        # PyVista plot
        plotter = pv.Plotter()
        opacity_values = [0.0, 0.02, 0.05, 0.12, 0.22, 0.35, 0.55, 0.8]

        plotter.add_volume(
            grid,
            scalars="V (eV)",
            cmap="plasma",
            opacity=opacity_values,
            shade=True,
            scalar_bar_args={"title": "V (eV)"}
        )

        contours = grid.contour(isosurfaces=3)
        plotter.add_mesh(contours, color="black", opacity=0.12)

        plotter.set_background("white", top="lightblue")
        plotter.show_grid()
        plotter.add_axes()
        plotter.add_text("Transparent 3D Potential Landscape", font_size=10)

        plotter.show(screenshot="Veff_3D_whitebg.png")

    def animate_hopping(self, n_steps=20, save_gif=True,
                        localization_energy_eV=0.3,
                        attempt_freq=1e13,
                        color_map="coolwarm"):
        """
        Physically realistic hopping animation (elastic case).
        Uses Mott’s distance-weighted rate:
            R_ij = ν0 * exp(-2 * r_ij / r_D)
        where r_D = ħ / sqrt(2 m* E_D), imported from phy.phy.

        Parameters
        ----------
        n_steps : int
            Number of hopping events to simulate.
        save_gif : bool
            Whether to save animation as GIF.
        localization_energy_eV : float
            Defect depth (E_D) in eV, sets localization radius.
        attempt_freq : float
            Attempt frequency (ν₀) in Hz.
        color_map : str
            Colormap for 3D visualization.
        """
        # --- renderer setup ---
        if platform.system() == "Linux":
            os.environ["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "1"
            os.environ["PYVISTA_OFF_SCREEN"] = "true"
            offscreen = True
        else:
            offscreen = False

        plotter = pv.Plotter(off_screen=offscreen, window_size=[800, 600])
        plotter.set_background("white")
        xs, ys, zs = self.x * 1e9, self.y * 1e9, self.z * 1e9
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        grid = pv.StructuredGrid(X, Y, Z)
        grid["V (eV)"] = (self.Veff / q).ravel(order="F")

        # --- volume visualization setup ---
        V_data = grid["V (eV)"]
        vmin, vmax = np.percentile(V_data, [2, 98])
        grid["V (eV)"] = np.clip(V_data, vmin, vmax)
        opacity_values = [0.0, 0.0, 0.05, 0.15, 0.35, 0.55, 0.75, 0.85, 0.95, 1.0]

        plotter.add_volume(
            grid,
            scalars="V (eV)",
            cmap=color_map,
            opacity=opacity_values,
            shade=True,
            scalar_bar_args={"title": "V (eV)", "color": "black"},
        )
        plotter.add_axes()
        plotter.show_grid()

        if save_gif:
            plotter.open_gif("hopping_realistic.gif")

        rng = np.random.default_rng()

        # ensure occupancy array exists
        if not hasattr(self, "defect_occupancy"):
            self.defect_occupancy = np.ones(self.Nd, dtype=int)

        print(f"[INFO] Elastic hopping with E_D = {localization_energy_eV} eV, ν₀ = {attempt_freq:.1e} Hz")

        # --- main hopping loop ---
        for step in range(n_steps):
            occupied = np.where(self.defect_occupancy == 1)[0]
            if len(occupied) == 0:
                print("[INFO] No occupied defects left.")
                break

            # pick one occupied site
            i = rng.choice(occupied)
            r_i = self.defect_positions[i]

            # compute hopping rates from defect i to all others
            rates = np.array([
                elastic_hopping_rate(r_i, r_j,
                                    E_D=localization_energy_eV*q,
                                    nu=attempt_freq)
                for r_j in self.defect_positions
            ])
            rates[i] = 0.0  # no self-hop
            P = rates / rates.sum()

            # select target based on weighted probability
            j = rng.choice(np.arange(self.Nd), p=P)
            r_j = self.defect_positions[j]

            # perform hop (updates occupancy & potential)
            self.perform_hop(i, j)

            # update visualization field
            V_data = (self.Veff / q).ravel(order="F")
            vmin, vmax = np.percentile(V_data, [2, 98])
            grid["V (eV)"] = np.clip(V_data, vmin, vmax)

            # draw transition line and spheres
            line = pv.Line(r_i * 1e9, r_j * 1e9)
            plotter.add_mesh(line, color="orange", line_width=6, opacity=0.9, lighting=False)
            plotter.add_mesh(pv.Sphere(radius=0.2, center=r_i * 1e9), color="blue", opacity=0.9)
            plotter.add_mesh(pv.Sphere(radius=0.2, center=r_j * 1e9), color="red", opacity=0.9)

            # render and record
            plotter.render()
            if save_gif:
                plotter.write_frame()

            # reset scene
            plotter.clear_actors()
            plotter.add_volume(grid, scalars="V (eV)", cmap=color_map,
                            opacity=opacity_values, shade=True)
            plotter.add_axes()
            plotter.show_grid()

            print(f"[DEBUG] Hop {i}->{j} | r_ij={np.linalg.norm(r_i-r_j)*1e9:.2f} nm | Rate={rates[j]:.2e} s⁻¹")

        if save_gif:
            plotter.close()
            print("[INFO] Saved 'hopping_realistic.gif'")
        else:
            plotter.show()


    def animate_hopping2(self, n_steps=20, save_gif=True):
        """
        Visually exaggerated hopping animation (debug mode).
        Shows large local potential changes and highlights hopping sites.
        """
        import pyvista as pv
        import numpy as np
        import os, platform
        from const.constants import q

        # --- detect OS and configure renderer ---
        if platform.system() == "Linux":
            os.environ["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "1"
            os.environ["PYVISTA_OFF_SCREEN"] = "true"
            offscreen = True
        else:
            offscreen = False  # Windows/macOS use native OpenGL

        print("[INFO] Launching exaggerated hopping animation...")

        # --- setup grid ---
        xs, ys, zs = self.x * 1e9, self.y * 1e9, self.z * 1e9
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        grid = pv.StructuredGrid(X, Y, Z)
        grid["V (eV)"] = (self.Veff / q).ravel(order="F")

        # --- plot setup ---
        plotter = pv.Plotter(off_screen=offscreen, window_size=[800, 600])
        plotter.add_volume(grid, scalars="V (eV)", cmap="plasma",
                        opacity="sigmoid", shade=True)
        plotter.set_background("white")
        plotter.add_axes()
        plotter.show_grid()
        plotter.add_text("Elastic Hopping (exaggerated)", font_size=12)

        if save_gif:
            plotter.open_gif("hopping_debug.gif")

        # --- animation loop ---
        for step in range(n_steps):
            # pick occupied defect i and empty defect j
            occupied = np.where(self.defect_occupancy == 1)[0]
            empty = np.where(self.defect_occupancy == 0)[0]
            if len(occupied) == 0 or len(empty) == 0:
                print("[INFO] No available hops left.")
                break
            i = np.random.choice(occupied)
            j = np.random.choice(empty)

            # perform hop and recalc potential
            self.perform_hop(i, j)

            # update visualization
            grid["V (eV)"] = (self.Veff / q).ravel(order="F")

            # mark hopping sites
            sphere_i = pv.Sphere(radius=0.2, center=self.defect_positions[i] * 1e9)
            sphere_j = pv.Sphere(radius=0.2, center=self.defect_positions[j] * 1e9)
            plotter.add_mesh(sphere_i, color="red", opacity=0.8)
            plotter.add_mesh(sphere_j, color="blue", opacity=0.8)
            plotter.add_text(f"Step {step+1}/{n_steps}: {i}→{j}", font_size=10)

            if save_gif:
                plotter.write_frame()
            else:
                plotter.render()

            plotter.clear_actors()
            plotter.add_volume(grid, scalars="V (eV)", cmap="plasma",
                            opacity="sigmoid", shade=True)
            plotter.add_axes()
            plotter.show_grid()


    # --------------------- elastic defect to defect -------------------------

    def hopping_rate(self, i, j, E_D=0.3*q):
        r_i = self.defect_positions[i]
        r_j = self.defect_positions[j]
        return elastic_defect_to_defect.elastic_hopping_rate(r_i, r_j, E_D=E_D)
    
    def update_potential_due_to_hop(self, i, j):
        """
        Recalculate electrostatic potential when an electron hops i -> j.
        Removes Coulomb potential of defect i and adds potential of defect j.
        """
        import numpy as np
        from const.constants import q, eps0

        # charge magnitude
        e = q
        r_i = self.defect_positions[i]
        r_j = self.defect_positions[j]

        # 3D coordinate grids (m)
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

        # distances
        r_i_grid = np.sqrt((X - r_i[0])**2 + (Y - r_i[1])**2 + (Z - r_i[2])**2)
        r_j_grid = np.sqrt((X - r_j[0])**2 + (Y - r_j[1])**2 + (Z - r_j[2])**2)

        # avoid divide by zero
        r_i_grid[r_i_grid < 1e-12] = 1e-12
        r_j_grid[r_j_grid < 1e-12] = 1e-12

        # change in potential due to hop
        deltaV = (e / (4 * np.pi * eps0 * self.eps_r)) * (1 / r_j_grid - 1 / r_i_grid)

        # update the potential grid
        self.Veff += deltaV

    def perform_hop(self, i, j):
        """
        Move one electron from defect i to j.
        Update occupancies and potential.
        """
        if self.defect_occupancy[i] == 0:
            print(f"[WARN] Defect {i} empty — skipping hop.")
            return

        # update occupancies
        self.defect_occupancy[i] = 0
        self.defect_occupancy[j] = 1

        # update potential accordingly
        self.update_potential_due_to_hop(i, j)

    
