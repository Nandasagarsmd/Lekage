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
        plotter.add_volume(grid, scalars="V (eV)", cmap="inferno", opacity="sigmoid", shade=True)
        plotter.add_axes()
        plotter.add_text("3D Effective Potential (eV)", font_size=10)
        plotter.show_grid()
        plotter.show()
