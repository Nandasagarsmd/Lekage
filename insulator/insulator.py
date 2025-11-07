"""
insulator.py
-------------
Defines the dielectric slab geometry and defect (trap) distribution
Imports physical constants from const.py
Later will call the effective_barrier_potential() routine to compute V(x,y,z)
"""

import numpy as np
from const.constants import q, eps_opt, eps_r

class DielectricSlab:
    def __init__(self, lx_nm=10.0, ly_nm=5.0, lz_nm=5.0,
                 nx=100, ny=50, nz=50,
                 Nd=100, seed=42):
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
        self.lx = lx_nm * 1e-9
        self.ly = ly_nm * 1e-9
        self.lz = lz_nm * 1e-9
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Nd = Nd
        self.seed = seed

        # grid setup
        self.x = np.linspace(0, self.lx, nx)
        self.y = np.linspace(0, self.ly, ny)
        self.z = np.linspace(0, self.lz, nz)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

        # initialize dielectric properties
        self.eps_r = eps_r
        self.eps_opt = eps_opt

        # generate defect distribution
        self._seed_defects()

        # placeholder for potential map
        self.Veff = None

    def _seed_defects(self):
        """Randomly place Nd defects inside the slab (uniform distribution)."""
        rng = np.random.default_rng(self.seed)
        self.defect_positions = np.column_stack((
            rng.uniform(0, self.lx, self.Nd),
            rng.uniform(0, self.ly, self.Nd),
            rng.uniform(0, self.lz, self.Nd)
        ))
        # one electron per defect (occupancy = 1)
        self.defect_occupancy = np.ones(self.Nd, dtype=int)
        # charge per defect (for now −e since electrons)
        self.defect_charge = -q * self.defect_occupancy

    def compute_effective_potential(self, bias_V=1.0):
        """
        Placeholder: compute effective barrier potential inside the slab.
        Will later combine base barrier, field tilt, image term, and defect contributions.
        """
        print(f"Computing effective barrier potential for {self.Nd} defects at {bias_V:.2f} V...")
        # TODO: implement the actual potential computation
        self.Veff = np.zeros((self.nx, self.ny, self.nz))
        return self.Veff

    def summary(self):
        """Print quick summary of the system."""
        print("---- Dielectric Slab ----")
        print(f"Dimensions (nm): {self.lx*1e9:.1f} × {self.ly*1e9:.1f} × {self.lz*1e9:.1f}")
        print(f"Grid points: {self.nx} × {self.ny} × {self.nz}")
        print(f"Defects: {self.Nd}")
        print(f"Dielectric constants: eps_r={self.eps_r}, eps_opt={self.eps_opt}")
        print("-------------------------")



