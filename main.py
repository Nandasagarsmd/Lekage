"""
main.py
--------
Entry point for leakage simulation.
Creates a dielectric slab, computes potential, visualizes, and animates hopping.
"""

from insulator.insulator import DielectricSlab


def main():

    lx_nm = 1.0          # slab thickness (transport direction) in nm
    ly_nm = 5.0          # width in nm
    lz_nm = 5.0          # height in nm

    nx, ny, nz = 20, 20, 20  
    Nd = 15                

    bias_V = 1.0          # applied bias [V]
    temperature_K = 300   # lattice temperature [K]
    t_stop = 1e-6         # total simulated time [s]
    sample_interval = 1e-9  # current sampling interval [s]
    EB_eV = 2.0           # barrier height [eV]
    # ----------------------------------------------------------------------

    # --- Initialize the dielectric slab ---
    slab = DielectricSlab(
        lx_nm=lx_nm, ly_nm=ly_nm, lz_nm=lz_nm,
        nx=nx, ny=ny, nz=nz, Nd=Nd
    )

    # --- Print simulation summary ---
    slab.summary()

    # --- Compute static potential landscape ---
    slab.compute_effective_potential(bias_V=bias_V, EB_eV=EB_eV)

    # --- kinetic Monte Carlo transport simulation ---
    slab.run_kmc(
        t_stop=t_stop,
        sample_interval=sample_interval,
        temperature_K=temperature_K,
        bias_V=bias_V
    )


if __name__ == "__main__":
    main()
