"""
main.py
--------
Entry point for leakage simulation.
Creates a dielectric slab, computes potential, visualizes, and animates hopping.
"""

from insulator.insulator import DielectricSlab


def main():
    print("\n=== Leakage Simulation Setup ===")

    # --- fixed simulation parameters ---
    lx = 2.0       # slab thickness in nm
    ly = 5.0       # slab width in nm
    lz = 5.0       # slab height in nm
    Nd = 100       # number of defects
    bias_V = 6.0  # applied voltage across slab

    # --- initialize the dielectric slab ---
    slab = DielectricSlab(lx_nm=lx, ly_nm=ly, lz_nm=lz, Nd=Nd)
    slab.summary()

    # --- compute and visualize static potential ---
    slab = DielectricSlab()
    slab.compute_effective_potential(bias_V=bias_V)
    slab.animate_hopping_with_potential(n_steps=40,localization_energy_eV=0.05)



if __name__ == "__main__":
    main()
