"""
main.py
--------
Entry point for leakage simulation.
Creates a dielectric slab, computes potential, and plots slices.
"""

from insulator.insulator import DielectricSlab


def main():
    print("\n=== Leakage Simulation Setup ===")

    # --- fixed simulation parameters ---
    lx = 10.0      # slab thickness in nm
    ly = 5.0       # slab width in nm
    lz = 5.0       # slab height in nm
    Nd = 100       # number of defects
    bias_V = 1.0   # applied voltage across slab

    # --- initialize the dielectric slab ---
    slab = DielectricSlab(lx_nm=lx, ly_nm=ly, lz_nm=lz, Nd=Nd)
    slab.summary()

    # --- compute and visualize ---
    slab.compute_effective_potential(bias_V=bias_V)
    slab.plot_defects()
    #slab.plot_potential_slice(axis="z")  # mid-plane slice (z)
    slab.plot_potential_3d(downsample=2)

    print(f"[INFO] Potential array shape: {slab.Veff.shape}")


if __name__ == "__main__":
    main()
