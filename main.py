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
    Nd = 30       # number of defects
    bias_V = 600.0  # applied voltage across slab

    # --- initialize the dielectric slab ---
    slab = DielectricSlab(lx_nm=lx, ly_nm=ly, lz_nm=lz, Nd=Nd)
    slab.summary()

    # --- compute and visualize static potential ---
    slab.compute_effective_potential(bias_V=bias_V)
    slab.plot_defects()
    slab.plot_potential_3d(downsample=1)

    print(f"[INFO] Potential array shape: {slab.Veff.shape}")

    # --- simulate elastic hopping and visualize evolution ---
    print("\n[INFO] Starting elastic hopping animation...")
    slab.animate_hopping(n_steps=40, save_gif=True)
    print("[INFO] Animation complete. File saved as 'hopping_animation.gif'.")


if __name__ == "__main__":
    main()
