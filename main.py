"""
main.py
--------
Entry point for leakage simulation.
Creates a dielectric slab, computes potential, visualizes, and animates hopping.
"""

from insulator.insulator import DielectricSlab


def main():
    print("\n=== Leakage Simulation Setup ===")

    lx, ly, lz = 2.0, 4.0, 4.0  # nm
    Nd = 5
    bias_V = 2.0

    slab = DielectricSlab(lx_nm=lx, ly_nm=ly, lz_nm=lz, Nd=Nd)
    slab.compute_effective_potential(bias_V=bias_V)

    # Run kinetic Monte Carlo transport
    slab.run_kmc(t_stop=1e-6, sample_interval=5e-9,
                 temperature_K=100, bias_V=bias_V)


if __name__ == "__main__":
    main()
