#!/usr/bin/env python3
"""StapleTIS (i*) on the Z potential with a coupled slow coordinate w (potentials/zpot_w.ZChannelsW).

Everything as the 2D Z run sim_istarz_br_g10 (overdamped, gamma 10, dt 0.002, kT 0.05, the same 11 interfaces
on y, 100000 cycles), with dim = 3: positions (y, x, w). order.txt gets the columns y, x, w. The sampler is not
changed: nd_patches makes the order parameter write all three coordinates and lets kick_star make 3D kick
points (w drawn near one of the two minima, channel at random).

Examples:
    python notebooks/zpot_w_run.py --seed 1                       # fresh run: simulations/sim_istarzw_br_g10_dV3_s1
    python notebooks/zpot_w_run.py --seed 1 --max-cycles 200      # pilot
    python notebooks/zpot_w_run.py --seed 1 --restart --max-cycles 150000
Long runs: in their own unit, e.g.
    systemd-run --user --unit=zw-s1 --collect -p MemoryMax=8G -p MemorySwapMax=0 \\
        --working-directory=$PWD python notebooks/zpot_w_run.py --seed 1
"""

from __future__ import annotations

import os

# One BLAS thread per run, so parallel runs do not oversubscribe the cores.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "notebooks"))

import nd_patches  # noqa: E402  (before Simulation and before loading a restart.pkl)
nd_patches.install()

from simulation import Simulation  # noqa: E402
from potentials.zpot_w import ZChannelsW  # noqa: E402
from staple_test import open_run_directory, configure_logging  # noqa: E402


INTERFACES = [0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.53, 0.61, 0.69, 0.77, 0.85]   # as sim_istarz_br_g10
TEMPERATURE = 0.05


def build_settings(dV_kT: float, w0: float, delta: float) -> dict:
    pot = ZChannelsW(w0=w0, dV_w=dV_kT * TEMPERATURE, delta=delta, Lx=0.3, angle=45.0)
    return {
        "interfaces": list(INTERFACES),
        "simtype": "i*",
        "method": "load",
        "max_len": 200000,
        "dt": 0.002,
        "temperature": TEMPERATURE,
        "friction": 10.0,
        "high_friction": True,
        "max_cycles": 100000,
        "p_shoot": 0.9,
        "include_stateB": False,
        "prime_both_starts": True,
        "snake_Lmax": 0,
        "max_paths": 5,
        "dim": 3,
        "mass": 1,
        "permeability": False,
        "zero_left": 0.1,
        "v_ord": False,
        "potential": pot,
    }


def plot_potential_slices(pot: ZChannelsW, intfs: list[float], output_dir: Path) -> None:
    """Effective potential (y, x) in the two channels w = -w0, +w0, and along w at the X crossing."""
    yv = np.linspace(-0.02, pot.Ly + 0.02, 240)
    xv = np.linspace(-0.01, pot.Lx + 0.01, 80)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw=dict(width_ratios=[1, 1, 0.6]))
    for ax, w in zip(axes[:2], (-pot.w0, pot.w0)):
        u = np.array([[pot.effective_potential((np.array([y, x, w]), None)) for y in yv] for x in xv])
        cs = ax.contourf(yv, xv, np.clip(u - u.min(), None, 0.6) / TEMPERATURE, 60, cmap="viridis")
        for i in intfs:
            ax.axvline(i, color="white", ls="--", lw=0.8)
        ax.set_title(f"w = {w:+g}")
        ax.set_xlabel("y (order parameter)")
        ax.set_ylabel("x")
        fig.colorbar(cs, ax=ax, label="U_eff / kT (clipped at 12)")
    wv = np.linspace(-2 * pot.w0, 2 * pot.w0, 300)
    for (y, x), c in zip([(0.30, 0.02), (0.45, 0.15), (0.30, 0.28)], ("C0", "C1", "C2")):
        u = [pot.potential((np.array([y, x, w]), None)) / TEMPERATURE for w in wv]
        axes[2].plot(wv, np.array(u) - min(u), color=c, label=f"y={y}, x={x}")
    axes[2].set_xlabel("w")
    axes[2].set_ylabel("U / kT (min 0)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "potential_slices.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=1, help="np.random seed (different seeds for parallel runs)")
    p.add_argument("--dV", type=float, default=3.0, help="w barrier dV_w in kT (default 3)")
    p.add_argument("--w0", type=float, default=0.1, help="w minima at +-w0")
    p.add_argument("--delta", type=float, default=0.02, help="width of the channel switch in w")
    p.add_argument("--work-dir", type=Path,
                   help="relative to the project root (default simulations/sim_istarzw_br_g10_dV<dV>_s<seed>); "
                        "a fresh run DELETES this directory first")
    p.add_argument("--max-cycles", type=int, help="maximum number of cycles (default 100000)")
    p.add_argument("--restart", action="store_true",
                   help="continue the run in --work-dir from its restart.pkl (a larger --max-cycles extends it)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    os.chdir(project_root)
    work_dir = args.work_dir or Path("simulations") / f"sim_istarzw_br_g10_dV{args.dV:g}_s{args.seed}"
    work_dir = open_run_directory(project_root, project_root / work_dir, args.restart)
    os.chdir(work_dir)
    print(os.getcwd())

    settings = build_settings(args.dV, args.w0, args.delta)
    if args.max_cycles is not None:
        settings["max_cycles"] = args.max_cycles
    if args.restart:
        settings["method"] = "restart"
    pot = settings["potential"]

    logger = configure_logging(work_dir / "logging.log")
    logger.info("\ninterfaces = {}\n".format(settings["interfaces"]) + "timestep = {}\n".format(settings["dt"]))
    logger.info("Seed: {}".format(args.seed))
    logger.info("zpot_w: dV_w = {:g} kT, w0 = {:g}, delta = {:g}".format(args.dV, args.w0, args.delta))
    logger.info("Potential info:\n{}".format(pot.info()))
    if not args.restart:
        plot_potential_slices(pot, settings["interfaces"], work_dir)

    sim = Simulation(settings)
    logger.info("Full settings:\n{}".format(sim.settings))
    import logging
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    sim.run()


if __name__ == "__main__":
    main()
