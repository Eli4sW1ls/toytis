#!/usr/bin/env python3
"""Run RETIS, REPPTIS or stapleTIS (i*) on the staggered two-channel potential.

Examples:
    python notebooks/staggered_channels_run.py --simtype retis --seed 1
    python notebooks/staggered_channels_run.py --simtype repptis --seed 1
    python notebooks/staggered_channels_run.py --simtype istar --seed 1
"""

from __future__ import annotations

import os

# One BLAS thread per run, so parallel runs do not oversubscribe the cores.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import logging
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "notebooks"))

from simulation import Simulation
from potentials.potential_staggered_channels import PotentialStaggeredChannels
from staple_test import prepare_run_directory, configure_logging


TEMPERATURE = 0.25
# Barrier 7.1 kT, dips 3 kT. Same landscape in kT units as the T = 0.07
# defaults, but diffusion is faster, so paths are ~10x shorter in steps.
V_PEAK = 0.5 / 0.07 * TEMPERATURE
V_DIP = 3.0 * TEMPERATURE
# States at -/+1.6, clear of the obstacle wall. Spacing ~0.2 inside the
# channels, with interfaces on the kinks at -/+0.6 (x_kink = 0.4), so that no
# interval contains a barrier top or dip bottom. No interface at x = 0: there
# the two channels are equally populated, and a local ensemble at 0 would keep
# whichever channel it starts in. At -/+0.1 one channel dominates (92%).
# No interface at -/+1.0 either: both channels are populated at the mouths, and
# an ensemble ending at -/+1.6 cannot switch channel (the obstacle wall reaches
# x = -/+1.5). From -/+0.8 inwards every ensemble is >= 92% one channel.
INTERFACES = [-1.6, -0.8, -0.6, -0.4, -0.25, -0.1, 0.1, 0.25, 0.4, 0.6, 0.8, 1.6]


def build_settings(simtype: str) -> dict:
    settings = {
        "interfaces": INTERFACES,
        "simtype": "i*" if simtype == "istar" else simtype,
        "method": "load",
        "max_len": 200000,
        "dt": 0.02,
        "temperature": TEMPERATURE,
        # Ballistic length sqrt(kT/m)/gamma = 0.5, about 2.5 interface spacings:
        # turns then identify the channel (staple paths stay in one channel),
        # while velocity still decorrelates within a turn.
        "friction": 1.0,
        "high_friction": False,
        "max_cycles": 100000,
        "p_shoot": 0.9,
        "include_stateB": False,
        "prime_both_starts": True,
        "snake_Lmax": 0,
        "max_paths": 5,
        "dim": 2,
        "mass": 1,
        "permeability": False,
        "zero_left": 0.2,
        "v_ord": True,
        "potential": PotentialStaggeredChannels(V_peak=V_PEAK, V_dip=V_DIP),
    }
    if simtype == "retis":
        settings["prime_both_starts"] = False
        # The high ensembles are nearly identical (crossing probability ~1 past
        # x = 0.6), so swaps between them are cheap and mix paths well.
        settings["p_shoot"] = 0.5
    return settings


def plot_potential_surface(pot, intfs: list[float], output_dir: Path) -> None:
    xvals = np.linspace(-2.9, 2.9, 290)
    yvals = np.linspace(-1.9, 1.9, 190)
    potvals = np.zeros((yvals.size, xvals.size))
    for j, yval in enumerate(yvals):
        for i, xval in enumerate(xvals):
            potvals[j, i] = pot.potential((np.array([xval, yval]), np.zeros(2)))

    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = 1.5 * pot.params["V_peak"]
    cs = ax.contourf(xvals, yvals, np.clip(potvals, None, vmax), 100, cmap="viridis")
    for intf in intfs:
        ax.axvline(intf, color="white", linestyle="--", linewidth=1.0)
    ax.set_xlabel("x (order parameter)")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title("Staggered channels: V_peak = {:.2f}, V_dip = {:.2f}, T = {:g}".format(
        pot.params["V_peak"], pot.params["V_dip"], TEMPERATURE))
    fig.colorbar(cs, ax=ax, label="potential (clipped)")
    fig.tight_layout()
    fig.savefig(output_dir / "potential_surface.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staggered two-channel benchmark.")
    parser.add_argument(
        "--simtype",
        choices=["retis", "repptis", "istar"],
        required=True,
        help="Simulation type: retis (benchmark), repptis, or istar (stapleTIS).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed (use different seeds for parallel runs).")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Working directory relative to the project root (default: simulations/stag_<simtype>_s<seed>).",
    )
    parser.add_argument("--max-cycles", type=int, help="Maximum number of cycles.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(project_root)
    np.random.seed(args.seed)

    work_dir = args.work_dir or Path("simulations") / "stag_{}_s{}".format(args.simtype, args.seed)
    work_dir = prepare_run_directory(project_root, project_root / work_dir)
    os.chdir(work_dir)
    print(os.getcwd())

    settings = build_settings(args.simtype)
    if args.max_cycles is not None:
        settings["max_cycles"] = args.max_cycles
    pot = settings["potential"]

    logger = configure_logging(work_dir / "logging.log")
    logger.info("\ninterfaces = {}\n".format(INTERFACES) + "timestep = {}\n".format(settings["dt"]))
    logger.info("Potential params:\n{}".format(pot.params))
    logger.info("Seed: {}".format(args.seed))
    plot_potential_surface(pot, INTERFACES, work_dir)

    sim = Simulation(settings)
    logger.info("Full settings:\n{}".format(sim.settings))
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    sim.run()


if __name__ == "__main__":
    main()
