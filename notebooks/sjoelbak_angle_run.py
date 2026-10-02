#!/usr/bin/env python3
"""Run the angled sjoelbak potential at a given barrier angle.

Example:
    python notebooks/sjoelbak_angle_run.py --angle 30
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
from potentials.sjoelbak_angle import RectangularGridWithAngledBarrierPotential
from staple_test import prepare_run_directory, configure_logging


INTERFACES = [0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.53, 0.61, 0.69, 0.77, 0.85]


def build_settings(angle: float, simtype: str) -> dict:
    settings = {
        "interfaces": INTERFACES,
        "simtype": "i*" if simtype == "istar" else simtype,
        "method": "load",
        "max_len": 1000000,
        "dt": 0.002,
        "temperature": 0.05,
        "friction": 2.0,
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
        "zero_left": 0.1,
        "v_ord": True,
        "potential": RectangularGridWithAngledBarrierPotential(angle=angle),
    }
    if simtype == "retis":
        settings["prime_both_starts"] = False
    return settings


def plot_potential_surface(pot, intfs: list[float], output_dir: Path) -> None:
    """Potential with the order parameter y on the horizontal axis."""
    yvals = np.linspace(-0.02, pot.Ly + 0.02, 300)
    xvals = np.linspace(-0.01, pot.Lx + 0.01, 100)
    potvals = np.zeros((xvals.size, yvals.size))
    for j, xval in enumerate(xvals):
        for i, yval in enumerate(yvals):
            potvals[j, i] = pot.potential((np.array([yval, xval]), np.zeros(2)))

    fig, ax = plt.subplots(figsize=(10, 4))
    cs = ax.contourf(yvals, xvals, np.clip(potvals, None, 1.0), 100, cmap="viridis")
    for intf in intfs:
        ax.axvline(intf, color="white", linestyle="--", linewidth=1.0)
    ax.set_xlabel("y (order parameter)")
    ax.set_ylabel("x")
    ax.set_aspect("equal")
    ax.set_title("Angled sjoelbak, angle = {:g} deg, Lx = {:.4g}".format(pot.angle_deg, pot.Lx))
    fig.colorbar(cs, ax=ax, label="potential (clipped at 1)")
    fig.tight_layout()
    fig.savefig(output_dir / "potential_surface.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the angled sjoelbak at one barrier angle.")
    parser.add_argument("--angle", type=float, required=True, help="Barrier tilt in degrees.")
    parser.add_argument(
        "--simtype",
        choices=["istar", "retis", "repptis"],
        default="istar",
        help="Simulation type: istar (i*), retis, or repptis.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Working directory relative to the project root (default: simulations/sjoel_<simtype>_angle<angle>).",
    )
    parser.add_argument("--max-cycles", type=int, help="Maximum number of cycles.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(project_root)

    work_dir = args.work_dir or Path("simulations") / "sjoel_{}_angle{:g}".format(args.simtype, args.angle)
    work_dir = prepare_run_directory(project_root, project_root / work_dir)
    os.chdir(work_dir)
    print(os.getcwd())

    settings = build_settings(args.angle, args.simtype)
    if args.max_cycles is not None:
        settings["max_cycles"] = args.max_cycles
    pot = settings["potential"]

    logger = configure_logging(work_dir / "logging.log")
    logger.info("\ninterfaces = {}\n".format(INTERFACES) + "timestep = {}\n".format(settings["dt"]))
    logger.info("Barrier info:\n{}".format(pot.barrier_info()))
    plot_potential_surface(pot, INTERFACES, work_dir)

    sim = Simulation(settings)
    logger.info("Full settings:\n{}".format(sim.settings))
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    sim.run()


if __name__ == "__main__":
    main()
