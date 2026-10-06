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

TEMPERATURE = 0.05
# np.trapz was removed in NumPy 2.0 in favour of np.trapezoid.
trapezoid = getattr(np, "trapezoid", None) or np.trapz


def retis_interfaces(pot, temperature: float, interfaces: list[float],
                     n_grid: int = 801, n_x: int = 300) -> list[float]:
    """RETIS interfaces spaced by equal free-energy increments.

    With the uniform INTERFACES the whole barrier can land inside a single
    RETIS ensemble. Measured on the angle-0 run: [4+] (0.37 -> 0.45) carries
    6.8 of the 11.7 kT and crosses with probability ~1e-3 (7 crossing paths out
    of 28306), while the outer five ensembles each carry 0.0 kT and contribute
    nothing. Spacing instead by equal increments of the running maximum of
    F(lambda) = -kT ln int dx exp(-U_eff/kT) gives every ensemble a comparable
    crossing probability -- worst local probability 0.23 rather than 1e-3 at
    angle 0, and ~0.34 at the larger angles.

    lambda_A and lambda_B are left where they are: they define the states. Only
    the interior interfaces move, so the crossing probability being estimated is
    unchanged. i* and REPPTIS keep the uniform set, so their ensembles stay
    comparable with the earlier runs.
    """
    lam_a, lam_b, n_int = interfaces[0], interfaces[-1], len(interfaces)
    lam = np.linspace(lam_a, lam_b, n_grid)
    xs = np.linspace(-0.03, pot.Lx + 0.03, n_x)

    free = np.empty(n_grid)
    for i, lam_i in enumerate(lam):
        # Phasepoint positions are (y, x); U_eff is the potential the force
        # actually derives from, so it is the one the Boltzmann weight needs.
        u = np.array([pot.effective_potential((np.array([lam_i, x]), np.zeros(2)))
                      for x in xs])
        u_min = u.min()
        free[i] = -temperature * np.log(
            trapezoid(np.exp(-(u - u_min) / temperature), xs)) + u_min

    # Running maximum: the crossing probability from lambda_A out to lambda is
    # set by the highest free energy reached on the way, not by F(lambda).
    barrier = np.maximum.accumulate(free / temperature)
    barrier -= barrier[0]

    step = (lam_b - lam_a) / (n_grid - 1)
    out: list[float] = []
    for target in np.linspace(0.0, barrier[-1], n_int - 1):
        value = lam[min(int(np.searchsorted(barrier, target)), n_grid - 1)]
        if out and value <= out[-1]:
            value = out[-1] + step
        out.append(round(float(value), 6))
    out.append(float(lam_b))
    return out


def build_settings(angle: float, simtype: str) -> dict:
    pot = RectangularGridWithAngledBarrierPotential(angle=angle)
    interfaces = (retis_interfaces(pot, TEMPERATURE, INTERFACES)
                  if simtype == "retis" else list(INTERFACES))
    settings = {
        "interfaces": interfaces,
        "simtype": "i*" if simtype == "istar" else simtype,
        "method": "load",
        "max_len": 1000000,
        "dt": 0.002,
        "temperature": TEMPERATURE,
        "friction": 5.0,
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
        "potential": pot,
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

    interfaces = settings["interfaces"]
    logger = configure_logging(work_dir / "logging.log")
    logger.info("\ninterfaces = {}\n".format(interfaces) + "timestep = {}\n".format(settings["dt"]))
    if args.simtype == "retis":
        logger.info("RETIS interfaces spaced by equal free-energy increments "
                    "(uniform set was {})".format(INTERFACES))
    logger.info("Barrier info:\n{}".format(pot.barrier_info()))
    plot_potential_surface(pot, interfaces, work_dir)

    sim = Simulation(settings)
    logger.info("Full settings:\n{}".format(sim.settings))
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    sim.run()


if __name__ == "__main__":
    main()
