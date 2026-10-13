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
import heapq
import logging
from pathlib import Path
import re
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

# RETIS only: 12 interfaces (11 ensembles past [0-]) instead of 11. The extra
# one splits the widest uphill step -- 0.265 -> 0.401 at angle 0, the one
# ensemble the two mis-analysed angle-0/10 reports could not bound from below.
N_RETIS_INTERFACES = 12

TEMPERATURE = 0.05


def retis_interfaces(pot, temperature: float, lam_a: float, lam_b: float,
                     n_int: int = N_RETIS_INTERFACES,
                     n_y: int = 801, n_x: int = 60) -> list[float]:
    """RETIS interfaces at equal steps of the estimated -ln P_A(lambda).

    lambda_A and lambda_B stay where they are (they define the states); the
    n_int - 2 interior interfaces are placed so that every ensemble crosses
    with a comparable probability. With the uniform INTERFACES the whole
    barrier can land in one ensemble: at angle 0, [4+] carried ~7 kT and
    crossed 7 times in 28306 paths while the five ensembles after it carried
    nothing.

    -ln P_A(lambda) is estimated in kT as the larger of two lower bounds, each
    blind to something the other sees:

    * the running maximum of the lambda-projected free energy
      F(lambda) = -kT ln int dx exp(-U_eff/kT). Right at small angles, where
      the barrier is perpendicular to lambda (angle 0: 6.8 kT for 0.37 -> 0.45,
      measured 6.5-7.3), but it cannot see a tilted ridge: once the corridor
      opens, every lambda has barrier-free configurations, so the projection
      misses the ridge entirely (angle 45: 0.6 kT for 0.45 -> 0.53, measured
      3.3-3.8).

    * the mountain-pass energy: the lowest possible maximum of U_eff along any
      path from lambda_A to the level lambda. It sees that beyond the corridor
      a path has to cross the ridge at its cheapest point (angle 45: 4.8 kT),
      but has no entropy, so it under-rates broad uphill stretches.

    Their maximum put the biggest step in the ensemble that actually
    bottlenecked in every earlier uniform run at 10, 25, 35 and 45 degrees
    (estimated 6.2 / 3.7 / 4.3 / 4.1 kT, measured 7.2 / 3.6 / 4.6 / 4.0); the
    free energy alone got 25-45 degrees wrong every time. Both are static, so
    the placement does not depend on the friction.

    i* and REPPTIS keep the uniform set, so their ensembles stay comparable
    with the earlier runs.
    """
    ys = np.linspace(lam_a, lam_b, n_y)
    # Inside the channel only: the walls are no-go regions, not part of a path.
    xs = (np.arange(n_x) + 0.5) * pot.Lx / n_x
    # Phasepoint positions are (y, x); U_eff is the potential the force derives
    # from, so it is the one the Boltzmann weight and the energy bound need.
    beta_u = np.array([[pot.effective_potential((np.array([y, x]), np.zeros(2)))
                        for x in xs] for y in ys]) / temperature

    shift = beta_u.min(axis=1)
    free = shift - np.log(np.exp(-(beta_u - shift[:, None])).sum(axis=1))
    free = np.maximum.accumulate(free - free[0])

    # Lowest achievable maximum of beta*U_eff on the way to each grid point
    # (Dijkstra with max instead of +), started from the whole lambda_A row.
    best = np.full(beta_u.shape, np.inf)
    best[0] = beta_u[0]
    heap = [(beta_u[0, j], 0, j) for j in range(n_x)]
    heapq.heapify(heap)
    while heap:
        value, i, j = heapq.heappop(heap)
        if value > best[i, j]:
            continue
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < n_y and 0 <= b < n_x:
                new = max(value, beta_u[a, b])
                if new < best[a, b]:
                    best[a, b] = new
                    heapq.heappush(heap, (new, a, b))
    mountain = best.min(axis=1)              # any path to y >= lambda crosses row lambda
    mountain = mountain - mountain[0]

    cost = np.maximum.accumulate(np.maximum(free, mountain))

    step = (lam_b - lam_a) / (n_y - 1)
    out: list[float] = []
    for target in np.linspace(0.0, cost[-1], n_int - 1):
        value = ys[min(int(np.searchsorted(cost, target)), n_y - 1)]
        if out and value <= out[-1]:
            value = out[-1] + step
        out.append(round(float(value), 6))
    out.append(float(lam_b))
    return out


def write_retis_rst(work_dir: Path, settings: dict, logger: logging.Logger) -> None:
    """Write work_dir/retis.rst for pyretisanalyse, with this run's interfaces.

    The shared simulations/retis.rst hard-codes the uniform set. Analysing a
    RETIS run made with the free-energy set against it silently scores every
    ensemble past [0+] against the wrong interface (the reports that gave
    Pcross ~1e-12 at angles 0 and 10). This copy carries the values the run
    actually used; analyse from inside the run directory.
    """
    template = next((p for p in (work_dir.parent / "retis.rst",
                                 project_root / "simulations" / "retis.rst") if p.is_file()), None)
    line = "interfaces = {}".format(settings["interfaces"])
    if template is None:
        logger.warning("No retis.rst template found; for pyretisanalyse use: %s", line)
        return
    values = {
        "interfaces": settings["interfaces"],
        "zero_left": settings["zero_left"],
        "temperature": settings["temperature"],
        "timestep": settings["dt"],
        "gamma": settings["friction"],
    }
    lines = template.read_text().splitlines()
    for key, value in values.items():
        pattern = re.compile(r"^(\s*){}\s*=".format(key))
        for k, text in enumerate(lines):
            match = pattern.match(text)
            if match:
                lines[k] = "{}{} = {}".format(match.group(1), key, value)
                break
        else:
            logger.warning("Key %r not found in %s; left unset in retis.rst", key, template)
    (work_dir / "retis.rst").write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s from %s with %s", work_dir / "retis.rst", template, line)


def build_settings(angle: float, simtype: str) -> dict:
    pot = RectangularGridWithAngledBarrierPotential(angle=angle)
    interfaces = (retis_interfaces(pot, TEMPERATURE, INTERFACES[0], INTERFACES[-1])
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
        logger.info("RETIS interfaces placed at equal steps of the estimated -ln P_A "
                    "(uniform set was {})".format(INTERFACES))
        write_retis_rst(work_dir, settings, logger)
    logger.info("Barrier info:\n{}".format(pot.barrier_info()))
    plot_potential_surface(pot, interfaces, work_dir)

    sim = Simulation(settings)
    logger.info("Full settings:\n{}".format(sim.settings))
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    sim.run()


if __name__ == "__main__":
    main()
