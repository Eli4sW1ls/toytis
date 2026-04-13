#!/usr/bin/env python3
"""Standalone version of staple_test.ipynb.

This script reproduces the notebook's 2D setup, runs the simulation, and
creates the same diagnostic plots without requiring Jupyter.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from simulation import Simulation


DEFAULT_WORK_DIR = Path("simulations") / "sim_script"


def build_settings_1d(intfs: list[float], simtype: str) -> dict:
    settings = {
        "interfaces": intfs,
        "simtype": "i*" if simtype == "istar" else simtype,
        "method": "load",
        "max_len": 200000,
        "dt": 0.005,
        "temperature": 1.0,
        "friction": 1.0,
        "high_friction": False,
        "max_cycles": 50000,
        "p_shoot": 0.9,
        "include_stateB": False,
        "prime_both_starts": True,
        "snake_Lmax": 0,
        "max_paths": 5,
        "dim": 1,
        "mass": 1,
    }
    if simtype == "retis":
        settings["max_cycles"] = 100000
        settings["prime_both_starts"] = False
    elif simtype == "istar":
        settings["v_ord"] = True
    return settings


def build_settings_2d(intfs: list[float], simtype: str) -> dict:
    settings = {
        "interfaces": intfs,
        "simtype": "i*" if simtype == "istar" else simtype,
        "method": "load",
        "max_len": 200000,
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
        "v_ord": False,
    }
    if simtype == "retis":
        settings["prime_both_starts"] = False
    elif simtype == "repptis":
        settings["dt"] = 0.01
        settings["temperature"] = 0.07
        settings["friction"] = 5.0
        settings["permeability"] = True
    return settings


def prepare_run_directory(project_root: Path, work_dir: Path) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    for entry in project_root.iterdir():
        if entry.is_file() and entry.suffix in {".py", ".png"}:
            shutil.copy2(entry, work_dir / entry.name)

    potentials_src = project_root / "potentials"
    if potentials_src.exists():
        shutil.copytree(potentials_src, work_dir / "potentials", dirs_exist_ok=True)

    return work_dir


def configure_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("[%(levelname)s] %(name)s %(funcName)s %(lineno)d: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def plot_potential_1d(sim: Simulation, intfs: list[float], output_dir: Path) -> None:
    x = np.linspace(-0.3, 1.5, 1000)
    pot = np.zeros_like(x)
    for i, xval in enumerate(x):
        pot[i], _ = sim.ensembles[0].engine.potential.potential_and_force((xval, 0.0))

    fig, ax = plt.subplots()
    ax.plot(x, pot)
    for intf in intfs:
        ax.axvline(intf, color="black", linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("potential")
    ax.set_title("Potential profile")
    fig.tight_layout()
    fig.savefig(output_dir / "potential_profile.png", dpi=200)


def plot_potential_surface(sim: Simulation, intfs: list[float], output_dir: Path) -> None:
    n_grid = 200
    xvals = np.linspace(0.0, 0.92, n_grid)
    yvals = np.linspace(-0.01, 0.31, n_grid)
    potvals = np.zeros((n_grid, n_grid))

    for i, xval in enumerate(xvals):
        for j, yval in enumerate(yvals):
            potvals[j, i], _ = sim.ensembles[0].engine.potential.potential_and_force(
                (np.array([yval, xval]), np.array([0.0, 0.0]))
            )

    fig, ax = plt.subplots(figsize=(8, 6))
    vmin = np.min(potvals)
    vmax = np.max(potvals)
    cs = ax.contourf(xvals, yvals, potvals, 100, cmap="viridis", vmin=vmin, vmax=vmax, extend="both")

    for idx, intf in enumerate(intfs):
        ax.axvline(
            intf,
            color="white",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            label="interfaces" if idx == 0 else None,
        )

    ax.set_title("Potential surface (potential_and_force)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    fig.colorbar(cs, ax=ax, label="potential", extend="both")
    fig.tight_layout()
    fig.savefig(output_dir / "potential_surface.png", dpi=200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staple_test simulation standalone.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Simulation working directory relative to the project root.",
    )
    parser.add_argument(
        "--mode",
        choices=["1d", "2d"],
        required=True,
        help="Simulation mode: 1d uses the 1D setup, 2d uses the 2D setup.",
    )
    parser.add_argument(
        "--simtype",
        choices=["istar", "retis", "repptis"],
        default="istar",
        help="Simulation type: istar (i*), retis, or repptis.",
    )
    parser.add_argument(
        "--interfaces",
        type=float,
        nargs="+",
        required=True,
        help="Space-separated interface values (example: --interfaces 0.05 0.13 0.21).",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots interactively after saving them.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        help="Time step (dt) for the simulation.",
    )
    parser.add_argument(
        "--friction",
        type=float,
        help="Friction coefficient.",
    )
    parser.add_argument(
        "--high-friction",
        action="store_true",
        help="Enable high friction mode (default is False).",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        help="Maximum number of cycles.",
    )
    parser.add_argument(
        "--mass",
        type=float,
        help="Mass scale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    work_dir = prepare_run_directory(project_root, project_root / args.work_dir)
    os.chdir(work_dir)
    print(os.getcwd())

    intfs = args.interfaces

    if args.mode == "1d":
        settings = build_settings_1d(intfs, args.simtype)
    else:
        settings = build_settings_2d(intfs, args.simtype)

    if args.dt is not None:
        settings["dt"] = args.dt
    if args.friction is not None:
        settings["friction"] = args.friction
    if args.high_friction:
        settings["high_friction"] = True
    if args.max_cycles is not None:
        settings["max_cycles"] = args.max_cycles
    if args.mass is not None:
        settings["mass"] = args.mass

    logger = configure_logging(work_dir / "logging.log")
    logger.info("\ninterfaces = {}\n".format(intfs) + "timestep = {}\n".format(settings["dt"]))
    
    sim = Simulation(settings)
    logger.info("Full settings:\n{}".format(sim.settings))

    logging.getLogger("matplotlib").setLevel(logging.DEBUG)
    sim.run()

    for ens in sim.ensembles:
        print(ens.intfs)

    plot_potential_1d(sim, intfs, work_dir)
    if args.mode == "2d":
        plot_potential_surface(sim, intfs, work_dir)

    if args.show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
