#!/usr/bin/env python3
"""Run one long Langevin MD trajectory on any 1D or nD potential.

The potential is given as module:Class, where the module is looked up in
potentials/ first and then in the project root. Constructor arguments are
passed as KEY=VALUE pairs (values are Python literals). The dimension follows
from the number of --start coordinates: 1 uses LangevinEngine, more uses
ndLangevinEngine.

The trajectory is written to md_path.txt (columns step, x1..x<dim>,
v1..v<dim>; one row every --stride steps; the start point, step 0, is not
included). It is appended to and flushed after every chunk, so a killed run
keeps what it had done.

Examples:
    python notebooks/md_run.py --potential cos_well:CosWellWalls --start -0.355 \\
        --dt 0.005 --temperature 1 --friction 1
    python notebooks/md_run.py --potential potential_staggered_channels:PotentialStaggeredChannels \\
        --pot-kwargs V_peak=1.7857 V_dip=0.75 --start -2.0 0.0 \\
        --dt 0.02 --temperature 0.25 --friction 1 --n-steps 10000000 --stride 10

Read back:
    d = np.loadtxt("md_path.txt"); step, x, v = d[:, 0], d[:, 1:dim + 1], d[:, dim + 1:]
(for very long runs, pandas.read_csv("md_path.txt", sep=" ", comment="#",
header=None) is much faster, or read it in pieces with usecols / max_rows)
"""

from __future__ import annotations

import os

# One BLAS thread per run, so parallel runs do not oversubscribe the cores.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import ast
import importlib
import logging
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "notebooks"))

from engine import LangevinEngine, ndLangevinEngine
from staple_test import prepare_run_directory, configure_logging


def load_potential(spec: str, kwargs: dict):
    """Instantiate a potential from 'module:Class', e.g. 'cos_well:CosWellWalls'."""
    module_name, sep, class_name = spec.partition(":")
    if not sep or not class_name:
        raise ValueError("--potential must be module:Class, got {!r}".format(spec))
    try:
        module = importlib.import_module("potentials." + module_name)
    except ModuleNotFoundError:
        module = importlib.import_module(module_name)
    return getattr(module, class_name)(**kwargs)


def parse_kwargs(pairs: list[str]) -> dict:
    kwargs = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError("--pot-kwargs entries must be KEY=VALUE, got {!r}".format(pair))
        try:
            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kwargs[key] = value  # plain string, e.g. a file path
    return kwargs


def plot_potential(pot, output_dir: Path) -> None:
    if not hasattr(pot, "plot_potential"):
        return
    fig, ax = plt.subplots()
    try:
        pot.plot_potential(ax)
        fig.savefig(output_dir / "potential.png", dpi=200)
    except Exception as exc:  # plotting is a convenience, never fail the run on it
        logging.getLogger().warning("Could not plot the potential: {}".format(exc))
    plt.close("all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a long Langevin MD trajectory on any 1D/nD potential.")
    parser.add_argument(
        "--potential", required=True,
        help="Potential as module:Class; the module is looked up in potentials/ first (e.g. cos_well:CosWellWalls).",
    )
    parser.add_argument(
        "--pot-kwargs", nargs="+", default=[], metavar="KEY=VALUE",
        help="Constructor arguments of the potential (Python literals, e.g. V_peak=1.5 mazefig='maze.png').",
    )
    parser.add_argument(
        "--start", type=float, nargs="+", required=True,
        help="Start position; its length sets the dimension. Velocities are drawn from Maxwell-Boltzmann.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed (use different seeds for parallel runs).")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Working directory relative to the project root (default: simulations/md_<Class>_s<seed>).",
    )
    parser.add_argument("--n-steps", type=int, default=50_000_000, help="Number of MD steps.")
    parser.add_argument("--stride", type=int, default=1, help="Save every stride-th step.")
    parser.add_argument(
        "--chunk", type=int, default=100_000,
        help="Saved rows per chunk; rows are written and the progress bar updated after each chunk.",
    )
    parser.add_argument("--dt", type=float, default=0.005, help="Time step.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature (kB = 1).")
    parser.add_argument("--friction", type=float, default=1.0, help="Friction coefficient.")
    parser.add_argument("--mass", type=float, default=1.0, help="Mass.")
    parser.add_argument("--high-friction", action="store_true", help="Use overdamped (Brownian) dynamics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(project_root)
    np.random.seed(args.seed)

    class_name = args.potential.partition(":")[2]
    work_dir = args.work_dir or Path("simulations") / "md_{}_s{}".format(class_name, args.seed)
    work_dir = prepare_run_directory(project_root, project_root / work_dir)
    os.chdir(work_dir)
    print(os.getcwd())

    dim = len(args.start)
    pot_kwargs = parse_kwargs(args.pot_kwargs)
    settings = {
        "dt": args.dt,
        "temperature": args.temperature,
        "friction": args.friction,
        "high_friction": args.high_friction,
        "mass": args.mass,
        "dim": dim,
        "potential": load_potential(args.potential, pot_kwargs),
    }
    engine = (LangevinEngine if dim == 1 else ndLangevinEngine)(settings)
    pot = settings["potential"]

    n_rows = args.n_steps // args.stride
    logger = configure_logging(work_dir / "logging.log")
    logger.info("Long MD: n_steps = {}, stride = {} ({} rows), start = {}".format(
        n_rows * args.stride, args.stride, n_rows, args.start))
    logger.info("timestep = {}, time per row = {}".format(args.dt, args.dt * args.stride))
    logger.info("Potential: {} {}".format(args.potential, pot_kwargs))
    logger.info("Potential params:\n{}".format(getattr(pot, "params", None)))
    logger.info("Seed: {}".format(args.seed))
    logger.info("Engine settings:\n{}".format(settings))
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    plot_potential(pot, work_dir)

    x0 = args.start[0] if dim == 1 else np.array(args.start)
    ph = (x0, engine.draw_velocities())
    step = engine.step
    buf = np.empty((args.chunk, 1 + 2 * dim))
    header = "step " + " ".join(["x{}".format(i + 1) for i in range(dim)] + ["v{}".format(i + 1) for i in range(dim)])
    fmt = ["%d"] + ["%.7g"] * (2 * dim)

    with open("md_path.txt", "w") as out, \
            tqdm(total=n_rows * args.stride, unit="step", unit_scale=True, smoothing=0.05) as pbar:
        out.write("# " + header + "\n")
        for row0 in range(0, n_rows, args.chunk):
            m = min(args.chunk, n_rows - row0)
            for k in range(m):
                for _ in range(args.stride):
                    ph = step(ph)
                buf[k, 0] = (row0 + k + 1) * args.stride
                buf[k, 1:dim + 1] = ph[0]
                buf[k, dim + 1:] = ph[1]
            np.savetxt(out, buf[:m], fmt=fmt)
            out.flush()
            pbar.update(m * args.stride)
            pbar.set_postfix(x=np.array2string(np.atleast_1d(ph[0]), precision=2))

    logger.info("Done: {} MD steps, final phasepoint {}".format(engine.mdsteps, ph))


if __name__ == "__main__":
    main()
