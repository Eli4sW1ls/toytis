#!/usr/bin/env python3
"""Where the staggered two-channel runs put their phase points, turns and joins.

Companion of stag_analysis.py (same run folders, same tistools weights). It
shows that the orthogonal coordinate y matters, and where REPPTIS and
StapleTIS concatenate their paths. The momentum version for the 1D well is
APPTIS/figures/scripts/lmstar_md_compare.py; the event definitions (blocks,
turns, closing crossings) are the same as there.

Subcommands
-----------
extract RUN...   Read every ensemble's order.txt once and cache, per ensemble, the
                 phase-point densities and the crossing / turn events below in
                 RUN/joins/NNN.npz. Works for i* and RETIS runs (RETIS: density only).
fig              All figures from the caches of one or more i* seeds (pooled) and
                 optionally RETIS runs (density reference):
                 PREFIX_density.png       P(y | x) of the sampled phase points and the
                                          upper-channel share along x
                 PREFIX_where.png         where the joins happen: y at each REPPTIS join
                                          vs y at each StapleTIS join, on the potential
                 PREFIX_joins_VAR.png     per interface, the two distributions each
                                          method glues together (VAR = y, px, vy)
                 PREFIX_summary.png       upper-channel share on both sides of every
                                          join and the overlap of the two distributions

Joins, in block notation (crossings of the same interface merged into one block;
three blocks i, j, i form a turn at lambda_j; see lmstar_md_compare.py)
----------------------------------------------------------------------
Ensemble folder e (e >= 2) is [(e-1)+-] / [(e-1)*] with window lambda_{e-2},
lambda_{e-1}, lambda_{e}. Column m below is the interface lambda_m at which the
variable is taken (an upward crossing, or downward with --down).

REPPTIS at lambda_m (up): the MSM continues a [(m-1)+-] LMR path, which leaves its
    window over lambda_m (blocks m-2, m-1, m), with the [m+-] LM* statistics, whose
    first lambda_m crossing comes either straight from below (m-2, m-1, m) or after a
    turn at lambda_{m-1} (m, m-1, m). Memoryless only if both give the same y.
    From the full staple path around each [m+-] window the two histories are
    separated (dashed lines); the "straight" one must match the [(m-1)+-] LMR exits.
    An LML window stored with its forward extension starts at its opening crossing,
    so its history is unknown; those weights go to the known LML windows (the
    extension side is chosen independently of the history).
StapleTIS at lambda_m (up): the MSM joins at turns. A turn at lambda_{m-1} (blocks
    m, m-1, m; minimum in (lambda_{m-2}, lambda_{m-1}]) is the window of a [(m-1)*]
    RMR path, and an extension of the [m*] paths that end in it or start with it.
    These are the same events, so the y at the closing lambda_m crossing must agree
    between [(m-1)*] and [m*] (ends and starts).
Down: mirror images (RML exits of [(m+1)+-] over lambda_m vs [m+-] RM*; turns at
    lambda_{m+1} as [(m+1)*] LML windows vs [m*] extensions).

Weights: REPPTIS events use the path weights of the window ensemble (tistools
get_weights, as in stag_analysis.repptis_profile); StapleTIS events the staple
weights (get_weights_staple). Densities: each ensemble normalised to one path,
then summed (a sampling density, not the equilibrium one); P(y | x) normalises
every x column.

Typical use (from the toytis root, workstation venv with tistools):
    python notebooks/stag_joins.py extract simulations/stagd3g01_istar_s{1,2,3,4} simulations/stagd3g01_retis_s6
    python notebooks/stag_joins.py fig --istar simulations/stagd3g01_istar_s{1,2,3,4} \\
        --retis simulations/stagd3g01_retis_s6 -o figs/g01
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stag_analysis import X_IN, Y_IN, ensembles, interfaces, load_pes, quiet, simtype, staple_weights, tistools  # noqa: E402

VERSION = "v1"
# Density grid
XB = np.linspace(-2.2, 2.2, 177)
YB = np.linspace(-2.0, 2.0, 101)
# Event types
EXIT_R, EXIT_L, FIRST_UP, FIRST_DN, TURN = range(5)
# History of a first crossing of the middle interface
H_UNKNOWN, H_STRAIGHT, H_TURNED, H_OTHER = range(4)
LMR_CODE = {"LML": 0, "LMR": 1, "RML": 2, "RMR": 3}
FIELDS = ("etype", "col", "y", "vx", "vy", "ws", "wr", "hist", "lmr", "dir", "side", "kind", "ext_x", "ext_y", "cyc")


# ---------------------------------------------------------------------------
# geometry of one path
# ---------------------------------------------------------------------------
def crossings(x, intf):
    """Every interface crossing of x in time order: interface index, step t (t -> t+1),
    upward flag and interpolation fraction. A step over several interfaces gives one
    crossing per interface, in the order they are passed (as lmstar_md_compare)."""
    level = np.searchsorted(intf, x, side="left")
    step = np.flatnonzero(level[1:] != level[:-1])
    lo, hi = level[step], level[step + 1]
    n = np.abs(hi - lo)
    step = np.repeat(step, n)
    rank = np.arange(len(step)) - np.repeat(np.cumsum(n) - n, n)
    up = np.repeat(hi > lo, n)
    lo = np.repeat(lo, n)
    iface = np.where(up, lo + rank, lo - 1 - rank)
    frac = (intf[iface] - x[step]) / (x[step + 1] - x[step])
    return iface, step, up, frac


def blocks_of(iface):
    if not len(iface):
        return np.empty(0, int), np.empty(0, int)
    first = np.concatenate([[0], np.flatnonzero(iface[1:] != iface[:-1]) + 1])
    return iface[first], first


def path_events(x, y, vx, vy, s0, s1, e, lmr, dirn, ws, wr, cyc, intf, window):
    """Crossing and turn events of one staple (or window-only) path of folder e."""
    out = []
    iface, step, up, frac = crossings(x, intf)
    if not len(iface):
        return out
    bval, first = blocks_of(iface)
    at = lambda c, a: a[step[c]] + frac[c] * (a[step[c] + 1] - a[step[c]])
    base = dict(ws=ws, wr=wr, lmr=LMR_CODE.get(lmr, -1), dir=dirn, cyc=cyc, hist=H_UNKNOWN,
                side=-1, kind=-1, ext_x=np.nan, ext_y=np.nan)

    def add(etype, col, c, **kw):
        out.append(dict(base, etype=etype, col=col, y=at(c, y), vx=at(c, vx), vy=at(c, vy), **kw))

    # turns at either end: blocks (k+1, k, k+1) = minimum at lambda_k, (k-1, k, k-1) = maximum
    n_last = len(x) - 1
    for side, inside in ((0, intf[0] < x[0] < intf[-1]), (1, intf[0] < x[-1] < intf[-1])):
        if not inside or len(bval) < 3:
            continue
        a, b, c = bval[:3] if side == 0 else bval[-3:]
        if a != c or abs(a - b) != 1:
            continue
        ci = first[2] if side == 0 else first[-1]
        kind = 0 if a == b + 1 else 1
        if up[ci] != (kind == 0):
            continue
        lo_, hi_ = (0, step[ci] + 1) if side == 0 else (step[first[-3]], n_last + 1)
        j = lo_ + (np.argmin(x[lo_:hi_]) if kind == 0 else np.argmax(x[lo_:hi_]))
        add(TURN, a, ci, side=side, kind=kind, ext_x=x[j], ext_y=y[j])

    if not window or e < 2:
        return out
    # REPPTIS window: steps s0-1 .. s1 (frames s0-1 .. s1+1), interfaces e-2, e-1, e
    inwin = (step >= s0 - 1) & (step <= s1)
    sel = lambda k, upward: np.flatnonzero(inwin & (iface == k) & (up == upward))
    if lmr == "LMR" and len(c := sel(e, True)):
        add(EXIT_R, e, c[-1])
    if lmr == "RML" and len(c := sel(e - 2, False)):
        add(EXIT_L, e - 2, c[-1])
    for upward, etype, entry, straight, turned, state in (
            (True, FIRST_UP, "L", e - 3, e - 1, x[0] <= intf[0]),
            (False, FIRST_DN, "R", e + 1, e - 1, x[0] >= intf[-1])):
        if lmr[:1] != entry or not len(c := sel(e - 1, upward)):
            continue
        c = c[0]
        b = np.searchsorted(first, c, side="right") - 1
        if b >= 2:
            h = bval[b - 2]
            hist = H_STRAIGHT if h == straight else H_TURNED if h == turned else H_OTHER
        else:
            hist = H_STRAIGHT if state else H_UNKNOWN
        add(etype, e - 1, c, hist=hist)
    return out


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------
def iter_blocks(fn, chunksize):
    """(block index, frames) per cycle block of order.txt: the time column restarts at 0
    for every block, so the blocks are found without parsing the headers."""
    import pandas as pd
    reader = pd.read_csv(fn, sep=r"\s+", comment="#", header=None, names=range(5), dtype=np.float64,
                         chunksize=chunksize, engine="c")
    carry, b = None, 0
    for chunk in reader:
        a = chunk.to_numpy()
        a = a[~np.isnan(a[:, 1])]   # a run still writing can end on a partial line
        if carry is not None:
            a = np.vstack([carry, a])
        starts = np.flatnonzero(a[:, 0] == 0)
        if not len(starts):
            carry = a
            continue
        for j in range(len(starts) - 1):
            yield b, a[starts[j]:starts[j + 1]]
            b += 1
        carry = a[starts[-1]:]
    if carry is not None and len(carry):
        yield b, carry


def extract_run(run, skip=0, chunksize=4_000_000, force=False):
    tt = tistools()
    is_istar = simtype(run) == "i*"
    intf = np.array(interfaces(run))
    with quiet():
        pes, _ = load_pes(run)
    W = staple_weights(pes, len(intf)) if is_istar else {i: np.asarray(pe.weights, float) for i, pe in enumerate(pes)}
    outdir = os.path.join(run, "joins")
    os.makedirs(outdir, exist_ok=True)
    nx, ny = len(XB) - 1, len(YB) - 1
    for e, ens in enumerate(ensembles(run)):
        if e == 0:
            continue    # [0-] lies in A
        fn = os.path.join(ens, "order.txt")
        cache = os.path.join(outdir, f"{e:03d}.npz")
        key = f"{VERSION} {os.path.getsize(fn)} {skip}"
        if not force and os.path.exists(cache):
            with np.load(cache) as d:
                if str(d["key"]) == key:
                    print(f"  {run}: {e:03d} cached", flush=True)
                    continue
        pe = pes[e]
        ws = np.array(W[e], float)
        wr = np.asarray(pe.weights, float).copy()
        cycles = np.asarray(pe.cyclenumbers)
        ws[cycles < skip] = 0
        wr[cycles < skip] = 0
        acc = np.asarray(pe.flags) == "ACC"
        Hs, Hr = np.zeros(nx * ny), np.zeros(nx * ny)
        ev, n_used, n_bad = [], 0, 0
        idx_s, w_s, idx_r, w_r = [], [], [], []

        def flush():
            if idx_s:
                np.add.at(Hs, np.concatenate(idx_s), np.concatenate(w_s))
            if idx_r:
                np.add.at(Hr, np.concatenate(idx_r), np.concatenate(w_r))
            idx_s.clear(); w_s.clear(); idx_r.clear(); w_r.clear()

        for b, a in iter_blocks(fn, chunksize):
            if b >= len(acc):
                break
            if not acc[b] or (ws[b] == 0 and wr[b] == 0):
                continue
            if len(a) != pe.lengths[b]:
                n_bad += 1
                continue
            x, y = a[:, 1], a[:, 2]
            vx = a[:, 3] if a.shape[1] > 3 else np.full(len(x), np.nan)
            vy = a[:, 4] if a.shape[1] > 4 else np.full(len(x), np.nan)
            s0, s1 = (int(pe.istar_idx[b][0]), int(pe.istar_idx[b][1])) if is_istar else (1, len(x) - 2)
            s0, s1 = max(s0, 1), min(s1, len(x) - 2)
            ix = np.clip(np.searchsorted(XB, x) - 1, 0, nx - 1)
            iy = np.clip(np.searchsorted(YB, y) - 1, 0, ny - 1)
            flat = ix * ny + iy
            if ws[b]:
                idx_s.append(flat); w_s.append(np.full(len(flat), ws[b]))
            if wr[b] and is_istar:
                win = slice(s0 - 1, s1 + 2)
                idx_r.append(flat[win]); w_r.append(np.full(len(flat[win]), wr[b]))
            if is_istar:
                ev += path_events(x, y, vx, vy, s0, s1, e, pe.lmrs[b], int(pe.dirs[b]), ws[b], wr[b], int(cycles[b]),
                                  intf, window=True)
            n_used += 1
            if len(idx_s) > 2000:
                flush()
        flush()
        tot_s, tot_r = ws[: len(acc)].sum(), wr[: len(acc)].sum()
        arrays = {f: np.array([d[f] for d in ev], float) for f in FIELDS}
        np.savez(cache, key=key, Hs=Hs.reshape(nx, ny), Hr=Hr.reshape(nx, ny), tot_s=tot_s, tot_r=tot_r,
                 intf=intf, window=np.array(pes[e].interfaces[0], float) if is_istar else np.zeros(3), **arrays)
        print(f"  {run}: {e:03d} {n_used} paths, {len(ev)} events"
              + (f", {n_bad} blocks with a length mismatch skipped" if n_bad else ""), flush=True)


def load_run(run):
    out = {}
    for fn in sorted(Path(run, "joins").glob("[0-9][0-9][0-9].npz")):
        with np.load(fn) as d:
            out[int(fn.stem)] = {k: d[k] for k in d.files}
    if not out:
        raise SystemExit(f"{run}: no joins/ cache, run `extract` first")
    return out


def pooled_events(runs):
    """Events of all runs, per folder, concatenated (seeds pooled with their own weights)."""
    ev = {}
    for data in runs:
        for e, d in data.items():
            cur = ev.setdefault(e, {f: [] for f in FIELDS})
            for f in FIELDS:
                cur[f].append(d[f])
    return {e: {f: np.concatenate(v) for f, v in cur.items()} for e, cur in ev.items()}


def density(runs, which):
    """Sum over ensembles of the per-ensemble normalised density (one path per ensemble)."""
    H = 0
    for data in runs:
        for e, d in data.items():
            tot = d["tot_s" if which == "Hs" else "tot_r"]
            if tot > 0 and d[which].sum() > 0:
                H = H + d[which] / tot
    return np.asarray(H, float)


# ---------------------------------------------------------------------------
# selections of the joins
# ---------------------------------------------------------------------------
def var_of(d, m, var, down):
    v = {"y": d["y"], "px": d["vx"], "vy": d["vy"]}[var][m]
    return -v if (var == "px" and down) else v


def rep_join(ev, m, down, var):
    """REPPTIS join at lambda_m: (arriving exits, departing first crossings, departing by history)."""
    lower, upper = (m + 2, m + 1) if down else (m, m + 1)
    if lower not in ev or upper not in ev:
        return None
    a, d = ev[lower], ev[upper]
    ma = (a["etype"] == (EXIT_L if down else EXIT_R)) & (a["col"] == m) & (a["wr"] > 0)
    md = (d["etype"] == (FIRST_DN if down else FIRST_UP)) & (d["col"] == m) & (d["wr"] > 0)
    if not ma.any() or not md.any():
        return None
    # self-turn windows (LML up, RMR down) only know their history on one extension side
    self_turn = d["lmr"] == LMR_CODE["RMR" if down else "LML"]
    known = d["hist"] != H_UNKNOWN
    w = d["wr"].copy()
    wk = w[md & self_turn & known].sum()
    scale = w[md & self_turn].sum() / wk if wk > 0 else 0.0
    wh = np.where(md & self_turn, w * scale, w) * known
    hists = {h: (var_of(d, md & (d["hist"] == h), var, down), wh[md & (d["hist"] == h)])
             for h in (H_STRAIGHT, H_TURNED)}
    return ((var_of(a, ma, var, down), a["wr"][ma]), (var_of(d, md, var, down), d["wr"][md]), hists,
            float(w[md & ~known].sum() / w[md].sum()))


def staple_join(ev, m, down, var):
    """StapleTIS join at lambda_m: the turn at lambda_{m-1} (up) or lambda_{m+1} (down).

    Returns (window ensemble: turn is its window, ext ensemble ending in it, ext ensemble
    starting with it); y etc. at the closing lambda_m crossing of the turn."""
    k, kind = (m + 1, 1) if down else (m - 1, 0)
    win_e, ext_e = k + 1, m + 1
    if win_e not in ev or ext_e not in ev:
        return None
    res = []
    for e, side in ((win_e, None), (ext_e, 1), (ext_e, 0)):
        d = ev[e]
        msk = (d["etype"] == TURN) & (d["kind"] == kind) & (d["col"] == m) & (d["ws"] > 0)
        if side is not None:
            msk &= d["side"] == side
        res.append((var_of(d, msk, var, down), d["ws"][msk]))
    return res if all(len(r[0]) for r in res) else None


def whist(v, w, edges):
    h, _ = np.histogram(v, bins=edges, weights=w)
    s = h.sum()
    return h / s if s > 0 else h


def overlap(a, b, edges):
    return float(np.minimum(whist(*a, edges), whist(*b, edges)).sum())


def upper(v, w):
    return float(w[v > 0].sum() / w.sum()) if w.sum() > 0 else np.nan


def var_edges(var):
    return {"y": np.linspace(-1.8, 1.8, 61), "px": np.linspace(0.0, 2.5, 51), "vy": np.linspace(-2.0, 2.0, 51)}[var]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def potential_grid(run):
    import re
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from potentials.potential_staggered_channels import PotentialStaggeredChannels
    head = open(os.path.join(run, "logging.log")).read(200000)
    T = float(re.search(r"'temperature': ([0-9.]+)", head).group(1))
    Vp = float(re.search(r"'V_peak': ([0-9.eE+-]+)", head).group(1))
    Vd = float(re.search(r"'V_dip': ([0-9.eE+-]+)", head).group(1))
    pot = PotentialStaggeredChannels(V_peak=Vp, V_dip=Vd)
    xs, ys = np.linspace(-2.2, 2.2, 177), np.linspace(-2.0, 2.0, 101)
    V = np.array([[pot.potential((np.array([x_, y_]), np.zeros(2))) for x_ in xs] for y_ in ys]) / T
    return xs, ys, V


def decorate(ax, intf, pot=None):
    import matplotlib.pyplot as plt
    if pot is not None:
        xs, ys, V = pot
        ax.contour(xs, ys, np.clip(V, -5, 10), levels=np.arange(-4, 10, 1.0), colors="0.55", linewidths=0.4, zorder=1)
    for l in intf:
        ax.axvline(l, color="0.6", lw=0.5, ls="--", zorder=2)
    ax.add_patch(plt.Rectangle((-X_IN, -Y_IN), 2 * X_IN, 2 * Y_IN, fc="none", ec="k", lw=0.8, zorder=4))
    ax.set_xlim(XB[0], XB[-1]); ax.set_ylim(YB[0], YB[-1])


def cond_y(H):
    """P(y | x): every x column normalised to its maximum (empty columns stay empty)."""
    m = H.max(axis=1, keepdims=True)
    return np.where(H > 0, H / np.where(m > 0, m, 1), np.nan)


def upper_share_x(H):
    yc = 0.5 * (YB[1:] + YB[:-1])
    t = H.sum(axis=1)
    return np.where(t > 0, H[:, yc > 0].sum(axis=1) / np.where(t > 0, t, 1), np.nan)


def fig_density(ist, ret, intf, pot, out):
    import matplotlib.pyplot as plt
    panels = ([("RETIS paths", density(ret, "Hs"))] if ret else []) + [
        ("StapleTIS: staple paths", density(ist, "Hs")), ("REPPTIS: window segments", density(ist, "Hr"))]
    fig, axes = plt.subplots(2, len(panels), figsize=(4.6 * len(panels), 7.2), squeeze=False,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    xc = 0.5 * (XB[1:] + XB[:-1])
    for ax, (name, H) in zip(axes[0], panels):
        im = ax.pcolormesh(XB, YB, cond_y(H).T, cmap="magma_r", vmin=0, vmax=1, zorder=0)
        decorate(ax, intf, pot)
        ax.set_title(f"{name}: $P(y\\,|\\,x)$", fontsize=10); ax.set_xlabel("x")
    axes[0][0].set_ylabel("y")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="density / max at that x")
    gs = axes[1][0].get_gridspec()
    for ax in axes[1]:
        ax.remove()
    ax = fig.add_subplot(gs[1, :])
    colors = {"RETIS paths": "k", "StapleTIS: staple paths": "tab:blue", "REPPTIS: window segments": "tab:orange"}
    for name, H in panels:
        ax.plot(xc, upper_share_x(H), color=colors[name], lw=1.6, label=name)
    for l in intf:
        ax.axvline(l, color="0.8", lw=0.5, ls="--")
    ax.axvspan(-X_IN, X_IN, color="0.93", zorder=0)
    ax.set_xlim(XB[0], XB[-1]); ax.set_ylim(0, 1); ax.set_xlabel("x"); ax.set_ylabel("upper-channel share (y > 0)")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig); print(f"saved {out}")


def split_violin(ax, x0, left, right, edges, width, cl, cr):
    yc = 0.5 * (edges[1:] + edges[:-1])
    hl, hr = whist(*left, edges), whist(*right, edges)
    s = width / max(hl.max(), hr.max(), 1e-300)
    ax.fill_betweenx(yc, x0 - s * hl, x0, color=cl, alpha=0.75, lw=0, zorder=5)
    ax.fill_betweenx(yc, x0, x0 + s * hr, color=cr, alpha=0.75, lw=0, zorder=5)
    ax.plot([x0, x0], [edges[0], edges[-1]], color="k", lw=0.4, zorder=6)


def fig_where(ev, intf, pot, cols, down, out, max_points=40000, seed=1):
    """Split violins of y at every join (left: arriving side, right: departing side), and
    StapleTIS turn locations (self-turn windows, a weighted subsample of each class)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    rng = np.random.default_rng(seed)
    edges = var_edges("y")
    gap = np.min(np.diff(intf[1:-1])) if len(intf) > 3 else 0.2
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    ax = axes[0]
    for m in cols:
        r = rep_join(ev, m, down, "y")
        if r:
            split_violin(ax, intf[m], r[0], r[1], edges, 0.45 * gap, "tab:orange", "tab:blue")
    handles = [Patch(color="tab:orange", label=r"left: $[(m{-}1)^\pm]$ LMR leaves over $\lambda_m$" if not down
                     else r"left: $[(m{+}1)^\pm]$ RML leaves over $\lambda_m$"),
               Patch(color="tab:blue", label=r"right: $[m^\pm]$ LM$^*$, first $\lambda_m$ crossing" if not down
                     else r"right: $[m^\pm]$ RM$^*$, first $\lambda_m$ crossing")]
    ax.legend(handles=handles, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    ax.set_title(r"REPPTIS joins: $y$ at the interface" + (" (down)" if down else " (up)"), fontsize=10)
    ax = axes[1]
    for kind, col in ((0, "tab:green"), (1, "tab:purple")):
        xs, ys, ws = [], [], []
        for e, d in ev.items():  # turn at lambda_{e-1} as the window of [(e-1)*]
            mk = (d["etype"] == TURN) & (d["ws"] > 0) & (d["kind"] == kind) & (d["col"] == (e if kind == 0 else e - 2))
            if mk.any():
                xs.append(d["ext_x"][mk]); ys.append(d["ext_y"][mk]); ws.append(d["ws"][mk] / d["ws"][mk].sum())
        if xs:
            xs, ys, ws = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
            pick = rng.choice(len(xs), size=min(max_points, len(xs)), p=ws / ws.sum())
            ax.scatter(xs[pick], ys[pick], s=2, color=col, alpha=0.15, lw=0, zorder=3, rasterized=True)
    for m in cols:
        r = staple_join(ev, m, down, "y")
        if r:
            split_violin(ax, intf[m], r[1], r[0], edges, 0.45 * gap, "tab:red", "tab:cyan")
    turn = r"$\lambda_{m+1}$" if down else r"$\lambda_{m-1}$"
    handles = [Line2D([], [], ls="", marker="o", ms=3, color="tab:green", label="minimum turns ($[k^*]$ RMR windows)"),
               Line2D([], [], ls="", marker="o", ms=3, color="tab:purple", label="maximum turns ($[k^*]$ LML windows)"),
               Patch(color="tab:red", label=rf"left: $[m^*]$ paths ending in the turn at {turn}"),
               Patch(color="tab:cyan", label=rf"right: the same turn as the window of $[({'m+1' if down else 'm-1'})^*]$")]
    ax.legend(handles=handles, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    ax.set_title(r"StapleTIS joins: $y$ at the closing $\lambda_m$ crossing of a turn, and where turns are", fontsize=10)
    for ax in axes:
        decorate(ax, intf, pot); ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig); print(f"saved {out}")


def fig_joins(ev, intf, cols, down, var, out):
    import matplotlib.pyplot as plt
    edges = var_edges(var)
    label = {"y": "y", "px": r"$|p_x|$" if down else r"$p_x$", "vy": r"$p_y$"}[var]
    fig, axes = plt.subplots(2, len(cols), figsize=(2.3 * len(cols), 5.0), sharex=True, squeeze=False,
                             gridspec_kw={"hspace": 0.45})
    fill = dict(bins=edges, density=True, alpha=0.35)
    step = dict(bins=edges, density=True, histtype="step", lw=1.2)
    for j, m in enumerate(cols):
        ax = axes[0][j]
        r = rep_join(ev, m, down, var)
        if r:
            arr, dep, hists, unk = r
            whisto(ax, arr, color="tab:orange", label="arrives: LMR exit" if not down else "arrives: RML exit", **fill)
            whisto(ax, dep, color="tab:blue", label="continues: LM$^*$" if not down else "continues: RM$^*$", **fill)
            for h, ls, name in ((H_STRAIGHT, "-", "straight"), (H_TURNED, "--", "after a turn")):
                whisto(ax, hists[h], color="0.15", ls=ls, label=f"continues, {name}", **step)
            ax.text(0.03, 0.96, f"OVL {overlap(arr, dep, edges):.2f}", transform=ax.transAxes, va="top", fontsize=7)
        ax.set_title(rf"$\lambda_{{{m}}}={intf[m]:+.2f}$", fontsize=9)
        ax = axes[1][j]
        s = staple_join(ev, m, down, var)
        if s:
            whisto(ax, s[0], color="tab:cyan", label="turn = window", **fill)
            whisto(ax, s[1], color="tab:red", label="ends in turn", **step)
            whisto(ax, s[2], color="tab:red", ls="--", label="starts with turn", **step)
            ax.text(0.03, 0.96, f"OVL {overlap(s[0], s[1], edges):.2f}", transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xlabel(label)
    axes[0][0].set_ylabel("REPPTIS\ndensity"); axes[1][0].set_ylabel("StapleTIS\ndensity")
    for row, y0 in ((0, 0.5), (1, 0.0)):  # one legend per row, from the first panel that has every entry
        hl = max((a.get_legend_handles_labels() for a in axes[row]), key=lambda t: len(t[1]))
        fig.legend(*hl, fontsize=7, ncol=len(hl[1]), frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, y0 - 0.01), bbox_transform=fig.transFigure)
    if var == "y":
        for ax in axes.ravel():
            ax.axvspan(-Y_IN, Y_IN, color="0.92", zorder=0)
    fig.suptitle(("Downward" if down else "Upward") + f" joins: {label} at the join, the two sides the MSM glues together",
                 fontsize=10)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig); print(f"saved {out}")


def whisto(ax, vw, **kw):
    """Weighted histogram of a (values, weights) pair."""
    if len(vw[0]) and vw[1].sum() > 0:
        ax.hist(vw[0], weights=vw[1], **kw)


def fig_summary(ev, intf, cols, down, out):
    import matplotlib.pyplot as plt
    rows = []
    for m in cols:
        r, s = rep_join(ev, m, down, "y"), staple_join(ev, m, down, "y")
        rp, sp = rep_join(ev, m, down, "px"), staple_join(ev, m, down, "px")
        rows.append(dict(
            m=m, lam=intf[m],
            rep_arr=upper(*r[0]) if r else np.nan, rep_dep=upper(*r[1]) if r else np.nan,
            rep_str=upper(*r[2][H_STRAIGHT]) if r and r[2][H_STRAIGHT][1].sum() > 0 else np.nan,
            rep_turn=upper(*r[2][H_TURNED]) if r and r[2][H_TURNED][1].sum() > 0 else np.nan,
            rep_unknown=r[3] if r else np.nan,
            sta_win=upper(*s[0]) if s else np.nan, sta_end=upper(*s[1]) if s else np.nan, sta_start=upper(*s[2]) if s else np.nan,
            ovl_rep_y=overlap(r[0], r[1], var_edges("y")) if r else np.nan,
            ovl_sta_y=overlap(s[0], s[1], var_edges("y")) if s else np.nan,
            ovl_rep_px=overlap(rp[0], rp[1], var_edges("px")) if rp else np.nan,
            ovl_sta_px=overlap(sp[0], sp[1], var_edges("px")) if sp else np.nan))
    lam = np.array([r_["lam"] for r_ in rows])
    g = lambda k: np.array([r_[k] for r_ in rows])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    ax.plot(lam, g("rep_arr"), "o-", color="tab:orange", label="REPPTIS, arriving (LMR exit)" if not down else "REPPTIS, arriving (RML exit)")
    ax.plot(lam, g("rep_dep"), "s-", color="tab:blue", label="REPPTIS, continuing (first crossing)")
    ax.plot(lam, g("rep_turn"), "s:", color="tab:blue", mfc="none", label="  of which after a turn")
    ax.plot(lam, g("sta_win"), "D-", color="tab:cyan", label="StapleTIS, turn = window")
    ax.plot(lam, g("sta_end"), "v--", color="tab:red", label="StapleTIS, ends in turn")
    ax.plot(lam, g("sta_start"), "^:", color="tab:red", label="StapleTIS, starts with turn")
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel(r"join interface $\lambda_m$"); ax.set_ylabel("upper-channel share at the join")
    ax.axvspan(-X_IN, X_IN, color="0.93", zorder=0); ax.legend(fontsize=7)
    ax.set_title("Channel on either side of each join", fontsize=10)
    ax2.plot(lam, g("ovl_rep_y"), "o-", color="tab:orange", label="REPPTIS, y")
    ax2.plot(lam, g("ovl_rep_px"), "o--", color="tab:orange", mfc="none", label=r"REPPTIS, $p_x$")
    ax2.plot(lam, g("ovl_sta_y"), "D-", color="tab:blue", label="StapleTIS, y")
    ax2.plot(lam, g("ovl_sta_px"), "D--", color="tab:blue", mfc="none", label=r"StapleTIS, $p_x$")
    ax2.set_ylim(0, 1.02); ax2.set_xlabel(r"join interface $\lambda_m$"); ax2.set_ylabel("overlap of the two joined distributions")
    ax2.axvspan(-X_IN, X_IN, color="0.93", zorder=0); ax2.legend(fontsize=7, loc="lower left")
    ax2.set_title(r"Overlap $\sum_b \min(p_b, q_b)$ (1 = identical)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig); print(f"saved {out}")
    print(f"\n{'m':>3} {'lam':>6} | REPPTIS upper share: arr  dep  straight turned  unknown-hist | "
          f"StapleTIS: window end start | OVL y rep/sta  px rep/sta")
    for r_ in rows:
        print(f"{r_['m']:>3} {r_['lam']:+6.2f} | {r_['rep_arr']:20.2f} {r_['rep_dep']:4.2f} {r_['rep_str']:8.2f} "
              f"{r_['rep_turn']:7.2f} {r_['rep_unknown']:8.2f}     | {r_['sta_win']:16.2f} {r_['sta_end']:4.2f} "
              f"{r_['sta_start']:5.2f} | {r_['ovl_rep_y']:5.2f}/{r_['ovl_sta_y']:4.2f}  {r_['ovl_rep_px']:4.2f}/{r_['ovl_sta_px']:4.2f}")
    return rows


def fig_all(istar_runs, retis_runs, prefix, variables, down, lam_range):
    ist = [load_run(r) for r in istar_runs]
    ret = [load_run(r) for r in retis_runs]
    intf = np.array(ist[0][min(ist[0])]["intf"], float)
    ev = pooled_events(ist)
    n = len(intf)
    cols = [m for m in (range(1, n - 3) if down else range(2, n - 1))
            if lam_range[0] <= intf[m] <= lam_range[1]]
    pot = potential_grid(istar_runs[0])
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    tag = "_down" if down else ""
    fig_density(ist, ret, intf, pot, f"{prefix}_density.png")
    fig_where(ev, intf, pot, cols, down, f"{prefix}_where{tag}.png")
    for var in variables:
        fig_joins(ev, intf, cols, down, var, f"{prefix}_joins_{var}{tag}.png")
    fig_summary(ev, intf, cols, down, f"{prefix}_summary{tag}.png")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract"); p.add_argument("runs", nargs="+")
    p.add_argument("--skip", type=int, default=0, help="Drop cycles below this number (startup).")
    p.add_argument("--chunksize", type=int, default=4_000_000, help="order.txt rows per read.")
    p.add_argument("--force", action="store_true", help="Recompute existing caches.")
    p = sub.add_parser("fig")
    p.add_argument("--istar", nargs="+", required=True, help="i* runs (pooled).")
    p.add_argument("--retis", nargs="*", default=[], help="RETIS runs for the density reference.")
    p.add_argument("--var", nargs="+", default=["y", "px"], choices=["y", "px", "vy"])
    p.add_argument("--down", action="store_true", help="Downward joins instead of upward ones.")
    p.add_argument("--lam-range", nargs=2, type=float, default=[-9, 9], help="Only joins with lambda_m in this range.")
    p.add_argument("-o", default="stag_joins", help="Output prefix.")
    a = ap.parse_args()
    if a.cmd == "extract":
        for r in a.runs:
            extract_run(r, a.skip, a.chunksize, a.force)
    elif a.cmd == "fig":
        fig_all(a.istar, a.retis, a.o, a.var, a.down, a.lam_range)


if __name__ == "__main__":
    main()
