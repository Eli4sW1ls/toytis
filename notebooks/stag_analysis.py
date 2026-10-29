#!/usr/bin/env python3
"""Analysis of the staggered two-channel runs (staggered_channels_run.py).

Run it next to the data (workstation), in an environment with tistools.

Subcommands
-----------
labels RUN...        Channel label of every path in order.txt (upper/lower/neither),
                     written to RUN/channels/NNN.chan. Needed by diag and switch.
diag RUN...          i*: per-ensemble channel mix, cross-ensemble staple-class table,
                     StapleTIS (all staples / one channel only) and REPPTIS P_AB.
                     RETIS: channel mix of accepted paths per third of the run.
                     Results are also saved to RUN/stag_analysis.json.
switch RUN...        i*: sample reactive paths from the tistools StapleTIS MSM and count
                     where consecutive staples switch channel at a turn (tunnelling).
fig-main             Crossing-probability profiles of several i* seeds vs the RETIS
                     reference (single-channel value x 2).
fig-friction         StapleTIS / REPPTIS vs reference and the switching fraction, per
                     friction (one group of i* seeds per friction).
fig-tunnel RUN       Where the StapleTIS MSM switches channel, with example paths.

Typical use (from the toytis root):
    python notebooks/stag_analysis.py labels simulations/stagd3g01_istar_s5
    python notebooks/stag_analysis.py diag   simulations/stagd3g01_istar_s5
    python notebooks/stag_analysis.py switch simulations/stagd3g01_istar_s5
    python notebooks/stag_analysis.py fig-main --istar simulations/stagd3g01_istar_s{5,6,7,8} \\
        --retis simulations/stagd3g01blockU_retis_s1 --factor 2 -o fig_main.png
    python notebooks/stag_analysis.py fig-friction \\
        --group "gamma=1"    "simulations/stag_istar_s1,simulations/stag_istar_s2,simulations/stag_istar_s3" 2.05e-4 \\
        --group "gamma=0.25" "simulations/stag_istar_s4" 2.0e-4 \\
        --group "gamma=0.1"  "simulations/stagd3g01_istar_s5,simulations/stagd3g01_istar_s6" 1.65e-4 -o fig_friction.png

RETIS references need `pyretisanalyse -i retis.rst` to have been run (report/ folder).
The full two-channel P_AB is twice the single-channel RETIS value (mirror symmetry
plus time reversal); --factor 2 applies that.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import gzip
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path = [p for p in sys.path if "inftools" not in p]

# Channel geometry of PotentialStaggeredChannels: the obstacle spans -1 < x < 1, |y| < 0.6.
X_IN, Y_IN = 1.0, 0.6
HDR = re.compile(r"# Cycle: (\d+), status: (\S+), move: (\S+), path length: (\d+).*staridx: \((-?\d+), (-?\d+)\)")
HDR_RETIS = re.compile(r"# Cycle: (\d+), status: (\S+), move: (\S+),")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ensembles(run):
    return sorted(glob.glob(os.path.join(run, "0[0-9][0-9]")))


def interfaces(run):
    for line in open(os.path.join(run, "logging.log")):
        if "interfaces =" in line:
            return [float(v) for v in line.split("[")[1].split("]")[0].split(",")]
    raise ValueError(f"no interfaces in {run}/logging.log")


def simtype(run):
    with open(os.path.join(run, "logging.log")) as f:
        head = f.read(200000)
    m = re.search(r"'simtype': '([^']+)'", head)
    return m.group(1) if m else "unknown"


def save_json(run, key, value):
    fn = os.path.join(run, "stag_analysis.json")
    data = json.load(open(fn)) if os.path.exists(fn) else {}
    data[key] = value
    txt = json.dumps(data, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(fn + ".tmp", "w") as f: f.write(txt)
    os.replace(fn + ".tmp", fn)


def load_json(run):
    fn = os.path.join(run, "stag_analysis.json")
    return json.load(open(fn)) if os.path.exists(fn) else {}


def tistools():
    import tistools as tt
    return tt


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def extract_labels(run):
    """One line per path block: cycle status move pathlen sa sb staple_up staple_lo path_up path_lo."""
    out = os.path.join(run, "channels"); os.makedirs(out, exist_ok=True)
    for ens in ensembles(run):
        e = os.path.basename(ens)
        rows, cur = [], None
        with open(os.path.join(ens, "order.txt"), buffering=8 << 20) as f:
            for line in f:
                if line[0] == "#":
                    if line.startswith("# Cycle"):
                        if cur is not None: rows.append(cur)
                        m = HDR.match(line)
                        cur = [m.group(1), m.group(2).rstrip(","), m.group(3).rstrip(","), m.group(4),
                               int(m.group(5)), int(m.group(6)), 0, 0, 0, 0] if m else None
                    continue
                if cur is None: continue
                p = line.split()
                if len(p) < 3: continue
                x = float(p[1])
                if -X_IN < x < X_IN:
                    y = float(p[2]); t = int(p[0])
                    if y > Y_IN:
                        cur[8] = 1
                        if cur[4] <= t <= cur[5]: cur[6] = 1
                    elif y < -Y_IN:
                        cur[9] = 1
                        if cur[4] <= t <= cur[5]: cur[7] = 1
        if cur is not None: rows.append(cur)
        with open(os.path.join(out, f"{e}.chan"), "w") as fo:
            for r in rows: fo.write(" ".join(str(v) for v in r) + "\n")
        print(f"  {run}: {e} {len(rows)} paths", flush=True)


def read_labels(run, n_rows, which="staple"):
    """Per ensemble array of 'U', 'L', 'N' (neither) or 'X' (no label, e.g. run still going)."""
    lab = {}
    for i, ens in enumerate(ensembles(run)):
        fn = os.path.join(run, "channels", os.path.basename(ens) + ".chan")
        a = np.loadtxt(fn, dtype=str, ndmin=2)
        cu, cl = (6, 7) if which == "staple" else (8, 9)
        m = min(len(a), n_rows[i])
        lab[i] = np.full(n_rows[i], "X", dtype="<U1")
        up, lo = a[:m, cu].astype(int) == 1, a[:m, cl].astype(int) == 1
        lab[i][:m] = np.where(up & lo, "B", np.where(up, "U", np.where(lo, "L", "N")))
    return lab


# ---------------------------------------------------------------------------
# tistools StapleTIS / REPPTIS
# ---------------------------------------------------------------------------
def load_pes(run):
    tt = tistools(); intf = interfaces(run)
    LMR, LMRs = tt.get_LMR_interfaces(intf, None); pes = []
    for i, ens in enumerate(ensembles(run)):
        pe = tt.read_pathensemble(os.path.join(ens, "pathensemble.txt")); pe.set_name(ens)
        pe.set_interfaces([LMR[i], LMRs[i]])
        if i == 0: pe.set_zero_minus_one(False); pe.set_in_zero_minus(True)
        if i == 1: pe.set_in_zero_plus(True)
        w, _ = tt.get_weights(pe.flags, tt.ACCFLAGS, tt.REJFLAGS, verbose=False); pe.set_weights(w)
        pes.append(pe)
    return pes, intf


def staple_weights(pes, n):
    tt = tistools()
    with quiet():
        return {i: np.asarray(tt.get_weights_staple(i, pe.flags, pe.generation, pe.lmrs, n, tt.ACCFLAGS, tt.REJFLAGS,
                                                    ha_factors=None, verbose=False), float) for i, pe in enumerate(pes)}


def istar_layout(tt, n):
    """State layout of the tistools StapleTIS MSM for n interfaces, for either tistools version.

    Returns (NS, A, start, B, state_turn): the number of states, the absorbing "in A" and
    "reached B" states, the "just left A" start state, and state_turn(s) -> (interface index,
    "min"|"max") of the last turn of state s (None, None for A/B; (0, "min") for the start).
    tistools >= 64f0940 (istar_n_states): 0 = in A, 1 = left A, 1+i = turned at a minimum at
    lambda_i, n-1+i = turned at a maximum at lambda_i, 2n-2 = B. Older versions used 2n states:
    1 = back in A, 2 = left A, 2+i = minimum at lambda_i, n+i = maximum at lambda_i, 2n-1 = B.
    """
    if hasattr(tt, "istar_n_states"):
        NS = tt.istar_n_states(n)
        def state_turn(s):
            if s == 1: return 0, "min"
            if 2 <= s <= n - 1: return s - 1, "min"
            if n <= s <= 2 * n - 3: return s - n + 1, "max"
            return None, None
        return NS, 0, 1, NS - 1, state_turn
    NS = 2 * n
    def state_turn(s):
        if s == 2: return 0, "min"
        if 2 < s < 2 + n: return s - 2, "min"
        if n <= s < 2 * n: return s - n, "max"
        return None, None
    return NS, 1, 2, NS - 1, state_turn


def staple_profile(pes, intf, weights=None):
    """Same calls as tistools-retis-istar-compare."""
    tt = tistools(); out = np.ones(len(intf))
    with quiet():
        for lint in range(2, len(intf) + 1):
            X = tt.compute_weight_matrices(pes[:lint], intf[:lint], len(intf), tr=False, correct_ha=False, norm=False,
                                           weights=None if weights is None else {i: weights[i] for i in range(lint)})
            P, _ = tt.get_transition_probs_weights(X)
            M = tt.construct_M_istar(P, istar_layout(tt, lint)[0], lint)
            _, _, y1, _ = tt.global_pcross_msm_star(M)
            out[lint - 1] = y1[0][0]
    return out


def repptis_profile(pes):
    tt = tistools()
    with quiet():
        ploc = [None] + [tt.get_local_probs(pe, w=pe.weights, tr=False) for pe in pes[1:]]
        _, _, rep = tt.get_global_probs_from_dict(ploc)
    return np.array(rep, float)


# ---------------------------------------------------------------------------
# diag
# ---------------------------------------------------------------------------
def diag_istar(run):
    pes, intf = load_pes(run); n = len(intf)
    W = staple_weights(pes, n)
    st, rep = staple_profile(pes, intf), repptis_profile(pes)
    res = {"interfaces": intf, "staple": st.tolist(), "repptis": rep.tolist(),
           "cycles": int(max(len(w) for w in W.values()))}
    print(f"\n== {run} (i*, {res['cycles']} cycles)")
    print(f"StapleTIS P_AB {st[-1]:.3e}   REPPTIS P_AB {rep[-1]:.3e}")
    if not os.path.isdir(os.path.join(run, "channels")):
        print("(no channel labels: run `labels` first for the channel diagnostics)")
        save_json(run, "diag", res); return res
    lab = read_labels(run, [len(W[i]) for i in range(n)])
    share = []
    for i in range(n):
        t = W[i][lab[i] != "X"].sum()
        share.append([float(W[i][lab[i] == "U"].sum() / t) if t else 0.0, float(W[i][lab[i] == "L"].sum() / t) if t else 0.0])
    print("staple weight share upper/lower per ensemble: " + "  ".join(f"e{i}:{u:.2f}/{l:.2f}" for i, (u, l) in enumerate(share)))
    print("staples from A (forward) reaching >= lambda: upper share | weight per ensemble "
          "(equal in every ensemble that contains the class, in equilibrium)")
    table = {}
    for lamk in [v for v in intf if 0.2 < v < 0.7]:
        row = {}
        for i in range(2, n):
            pe = pes[i]; m = (pe.lambmins <= intf[0]) & (pe.dirs == 1) & (pe.lambmaxs >= lamk) & (W[i] > 0) & (lab[i] != "X")
            w = W[i][m]; t = w.sum()
            row[i] = [float(w[lab[i][m] == "U"].sum() / t) if t > 0 else None, float(t)]
        table[str(lamk)] = row
        print(f"  >= {lamk:+.2f}: " + "  ".join(f"e{i}:{v[0]:.2f}|{v[1]:.0f}" if v[0] is not None else f"e{i}:-" for i, v in row.items()))
    masked = lambda keep: {i: W[i] * np.isin(lab[i], keep) for i in range(n)}
    lo, up = staple_profile(pes, intf, masked(["L", "N"])), staple_profile(pes, intf, masked(["U", "N"]))
    print(f"StapleTIS one channel only: lower {lo[-1]:.3e}   upper {up[-1]:.3e}   (equal by symmetry when converged)")
    res.update({"share": share, "class_table": table, "staple_lower": lo.tolist(), "staple_upper": up.tolist()})
    save_json(run, "diag", res); return res


def diag_retis(run):
    intf = interfaces(run)
    print(f"\n== {run} (RETIS): upper share of accepted paths per third [n, channel changes]")
    res = {}
    for i, ens in enumerate(ensembles(run)[1:], start=1):
        ch, cur = [], None
        with open(os.path.join(ens, "order.txt"), buffering=8 << 20) as f:
            for line in f:
                if line.startswith("# Cycle"):
                    if cur and cur[1] and cur[2] != "ld" and (cur[3] or cur[4]): ch.append((cur[0], "U" if cur[3] else "L"))
                    m = HDR_RETIS.match(line)
                    cur = [int(m.group(1)), m.group(2).rstrip(",") == "ACC", m.group(3).rstrip(","), 0, 0] if m else None
                elif line[0] != "#" and cur is not None:
                    p = line.split()
                    if len(p) < 3: continue
                    x = float(p[1])
                    if -X_IN < x < X_IN:
                        y = float(p[2])
                        if y > Y_IN: cur[3] = 1
                        elif y < -Y_IN: cur[4] = 1
        if not ch: continue
        cmax = ch[-1][0]; parts = []
        for k in range(3):
            part = [c for cyc, c in ch if k * cmax / 3 <= cyc < (k + 1) * cmax / 3 + (k == 2)]
            parts.append([part.count("U") / max(len(part), 1), len(part), sum(a != b for a, b in zip(part, part[1:]))])
        res[i] = parts
        print(f"  [{i-1}+] ({intf[i-1]:+.2f}) | " + " | ".join(f"{u:.2f} [{n_}, {s}]" for u, n_, s in parts), flush=True)
    save_json(run, "retis_mix", res); return res


# ---------------------------------------------------------------------------
# switch: reactive paths of the StapleTIS MSM and their channel switches
# ---------------------------------------------------------------------------
def switch(run, nrun=200000, seed=3):
    tt = tistools(); rng = np.random.default_rng(seed)
    pes, intf = load_pes(run); n = len(intf)
    NS, sA, s0, sB, state_turn = istar_layout(tt, n)
    W = staple_weights(pes, n); lab = read_labels(run, [len(W[i]) for i in range(n)])
    masked = lambda keep: {i: W[i] * np.isin(lab[i], keep) for i in range(n)}
    with quiet():
        X = tt.compute_weight_matrices(pes, intf, n, tr=False, correct_ha=False, norm=False)
        P, _ = tt.get_transition_probs_weights(X)
        Xc = {c: tt.compute_weight_matrices(pes, intf, n, tr=False, correct_ha=False, norm=False, weights=masked([c]))
              for c in "ULN"}
    M = np.nan_to_num(tt.construct_M_istar(P, NS, n))
    T = {c: sum(Xc[c][i] for i in range(n)) for c in Xc}

    def comp(j, k):
        v = np.array([T[c][j, k] for c in "ULN"]); s = v.sum()
        return v / s if s > 0 else np.array([0, 0, 1.0])

    edge = {}
    for a in range(NS):
        ta, _ = state_turn(a)
        if ta is None: continue
        for b in np.nonzero(M[a] > 0)[0]:
            tb = 0 if b == sA else (n - 1 if b == sB else state_turn(b)[0])
            edge[(a, b)] = comp(ta, tb)
    trans = [s for s in range(NS) if s not in (sA, sB) and (s == s0 or state_turn(s)[0] is not None)]
    q = np.zeros(NS); q[sB] = 1.0
    q[trans] = np.linalg.solve(np.eye(len(trans)) - M[np.ix_(trans, trans)], M[np.ix_(trans, [sB])].ravel())

    def inside(s):
        t, kind = state_turn(s)
        if t is None: return False
        lo_, hi_ = (intf[t - 1], intf[t]) if kind == "min" else (intf[t], intf[t + 1] if t + 1 < n else 9)
        return lo_ >= -X_IN and hi_ <= X_IN

    sw, nsw, nany, examples = {}, [], [], []
    for _ in range(nrun):
        s, last, k, kany, path = s0, None, 0, 0, []
        while s not in (sA, sB):
            nxt = np.nonzero(M[s] > 0)[0]; pr = M[s, nxt] * q[nxt]; pr /= pr.sum()
            b = nxt[rng.choice(len(nxt), p=pr)]; c = "ULN"[rng.choice(3, p=edge[(s, b)])]
            if c != "N":
                kany += last is not None and c != last
                if last is not None and c != last and inside(s):
                    t, kind = state_turn(s); key = f"{kind} [{(intf[t-1] if kind == 'min' else intf[t]):+.2f},{(intf[t] if kind == 'min' else intf[t+1]):+.2f}] {last}->{c}"
                    sw[key] = sw.get(key, 0) + 1; k += 1
                last = c
            path.append([list(state_turn(s)), list(state_turn(b)) if b not in (sA, sB) else ("A" if b == sA else "B"), c]); s = b
        nsw.append(k); nany.append(kany)
        if k and len(examples) < 50: examples.append(path)
    nsw = np.array(nsw)
    res = {"frac_switch": float(np.mean(nsw > 0)), "frac_switch_any": float(np.mean(np.array(nany) > 0)), "mean_switch": float(nsw.mean()), "nrun": nrun,
           "switch_at": {k: v / nrun for k, v in sorted(sw.items(), key=lambda kv: -kv[1])},
           "committor": float(q[s0]), "examples": examples}
    print(f"\n== {run}: {res['frac_switch']:.1%} of reactive StapleTIS-MSM paths switch channel at a turn inside the obstacle "
          f"({res['frac_switch_any']:.1%} switch anywhere, incl. the open region)")
    for k_, v in list(res["switch_at"].items())[:8]: print(f"   {k_}: {v:.3f} per reactive path")
    save_json(run, "switch", res); return res


# ---------------------------------------------------------------------------
# RETIS reference
# ---------------------------------------------------------------------------
def retis_reference(run):
    """(lambda, P) matched profile and (P_AB, rel. error) from pyretisanalyse output."""
    blocks, lam, pc = [], [], []
    for line in gzip.open(os.path.join(run, "report", "matched-probability.txt.gz"), "rt"):
        line = line.strip()
        if not line: continue
        if line.startswith("#"):
            if lam: blocks.append((np.array(lam), np.array(pc)))
            lam, pc = [], []; continue
        a, b = line.split()[:2]; lam.append(float(a)); pc.append(float(b))
    blocks.append((np.array(lam), np.array(pc)))
    sel = [(b[0] < (blocks[k + 1][0][0] if k + 1 < len(blocks) else 9e9)) for k, b in enumerate(blocks)]
    rl = np.concatenate([b[0][m] for b, m in zip(blocks, sel)]); rp = np.concatenate([b[1][m] for b, m in zip(blocks, sel)])
    err = None
    for fn in (os.path.join(run, "report.txt"), os.path.join(run, "report", "retis_report.rst")):
        if os.path.exists(fn):
            txt = re.sub(r"\x1b\[[0-9;]*m", "", open(fn).read())
            m = re.search(r"RETIS Crossing probability: *([0-9.eE+-]+)\s*\(Relative error: *([0-9.]+) *%\)", txt)
            if m: err = float(m.group(2)) / 100; break
    return rl, rp, float(rp[-1]), err


def istar_values(run):
    d = load_json(run).get("diag")
    if d is None:
        pes, intf = load_pes(run)
        d = {"interfaces": intf, "staple": staple_profile(pes, intf).tolist(), "repptis": repptis_profile(pes).tolist()}
    return d


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def combine_references(refs, factor, names=None):
    """Combine the RETIS estimates of P_AB into one reference.

    Each run i gives P_i with the pyretis block-error relative error r_i. P_AB is a
    product of local crossing probabilities, so the runs are combined in log space:
    ln P_ref = sum(w_i ln P_i) / sum(w_i), w_i = 1 / r_i^2, with error 1 / sqrt(sum w_i).
    If the runs scatter more than their block errors allow (chi2 = sum w_i (ln P_i -
    ln P_ref)^2 > dof), that error is inflated by the Birge ratio sqrt(chi2 / dof).
    ``factor`` multiplies the value (2 for one-channel runs); it is exact by symmetry
    and carries no error. Returns (P_ref, relative error, per-run [name, factor*P_i, r_i]).
    """
    names = names or [str(k) for k in range(len(refs))]
    v = np.array([r[2] for r in refs]); errs = [r[3] for r in refs]
    runs = [[n, factor * a, e] for n, a, e in zip(names, v, errs)]
    if any(e is None for e in errs):
        return factor * np.exp(np.log(v).mean()), None, runs
    l, r = np.log(v), np.array(errs); w = 1 / r ** 2
    m = (w * l).sum() / w.sum(); sm = w.sum() ** -0.5; birge = 1.0; chi2 = 0.0
    if len(v) > 1:
        chi2 = float((w * (l - m) ** 2).sum()); birge = max(1.0, np.sqrt(chi2 / (len(v) - 1)))
    print("RETIS references (x{:g}): ".format(factor) + ", ".join(f"{n} {a:.3e} +- {e:.1%}" for n, a, e in runs))
    print(f"  weighted (log) mean {factor * np.exp(m):.3e}, block-error-only error {sm:.1%}, "
          f"chi2 {chi2:.2f} / dof {len(v) - 1}, Birge {birge:.2f} -> +- {sm * birge:.1%}")
    return factor * np.exp(m), sm * birge, runs


def geo_ratio(r, rel):
    """Geometric mean of per-seed ratios; error = seed standard error (log) combined with the reference error."""
    l = np.log(r); se = l.std(ddof=1) / np.sqrt(len(l)) if len(l) > 1 else 0.0
    return np.exp(l.mean()), float(np.hypot(se, rel or 0.0))


def fig_main(istar_runs, retis_runs, factor, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    refs = [retis_reference(r) for r in retis_runs]
    P_ref, rel, ref_runs = combine_references(refs, factor, [os.path.basename(os.path.normpath(r)) for r in retis_runs])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2.4, 1]})
    for rl, rp, _, _ in refs: ax.semilogy(rl, rp, color="0.55", lw=1.2)
    ax.plot([], [], color="0.55", lw=1.2, label="RETIS, one channel")
    sts, reps = [], []
    for run in istar_runs:
        d = istar_values(run); intf = np.array(d["interfaces"])
        sts.append(d["staple"]); reps.append(d["repptis"])
        ax.semilogy(intf, d["staple"], "-", color="tab:blue", alpha=0.3, lw=1)
        ax.semilogy(intf, d["repptis"], "-", color="tab:orange", alpha=0.3, lw=1)
    sts, reps = np.array(sts), np.array(reps)
    gm = lambda a: np.exp(np.log(a).mean(0))
    ax.semilogy(intf, gm(sts), "o-", color="tab:blue", lw=2.2, label=f"StapleTIS ({len(istar_runs)} seeds)")
    ax.semilogy(intf, gm(reps), "s-", color="tab:orange", lw=2.2, label=f"REPPTIS ({len(istar_runs)} seeds)")
    ax.axhline(P_ref, color="k", lw=2, label=f"RETIS reference ({factor:g} x one channel)")
    if rel: ax.axhspan(P_ref * np.exp(-rel), P_ref * np.exp(rel), color="k", alpha=0.2, lw=0)
    ax.set_ylabel(r"$P_A(\lambda\,|\,\lambda_A)$"); ax.legend(fontsize=9, loc="lower left")
    for k, (lab_, a, col) in enumerate((("StapleTIS", sts[:, -1], "tab:blue"), ("REPPTIS", reps[:, -1], "tab:orange"))):
        r = a / P_ref
        g, tot = geo_ratio(r, rel)
        ax2.scatter(np.full(len(r), k), r, color=col, alpha=0.6)
        ax2.errorbar(k, g, yerr=[[g - g * np.exp(-tot)], [g * np.exp(tot) - g]], fmt="D", color=col, ms=9, capsize=4)
        ax2.text(k + 0.15, g, f"{g:.2f}x\n$\\pm${tot:.0%}", va="center")
        print(f"{lab_}: geometric mean {g:.3f}x +- {tot:.1%} (seed SE + reference); per seed " + " ".join(f"{v:.2f}" for v in r))
    # the RETIS runs themselves, each with its own block error, and the combined reference band
    xr = -1 + np.linspace(-0.25, 0.25, len(ref_runs)) if len(ref_runs) > 1 else np.array([-1.0])
    for x_, (n, a, e) in zip(xr, ref_runs):
        q = a / P_ref
        ax2.errorbar(x_, q, yerr=None if e is None else [[q - q * np.exp(-e)], [q * np.exp(e) - q]],
                     fmt="o", color="0.3", ms=5, capsize=3)
    if rel:
        ax2.axhspan(np.exp(-rel), np.exp(rel), color="k", alpha=0.2, lw=0)
        ax2.text(-1, np.exp(rel) * 1.15, f"$\\pm${rel:.0%}", ha="center", fontsize=9)
    ax2.axhline(1, color="k", lw=1); ax2.set_yscale("log"); ax2.set_xlim(-1.6, 1.8)
    ax2.set_xticks([-1, 0, 1]); ax2.set_xticklabels([f"RETIS\n(x{factor:g})", "StapleTIS", "REPPTIS"])
    ax2.set_ylabel(r"$P_{AB}$ / reference")
    ax.set_xlabel(r"$\lambda$"); fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"saved {out}; reference P_AB = {P_ref:.3e}" + (f" +- {rel:.1%}" if rel else ""))


def fig_friction(groups, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    labels, st, rep, swf = [], [], [], []
    for label, runs, ref in groups:
        runs = [r for r in runs.split(",") if r]; ref = float(ref)
        s = [istar_values(r)["staple"][-1] / ref for r in runs]; p = [istar_values(r)["repptis"][-1] / ref for r in runs]
        f = [load_json(r).get("switch", {}).get("frac_switch") for r in runs]
        labels.append(label); st.append(s); rep.append(p); swf.append([v for v in f if v is not None])
        print(f"{label}: StapleTIS {np.round(s, 2)}  REPPTIS {np.round(p, 1)}  switching {np.round(swf[-1], 3)}")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(labels))
    for k, (vals, col, name) in enumerate(((st, "tab:blue", "StapleTIS"), (rep, "tab:orange", "REPPTIS"))):
        for xi, v in zip(x, vals):
            ax.scatter(np.full(len(v), xi + (k - 0.5) * 0.15), v, color=col, alpha=0.6)
            ax.errorbar(xi + (k - 0.5) * 0.15, np.exp(np.log(v).mean()), fmt="D", color=col, ms=8, label=name if xi == 0 else None)
    ax.axhline(1, color="k", lw=1); ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"$P_{AB}$ / RETIS reference"); ax.legend(); ax.set_title("Rate estimate vs friction")
    ax2.bar(x, [np.mean(v) if v else np.nan for v in swf], color="tab:blue", alpha=0.7)
    for xi, v in zip(x, swf): ax2.scatter(np.full(len(v), xi), v, color="k", s=12)
    ax2.set_xticks(x); ax2.set_xticklabels(labels); ax2.set_ylim(0, 1)
    ax2.set_ylabel("reactive StapleTIS-MSM paths\nthat switch channel at a turn"); ax2.set_title("Tunnelling at turns")
    fig.tight_layout(); fig.savefig(out, dpi=150); print(f"saved {out}")


def fig_tunnel(run, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from potentials.potential_staggered_channels import PotentialStaggeredChannels
    d = load_json(run); sw = d.get("switch")
    if sw is None: raise SystemExit("run `switch` first")
    intf = d["diag"]["interfaces"] if "diag" in d else interfaces(run)
    head = open(os.path.join(run, "logging.log")).read(200000)
    T = float(re.search(r"'temperature': ([0-9.]+)", head).group(1))
    Vp = float(re.search(r"'V_peak': ([0-9.eE+-]+)", head).group(1)); Vd = float(re.search(r"'V_dip': ([0-9.eE+-]+)", head).group(1))
    pot = PotentialStaggeredChannels(V_peak=Vp, V_dip=Vd)
    xl = np.linspace(-1.25, 1.25, 500)
    Vu = [pot._background_potential_and_force(x, 1.3)[0] / T for x in xl]; Vl = [pot._background_potential_and_force(x, -1.3)[0] / T for x in xl]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax.plot(xl, Vu, color="tab:red", lw=2, label="upper channel"); ax.plot(xl, Vl, color="tab:blue", lw=2, label="lower channel")
    for l in intf: ax.axvline(l, color="0.88", lw=0.6)
    for key, v in sw["switch_at"].items():
        kind, rng_, d_ = key.split(" "); lo_, hi_ = [float(s) for s in rng_.strip("[]").split(",")]
        ax.axvspan(lo_, hi_, color="tab:purple" if d_ == "L->U" else "tab:brown", alpha=min(0.6, 0.1 + 2 * v))
        ax.text(0.5 * (lo_ + hi_), 8.2, f"{kind}\n{d_}\n{v:.2f}", ha="center", fontsize=7)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-4, 10); ax.set_xlabel("x"); ax.set_ylabel("V / kT"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Turns where the StapleTIS MSM switches channel ({sw['frac_switch']:.0%} of reactive paths)")
    def tx(tk):
        if tk in ("A", "B"): return intf[0] if tk == "A" else intf[-1]
        t, kind = tk
        if t == 0: return intf[0]
        return 0.5 * (intf[t - 1] + intf[t]) if kind == "min" else 0.5 * (intf[t] + intf[min(t + 1, len(intf) - 1)])
    for j, path in enumerate(sorted(sw["examples"], key=len)[:4]):
        xprev, prev = intf[0], None
        for a, b, c in path:
            y = {"U": 1, "L": -1, "N": 0}[c] * (1 + 0.08 * j); x1 = tx(b)
            col = {"U": "tab:red", "L": "tab:blue", "N": "0.6"}[c]
            if prev is not None and prev != c and "N" not in (prev, c):
                ax2.plot([xprev, xprev], [yprev, y], "k:", lw=1.5)
            ax2.annotate("", xy=(x1, y), xytext=(xprev, y), arrowprops=dict(arrowstyle="->", color=col, lw=1.8),
                         annotation_clip=False, clip_on=True)
            xprev, yprev, prev = x1, y, c
    ax2.add_patch(plt.Rectangle((-X_IN, -Y_IN), 2 * X_IN, 2 * Y_IN, color="0.4"))
    for l in intf: ax2.axvline(l, color="0.88", lw=0.6, zorder=0)
    ax2.set_xlim(intf[0] - 0.05, intf[-1] + 0.05); ax2.set_ylim(-1.5, 1.5); ax2.set_yticks([-1, 1]); ax2.set_yticklabels(["lower", "upper"])
    ax2.set_xlabel("x (arrows: staple from turn to turn)"); ax2.set_title("Example reactive MSM paths that switch channel (dotted)")
    fig.tight_layout(); fig.savefig(out, dpi=150); print(f"saved {out}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("labels", "diag", "switch"):
        p = sub.add_parser(name); p.add_argument("runs", nargs="+")
        if name == "switch": p.add_argument("--nrun", type=int, default=200000)
    p = sub.add_parser("fig-main"); p.add_argument("--istar", nargs="+", required=True); p.add_argument("--retis", nargs="+", required=True)
    p.add_argument("--factor", type=float, default=2.0, help="P_AB(full) / P_AB(RETIS run); 2 for single-channel RETIS")
    p.add_argument("-o", default="fig_main.png")
    p = sub.add_parser("fig-friction"); p.add_argument("--group", nargs=3, action="append", required=True,
                                                       metavar=("LABEL", "ISTAR_RUNS_COMMA_SEPARATED", "REFERENCE_PAB"))
    p.add_argument("-o", default="fig_friction.png")
    p = sub.add_parser("fig-tunnel"); p.add_argument("run"); p.add_argument("-o", default="fig_tunnel.png")
    a = ap.parse_args()
    if a.cmd == "labels":
        for r in a.runs: extract_labels(r)
    elif a.cmd == "diag":
        for r in a.runs: (diag_retis if simtype(r) == "retis" else diag_istar)(r)
    elif a.cmd == "switch":
        for r in a.runs: switch(r, a.nrun)
    elif a.cmd == "fig-main": fig_main(a.istar, a.retis, a.factor, a.o)
    elif a.cmd == "fig-friction": fig_friction(a.group, a.o)
    elif a.cmd == "fig-tunnel": fig_tunnel(a.run, a.o)


if __name__ == "__main__":
    main()
