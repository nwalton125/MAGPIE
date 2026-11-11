#!/usr/bin/env python3
"""Exact equilibrium of a pegeq game tree via the sequence-form linear program.

Reads the tree written by bin/pegeq, maps each leaf to a payoff with a value
function, and solves max_x min_y x'Ay for the root mover (x) and opponent (y).

Value functions (payoff to the root mover; the game is zero-sum):
  wtl      1 / 0.5 / 0 for win / tie / loss                     (default)
  spread   final spread in points
  win+spread:K   win/tie/loss plus K * spread as a tiebreak
A tree built with -window carries only win/tie/loss classes, so it supports wtl
only. Any monotone function of the final spread is valid with exact trees,
because the endgame leaves are minimax over spread.

usage: pegeq_solve.py TREE [TREE ...] [-value wtl] [-show N]
"""
import argparse
import collections
import sys

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def value_fn(name, window):
    if name == "wtl":
        return lambda v: 1.0 if v > 0 else (0.5 if v == 0 else 0.0)
    if window:
        sys.exit("tree has win/tie/loss classes only; use -value wtl")
    if name == "spread":
        return lambda v: float(v)
    if name.startswith("win+spread:"):
        k = float(name.split(":")[1])
        return lambda v: (1.0 if v > 0 else 0.5 if v == 0 else 0.0) + k * v
    sys.exit(f"unknown value function {name}")


def load(paths):
    hdr, infosets, leaves = {}, {}, collections.defaultdict(float)
    approx = collections.defaultdict(float)  # (seq0, seq1) -> chance prob
    for path in paths:
        with open(path) as f:
            for line in f:
                t = line.split()
                if not t:
                    continue
                if t[0] == "H":
                    hdr[t[1]] = t[2] if len(t) > 2 else ""
                elif t[0] == "I":
                    player, key, parent, n = int(t[1]), t[2], t[3], int(t[4])
                    acts = tuple(t[5:5 + n])
                    old = infosets.get(key)
                    if old is not None and old[2] != acts:
                        sys.exit(f"inconsistent actions at infoset {key}")
                    infosets[key] = (player, parent, acts)
                elif t[0] == "L":
                    p, s0, s1, v, exact = float(t[1]), t[2], t[3], int(t[4]), int(t[5])
                    leaves[(s0, s1, v)] += p
                    if not exact:
                        approx[(s0, s1)] += p
    return hdr, infosets, leaves, approx


def build(infosets, player):
    seqs = {"root": 0}
    for key, (pl, parent, acts) in infosets.items():
        if pl == player:
            for a in acts:
                seqs[f"{key}#{a}"] = len(seqs)
    isets = [k for k, v in infosets.items() if v[0] == player]
    rows, cols, vals = [0], [0], [1.0]  # root sequence = 1
    for r, key in enumerate(isets, start=1):
        _, parent, acts = infosets[key]
        rows.append(r); cols.append(seqs[parent]); vals.append(-1.0)
        for a in acts:
            rows.append(r); cols.append(seqs[f"{key}#{a}"]); vals.append(1.0)
    M = sp.csr_matrix((vals, (rows, cols)), shape=(len(isets) + 1, len(seqs)))
    rhs = np.zeros(len(isets) + 1)
    rhs[0] = 1.0
    return seqs, isets, M, rhs


def solve_side(A, E, e, F, f):
    """max_x min_y x'Ay  s.t. Ex=e, x>=0, Fy=f, y>=0. Returns (value, x)."""
    nx, nq = E.shape[1], F.shape[0]
    # variables [x, q]; maximize f'q  s.t.  F'q - A'x <= 0,  Ex = e
    c = np.concatenate([np.zeros(nx), -f])
    A_ub = sp.hstack([-A.T, F.T]).tocsr()
    b_ub = np.zeros(A.shape[1])
    A_eq = sp.hstack([E, sp.csr_matrix((E.shape[0], nq))]).tocsr()
    bounds = [(0, None)] * nx + [(None, None)] * nq
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=e, bounds=bounds,
                  method="highs")
    if res.status != 0:
        sys.exit("LP failed: " + res.message)
    return -res.fun, res.x[:nx]


def behaviour(infosets, seqs, x, player):
    out = {}
    for key, (pl, parent, acts) in infosets.items():
        if pl != player:
            continue
        reach = x[seqs[parent]]
        probs = [x[seqs[f"{key}#{a}"]] for a in acts]
        tot = sum(probs)
        out[key] = (reach, acts, [p / tot if tot > 1e-12 else 1 / len(acts)
                                  for p in probs])
    return out


def rack_of(seq):
    """Initial rack in a sequence key '<role>|<rack>+<draws>|<public>#<act>'."""
    return seq.split("|")[1].split("+")[0] if seq != "root" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trees", nargs="+")
    ap.add_argument("-value", default="wtl")
    ap.add_argument("-show", type=int, default=12)
    args = ap.parse_args()
    hdr, infosets, leaves, approx = load(args.trees)
    u = value_fn(args.value, hdr.get("window") == "1")
    sx, ix, E, e = build(infosets, 0)
    sy, iy, F, f = build(infosets, 1)
    A = sp.dok_matrix((len(sx), len(sy)))
    for (s0, s1, v), p in leaves.items():
        A[sx[s0], sy[s1]] += p * u(v)
    A = A.tocsr()
    print(f"deals {hdr.get('deals')}  mover infosets {len(ix)}  opp infosets "
          f"{len(iy)}  sequences {len(sx)}/{len(sy)}  leaves {len(leaves)}")
    v1, x = solve_side(A, E, e, F, f)
    v2, y = solve_side(-A.T, F, f, E, e)
    if approx:
        m = sum(p * x[sx[s0]] * y[sy[s1]] for (s0, s1), p in approx.items())
        print(f"probability of reaching a greedy-playout (inexact) leaf "
              f"under the equilibrium: {m:.4f}")
    print(f"game value for root mover ({args.value}): {v1:.6f}   "
          f"(check from opponent's LP: {-v2:.6f})")
    bx, by = behaviour(infosets, sx, x, 0), behaviour(infosets, sy, y, 1)

    # value conditional on the mover's real rack
    r0, o0 = hdr["mover_rack"], hdr["opp_rack"]
    num = 0.0
    for (s0, s1, v), p in leaves.items():
        if rack_of(s0) != r0:
            continue
        num += p * u(v) * x[sx[s0]] * y[sy[s1]]
    # reach-weighted: leaf prob already includes chance; multiply realization plans
    den = sum(p * x[sx[s0]] * y[sy[s1]] for (s0, s1, v), p in leaves.items()
              if rack_of(s0) == r0)
    if den > 0:
        print(f"value given the mover's real rack {r0}: {num / den:.6f}")

    def show(beh, prefix_filter, title):
        print(title)
        rows = [(k, r, a, pr) for k, (r, a, pr) in beh.items()
                if prefix_filter(k) and r > 1e-9]
        rows.sort(key=lambda t: -t[1])
        for k, r, a, pr in rows[:args.show]:
            mix = ", ".join(f"{act} {q:.3f}" for act, q in
                            sorted(zip(a, pr), key=lambda t: -t[1]) if q > 5e-4)
            print(f"  {k}\n      {mix}")

    show(bx, lambda k: k.split("|")[1] == r0 and k.split("|")[2] == "",
         f"Mover's first move with {r0}:")
    show(by, lambda k: k.split("|")[1].split("+")[0] == o0,
         f"Opponent's replies with {o0}:")


if __name__ == "__main__":
    main()
