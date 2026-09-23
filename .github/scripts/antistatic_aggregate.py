#!/usr/bin/env python3
"""Combines antistatic endgame experiment shards into one report.

usage: antistatic_aggregate.py <results_dir> <report_dir>

<results_dir> holds the output of antistatic_shard.sh runs (part_*.txt summaries
and part_*/ game logs, possibly nested one level per shard). Writes to
<report_dir>:
  summary.md   the report (also suitable for $GITHUB_STEP_SUMMARY)
  results.csv  one row per endgame: seed and player one's final spread in
               each variation
  changed/     logs of the games where the antistatic endgame changed the result
  changed.html a viewer for those games (open it in a browser)
"""

import csv
import glob
import math
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import antistatic_viewer  # noqa: E402

ROW = re.compile(
    r"^\d+\s+(\d+)\s+([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)\s+(\d+)\s*$")
NO_ENDGAME = re.compile(r"^\d+\s+(\d+)\s+\(game ended before the endgame\)")
HEADER = re.compile(r"^antistatic endgame experiment: .*?, (.*)$")
SOLVER = re.compile(
    r"^Antistatic solver: (\d+) moves, ([\d.]+)s total, (\d+) timed out")


def points(spread):
    """Game points for a final spread: 1 for a win, 0.5 for a draw."""
    return 1.0 if spread > 0 else 0.5 if spread == 0 else 0.0


def mean_and_se(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def pct(x):
    return f"{100 * x:.2f}%"


def main():
    results_dir, report_dir = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(report_dir, "changed"), exist_ok=True)

    rows = {}
    no_endgame = set()
    settings = set()
    solver_moves = solver_seconds = solver_timeouts = 0
    summaries = sorted(
        glob.glob(os.path.join(results_dir, "**", "part_*.txt"), recursive=True))
    for path in summaries:
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if m := ROW.match(line):
                    seed = int(m.group(1))
                    rows[seed] = tuple(int(m.group(i)) for i in range(2, 6))
                elif m := NO_ENDGAME.match(line):
                    no_endgame.add(int(m.group(1)))
                elif m := HEADER.match(line):
                    settings.add(m.group(1))
                elif m := SOLVER.match(line):
                    solver_moves += int(m.group(1))
                    solver_seconds += float(m.group(2))
                    solver_timeouts += int(m.group(3))

    seeds = sorted(rows)
    with open(os.path.join(report_dir, "results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["seed", "static", "antistatic_p1", "antistatic_p2", "timeouts"])
        for seed in seeds:
            writer.writerow([seed, *rows[seed]])

    # Per seat: the antistatic player's game points minus what the static
    # player earned from the same seat in the same endgame.
    seats = []
    changed = []
    for seat, index, sign in ((1, 1, 1), (2, 2, -1)):
        static_points = [points(sign * rows[s][0]) for s in seeds]
        anti_points = [points(sign * rows[s][index]) for s in seeds]
        diffs = [a - b for a, b in zip(anti_points, static_points)]
        spread_gain, _ = mean_and_se(
            [sign * (rows[s][index] - rows[s][0]) for s in seeds])
        better = [s for s, d in zip(seeds, diffs) if d > 0]
        worse = [s for s, d in zip(seeds, diffs) if d < 0]
        changed += [(seat, s, "better") for s in better]
        changed += [(seat, s, "worse") for s in worse]
        seats.append({
            "seat": seat,
            "static": mean_and_se(static_points)[0],
            "anti": mean_and_se(anti_points)[0],
            "gain": mean_and_se(diffs),
            "spread_gain": spread_gain,
            "better": better,
            "worse": worse,
        })
    both = mean_and_se([
        (points(rows[s][1]) - points(rows[s][0]) +
         points(-rows[s][2]) - points(-rows[s][0])) / 2 for s in seeds
    ])

    # Copy the logs of games whose result changed.
    logs = {}
    for path in glob.glob(os.path.join(results_dir, "**", "game_*_seed_*.txt"),
                          recursive=True):
        m = re.search(r"_seed_(\d+)\.txt$", path)
        logs[int(m.group(1))] = path
    for _, seed, _ in changed:
        if seed in logs:
            shutil.copy(logs[seed], os.path.join(report_dir, "changed"))
    changed_logs = sorted({logs[s] for _, s, _ in changed if s in logs})
    if changed_logs:
        mode = "spread" if any("max spread" in x for x in settings) else \
            "firstwin"
        html = antistatic_viewer.build_page(
            {mode: changed_logs}, "Antistatic Result Changes",
            f"MAGPIE · {'; '.join(sorted(settings))}",
            intro=("The endgames where the antistatic solver changed the "
                   "result compared with static play, out of "
                   f"{len(rows)} endgames. Both players made static moves "
                   "until the bag was empty; each endgame was then played "
                   "static vs static and with the antistatic solver on each "
                   "side."),
            subset=True)
        with open(os.path.join(report_dir, "changed.html"), "w") as f:
            f.write(html)

    n = len(seeds)
    out = ["# Antistatic endgame experiment", ""]
    out.append(f"Settings: {'; '.join(sorted(settings)) or 'unknown'}")
    out.append("")
    out.append(
        f"Endgames: **{n}** (plus {len(no_endgame)} games that ended before "
        f"the bag emptied), seeds {seeds[0] if seeds else '-'}–"
        f"{seeds[-1] if seeds else '-'}")
    out.append("")
    out.append("Win% counts a draw as half a win. The gain is the antistatic "
               "player's win% minus the static player's from the same seat in "
               "the same endgame, with its standard error.")
    out.append("")
    out.append("| Seat | Static win% | Antistatic win% | Gain | Results "
               "better / worse | Mean spread gain |")
    out.append("|---|---|---|---|---|---|")
    for s in seats:
        gain, se = s["gain"]
        out.append(
            f"| P{s['seat']} | {pct(s['static'])} | {pct(s['anti'])} | "
            f"{100 * gain:+.2f} ± {100 * se:.2f} pts | {len(s['better'])} / "
            f"{len(s['worse'])} | {s['spread_gain']:+.2f} |")
    out.append("")
    out.append(f"Average over both seats: **{100 * both[0]:+.2f} ± "
               f"{100 * both[1]:.2f}** win% points per endgame.")
    out.append("")
    out.append(f"Solver: {solver_moves} moves, {solver_seconds:.1f}s total, "
               f"**{solver_timeouts} timed out**.")
    if changed:
        out.append("")
        out.append("Games where the result changed (logs in `changed/`):")
        out.append("")
        for seat, seed, direction in sorted(changed, key=lambda c: c[1]):
            out.append(f"- seed {seed}: antistatic as P{seat}, result "
                       f"{direction}")
    text = "\n".join(out) + "\n"
    with open(os.path.join(report_dir, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
