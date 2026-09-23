#!/usr/bin/env python3
"""Combines antistatic endgame experiment shards into one report.

usage: antistatic_aggregate.py <results_dir> <report_dir>

<results_dir> holds the output of antistatic_shard.sh runs (part_*.txt summaries
and part_*/ game logs, possibly nested one level per shard). Writes to
<report_dir>:
  summary.md   the report (also suitable for $GITHUB_STEP_SUMMARY)
  results.csv  one row per endgame: seed and player one's final spread in
               each variation
  changed/     logs of the notable games (see NOTABLE below)
  changed.html a viewer for those games, with filters (open it in a browser)
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

# Rows: game, seed, P1's final spread in each variation, then solve counts.
# Current logs: static, antistatic P1/P2, normal P1/P2, antistatic incomplete,
# normal incomplete, no move. Older logs: static, antistatic P1/P2,
# incomplete, and optionally no move.
INT = r"\s+([+-]?\d+)"
ROW_CURRENT = re.compile(r"^\d+\s+(\d+)" + INT * 8 + r"\s*$")
ROW_OLD = re.compile(r"^\d+\s+(\d+)" + INT * 4 + r"(?:" + INT + r")?\s*$")
NO_ENDGAME = re.compile(r"^\d+\s+(\d+)\s+\(game ended before the endgame\)")
HEADER = re.compile(r"^antistatic endgame experiment: .*?, (.*)$")
SOLVER = re.compile(r"^(Antistatic|Normal) solver: (\d+) moves, ([\d.]+)s "
                    r"total, (\d+) (?:incomplete|timed out)"
                    r"(?:, (\d+) no move)?")
SOLVERS = ("antistatic", "normal")

# The most games changed.html shows; every notable game's log is still in
# changed/. Games where static beat antistatic come first.
MAX_VIEWER_GAMES = 3000


def points(spread):
    """Game points for a final spread: 1 for a win, 0.5 for a draw."""
    return 1.0 if spread > 0 else 0.5 if spread == 0 else 0.0


def seat_points(spread, seat):
    """Game points from seat 1 or 2 for player one's final spread."""
    return points(spread if seat == 1 else -spread)


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


def pts(mean_se):
    mean, se = mean_se
    return f"{100 * mean:+.2f} ± {100 * se:.2f}"


def read_results(results_dir):
    rows, no_endgame, settings = {}, set(), set()
    solver_totals = {s: [0, 0.0, 0, 0] for s in SOLVERS}
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "part_*.txt"),
                                 recursive=True)):
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if m := ROW_CURRENT.match(line):
                    v = [int(x) for x in m.groups()]
                    rows[v[0]] = {"static": v[1], "antistatic": (v[2], v[3]),
                                  "normal": (v[4], v[5])}
                elif m := ROW_OLD.match(line):
                    v = [int(x) if x is not None else 0 for x in m.groups()]
                    rows[v[0]] = {"static": v[1], "antistatic": (v[2], v[3]),
                                  "normal": None}
                elif m := NO_ENDGAME.match(line):
                    no_endgame.add(int(m.group(1)))
                elif m := HEADER.match(line):
                    settings.add(m.group(1))
                elif m := SOLVER.match(line):
                    t = solver_totals[m.group(1).lower()]
                    t[0] += int(m.group(2))
                    t[1] += float(m.group(3))
                    t[2] += int(m.group(4))
                    t[3] += int(m.group(5) or 0)
    return rows, no_endgame, settings, solver_totals


def main():
    results_dir, report_dir = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(report_dir, "changed"), exist_ok=True)
    rows, no_endgame, settings, solver_totals = read_results(results_dir)
    seeds = sorted(rows)
    has_normal = bool(seeds) and all(rows[s]["normal"] for s in seeds)
    solvers = SOLVERS if has_normal else SOLVERS[:1]

    with open(os.path.join(report_dir, "results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "static", "antistatic_p1", "antistatic_p2",
                         "normal_p1", "normal_p2"])
        for s in seeds:
            normal = rows[s]["normal"] or ("", "")
            writer.writerow([s, rows[s]["static"], *rows[s]["antistatic"],
                             *normal])

    def solver_points(s, solver, seat):
        return seat_points(rows[s][solver][seat - 1], seat)

    def static_points(s, seat):
        return seat_points(rows[s]["static"], seat)

    # Notable games, by filter.
    static_beat_anti = [s for s in seeds if any(
        static_points(s, seat) > solver_points(s, "antistatic", seat)
        for seat in (1, 2))]
    anti_changed = [s for s in seeds if any(
        static_points(s, seat) != solver_points(s, "antistatic", seat)
        for seat in (1, 2))]
    anti_vs_normal = [s for s in seeds if has_normal and any(
        solver_points(s, "antistatic", seat) !=
        solver_points(s, "normal", seat) for seat in (1, 2))]
    normal_changed = [s for s in seeds if has_normal and any(
        static_points(s, seat) != solver_points(s, "normal", seat)
        for seat in (1, 2))]
    notable = sorted(set(static_beat_anti) | set(anti_changed) |
                     set(anti_vs_normal) | set(normal_changed))

    logs = {}
    for path in glob.glob(os.path.join(results_dir, "**", "game_*_seed_*.txt"),
                          recursive=True):
        m = re.search(r"_seed_(\d+)\.txt$", path)
        logs[int(m.group(1))] = path
    for s in notable:
        if s in logs:
            shutil.copy(logs[s], os.path.join(report_dir, "changed"))
    first = [s for s in static_beat_anti if s in logs]
    shown = first + [s for s in notable if s in logs and s not in set(first)]
    shown = sorted(shown[:MAX_VIEWER_GAMES])
    if shown:
        mode = ("spread" if any("max spread" in x for x in settings)
                else "firstwin")
        capped = len(shown) < len([s for s in notable if s in logs])
        intro = (
            f"Notable endgames out of {len(rows)}: where a solver's result "
            "differs from static's in the same seat, or the two solvers' "
            "results differ. Both players made static moves until the bag "
            "was empty; each endgame was then played static vs static and "
            "with each solver on each side.")
        if capped:
            intro += (f" Showing {len(shown)} of the notable games, including "
                      "every game where static beat antistatic; all of their "
                      "logs are in changed/.")
        html = antistatic_viewer.build_page(
            {mode: [logs[s] for s in shown]}, "Antistatic Result Changes",
            f"MAGPIE · {'; '.join(sorted(settings))}", intro=intro,
            subset=True)
        with open(os.path.join(report_dir, "changed.html"), "w") as f:
            f.write(html)

    # Report.
    n = len(seeds)
    out = ["# Antistatic endgame experiment", ""]
    out.append(f"Settings: {'; '.join(sorted(settings)) or 'unknown'}")
    out.append("")
    out.append(f"Endgames: **{n}** (plus {len(no_endgame)} games that ended "
               f"before the bag emptied), seeds "
               f"{seeds[0] if seeds else '-'}–{seeds[-1] if seeds else '-'}")
    out.append("")
    out.append("Win% counts a draw as half a win. A gain is the solver's win% "
               "minus static's from the same seat in the same endgame, with "
               "its standard error, in win% points.")
    out.append("")
    out.append("| Player | Seat | Win% | Gain over static | Results better / "
               "worse than static |")
    out.append("|---|---|---|---|---|")
    for seat in (1, 2):
        out.append(f"| Static | P{seat} | "
                   f"{pct(mean_and_se([static_points(s, seat) for s in seeds])[0])}"
                   f" | | |")
    for solver in solvers:
        for seat in (1, 2):
            diffs = [solver_points(s, solver, seat) - static_points(s, seat)
                     for s in seeds]
            win = mean_and_se([solver_points(s, solver, seat)
                               for s in seeds])[0]
            better = sum(d > 0 for d in diffs)
            worse = sum(d < 0 for d in diffs)
            out.append(f"| {solver.capitalize()} | P{seat} | {pct(win)} | "
                       f"{pts(mean_and_se(diffs))} | {better} / {worse} |")
    out.append("")
    for solver in solvers:
        both = mean_and_se([
            (solver_points(s, solver, 1) - static_points(s, 1) +
             solver_points(s, solver, 2) - static_points(s, 2)) / 2
            for s in seeds])
        out.append(f"- {solver.capitalize()} over static, both seats: "
                   f"**{pts(both)}** win% points per endgame")
    if has_normal:
        both = mean_and_se([
            (solver_points(s, "antistatic", 1) - solver_points(s, "normal", 1)
             + solver_points(s, "antistatic", 2) -
             solver_points(s, "normal", 2)) / 2 for s in seeds])
        out.append(f"- Antistatic over normal, both seats: **{pts(both)}** "
                   "win% points per endgame")
    out.append("")
    out.append("Notable games (the filters in changed.html):")
    out.append("")
    out.append(f"- Antistatic changed the result: {len(anti_changed)}")
    out.append(f"- Static did better than antistatic: "
               f"**{len(static_beat_anti)}**")
    if has_normal:
        out.append(f"- Normal solver changed the result: "
                   f"{len(normal_changed)}")
        out.append(f"- Antistatic and normal results differ: "
                   f"{len(anti_vs_normal)}")
    out.append("")
    for solver in solvers:
        moves, secs, incomplete, no_move = solver_totals[solver]
        out.append(f"{solver.capitalize()} solver: {moves} moves, "
                   f"{secs:.1f}s total. **{incomplete}** cut short by the time "
                   f"limit (played the best move from the deepest completed "
                   f"depth); **{no_move}** returned no move (played the static "
                   f"move).")
        out.append("")
    if static_beat_anti:
        out.append("Games where static did better than antistatic:")
        out.append("")
        for s in static_beat_anti[:200]:
            seats = [seat for seat in (1, 2) if static_points(s, seat) >
                     solver_points(s, "antistatic", seat)]
            out.append(f"- seed {s}: as " +
                       " and ".join(f"P{seat}" for seat in seats))
        if len(static_beat_anti) > 200:
            out.append(f"- … and {len(static_beat_anti) - 200} more")
    text = "\n".join(out) + "\n"
    with open(os.path.join(report_dir, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
