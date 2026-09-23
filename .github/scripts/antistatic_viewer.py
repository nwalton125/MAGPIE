#!/usr/bin/env python3
"""Builds a self-contained HTML viewer for antistatic endgame game logs.

usage: antistatic_viewer.py <out.html> <mode> <log> [<log> ...]
       [--title T] [--eyebrow E] [--intro I] [--subset]

<mode> is "firstwin" or "spread": the solver mode the logs were run in. The
viewer shows each game's endgame board and its three endgame variations, with
every move replayable on the board.
"""

import argparse
import json
import os
import re

MOVE = re.compile(r"^  P([12]) (\S*)  (.*?)  (-?\d+)-(-?\d+)"
                  r"(?:  \[solver ([\d.]+)s(, TIMED OUT)?\])?$")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "antistatic_viewer.html")


def parse_move(line):
    m = MOVE.match(line)
    if not m:
        return None
    player, rack, text, s1, s2, solver, timed_out = m.groups()
    move = {"player": int(player), "rack": rack, "s1": int(s1),
            "s2": int(s2)}
    if solver is not None:
        move["solver"] = float(solver)
        move["timedOut"] = bool(timed_out)
    text = text.strip()
    if m := re.match(r"^\(exch (\S+)\)$", text):
        move.update(kind="exchange", text=f"exchange {m.group(1)}", score=0)
    elif text.startswith("pass"):
        move.update(kind="pass", text="pass", score=0)
    else:
        coord, word, score = re.match(r"^(\S+) (\S+) (-?\d+)$", text).groups()
        move.update(kind="play", coord=coord, word=word, score=int(score),
                    text=f"{coord} {word}")
    return move


def parse_log(path):
    """Parses one game log written by magpie_test antistaticeg."""
    with open(path) as f:
        lines = f.read().split("\n")
    game = {"seed": int(re.search(r"seed (\d+)", lines[0]).group(1)),
            "pre": [], "variations": []}
    section = None
    current = None
    for line in lines[1:]:
        if line.startswith("Static play until"):
            section = "pre"
        elif line.startswith("Endgame position"):
            section = "cgp"
        elif section == "cgp" and line.startswith("  "):
            game["cgp"] = line.strip()
            section = None
        elif line and not line.startswith(" ") and line.endswith(":"):
            current = {"name": line[:-1], "moves": []}
            game["variations"].append(current)
            section = "variation"
        elif line.startswith("  Final:"):
            m = re.match(r"  Final: (-?\d+)-(-?\d+), P1 spread ([+-]?\d+)", line)
            current["final"] = [int(m.group(1)), int(m.group(2))]
            current["spread"] = int(m.group(3))
        elif (move := parse_move(line)) is not None:
            target = game["pre"] if section == "pre" else current["moves"]
            target.append(move)
    return game


def build_page(logs_by_mode, title, eyebrow, intro="", subset=False):
    """Returns the viewer HTML for {mode: [log paths]}."""
    games = {}
    for mode, paths in logs_by_mode.items():
        for path in paths:
            parsed = parse_log(path)
            entry = games.setdefault(parsed["seed"], {
                "seed": parsed["seed"], "pre": parsed["pre"],
                "cgp": parsed["cgp"], "modes": {}})
            entry["modes"][mode] = parsed["variations"]
    ordered = [games[seed] for seed in sorted(games)]
    meta = {"title": title, "eyebrow": eyebrow, "intro": intro,
            "subset": subset}
    with open(TEMPLATE) as f:
        page = f.read()
    # json.dumps output can't contain "</script>" once "</" is escaped.
    def embed(value):
        return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")
    return (page.replace("__TITLE__", title)
            .replace("__META__", embed(meta))
            .replace("__GAMES__", embed(ordered)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("mode", choices=["firstwin", "spread"])
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--title", default="Antistatic Endgames")
    parser.add_argument("--eyebrow", default="MAGPIE")
    parser.add_argument("--intro", default="")
    parser.add_argument("--subset", action="store_true")
    args = parser.parse_args()
    html = build_page({args.mode: args.logs}, args.title, args.eyebrow,
                      args.intro, args.subset)
    with open(args.out, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
