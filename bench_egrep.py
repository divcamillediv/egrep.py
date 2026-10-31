#!/usr/bin/env python3
"""
Le programme sert exclusivement à générer les graphes nécessaires pour le rapport de TP.

Performance tests for egrep.py.

Experiments (select with --only, default: all):
  per_pattern      End-to-end wall time of GNU egrep (grep -E) and egrep.py on the
                   book, one pattern per operator family. Output is checked to be
                   identical before timing.
  breakdown        Where egrep.py spends its time: interpreter startup, compiling
                   the pattern (KMP table or NFA -> DFA -> min-DFA), searching.
  scaling          Wall time as the input grows (book concatenated k times), for a
                   literal pattern (KMP) and a regex (automaton).
  blowup           NFA / DFA / min-DFA state counts for (a|b)*a(a|b)^k, k = 1..12:
                   the NFA grows linearly, the DFA as 2^(k+1).
  kmp_book         Literal search on the book: naive vs KMP vs automaton (the
                   lecture-1 method), matching time only, same lines in memory.
  kmp_adversarial  Worst case for the naive algorithm: lines "aaa...a", pattern
                   "a...ab" of length m. Naive is O(n*m), KMP and automaton O(n).

Usage:
  python bench_egrep.py                          # everything, 10 runs (a few minutes)
  python bench_egrep.py --quick                  # 3 runs, smaller inputs
  python bench_egrep.py --only kmp_book kmp_adversarial
  python bench_egrep.py --runs 20 --file other_book.txt --out results/

Outputs in --out (default bench_results/): one CSV per experiment and, if
matplotlib is installed, one PNG chart per experiment.
"""
import argparse
import csv
import io
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
EGREP_PY = os.path.join(HERE, "egrep.py")
sys.path.insert(0, HERE)
from alphabet import Alphabet  # noqa: E402
from dfa import DFA  # noqa: E402
from egrep import build_matcher, process_stream  # noqa: E402
from kmp import KMP, Naive  # noqa: E402
from nfa import NFA  # noqa: E402
from syntax_tree import make_search_tree, parse_regex  # noqa: E402

EXPERIMENTS = ["per_pattern", "breakdown", "scaling", "blowup", "kmp_book", "kmp_adversarial"]
NEEDS_BOOK = {"per_pattern", "breakdown", "scaling", "kmp_book"}
NEEDS_GREP = {"per_pattern", "scaling"}

# (label, pattern): one family of operators per entry
PATTERNS = [
    ("literal",         "Babylon"),
    ("rare literal",    "Nebuchadnezzar"),
    ("dot",             "S.r.on"),
    ("star",            "th(e|a)*n"),
    ("plus",            "S(a|g|r)+on"),
    ("option",          "Ba(b|r)+y?lon"),
    ("alternation x2",  "Assyria|Babylon"),
    ("alternation x6",  "Sargon|Babylon|Assyria|Persia|Egypt|Elam"),
    ("nested groups",   "((a|e)(r|n))+t"),
    ("dense (.*)",      "a.*e.*i.*o.*u"),
]

SCALING_PATTERNS = [("literal (KMP)", "Babylon"), ("regex (DFA)", "S(a|g|r)+on")]

# Literal patterns for naive vs KMP on the book: from very frequent to absent.
BOOK_LITERALS = ["the", "of the", "Babylon", "Hammurabi", "Nebuchadnezzar",
                 "Merodach-baladan", "the First Dynasty of Babylon", "xylophone"]

# Colors: reference categorical palette, fixed order, one color per entity.
COLORS = {
    "egrep": "#2a78d6",      # slot 1, blue
    "egrep.py": "#eb6834",   # slot 2, orange
    "KMP": "#1baf7a",        # slot 3, aqua
    "naive": "#eda100",      # slot 4, yellow
    "automaton": "#e87ba4",  # slot 5, magenta
}
NEUTRAL = "#a3a29b"


# -----------------------------------------------------------------------------
# Measurement helpers
# -----------------------------------------------------------------------------
def find_system_egrep():
    """Prefer 'grep -E' (modern 'egrep' is a wrapper that may print warnings)."""
    grep = shutil.which("grep")
    if grep:
        return [grep, "-E"]
    egrep_bin = shutil.which("egrep")
    return [egrep_bin] if egrep_bin else None


def commands(system_egrep):
    return {
        "egrep": lambda p, f: system_egrep + ["--", p, f],
        "egrep.py": lambda p, f: [sys.executable, EGREP_PY, "--", p, f],
    }


def run_output(cmd):
    proc = subprocess.run(cmd, capture_output=True)
    return proc.stdout.decode("utf-8", errors="replace").replace("\r", "")


def time_command(cmd, runs, warmup=1):
    """
    Wall-clock times (seconds) of `runs` executions.

    Output is read through a pipe, NOT sent to /dev/null: GNU grep detects
    /dev/null and stops at the first match (like -q), which would make the
    comparison meaningless.
    """
    for _ in range(warmup):
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        times.append(time.perf_counter() - t0)
    return times


def time_function(fn, runs, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def mean_std(values):
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def ms(pair):
    """(mean, std) in seconds -> (mean, std) in milliseconds."""
    return pair[0] * 1e3, pair[1] * 1e3


def read_lines(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\r\n") for line in f]


def build_automaton(pattern, search=True):
    """tree -> NFA -> DFA -> min-DFA, always (even for literal patterns)."""
    tree = parse_regex(pattern)
    if search:
        tree = make_search_tree(tree)
    nfa = NFA.from_tree(tree)
    alphabet = Alphabet(nfa.collect_explicit_characters())
    dfa = DFA.determinize(nfa, alphabet)
    return nfa, dfa, alphabet, DFA.minimize(dfa, alphabet)


def build_engine(name, pattern):
    """Matcher for one search strategy: 'naive', 'KMP' or 'automaton'."""
    if name == "naive":
        return Naive(pattern).match_line
    if name == "KMP":
        return KMP(pattern).match_line
    _, _, alphabet, min_dfa = build_automaton(pattern)
    return lambda line: min_dfa.match_line(line, alphabet)


def count_matches(match, lines):
    return sum(1 for line in lines if match(line))


# -----------------------------------------------------------------------------
# Experiment: per pattern, end to end vs GNU egrep
# -----------------------------------------------------------------------------
def bench_per_pattern(cmds, text_file, runs):
    print(f"\n=== per_pattern: GNU egrep vs egrep.py ({runs} runs, {os.path.basename(text_file)}) ===")
    print(f"{'pattern':<16} {'regex':<30} {'engine':<6} {'egrep (ms)':>16} {'egrep.py (ms)':>18} {'ratio':>7}")
    rows = []
    for label, pattern in PATTERNS:
        ref = run_output(cmds["egrep"](pattern, text_file))
        if run_output(cmds["egrep.py"](pattern, text_file)) != ref:
            print(f"{label:<16} {pattern:<30} SKIPPED: outputs differ")
            continue
        engine = build_matcher(pattern)[1]["engine"]
        stats = {tool: ms(mean_std(time_command(make(pattern, text_file), runs)))
                 for tool, make in cmds.items()}
        ratio = stats["egrep.py"][0] / stats["egrep"][0]
        print(f"{label:<16} {pattern[:30]:<30} {engine:<6} "
              f"{stats['egrep'][0]:9.2f} ± {stats['egrep'][1]:5.2f} "
              f"{stats['egrep.py'][0]:11.2f} ± {stats['egrep.py'][1]:5.2f} {ratio:6.1f}x")
        rows.append({
            "label": label, "pattern": pattern, "engine": engine,
            "matching_lines": ref.count("\n"),
            "egrep_mean_ms": stats["egrep"][0], "egrep_std_ms": stats["egrep"][1],
            "py_mean_ms": stats["egrep.py"][0], "py_std_ms": stats["egrep.py"][1],
            "ratio": ratio,
        })
    return rows


# -----------------------------------------------------------------------------
# Experiment: where does egrep.py spend its time?
# -----------------------------------------------------------------------------
def bench_breakdown(text_file, runs):
    print(f"\n=== breakdown: egrep.py time split ({runs} runs) ===")
    startup = ms(mean_std(time_command([sys.executable, "-c", "pass"], runs)))
    print(f"Python interpreter startup: {startup[0]:.2f} ± {startup[1]:.2f} ms")
    print(f"{'pattern':<16} {'engine':<6} {'NFA':>5} {'DFA':>5} {'min':>5} "
          f"{'build (ms)':>16} {'search (ms)':>16}")
    rows = []
    for label, pattern in PATTERNS:
        build_t, search_t = [], []
        for _ in range(runs):
            t0 = time.perf_counter()
            match, info = build_matcher(pattern)
            t1 = time.perf_counter()
            with open(text_file, "r", encoding="utf-8", errors="replace") as f, \
                 redirect_stdout(io.StringIO()):
                process_stream(f, None, match)
            t2 = time.perf_counter()
            build_t.append(t1 - t0)
            search_t.append(t2 - t1)
        b, s = ms(mean_std(build_t)), ms(mean_std(search_t))
        sizes = [info.get(k, "") for k in ("nfa_states", "dfa_states", "min_dfa_states")]
        print(f"{label:<16} {info['engine']:<6} {sizes[0]!s:>5} {sizes[1]!s:>5} {sizes[2]!s:>5} "
              f"{b[0]:9.3f} ± {b[1]:5.3f} {s[0]:9.2f} ± {s[1]:5.2f}")
        rows.append({
            "label": label, "pattern": pattern, "engine": info["engine"],
            "nfa_states": sizes[0], "dfa_states": sizes[1], "min_dfa_states": sizes[2],
            "startup_mean_ms": startup[0], "startup_std_ms": startup[1],
            "build_mean_ms": b[0], "build_std_ms": b[1],
            "search_mean_ms": s[0], "search_std_ms": s[1],
        })
    return rows


# -----------------------------------------------------------------------------
# Experiment: scaling with input size
# -----------------------------------------------------------------------------
def bench_scaling(cmds, text_file, runs, factors):
    print(f"\n=== scaling: wall time vs file size ({runs} runs) ===")
    with open(text_file, "rb") as f:
        book = f.read()
    tmpdir = tempfile.mkdtemp(prefix="egrep_bench_")
    rows = []
    try:
        print(f"{'size (MB)':>10} {'pattern':<14} {'egrep (ms)':>16} {'egrep.py (ms)':>18} {'egrep.py MB/s':>14}")
        for k in factors:
            path = os.path.join(tmpdir, f"book_x{k}.txt")
            with open(path, "wb") as f:
                f.write(book * k)
            size_mb = len(book) * k / 1e6
            for label, pattern in SCALING_PATTERNS:
                stats = {tool: ms(mean_std(time_command(make(pattern, path), runs)))
                         for tool, make in cmds.items()}
                print(f"{size_mb:10.2f} {label:<14} "
                      f"{stats['egrep'][0]:9.2f} ± {stats['egrep'][1]:5.2f} "
                      f"{stats['egrep.py'][0]:11.2f} ± {stats['egrep.py'][1]:5.2f} "
                      f"{size_mb / (stats['egrep.py'][0] / 1e3):14.2f}")
                rows.append({
                    "factor": k, "size_mb": size_mb, "label": label, "pattern": pattern,
                    "egrep_mean_ms": stats["egrep"][0], "egrep_std_ms": stats["egrep"][1],
                    "py_mean_ms": stats["egrep.py"][0], "py_std_ms": stats["egrep.py"][1],
                })
            os.remove(path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return rows


# -----------------------------------------------------------------------------
# Experiment: exponential state blow-up of the subset construction
# -----------------------------------------------------------------------------
def blowup_pattern(k):
    """(a|b)*a(a|b)^k, i.e. "the (k+1)-th character from the end is an a".
    The ERE subset has no {k} counter, so (a|b) is written out k times."""
    return "(a|b)*a" + "(a|b)" * k


def bench_blowup(k_max=12):
    """
    Classic worst case for determinization: the DFA has to remember the last
    k+1 characters, so it needs 2^(k+1) states, and no smaller DFA exists
    (each suffix of length k+1 is a distinct Myhill-Nerode class). The Thompson
    NFA only grows by a constant number of states per (a|b).

    Automata are built for the pattern itself (no .* prefix) so that the counts
    are those of the textbook language. The search-mode count (implicit .*
    prefix, what egrep.py builds) is in the CSV as well.
    """
    print(f"\n=== blowup: (a|b)*a(a|b)^k, k = 1..{k_max} ===")
    print(f"{'k':>3} {'NFA':>6} {'DFA':>7} {'min-DFA':>8} {'2^(k+1)':>8} "
          f"{'min-DFA search':>15} {'build (ms)':>11}")
    rows = []
    for k in range(1, k_max + 1):
        pattern = blowup_pattern(k)
        t0 = time.perf_counter()
        nfa, dfa, _, min_dfa = build_automaton(pattern, search=False)
        build_ms = (time.perf_counter() - t0) * 1e3
        min_search = build_automaton(pattern, search=True)[3]
        print(f"{k:3d} {len(nfa.states):6d} {dfa.state_count:7d} {min_dfa.state_count:8d} "
              f"{2 ** (k + 1):8d} {min_search.state_count:15d} {build_ms:11.1f}")
        rows.append({
            "k": k, "pattern": pattern, "nfa_states": len(nfa.states),
            "dfa_states": dfa.state_count, "min_dfa_states": min_dfa.state_count,
            "theory_2^(k+1)": 2 ** (k + 1),
            "min_dfa_states_search_mode": min_search.state_count,
            "build_ms": build_ms,
        })
    return rows


# -----------------------------------------------------------------------------
# Experiments: naive vs KMP vs automaton on literal patterns
# -----------------------------------------------------------------------------
ENGINES = ["naive", "KMP", "automaton"]


def compare_engines(pattern, lines, runs):
    """Time the search of every line with each engine. Returns one row per
    engine; raises if the engines disagree on the number of matching lines."""
    rows, counts = [], {}
    for engine in ENGINES:
        t0 = time.perf_counter()
        match = build_engine(engine, pattern)
        build_ms = (time.perf_counter() - t0) * 1e3
        counts[engine] = count_matches(match, lines)
        mean, std = ms(mean_std(time_function(lambda: count_matches(match, lines), runs)))
        rows.append({"engine": engine, "build_ms": build_ms,
                     "search_mean_ms": mean, "search_std_ms": std,
                     "matching_lines": counts[engine]})
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"engines disagree on {pattern!r}: {counts}")
    return rows


def bench_kmp_book(text_file, runs):
    print(f"\n=== kmp_book: naive vs KMP vs automaton on the book ({runs} runs) ===")
    lines = read_lines(text_file)
    print(f"{len(lines)} lines, {sum(map(len, lines))} characters (search time only, lines in memory)")
    print(f"{'pattern':<30} {'m':>3} {'lines':>6} " + " ".join(f"{e + ' (ms)':>18}" for e in ENGINES))
    rows = []
    for pattern in BOOK_LITERALS:
        results = compare_engines(pattern, lines, runs)
        print(f"{pattern:<30} {len(pattern):3d} {results[0]['matching_lines']:6d} " +
              " ".join(f"{r['search_mean_ms']:10.1f} ± {r['search_std_ms']:5.1f}" for r in results))
        for r in results:
            rows.append({"pattern": pattern, "m": len(pattern), **r})
    return rows


def bench_kmp_adversarial(runs, n_lines, line_length, lengths):
    print(f"\n=== kmp_adversarial: {n_lines} lines of 'a' x {line_length}, "
          f"pattern 'a' x (m-1) + 'b' ({runs} runs) ===")
    lines = ["a" * line_length] * n_lines
    print(f"{'m':>4} " + " ".join(f"{e + ' (ms)':>18}" for e in ENGINES) + f" {'naive/KMP':>10}")
    rows = []
    for m in lengths:
        pattern = "a" * (m - 1) + "b"
        results = compare_engines(pattern, lines, runs)
        by = {r["engine"]: r for r in results}
        print(f"{m:4d} " + " ".join(f"{r['search_mean_ms']:10.1f} ± {r['search_std_ms']:5.1f}"
                                    for r in results) +
              f" {by['naive']['search_mean_ms'] / by['KMP']['search_mean_ms']:9.1f}x")
        for r in results:
            rows.append({"m": m, "pattern": pattern, "n_lines": n_lines,
                         "line_length": line_length, **r})
    return rows


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
    })
    return plt


def save(fig, plt, out_dir, name):
    fig.tight_layout()
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_grouped_bars(plt, out_dir, name, title, groups, series, ylabel, log=True):
    """series: list of (label, color, means, stds), one value per group."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    n = len(series)
    width = 0.8 / n
    for i, (label, color, means, stds) in enumerate(series):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        ax.bar(xs, means, width * 0.92, yerr=stds, capsize=3, color=color,
               label=label, error_kw={"elinewidth": 1})
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha="right")
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend()
    return save(fig, plt, out_dir, name)


def plot_per_pattern(plt, out_dir, rows):
    groups = [f"{r['label']}\n[{r['engine']}]" for r in rows]
    return plot_grouped_bars(
        plt, out_dir, "per_pattern.png", "Time per pattern, end to end (mean ± std)", groups,
        [("egrep (GNU)", COLORS["egrep"], [r["egrep_mean_ms"] for r in rows], [r["egrep_std_ms"] for r in rows]),
         ("egrep.py", COLORS["egrep.py"], [r["py_mean_ms"] for r in rows], [r["py_std_ms"] for r in rows])],
        "wall time (ms, log scale)")


def plot_breakdown(plt, out_dir, rows):
    labels = [f"{r['label']}\n[{r['engine']}]" for r in rows]
    startup = [r["startup_mean_ms"] for r in rows]
    build = [r["build_mean_ms"] for r in rows]
    search = [r["search_mean_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, startup, color=NEUTRAL, label="interpreter startup", edgecolor="white")
    ax.bar(labels, build, bottom=startup, color=COLORS["naive"], label="compile pattern",
           edgecolor="white")
    ax.bar(labels, search, bottom=[a + b for a, b in zip(startup, build)], color=COLORS["egrep.py"],
           yerr=[r["search_std_ms"] for r in rows], capsize=3, label="search", edgecolor="white")
    ax.set_ylabel("time (ms)")
    ax.set_title("egrep.py: where the time goes (mean, search ± std)")
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend()
    return save(fig, plt, out_dir, "breakdown.png")


def plot_scaling(plt, out_dir, rows):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    styles = {SCALING_PATTERNS[0][0]: ("--", "s"), SCALING_PATTERNS[1][0]: ("-", "o")}
    for label, _ in SCALING_PATTERNS:
        sub = [r for r in rows if r["label"] == label]
        sizes = [r["size_mb"] for r in sub]
        linestyle, marker = styles[label]
        for tool, key in (("egrep (GNU)", "egrep"), ("egrep.py", "py")):
            ax.errorbar(sizes, [r[f"{key}_mean_ms"] for r in sub], yerr=[r[f"{key}_std_ms"] for r in sub],
                        color=COLORS["egrep" if key == "egrep" else "egrep.py"],
                        linestyle=linestyle, marker=marker, capsize=3, label=f"{tool}, {label}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("file size (MB, log scale)")
    ax.set_ylabel("wall time (ms, log scale)")
    ax.set_title("Scaling with input size")
    ax.legend(fontsize=9)
    return save(fig, plt, out_dir, "scaling.png")


def plot_blowup(plt, out_dir, rows):
    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ks, [r["theory_2^(k+1)"] for r in rows], color="#dddcd6", linewidth=9, zorder=0,
            solid_capstyle="round", label="$2^{k+1}$ (theory)")
    ax.plot(ks, [r["dfa_states"] for r in rows], marker="o", color=COLORS["egrep"],
            label="DFA (subset construction)")
    ax.plot(ks, [r["min_dfa_states"] for r in rows], marker="x", linestyle="--",
            color=COLORS["egrep.py"], label="minimal DFA (Moore)")
    ax.plot(ks, [r["nfa_states"] for r in rows], marker="s", color=COLORS["KMP"],
            label="NFA (Thompson)")
    ax.set_yscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xlabel("k")
    ax.set_ylabel("number of states (log$_2$ scale)")
    ax.set_title("State blow-up for (a|b)*a(a|b)$^k$")
    ax.legend()
    return save(fig, plt, out_dir, "blowup.png")


def plot_kmp_book(plt, out_dir, rows):
    patterns = list(dict.fromkeys(r["pattern"] for r in rows))
    groups = [f"{p}\n(m={len(p)})" for p in patterns]
    series = []
    for engine in ENGINES:
        sub = {r["pattern"]: r for r in rows if r["engine"] == engine}
        series.append((engine, COLORS[engine], [sub[p]["search_mean_ms"] for p in patterns],
                       [sub[p]["search_std_ms"] for p in patterns]))
    return plot_grouped_bars(plt, out_dir, "kmp_book.png",
                             "Literal search on the book: naive vs KMP vs automaton (mean ± std)",
                             groups, series, "search time (ms)", log=False)


def plot_kmp_adversarial(plt, out_dir, rows):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    markers = {"naive": "o", "KMP": "s", "automaton": "^"}
    for engine in ENGINES:
        sub = [r for r in rows if r["engine"] == engine]
        ax.errorbar([r["m"] for r in sub], [r["search_mean_ms"] for r in sub],
                    yerr=[r["search_std_ms"] for r in sub], color=COLORS[engine],
                    marker=markers[engine], capsize=3, label=engine)
    ax.set_xscale("log", base=2)
    ax.set_xticks([r["m"] for r in rows if r["engine"] == "KMP"])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("pattern length m (log$_2$ scale)")
    ax.set_ylabel("search time (ms)")
    first = rows[0]
    ax.set_title(f"Worst case for naive search: {first['n_lines']} lines of "
                 f"'a'×{first['line_length']}, pattern 'a'×(m−1)+'b'", fontsize=10)
    ax.legend()
    return save(fig, plt, out_dir, "kmp_adversarial.png")


PLOTS = {
    "per_pattern": plot_per_pattern, "breakdown": plot_breakdown, "scaling": plot_scaling,
    "blowup": plot_blowup, "kmp_book": plot_kmp_book, "kmp_adversarial": plot_kmp_adversarial,
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark egrep.py (vs GNU egrep, naive vs KMP).")
    parser.add_argument("--file", default=os.path.join(HERE, "56667-0.txt"),
                        help="text file to search (default: 56667-0.txt next to this script)")
    parser.add_argument("--runs", type=int, default=10, help="repetitions per measurement (default 10)")
    parser.add_argument("--out", default=os.path.join(HERE, "bench_results"),
                        help="output directory for CSV and PNG files")
    parser.add_argument("--quick", action="store_true", help="3 runs and smaller inputs (for CI)")
    parser.add_argument("--only", nargs="+", choices=EXPERIMENTS, metavar="EXPERIMENT",
                        help=f"run only these experiments: {', '.join(EXPERIMENTS)}")
    parser.add_argument("--kmax", type=int, default=12, help="largest k for the blowup experiment")
    args = parser.parse_args()

    selected = args.only or EXPERIMENTS
    runs = 3 if args.quick else args.runs

    if NEEDS_BOOK & set(selected) and not os.path.isfile(args.file):
        sys.exit(f"Text file not found: {args.file}\n"
                 "Download it from https://www.gutenberg.org/files/56667/56667-0.txt")
    system_egrep = find_system_egrep()
    if NEEDS_GREP & set(selected) and system_egrep is None:
        sys.exit("Neither grep nor egrep found on PATH.")
    os.makedirs(args.out, exist_ok=True)

    print(f"Python       : {sys.version.split()[0]}")
    if system_egrep:
        print(f"System egrep : {' '.join(system_egrep)}")
    if NEEDS_BOOK & set(selected):
        print(f"Text file    : {args.file} ({os.path.getsize(args.file) / 1e3:.0f} kB)")
    print(f"Experiments  : {', '.join(selected)}")

    results = {}
    for name in selected:
        if name == "per_pattern":
            results[name] = bench_per_pattern(commands(system_egrep), args.file, runs)
        elif name == "breakdown":
            results[name] = bench_breakdown(args.file, runs)
        elif name == "scaling":
            factors = [1, 2, 4, 8] if args.quick else [1, 2, 4, 8, 16, 32]
            results[name] = bench_scaling(commands(system_egrep), args.file, runs, factors)
        elif name == "blowup":
            results[name] = bench_blowup(min(args.kmax, 10) if args.quick else args.kmax)
        elif name == "kmp_book":
            results[name] = bench_kmp_book(args.file, runs)
        elif name == "kmp_adversarial":
            n_lines = 50 if args.quick else 200
            results[name] = bench_kmp_adversarial(runs, n_lines, 500, [2, 4, 8, 16, 32, 64])
        write_csv(os.path.join(args.out, f"{name}.csv"), results[name])

    plt = get_pyplot()
    charts = []
    if plt is None:
        print("\nmatplotlib not installed: CSV files only.")
    else:
        charts = [PLOTS[name](plt, args.out, rows) for name, rows in results.items() if rows]
    print(f"\nResults written to {args.out}/ "
          f"({', '.join(os.path.basename(c) for c in charts) or 'CSV only'})")


if __name__ == "__main__":
    main()
