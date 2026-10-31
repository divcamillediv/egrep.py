#!/usr/bin/env python3
"""
Clone de la commande Linux egrep en Python : point d'entree en ligne de commande.

    python egrep.py [--stats] PATTERN [FILE...]

Deux strategies de recherche, choisies automatiquement d'apres le motif :

  * motif litteral (simple concatenation de caracteres, ex: "Babylon")
      -> KMP (kmp.py), sans construire d'automate ;
  * tout autre motif (|, *, +, ?, ., parentheses...)
      -> chaine du cours, chaque etape dans son propre module :
         1. syntax_tree.py : arbre syntaxique (parseur)
         2. nfa.py         : NFA avec transitions epsilon (Thompson / Aho-Ullman)
         3. dfa.py         : determinisation (methode des sous-ensembles)
         4. dfa.py         : minimisation (algorithme de Moore)
         5. dfa.py         : reconnaissance ligne par ligne
            alphabet.py    : classes d'equivalence des caracteres

Option : --stats affiche sur stderr la strategie utilisee, la taille des
automates et les temps de construction et de recherche.

Codes de sortie (comme egrep) : 0 si au moins une ligne correspond,
1 sinon, 2 en cas d'erreur (motif invalide, option inconnue...).
Edge case : egrep "" affiche le texte en entier.
"""

import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

from alphabet import Alphabet
from dfa import DFA
from kmp import KMP, literal_of
from nfa import NFA
from syntax_tree import make_search_tree, parse_regex

USAGE = "Usage: python egrep.py [--stats] PATTERN [FILE...]"

Matcher = Callable[[str], bool]


class UsageError(Exception):
    pass


def build_matcher(pattern: str) -> Tuple[Matcher, Dict[str, object]]:
    """
    Compile le motif et renvoie (match, infos) : match(ligne) -> bool, et un
    dictionnaire decrivant la strategie choisie ("engine" vaut "KMP" ou "DFA").
    """
    tree = parse_regex(pattern)

    literal = literal_of(tree)
    if literal is not None:
        kmp = KMP(literal)
        return kmp.match_line, {"engine": "KMP", "pattern_length": len(literal)}

    nfa = NFA.from_tree(make_search_tree(tree))
    alphabet = Alphabet(nfa.collect_explicit_characters())
    dfa = DFA.determinize(nfa, alphabet)
    min_dfa = DFA.minimize(dfa, alphabet)

    def match(line: str) -> bool:
        return min_dfa.match_line(line, alphabet)

    return match, {
        "engine": "DFA",
        "nfa_states": len(nfa.states),
        "dfa_states": dfa.state_count,
        "min_dfa_states": min_dfa.state_count,
    }


def process_stream(stream, prefix: Optional[str], match: Matcher) -> int:
    """Affiche les lignes qui correspondent (prefixees par le nom du fichier
    s'il y en a plusieurs) et renvoie leur nombre."""
    match_count = 0
    for line in stream:
        line = line.rstrip("\r\n")
        if match(line):
            match_count += 1
            print(line if prefix is None else f"{prefix}:{line}")
    return match_count


def parse_args(args: List[str]) -> Tuple[bool, str, List[str]]:
    """Renvoie (stats, motif, fichiers). '--' termine les options, ce qui
    permet de chercher un motif commencant par '-'."""
    stats = False
    positional: List[str] = []
    options_done = False
    for arg in args:
        if not options_done and arg == "--":
            options_done = True
        elif not options_done and arg == "--stats":
            stats = True
        elif not options_done and arg.startswith("-") and arg != "-":
            raise UsageError(f"unrecognized option '{arg}'")
        else:
            positional.append(arg)
    if not positional:
        raise UsageError("missing PATTERN")
    return stats, positional[0], positional[1:]


def format_stats(info: Dict[str, object], build_s: float, search_s: float) -> str:
    if info["engine"] == "KMP":
        details = f"pattern length: {info['pattern_length']}"
    else:
        details = (f"NFA states: {info['nfa_states']} | DFA states: {info['dfa_states']} | "
                   f"Min-DFA states: {info['min_dfa_states']}")
    return (f"[Stats] engine: {info['engine']} | {details} | "
            f"build: {build_s * 1000:.2f} ms | search: {search_s * 1000:.2f} ms\n")


def main() -> None:
    try:
        stats, pattern, files = parse_args(sys.argv[1:])
    except UsageError as e:
        sys.stderr.write(f"egrep: {e}\n{USAGE}\n")
        sys.exit(2)

    try:
        t0 = time.perf_counter()
        match, info = build_matcher(pattern)
        t1 = time.perf_counter()

        total_matches = 0
        if not files:
            total_matches += process_stream(sys.stdin, None, match)
        for path in files:
            if not os.path.isfile(path):
                sys.stderr.write(f"egrep: {path}: No such file or permission denied\n")
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    prefix = path if len(files) > 1 else None
                    total_matches += process_stream(f, prefix, match)
            except OSError as err:
                sys.stderr.write(f"egrep: {path}: {err}\n")
        t2 = time.perf_counter()

        if stats:
            sys.stderr.write(format_stats(info, t1 - t0, t2 - t1))
    except Exception as e:
        sys.stderr.write(f"egrep: error: {e}\n")
        sys.exit(2)

    sys.exit(0 if total_matches > 0 else 1)


if __name__ == "__main__":
    main()
