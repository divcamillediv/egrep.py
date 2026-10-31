"""
Strategie alternative pour les motifs litteraux : Knuth-Morris-Pratt (KMP).

Le module contient aussi l'algorithme naif,
qui sert uniquement de point de comparaison dans les tests de performance.
"""
from typing import List, Optional

from syntax_tree import CONCAT, RegExTree

MAX_CODE_POINT = 0x10FFFF  # au-dela : codes speciaux (CONCAT, DOT, EPSILON...)


def literal_of(tree: RegExTree) -> Optional[str]:
    chars: List[str] = []
    stack = [tree]
    while stack:
        node = stack.pop()
        if node.root == CONCAT:
            stack.append(node.sub_trees[1])  # empile la droite d'abord :
            stack.append(node.sub_trees[0])  # la gauche est traitee en premier
        elif not node.sub_trees and node.root <= MAX_CODE_POINT:
            chars.append(chr(node.root))
        else:
            return None
    return "".join(chars)


def failure_table(pattern: str) -> List[int]:
    fail = [0] * len(pattern)
    k = 0
    for i in range(1, len(pattern)):
        while k > 0 and pattern[i] != pattern[k]:
            k = fail[k - 1]
        if pattern[i] == pattern[k]:
            k += 1
        fail[i] = k
    return fail


class KMP:
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.fail = failure_table(pattern)

    def find(self, text: str) -> int:
        pattern, fail = self.pattern, self.fail
        m = len(pattern)
        if m == 0:
            return 0
        j = 0  # nombre de caracteres du motif deja reconnus
        for i, c in enumerate(text):
            while j > 0 and c != pattern[j]:
                j = fail[j - 1]
            if c == pattern[j]:
                j += 1
                if j == m:
                    return i - m + 1
        return -1

    def match_line(self, line: str) -> bool:
        return self.find(line) != -1

"""
Algorithme naif pour les comparaisons de performance.
"""

def naive_find(text: str, pattern: str) -> int:
    n, m = len(text), len(pattern)
    for i in range(n - m + 1):
        j = 0
        while j < m and text[i + j] == pattern[j]:
            j += 1
        if j == m:
            return i
    return -1

class Naive:
    def __init__(self, pattern: str):
        self.pattern = pattern

    def find(self, text: str) -> int:
        return naive_find(text, self.pattern)

    def match_line(self, line: str) -> bool:
        return naive_find(line, self.pattern) != -1
