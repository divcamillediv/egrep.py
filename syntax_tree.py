"""
Etape 1 : arbre syntaxique d'une expression reguliere (ERE simplifiee).

Contient les codes des noeuds speciaux, la classe RegExTree, le parseur a
descente recursive et make_search_tree (prefixe .* pour la recherche).
"""
from typing import List, Optional

# Tous les codes speciaux sont > 0x10FFFF (plus grand point de code Unicode),
# donc ils ne peuvent jamais entrer en collision avec un caractere litteral.
CONCAT = 0xC04CA7
ETOILE = 0xE7011E
ALTERN = 0xA17E54
PLUS = 0x7105E5      # x+  : une ou plusieurs fois
OPTION = 0x9E5710    # x?  : zero ou une fois
DOT = 0xD07D07       # .   : caractere universel
EPSILON = 0xE95110   # mot vide (ex: "", "()", "a|")

_QUANTIFIERS = {"*": ETOILE, "+": PLUS, "?": OPTION}


class RegExTree:
    def __init__(self, root: int, sub_trees: Optional[List["RegExTree"]] = None):
        self.root = root
        self.sub_trees: List["RegExTree"] = sub_trees if sub_trees is not None else []

    def __repr__(self) -> str:
        if not self.sub_trees:
            return self._root_to_str()
        children = ",".join(repr(child) for child in self.sub_trees)
        return f"{self._root_to_str()}({children})"

    def _root_to_str(self) -> str:
        names = {CONCAT: "CONCAT", ETOILE: "*", ALTERN: "|", PLUS: "+",
                 OPTION: "?", DOT: ".", EPSILON: "EPS"}
        return names.get(self.root) or chr(self.root)


class _Parser:
    """
    Parseur a descente recursive. Grammaire (priorite croissante) :
        alternation := concat ('|' concat)*
        concat      := repeat*                (vide -> EPSILON)
        repeat      := atom ('*' | '+' | '?')*
        atom        := '(' alternation ')' | '.' | '\\' char | char
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.pos = 0
        self.depth = 0  # nombre de '(' ouvertes non encore fermees

    def _peek(self) -> Optional[str]:
        return self.pattern[self.pos] if self.pos < len(self.pattern) else None

    def parse(self) -> RegExTree:
        return self._parse_alternation()

    def _parse_alternation(self) -> RegExTree:
        left = self._parse_concat()
        while self._peek() == "|":
            self.pos += 1
            right = self._parse_concat()
            left = RegExTree(ALTERN, [left, right])
        return left

    def _parse_concat(self) -> RegExTree:
        items: List[RegExTree] = []
        while self._peek() is not None and self._peek() != "|" \
                and not (self._peek() == ")" and self.depth > 0):
            items.append(self._parse_repeat())
        if not items:
            return RegExTree(EPSILON)
        tree = items[0]
        for item in items[1:]:
            tree = RegExTree(CONCAT, [tree, item])
        return tree

    def _parse_repeat(self) -> RegExTree:
        atom = self._parse_atom()
        while self._peek() is not None and self._peek() in _QUANTIFIERS:
            atom = RegExTree(_QUANTIFIERS[self._peek()], [atom])
            self.pos += 1
        return atom

    def _parse_atom(self) -> RegExTree:
        c = self.pattern[self.pos]
        self.pos += 1
        if c == "(":
            self.depth += 1
            inner = self._parse_alternation()
            if self._peek() != ")":
                raise ValueError("Unmatched ( or \\(")
            self.pos += 1
            self.depth -= 1
            return inner
        if c == ".":
            return RegExTree(DOT)
        if c == "\\":
            if self.pos >= len(self.pattern):
                raise ValueError("Trailing backslash")
            c = self.pattern[self.pos]
            self.pos += 1
        elif c in _QUANTIFIERS:
            # Quantificateur sans operande (ex: "*a") : ignore, comme GNU egrep.
            return RegExTree(EPSILON)
        # Note : une ')' sans '(' correspondante arrive ici et est litterale,
        # comme dans GNU egrep.
        return RegExTree(ord(c))


def parse_regex(reg_ex: str) -> RegExTree:
    return _Parser(reg_ex).parse()


def make_search_tree(tree: RegExTree) -> RegExTree:
    """Prefixe .* devant l'arbre syntaxique pour la recherche de sous-motifs."""
    dot = RegExTree(DOT, [])
    dot_star = RegExTree(ETOILE, [dot])
    return RegExTree(CONCAT, [dot_star, tree])
