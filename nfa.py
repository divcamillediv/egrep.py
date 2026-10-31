"""
Etape 2 : NFA avec transitions epsilon, construit a partir de l'arbre
syntaxique.
"""
from typing import Dict, List, Optional, Set, Tuple

from syntax_tree import ALTERN, CONCAT, DOT, EPSILON, ETOILE, OPTION, PLUS, RegExTree


class NFAState:
    def __init__(self, state_id: int):
        self.id = state_id
        self.transitions: Dict[int, List["NFAState"]] = {}
        self.epsilon_transitions: List["NFAState"] = []

    def add_transition(self, symbol: int, to_state: "NFAState") -> None:
        self.transitions.setdefault(symbol, []).append(to_state)

    def add_epsilon(self, to_state: "NFAState") -> None:
        self.epsilon_transitions.append(to_state)


class NFA:
    def __init__(self):
        self.state_counter = 0
        self.states: List[NFAState] = []
        self.start: Optional[NFAState] = None
        self.accept: Optional[NFAState] = None

    def create_state(self) -> NFAState:
        s = NFAState(self.state_counter)
        self.state_counter += 1
        self.states.append(s)
        return s

    @classmethod
    def from_tree(cls, tree: RegExTree) -> "NFA":
        nfa = cls()
        start, accept = nfa._build(tree)
        nfa.start = start
        nfa.accept = accept
        return nfa

    def _build(self, tree: RegExTree) -> Tuple[NFAState, NFAState]:
        if tree.root == EPSILON:
            s = self.create_state()
            e = self.create_state()
            s.add_epsilon(e)
            return s, e

        if not tree.sub_trees:
            s = self.create_state()
            e = self.create_state()
            s.add_transition(tree.root, e)
            return s, e

        if tree.root == CONCAT:
            l_start, l_accept = self._build(tree.sub_trees[0])
            r_start, r_accept = self._build(tree.sub_trees[1])
            l_accept.add_epsilon(r_start)
            return l_start, r_accept

        if tree.root == ALTERN:
            l_start, l_accept = self._build(tree.sub_trees[0])
            r_start, r_accept = self._build(tree.sub_trees[1])
            s = self.create_state()
            e = self.create_state()
            s.add_epsilon(l_start)
            s.add_epsilon(r_start)
            l_accept.add_epsilon(e)
            r_accept.add_epsilon(e)
            return s, e

        if tree.root == ETOILE:
            c_start, c_accept = self._build(tree.sub_trees[0])
            s = self.create_state()
            e = self.create_state()
            s.add_epsilon(c_start)
            s.add_epsilon(e)
            c_accept.add_epsilon(c_start)
            c_accept.add_epsilon(e)
            return s, e

        if tree.root == PLUS:
            # Comme l'etoile, mais sans le raccourci s -> e : au moins un passage
            c_start, c_accept = self._build(tree.sub_trees[0])
            s = self.create_state()
            e = self.create_state()
            s.add_epsilon(c_start)
            c_accept.add_epsilon(c_start)
            c_accept.add_epsilon(e)
            return s, e

        if tree.root == OPTION:
            # Comme l'etoile, mais sans la boucle de retour : au plus un passage
            c_start, c_accept = self._build(tree.sub_trees[0])
            s = self.create_state()
            e = self.create_state()
            s.add_epsilon(c_start)
            s.add_epsilon(e)
            c_accept.add_epsilon(e)
            return s, e

        raise ValueError(f"Unknown tree root node: {tree.root}")

    def collect_explicit_characters(self) -> Set[int]:
        chars: Set[int] = set()
        for s in self.states:
            for sym in s.transitions.keys():
                if sym != DOT:
                    chars.add(sym)
        return chars
