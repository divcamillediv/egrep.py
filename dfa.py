"""
Etapes 3, 4 et 5 : determinisation du NFA (methode des sous-ensembles),
minimisation (algorithme de Moore) et reconnaissance d'une ligne.
"""
from collections import deque
from typing import Dict, List, Set, Tuple

from alphabet import Alphabet
from nfa import NFA, NFAState
from syntax_tree import DOT


class DFA:
    def __init__(self, state_count: int, start_state: int, is_accept: List[bool], transitions: List[List[int]]):
        self.state_count = state_count
        self.start_state = start_state
        self.is_accept = is_accept
        self.transitions = transitions

    @classmethod
    def determinize(cls, nfa: NFA, alphabet: Alphabet) -> "DFA":
        def epsilon_closure(states: Set[NFAState]) -> Set[NFAState]:
            closure = set(states)
            stack = list(states)
            while stack:
                curr = stack.pop()
                for nxt in curr.epsilon_transitions:
                    if nxt not in closure:
                        closure.add(nxt)
                        stack.append(nxt)
            return closure

        def move(states: Set[NFAState], sym_idx: int) -> Set[NFAState]:
            targets: Set[NFAState] = set()
            for s in states:
                for label, next_states in s.transitions.items():
                    if label == DOT:
                        if sym_idx != alphabet.newline_index:
                            targets.update(next_states)
                    elif alphabet.get_symbol_index(label) == sym_idx:
                        targets.update(next_states)
            return epsilon_closure(targets)

        start_closure = epsilon_closure({nfa.start})
        start_ids = frozenset(s.id for s in start_closure)

        subset_to_id: Dict[frozenset, int] = {start_ids: 0}
        id_to_states: Dict[frozenset, Set[NFAState]] = {start_ids: start_closure}
        queue = deque([start_ids])

        accept_list: List[bool] = []
        transitions_list: List[List[int]] = []

        while queue:
            curr_subset = queue.popleft()
            curr_states = id_to_states[curr_subset]

            accept_list.append(any(s == nfa.accept for s in curr_states))
            row = [-1] * alphabet.size

            for sym_idx in range(alphabet.size):
                next_states = move(curr_states, sym_idx)
                if next_states:
                    next_ids = frozenset(s.id for s in next_states)
                    if next_ids not in subset_to_id:
                        new_id = len(subset_to_id)
                        subset_to_id[next_ids] = new_id
                        id_to_states[next_ids] = next_states
                        queue.append(next_ids)
                    row[sym_idx] = subset_to_id[next_ids]

            transitions_list.append(row)

        return cls(len(subset_to_id), 0, accept_list, transitions_list)

    @classmethod
    def minimize(cls, dfa: "DFA", alphabet: Alphabet) -> "DFA":
        if dfa.state_count <= 1:
            return dfa

        accept_group = [i for i in range(dfa.state_count) if dfa.is_accept[i]]
        non_accept_group = [i for i in range(dfa.state_count) if not dfa.is_accept[i]]

        partition: List[List[int]] = []
        if accept_group:
            partition.append(accept_group)
        if non_accept_group:
            partition.append(non_accept_group)

        group_of = [0] * dfa.state_count
        for g, group in enumerate(partition):
            for s in group:
                group_of[s] = g

        changed = True
        while changed:
            changed = False
            new_partition: List[List[int]] = []

            for group in partition:
                if len(group) <= 1:
                    new_partition.append(group)
                    continue

                subgroups: Dict[Tuple[int, ...], List[int]] = {}
                for s in group:
                    sig = tuple(
                        group_of[dfa.transitions[s][a]] if dfa.transitions[s][a] != -1 else -1
                        for a in range(alphabet.size)
                    )
                    subgroups.setdefault(sig, []).append(s)

                if len(subgroups) > 1:
                    changed = True
                new_partition.extend(subgroups.values())

            partition = new_partition
            for g, group in enumerate(partition):
                for s in group:
                    group_of[s] = g

        min_count = len(partition)
        min_start = group_of[dfa.start_state]
        min_accept = [False] * min_count
        min_transitions = [[-1] * alphabet.size for _ in range(min_count)]

        for g, group in enumerate(partition):
            rep = group[0]
            min_accept[g] = dfa.is_accept[rep]
            for a in range(alphabet.size):
                dest = dfa.transitions[rep][a]
                min_transitions[g][a] = group_of[dest] if dest != -1 else -1

        return cls(min_count, min_start, min_accept, min_transitions)

    def accepts(self, word: str, alphabet: Alphabet) -> bool:
        """Vrai si le mot ENTIER appartient au langage de l'automate."""
        state = self.start_state
        for ch in word:
            state = self.transitions[state][alphabet.get_symbol_index(ord(ch))]
            if state == -1:
                return False
        return self.is_accept[state]

    def match_line(self, line: str, alphabet: Alphabet) -> bool:
        """
        Recherche d'un facteur : l'automate doit avoir ete construit sur
        make_search_tree(arbre), c'est-a-dire avec le prefixe .* implicite.
        On s'arrete des qu'un etat acceptant est atteint (le reste de la
        ligne n'a pas besoin d'etre lu).
        """
        state = self.start_state
        if self.is_accept[state]:
            return True
        for ch in line:
            state = self.transitions[state][alphabet.get_symbol_index(ord(ch))]
            if state == -1:
                return False
            if self.is_accept[state]:
                return True
        return False
