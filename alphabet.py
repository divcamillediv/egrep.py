from typing import Dict, Set


class Alphabet:
    def __init__(self, explicit_chars: Set[int]):
        self.ascii_to_symbol = [-1] * 256
        self.unicode_to_symbol: Dict[int, int] = {}
        index = 0

        for ch in sorted(explicit_chars):
            if 0 <= ch < 256:
                self.ascii_to_symbol[ch] = index
            else:
                self.unicode_to_symbol[ch] = index
            index += 1

        # Caractere de saut de ligne '\n' (distinct car '.' n'y correspond pas)
        newline_ord = ord("\n")
        if self.ascii_to_symbol[newline_ord] == -1:
            self.newline_index = index
            self.ascii_to_symbol[newline_ord] = self.newline_index
            index += 1
        else:
            self.newline_index = self.ascii_to_symbol[newline_ord]

        # Classe representative pour tout autre caractere
        self.other_index = index
        for i in range(256):
            if self.ascii_to_symbol[i] == -1:
                self.ascii_to_symbol[i] = self.other_index
        index += 1

        self.size = index

    def get_symbol_index(self, ch: int) -> int:
        if 0 <= ch < 256:
            return self.ascii_to_symbol[ch]
        return self.unicode_to_symbol.get(ch, self.other_index)
