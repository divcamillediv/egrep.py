"""
Unit tests for egrep.py and its modules.

Run:       python -m unittest -v test_egrep
Coverage:  python -m coverage run --branch -m unittest test_egrep
           python -m coverage report -m --include="egrep.py,syntax_tree.py,alphabet.py,nfa.py,dfa.py,kmp.py"
"""
import io
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import egrep
from alphabet import Alphabet
from dfa import DFA
from egrep import UsageError, build_matcher, format_stats, parse_args, process_stream
from kmp import KMP, Naive, failure_table, literal_of, naive_find
from nfa import NFA
from syntax_tree import (
    ALTERN,
    CONCAT,
    DOT,
    EPSILON,
    ETOILE,
    OPTION,
    PLUS,
    RegExTree,
    make_search_tree,
    parse_regex,
)

HERE = os.path.dirname(os.path.abspath(__file__))
EGREP_PY = os.path.join(HERE, "egrep.py")
GUTENBERG = os.path.join(HERE, "56667-0.txt")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def build_dfa(pattern: str, search: bool):
    """tree -> NFA -> DFA -> min-DFA. With search=True, the .* prefix is added."""
    tree = parse_regex(pattern)
    if search:
        tree = make_search_tree(tree)
    nfa = NFA.from_tree(tree)
    alphabet = Alphabet(nfa.collect_explicit_characters())
    dfa = DFA.determinize(nfa, alphabet)
    return alphabet, dfa, DFA.minimize(dfa, alphabet)


def accepts(pattern: str, word: str) -> bool:
    """Does the WHOLE word belong to L(pattern)? (automaton without .* prefix)"""
    alphabet, _, dfa = build_dfa(pattern, search=False)
    return dfa.accepts(word, alphabet)


def matches(pattern: str, line: str) -> bool:
    """Does the line CONTAIN a factor in L(pattern)? (what egrep prints)"""
    match, _ = build_matcher(pattern)
    return match(line)


def run_main(argv, stdin_text=""):
    """Run egrep.main() in-process. Returns (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "argv", ["egrep.py"] + argv), \
         mock.patch.object(sys, "stdin", io.StringIO(stdin_text)), \
         redirect_stdout(out), redirect_stderr(err):
        try:
            egrep.main()
        except SystemExit as e:
            return e.code, out.getvalue(), err.getvalue()
    raise AssertionError("main() returned without calling sys.exit()")


# -----------------------------------------------------------------------------
# Step 1: parser and syntax tree
# -----------------------------------------------------------------------------
class TestParserAtoms(unittest.TestCase):
    def test_single_character(self):
        tree = parse_regex("a")
        self.assertEqual(tree.root, ord("a"))
        self.assertEqual(tree.sub_trees, [])

    def test_dot(self):
        self.assertEqual(parse_regex(".").root, DOT)

    def test_empty_pattern_is_epsilon(self):
        self.assertEqual(parse_regex("").root, EPSILON)

    def test_empty_group_is_epsilon(self):
        self.assertEqual(parse_regex("()").root, EPSILON)

    def test_escaped_metacharacters_are_literal(self):
        for meta in "()|*+?.\\":
            with self.subTest(meta=meta):
                tree = parse_regex("\\" + meta)
                self.assertEqual(tree.root, ord(meta))
                self.assertEqual(tree.sub_trees, [])

    def test_escaped_ordinary_character(self):
        self.assertEqual(parse_regex("\\a").root, ord("a"))

    def test_non_ascii_character(self):
        self.assertEqual(parse_regex("é").root, ord("é"))

    def test_special_codes_cannot_collide_with_unicode(self):
        codes = {CONCAT, ETOILE, ALTERN, PLUS, OPTION, DOT, EPSILON}
        self.assertEqual(len(codes), 7)
        for code in codes:
            self.assertGreater(code, 0x10FFFF)


class TestParserOperators(unittest.TestCase):
    def test_concatenation_is_left_associative(self):
        self.assertEqual(repr(parse_regex("abc")), "CONCAT(CONCAT(a,b),c)")

    def test_quantifiers(self):
        self.assertEqual(repr(parse_regex("a*")), "*(a)")
        self.assertEqual(repr(parse_regex("a+")), "+(a)")
        self.assertEqual(repr(parse_regex("a?")), "?(a)")

    def test_stacked_quantifiers(self):
        self.assertEqual(repr(parse_regex("a*+?")), "?(+(*(a)))")

    def test_quantifier_binds_tighter_than_concat(self):
        self.assertEqual(repr(parse_regex("ab*")), "CONCAT(a,*(b))")

    def test_alternation_three_branches(self):
        # Regression: this used to raise "Invalid alternation usage".
        self.assertEqual(repr(parse_regex("a|g|r")), "|(|(a,g),r)")

    def test_concat_binds_tighter_than_alternation(self):
        self.assertEqual(repr(parse_regex("ab|cd")), "|(CONCAT(a,b),CONCAT(c,d))")

    def test_empty_alternation_branches(self):
        self.assertEqual(repr(parse_regex("a|")), "|(a,EPS)")
        self.assertEqual(repr(parse_regex("|a")), "|(EPS,a)")


class TestParserParentheses(unittest.TestCase):
    def test_quantifier_on_group(self):
        self.assertEqual(repr(parse_regex("(ab)*")), "*(CONCAT(a,b))")

    def test_assignment_example(self):
        self.assertEqual(repr(parse_regex("S(a|g|r)+on")),
                         "CONCAT(CONCAT(CONCAT(S,+(|(|(a,g),r))),o),n)")

    def test_nested_groups(self):
        self.assertEqual(repr(parse_regex("((a|e)(r|n))+")), "+(CONCAT(|(a,e),|(r,n)))")

    def test_redundant_parentheses_are_transparent(self):
        self.assertEqual(repr(parse_regex("((a))")), "a")


class TestParserEdgeCasesAndErrors(unittest.TestCase):
    def test_unmatched_open_parenthesis(self):
        for pattern in ("(a|b", "(", "((a)", "a(b"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    parse_regex(pattern)

    def test_trailing_backslash(self):
        with self.assertRaises(ValueError):
            parse_regex("ab\\")

    def test_stray_closing_parenthesis_is_literal(self):
        # GNU egrep treats an unmatched ')' as a literal character.
        self.assertEqual(repr(parse_regex("a)")), "CONCAT(a,))")

    def test_leading_quantifier_is_ignored(self):
        # GNU egrep ignores a quantifier with no operand ("*a" == "a").
        for q in "*+?":
            with self.subTest(q=q):
                self.assertEqual(repr(parse_regex(q + "a")), "CONCAT(EPS,a)")


class TestRegExTree(unittest.TestCase):
    def test_repr_of_all_node_kinds(self):
        a = RegExTree(ord("a"))
        self.assertEqual(repr(RegExTree(DOT)), ".")
        self.assertEqual(repr(RegExTree(EPSILON)), "EPS")
        self.assertEqual(repr(RegExTree(ETOILE, [a])), "*(a)")
        self.assertEqual(repr(RegExTree(PLUS, [a])), "+(a)")
        self.assertEqual(repr(RegExTree(OPTION, [a])), "?(a)")
        self.assertEqual(repr(RegExTree(CONCAT, [a, a])), "CONCAT(a,a)")
        self.assertEqual(repr(RegExTree(ALTERN, [a, a])), "|(a,a)")

    def test_default_sub_trees_are_not_shared(self):
        t1, t2 = RegExTree(1), RegExTree(2)
        t1.sub_trees.append(t2)
        self.assertEqual(t2.sub_trees, [])

    def test_make_search_tree_prefixes_dot_star(self):
        self.assertEqual(repr(make_search_tree(parse_regex("ab"))), "CONCAT(*(.),CONCAT(a,b))")


# -----------------------------------------------------------------------------
# Alphabet equivalence classes
# -----------------------------------------------------------------------------
class TestAlphabet(unittest.TestCase):
    def test_explicit_chars_get_distinct_indices(self):
        alpha = Alphabet({ord("a"), ord("b")})
        ia, ib = alpha.get_symbol_index(ord("a")), alpha.get_symbol_index(ord("b"))
        self.assertNotEqual(ia, ib)
        self.assertNotIn(ia, (alpha.newline_index, alpha.other_index))

    def test_other_characters_share_one_class(self):
        alpha = Alphabet({ord("a")})
        for ch in ("z", "1", "中"):
            self.assertEqual(alpha.get_symbol_index(ord(ch)), alpha.other_index)

    def test_newline_has_its_own_class(self):
        alpha = Alphabet({ord("a")})
        self.assertEqual(alpha.get_symbol_index(ord("\n")), alpha.newline_index)
        self.assertNotEqual(alpha.newline_index, alpha.other_index)
        self.assertEqual(alpha.size, 3)  # a, \n, other

    def test_explicit_newline_is_reused(self):
        alpha = Alphabet({ord("\n"), ord("a")})
        self.assertEqual(alpha.newline_index, alpha.get_symbol_index(ord("\n")))
        self.assertEqual(alpha.size, 3)

    def test_explicit_non_ascii_character(self):
        alpha = Alphabet({ord("é"), 0x4E2D})
        self.assertNotEqual(alpha.get_symbol_index(ord("é")), alpha.other_index)
        self.assertNotEqual(alpha.get_symbol_index(0x4E2D), alpha.other_index)
        self.assertEqual(alpha.size, 4)

    def test_empty_alphabet(self):
        self.assertEqual(Alphabet(set()).size, 2)  # \n, other


# -----------------------------------------------------------------------------
# Step 2: Thompson NFA
# -----------------------------------------------------------------------------
class TestNFA(unittest.TestCase):
    def test_state_counts_follow_thompson_construction(self):
        expected = {"a": 2, "": 2, ".": 2, "ab": 4, "a|b": 6, "a*": 4, "a+": 4, "a?": 4}
        for pattern, n in expected.items():
            with self.subTest(pattern=pattern):
                self.assertEqual(len(NFA.from_tree(parse_regex(pattern)).states), n)

    def test_state_ids_are_sequential(self):
        nfa = NFA.from_tree(parse_regex("(a|b)*c"))
        self.assertEqual([s.id for s in nfa.states], list(range(len(nfa.states))))

    def test_epsilon_leaf_has_only_an_epsilon_edge(self):
        nfa = NFA.from_tree(parse_regex(""))
        self.assertEqual(nfa.start.transitions, {})
        self.assertEqual(nfa.start.epsilon_transitions, [nfa.accept])

    def test_plus_has_no_bypass_edge(self):
        nfa = NFA.from_tree(parse_regex("a+"))
        self.assertNotIn(nfa.accept, nfa.start.epsilon_transitions)

    def test_star_and_option_have_bypass_edge(self):
        for pattern in ("a*", "a?"):
            with self.subTest(pattern=pattern):
                nfa = NFA.from_tree(parse_regex(pattern))
                self.assertIn(nfa.accept, nfa.start.epsilon_transitions)

    def test_collect_explicit_characters_excludes_dot(self):
        nfa = NFA.from_tree(parse_regex("a.b|c"))
        self.assertEqual(nfa.collect_explicit_characters(), {ord("a"), ord("b"), ord("c")})

    def test_unknown_root_raises(self):
        with self.assertRaises(ValueError):
            NFA.from_tree(RegExTree(-42, [RegExTree(ord("a"))]))


# -----------------------------------------------------------------------------
# Steps 3 & 4: determinization and minimization
# -----------------------------------------------------------------------------
class TestDFAConstruction(unittest.TestCase):
    def test_determinize_table_shape(self):
        alphabet, dfa, _ = build_dfa("(a|b)*c", search=False)
        self.assertEqual(dfa.start_state, 0)
        self.assertEqual(len(dfa.transitions), dfa.state_count)
        self.assertEqual(len(dfa.is_accept), dfa.state_count)
        for row in dfa.transitions:
            self.assertEqual(len(row), alphabet.size)
            for dest in row:
                self.assertTrue(dest == -1 or 0 <= dest < dfa.state_count)

    def test_minimal_state_counts(self):
        # Minimal DFAs (without the implicit dead state).
        expected = {"a": 2, "ab": 3, "a*": 1, "a+": 2, "a?": 2, "(a|b)*c": 2, "": 1}
        for pattern, n in expected.items():
            with self.subTest(pattern=pattern):
                self.assertEqual(build_dfa(pattern, search=False)[2].state_count, n)

    def test_minimization_merges_equivalent_states(self):
        _, dfa, min_dfa = build_dfa("ab|cb", search=False)
        self.assertLess(min_dfa.state_count, dfa.state_count)

    def test_minimize_single_state_returns_same_object(self):
        _, dfa, min_dfa = build_dfa("", search=False)
        self.assertEqual(dfa.state_count, 1)
        self.assertIs(min_dfa, dfa)

    def test_minimize_all_or_none_accepting(self):
        alphabet = Alphabet(set())
        for accept in (True, False):
            with self.subTest(accept=accept):
                dfa = DFA(2, 0, [accept, accept], [[1, -1], [0, -1]])
                self.assertEqual(DFA.minimize(dfa, alphabet).state_count, 1)

    def test_minimization_preserves_language(self):
        words = ["", "a", "b", "ab", "ba", "abc", "abab", "aabb", "c", "bc", "abcc"]
        for pattern in ("(a|b)*c", "a+b?", "(ab)*", "a|bc*", "(a|b|c)+"):
            alphabet, dfa, min_dfa = build_dfa(pattern, search=False)
            for w in words:
                with self.subTest(pattern=pattern, word=w):
                    self.assertEqual(dfa.accepts(w, alphabet), min_dfa.accepts(w, alphabet))

    def test_blowup_family_needs_exponentially_many_states(self):
        # (a|b)*a(a|b)^k: NFA linear in k, minimal DFA exactly 2^(k+1) states.
        for k in range(1, 6):
            with self.subTest(k=k):
                nfa = NFA.from_tree(parse_regex("(a|b)*a" + "(a|b)" * k))
                self.assertEqual(len(nfa.states), 6 * k + 10)
                _, _, min_dfa = build_dfa("(a|b)*a" + "(a|b)" * k, search=False)
                self.assertEqual(min_dfa.state_count, 2 ** (k + 1))


# -----------------------------------------------------------------------------
# Step 5: DFA language (whole word) and line search
# -----------------------------------------------------------------------------
class TestDFAAccepts(unittest.TestCase):
    def test_exact_language(self):
        for w in ("ad", "abd", "acd", "abbcbcd"):
            self.assertTrue(accepts("a(b|c)*d", w), w)
        for w in ("a", "adx", "xad", ""):
            self.assertFalse(accepts("a(b|c)*d", w), w)

    def test_aho_ullman_example(self):
        for w in ("a", "b", "bc", "bccc"):
            self.assertTrue(accepts("a|bc*", w), w)
        for w in ("ac", "c", ""):
            self.assertFalse(accepts("a|bc*", w), w)

    def test_quantifiers(self):
        self.assertTrue(accepts("a+", "aaa"))
        self.assertFalse(accepts("a+", ""))
        self.assertTrue(accepts("colou?r", "color"))
        self.assertTrue(accepts("colou?r", "colour"))
        self.assertFalse(accepts("colou?r", "colouur"))
        self.assertTrue(accepts("(ab)*", ""))
        self.assertFalse(accepts("(ab)*", "aba"))

    def test_dot(self):
        self.assertTrue(accepts("a.b", "axb"))
        self.assertTrue(accepts("a.b", "aéb"))
        self.assertFalse(accepts("a.b", "ab"))
        self.assertFalse(accepts("a.b", "a\nb"))

    def test_empty_pattern_accepts_only_empty_word(self):
        self.assertTrue(accepts("", ""))
        self.assertFalse(accepts("", "x"))


class TestDFAMatchLine(unittest.TestCase):
    def test_substring_search(self):
        alphabet, _, dfa = build_dfa("ne(e|a)dle", search=True)
        for w in ("needle", "before_needle_after", "a neadle"):
            self.assertTrue(dfa.match_line(w, alphabet), w)
        for w in ("needl", "no match here", ""):
            self.assertFalse(dfa.match_line(w, alphabet), w)

    def test_overlapping_prefix(self):
        self.assertTrue(matches("a(b|c)", "aab"))
        self.assertTrue(matches("(ab)+ab", "xababab"))

    def test_empty_like_patterns_match_every_line(self):
        for pattern in ("", "a|", "|a", "()", "(|a)", "x*"):
            with self.subTest(pattern=pattern):
                self.assertTrue(matches(pattern, "zzz"))
                self.assertTrue(matches(pattern, ""))

    def test_dead_state_stops_scanning(self):
        # Without the .* prefix the automaton can die: match_line returns False.
        alphabet, _, dfa = build_dfa("ab", search=False)
        self.assertFalse(dfa.match_line("xab", alphabet))
        self.assertTrue(dfa.match_line("abx", alphabet))

    def test_assignment_example(self):
        for w in ("Sargon of Akkad", "Saon", "the Sagon"):
            self.assertTrue(matches("S(a|g|r)+on", w), w)
        for w in ("Son", "Sxon", "sargon"):
            self.assertFalse(matches("S(a|g|r)+on", w), w)

    def test_newline_breaks_dot(self):
        self.assertFalse(matches("a.b", "a\nb"))


# -----------------------------------------------------------------------------
# KMP and naive search (kmp.py)
# -----------------------------------------------------------------------------
class TestLiteralDetection(unittest.TestCase):
    def test_literals(self):
        cases = {"a": "a", "abc": "abc", "Babylon": "Babylon", "a(bc)": "abc",
                 "(a)": "a", "((ab)c)d": "abcd", "a\\.b": "a.b", "\\(x\\|y\\)": "(x|y)",
                 "a\\*": "a*", "end)": "end)", "café": "café"}
        for pattern, literal in cases.items():
            with self.subTest(pattern=pattern):
                self.assertEqual(literal_of(parse_regex(pattern)), literal)

    def test_non_literals(self):
        for pattern in ("a.b", ".", "a|b", "a*", "ab+", "colou?r", "S(a|g|r)+on",
                        "", "()", "*ab", "a(b|c)d"):
            with self.subTest(pattern=pattern):
                self.assertIsNone(literal_of(parse_regex(pattern)))

    def test_long_literal_does_not_hit_recursion_limit(self):
        pattern = "ab" * 5000
        self.assertEqual(literal_of(parse_regex(pattern)), pattern)


class TestFailureTable(unittest.TestCase):
    def test_known_tables(self):
        cases = {
            "": [],
            "a": [0],
            "abcd": [0, 0, 0, 0],
            "aaaa": [0, 1, 2, 3],
            "ABABCABAB": [0, 0, 1, 2, 0, 1, 2, 3, 4],
            "aabaaab": [0, 1, 0, 1, 2, 2, 3],
            "mamamia": [0, 0, 1, 2, 3, 0, 0],
        }
        for pattern, table in cases.items():
            with self.subTest(pattern=pattern):
                self.assertEqual(failure_table(pattern), table)

    def test_table_matches_brute_force_definition(self):
        def longest_border(s):
            return max(k for k in range(len(s)) if s[:k] == s[len(s) - k:])
        rng = random.Random(1)
        for _ in range(200):
            p = "".join(rng.choice("ab") for _ in range(rng.randint(1, 12)))
            self.assertEqual(failure_table(p), [longest_border(p[:i + 1]) for i in range(len(p))], p)


class TestStringSearch(unittest.TestCase):
    ENGINES = {"KMP": lambda p: KMP(p).find, "naive": lambda p: (lambda t: naive_find(t, p))}

    def test_examples(self):
        cases = [("abcabd", "abd", 3), ("aaaa", "aaa", 0), ("abababc", "ababc", 2),
                 ("hello", "xyz", -1), ("ab", "abc", -1), ("", "a", -1),
                 ("abc", "", 0), ("", "", 0), ("café au lait", "é au", 3),
                 ("aaaaaaaab", "aaab", 5), ("Sargon of Akkad", "Akkad", 10)]
        for name, make in self.ENGINES.items():
            for text, pattern, expected in cases:
                with self.subTest(engine=name, text=text, pattern=pattern):
                    self.assertEqual(make(pattern)(text), expected)

    def test_random_against_str_find(self):
        rng = random.Random(42)
        for _ in range(1000):
            text = "".join(rng.choice("abc") for _ in range(rng.randint(0, 40)))
            pattern = "".join(rng.choice("abc") for _ in range(rng.randint(1, 6)))
            for name, make in self.ENGINES.items():
                self.assertEqual(make(pattern)(text), text.find(pattern),
                                 f"{name}: {pattern!r} in {text!r}")

    def test_match_line_interface(self):
        for cls in (KMP, Naive):
            with self.subTest(cls=cls.__name__):
                m = cls("needle")
                self.assertEqual(m.pattern, "needle")
                self.assertTrue(m.match_line("a needle here"))
                self.assertFalse(m.match_line("a needl here"))
                self.assertEqual(m.find("xxneedle"), 2)


# -----------------------------------------------------------------------------
# egrep.py: engine selection, output, argument parsing
# -----------------------------------------------------------------------------
class TestBuildMatcher(unittest.TestCase):
    def test_literal_uses_kmp(self):
        match, info = build_matcher("Babylon")
        self.assertEqual(info, {"engine": "KMP", "pattern_length": 7})
        self.assertTrue(match("old Babylon"))
        self.assertFalse(match("Babylo"))

    def test_regex_uses_dfa(self):
        match, info = build_matcher("S(a|g|r)+on")
        self.assertEqual(info["engine"], "DFA")
        self.assertEqual((info["nfa_states"], info["dfa_states"], info["min_dfa_states"]), (22, 8, 5))
        self.assertTrue(match("Sargon"))

    def test_both_engines_agree(self):
        # Same literal through KMP and through the automaton.
        lines = ["Babylon", "Babylonia", "babylon", "", "xBabylonx", "Baby lon"]
        kmp_match, _ = build_matcher("Babylon")
        alphabet, _, dfa = build_dfa("Babylon", search=True)
        for line in lines:
            self.assertEqual(kmp_match(line), dfa.match_line(line, alphabet), line)

    def test_invalid_pattern_raises(self):
        with self.assertRaises(ValueError):
            build_matcher("(a|b")


class TestProcessStream(unittest.TestCase):
    TEXT = "alpha\nbeta\ngamma\nalphabeta\ndelta\n"

    def _run(self, pattern, text, prefix=None):
        match, _ = build_matcher(pattern)
        out = io.StringIO()
        with redirect_stdout(out):
            count = process_stream(io.StringIO(text), prefix, match)
        return count, out.getvalue()

    def test_prints_matching_lines(self):
        self.assertEqual(self._run("alpha", self.TEXT), (2, "alpha\nalphabeta\n"))

    def test_prefix(self):
        self.assertEqual(self._run("lta", self.TEXT, "f.txt"), (1, "f.txt:delta\n"))

    def test_crlf_line_endings_are_stripped(self):
        self.assertEqual(self._run("eta", "beta\r\nalpha\r\n"), (1, "beta\n"))

    def test_last_line_without_newline(self):
        self.assertEqual(self._run("alpha", "xx\nalpha"), (1, "alpha\n"))

    def test_empty_stream(self):
        self.assertEqual(self._run("a", ""), (0, ""))


class TestParseArgs(unittest.TestCase):
    def test_pattern_and_files(self):
        self.assertEqual(parse_args(["p"]), (False, "p", []))
        self.assertEqual(parse_args(["p", "f1", "f2"]), (False, "p", ["f1", "f2"]))

    def test_stats_anywhere_before_double_dash(self):
        self.assertEqual(parse_args(["--stats", "p"]), (True, "p", []))
        self.assertEqual(parse_args(["p", "--stats", "f"]), (True, "p", ["f"]))

    def test_double_dash_ends_options(self):
        self.assertEqual(parse_args(["--", "-n"]), (False, "-n", []))
        self.assertEqual(parse_args(["--", "--stats"]), (False, "--stats", []))

    def test_single_dash_is_positional(self):
        self.assertEqual(parse_args(["-"]), (False, "-", []))

    def test_removed_options_are_rejected(self):
        for opt in ("-n", "-c", "-v", "-x", "-h", "--help", "--count"):
            with self.subTest(opt=opt):
                with self.assertRaises(UsageError):
                    parse_args([opt, "p"])

    def test_missing_pattern(self):
        for args in ([], ["--stats"], ["--"]):
            with self.assertRaises(UsageError):
                parse_args(args)


class TestFormatStats(unittest.TestCase):
    def test_kmp(self):
        line = format_stats({"engine": "KMP", "pattern_length": 7}, 0.001, 0.25)
        self.assertEqual(line, "[Stats] engine: KMP | pattern length: 7 | "
                               "build: 1.00 ms | search: 250.00 ms\n")

    def test_dfa(self):
        info = {"engine": "DFA", "nfa_states": 22, "dfa_states": 8, "min_dfa_states": 5}
        line = format_stats(info, 0.0005, 0.1)
        self.assertIn("engine: DFA", line)
        self.assertIn("NFA states: 22 | DFA states: 8 | Min-DFA states: 5", line)
        self.assertIn("build: 0.50 ms | search: 100.00 ms", line)


# -----------------------------------------------------------------------------
# CLI: main() in-process (counted by coverage)
# -----------------------------------------------------------------------------
class TestMainInProcess(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.f1 = os.path.join(self.tmp, "one.txt")
        self.f2 = os.path.join(self.tmp, "two.txt")
        with open(self.f1, "w", encoding="utf-8") as f:
            f.write("foo\nbar\nbaz\n")
        with open(self.f2, "w", encoding="utf-8") as f:
            f.write("bar\nqux\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_stdin_regex(self):
        self.assertEqual(run_main(["foo|bar"], "foo\nbaz\nbar\nqux\n")[:2], (0, "foo\nbar\n"))

    def test_stdin_literal(self):
        self.assertEqual(run_main(["ba"], "foo\nbaz\nbar\n")[:2], (0, "baz\nbar\n"))

    def test_no_match_exit_code(self):
        self.assertEqual(run_main(["notfound"], "hello\nworld\n")[:2], (1, ""))

    def test_stats_kmp(self):
        code, out, err = run_main(["--stats", "foo"], "foo\n")
        self.assertEqual((code, out), (0, "foo\n"))
        self.assertIn("engine: KMP", err)
        self.assertIn("pattern length: 3", err)

    def test_stats_dfa(self):
        code, out, err = run_main(["--stats", "fo+"], "foo\n")
        self.assertEqual((code, out), (0, "foo\n"))
        self.assertIn("engine: DFA", err)
        self.assertIn("Min-DFA states", err)

    def test_no_stats_by_default(self):
        self.assertEqual(run_main(["foo"], "foo\n")[2], "")

    def test_unknown_option(self):
        code, out, err = run_main(["-n", "foo"], "foo\n")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("unrecognized option '-n'", err)
        self.assertIn("Usage", err)

    def test_missing_pattern(self):
        code, _, err = run_main([])
        self.assertEqual(code, 2)
        self.assertIn("Usage", err)

    def test_pattern_starting_with_dash(self):
        self.assertEqual(run_main(["--", "-x"], "a-x\nb\n")[:2], (0, "a-x\n"))

    def test_single_file_has_no_prefix(self):
        self.assertEqual(run_main(["ba", self.f1])[:2], (0, "bar\nbaz\n"))

    def test_multiple_files_get_prefix(self):
        _, out, _ = run_main(["bar", self.f1, self.f2])
        self.assertEqual(out, f"{self.f1}:bar\n{self.f2}:bar\n")

    def test_missing_file_reports_and_continues(self):
        missing = os.path.join(self.tmp, "nope.txt")
        code, out, err = run_main(["foo", missing, self.f1])
        self.assertEqual(code, 0)
        self.assertIn("nope.txt", err)
        self.assertIn(f"{self.f1}:foo", out)

    def test_directory_is_rejected(self):
        code, _, err = run_main(["foo", self.tmp])
        self.assertEqual(code, 1)
        self.assertIn(self.tmp, err)

    def test_os_error_while_reading(self):
        with mock.patch("builtins.open", side_effect=OSError("boom")):
            code, _, err = run_main(["foo", self.f1])
        self.assertEqual(code, 1)
        self.assertIn("boom", err)

    def test_syntax_error_exit_code(self):
        code, _, err = run_main(["(a|b"], "a\n")
        self.assertEqual(code, 2)
        self.assertIn("Unmatched", err)

    def test_empty_pattern_prints_everything(self):
        self.assertEqual(run_main([""], "x\n\ny\n")[:2], (0, "x\n\ny\n"))


# -----------------------------------------------------------------------------
# CLI: real subprocess (end-to-end)
# -----------------------------------------------------------------------------
class TestCLISubprocess(unittest.TestCase):
    def _run(self, args, stdin=""):
        return subprocess.run([sys.executable, EGREP_PY] + args, input=stdin,
                              capture_output=True, text=True, encoding="utf-8")

    def test_assignment_example_from_stdin(self):
        proc = self._run(["S(a|g|r)+on"], "Sargon of Akkad\nnothing here\nSaon\nSon\n")
        self.assertEqual((proc.returncode, proc.stdout), (0, "Sargon of Akkad\nSaon\n"))

    def test_exit_codes(self):
        self.assertEqual(self._run(["notfound"], "hello\n").returncode, 1)
        self.assertEqual(self._run(["(a|b"], "a\n").returncode, 2)
        self.assertEqual(self._run(["-c", "a"], "a\n").returncode, 2)

    def test_stats_go_to_stderr(self):
        proc = self._run(["--stats", "Akkad"], "Sargon of Akkad\n")
        self.assertEqual(proc.stdout, "Sargon of Akkad\n")
        self.assertIn("engine: KMP", proc.stderr)


# -----------------------------------------------------------------------------
# Differential testing against the system's grep -E (skipped if unavailable)
# -----------------------------------------------------------------------------
GREP = shutil.which("grep")

SAMPLE_TEXT = "\n".join([
    "Sargon of Akkad", "Sagon", "Son", "abc", "aab", "ababab", "color", "colour",
    "", "the cat sat", "then", "than", "x(yz)", "a.b", "a|b", "Babylon", "Babbylon",
    "Assyria", "Assiria", "café", "end)", "123", "...", "aaaaaaab", "mamamia",
]) + "\n"

DIFF_PATTERNS = [
    # DFA engine
    "S(a|g|r)+on", "a|b|c", "(a|b|c|d)+e", "Ba(b|r)+y?lon", "(ab)*c", "Sar.on",
    "Ass(y|i)ria|Babylon", "((a|e)(r|n))+t", "x(yz)?", "colou?r", "th(e|a)*n",
    "", "()", "a|", "(|a)", "...", ".", "caf.", "(ab)+", "z*", "*ab",
    # KMP engine
    "Sargon", "Babylon", "a\\.b", "a\\|b", "end)", "café", "aaab", "mamamia",
    "ab", "x", "notpresent", "a(bc)",
]


@unittest.skipUnless(GREP, "grep not available")
class TestAgainstSystemGrep(unittest.TestCase):
    def _grep(self, pattern, path):
        proc = subprocess.run([GREP, "-E", "--", pattern, path], capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        return proc.stdout.replace("\r", "")

    def _ours(self, pattern, path):
        proc = subprocess.run([sys.executable, EGREP_PY, "--", pattern, path], capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        return proc.stdout

    def test_sample_text(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(SAMPLE_TEXT)
            path = f.name
        try:
            for pattern in DIFF_PATTERNS:
                with self.subTest(pattern=pattern):
                    self.assertEqual(self._ours(pattern, path), self._grep(pattern, path))
        finally:
            os.unlink(path)

    @unittest.skipUnless(os.path.exists(GUTENBERG), "56667-0.txt not found next to the tests")
    def test_gutenberg_book(self):
        for pattern in ("S(a|g|r)+on", "Ass(y|i)ria|Babylon", "th(e|a)*n",
                        "Babylon", "Nebuchadnezzar", "the"):
            with self.subTest(pattern=pattern):
                self.assertEqual(self._ours(pattern, GUTENBERG), self._grep(pattern, GUTENBERG))


if __name__ == "__main__":
    unittest.main()
