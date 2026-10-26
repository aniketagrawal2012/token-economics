"""
entropy_pruner.py - Mutual-Information Entropy Pruner & Polarity-Aware 4-Tuple Deduplicator
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Implements Stage 2 of the Tokenomics Compiler:
1. Sub-clause Aho-Corasick / regex span stripping of 18 conversational pleasantry templates
   prior to clause scoring so mixed clauses preserve their constraint core.
2. Lexical mutual-information proxy I_hat(c_k; Y) (Eq. 6) with weight 1.0 on domain entities,
   weight 1.0 on symbolic/relational operators, and weight 0.8 on numeric/string literals.
3. Inverse-length threshold scaling tau_eff(c_tilde_k) = tau_info * min(1.0, |T_M(c_tilde_k)| / L_ref)
   (Eq. 7) with L_ref = 5.
4. Polarity-aware 4-tuple constraint signature extraction Sig(c_k) = {(attr, pi, op, val)}
   with pi in {+1, -1} to deduplicate redundant constraints without folding negated rules.
"""

from __future__ import annotations
import re
from typing import List, Tuple, Set, FrozenSet
from bpe_transpiler import BPETranspiler


class EntropyPruner:
    """
    Information-theoretic prompt pruner and polarity-aware constraint deduplicator.
    """

    # 18 Conversational pleasantry & filler span templates stripped at sub-clause level
    PLEASANTRY_SPAN_PATTERNS = [
        r"^\s*hello(?:\s+dear|\s+team|\s+google\s+cloud[^.!?\n]*)?[!,.]*\s*",
        r"^\s*hi\s+there[!,.]*\s*",
        r"^\s*greetings(?:\s+agent\s+controller)?[!,.]*\s*",
        r"^\s*dear\s+[a-z\s]+(?:assistant|controller|team|support)[!,.]*\s*",
        r"\bi\s+hope\s+you\s+are\s+(?:having\s+a\s+wonderful\s+day|doing\s+well)[!,.]*\s*",
        r"\bhope\s+you\s+have\s+a\s+great\s+day[!,.]*\s*",
        r"\bcan\s+you\s+please\s+kindly\s+help\s+me\s+with[^?.!]*[?.!]\s*",
        r"\bthank\s+you\s+(?:so\s+much|very\s+much)?(?:\s+in\s+advance)?[^.!?\n]*[!.]*\s*",
        r"\bi\s+really\s+appreciate\s+your\s+(?:help|assistance)[^.!?\n]*[!.]*\s*",
        r"\bthanks\s+a\s+lot[^.!?\n]*[!.]*\s*",
        r"\bplease\s+make\s+sure\s+to\s+carefully\b",
        r"\bplease\s+make\s+sure\s+(?:that\s+)?(?:the\s+response\s+is\s+)?",
        r"\bplease\s+ensure\s+that\s+the\s+query\s+is\s+formatted\s+cleanly[.!]*\s*",
        r"\bkindly\s+process\s+this\s+user\s+transaction\s+via\s+available\s+api\s+tools[.!]*\s*",
        r"\bas\s+an\s+ai\s+assistant[,]*\s*",
        r"\bif\s+you\s+don't\s+mind[,]*\s*",
        r"\bwould\s+you\s+be\s+so\s+kind\s+as\s+to\b",
        r"\bi\s+would\s+like\s+to\s+request\s+that\s+you\b",
    ]

    # Hard invariant & deontic modal keywords (Disjunct 1 of Eq. 5)
    HARD_INVARIANT_KEYWORDS = {
        "must", "never", "strictly", "required", "mandatory", "invariant", "filter",
        "where", "select", "from", "order", "limit", "schema", "json", "error",
        "rollback", "abort", "escalate", "retry", "circuit", "breaker", "etag",
        "validate", "idempotency", "aml", "ctr", "fincen", "spanner", "transaction"
    }

    # Symbolic & relational operators V_op (weighted 1.0 in Eq. 6)
    SYMBOLIC_OPERATORS = {
        "=", "==", "!=", "<>", ">=", "<=", ">", "<", "&&", "||", "->", "=>", "::"
    }

    # Word-level logical/relational operators included in V_op
    LEXICAL_OPERATORS = {
        "and", "or", "not", "in", "between", "like", "is", "null", "true", "false",
        "asc", "desc", "eq", "neq", "gt", "lt", "gte", "lte"
    }

    def __init__(self, info_threshold: float = 0.30, l_ref: int = 5):
        self.info_threshold = info_threshold
        self.l_ref = l_ref
        self.transpiler = BPETranspiler()
        self._compiled_pleasantries = [re.compile(p, re.IGNORECASE) for p in self.PLEASANTRY_SPAN_PATTERNS]

    def strip_pleasantry_spans(self, clause: str) -> str:
        """
        Sub-clause Aho-Corasick / regex span stripper: removes conversational pleasantry
        prefixes/suffixes while preserving any embedded technical constraint in the same sentence.
        """
        cleaned = clause
        for pat in self._compiled_pleasantries:
            cleaned = pat.sub(" ", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;-")
        return cleaned

    def compute_clause_information_score(self, cleaned_clause: str) -> Tuple[float, float, int, bool]:
        r"""
        Evaluates Eq. (6) and Eq. (7):
          \hat{I}(c_k; Y) = (1.0 * |c_k \cap V_entity| + 1.0 * |c_k \cap V_op| + 0.8 * |c_k \cap V_num|) / |T_M(c_k)|
          \tau_eff(c_k)   = \tau_info * min(1.0, |T_M(c_k)| / L_ref)
        Returns (info_score, tau_eff, token_count, has_hard_invariant).
        """
        if not cleaned_clause:
            return 0.0, self.info_threshold, 0, False

        tok_count = max(1, self.transpiler.estimate_bpe_tokens(cleaned_clause))
        tau_eff = self.info_threshold * min(1.0, tok_count / float(self.l_ref))

        words_lower = re.findall(r"[a-z0-9_']+", cleaned_clause.lower())

        # Check Disjunct 1: Hard Invariant or Deontic Modal presence
        has_hard_inv = any(w.strip("'") in self.HARD_INVARIANT_KEYWORDS for w in words_lower)

        # Count V_entity: snake_case, CamelCase, UPPER_CASE identifiers, quoted field names, SQL/API identifiers
        entity_matches = re.findall(
            r"\b(?:[a-z]+_[a-z0-9_]+|[a-z]+[A-Z][a-zA-Z0-9]*|[A-Z]{2,}[A-Z0-9_]*)\b|'[^']+'|\"[^\"]+\"",
            cleaned_clause
        )
        n_entity = len(entity_matches)
        # Also count domain nouns in HARD_INVARIANT_KEYWORDS as domain entities if not already counted
        for w in words_lower:
            w_clean = w.strip("'")
            if w_clean in self.HARD_INVARIANT_KEYWORDS and w_clean not in ("must", "never", "strictly", "required"):
                n_entity += 1

        # Count V_op: symbolic relational operators + lexical operators (weight 1.0 in Eq. 6)
        sym_ops = re.findall(r"(?:>=|<=|!=|==|<>|&&|\|\||->|=>|[=><])", cleaned_clause)
        lex_ops = [w for w in words_lower if w in self.LEXICAL_OPERATORS]
        n_op = len(sym_ops) + len(lex_ops)

        # Count V_num: numeric literals, dates, currency amounts, version/hash codes (weight 0.8 in Eq. 6)
        num_matches = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned_clause)
        n_num = len(num_matches)

        numerator = 1.0 * n_entity + 1.0 * n_op + 0.8 * n_num
        info_score = numerator / float(tok_count)

        return info_score, tau_eff, tok_count, has_hard_inv

    @classmethod
    def extract_polarity_signature(cls, clause: str) -> FrozenSet[Tuple[str, int, str, str]]:
        r"""
        Extracts the polarity-aware 4-tuple signature Sig(c_k) = {(attr, \pi, op, val)}:
        - \pi \in {+1, -1} captures affirmative (+1) vs. negative (-1) constraint polarity.
        - Consolidates semantically equivalent output-format directives (e.g., 'return output strictly as JSON')
          under canonical tuple ('output_format', +1, '=', 'json').
        """
        lower = clause.lower().strip()
        polarity = -1 if re.search(r"\b(?:never|not|cannot|do\s+not|don't|prohibited|without|!=|<>)\b", lower) else +1

        tuples: Set[Tuple[str, int, str, str]] = set()

        # Check canonical JSON output format constraint (affirmative only, without additional field lists)
        if "json" in lower and polarity == +1:
            has_specific_fields = bool(re.search(r"\b(?:key|keys|field|fields|matching|schema)\b", lower))
            if not has_specific_fields:
                return frozenset([("output_format", +1, "=", "json")])

        # Extract explicit relational triples: attr op val
        rel_matches = re.findall(
            r"([a-zA-Z_][a-zA-Z0-9_.]*)\s*(>=|<=|!=|==|<>|=|>|<)\s*('[^']*'|\"[^\"]*\"|[a-zA-Z0-9_.-]+)",
            clause
        )
        for attr, op, val in rel_matches:
            rel_pol = -1 if op in ("!=", "<>") else polarity
            norm_op = "=" if op in ("==", "!=", "<>") else op
            tuples.add((attr.lower(), rel_pol, norm_op, val.strip("'\"").lower()))

        if tuples:
            return frozenset(tuples)

        # Fallback signature based on normalized content tokens + polarity
        content_words = [
            w for w in re.findall(r"[a-z0-9_]+", lower)
            if w not in {"the", "a", "an", "please", "ensure", "make", "sure", "that", "is", "be", "to", "in"}
        ]
        if content_words:
            attr_key = "_".join(content_words[:3])
            val_key = "_".join(content_words[3:6]) if len(content_words) > 3 else "true"
            tuples.add((attr_key, polarity, "=", val_key))

        return frozenset(tuples)

    def prune_prompt(self, raw_prompt: str) -> Tuple[str, int, int, float]:
        """
        Executes Stage 2 (Pleasantry Span Stripping + Mutual-Information Pruning + 4-Tuple Deduplication).
        Returns (pruned_prompt, raw_tokens, pruned_tokens, compression_ratio).
        """
        if not raw_prompt:
            return "", 0, 0, 1.0

        raw_tokens = self.transpiler.estimate_bpe_tokens(raw_prompt)

        # Split into lines and sentence clauses while preserving list bullets
        raw_lines = raw_prompt.splitlines()
        retained_clauses: List[str] = []
        seen_signatures: Set[FrozenSet[Tuple[str, int, str, str]]] = set()

        for line in raw_lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Preserve bullet items or header lines as single units; split prose lines on sentence boundaries
            if line_stripped.startswith(("-", "*", "#")) or ":" in line_stripped:
                sub_clauses = [line_stripped]
            else:
                sub_clauses = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line_stripped) if s.strip()]

            kept_in_line: List[str] = []
            for sc in sub_clauses:
                cleaned_sc = self.strip_pleasantry_spans(sc)
                if not cleaned_sc or len(cleaned_sc) < 3:
                    continue

                info_score, tau_eff, tok_cnt, has_hard_inv = self.compute_clause_information_score(cleaned_sc)

                # Retention predicate (Eq. 5): Disjunct 1 (hard invariant) OR Disjunct 2 (info_score >= tau_eff)
                if has_hard_inv or info_score >= tau_eff:
                    sig = self.extract_polarity_signature(cleaned_sc)
                    if sig and sig in seen_signatures:
                        # Duplicate constraint with identical polarity -> fold!
                        continue
                    if sig:
                        seen_signatures.add(sig)
                    kept_in_line.append(cleaned_sc)

            if kept_in_line:
                retained_clauses.append(" ".join(kept_in_line))

        pruned_text = "\n".join(retained_clauses)
        pruned_tokens = self.transpiler.estimate_bpe_tokens(pruned_text)
        ratio = round(pruned_tokens / max(1, raw_tokens), 4)
        return pruned_text, raw_tokens, pruned_tokens, ratio
