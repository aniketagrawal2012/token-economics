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

    # 64-token deterministic invariant lexicon K_inv across 4 sub-lexicons (matched case-insensitively, #11)
    K_INV_LEXICON = {
        # (i) 16 deontic modals
        "must", "shall", "required", "mandatory", "enforce", "ensure", "always", "strictly",
        "only", "exact", "bound", "limit", "verify", "assert", "guarantee", "comply",
        # (ii) 16 negative polarity markers
        "never", "not", "cannot", "prohibit", "exclude", "disallow", "forbid", "omit",
        "reject", "deny", "prevent", "without", "none", "neither", "invalid", "illegal",
        # (iii) 16 relational/SQL keywords
        "select", "where", "join", "group", "having", "order", "distinct", "from",
        "union", "intersect", "exists", "between", "asc", "desc", "ascend", "descend",
        # (iv) 16 schema/domain contract specifiers (including case-insensitive 'utc')
        "type", "enum", "format", "nullable", "unique", "primary", "foreign", "index",
        "schema", "pattern", "minimum", "maximum", "integer", "boolean", "string", "utc",
    }

    # 28-operator symbolic/structural lexicon S_sym (#11)
    S_SYM_OPERATORS = {
        "=", "==", "!=", "<>", "<", ">", "<=", ">=", "IN", "NOT IN", "LIKE",
        "IS NULL", "NOT NULL", "::", "->", "=>", "+", "-", "*", "/", "%",
        "&&", "||", "!", "{", "}", "[", "]",
    }

    # Hard invariant & deontic modal keywords
    HARD_INVARIANT_KEYWORDS = K_INV_LEXICON | {
        "invariant", "filter", "json", "error", "rollback", "abort", "escalate",
        "retry", "circuit", "breaker", "etag", "validate", "idempotency", "aml",
        "ctr", "fincen", "spanner", "transaction"
    }

    # Symbolic & relational operators V_op (weighted 1.0 in Eq. 6)
    SYMBOLIC_OPERATORS = {
        "=", "==", "!=", "<>", ">=", "<=", ">", "<", "&&", "||", "->", "=>", "::",
        "+", "-", "*", "/", "%", "!", "{", "}", "[", "]"
    }

    # Word-level logical/relational operators included in V_op
    LEXICAL_OPERATORS = {
        "and", "or", "not", "in", "between", "like", "is", "null", "true", "false",
        "asc", "desc", "eq", "neq", "gt", "lt", "gte", "lte"
    }

    def __init__(self, info_threshold: float = 0.30, l_ref: int = 5, custom_lexicon: Set[str] | None = None):
        self.info_threshold = info_threshold
        self.l_ref = l_ref
        self.transpiler = BPETranspiler()
        self.active_k_inv = set(self.K_INV_LEXICON)
        if custom_lexicon:
            self.active_k_inv |= {w.lower() for w in custom_lexicon}
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

    def compute_empirical_invariant_density(self, cleaned_clause: str) -> Tuple[float, int, int, int]:
        r"""
        Evaluates Eq. (6) on residual sub-clause \tilde{c}_k:
          D_inv(\tilde{c}_k) = (2.0 * Count(\tilde{c}_k, K_inv) + 1.0 * Count(\tilde{c}_k, S_sym)) / (|T_M(\tilde{c}_k)| + 1)
        Returns (d_inv, k_inv_count, s_sym_count, tok_count).
        """
        if not cleaned_clause:
            return 0.0, 0, 0, 0
        tok_count = max(1, self.transpiler.estimate_bpe_tokens(cleaned_clause))
        words_lower = [w.strip("'") for w in re.findall(r"[a-z0-9_']+", cleaned_clause.lower())]
        k_inv_count = sum(1 for w in words_lower if w in self.active_k_inv)

        upper_clause = cleaned_clause.upper()
        multi_word_ops = sum(upper_clause.count(op) for op in ("NOT IN", "IS NULL", "NOT NULL"))
        sym_tokens = re.findall(r"(?:>=|<=|!=|==|<>|&&|\|\||->|=>|::|[=><+\-*/%!{}\[\]])", cleaned_clause)
        word_ops = sum(1 for w in words_lower if w in ("in", "like"))
        s_sym_count = multi_word_ops + len(sym_tokens) + word_ops

        weighted_num = 2.0 * k_inv_count + 1.0 * s_sym_count
        d_inv = weighted_num / float(tok_count + 1)
        return d_inv, k_inv_count, s_sym_count, tok_count

    def evaluate_retention_predicate(self, cleaned_clause: str, has_novel_sig: bool = False) -> bool:
        r"""
        Evaluates Eq. (7) non-redundant 3-disjunct retention filter (#1):
          Retain(c_k) <=> (Sig_new(c_k) != empty)
                      OR  (|T_M(\tilde{c}_k)| <= 2  AND  D_inv(\tilde{c}_k) >= \tau_info)
                      OR  (|T_M(\tilde{c}_k)| >= 1  AND  D_inv(\tilde{c}_k) >= 2.0 / (|T_M(\tilde{c}_k)| + 1))
        """
        if has_novel_sig:
            return True
        d_inv, _, _, tok_count = self.compute_empirical_invariant_density(cleaned_clause)
        if tok_count == 0:
            return False
        disjunct_2 = (tok_count <= 2) and (d_inv >= self.info_threshold)
        disjunct_3 = (tok_count >= 1) and (d_inv + 1e-12 >= 2.0 / float(tok_count + 1))
        return disjunct_2 or disjunct_3

    def compute_clause_information_score(self, cleaned_clause: str) -> Tuple[float, float, int, bool]:
        r"""
        Evaluates Eq. (6) and Eq. (7) and returns (info_score, tau_eff, token_count, has_hard_invariant).
        """
        if not cleaned_clause:
            return 0.0, self.info_threshold, 0, False

        d_inv, k_inv_cnt, s_sym_cnt, tok_count = self.compute_empirical_invariant_density(cleaned_clause)
        tau_eff = self.info_threshold if tok_count <= 2 else (2.0 / float(tok_count + 1))

        words_lower = re.findall(r"[a-z0-9_']+", cleaned_clause.lower())
        has_hard_inv = (k_inv_cnt >= 1) or (s_sym_cnt >= 2) or any(
            w.strip("'") in self.HARD_INVARIANT_KEYWORDS for w in words_lower
        )

        entity_matches = re.findall(
            r"\b(?:[a-z]+_[a-z0-9_]+|[a-z]+[A-Z][a-zA-Z0-9]*|[A-Z]{2,}[A-Z0-9_]*)\b|'[^']+'|\"[^\"]+\"",
            cleaned_clause
        )
        n_entity = len(entity_matches) + k_inv_cnt
        sym_ops = re.findall(r"(?:>=|<=|!=|==|<>|&&|\|\||->|=>|[=><])", cleaned_clause)
        lex_ops = [w for w in words_lower if w in self.LEXICAL_OPERATORS]
        n_op = max(s_sym_cnt, len(sym_ops) + len(lex_ops))
        num_matches = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned_clause)
        n_num = len(num_matches)

        numerator = max(2.0 * k_inv_cnt + 1.0 * s_sym_cnt, 1.0 * n_entity + 1.0 * n_op + 0.8 * n_num)
        info_score = max(d_inv, numerator / float(tok_count + 1))

        return info_score, tau_eff, tok_count, has_hard_inv

    @classmethod
    def extract_polarity_signature(cls, clause: str) -> FrozenSet[Tuple[str, int, str, str]]:
        r"""
        Extracts the polarity-aware 4-tuple signature Sig(c_k) = {(attr, \pi, op, val)}:
        - \pi \in {+1, -1} captures affirmative (+1) vs. negative (-1) constraint polarity.
        - Propagates clause-initial negative polarity markers across coordinated conjuncts
          ('or', 'nor', 'and') beyond the local 5-token window (#3).
        """
        lower = clause.lower().strip()
        clause_level_neg = bool(re.search(
            r"\b(?:never|not|cannot|do\s+not|don't|prohibit(?:ed)?|forbid(?:den)?|disallow(?:ed)?|"
            r"exclude|omit|reject|deny|prevent|without|none|neither)\b",
            lower
        ))
        default_polarity = -1 if (clause_level_neg or re.search(r"(?:!=|<>)", lower)) else +1

        tuples: Set[Tuple[str, int, str, str]] = set()

        # Check canonical JSON output format constraint (affirmative only, without additional field lists)
        if "json" in lower and default_polarity == +1:
            has_specific_fields = bool(re.search(r"\b(?:key|keys|field|fields|matching|schema)\b", lower))
            if not has_specific_fields:
                return frozenset([("output_format", +1, "=", "json")])

        # Extract explicit relational triples: attr op val
        rel_iter = re.finditer(
            r"([a-zA-Z_][a-zA-Z0-9_.]*)\s*(>=|<=|!=|==|<>|=|>|<)\s*('[^']*'|\"[^\"]*\"|[a-zA-Z0-9_.-]+)",
            clause
        )
        for m in rel_iter:
            attr, op, val = m.group(1), m.group(2), m.group(3)
            prefix_window = clause[max(0, m.start() - 40):m.start()].lower()
            local_neg = bool(re.search(
                r"\b(?:never|not|cannot|do\s+not|don't|prohibit(?:ed)?|forbid(?:den)?|disallow(?:ed)?|without|neither|nor)\b",
                prefix_window
            ))
            # Propagate clause-level negation across coordinated conjuncts (#3)
            if op in ("!=", "<>") or local_neg or clause_level_neg:
                rel_pol = -1
            else:
                rel_pol = +1
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
            tuples.add((attr_key, default_polarity, "=", val_key))

        return frozenset(tuples)

    def prune_prompt(self, raw_prompt: str) -> Tuple[str, int, int, float]:
        """
        Executes Stage 2 (Pleasantry Span Stripping + Invariant Density Pruning + 4-Tuple Deduplication).
        Emits span-stripped residual clause c_tilde_k into I_inv when Retain(c_k) holds (#2).
        Returns (pruned_prompt, raw_tokens, pruned_tokens, compression_ratio).
        """
        if not raw_prompt:
            return "", 0, 0, 1.0

        raw_tokens = self.transpiler.estimate_bpe_tokens(raw_prompt)

        raw_lines = raw_prompt.splitlines()
        retained_clauses: List[str] = []
        seen_signatures: Set[Tuple[str, int, str, str]] = set()

        for line in raw_lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            if line_stripped.startswith(("-", "*", "#")) or ":" in line_stripped:
                sub_clauses = [line_stripped]
            else:
                sub_clauses = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line_stripped) if s.strip()]

            kept_in_line: List[str] = []
            for sc in sub_clauses:
                cleaned_sc = self.strip_pleasantry_spans(sc)
                if not cleaned_sc or len(cleaned_sc) < 3:
                    continue

                sig = self.extract_polarity_signature(cleaned_sc)
                sig_new = set(sig) - seen_signatures if sig else set()
                has_novel_sig = len(sig_new) > 0

                # If all extracted tuples in sig were already seen, fold duplicate constraint (#2)
                if sig and not has_novel_sig:
                    continue

                info_score, tau_eff, _, has_hard_inv = self.compute_clause_information_score(cleaned_sc)
                retain = self.evaluate_retention_predicate(cleaned_sc, has_novel_sig=False) or has_hard_inv or (info_score >= tau_eff)

                if retain:
                    if sig:
                        seen_signatures.update(sig)
                    kept_in_line.append(cleaned_sc)

            if kept_in_line:
                retained_clauses.append(" ".join(kept_in_line))

        pruned_text = "\n".join(retained_clauses)
        pruned_tokens = self.transpiler.estimate_bpe_tokens(pruned_text)
        ratio = round(pruned_tokens / max(1, raw_tokens), 4)
        return pruned_text, raw_tokens, pruned_tokens, ratio

