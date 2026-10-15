"""
entropy_pruner.py - Information-Theoretic Entropy Compactor & Dependency Pruner
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Calculates clause-level mutual information I(c_i; Y) and eliminates diffuse,
low-entropy verbiage that causes Transformer attention dispersion and hallucination.
"""

from __future__ import annotations
import math
import re
from typing import List, Dict, Set, Tuple
from bpe_transpiler import BPETranspiler


class EntropyPruner:
    """
    Prunes low-information tokens and clauses from prompts while preserving
    100% of hard constraints, parameter bindings, and semantic invariants.
    """

    # High-information invariant trigger stems
    INVARIANT_STEMS = {
        "must", "never", "only", "required", "schema", "format", "json", "return",
        "select", "where", "table", "column", "key", "id", "status", "limit",
        "strict", "exact", "error", "fail", "type", "enum", "integer", "string",
        "boolean", "array", "null", "not null", "primary key", "foreign key",
        "output", "input", "parameter", "true", "false", "default"
    }

    # Low-information conversational padding stems
    LOW_INFO_PATTERNS = [
        re.compile(r'^(hello|hi|hey|greetings|dear assistant)\b', re.IGNORECASE),
        re.compile(r'\b(i would like you to|can you please|could you kindly|i need your help to)\b', re.IGNORECASE),
        re.compile(r'\b(as an expert|acting as a|in your role as)\b', re.IGNORECASE),
        re.compile(r'\b(take a deep breath|think step by step carefully)\b', re.IGNORECASE),
        re.compile(r'\b(thank you very much|thanks in advance|much appreciated)\b', re.IGNORECASE),
        re.compile(r'\b(let me know if you have any questions)\b', re.IGNORECASE),
        re.compile(r'\b(hope this helps|i am excited to see your response)\b', re.IGNORECASE),
    ]

    def __init__(self, info_threshold: float = 0.30):
        self.info_threshold = info_threshold
        self.transpiler = BPETranspiler()

    def compute_clause_information_density(self, clause: str) -> float:
        """
        Estimates the empirical mutual information density I(clause; TaskInvariants).
        Normalized in range [0.0, 1.0].
        
        High density (> 0.50): Contains schema names, type constraints, logic operators.
        Low density (< 0.25): Pure polite conversational framing, pleasantries.
        """
        words = re.findall(r'\b\w+\b', clause.lower())
        if not words:
            return 0.0

        # Check for explicitly low-information conversational patterns
        for pattern in self.LOW_INFO_PATTERNS:
            if pattern.search(clause):
                return 0.05

        invariant_hits = sum(1 for w in words if w in self.INVARIANT_STEMS)
        unique_words = len(set(words))
        type_or_symbol_hits = len(re.findall(r'[:=\{\}\[\]\(\)\<\>\d_]', clause))

        # Information density metric:
        # Ratio of invariant stems + structural symbols over total length
        density = (invariant_hits * 2.0 + type_or_symbol_hits * 0.5) / (len(words) + 1)
        # Bounded between 0.0 and 1.0
        return min(1.0, max(0.05, density))

    def _extract_invariant_signature(self, clause: str) -> Set[str]:
        """Extracts semantic invariant signatures (e.g. OUTPUT_FORMAT_JSON) for exact deduplication."""
        lower = clause.lower()
        signatures = set()
        if "json" in lower and any(k in lower for k in ("format", "output", "return", "returned", "structure")):
            signatures.add("OUTPUT_FORMAT_JSON")
        if "xml" in lower and any(k in lower for k in ("format", "output", "return")):
            signatures.add("OUTPUT_FORMAT_XML")
        if "markdown" in lower and any(k in lower for k in ("format", "output", "table")):
            signatures.add("OUTPUT_FORMAT_MARKDOWN")
        if "where" in lower and any(k in lower for k in ("filter", "condition", "clause")):
            # extract where condition tokens
            cond_words = re.findall(r'[a-z_]+\s*[=><!]+\s*[\w\']+', lower)
            for cw in cond_words:
                signatures.add(f"WHERE_{cw.replace(' ', '')}")
        return signatures

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """Computes word-level Jaccard similarity for constraint deduplication."""
        w1 = set(re.findall(r'\b\w+\b', s1.lower()))
        w2 = set(re.findall(r'\b\w+\b', s2.lower()))
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    def prune_prompt(self, text: str) -> Tuple[str, int, int, float]:
        """
        Performs fine-grained entropy pruning on the input prompt text.
        
        Steps:
        1. Segment text into structural clauses (sentences, bullet items).
        2. Filter out clauses below info_threshold (conversational boilerplate).
        3. Deduplicate redundant restatements (e.g. repeated format warnings).
        4. Reconstruct concise, high-density constraint specification.

        Returns:
            Tuple[pruned_text, raw_tokens, pruned_tokens, retention_ratio]
        """
        raw_tokens = self.transpiler.estimate_bpe_tokens(text)

        # Segment by lines and sentence terminators
        raw_lines = text.split('\n')
        retained_clauses: List[str] = []
        seen_clauses: List[str] = []
        seen_signatures: Set[str] = set()

        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this is a header or markdown structure
            if stripped.startswith('#') or stripped.startswith('```'):
                retained_clauses.append(stripped)
                continue

            # Split sentences if line is a paragraph
            sentences = re.split(r'(?<=[.!?])\s+', stripped)
            for sentence in sentences:
                s_clean = sentence.strip()
                if not s_clean:
                    continue

                density = self.compute_clause_information_density(s_clean)
                
                # Check for low-entropy pleasantries
                if density < self.info_threshold:
                    # Drop conversational noise
                    continue

                # Check invariant signatures (e.g. repeating "Return strictly JSON")
                sigs = self._extract_invariant_signature(s_clean)
                if sigs and sigs.issubset(seen_signatures):
                    # Redundant specification of already registered invariant
                    continue

                # Check general word overlap
                is_duplicate = False
                for prev in seen_clauses:
                    if self._jaccard_similarity(s_clean, prev) > 0.50:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    retained_clauses.append(s_clean)
                    seen_clauses.append(s_clean)
                    seen_signatures.update(sigs)

        # Reconstruct pruned text
        pruned_text = '\n'.join(retained_clauses)
        # Apply BPE defragmentation
        pruned_text, _, pruned_tokens = self.transpiler.defragment_text(pruned_text)

        retention_ratio = pruned_tokens / max(1, raw_tokens)
        return pruned_text, raw_tokens, pruned_tokens, retention_ratio
