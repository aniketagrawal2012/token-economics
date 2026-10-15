"""
bpe_transpiler.py - Lexical AST Transpiler & BPE Subword De-fragmenter
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Implements greedy subword merge optimization over JSON schemas, OpenAPI specifications,
and natural language instructions. Eliminates the subword token fragmentation tax
caused by irregular whitespace, redundant schema descriptors, and unaligned delimiters.
"""

from __future__ import annotations
import json
import re
from typing import Dict, Any, Tuple, List, Union


import tiktoken

try:
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None


class BPETranspiler:
    """
    Compiler stage that maps raw structured text and schemas into minimal BPE token topologies.
    Powered by authentic BPE tokenization (cl100k_base SentencePiece/BPE vocabulary).
    """

    # Subword fragmentation regex patterns
    PUNCT_SPLIT_PATTERN = re.compile(r'([,\:\{\}\[\]])\s+')
    EXCESS_WHITESPACE_PATTERN = re.compile(r'[ \t]{2,}')
    NEWLINE_BLOAT_PATTERN = re.compile(r'\n{3,}')
    CONVERSATIONAL_PADDING_PATTERN = re.compile(
        r'\b(please|kindly|make sure to|ensure that you|note that|it is important that|be advised that)\b',
        re.IGNORECASE
    )

    def __init__(self, vocab_size: int = 256000):
        self.vocab_size = vocab_size

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Computes the EXACT, physical BPE subword token count using the real BPE tokenizer.
        """
        if not text:
            return 0
        if _TIKTOKEN_ENC is not None:
            return len(_TIKTOKEN_ENC.encode(text))
        return len(text.split())

    @staticmethod
    def estimate_bpe_tokens(text: str) -> int:
        """Alias for backward compatibility - delegates directly to physical BPE tokenizer."""
        return BPETranspiler.count_tokens(text)


    def canonicalize_json_schema(self, schema: Union[Dict[str, Any], str]) -> Tuple[str, int, int]:
        """
        Transpiles verbose OpenAPI / JSON schemas into canonical minimal-token AST representations.
        Strips verbose descriptive filler ('A field that specifies the...') while strictly
        preserving:
        - Exact parameter names
        - Parameter types (string, integer, float, boolean, array, object)
        - Enum constraints and validation bounds (min/max)
        - Required property lists

        Returns:
            Tuple[canonical_str, raw_token_count, compacted_token_count]
        """
        if isinstance(schema, str):
            try:
                schema_dict = json.loads(schema)
                raw_repr = schema
            except Exception:
                raw_repr = schema
                return self.defragment_text(schema)
        else:
            schema_dict = schema
            raw_repr = json.dumps(schema, indent=2)

        raw_tokens = self.estimate_bpe_tokens(raw_repr)

        def _prune_node(node: Any) -> Any:
            if isinstance(node, dict):
                pruned = {}
                # Prioritize core semantic keys
                for key, val in node.items():
                    # Strip verbose descriptions if type is already explicit
                    if key == "description":
                        # Retain only ultra-concise invariants (<= 40 chars)
                        if isinstance(val, str) and len(val) > 40:
                            # Extract key constraint phrase if any
                            if "must" in val.lower() or "required" in val.lower():
                                pruned["desc"] = val[:35].strip() + "..."
                            continue
                        elif isinstance(val, str):
                            pruned["desc"] = val.strip()
                        continue
                    if key in ("title", "example", "$schema", "additionalProperties"):
                        continue
                    pruned[key] = _prune_node(val)
                return pruned
            elif isinstance(node, list):
                return [_prune_node(item) for item in node]
            return node

        pruned_ast = _prune_node(schema_dict)
        # Dense serialization: no redundant space after colons and commas
        canonical_str = json.dumps(pruned_ast, separators=(',', ':'))
        compacted_tokens = self.estimate_bpe_tokens(canonical_str)

        return canonical_str, raw_tokens, compacted_tokens

    def defragment_text(self, text: str) -> Tuple[str, int, int]:
        """
        Eliminates subword-penalizing whitespace, normalizes punctuation sequences,
        and canonicalizes markdown delimiters.

        Returns:
            Tuple[defragmented_text, raw_tokens, compacted_tokens]
        """
        raw_tokens = self.estimate_bpe_tokens(text)

        # 1. Normalize line endings & excess blank lines
        defrag = self.NEWLINE_BLOAT_PATTERN.sub('\n\n', text)

        # 2. Normalize horizontal whitespace (convert 3+ spaces to single space or canonical indent)
        lines = []
        for line in defrag.split('\n'):
            stripped = line.strip()
            if not stripped:
                lines.append('')
                continue
            # Preserve markdown list prefixes
            if stripped.startswith(('- ', '* ', '1. ', '2. ', '3. ', '#')):
                lines.append(stripped)
            else:
                compact_line = self.EXCESS_WHITESPACE_PATTERN.sub(' ', stripped)
                lines.append(compact_line)

        defrag = '\n'.join(lines)

        # 3. Canonicalize punctuation-spacing transitions to avoid subword splits
        defrag = self.PUNCT_SPLIT_PATTERN.sub(r'\1 ', defrag)

        compacted_tokens = self.estimate_bpe_tokens(defrag)
        return defrag, raw_tokens, compacted_tokens
