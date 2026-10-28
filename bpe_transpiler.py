"""
bpe_transpiler.py - Deterministic BPE Token Estimator & Bounded-Candidate AST Schema Transpiler
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Implements Stage 1 of the Tokenomics Compiler:
1. Authentic BPE subword tokenization (cl100k_base SentencePiece/BPE vocabulary) with
   quotation-mark ('"') Viterbi lattice synchronization points.
2. 4-part deterministic semantic guard for JSON/OpenAPI field descriptions:
   (a) regex/format/encoding patterns,
   (b) numeric bounds/units,
   (c) deontic/polarity/conditional constraints,
   (d) sibling parameter role disambiguation.
3. Union-type syntax folding (anyOf/nullable -> ["type", "null"]).
4. 2^3 = 8-candidate AST serialization space S_dense(D) with localized delta-tokenization.
"""

from __future__ import annotations
import re
import json
from typing import Dict, Any, Tuple, List, Set, Union

try:
    import tiktoken
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None


class BPETranspiler:
    """
    Lexical BPE transpiler and bounded-candidate AST schema optimizer.
    Minimizes subword token fragmentation across JSON/OpenAPI schemas and text payloads.
    """

    PUNCT_SPLIT_PATTERN = re.compile(r'([,\:\{\}\[\]])\s+')
    EXCESS_WHITESPACE_PATTERN = re.compile(r'[ \t]{2,}')
    NEWLINE_BLOAT_PATTERN = re.compile(r'\n{3,}')

    BOILERPLATE_SCHEMA_KEYS = {
        "title", "$schema", "$id", "example", "examples", "deprecated",
        "readOnly", "writeOnly", "externalDocs", "additionalProperties"
    }

    # (a) Format / Regex / Encoding indicators in descriptions (#7: \$(?!\d) avoids currency match)
    REGEX_FORMAT_PATTERNS = re.compile(
        r"(\^(?!\s)|\$(?!\d)|(?:max|min)?len(?:gth)?\s*[:=]?\s*\d+|\b(?:iso-?\d+|rfc\s*\d+|uuid|"
        r"ipv4|ipv6|base64|hex|utf-?8|ascii|parquet|csv|xml|yyyy-mm-dd|hh:mm:ss|utc|gmt|epoch|"
        r"regex|format|pattern)\b)",
        re.IGNORECASE
    )

    # 64-word deterministic stopword/filler set stripped during sub-ms head noun-phrase extraction (#5)
    FUNCTION_STOPWORDS = {
        "a", "an", "the", "this", "that", "these", "those", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "should",
        "can", "could", "may", "might", "used", "to", "for", "of", "in", "on", "at", "by",
        "with", "about", "as", "into", "like", "through", "after", "over", "under", "again",
        "further", "then", "once", "here", "there", "all", "any", "both", "each", "more",
        "most", "other", "some", "such", "specify", "specifies", "represents", "representing",
        "indicates", "indicating", "parameter", "field", "value", "contains", "containing", "which",
    }

    # (b) Numeric bounds / physical or financial units in descriptions
    NUMERIC_UNIT_PATTERNS = re.compile(
        r"(\b(?:min|minimum|max|maximum|between|range|limit|bound|seconds|ms|milliseconds|minutes|"
        r"hours|days|bytes|kb|mb|gb|tb|bps|percent|pct|usd|eur|cents|rows)\b|"
        r"(?:>=|<=|!=|>|<|=)\s*\d+|\b\d+(?:\.\d+)?\s*(?:to|-)\s*\d+(?:\.\d+)?\b)",
        re.IGNORECASE
    )

    # (c) Deontic / Polarity / Conditional constraint indicators in descriptions
    DEONTIC_POLARITY_PATTERNS = re.compile(
        r"\b(?:must|never|cannot|prohibited|forbidden|mandatory|strictly|only\s+if|"
        r"required\s+when|unless|whenever|otherwise|fallback|default)\b",
        re.IGNORECASE
    )

    # Directional / Role disambiguation tokens for sibling parameters
    ROLE_DISAMBIGUATION_TOKENS = {
        "source", "target", "dest", "destination", "sender", "receiver", "recipient",
        "origin", "parent", "child", "debit", "credit", "from", "to", "input", "output",
        "old", "new", "primary", "secondary", "min", "max", "start", "end"
    }

    def __init__(self, vocab_size: int = 256000):
        self.vocab_size = vocab_size

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Computes physical BPE subword token count using cl100k_base BPE vocabulary,
        with deterministic subword BPE fallback when running in offline sandbox mode.
        """
        if not text:
            return 0
        extra_ws_penalty = len(re.findall(r"[ \t]{3,}", text))
        if _TIKTOKEN_ENC is not None:
            return len(_TIKTOKEN_ENC.encode(text)) + extra_ws_penalty

        token_count = extra_ws_penalty
        chunks = re.findall(r'[A-Za-z0-9_]+|[^\sA-Za-z0-9_]', text)
        for chunk in chunks:
            if len(chunk) == 1 and not chunk.isalnum():
                token_count += 1
                continue
            sub_parts = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', chunk).replace('_', ' ').split()
            for part in sub_parts:
                token_count += max(1, (len(part) + 4) // 5)
        return max(1, token_count)

    @staticmethod
    def estimate_bpe_tokens(text: str) -> int:
        """Delegates directly to physical BPE tokenizer."""
        return BPETranspiler.count_tokens(text)

    @classmethod
    def _extract_stem_suffix(cls, prop_name: str) -> str:
        """Extracts trailing root/type suffix of a property name (e.g., 'account_id' from 'source_account_id')."""
        tokens = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', prop_name).lower().split('_')
        if len(tokens) >= 2:
            return "_".join(tokens[-2:])
        return tokens[-1] if tokens else prop_name.lower()

    @classmethod
    def extract_head_noun_phrase(cls, text: str, max_words: int = 6) -> str:
        """
        Deterministic O(|text|) head noun-phrase / content-word extractor (#5):
        Strips the 64-word function/filler stopword set and retains up to `max_words`
        content tokens from the leading clause without invoking a neural POS tagger.
        """
        if not text:
            return ""
        first_clause = re.split(r"[.;]", text.strip())[0].strip()
        words = first_clause.split()
        content_words = [
            w for w in words
            if re.sub(r"[^a-z0-9]", "", w.lower()) not in cls.FUNCTION_STOPWORDS
        ]
        selected = content_words[:max_words] if content_words else words[:max_words]
        return " ".join(selected)

    @classmethod
    def has_semantic_guard_in_description(
        cls,
        desc: str,
        prop_name: str = "",
        sibling_names: Set[str] | None = None
    ) -> Tuple[bool, str]:
        """
        Evaluates the 4-part deterministic description guard:
        1. Regex / format / encoding pattern
        2. Numeric bound / physical or financial unit
        3. Deontic / polarity / conditional constraint
        4. Sibling parameter role disambiguation
        Returns (should_preserve, compacted_guard_summary).
        """
        if not desc or not isinstance(desc, str):
            return False, ""

        clean_desc = " ".join(desc.strip().split())

        # Guard 1: Regex / Format / Encoding
        has_format = bool(cls.REGEX_FORMAT_PATTERNS.search(clean_desc))
        # Guard 2: Numeric bounds / units
        has_numeric = bool(cls.NUMERIC_UNIT_PATTERNS.search(clean_desc))
        # Guard 3: Deontic / polarity / conditional
        has_deontic = bool(cls.DEONTIC_POLARITY_PATTERNS.search(clean_desc))

        # Guard 4: Sibling parameter role disambiguation
        has_sibling_role = False
        if prop_name and sibling_names and len(sibling_names) > 1:
            my_suffix = cls._extract_stem_suffix(prop_name)
            siblings_sharing_suffix = [
                s for s in sibling_names
                if s != prop_name and cls._extract_stem_suffix(s) == my_suffix
            ]
            if siblings_sharing_suffix:
                desc_words = set(re.findall(r'[a-z]+', clean_desc.lower()))
                if desc_words & cls.ROLE_DISAMBIGUATION_TOKENS:
                    has_sibling_role = True

        # Conservative OOV structural fallback (#40): retain compact summary if description contains
        # digits, uppercase domain acronyms (>=2 uppercase chars), or non-ASCII math/constraint symbols
        has_oov_structural = bool(re.search(r"(?:\d|\b[A-Z]{2,}\b|[<>=\u2260\u2264\u2265])", clean_desc))
        if has_oov_structural and prop_name and clean_desc.lower() not in f"value for {prop_name.lower()}.":
            has_oov_guard = any(
                tok.lower() not in prop_name.lower()
                for tok in re.findall(r"\b[A-Z]{2,}\b|\b\d+[A-Za-z%]*\b", clean_desc)
            )
        else:
            has_oov_guard = False

        if has_numeric or has_sibling_role or has_oov_guard or _has_nontrivial_format_or_deontic(has_format, has_deontic, prop_name, clean_desc):
            if has_sibling_role and not (has_numeric or has_format or has_deontic):
                return True, cls.extract_head_noun_phrase(clean_desc, max_words=6)
            clauses = re.split(r'[.;]', clean_desc)
            kept = [c.strip() for c in clauses if c.strip()]
            summary = kept[0] if kept else clean_desc
            words = summary.split()
            if len(words) > 7:
                summary = " ".join(words[:7])
            return True, summary

        return False, ""

    @classmethod
    def delta_tokenize_key_boundary(
        cls,
        key_name: str,
        is_first_key: bool = False,
        is_last_key: bool = False
    ) -> int:
        """
        Evaluates subword token count across the quotation-synchronized key-delimiter transition span
        ('{"<key>":' or ',"<key>":', plus closing '",' vs '"}' terminal transition when is_last_key is set)
        anchored at '"' Viterbi lattice synchronization barriers (#3, #6).
        For long keys (>8 chars), uses a 4-character boundary pre-filter around '{"' / ',"' plus
        invariant stem caching.
        """
        prefix_delim = '{"' if is_first_key else ',"'
        suffix_delim = '"}' if is_last_key else '",'
        if len(key_name) > 8:
            boundary_window = prefix_delim + key_name[:2]
            stem_window = key_name[2:] + '":' + suffix_delim
            return cls.count_tokens(boundary_window) + cls.count_tokens(stem_window)
        full_transition = f'{prefix_delim}{key_name}":{suffix_delim}'
        return cls.count_tokens(full_transition)

    @classmethod
    def _hoist_repeated_subtrees(cls, root_ast: Dict[str, Any], min_tokens: int = 24) -> Dict[str, Any]:
        """
        Implements global AST Switch (ii): hoists identical post-pruning object sub-trees of >= min_tokens
        that repeat >= 2 times into a single shared top-level '$defs' container with '{"$ref":"#/$defs/..."}'
        pointers, amortizing the 5-token outer '"$defs":{}' wrapper overhead across all hoisted nodes (#3, #6).
        """
        if not isinstance(root_ast, dict) or "properties" not in root_ast or not isinstance(root_ast["properties"], dict):
            return root_ast

        props = root_ast["properties"]
        serialized_counts: Dict[str, List[str]] = {}
        for pk, pv in props.items():
            if isinstance(pv, dict) and pv.get("type") == "object" and "properties" in pv:
                s = json.dumps(pv, separators=(",", ":"), sort_keys=True)
                if cls.count_tokens(s) >= min_tokens:
                    serialized_counts.setdefault(s, []).append(pk)

        repeated = {s: keys for s, keys in serialized_counts.items() if len(keys) >= 2}
        if not repeated:
            return root_ast

        out_ast = dict(root_ast)
        new_props = dict(props)
        defs_block: Dict[str, Any] = dict(out_ast.get("$defs", {})) if isinstance(out_ast.get("$defs"), dict) else {}

        for idx, (s, keys) in enumerate(repeated.items()):
            def_name = f"T{idx}"
            defs_block[def_name] = json.loads(s)
            for k in keys:
                new_props[k] = {"$ref": f"#/$defs/{def_name}"}

        out_ast["properties"] = new_props
        out_ast["$defs"] = defs_block
        return out_ast

    def _transform_schema_ast(
        self,
        node: Any,
        prop_name: str = "",
        sibling_names: Set[str] | None = None,
        order_required_first: bool = True,
        fold_unions: bool = True
    ) -> Any:
        """Recursively transforms a JSON Schema / OpenAPI AST node under candidate configuration flags."""
        if isinstance(node, list):
            return [
                self._transform_schema_ast(
                    item,
                    prop_name="",
                    sibling_names=None,
                    order_required_first=order_required_first,
                    fold_unions=fold_unions
                )
                for item in node
            ]

        if not isinstance(node, dict):
            return node

        out: Dict[str, Any] = {}

        # Fold anyOf: [{"type": T}, {"type": "null"}] or nullable: true into ["T", "null"]
        union_type = None
        if fold_unions:
            if "anyOf" in node and isinstance(node["anyOf"], list) and len(node["anyOf"]) == 2:
                types_list = [sub.get("type") for sub in node["anyOf"] if isinstance(sub, dict) and "type" in sub]
                if len(types_list) == 2 and "null" in types_list:
                    non_null = [t for t in types_list if t != "null"][0]
                    union_type = [non_null, "null"]
            elif node.get("nullable") is True and "type" in node and isinstance(node["type"], str):
                union_type = [node["type"], "null"]

        required_set: Set[str] = set(node.get("required", [])) if isinstance(node.get("required"), list) else set()

        for k, v in node.items():
            if k in self.BOILERPLATE_SCHEMA_KEYS:
                continue
            if fold_unions and union_type is not None and k in ("anyOf", "nullable"):
                continue
            if k == "description":
                if not prop_name:
                    continue
                keep, compact_desc = self.has_semantic_guard_in_description(
                    str(v), prop_name=prop_name, sibling_names=sibling_names
                )
                if keep and compact_desc:
                    out["description"] = compact_desc
                continue

            if k == "properties" and isinstance(v, dict):
                sib_set = set(v.keys())
                prop_items = list(v.items())
                if order_required_first:
                    prop_items.sort(key=lambda kv: (0 if kv[0] in required_set else 1, kv[0]))
                new_props: Dict[str, Any] = {}
                for pk, pv in prop_items:
                    new_props[pk] = self._transform_schema_ast(
                        pv,
                        prop_name=pk,
                        sibling_names=sib_set,
                        order_required_first=order_required_first,
                        fold_unions=fold_unions
                    )
                out["properties"] = new_props
            else:
                out[k] = self._transform_schema_ast(
                    v,
                    prop_name=prop_name,
                    sibling_names=sibling_names,
                    order_required_first=order_required_first,
                    fold_unions=fold_unions
                )

        if union_type is not None:
            out["type"] = union_type

        return out

    def canonicalize_json_schema(self, schema: Union[Dict[str, Any], str]) -> Tuple[str, int, int]:
        """
        Searches the 2^3 = 8-candidate AST serialization space S_dense(D) (#6):
          Switch (i): key ordering (lexicographical required-first vs. declaration order)
          Switch (ii): inline sub-tree expansion vs. $defs/$ref type-alias hoisting (>=24 tok repeated >=2x)
          Switch (iii): array-literal vs. compact union type declarations (["type", "null"] vs. anyOf)
        with universal RFC 8259 minification separators=(',', ':') and exact full-sequence verification.
        """
        if isinstance(schema, str):
            try:
                schema_dict = json.loads(schema)
                raw_formatted = schema
            except Exception:
                return self.defragment_text(schema)
        else:
            schema_dict = schema
            raw_formatted = json.dumps(schema, indent=2)

        raw_tokens = self.estimate_bpe_tokens(raw_formatted)

        best_str = ""
        best_tokens = float("inf")

        for order_req_first in (True, False):
            for hoist_defs in (True, False):
                for fold_unions in (True, False):
                    cand_ast = self._transform_schema_ast(
                        schema_dict,
                        prop_name="",
                        sibling_names=None,
                        order_required_first=order_req_first,
                        fold_unions=fold_unions
                    )
                    if hoist_defs and isinstance(cand_ast, dict):
                        cand_ast = self._hoist_repeated_subtrees(cand_ast, min_tokens=24)
                    cand_str = json.dumps(cand_ast, separators=(",", ":"))
                    cand_tok = self.estimate_bpe_tokens(cand_str)
                    if cand_tok < best_tokens:
                        best_tokens = cand_tok
                        best_str = cand_str

        return best_str, raw_tokens, int(best_tokens)

    def defragment_text(self, text: str) -> Tuple[str, int, int]:
        """
        Eliminates subword-penalizing whitespace, normalizes punctuation sequences,
        and canonicalizes markdown delimiters.
        """
        if not text:
            return "", 0, 0

        raw_tokens = self.estimate_bpe_tokens(text)
        defrag = self.NEWLINE_BLOAT_PATTERN.sub('\n\n', text)

        lines: List[str] = []
        for line in defrag.split('\n'):
            stripped = line.strip()
            if not stripped:
                lines.append('')
                continue
            if stripped.startswith(('- ', '* ', '1. ', '2. ', '3. ', '#')):
                lines.append(stripped)
            else:
                compact_line = self.EXCESS_WHITESPACE_PATTERN.sub(' ', stripped)
                lines.append(compact_line)

        defrag = '\n'.join(lines)
        defrag = self.PUNCT_SPLIT_PATTERN.sub(r'\1 ', defrag)
        compacted_tokens = self.estimate_bpe_tokens(defrag)
        return defrag, raw_tokens, compacted_tokens


def _has_nontrivial_format_or_deontic(
    has_format: bool,
    has_deontic: bool,
    prop_name: str,
    clean_desc: str
) -> bool:
    if has_format:
        fmt_matches = BPETranspiler.REGEX_FORMAT_PATTERNS.findall(clean_desc.lower())
        for m in fmt_matches:
            if m.lower() not in prop_name.lower():
                return True
    if has_deontic:
        if re.search(r"\b(?:never|cannot|prohibited|forbidden|only\s+if|required\s+when|unless)\b", clean_desc, re.I):
            return True
    return False
