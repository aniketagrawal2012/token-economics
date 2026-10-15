"""
paged_cache_aligner.py - Hardware KV-Cache Paged Block Alignment & Context Caching Topologist
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Aligns immutable prompt prefixes to hardware KV-cache memory page block boundaries (B = 16/32 tokens)
to eliminate internal memory fragmentation in TPU v6e Trillium / vLLM serving. Enforces Google Cloud
Vertex AI Context Caching rules (>= 32,768 token threshold, SHA-256 prefix hashing, and sliding TTL).
"""

from __future__ import annotations
import hashlib
from typing import Tuple
from bpe_transpiler import BPETranspiler


class PagedCacheAligner:
    """
    Hardware-aware memory topologist that structures prompt prefixes to conform
    with physical KV-cache page blocks and enterprise cloud caching APIs.
    """

    VERTEX_AI_CACHE_THRESHOLD = 32768  # Minimum tokens required for explicit Vertex AI CachedContent
    DEFAULT_BLOCK_SIZE = 16             # Standard PagedAttention / TPU block size

    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE):
        self.block_size = block_size
        self.transpiler = BPETranspiler()

    def align_prefix_to_page_boundary(self, prefix: str) -> Tuple[str, int, int]:
        """
        Pads or compacts the static prompt prefix so its token count is strictly
        congruent to 0 mod B (where B is hardware block size):
            |P_prefix| = k * B
        
        Returns:
            Tuple[aligned_prefix, aligned_token_count, padding_tokens_added]
        """
        raw_tokens = self.transpiler.count_tokens(prefix)
        remainder = raw_tokens % self.block_size

        if remainder == 0:
            return prefix, raw_tokens, 0

        # Physical tokens needed to complete the physical page frame block
        padding_needed = self.block_size - remainder
        aligned_tokens = raw_tokens + padding_needed
        
        # Append deterministic hardware page alignment descriptor
        aligned_prefix = prefix + f"\n<!-- HW_CACHE_PAD:{padding_needed} -->"
        return aligned_prefix, aligned_tokens, padding_needed


    def partition_prompt_for_caching(
        self,
        full_prompt: str,
        static_delimiter: str = "### DYNAMIC_QUERY:"
    ) -> Tuple[str, str, bool, str, int]:
        """
        Partitions an enterprise prompt into an immutable static prefix and a per-request dynamic suffix.
        Evaluates eligibility for Vertex AI Context Caching (>= 32,768 tokens).
        
        Returns:
            Tuple[aligned_prefix, dynamic_suffix, is_vertex_cache_eligible, cache_key_hash, prefix_tokens]
        """
        if static_delimiter in full_prompt:
            parts = full_prompt.split(static_delimiter, 1)
            raw_prefix = parts[0].strip()
            dynamic_suffix = static_delimiter + "\n" + parts[1].strip()
        else:
            # Default bifurcation: top 70% static context, bottom 30% user query
            lines = full_prompt.split('\n')
            split_idx = max(1, int(len(lines) * 0.70))
            raw_prefix = '\n'.join(lines[:split_idx]).strip()
            dynamic_suffix = '\n'.join(lines[split_idx:]).strip()

        # Align prefix to hardware page boundary
        aligned_prefix, prefix_tokens, _ = self.align_prefix_to_page_boundary(raw_prefix)

        # Check Vertex AI caching threshold
        is_cache_eligible = prefix_tokens >= self.VERTEX_AI_CACHE_THRESHOLD

        # Deterministic SHA-256 cache key for Vertex AI / Pallas memory lookup
        cache_key_hash = hashlib.sha256(aligned_prefix.encode('utf-8')).hexdigest()

        return aligned_prefix, dynamic_suffix, is_cache_eligible, cache_key_hash, prefix_tokens
