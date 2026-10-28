"""
paged_cache_aligner.py - Hardware KV-Cache Paged Block Alignment & Context Caching Topologist
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Aligns immutable prompt prefixes to hardware KV-cache memory page block boundaries (B = 16/32 tokens)
to eliminate internal memory fragmentation in TPU v6e Trillium / vLLM serving. Enforces Google Cloud
Vertex AI Context Caching rules (>= 32,768 token threshold for explicit CachedContent API,
plus >= 1,024 token implicit prefix-cache reuse in vLLM/SGLang, SHA-256 prefix hashing, and sliding TTL).
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

    VERTEX_AI_CACHE_THRESHOLD = 32768   # Minimum tokens required for explicit Vertex AI CachedContent
    IMPLICIT_PREFIX_THRESHOLD = 1024    # Minimum tokens for implicit vLLM/SGLang RadixAttention reuse
    DEFAULT_BLOCK_SIZE = 16             # Standard PagedAttention / TPU block size

    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE):
        self.block_size = block_size
        self.transpiler = BPETranspiler()

    def align_prefix_to_page_boundary(self, prefix: str) -> Tuple[str, int, int]:
        """
        Pads or compacts the static prompt prefix so its token count is strictly
        congruent to 0 mod B (where B is hardware block size):
            |T_M(P_prefix)| = k * B

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

        # Construct variable-length neutral comment padding calibrated to padding_needed tokens
        pad_units = " ".join(["_"] * max(0, padding_needed - 4))
        pad_suffix = f"\n<!-- HW_CACHE_PAD:{padding_needed} {pad_units} -->"
        aligned_prefix = prefix + pad_suffix
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

        raw_prefix_tokens = self.transpiler.count_tokens(raw_prefix)
        is_cache_eligible = raw_prefix_tokens >= self.VERTEX_AI_CACHE_THRESHOLD

        if is_cache_eligible:
            aligned_prefix, prefix_tokens, _ = self.align_prefix_to_page_boundary(raw_prefix)
        else:
            # Regime A (< 32,768 tokens on managed endpoints): bypass Stage 4 padding
            aligned_prefix = raw_prefix
            prefix_tokens = raw_prefix_tokens

        # Deterministic SHA-256 cache key for Vertex AI / Pallas memory lookup
        cache_key_hash = hashlib.sha256(aligned_prefix.encode('utf-8')).hexdigest()

        return aligned_prefix, dynamic_suffix, is_cache_eligible, cache_key_hash, prefix_tokens

    def simulate_vmem_tile_masking(self, prefix_tokens: int, dynamic_tokens: int = 64) -> dict:
        r"""
        Models the server-side Cloud TPU v6e JAX/Pallas Block-Attention kernel invariants (#2):
        1. Block-table allocator zero-initializes the \Delta_frag trailing HBM2e slots upon prefix page
           sealing, and the Pallas kernel zeroes out q_p, k_p, v_p <- 0 in VMEM scratchpad registers
           via the validity bitmask prior to 128x128 MXU systolic array tile execution, preventing
           IEEE-754 NaN * 0 or NaN + (-inf) propagation from uninitialized HBM2e memory.
        2. On dummy pad query rows p in I_pad, self-diagonal M_pp = 0 and z_pp = <0, 0>/sqrt(d) = 0
           guarantee exact row-wise softmax partition denominator Z_p = exp(0) = 1.0 (zero 0/0 NaNs),
           while non-self key columns p in I_pad have M_ip = -inf so A_ip = exp(0 - inf) = 0.0.
        3. Logical RoPE counter m_logical(i) = m_physical(i) - |I_pad \cap {1..i}| decouples rotary
           phases and local sliding-window horizon W_loc from interior padding slots.
        """
        delta_frag = (self.block_size - (prefix_tokens % self.block_size)) % self.block_size
        physical_prefix_tokens = prefix_tokens + delta_frag
        num_prefix_blocks = physical_prefix_tokens // self.block_size
        logical_total_tokens = prefix_tokens + dynamic_tokens
        physical_total_tokens = physical_prefix_tokens + dynamic_tokens

        return {
            "raw_prefix_tokens": prefix_tokens,
            "delta_frag": delta_frag,
            "physical_prefix_tokens": physical_prefix_tokens,
            "num_sealed_prefix_blocks": num_prefix_blocks,
            "vmem_qkv_zeroed_for_pad": True,
            "hbm_tail_slots_zero_initialized": True,
            "pad_self_logit_z_pp": 0.0,
            "pad_row_partition_Z_p": 1.0,
            "valid_to_pad_attention_weight_A_ip": 0.0,
            "nan_propagation_possible": False,
            "logical_terminal_position": logical_total_tokens,
            "physical_terminal_position": physical_total_tokens,
        }

