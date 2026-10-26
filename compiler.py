"""
compiler.py - Unified Tokenomics Compiler & End-to-End Orchestrator
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Coordinates lexical BPE transpilation, mutual-information entropy pruning,
attention-sink landmark injection, hardware paged cache alignment, and Pareto routing.
"""

from __future__ import annotations
import json
from typing import Optional, Dict, Any, Tuple
from tokenomics_types import (
    PromptProfile,
    CompiledPrompt,
    TokenStats,
    RoutingDecision,
    EnterpriseCohort
)
from bpe_transpiler import BPETranspiler
from entropy_pruner import EntropyPruner
from attention_anchor import AttentionAnchorInjector
from paged_cache_aligner import PagedCacheAligner
from pareto_router import ParetoEconomicRouter


class TokenomicsCompiler:
    """
    Unified end-to-end compiler translating raw enterprise prompts into
    hardware-aligned, attention-stabilized, cost-optimal execution packages.
    """

    def __init__(self, block_size: int = 16, info_threshold: float = 0.30):
        self.transpiler = BPETranspiler()
        self.pruner = EntropyPruner(info_threshold=info_threshold)
        self.anchor_injector = AttentionAnchorInjector(sink_boost=1.75)
        self.cache_aligner = PagedCacheAligner(block_size=block_size)
        self.router = ParetoEconomicRouter()

    def compile(self, profile: PromptProfile) -> Tuple[CompiledPrompt, RoutingDecision]:
        """
        Executes the 5-stage compilation and routing pipeline:
        1. BPE schema and text canonicalization
        2. Mutual-information entropy pruning
        3. Attention-sink landmark injection
        4. Hardware page-block alignment
        5. Closed-form Pareto economic routing
        """
        # Assemble raw text
        raw_parts = [profile.raw_instruction]
        if profile.schema_definitions:
            raw_parts.append(json.dumps(profile.schema_definitions, indent=2))
        if profile.context_payload:
            raw_parts.append(profile.context_payload)

        raw_full_text = "\n\n".join(raw_parts)
        raw_tokens = self.transpiler.estimate_bpe_tokens(raw_full_text)

        # Stage 1 & 2: Transpile schemas & prune conversational noise
        pruned_instruction, _, _, _ = self.pruner.prune_prompt(profile.raw_instruction)

        canonical_schema_str = ""
        if profile.schema_definitions:
            canonical_schema_str, _, _ = self.transpiler.canonicalize_json_schema(profile.schema_definitions)

        # Stage 3: Inject attention-sink landmark anchors
        staged_text_parts = [pruned_instruction]
        if canonical_schema_str:
            staged_text_parts.append(f"### CANONICAL SCHEMA CONTRACT:\n{canonical_schema_str}")
        if profile.context_payload:
            defrag_ctx, _, _ = self.transpiler.defragment_text(profile.context_payload)
            staged_text_parts.append(f"### VERIFIED INPUT PAYLOAD:\n{defrag_ctx}")

        intermediate_text = "\n\n".join(staged_text_parts)
        anchored_text, anchors = self.anchor_injector.inject_anchors(intermediate_text)

        # Stage 4: Hardware KV-cache page boundary alignment
        aligned_prefix, dynamic_suffix, is_cache_eligible, cache_key, prefix_tokens = (
            self.cache_aligner.partition_prompt_for_caching(anchored_text)
        )

        compiled_full_prompt = f"{aligned_prefix}\n\n{dynamic_suffix}".strip()
        compacted_tokens = self.transpiler.estimate_bpe_tokens(compiled_full_prompt)

        compression_ratio = round(compacted_tokens / max(1, raw_tokens), 4)
        tokens_saved = max(0, raw_tokens - compacted_tokens)
        dynamic_tokens = max(0, compacted_tokens - prefix_tokens)

        # Compute formal attention entropy metrics
        attention_metrics = self.anchor_injector.calculate_attention_entropy(compacted_tokens, anchors)

        token_stats = TokenStats(
            raw_input_tokens=raw_tokens,
            compacted_input_tokens=compacted_tokens,
            compression_ratio=compression_ratio,
            tokens_saved=tokens_saved,
            prefix_tokens=prefix_tokens,
            dynamic_suffix_tokens=dynamic_tokens,
            cache_alignment_padding_tokens=prefix_tokens % self.cache_aligner.block_size,
            effective_cache_hit_rate=0.85 if is_cache_eligible else 0.0
        )

        compiled_prompt = CompiledPrompt(
            request_id=profile.request_id,
            raw_prompt=raw_full_text,
            compiled_prompt=compiled_full_prompt,
            static_prefix=aligned_prefix,
            dynamic_suffix=dynamic_suffix,
            stats=token_stats,
            attention_metrics=attention_metrics,
            anchors_injected=anchors,
            paged_cache_block_size=self.cache_aligner.block_size,
            is_cache_eligible=is_cache_eligible
        )

        # Stage 5: Closed-Form Pareto Economic Routing
        routing_decision = self.router.optimize_route(
            request_id=profile.request_id,
            cohort=profile.cohort,
            input_tokens=compacted_tokens,
            estimated_output_tokens=None,
            target_quality=profile.target_quality,
            max_latency_ms=profile.max_latency_ms,
            is_cached=is_cache_eligible,
            raw_uncompacted_input_tokens=raw_tokens,
            prefix_tokens=prefix_tokens
        )

        return compiled_prompt, routing_decision
