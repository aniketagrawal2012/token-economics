"""
token_economics.py - Unified Top-Level Facade for the Tokenomics Compiler (PromptVM)
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Re-exports all core data contracts, compilation stages (Stages 1-4), and the
Stage 5 discrete-choice Pareto economic router.
"""

from tokenomics_types import (
    ModelTier,
    PricingSpec,
    TokenStats,
    AttentionMetrics,
    EnterpriseCohort,
    PromptProfile,
    AttentionAnchor,
    CompiledPrompt,
    RoutingDecision,
    BenchmarkTrace,
)
from bpe_transpiler import BPETranspiler
from entropy_pruner import EntropyPruner
from attention_anchor import AttentionAnchorInjector
from paged_cache_aligner import PagedCacheAligner
from pareto_router import ParetoEconomicRouter
from compiler import TokenomicsCompiler

__all__ = [
    "ModelTier",
    "PricingSpec",
    "TokenStats",
    "AttentionMetrics",
    "EnterpriseCohort",
    "PromptProfile",
    "AttentionAnchor",
    "CompiledPrompt",
    "RoutingDecision",
    "BenchmarkTrace",
    "BPETranspiler",
    "EntropyPruner",
    "AttentionAnchorInjector",
    "PagedCacheAligner",
    "ParetoEconomicRouter",
    "TokenomicsCompiler",
]
