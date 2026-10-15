"""
types.py - Pydantic Data Contracts and Mathematical Types for Tokenomics Compiler
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator


class ModelTier(str, Enum):
    """Supported model serving tiers with distinct FLOPs, pricing, and latency regimes."""
    GEMINI_38_FLASH_LITE = "gemini-3.8-flash-lite"
    GEMINI_38_FLASH = "gemini-3.8-flash"
    GEMINI_25_PRO = "gemini-2.5-pro"
    CLAUDE_35_SONNET_BASELINE = "claude-3.5-sonnet-baseline"
    GPT_4O_MINI_BASELINE = "gpt-4o-mini-baseline"


class PricingSpec(BaseModel):
    """Hardware inference pricing specification per million (1M) tokens in USD."""
    input_per_million: float = Field(..., description="Cost per 1M standard input tokens in USD")
    output_per_million: float = Field(..., description="Cost per 1M output tokens in USD")
    cached_input_per_million: float = Field(..., description="Cost per 1M cached prefix tokens in USD")
    thinking_per_million: float = Field(..., description="Cost per 1M test-time thinking/reasoning tokens in USD")

    @classmethod
    def get_tier_pricing(cls, tier: ModelTier) -> "PricingSpec":
        """Official Google Cloud Vertex AI & frontier model pricing reference specs."""
        pricing_map = {
            ModelTier.GEMINI_38_FLASH_LITE: cls(
                input_per_million=0.075,
                output_per_million=0.30,
                cached_input_per_million=0.01875,
                thinking_per_million=0.30
            ),
            ModelTier.GEMINI_38_FLASH: cls(
                input_per_million=0.75,
                output_per_million=3.00,
                cached_input_per_million=0.1875,
                thinking_per_million=3.00
            ),
            ModelTier.GEMINI_25_PRO: cls(
                input_per_million=1.25,
                output_per_million=5.00,
                cached_input_per_million=0.3125,
                thinking_per_million=5.00
            ),
            ModelTier.CLAUDE_35_SONNET_BASELINE: cls(
                input_per_million=3.00,
                output_per_million=15.00,
                cached_input_per_million=0.30,
                thinking_per_million=15.00
            ),
            ModelTier.GPT_4O_MINI_BASELINE: cls(
                input_per_million=0.15,
                output_per_million=0.60,
                cached_input_per_million=0.075,
                thinking_per_million=0.60
            )
        }
        return pricing_map[tier]


class TokenStats(BaseModel):
    """Fine-grained token decomposition across prompt compilation stages."""
    raw_input_tokens: int = Field(..., ge=0, description="Raw unoptimized input token count")
    compacted_input_tokens: int = Field(..., ge=0, description="Compacted input token count after BPE and entropy pruning")
    compression_ratio: float = Field(..., ge=0.0, le=2.0, description="Compacted / Raw token ratio (rho)")
    tokens_saved: int = Field(..., ge=0, description="Absolute input tokens saved")
    prefix_tokens: int = Field(..., ge=0, description="Static prefix tokens eligible for KV-cache reuse")
    dynamic_suffix_tokens: int = Field(..., ge=0, description="Dynamic suffix tokens executed per request")
    cache_alignment_padding_tokens: int = Field(default=0, ge=0, description="Hardware KV-cache page boundary padding")
    effective_cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Observed KV-cache hit rate")


class AttentionMetrics(BaseModel):
    """Information-theoretic and attention entropy distribution metrics."""
    raw_attention_entropy: float = Field(..., description="Shannon entropy of raw attention distribution (nats or bits)")
    anchored_attention_entropy: float = Field(..., description="Entropy after structural landmark injection")
    entropy_reduction_ratio: float = Field(..., description="Relative entropy reduction (H_raw - H_anchor) / H_raw")
    attention_sink_density: float = Field(..., description="Ratio of probability mass concentrated at landmark anchors")
    dispersion_bounded: bool = Field(..., description="Whether attention dispersion is provably bounded by H_max")


class EnterpriseCohort(str, Enum):
    """Five distinct enterprise evaluation strata for empirical validation."""
    COHORT_A_SQL_DB = "cohort_a_sql_database_query"
    COHORT_B_AGENT_OPENAPI = "cohort_b_multi_tool_openapi_agent"
    COHORT_C_FINANCIAL_COMPLIANCE = "cohort_c_financial_compliance_ledger"
    COHORT_D_LONG_CONTEXT_CODE = "cohort_d_long_context_code_refactoring"
    COHORT_E_MULTI_TURN_REASONING = "cohort_e_multi_turn_dynamic_reasoning"


class PromptProfile(BaseModel):
    """Incoming enterprise prompt profile subject to compilation and routing."""
    request_id: str
    cohort: EnterpriseCohort
    raw_instruction: str
    schema_definitions: Optional[Dict[str, Any]] = None
    context_payload: Optional[str] = None
    target_quality: float = Field(default=0.95, ge=0.0, le=1.0, description="Minimum acceptable quality threshold (Q*)")
    max_latency_ms: float = Field(default=2000.0, gt=0.0, description="Hard SLA latency constraint (L_max)")
    max_budget_usd: Optional[float] = Field(default=None, gt=0.0, description="Budget cap per invocation")


class AttentionAnchor(BaseModel):
    """Structural landmark token anchor injected to bound Transformer attention dispersion."""
    tag: str = Field(..., description="Uppercase invariant header tag, e.g. ### MANDATORY INVARIANTS:")
    invariant_type: str = Field(..., description="Category of invariant: CONSTRAINT, SCHEMA, TERMINATION, ESCROW")
    byte_offset: int = Field(..., ge=0)
    embedding_weight_boost: float = Field(default=1.5, ge=1.0, description="Relative attention sink multiplier")


class CompiledPrompt(BaseModel):
    """The fully optimized, hardware-aligned prompt output ready for model dispatch."""
    request_id: str
    raw_prompt: str
    compiled_prompt: str
    static_prefix: str
    dynamic_suffix: str
    stats: TokenStats
    attention_metrics: AttentionMetrics
    anchors_injected: List[AttentionAnchor]
    paged_cache_block_size: int = Field(default=16, description="Hardware KV-cache allocation page size (e.g. 16 or 32)")
    is_cache_eligible: bool = Field(default=False, description="Whether prefix meets Vertex AI caching threshold (>= 32,768)")


class RoutingDecision(BaseModel):
    """Optimal execution route chosen by the convex Pareto economic optimizer."""
    request_id: str
    selected_tier: ModelTier
    thinking_budget: int = Field(default=0, ge=0, description="Test-time thinking token budget allocated")
    estimated_cost_usd: float = Field(..., ge=0.0)
    baseline_monolithic_cost_usd: float = Field(..., ge=0.0)
    net_cost_savings_pct: float = Field(..., description="Cost reduction percentage vs Gemini 2.5 Pro monolithic")
    estimated_latency_ms: float = Field(..., ge=0.0)
    expected_quality_score: float = Field(..., ge=0.0, le=1.0)
    meets_sla: bool = Field(default=True)


class BenchmarkTrace(BaseModel):
    """Empirical telemetry trace record for statistical evaluation and audit."""
    request_id: str
    cohort: EnterpriseCohort
    model_tier: ModelTier
    raw_input_tokens: int
    compacted_input_tokens: int
    output_tokens: int
    thinking_tokens: int
    raw_cost_usd: float
    compacted_cost_usd: float
    cost_savings_pct: float
    ttft_ms: float
    inter_token_latency_ms: float
    total_latency_ms: float
    task_success: bool
    quality_score: float
    attention_entropy: float
    cache_hit: bool
