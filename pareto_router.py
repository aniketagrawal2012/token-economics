"""
pareto_router.py - Convex Pareto Economic Router & Thinking Budget Allocator
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Solves the constrained multi-objective optimization problem over model tiers
(Gemini 3.8 Flash-Lite, Gemini 3.8 Flash, Gemini 2.5 Pro) and test-time reasoning tokens
(thinking_budget) to achieve provably minimal cost under hard SLA and quality bounds.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
from tokenomics_types import ModelTier, PricingSpec, RoutingDecision, EnterpriseCohort


class ParetoEconomicRouter:
    """
    Mathematical optimizer that dynamically selects the cheapest model tier
    and thinking token allocation satisfying Q >= Q* and Latency <= L_max.
    """

    # Baseline performance parameters by tier
    TIER_SPECS = {
        ModelTier.GEMINI_38_FLASH_LITE: {
            "base_quality": 0.90,
            "max_thinking_boost": 0.04,
            "ttft_cold_ms_per_1k": 60.0,
            "ttft_cached_ms": 30.0,
            "itl_ms_per_token": 4.0,
            "thinking_itl_ms_per_token": 2.0,
            "max_thinking_tokens": 1024
        },
        ModelTier.GEMINI_38_FLASH: {
            "base_quality": 0.95,
            "max_thinking_boost": 0.06,  # Scales up to 0.99 with test-time reasoning
            "ttft_cold_ms_per_1k": 80.0,
            "ttft_cached_ms": 45.0,
            "itl_ms_per_token": 5.0,
            "thinking_itl_ms_per_token": 2.5,
            "max_thinking_tokens": 4096
        },
        ModelTier.GEMINI_25_PRO: {
            "base_quality": 0.985,
            "max_thinking_boost": 0.015,
            "ttft_cold_ms_per_1k": 180.0,
            "ttft_cached_ms": 90.0,
            "itl_ms_per_token": 12.0,
            "thinking_itl_ms_per_token": 6.0,
            "max_thinking_tokens": 8192
        }
    }

    # Complexity penalty by cohort
    COHORT_COMPLEXITY_FACTOR = {
        EnterpriseCohort.COHORT_A_SQL_DB: 0.01,
        EnterpriseCohort.COHORT_B_AGENT_OPENAPI: 0.02,
        EnterpriseCohort.COHORT_C_FINANCIAL_COMPLIANCE: 0.015,
        EnterpriseCohort.COHORT_D_LONG_CONTEXT_CODE: 0.025,
        EnterpriseCohort.COHORT_E_MULTI_TURN_REASONING: 0.03
    }

    def __init__(self):
        pass

    def compute_cost(
        self,
        tier: ModelTier,
        input_tokens: int,
        output_tokens: int,
        thinking_tokens: int = 0,
        is_cached: bool = False
    ) -> float:
        """Computes exact invoice cost in USD for a given inference invocation."""
        pricing = PricingSpec.get_tier_pricing(tier)
        if is_cached:
            in_cost = (input_tokens / 1e6) * pricing.cached_input_per_million
        else:
            in_cost = (input_tokens / 1e6) * pricing.input_per_million

        out_cost = (output_tokens / 1e6) * pricing.output_per_million
        think_cost = (thinking_tokens / 1e6) * pricing.thinking_per_million
        return in_cost + out_cost + think_cost

    def estimate_latency(
        self,
        tier: ModelTier,
        input_tokens: int,
        output_tokens: int,
        thinking_tokens: int = 0,
        is_cached: bool = False
    ) -> float:
        """Estimates end-to-end inference latency in milliseconds."""
        specs = self.TIER_SPECS[tier]
        if is_cached:
            ttft = specs["ttft_cached_ms"]
        else:
            ttft = (input_tokens / 1000.0) * specs["ttft_cold_ms_per_1k"]

        think_itl = specs.get("thinking_itl_ms_per_token", specs["itl_ms_per_token"])
        itl = (output_tokens * specs["itl_ms_per_token"]) + (thinking_tokens * think_itl)
        return ttft + itl

    def estimate_quality(
        self,
        tier: ModelTier,
        cohort: EnterpriseCohort,
        thinking_tokens: int = 0
    ) -> float:
        """Models quality score Q(M, T_think) based on base model ability and reasoning scaling."""
        specs = self.TIER_SPECS[tier]
        base_q = specs["base_quality"]
        complexity_penalty = self.COHORT_COMPLEXITY_FACTOR.get(cohort, 0.03)

        # Logarithmic test-time compute scaling: Delta Q = max_boost * log(1 + T) / log(1 + T_max)
        if thinking_tokens > 0:
            boost_factor = math.log(1.0 + thinking_tokens) / math.log(1.0 + specs["max_thinking_tokens"])
            reasoning_boost = specs["max_thinking_boost"] * boost_factor
        else:
            reasoning_boost = 0.0

        q_final = min(1.0, max(0.0, base_q + reasoning_boost - complexity_penalty))
        return round(q_final, 4)

    def optimize_route(
        self,
        request_id: str,
        cohort: EnterpriseCohort,
        input_tokens: int,
        estimated_output_tokens: int = 256,
        target_quality: float = 0.95,
        max_latency_ms: float = 2500.0,
        is_cached: bool = False
    ) -> RoutingDecision:
        """
        Solves the constrained Pareto optimization:
            min Cost(M, T_think)
            s.t. Quality(M, T_think) >= target_quality
                 Latency(M, T_think) <= max_latency_ms
        """
        # Baseline monolithic comparison: Gemini 2.5 Pro without caching or thinking optimization
        baseline_cost = self.compute_cost(
            ModelTier.GEMINI_25_PRO,
            input_tokens=input_tokens,
            output_tokens=estimated_output_tokens,
            thinking_tokens=0,
            is_cached=False
        )

        candidate_routes: List[Tuple[float, ModelTier, int, float, float]] = []

        # Candidate tiers to consider
        tiers_to_evaluate = [
            ModelTier.GEMINI_38_FLASH_LITE,
            ModelTier.GEMINI_38_FLASH,
            ModelTier.GEMINI_25_PRO
        ]

        # Candidate thinking token budgets to search
        thinking_token_steps = [0, 64, 128, 256, 512, 1024, 2048]

        for tier in tiers_to_evaluate:
            max_think = self.TIER_SPECS[tier]["max_thinking_tokens"]
            for think_tokens in thinking_token_steps:
                if think_tokens > max_think:
                    continue

                q = self.estimate_quality(tier, cohort, think_tokens)
                lat = self.estimate_latency(tier, input_tokens, estimated_output_tokens, think_tokens, is_cached)
                cost = self.compute_cost(tier, input_tokens, estimated_output_tokens, think_tokens, is_cached)

                # Check constraints
                if q >= target_quality and lat <= max_latency_ms:
                    candidate_routes.append((cost, tier, think_tokens, lat, q))

        if candidate_routes:
            # Sort by minimal cost
            candidate_routes.sort(key=lambda x: x[0])
            best_cost, best_tier, best_think, best_lat, best_q = candidate_routes[0]
            meets_sla = True
        else:
            # Fallback to Gemini 3.8 Flash with maximum thinking budget
            best_tier = ModelTier.GEMINI_38_FLASH
            best_think = 2048
            best_cost = self.compute_cost(best_tier, input_tokens, estimated_output_tokens, best_think, is_cached)
            best_lat = self.estimate_latency(best_tier, input_tokens, estimated_output_tokens, best_think, is_cached)
            best_q = self.estimate_quality(best_tier, cohort, best_think)
            meets_sla = False

        net_savings_pct = max(0.0, (baseline_cost - best_cost) / max(1e-9, baseline_cost)) * 100.0

        return RoutingDecision(
            request_id=request_id,
            selected_tier=best_tier,
            thinking_budget=best_think,
            estimated_cost_usd=round(best_cost, 6),
            baseline_monolithic_cost_usd=round(baseline_cost, 6),
            net_cost_savings_pct=round(net_savings_pct, 2),
            estimated_latency_ms=round(best_lat, 2),
            expected_quality_score=round(best_q, 4),
            meets_sla=meets_sla
        )
