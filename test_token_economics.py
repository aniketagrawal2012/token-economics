"""
test_token_economics.py - Comprehensive Automated Test Suite (unittest)
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Deterministic unit testing covering all 5 pillars: BPE transpilation (with 4-part description
guard, union-type folding, and 8-candidate AST search), entropy pruning (with sub-clause
pleasantry span stripping, Eq. 6 symbolic operator weight 1.0, and polarity-aware 4-tuple
deduplication), attention anchoring (Theorem 1), paged-cache alignment, and Pareto routing (Eq. 17).
"""

import unittest
import json
import math
import sys
import os

# Ensure current directory is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenomics_types import (
    ModelTier,
    PricingSpec,
    EnterpriseCohort,
    PromptProfile,
    CompiledPrompt,
    RoutingDecision
)
from bpe_transpiler import BPETranspiler
from entropy_pruner import EntropyPruner
from attention_anchor import AttentionAnchorInjector
from paged_cache_aligner import PagedCacheAligner
from pareto_router import ParetoEconomicRouter
from compiler import TokenomicsCompiler


class TestTokenomicsCompiler(unittest.TestCase):

    def test_01_pricing_specs_and_tiers(self):
        """Verify all model tiers have valid pricing, cached discounts, and positive rates."""
        for tier in ModelTier:
            pricing = PricingSpec.get_tier_pricing(tier)
            self.assertGreater(pricing.input_per_million, 0.0)
            self.assertGreater(pricing.output_per_million, 0.0)
            self.assertGreater(pricing.cached_input_per_million, 0.0)
            self.assertLess(pricing.cached_input_per_million, pricing.input_per_million)

    def test_02_bpe_transpiler_token_estimation(self):
        """Verify subword token estimation behaves accurately on various syntaxes."""
        transpiler = BPETranspiler()
        self.assertEqual(transpiler.estimate_bpe_tokens(""), 0)
        self.assertGreaterEqual(transpiler.estimate_bpe_tokens("Hello world"), 2)

        tokens_camel = transpiler.estimate_bpe_tokens("EnterpriseOrderManagementSystemProxy")
        self.assertGreaterEqual(tokens_camel, 5)

        spaced_text = "word      anotherword"
        self.assertGreater(transpiler.estimate_bpe_tokens(spaced_text), transpiler.estimate_bpe_tokens("word anotherword"))

    def test_03_json_schema_canonicalization_and_4part_guard(self):
        """Verify verbose OpenAPI schemas are compressed by >= 25% while preserving 4-part semantic guards and folding unions."""
        transpiler = BPETranspiler()
        raw_schema = {
            "title": "DatabaseQueryParameters",
            "description": "A very long descriptive paragraph explaining in unnecessary detail what this schema is intended for.",
            "type": "object",
            "properties": {
                "query_string": {
                    "type": "string",
                    "description": "The exact SQL query string that must be executed against the Google Cloud Spanner database instance."
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Optional timeout parameter specifying maximum execution duration between 1 and 300 seconds."
                },
                "isolation_level": {
                    "type": "string",
                    "enum": ["SERIALIZABLE", "READ_COMMITTED"],
                    "description": "Transaction isolation level required for ACID consistency."
                },
                "cursor_token": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Optional pagination cursor."
                }
            },
            "required": ["query_string", "isolation_level"]
        }

        canonical_str, raw_tok, comp_tok = transpiler.canonicalize_json_schema(raw_schema)
        self.assertLess(comp_tok, raw_tok)
        reduction_pct = (raw_tok - comp_tok) / raw_tok * 100.0
        self.assertGreaterEqual(reduction_pct, 25.0, f"Expected >= 25% reduction, got {reduction_pct:.2f}%")

        parsed = json.loads(canonical_str)
        self.assertIn("query_string", parsed["properties"])
        self.assertIn("isolation_level", parsed["properties"])
        self.assertEqual(parsed["properties"]["isolation_level"]["enum"], ["SERIALIZABLE", "READ_COMMITTED"])
        self.assertEqual(parsed["required"], ["query_string", "isolation_level"])
        # Verify union folding: anyOf -> ["string", "null"]
        self.assertEqual(parsed["properties"]["cursor_token"]["type"], ["string", "null"])
        # Verify numeric bound guard preserved on timeout_seconds
        self.assertIn("description", parsed["properties"]["timeout_seconds"])

    def test_04_entropy_pruner_eliminates_conversational_fluff_and_retains_short_symbolic(self):
        """Verify polite greetings are stripped, mixed clauses keep constraints, and short symbolic clauses pass Eq. (6)."""
        pruner = EntropyPruner(info_threshold=0.30)
        fluffy_prompt = (
            "Hello dear AI assistant! I hope you are having a wonderful day.\n"
            "Can you please kindly help me with a critical database query?\n"
            "You must select the customer_id and transaction_total from the accounts table.\n"
            "The where condition must filter status = 'ACTIVE'.\n"
            "x >= 0 and y != 1\n"
            "Please return the output strictly formatted as valid JSON.\n"
            "Thank you so much in advance, I really appreciate your help!"
        )

        pruned_text, raw_tok, pruned_tok, ratio = pruner.prune_prompt(fluffy_prompt)
        self.assertLess(pruned_tok, raw_tok)
        self.assertIn("customer_id", pruned_text)
        self.assertIn("accounts table", pruned_text)
        self.assertIn("status = 'ACTIVE'", pruned_text)
        self.assertIn("x >= 0 and y != 1", pruned_text)
        self.assertIn("strictly formatted as valid JSON", pruned_text)
        self.assertNotIn("having a wonderful day", pruned_text)
        self.assertNotIn("really appreciate your help", pruned_text)

    def test_05_entropy_pruner_polarity_aware_deduplication(self):
        """Verify redundant repeating constraints are consolidated while opposite-polarity constraints are preserved."""
        pruner = EntropyPruner(info_threshold=0.30)
        repetitive_prompt = (
            "You must return the output formatted strictly as JSON.\n"
            "The query must filter where balance >= 1000.\n"
            "The query must filter where balance != 1000.\n"
            "Please make sure the response is returned strictly as JSON format.\n"
            "The output response must be strictly in JSON."
        )

        pruned_text, raw_tok, pruned_tok, _ = pruner.prune_prompt(repetitive_prompt)
        json_mentions = [line for line in pruned_text.split('\n') if 'json' in line.lower()]
        self.assertEqual(len(json_mentions), 1)
        # Both balance >= 1000 (+1 polarity) and balance != 1000 (-1 polarity) must be retained
        self.assertIn("balance >= 1000", pruned_text)
        self.assertIn("balance != 1000", pruned_text)

    def test_06_attention_anchor_injection(self):
        """Verify 7-token structural uppercase landmark anchors are injected."""
        injector = AttentionAnchorInjector(sink_boost=1.75)
        raw_prompt = (
            "You must follow these rules strictly:\n"
            "- Never disclose internal API keys\n"
            "- Validate all customer IDs before dispatch\n\n"
            "The output format specification requires JSON with fields code and message."
        )

        anchored_text, anchors = injector.inject_anchors(raw_prompt)
        self.assertGreaterEqual(len(anchors), 1)
        self.assertIn("### MANDATORY EXECUTION INVARIANTS:", anchored_text)
        self.assertIn("### REQUIRED OUTPUT SPECIFICATION:", anchored_text)

    def test_07_attention_entropy_reduction(self):
        """Verify Shannon entropy of anchored prompt is strictly bounded by Theorem 1 H_max and lower than ln(i)."""
        injector = AttentionAnchorInjector(sink_boost=1.75)
        _, anchors = injector.inject_anchors("You must obey rule 1 and rule 2. Return format is JSON.")
        metrics = injector.calculate_attention_entropy(num_tokens=500, anchors=anchors)

        self.assertTrue(metrics.dispersion_bounded)
        self.assertLess(metrics.anchored_attention_entropy, metrics.raw_attention_entropy)
        self.assertGreater(metrics.entropy_reduction_ratio, 0.0)
        self.assertGreater(metrics.attention_sink_density, 0.0)

    def test_08_paged_cache_block_boundary_alignment(self):
        """Verify prefix length is strictly congruent to 0 mod B (where B=16)."""
        aligner = PagedCacheAligner(block_size=16)
        prefix = "This is a static system instruction prefix containing enterprise policies."
        aligned_prefix, aligned_tokens, padding_added = aligner.align_prefix_to_page_boundary(prefix)

        self.assertEqual(aligned_tokens % 16, 0)
        self.assertGreaterEqual(padding_added, 0)
        self.assertTrue("<!-- HW_CACHE_PAD:" in aligned_prefix or padding_added == 0)

    def test_09_vertex_ai_cache_threshold_detection(self):
        """Verify prompts >= 32,768 tokens are flagged as cache eligible and SHA-256 is computed."""
        aligner = PagedCacheAligner(block_size=16)
        small_prompt = "Short prompt text for immediate execution."
        _, _, is_eligible_small, hash_small, _ = aligner.partition_prompt_for_caching(small_prompt)
        self.assertFalse(is_eligible_small)
        self.assertEqual(len(hash_small), 64)

        large_prefix = "word " * 33000
        large_prompt = f"{large_prefix}\n### DYNAMIC_QUERY:\nUser question here"
        _, _, is_eligible_large, hash_large, tokens_large = aligner.partition_prompt_for_caching(large_prompt)
        self.assertTrue(is_eligible_large)
        self.assertGreaterEqual(tokens_large, 32768)
        self.assertEqual(len(hash_large), 64)

    def test_10_pareto_router_cost_minimization_and_closed_form_thinking(self):
        """Verify router picks cheapest tier satisfying Q* and Eq. (17) yields T_think* = 16 for Cohort C."""
        router = ParetoEconomicRouter()
        decision_a = router.optimize_route(
            request_id="req_test_01",
            cohort=EnterpriseCohort.COHORT_A_SQL_DB,
            input_tokens=500,
            estimated_output_tokens=100,
            target_quality=0.85,
            max_latency_ms=2000.0,
            is_cached=False
        )
        self.assertEqual(decision_a.selected_tier, ModelTier.GEMINI_38_FLASH_LITE)
        self.assertTrue(decision_a.meets_sla)
        self.assertGreater(decision_a.net_cost_savings_pct, 70.0)

        # Verify Cohort C (Q* = 0.96, Q_base(Flash, C) = 0.94, Delta_Q_max = 0.06 -> Omega = 1/3 -> ceil(4097^(1/3)-1) = 16)
        t_think_c, q_c = router.compute_closed_form_thinking_budget(
            tier=ModelTier.GEMINI_38_FLASH,
            cohort=EnterpriseCohort.COHORT_C_FINANCIAL_COMPLIANCE,
            target_quality=0.96,
            c_task=1.0
        )
        self.assertEqual(t_think_c, 16)
        self.assertGreaterEqual(q_c, 0.96)

    def test_11_pareto_router_thinking_budget_boost(self):
        """Verify Gemini 3.8 Flash boosts quality using test-time thinking tokens."""
        router = ParetoEconomicRouter()
        decision = router.optimize_route(
            request_id="req_test_02",
            cohort=EnterpriseCohort.COHORT_D_LONG_CONTEXT_CODE,
            input_tokens=1500,
            estimated_output_tokens=300,
            target_quality=0.95,
            max_latency_ms=3500.0,
            is_cached=False
        )
        self.assertIn(decision.selected_tier, (ModelTier.GEMINI_38_FLASH, ModelTier.GEMINI_25_PRO))
        self.assertGreaterEqual(decision.expected_quality_score, 0.95)
        self.assertTrue(decision.meets_sla)

    def test_12_end_to_end_compiler_pipeline(self):
        """Verify complete end-to-end compilation of a realistic enterprise SQL prompt."""
        compiler = TokenomicsCompiler(block_size=16)
        profile = PromptProfile(
            request_id="req_e2e_001",
            cohort=EnterpriseCohort.COHORT_A_SQL_DB,
            raw_instruction=(
                "Hello team! Please make sure to carefully write an optimized SQL query for our analytics dashboard.\n"
                "You must select the transaction_id, user_email, and amount_usd from the transactions table.\n"
                "The filter condition must be timestamp >= '2026-01-01' and status = 'COMPLETED'.\n"
                "Please return the output strictly formatted as valid JSON.\n"
                "Output must be strictly JSON format.\n"
                "Thank you so much!"
            ),
            schema_definitions={
                "description": "SQL query parameters",
                "type": "object",
                "properties": {
                    "sql_query": {"type": "string", "description": "The exact executable SQL statement."},
                    "row_limit": {"type": "integer", "description": "Maximum rows to return, default 100."}
                },
                "required": ["sql_query"]
            },
            target_quality=0.92,
            max_latency_ms=2000.0
        )

        compiled_prompt, decision = compiler.compile(profile)

        self.assertGreater(compiled_prompt.stats.raw_input_tokens, compiled_prompt.stats.compacted_input_tokens)
        self.assertGreater(compiled_prompt.stats.tokens_saved, 0)
        self.assertLess(compiled_prompt.stats.compression_ratio, 0.85)
        self.assertGreaterEqual(len(compiled_prompt.anchors_injected), 1)
        self.assertTrue(compiled_prompt.attention_metrics.dispersion_bounded)
        self.assertLess(compiled_prompt.attention_metrics.anchored_attention_entropy, compiled_prompt.attention_metrics.raw_attention_entropy)
        self.assertTrue(decision.meets_sla)
        self.assertGreater(decision.net_cost_savings_pct, 30.0)
        self.assertGreaterEqual(decision.expected_quality_score, 0.90)


if __name__ == "__main__":
    unittest.main()
