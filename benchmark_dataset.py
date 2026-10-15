"""
benchmark_dataset.py - Enterprise Prompt Dataset Generator (N = 1,000 Traces)
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Generates N = 1,000 authentic enterprise prompt profiles stratified into 5 distinct
operational cohorts (n = 200 each) representing Google Cloud Spanner, Vertex AI,
BigQuery, and multi-agent tool execution workloads.
"""

from __future__ import annotations
import random
from typing import List, Dict, Any
from tokenomics_types import EnterpriseCohort, PromptProfile


class BenchmarkDatasetGenerator:
    """
    Synthesizes authentic, highly realistic enterprise prompts with complex schemas,
    conversational bloat, hard constraints, and variable context payloads.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate_cohort_a_sql(self, n: int = 200) -> List[PromptProfile]:
        """Cohort A: Structured SQL & Database Query Generation against Cloud Spanner."""
        tables = ["orders", "accounts", "settlements", "inventory", "ledger_entries", "audit_log"]
        columns = ["customer_id", "transaction_total", "created_at", "status", "currency", "merchant_id"]
        statuses = ["'ACTIVE'", "'COMPLETED'", "'PENDING_SETTLEMENT'", "'FLAGGED_FRAUD'"]
        
        profiles = []
        for i in range(n):
            table = random.choice(tables)
            cols = random.sample(columns, k=3)
            status = random.choice(statuses)
            limit = random.choice([50, 100, 500, 1000])

            raw_text = (
                f"Hello Google Cloud support team! Please make sure to write a high-performance Cloud Spanner SQL query.\n"
                f"I need you to select the fields {', '.join(cols)} from the {table} table.\n"
                f"The filter condition must strictly be status = {status} and created_at >= '2026-01-01'.\n"
                f"Order by transaction_total DESC with a hard limit of {limit} rows.\n"
                f"Please ensure that the query is formatted cleanly. Output must be strictly JSON format.\n"
                f"Return the response formatted strictly as JSON with key 'sql_statement'.\n"
                f"Thank you so much for your assistance, hope you have a great day!"
            )
            schema = {
                "title": "SpannerQueryRequest",
                "description": "Parameters for executing a transactional read-only query on Google Cloud Spanner.",
                "type": "object",
                "properties": {
                    "sql_statement": {"type": "string", "description": "The exact SQL string."},
                    "query_mode": {"type": "string", "enum": ["NORMAL", "PROFILE", "PLAN"]},
                    "priority": {"type": "string", "enum": ["PRIORITY_HIGH", "PRIORITY_LOW"]}
                },
                "required": ["sql_statement"]
            }

            profiles.append(PromptProfile(
                request_id=f"cohort_a_sql_{i:04d}",
                cohort=EnterpriseCohort.COHORT_A_SQL_DB,
                raw_instruction=raw_text,
                schema_definitions=schema,
                target_quality=0.92,
                max_latency_ms=1800.0
            ))
        return profiles

    def generate_cohort_b_agent_openapi(self, n: int = 200) -> List[PromptProfile]:
        """Cohort B: Multi-Tool Agent Orchestration with Rich OpenAPI Specifications."""
        tools = [
            ("charge_credit_card", ["amount_cents", "currency", "token", "idempotency_key"]),
            ("dispatch_warehouse_freight", ["shipment_id", "carrier_code", "pallet_count", "service_tier"]),
            ("update_customer_crm", ["account_id", "interaction_type", "representative_notes", "escalate_flag"]),
            ("issue_refund_voucher", ["transaction_id", "reason_code", "supervisor_pin", "override_approved"])
        ]

        profiles = []
        for i in range(n):
            tool_name, tool_params = random.choice(tools)
            raw_text = (
                f"Greetings agent controller. Kindly process this user transaction via available API tools.\n"
                f"The customer wants to invoke {tool_name} with specific parameter constraints.\n"
                f"You must strictly validate that all required parameters {', '.join(tool_params)} are present.\n"
                f"Never execute without a verified idempotency key. Do not hallucinate parameter types.\n"
                f"Please return strictly valid JSON matching the OpenAPI tool dispatch specification.\n"
                f"The response must be in JSON format. Do not include markdown codeblocks or apologies."
            )
            schema = {
                "title": f"{tool_name}_Spec",
                "description": f"OpenAPI 3.1 specification for enterprise microservice {tool_name}.",
                "type": "object",
                "properties": {param: {"type": "string", "description": f"Value for {param}."} for param in tool_params},
                "required": tool_params[:3]
            }

            profiles.append(PromptProfile(
                request_id=f"cohort_b_agent_{i:04d}",
                cohort=EnterpriseCohort.COHORT_B_AGENT_OPENAPI,
                raw_instruction=raw_text,
                schema_definitions=schema,
                target_quality=0.95,
                max_latency_ms=2200.0
            ))
        return profiles

    def generate_cohort_c_financial_compliance(self, n: int = 200) -> List[PromptProfile]:
        """Cohort C: Financial & Regulatory Ledger Compliance Extraction."""
        profiles = []
        for i in range(n):
            raw_text = (
                f"Dear Compliance Auditor Assistant,\n"
                f"Please carefully audit the attached cross-border treasury journal entry.\n"
                f"You must extract the sender_bic, receiver_iban, settlement_currency, and net_amount.\n"
                f"Strict invariant: If net_amount > 1000000 USD, you must flag AML_SURVEILLANCE_TRIGGER = true.\n"
                f"Ensure the output satisfies SOX compliance requirements. Format strictly as JSON.\n"
                f"Return output formatted strictly as valid JSON.\n"
                f"Thank you for ensuring zero audit violations."
            )
            context = f"JOURNAL_PAYLOAD_HASH: {random.randint(10000000, 99999999)}\nLINE_ITEMS: [DEBIT 540000 EUR, CREDIT 585000 USD, FEE 450 USD]"
            schema = {
                "title": "AML_Audit_Record",
                "type": "object",
                "properties": {
                    "sender_bic": {"type": "string"},
                    "receiver_iban": {"type": "string"},
                    "aml_flag": {"type": "boolean"},
                    "net_amount": {"type": "number"}
                },
                "required": ["sender_bic", "aml_flag", "net_amount"]
            }

            profiles.append(PromptProfile(
                request_id=f"cohort_c_fin_{i:04d}",
                cohort=EnterpriseCohort.COHORT_C_FINANCIAL_COMPLIANCE,
                raw_instruction=raw_text,
                schema_definitions=schema,
                context_payload=context,
                target_quality=0.96,
                max_latency_ms=2400.0
            ))
        return profiles

    def generate_cohort_d_long_context_code(self, n: int = 200) -> List[PromptProfile]:
        """Cohort D: Long-Context Enterprise Codebase Modernization (Large Prefixes)."""
        profiles = []
        for i in range(n):
            # Generate substantial context payload modeling code files
            code_lines = [f"// Class Module_{j}: Legacy Java Enterprise Bean implementation with JTA transactions" for j in range(80)]
            context = "\n".join(code_lines)

            raw_text = (
                f"Hello engineering team! Please refactor this legacy Java enterprise service into modern idiomatic Go.\n"
                f"You must preserve transaction rollback semantics, thread-safety, and connection pool bounds.\n"
                f"Never use global state variables. Always return error values explicitly.\n"
                f"Format the final output strictly as JSON with keys 'go_code' and 'unit_tests'.\n"
                f"Return strictly JSON formatted response."
            )

            profiles.append(PromptProfile(
                request_id=f"cohort_d_code_{i:04d}",
                cohort=EnterpriseCohort.COHORT_D_LONG_CONTEXT_CODE,
                raw_instruction=raw_text,
                context_payload=context,
                target_quality=0.95,
                max_latency_ms=3500.0
            ))
        return profiles

    def generate_cohort_e_multi_turn_reasoning(self, n: int = 200) -> List[PromptProfile]:
        """Cohort E: Multi-Turn Dynamic Tool Self-Correction & Reasoning."""
        profiles = []
        for i in range(n):
            raw_text = (
                f"Agent Task Dispatcher Turn {i % 5 + 1}:\n"
                f"The previous tool call returned HTTP 412 Precondition Failed due to an outdated ETag.\n"
                f"You must read the latest entity version, reconcile the conflicting field, and retry.\n"
                f"Invariant: If retry_count >= 3, you must trip the circuit breaker and abort with code ESCALATE.\n"
                f"Please format your response strictly as JSON with action, payload, and retry_flag.\n"
                f"Return output strictly in JSON format."
            )
            schema = {
                "title": "AgentRetryResolution",
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["RETRY_MUTATION", "TRIP_CIRCUIT_BREAKER", "ESCALATE"]},
                    "payload": {"type": "object"},
                    "retry_flag": {"type": "boolean"}
                },
                "required": ["action", "retry_flag"]
            }

            profiles.append(PromptProfile(
                request_id=f"cohort_e_reason_{i:04d}",
                cohort=EnterpriseCohort.COHORT_E_MULTI_TURN_REASONING,
                raw_instruction=raw_text,
                schema_definitions=schema,
                target_quality=0.94,
                max_latency_ms=2500.0
            ))
        return profiles

    def generate_complete_dataset(self) -> List[PromptProfile]:
        """Combines all 5 cohorts into the canonical N = 1,000 trace dataset."""
        dataset = []
        dataset.extend(self.generate_cohort_a_sql(200))
        dataset.extend(self.generate_cohort_b_agent_openapi(200))
        dataset.extend(self.generate_cohort_c_financial_compliance(200))
        dataset.extend(self.generate_cohort_d_long_context_code(200))
        dataset.extend(self.generate_cohort_e_multi_turn_reasoning(200))
        return dataset
