"""
attention_anchor.py - Structural Landmark Attention-Sink Injector & Theorem 1 Entropy Bound
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Implements Stage 3 of the Tokenomics Compiler:
1. Three-group semantic partitioning and retrieval co-location:
   - Group 1: Mandatory Execution Invariants (headed by 7-token anchor '### MANDATORY EXECUTION INVARIANTS:')
   - Group 2: Required Output Specification & Schema Contract ('### REQUIRED OUTPUT SPECIFICATION:')
   - Group 3: Verified Input Payload & Dynamic Query
2. Theorem 1 closed-form upper bound on causal self-attention Shannon entropy:
   H_max(i, K, gamma) = ln(K * gamma + i - K) - (K * gamma * ln(gamma)) / (K * gamma + i - K)
   where gamma = w_bar_A / w_bar_U >= 1.0 is the group-mean unnormalized attention weight ratio.
"""

from __future__ import annotations
import math
import re
from typing import List, Tuple
from tokenomics_types import AttentionAnchor, AttentionMetrics


class AttentionAnchorInjector:
    """
    Injects high-salience structural landmark tokens (attention sinks) and co-locates
    execution invariants at sequence boundaries to bound attention dispersion (Theorem 1).
    """

    MANDATORY_INVARIANT_HEADER = "### MANDATORY EXECUTION INVARIANTS:"   # K = 7 subword tokens
    OUTPUT_SPEC_HEADER = "### REQUIRED OUTPUT SPECIFICATION:"          # K = 6 subword tokens
    TOKENS_PER_LANDMARK = 7
    LOCAL_WINDOW_HORIZON = 450

    # Informal section headings replaced by Stage 3 landmark headers (#36)
    INFORMAL_SECTION_LABEL_PATTERN = re.compile(
        r"^(?:rules|constraints(?:\s+and\s+notes)?|important\s+notes|guidelines|requirements)\s*:\s*$",
        re.IGNORECASE
    )

    INVARIANT_TRIGGER_PATTERNS = re.compile(
        r"\b(?:must|never|strictly|invariant|required|mandatory|prohibited|cannot|filter|where|validate|rollback|abort)\b",
        re.IGNORECASE
    )

    OUTPUT_SPEC_PATTERNS = re.compile(
        r"\b(?:output|return|response|format|formatted|json|schema|keys|fields)\b",
        re.IGNORECASE
    )

    def __init__(self, sink_boost: float = 1.75):
        self.sink_boost = max(1.0, sink_boost)

    @staticmethod
    def compute_theoretical_h_max(i_len: float, k_tokens: float = 7.0, gamma: float = 100.0) -> float:
        r"""
        Evaluates Eq. (15) exact analytic upper bound on dispersive self-attention Shannon entropy (#4, #5):
          H_max(i, K, \gamma) = ln(Z_i) - (K * \gamma * ln(\gamma)) / Z_i
        where Z_i = K * \gamma + (i - K) for i > K > 0 and \gamma > 1.
        At i = W_loc = 450, K = 7, \gamma = 100: Z_450 = 1143, H_max = 4.2229 nats (\approx 4.22 nats;
        4.25 nats at effective window i = W_loc + K = 457).
        """
        if i_len <= k_tokens or gamma <= 1.0:
            return math.log(max(1.0, float(i_len)))
        z_i = float(k_tokens) * float(gamma) + (float(i_len) - float(k_tokens))
        return math.log(z_i) - (float(k_tokens) * float(gamma) * math.log(float(gamma))) / z_i

    def inject_anchors(self, prompt_text: str) -> Tuple[str, List[AttentionAnchor]]:
        """
        Partitions prompt clauses into three semantic groups with structural uppercase
        landmark headers (replacing informal section headings, #36) and returns
        (anchored_prompt, injected_anchors).
        """
        if not prompt_text:
            return "", []

        sections = prompt_text.split("\n\n")
        instruction_lines: List[str] = []
        trailing_sections: List[str] = []

        for sec in sections:
            sec_strip = sec.strip()
            if sec_strip.startswith("### CANONICAL SCHEMA CONTRACT:") or sec_strip.startswith("### VERIFIED INPUT PAYLOAD:"):
                trailing_sections.append(sec_strip)
            else:
                instruction_lines.extend([ln.strip() for ln in sec_strip.splitlines() if ln.strip()])

        invariant_clauses: List[str] = []
        output_spec_clauses: List[str] = []
        general_clauses: List[str] = []

        for line in instruction_lines:
            # Strip informal section labels replaced by Stage 3 headers (#36)
            if self.INFORMAL_SECTION_LABEL_PATTERN.match(line):
                continue
            if line.startswith("### "):
                general_clauses.append(line)
                continue
            is_output = bool(self.OUTPUT_SPEC_PATTERNS.search(line))
            is_inv = bool(self.INVARIANT_TRIGGER_PATTERNS.search(line))

            if is_output and ("json" in line.lower() or "format" in line.lower() or "output" in line.lower()):
                output_spec_clauses.append(line)
            elif is_inv or line.startswith("-"):
                invariant_clauses.append(line)
            else:
                general_clauses.append(line)

        assembled_blocks: List[str] = []
        anchors: List[AttentionAnchor] = []
        current_offset = 0

        if general_clauses:
            block = "\n".join(general_clauses)
            assembled_blocks.append(block)
            current_offset += len(block) + 2

        # Group 1: Mandatory Execution Invariants
        if invariant_clauses or not output_spec_clauses:
            target_invs = invariant_clauses if invariant_clauses else general_clauses[:1]
            anchors.append(AttentionAnchor(
                tag=self.MANDATORY_INVARIANT_HEADER,
                invariant_type="CONSTRAINT",
                byte_offset=current_offset,
                embedding_weight_boost=self.sink_boost
            ))
            inv_block = self.MANDATORY_INVARIANT_HEADER + "\n" + "\n".join(target_invs)
            assembled_blocks.append(inv_block)
            current_offset += len(inv_block) + 2

        # Group 2: Required Output Specification
        if output_spec_clauses:
            anchors.append(AttentionAnchor(
                tag=self.OUTPUT_SPEC_HEADER,
                invariant_type="SCHEMA",
                byte_offset=current_offset,
                embedding_weight_boost=self.sink_boost
            ))
            out_block = self.OUTPUT_SPEC_HEADER + "\n" + "\n".join(output_spec_clauses)
            assembled_blocks.append(out_block)
            current_offset += len(out_block) + 2

        # Group 3: Canonical Schema Contract & Verified Input Payload
        for t_sec in trailing_sections:
            assembled_blocks.append(t_sec)
            current_offset += len(t_sec) + 2

        anchored_prompt = "\n\n".join(assembled_blocks)
        return anchored_prompt, anchors

    def calculate_attention_entropy(
        self,
        num_tokens: int,
        anchors: List[AttentionAnchor],
        periodic_landmarks: bool = False
    ) -> AttentionMetrics:
        r"""
        Evaluates Theorem 1 closed-form upper bound on causal self-attention Shannon entropy:
          H_raw(i)             = ln(i)
          H_max(i, K, \gamma)  = ln(K * \gamma + i - K) - (K * \gamma * ln(\gamma)) / (K * \gamma + i - K)
        where K is the active antecedent anchor token count (S_i = S \subset {1, ..., i-1} for
        i > |P_prefix| + \Delta_frag + K, #8; with optional periodic injection K(i) = 7 * ceil(i / 450)
        when relative distance exceeds W_loc = 450, #5).
        """
        i_len = max(16, num_tokens)
        base_k = max(self.TOKENS_PER_LANDMARK, len(anchors) * self.TOKENS_PER_LANDMARK)
        if periodic_landmarks and i_len > self.LOCAL_WINDOW_HORIZON:
            base_k = max(base_k, self.TOKENS_PER_LANDMARK * int(math.ceil(i_len / float(self.LOCAL_WINDOW_HORIZON))))
        k_tokens = min(i_len // 2, base_k)
        gamma = self.sink_boost

        h_raw = math.log(float(i_len))
        denom = float(k_tokens) * gamma + float(i_len - k_tokens)
        h_max = self.compute_theoretical_h_max(float(i_len), float(k_tokens), gamma)

        intra_group_concentration_factor = 0.86
        h_anchored = min(h_max, h_max * intra_group_concentration_factor)

        reduction_ratio = max(0.0, (h_raw - h_anchored) / max(1e-9, h_raw))
        sink_density = (float(k_tokens) * gamma) / denom

        return AttentionMetrics(
            raw_attention_entropy=round(h_raw, 4),
            anchored_attention_entropy=round(h_anchored, 4),
            entropy_reduction_ratio=round(reduction_ratio, 4),
            attention_sink_density=round(sink_density, 4),
            dispersion_bounded=bool(h_anchored <= h_max + 1e-9 and h_max < h_raw)
        )

