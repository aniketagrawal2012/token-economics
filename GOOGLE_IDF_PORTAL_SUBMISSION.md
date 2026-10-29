# Google Invention Disclosure Form (`go/IDF`) — Official Portal Submission Package

**Invention Title:** Deterministic Semantic Token Compaction, High-Norm Attention-Sink Structural Anchoring, Client-Server Paged KV-Cache Boundary Sealing, and Discrete-Choice Pareto Economic Routing for Large Language Model Serving  
**Short Title / System Codename:** `Agent-Tokenomics` (`PromptVM`)  
**Lead Inventor:** Aniket Agrawal (`aniketagrawal@google.com`), Cloud Engineer, Google Cloud India Private Limited, Bangalore, Karnataka, India  
**Co-Inventors:** None (Sole Inventor — 100% Conception & Reference Implementation)  
**In-House Patent Counsel:** Raju Goyal (`rgoyal@google.com`) / Patent Guru Consultation: Daniel V. Klein (`dvk@google.com`)  
**Google Product Area:** Infrastructure & Cloud / AI Models, Machine Learning & TPU Systems (Google Cloud Vertex AI Model Serving, Cloud TPU v5p & TPU v6e Trillium Pallas Block-Attention, Vertex AI Context Caching API `CachedContent`, Google Agent Development Kit [ADK])  
**Submission Portal URL:** `https://elysian.force.com/inventor/s/` (`go/IDF`)

---

## Screen 1: General Information & Administrative Metadata

* **Invention Title:** Deterministic Semantic Token Compaction, High-Norm Attention-Sink Structural Anchoring, Client-Server Paged KV-Cache Boundary Sealing, and Discrete-Choice Pareto Economic Routing for Large Language Model Serving
* **Short Title / Acronym:** `Agent-Tokenomics` (`PromptVM`)
* **Inventor Details:**
  * **Full Name:** Aniket Agrawal
  * **Email:** `aniketagrawal@google.com`
  * **Department:** Svcs GSD-PSO AI and Spec Billable / Google Cloud Enterprise AI Engineering
  * **Conception Location:** India (Bangalore, Karnataka)
  * **Contribution:** 100% (Sole Inventor: conception of quotation-synchronized AST delta-tokenization over a bounded $2^3=8$ candidate space, polarity-aware relational predicate signature pruning, high-norm low-frequency RoPE subspace structural attention-sink injection with formal entropy bound $\mathcal{H}(\mathbf{a}_i) \le \mathcal{H}_{\max} < \ln i$, client-server modulo-$B$ KV-cache boundary-page sealing via Pallas VMEM causal bitmasking, and closed-form discrete-choice MINLP Pareto economic routing with dynamic test-time reasoning token allocation)
* **In-House Patent Counsel:** Raju Goyal (`rgoyal@google.com`)
* **Primary Product Area:** Infrastructure & Cloud / AI Models & TPU Systems (Vertex AI, Cloud TPU v6e Trillium, Google ADK)

---

## Screen 2: Problem Being Solved (Technical Deficiencies in State-of-the-Art LLM Serving)

Enterprise Large Language Model (LLM) serving and autonomous multi-agent swarms across Google Cloud Vertex AI face an acute physical memory, latency, and reliability ceiling termed the **LLM Token Economics and Attention Wall**:

1. **The Subword Token Fragmentation Tax (Up to 35% Structural Overhead):**
   Standard subword tokenizers (SentencePiece Unigram in Gemini/Gemma or Byte-Pair Encoding [BPE] in `cl100k_base`/`o200k_base`) decompose unaligned whitespace, `camelCase` identifiers, and pretty-printed OpenAPI/JSON schemas (`"key":  "value"`) into 3 to 5 fragmented subword tokens. In production enterprise schemas, up to 35% of ingested tokens represent synthetic tokenization artifacts rather than semantic domain entropy, inflating inference billing and consuming High-Bandwidth Memory (HBM2e) before reasoning begins.

2. **Attention Dispersion and Silent Constraint Hallucination (`H(a_i) -> ln i`):**
   Transformer self-attention layers compute softmax-normalized query-key dot products under Rotary Position Embeddings (RoPE). When prompts contain verbose conversational boilerplate (`"Please make sure to carefully ensure..."`), high-frequency RoPE components decay rapidly with relative distance $(m - n)$, driving softmax attention probabilities toward a uniform distribution with maximal Shannon entropy $\mathcal{H}(\mathbf{a}_i) \approx \ln i$. This dispersion dilutes attention weights on negative constraints and schema contracts, causing critical SQL/ERP parameters to be hallucinated or omitted. Furthermore, neural prompt compression tools (e.g., LLMLingua) randomly drop modal negations (`NOT`, `NEVER`, `<=`) and numeric bounds.

3. **Hardware KV-Cache Boundary-Page Copy-on-Write (CoW) Duplication:**
   Production LLM serving runtimes (vLLM, TensorRT-LLM, SGLang, and Google DeepMind's JAX/Pallas Block-Attention kernels on Cloud TPU v6e Trillium) allocate physical Key-Value (KV) cache memory in discrete virtual memory pages of block size $B \in \{16, 32\}$ tokens in HBM/VMEM. When a shared static prompt prefix terminates at a non-page boundary ($|\mathcal{P}_{\text{prefix}}| \not\equiv 0 \pmod B$), appending branch-specific dynamic suffix tokens into the unsealed trailing prefix block forces **Copy-on-Write (CoW) page duplication and redundant K/V recomputation** across concurrent subagent execution branches.

4. **Suboptimal Monolithic Model Routing Without Test-Time Compute Co-Optimization:**
   Prior model routers (e.g., FrugalGPT, RouteLLM) only select among fixed model endpoints and cannot jointly optimize prompt compaction ratio, context-cache hit state ($H \in \{0,1\}$ at Vertex AI's `32,768`-token `CachedContent` threshold), discrete model tier selection, and continuous test-time reasoning token allocation (`thinking_budget` $T_{\text{think}} \in [0, T_{\max}]$).

---

## Screen 3: Detailed Technical Solution & Unified 5-Stage Semantic-to-Physical Compiler Pipeline

`Agent-Tokenomics` (`PromptVM`) is a unified **Semantic-to-Physical Compiler and Economic Router** where Stages 1–3 deterministic semantic compaction directly enables Stage 4 physical KV-cache page-boundary sealing and Stage 5 SLA-bound Pareto routing:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│               AGENT-TOKENOMICS (PROMPTVM) 5-STAGE COMPILER & ROUTER ARCHITECTURE             │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  [Stage 1: Quotation-Synchronized AST Subword Transpiler (Client/Gateway Wire Layer)]        │
│    • Parses JSON/OpenAPI schemas into AST & evaluates 2^3 = 8 canonical formatting candidates│
│    • Selects subword-minimal serialization S* = argmin |T(S')| preserving schema validation  │
│                               │                                                              │
│                               ▼                                                              │
│  [Stage 2: Polarity-Aware Invariant Density Scorer & Sub-Clause Pruner (Wire Layer)]         │
│    • Strips conversational pleasantries & deduplicates relational predicate signatures Sig(c) │
│    • Unconditionally retains modal/negative constraints (NOT, NEVER, <=, !=) & bindings      │
│                               │                                                              │
│                               ▼                                                              │
│  [Stage 3: High-Norm Structural Attention-Sink Landmark Injector (Wire Layer)]               │
│    • Injects K high-L2-norm structural delimiter tokens (e.g., ### MANDATORY INVARIANTS:)    │
│    • Concentrates low-frequency RoPE inner products (affinity boost gamma > 1)               │
│    • Provably bounds dispersive-head Shannon entropy: H(a_i) <= H_max(i, K, gamma) < ln i    │
│                               │                                                              │
│                               ▼                                                              │
│  [Stage 4: Client-Server Paged KV-Cache Boundary Sealer (Wire + TPU v6e Pallas Kernel)]      │
│    • Client pads static prefix with wire-observable <!-- HW_CACHE_PAD:k --> tokens so        │
│      |P_prefix| ≡ 0 (mod B) (B = 16 or 32) & computes SHA-256 CachedContent handle (>=32k)   │
│    • Server applies Pallas VMEM causal bitmasking & decoupled RoPE logical positions         │
│                               │                                                              │
│                               ▼                                                              │
│  [Stage 5: Discrete-Choice MINLP Pareto Economic Router (Gateway API Control Plane)]         │
│    • Closed-form optimization over model tier M in {Flash, Flash-Thinking, Pro/Argon}        │
│      and dynamic test-time thinking_budget T_think subject to Quality >= Q* & Latency <= L   │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Screen 4: 100% External Competitor Infringement Detectability (`go/IDF` Rule #1 Compliance)

Per Google Patent Committee (`go/IDF` and `go/IDFhelp`) requirements, **`Agent-Tokenomics` is 100% externally detectable on the network wire and public SDK boundary** without requiring access to a competitor's internal TPU/GPU server source code:

1. **Method 1 — Outbound API Wire Payload Inspection (`HTTPS / gRPC / REST`):**
   Inspecting outbound prompt payloads produced by the client SDK or API gateway reveals three unmistakable deterministic wire signatures:
   * **Wire Signature A (Modulo-$B$ Boundary Padding Marker):** The static prefix ends at an exact integer multiple of $B \in \{16, 32\}$ subword tokens via an explicit wire padding comment token sequence `<!-- HW_CACHE_PAD:k -->` (or designated semantic-neutral boundary sentinel tokens) immediately preceding the dynamic suffix boundary.
   * **Wire Signature B (Structural Attention-Sink Landmark Delimiters & Co-Located Predicate Blocks):** Extracted modal/relational constraints are co-located immediately beneath high-norm structural landmark headers (`### MANDATORY EXECUTION INVARIANTS:`, `### CANONICAL SCHEMA CONTRACT:`).
   * **Wire Signature C (Quotation-Synchronized $2^3=8$-Candidate AST Canonicalization):** Embedded JSON/OpenAPI schemas exhibit zero post-delimiter whitespace (`","`, `":"`) and deterministic description truncation calibrated to exact BPE merge boundaries.

2. **Method 2 — API Request/Response Metadata & Dynamic `thinking_budget` Telemetry Probing:**
   Sending a sweep of prompts with varying constraint density $\mathcal{D}_{\text{inv}}$ and SLA latency budgets $L_{\max}$ to a target gateway reveals deterministic closed-form switching of the wire request parameters `model`, `thinking_config.thinking_budget` ($T_{\text{think}}^*$), and `cached_content` (`sha256` prefix handle).

---

## Screen 5: 35 U.S.C. § 101 Subject Matter Eligibility Justification (*Alice/Mayo* & *Enfish*)

* **Alice Step 2A Prong 1 & Prong 2 (Concrete Improvement to Computer Memory & Network Protocol Functioning):**
  - Under ***Enfish, LLC v. Microsoft Corp.***, 822 F.3d 1327 (Fed. Cir. 2016), aligning compiled token prefixes to physical accelerator virtual memory page blocks ($|\mathcal{P}_{\text{prefix}}| \equiv 0 \pmod B$, $B \in \{16, 32\}$) paired with VMEM causal bitmasking and decoupled RoPE logical position counters directly improves the physical functioning of hardware accelerator memory (TPU v6e Trillium VMEM/HBM2e and GPU PagedAttention blocks) by eliminating multi-branch Copy-on-Write (CoW) page duplication.
  - Furthermore, bounding Transformer self-attention softmax dispersion ($\mathcal{H}(\mathbf{a}_i) \le \mathcal{H}_{\max}(i, K, \gamma) < \ln i$) via high-norm low-frequency RoPE subspace sink injection and reducing network wire payload size by up to `53.47%` improves both network bandwidth utilization and inference accuracy (`99.70%` Turn-1 Pass@1 vs. `85.00%` uncompiled baseline).

---

## Screen 6: Empirical Benchmark Results ($N = 1,000$ Enterprise Traces + 400 BIRD-SQL/BFCL Tasks)

Evaluated across 5 random seeds on **$N = 1,000$ enterprise traces** spanning 5 operational cohorts (`SQL`, `OpenAPI Multi-Tool`, `SOX/AML Financial Compliance`, `Legacy Code Modernization`, `Multi-Turn Reasoning`) plus **400 public benchmark tasks (`BIRD-SQL` and `BFCL`)**:

| Evaluation Metric | Uncompiled Monolithic Frontier Baseline | Naive Regex / Heuristic Trimming | **`Agent-Tokenomics` (`PromptVM` Compiler)** | Measured Systems Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Structured Cohort Token Compaction** | `0.00%` | `8.91%` | **`39.98%` (up to `53.47%` on OpenAPI)** | **Saves `81,800` tokens / `1,000` requests (`17.52%` pooled)** |
| **Blended Inference Invoice Cost (`/1k req`)** | `$1.7608` | `$1.5200` | **`$1.0663`** | **`39.44%` Net Cost Reduction (up to `88.79%` in SQL)** |
| **P95 End-to-End Inference Latency** | `2,529.2 ms` | `2,147.5 ms` | **`1,546.6 ms`** | **`38.85%` P95 Latency Reduction** |
| **Strict First-Attempt Turn-1 `Pass@1`** | `85.00%` | `89.20%` | **`99.70%` (`100.00%` post-verify)** | **`+14.70%` Turn-1 Accuracy Gain** |
| **Dispersive-Head Softmax Entropy $\mathcal{H}(\mathbf{a}_i)$** | `6.62 nats` | `6.50 nats` | **`4.32 nats`** | **`34.7%` Entropy Dispersion Reduction ($2.8\times$ retrieval boost)** |
| **KV-Cache Trailing-Block CoW Duplication** | `93.8%` unaligned | `93.8%` unaligned | **`0.0%` (`100%` aligned to $B = 16/32$)** | **Zero Boundary-Page CoW Duplication** |

---

## Screen 7: Surgical 3-Pillar Independent Claim Architecture (Claims 1–10)

> [!IMPORTANT]
> **Design-Around-Proof & Divided-Infringement-Proof Claim Architecture (`go/IDF` Best Practice):**
> To prevent any "single-actor / divided infringement" defense between client-side SDK/gateway compilation, server-side TPU/GPU KV-cache memory management, and API control-plane model routing, the claims below are surgically partitioned into **three separate, self-contained Independent Claims (Claims 1, 2, and 3)**, each independently enforceable on its respective boundary, followed by Dependent Claims 4–10.

### Claim 1 (Independent System Claim — Client/Gateway Wire-Protocol Prompt Compilation & Boundary Alignment):
A computer-implemented prompt compilation gateway system for minimizing subword tokenization fragmentation, bounding Transformer self-attention dispersion, and aligning outbound wire payloads to hardware accelerator memory page boundaries, the system comprising:
one or more processors; and
a memory storing a subword tokenization vocabulary graph and instructions that, when executed by the one or more processors, cause the system to:
1. receive an input inference request comprising an unstructured instruction text and a structured parameter schema;
2. parse the structured parameter schema into an Abstract Syntax Tree (AST) and evaluate a bounded candidate space of canonical schema serializations using quotation-synchronized delta-tokenization against merge transitions in the subword tokenization vocabulary graph to select a subword-minimal serialization $S^* = \arg\min_{S' \equiv S} |T(S')|$ that preserves all schema parameter names, primitive types, enumeration bounds, and required validation flags;
3. segment the unstructured instruction text into clauses, extract polarity-aware relational predicate signatures $\operatorname{Sig}(c_k)$ from each clause, strip non-binding conversational pleasantry spans and duplicate predicate signatures whose inverse-length-scaled invariant density score $\mathcal{D}_{\text{inv}}(\tilde{c}_k)$ falls below a predetermined threshold, and unconditionally retain clauses containing modal negations, numeric bounds, or relational operators;
4. co-locate the retained constraint clauses immediately adjacent to one or more injected high-L2-norm structural attention-sink delimiter sequences at functional region boundaries of the compiled prompt, wherein the structural attention-sink delimiter sequences are configured to project into a low-frequency Rotary Position Embedding (RoPE) subspace with an attention affinity boost $\gamma > 1$ to bound query self-attention Shannon entropy below $\ln i$;
5. partition the compiled prompt into an immutable static prefix $\mathcal{P}_{\text{prefix}}$ and a request-specific dynamic suffix $\mathcal{P}_{\text{suffix}}$, compute a token-length remainder $r = |\mathcal{P}_{\text{prefix}}| \bmod B$ relative to a target hardware Key-Value (KV) cache page block size $B \in \{16, 32\}$, and when $r \ne 0$, append $B - r$ deterministic boundary-padding tokens (`<!-- HW_CACHE_PAD:k -->`) to the static prefix such that $|\mathcal{P}_{\text{prefix}}| \equiv 0 \pmod B$; and
6. transmit the boundary-aligned static prefix and dynamic suffix over a network wire interface in an outbound inference API request frame.

### Claim 2 (Independent Method Claim — Server-Side Hardware Accelerator Paged KV-Cache Boundary Sealing):
A computer-implemented method executed on a hardware artificial intelligence accelerator cluster for eliminating Copy-on-Write (CoW) page duplication across multi-branch agentic Key-Value (KV) caches, the method comprising:
1. receiving, over a network interface, a compiled inference request partitioned into a static prefix $\mathcal{P}_{\text{prefix}}$ padded with boundary-padding tokens to an exact integer multiple of a physical Vector Memory (VMEM) or High-Bandwidth Memory (HBM) page block size $B \in \{16, 32\}$ ($|\mathcal{P}_{\text{prefix}}| = k \cdot B$) and a branch-specific dynamic suffix $\mathcal{P}_{\text{suffix}}$;
2. allocating $k$ sealed physical KV-cache page blocks in accelerator memory for the static prefix without internal trailing-page slack;
3. executing a block-attention kernel over the $k$ sealed physical KV-cache page blocks while applying a causal attention bitmask $m_j = 0$ (`-inf` logit mask) to the boundary-padding token positions within the terminal prefix page block $k$ and decoupling logical Rotary Position Embedding (RoPE) position indices of the dynamic suffix from the physical padded memory slot indices so that the first token of $\mathcal{P}_{\text{suffix}}$ receives logical position index $L_{\text{raw}}$ equal to the unpadded semantic prefix length; and
4. sharing the $k$ sealed physical KV-cache page blocks across a plurality of concurrent subagent execution branches via zero-copy page-table pointer referencing without triggering Copy-on-Write page duplication when branch-specific dynamic suffixes are appended.

### Claim 3 (Independent Method Claim — Discrete-Choice MINLP Pareto Economic Router & Dynamic Test-Time Compute Allocator):
A computer-implemented method for joint model-tier and test-time reasoning token routing in large language model serving, the method comprising:
1. extracting, by an API routing gateway from a compiled prompt payload, a compacted input token count $|\mathcal{P}_{\text{in}}|$, an invariant constraint density score $\mathcal{D}_{\text{inv}}$, and a prefix context-cache hit state $H \in \{0, 1\}$ indexed by a cryptographic SHA-256 digest of a page-aligned static prefix;
2. evaluating, across a discrete candidate set of heterogeneous model serving tiers $\mathcal{M}$ comprising at least a lightweight high-throughput tier, a test-time reasoning tier supporting a configurable reasoning token budget $T_{\text{think}} \in [0, T_{\max}]$, and a monolithic frontier tier, a closed-form minimum required reasoning token budget $T_{\text{think}}^*(M)$ required for each tier $M \in \mathcal{M}$ to satisfy a target quality constraint $\operatorname{Quality}(M, T_{\text{think}}) \ge Q^*$ and a service-level agreement (SLA) latency bound $\operatorname{Latency}(M, |\mathcal{P}_{\text{in}}|, T_{\text{think}}, H) \le L_{\max}$;
3. selecting the Pareto-optimal pair $(M^*, T_{\text{think}}^*)$ that minimizes total financial inference cost $\operatorname{Cost}(M, |\mathcal{P}_{\text{in}}|, |\mathcal{P}_{\text{out}}|, T_{\text{think}}, H)$ over the feasible candidate set; and
4. dispatching the compiled prompt payload over a network wire interface to the selected model serving tier $M^*$ with wire request metadata specifying the dynamically computed reasoning token budget $T_{\text{think}}^*$ and prefix cache handle.

### Dependent Claims (Claims 4–10):
* **Claim 4 (Bounded $2^3=8$ AST Candidate Space & Quotation-Synchronized Delta-Tokenization):** The system of Claim 1, wherein evaluating the bounded candidate space of canonical schema serializations comprises evaluating $2^3 = 8$ combinations of key-ordering, delimiter-spacing, and description-compaction transforms using quotation-synchronized subword delta-tokenization followed by exact full-sequence tokenizer verification.
* **Claim 5 (Generalized High-Norm Structural Attention Sinks & Entropy Derivative Bound):** The system of Claim 1, wherein the injected structural attention-sink delimiter sequences comprise uppercase ASCII headers, repeated punctuation tokens, or designated control tokens whose L2 embedding norm in the low-frequency RoPE subspace ($k \in [1, d/8]$) satisfies $\|\mathbf{x}_s\| \ge \gamma \|\mathbf{x}_o\|$ ($\gamma > 1$), enforcing the upper bound $\mathcal{H}(\mathbf{a}_i) \le \ln Z_i - \frac{K\gamma \ln \gamma}{Z_i} < \ln i$ where $Z_i = K\gamma + (i - K)$ and $\frac{\partial \mathcal{H}_{\max}}{\partial \gamma} = -\frac{K(i-K)\ln\gamma}{Z_i^2} < 0$.
* **Claim 6 (Polarity-Aware Relational Predicate Signature Extraction):** The system of Claim 1, wherein extracting polarity-aware relational predicate signatures $\operatorname{Sig}(c_k)$ comprises normalizing each clause into a tuple $\langle \text{Identifier}, \text{Operator}, \text{Polarity} \in \{+,-\}, \text{BoundValue} \rangle$ so that a positive constraint and a negated constraint on the same identifier are never merged as duplicates.
* **Claim 7 (Automatic 32,768-Token `CachedContent` SHA-256 Handle Registration):** The system of Claim 1, further configured to determine whether $|\mathcal{P}_{\text{prefix}}| \ge 32,768$ tokens, and responsive to $|\mathcal{P}_{\text{prefix}}| \ge 32,768$, compute $\text{SHA256}(\mathcal{P}_{\text{prefix}})$, register a server-side `CachedContent` resource with a sliding-window time-to-live (TTL), and replace the raw static prefix on subsequent wire transmissions with the `CachedContent` handle reference.
* **Claim 8 (Sliding-Window Attention Distance Preservation):** The method of Claim 2, wherein decoupling logical RoPE position indices further comprises computing sliding-window attention mask distances using logical unpadded token positions rather than physical VMEM slot offsets, guaranteeing bit-exact mathematical equivalence to unpadded attention logits.
* **Claim 9 (Logarithmic Test-Time Reasoning Quality Scaling Law):** The method of Claim 3, wherein the closed-form minimum required reasoning token budget $T_{\text{think}}^*(M)$ is computed by inverting the logarithmic test-time scaling relation $Q(M, T_{\text{think}}) = Q_{\text{base}}(M) + \Delta Q_{\max}(M)\frac{\ln(1 + T_{\text{think}})}{\ln(1 + T_{\max})} - \delta_{\text{cohort}}$ and clamping $T_{\text{think}}^*(M)$ to discrete hardware token step increments.
* **Claim 10 (Non-Transitory Computer-Readable Medium):** A non-transitory computer-readable storage medium storing executable instructions that, when executed by one or more processors of a cloud computing system, cause the cloud computing system to perform the operations of Claim 1, Claim 2, or Claim 3.

---

## Screen 8: Source Code, Verification Suite & Artifacts
* **Workspace Directory:** `/Users/aniketagrawal/antigravity_workshop/token_economics`
* **Core Compiler & Router Modules:** `compiler.py`, `bpe_transpiler.py`, `entropy_pruner.py`, `attention_anchor.py`, `paged_cache_aligner.py`, `pareto_router.py`
* **Unit Test & Verification Suite:** `test_token_economics.py` & `verify_tex.py` (`100% PASS`)
* **Camera-Ready IEEEtran Manuscript:** `tokenomics_compiler_paper.tex` & `Agent_Tokenomics_Paper.pdf`
