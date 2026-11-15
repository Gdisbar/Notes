# PART D — LAST-DAY REVISION SHEET

## 1. One-line definitions (recite until fluent)

| Term | One-line definition |
|---|---|
| Waterfall | Sequential phase-gated SDLC; feedback arrives last; change is expensive. |
| Agile | Iterative sprints with continuous stakeholder feedback; scope flexes, time is fixed. |
| DevOps | Extends Agile to production via CI/CD; measured by DORA. |
| Continuous AI-augmentation | AI embedded in every phase; lifecycle becomes a continuous generate-verify loop. |
| Verification tax | Human review effort created by AI-generated output volume. |
| Inversion of Engineering Value | Code generation is free, so value moves to specifying intent and verifying outputs. |
| Technical debt | Accumulated shortcuts from generation outpacing refactoring/test investment. |
| Architectural drift | Actual system structure diverging from intended architecture via many unchecked local changes. |
| Hallucination | Confident, plausible, false AI content (APIs, requirements, logic, security). |
| Foundation model | Broadly trained, large, adaptable base model for many downstream tasks. |
| Tokenization | Subword splitting (BPE); the unit of cost, memory and failure (identifiers split badly). |
| Context window | The model's finite token working memory; context engineering = choosing what enters it. |
| Non-determinism | Same prompt → different outputs due to sampling, prompt sensitivity, provider state, agent stochasticity. |
| Temperature | Sampling randomness parameter; 0–0.1 for code generation. |
| Prompt engineering | Structured instruction: role + decomposed task + minimal context + constraints + output schema. |
| Agentic workflow | Goal → Plan → Act (tools) → Observe → Verify loop, bounded by guardrails. |
| RAG | Retrieval-augmented generation: ground answers in a permissioned, curated corpus at query time. |
| HITL | Human-in-the-loop: recorded human approval at irreversible/expensive-to-reverse steps. |
| Feedforward control | Constraints applied BEFORE generation (contracts, style guides, harness rules). |
| Feedback control | Deterministic checks AFTER generation (sensors: static patterns + dynamic execution). |
| Deterministic checkpoint | A boundary where variability is frozen: validated inputs, exact assertions, bounded attempts, audit logs. |
| Bounded repair loop | Agent may retry only N times (lab: 2) on machine-readable findings, then escalates. |
| Spec-driven development | Spec declares intent, code realizes it; change the spec, regenerate code+tests. |
| Spec-first / anchored / as-source | Spec guides once / spec synchronized with code / humans edit only specs. |
| ADR | Record of one Architectural Decision: title, status, context, decision, consequences. |
| ASR | Architecturally Significant Requirement: measurably affects architecture/quality. |
| STRIDE | Spoofing, Tampering, Repudiation, Info disclosure, DoS, Elevation of privilege. |
| Code smell | Surface symptom indicating deeper design problem (not a bug). |
| Characterization test | Test pinning current (legacy) behaviour before refactoring — the golden master. |
| Circular validation | AI-generated tests validating AI-generated code — correlated errors, no independent oracle. |
| Test oracle | The independent source of expected behaviour that makes a test evidential. |
| Mutation testing | Inject mutants; suites that kill no mutants prove nothing. |
| Smart test selection / TIA | Run the minimal high-value test subset per commit via impact analysis. |
| Canary / blue-green / feature flag | Gradual % rollout / two environments with instant switch / deploy-release decoupling. |
| Self-healing | Detect → diagnose → repair (pre-approved runbooks) → verify → record. |
| Shift-left enforcement | Security/quality policy checked at commit/design time, not release time. |
| Dynamic quality gate | Risk-adaptive blocking checkpoint (bigger change → stricter gates). |
| Autonomous limits | Written, enforced bounds on agent actions: attempts, allow-lists, forbidden ops. |
| Accountability principle | AI is a tool, not a legal person; the deploying humans/org own the outcome. |
| DORA metrics | Deployment Frequency, Lead Time, MTTR, Change Failure Rate (+2025 rework rate). |
| AI Code Share | % of committed code AI-generated/assisted (PR, line, commit level). |
| Code durability / turnover | Does code stick? % rewritten/deleted within 30/90 days, AI vs human. |
| CAT | Complexity-Adjusted Throughput: velocity weighted by work complexity. |
| Innovation rate | % of work that is new features vs maintenance/config. |
| Rework rate | 2025 DORA's fifth metric: unplanned fixes pushed to production. |
| J-curve | Adoption dip then recovery above baseline; forces: Learning, Verification tax, Pipeline adaptation. |
| Cognitive load | Intrinsic (task) + extraneous (tooling/waiting) + germane (learning) mental burden. |
| Lifecycle health | Balanced scorecard of productivity + quality + reliability, never speed alone. |
| Net ROI | (Productive Value of Time Saved − Rework Cost from Code Turnover) / Total AI Tool Cost. |

## 2. Mnemonics

- **D-HAT** — AI velocity risks: Debt, Hallucination, Architectural drift, automated errors at speed.
- **L-V-P** — J-curve trough forces: Learning curve, Verification tax, Pipeline adaptation.
- **5 pillars of Developer AI Impact: "All Coders Value Quality & ROI"** — Adoption, Code share, Velocity (CAT), Quality (turnover), ROI (Net ROI formula).
- **5 levels: "A-E-P-Q-B"** — Adoption, Engagement, Productivity, Quality, Business value ("orgs stop at 1, skip 4").
- **WAY** — Work-first (measure what happened), Agnostic (all tools or the picture lies), Yours (data survives audit).
- **Gate stack: "Schema → Static → Tests → Coverage → Canary → Human"** — six layers.
- **Harness loop: "Feed → Gen → Sense → Fix (≤2) → Audit"**.
- **Spec Kit order: Constitution → Specify → Clarify → Plan → Checklist → Tasks → Analyze → Implement → Converge**.
- **MoSCoW** — Must/Should/Could/Won't. **EARS** — "When [trigger] the system shall [response]".
- **Autonomy rule** — "Autonomous for reversible and verifiable; approval for irreversible, external-facing, data-touching."
- **Course thesis line** — "AI collapses the cost of producing code, not of verifying it" and "the pipeline is deterministic even when the model is not."

## 3. Formulas (write exactly)

- **ROI (%) = (Value − Investment) / Investment × 100**
- **Net ROI = (Productive Value of Time Saved − Rework Cost from Code Turnover) / Total AI Tool Cost**
- **J-curve dip cost = staff size × fully loaded salary × productivity drop × dip duration**
- **Time Saved Value = hours saved per engineer per week × loaded engineering cost**

## 4. Number anchors (drop these in answers for concreteness)

- Lab harness: **2** generation attempts max; **4** feedforward files; exact assertions **201/400**; temperature **0.1**; **3** API retries; generated files **untrusted** until both sensors pass; findings are **{code, message}** pairs.
- DORA: **4** metrics (2018, Accelerate) + **rework rate** (2025); MTTR **survives** AI era; churn **doubled** post-AI.
- Turnover windows: **30-day** (early warning) and **90-day** (deep quality), always split **AI vs human**.
- Developer AI Impact: **5** levels, **5** pillars; DX AI framework: **3** dimensions (utilization, impact, cost); AWS: **5** categories, **3** rollout phases (baseline & A/B, granular cost, guardrail validation); DX Core **4** = speed, effectiveness, quality, business impact (encapsulates DORA+SPACE+DevEx).
- Specification spectrum: **3** points; Spec Kit: **9** commands; STRIDE: **6** threats; MoSCoW: **4** classes.
- ADR fields: **5** (title, status, context, decision, consequences).
- Lab sensor: static checks → dynamic checks; dynamic = `node --test` exit **0**.

## 5. Comparison tables (reproduce any of these from memory)

1. **Pair programming vs agents** — initiative, control point, best-for, risk, verification, failure mode (C1).
2. **Spec-first vs anchored vs as-source** — maintenance, control, verification burden, use case (C2).
3. **RAG vs fine-tuning vs long context** — currency, access control, auditability, cost, failure mode (C3).
4. **Waterfall vs Agile vs AI-augmented** — control points, risk location, artifact, measurement (C4).
5. **Deterministic gates vs AI-judged gates** — reproducibility, auditability, role (decide vs advise) (C6).
6. **DORA metric → AI-era problem → AI-era companion** (E5 table).
7. **Autonomous vs approval-required operations** (Module 6.5 table).

## 6. Dispatch table — "if asked X, open with Y"

| If the question says… | Open with |
|---|---|
| "SDLC evolution / AI acceleration" | Waterfall→Agile→DevOps→AI loop; per-phase changes; end with D-HAT risks. |
| "40% faster but defects/lead-time unchanged" | Local optimization; constraint moved; verification tax; systemic enablers = platform + verification + spec infrastructure. |
| "different outputs each run / non-determinism" | 4 sources (sampling, prompt, provider, agent env); then the 6-layer validation stack; "pipeline deterministic even when model is not". |
| "regenerate from spec / disposable code / no tests" | Inversion of Engineering Value: they kept the cheap half, deleted the valuable half; spec-completeness problem; verification is what makes disposability safe. |
| "client transcript → requirements" | Scrub → RAG-grounded extraction → synthesis (BDD/EARS scoring) → traceability gap report → HITL sign-off. |
| "validate AI architecture" | 6 steps: traceability, risk analysis, STRIDE threat model, ADR capture, prototype, human gate. |
| "AI tests AI / circular validation" | 4 failure reasons; prevention: spec-derived exact assertions, independent oracles, context separation, integration level, human checkpoints. |
| "agent shipped a vulnerability" | Tool-not-person principle; RACI; which gate failed (HITL, shift-left, dynamic gates); liability with the org; audit trail. |
| "monolith → microservices with agents" | Pin behaviour (characterization) → seams → contract-first → guarded generation → sensors+parity → versioned canary → human gate per extraction. |
| "DORA under AI" | 4 metrics + assumptions; DF inflated, LT hides review queue, CFR misses churn, MTTR survives; add rework rate, code share, turnover, CAT. |
| "measure whether AI is beneficial" | 6-step framework: baseline/A-B, CAT, turnover+defects, reliability, human layer, Net ROI; J-curve horizon. |
| "500 developers economics" | Fully-loaded costs incl. review overhead and rework; 3 formulas; pilot→attribute→guardrail rollout. |
| "unlimited repair attempts" | Reward-hacking the sensor, repair drift, cost runaway, masking spec defects; bound = escalation trigger; humans still review intent. |
| "telemetry into AI tools" | 4 channels; sanitize/aggregate/permission/RAG-not-finetuning; curator; version the corpus; measure escape rate. |
| "governance plan for regulated module" | Classify → spec must-nots + permissioned RAG → deterministic deep gates → recorded HITL → canary with auto-rollback → RACI + audit. |

## 7. If you remember only ten lines, remember these

1. AI collapses the cost of producing code, not of verifying it.
2. The pipeline is deterministic even when the model is not — verified artifacts, not raw generations, reach production.
3. Inversion of Engineering Value: value moves from writing code to specifying and verifying it.
4. Feedforward before, sensors after, bounded repair between, audit always, human at irreversible steps.
5. AI tests AI = circular validation; tests are evidence only when the tester is independent of the accused.
6. Accountability: AI is a tool, not a legal person; gates make ownership enforceable.
7. Velocity risks scale together: D-HAT (debt, hallucination, drift, automated errors).
8. DORA was built for human code; pair it with rework rate, code share, turnover, CAT — MTTR survives.
9. J-curve: forecast the dip (Learn, Verify, Pipe); judge in quarters, provision verification first.
10. Lifecycle health = balanced productivity + quality + reliability, never speed alone.
