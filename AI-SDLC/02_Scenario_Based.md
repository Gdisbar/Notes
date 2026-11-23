# PART 2 — SCENARIO-BASED QUESTIONS
*Handle a concrete situation, failure or edge case → **Diagnose → Mitigate → Recover → Prevent***
REF format `[A:page]` → see `00_READ_FIRST.md`. Concepts in `01_Concepts.md` (**A…L**). ⚠ = not in your condensed PDF.

---

# 0. THE SCENARIO-ANSWER ALGORITHM

| Step | Do this | Write this |
|---|---|---|
| 1. **Classify** | which risk class? code quality/security · context/agent behaviour · testing · metrics · governance/IP · cost | "Primarily a *supply-chain* risk from a hallucinated dependency." |
| 2. **Locate the layer** | prompt · context · harness · loop/graph · spec · pipeline · human process | "Failure is in the *harness* (no sensor), not the model." |
| 3. **Diagnose** | evidence to check (logs/traces, diff, test output, metrics, audit trail) | "Check the lockfile diff and provenance; compare to spec." |
| 4. **Contain now** | stop/rollback/disable/escalate | "Revoke agent write access, revert the commit, block the package." |
| 5. **Recover** | restore trust (re-run verification, fix root cause, regenerate from spec) | "Update the spec, regenerate, re-run tests and mutation testing." |
| 6. **Prevent** | add a control: guide (feedforward) **and** sensor (feedback), gate, guardrail, ADR | "Add a CI gate + allow-list; update `constitution.md`." |
| 7. **Edge cases & trade-offs** | what the fix costs; what if the control itself fails | "Gate adds latency; rubber-stamp risk → escalation triggers." |
| 8. **Measure** | metric that shows recovery | "Turnover rate, CFR, rework rate, first-pass acceptance." |

> **AI-SDLC edge-case checklist:** *hallucination · non-determinism (same prompt, different output) · context loss/attention dilution · reviewer fatigue · circular validation · over-privileged agent · prompt injection · untrusted MCP tool · token/cost blow-up · model/data drift · spec–code drift · IP/licence contamination · accountability gap.*

---

# 1. CODE-QUALITY & SECURITY SCENARIOS

## S1.1 AI suggests a package that doesn't exist (or is malicious) `[A:21, A:124]`
- **Diagnose:** **hallucinated dependency** → supply-chain attack vector (attacker registers the fake name). Check lockfile diff, registry age/downloads, signature, licence.
- **Contain:** block the package, revert the commit, scan for installs.
- **Prevent:** **external dependency adoption = decision class with security review + named library owner** `[A:127]`; allow-listed registries, **SCA/SAST**, lockfile diff + supply-chain summary produced by the agent, **provenance (SLSA)** `[A:129]`.
- **Edge:** agent may "fix" a failing build by adding a new dependency — treat as an *irreversible* action needing a human gate `[A:125]`.

## S1.2 Generated code compiles and passes tests but is insecure `[A:21, A:93]`
- "**Plausible but insecure code**". **Mitigate:** review step 1 (run tests + static analysis first) → verify intent → scrutinise dependencies → spot AI-specific pitfalls `[A:93]`; **semantic AI reviewers flag OWASP Top 10** before human review `[A:9]`; **regulated paths (auth/payments/PII) need two approvers incl. a security-aware reviewer** `[A:126]`.
- **Prevent:** secure-coding skill/standards encoded for the agent `[A:81]`; guardrails on output `[A:84]`.
- **Edge:** if the *same AI wrote tests and code* the tests are **circular** (S3.2).

## S1.3 Architectural drift / "code slop" / code bloat `[A:21, A:73, A:76]`
- **Diagnose:** agent (no institutional memory `[A:71]`) generated code that is syntactically fine but ignores boundaries; maximal complexity.
- **Mitigate:** **conformance layer** — architectural fitness functions, dependency rules, lint gates *before* code propagates `[A:76]`; ArchUnit-style **structural tests as computational sensors** `[A:38]`.
- **Prevent:** **architect-authored constraints** (versioning policy, behavioural constraints, dependency rationale, exception logic) `[A:74]`; **simplicity rules in `constitution.md`** `[A:69]`; **deep modules / sinks** to contain blast radius `[A:72]`.
- **Edge:** constraints go stale → keep ADRs and knowledge layer updated `[A:70, A:76]`.

## S1.4 Spec-to-code drift in spec-driven development `[A:68-69]`
- **Diagnose:** code commits changed behaviour without updating `spec.md`/`plan.md`.
- **Mitigate/Prevent:** **CI check that fails if code touches architecture files without updated specs**; keep spec **anchored** (not merely first) `[A:67]`; regenerate from spec; ADR for any deliberate deviation.
- **Edge:** brownfield code has no spec → include schema dumps/OpenAPI contracts as reference files `[A:69]`; **false sense of control** from reviewing markdown rather than code `[A:62]`.

## S1.5 Plan was wrong but tasks/implementation proceeded (cascading errors) `[A:68-69]`
- Root cause: unchecked transition `/plan → /tasks`. **Fix:** treat **`plan.md` as an explicit PR gate**; mandatory senior review before `/tasks`; **task decomposition & dependency validation** (risk-control activity ⚠ orig. 149). **Measure:** spec-to-code drift incidents, rework rate.

## S1.6 Library upgrade across many repos is slow and error-prone `[A:87-88]`
- **Do not hand the whole task to an LLM.** **Decompose:** **static analysis (RevAPI)** finds breaking changes *completely and consistently*; **LLM** synthesises transformation strategy where no recipe exists; **automated tooling applies it**; **build + unit tests** are the feedback sensor. "Write once, roll out everywhere". **Edge:** if tests are weak the loop will converge on wrong code → require coverage/mutation checks `[A:98]`.

---

# 2. CONTEXT, AGENT & PROMPT SCENARIOS

## S2.1 The assistant keeps producing generic, misaligned code (frustration loop) `[A:79-82]`
- **Diagnose:** project context missing → "internet-average patterns"; effort spent regenerating/editing (⚠ orig. 168).
- **Fix with the five patterns:** **knowledge priming** (stack versions, structure, naming, examples) → **design-first** → **context anchoring** (living feature doc) → **encode team standards** → **feedback flywheel**.
- **Measure (not "lines generated"):** **first-pass acceptance rate, iteration cycles/task, post-merge rework, review burden** (⚠ orig. 169).
- **Edge:** priming docs need upkeep; for trivial one-offs the overhead isn't worth it (⚠ orig. 175).

## S2.2 Long session — the agent forgets constraints or contradicts itself `[A:32, A:41]`
- **Diagnose:** **context-window failure** — silent degradation, **attention dilution ("lost in the middle")**, inconsistent behaviour, cascading failures. LLMs have **no long-term memory** `[A:41]`.
- **Mitigate:** **context engineering** — selection, compression, sequencing, memory/tool integration `[A:40]`; **context anchoring** doc `[A:81]`; shorter sessions with summaries; **RAG** instead of stuffing everything `[A:45]`; **track token consumption** `[A:32]`.
- **Edge:** more context ≠ better; irrelevant text dilutes attention.

## S2.3 Agent takes a destructive action (drops a table / pushes to prod) `[A:83-84, A:125-128]`
- **Contain:** revoke credentials, restore from backup, audit trail from signed/logged agent actions.
- **Root cause:** **over-privilege / excessive agency**; relying on "please behave" prompts.
- **Prevent:** **action guardrails enforced outside the model**; **reversible vs irreversible** matrix — *agents never hold write capability for irreversible actions*; **HITL for high-consequence actions**; scoped credentials, sandbox, allow-lists, transaction limits, dual approval, immutable logs, **kill switch** (⚠ orig. 260); production-incident response = **human-only execution** `[A:127]`.
- **Edge:** approval fatigue → escalation triggers `[A:128]`.

## S2.4 Prompt injection / data leakage via the assistant `[A:84]` (⚠ orig. 257)
- **Input guardrails** (prompt-injection detection, PII redaction/tokenisation, topic filtering) and **output guardrails** (grounding, schema validation) `[A:84]`; secrets never in prompts; approved models/repo boundaries; logging linking AI use to developers `[A:124]`.
- **Edge:** untrusted **MCP servers/tools** can inject instructions — treat as untrusted, allow-list, least privilege.

## S2.5 The agent makes up facts/APIs — grounding `[A:45]`
- Use **RAG** (authoritative docs/API contracts retrieved and added to the prompt) + **grounding checks** `[A:84]`; **constrained decoding** for valid JSON/syntax `[A:30]`; delegate computations to a **code interpreter** `[A:30]`.

## S2.6 The model miscounts characters / mishandles identifiers `[A:29-30]`
- Cause: **subword tokenisation** (models see token IDs, not characters); cryptic identifiers cost many tokens. **Mitigate:** let the model **write and run code** (interpreter) for character-level work; use meaningful identifiers; constrained decoding.

## S2.7 Choose who controls the flow: loops that wander `[A:75]`
- An **agent harness (LLM owns control flow)** can wander in open-ended tasks → add human steering/**HOTL** monitoring; for repeatable stages use an **orchestrator (code owns control flow)**.

---

# 3. TESTING SCENARIOS

## S3.1 Tests are flaky / "passes sometimes" `[A:89-92]`
- **Diagnose:** **apparent non-determinism** — emergent (hidden couplings from AI-chosen libraries), concurrent (agent loops/parallel tools → races), exceptional (hallucinated fault handling). **One run is not enough.**
- **Mitigate:** repeated/statistical test runs, controllability & observability improvements, isolate and quarantine flaky tests (**AI TIA separates flaky from true regressions** ⚠ orig. 202), seed/record non-deterministic inputs, fix races.
- **Prevent:** reduce emergent coupling via **deep modules/explicit contracts** `[A:72]`.

## S3.2 All tests pass, yet bugs reach production — circular validation `[A:98]`
- **Diagnose:** AI generated code **and** tests from the same assumptions.
- **Fix:** **ground tests in the specification, not the implementation**; **TDD**; **mutation testing** (do tests catch injected faults?); **computational verification gates**; independent oracles; independent test approval (⚠ orig. 260).
- **Measure:** mutation score, coverage thresholds, defect escape rate.

## S3.3 20–30% of UI tests fail after every release `[A:95]`
- **Diagnose:** locator breakage (team ships every 2 weeks → ~2 engineer-days/sprint lost). **Mitigate:** **self-healing automation** — AI watches the DOM, finds updated locator via **multiple fallback strategies**, repairs, confirms before next run. **Edge:** heal wrongly (masks a real regression) → require human review of healed locators and verify intent; keep assertions on behaviour not just element IDs.

## S3.4 CI takes too long as PR volume explodes `[A:94]` (⚠ orig. 202)
- Move from **full-suite** to **AI-driven predictive Test Impact Analysis** (ML-selected subset; **70–90% shorter runs**; cross-service dependency mapping); keep periodic full runs as a safety net. **Edge:** missed indirect failures → nightly full regression; track escaped defects.

## S3.5 Where should we focus review & testing effort? `[A:96-97]`
**Defect prediction:** rank files by **churn + complexity + defect history**; JIT-score each PR; add extra review/tests to hotspots. *See §10 for calculations.*

## S3.6 Testing an LLM feature / chatbot `[A:99-101]`
- Exact-match assertions fail → **evals**: similarity metrics (cosine, ROUGE, BLEU), **LLM-as-judge** with rubrics, **assertion thresholds**; **continuous monitoring** — **data/model drift**, **shadow testing / A/B trajectory analysis** on real traffic. **Edge:** judge-model bias → calibrate with human-labelled samples.

## S3.7 Testing an autonomous agent `[A:100, A:102]`
Three tiers: **component** (tools, prompts) → **trajectory** (the reasoning/ReAct chain, correct tool choice, valid JSON parameters, graceful API failure handling) → **end-to-end** (goal achieved?). Use **simulated environments** and keep **human-in-the-loop** practices.

---

# 4. METRICS SCENARIOS

## S4.1 Change failure rate rose 20% after adopting AI `[A:105]`
- **Diagnose:** DORA can't attribute cause. Segment by **AI code share** (AI-assisted PR %), compare **AI vs human code turnover** and **PR revert rate**; check verification tax (review queues), **rework rate** (fifth DORA metric) `[A:106-107]`.
- **Act:** strengthen gates (sensors), reduce batch size, improve context/priming, targeted training. **Measure:** CFR, rework rate, code turnover 30/90-day.

## S4.2 Deployment frequency doubled — is that success? `[A:105, A:107]`
Not necessarily: **AI-generated boilerplate/micro-commits inflate it**. Use **complexity-adjusted throughput (CAT)**, **PR cycle time**, **innovation rate**; check **code durability** (does it stick?).

## S4.3 Code churn has doubled but no incidents `[A:105-106, A:110]`
DORA misses it: rewritten code didn't fail in production. Track **code turnover rate (30/90-day, AI vs human)**; investigate quality of AI contributions; add review/standards; calculate rework cost into ROI (`[A:111]`).

## S4.4 Leadership asks: "Is the AI investment paying off?" `[A:108, A:111, A:116]`
Use the **5-level hierarchy** (adoption → engagement → productivity → quality → business value) and **Net ROI = (value of time saved − rework cost) ÷ total AI tool cost**; include **J-curve cost** (training/learning dip). Present baseline vs control teams (AWS rollout ⚠ orig. 235). **Edge:** **70% weekly users but 10% AI code share** = shallow adoption ⚠.

## S4.5 Productivity dipped after rollout `[A:115]`
**J-curve**: expected temporary dip — learning curve, **verification tax**, pipeline adaptation. Mitigate with enablement/training, adapt review/CI capacity, set expectations, and measure recovery over the dip duration. **Edge:** if it doesn't recover, re-examine use cases, quality (turnover) and costs.

## S4.6 Token spend spikes `[A:19, A:33]`
**Tokenomics**: hidden costs, inefficient model routing, waste. Actions: route simple tasks to cheaper models, cap context size/output, cache, compress context, per-feature **cost per query**, budget alerts, **cost pocket telemetry** (agent runs declare model tier/expected cost/stop condition `[A:127]`). **Edge:** cheaper model → more rework — evaluate cost **per outcome**.

---

# 5. GOVERNANCE & COMPLIANCE SCENARIOS

## S5.1 Production bug in AI-written code — who is accountable? `[A:14, A:129]`
- Accountability is "foggier" when AI authors artefacts. **Answer:** a **named human owner** signs each decision class (reviewer for PR, library owner for dependencies, architecture owner for ADR); **audit trail** of signed commits/DCO, agent metadata, who asked/produced/approved (playbook commit chain `[A:60]`); no one hides behind "the AI did it".

## S5.2 Legal asks about IP/licence risk of AI-generated code `[A:129]`
Challenges: **opacity of training data, code overlap, licence conflicts.** Controls: **provenance tracking** (signed commits, DCO, SLSA, IDE/agent metadata), licence scanning (SCA), approved tools/models and repository boundaries `[A:124]` (⚠ orig. 260), human review.

## S5.3 Reviewers rubber-stamp AI pull requests `[A:128]`
AI generates PRs faster than humans can inspect. **Fix:** mandatory **escalation triggers** (low confidence, unfamiliar legacy dependency, architecture-boundary violation); route routine reversible work to **HOTL**; AI pre-review for triage; **cryptographic audit trails**; limit PR size; track review burden metrics.

## S5.4 Classify a new AI use case `[A:119-121]`
Decide: **prohibited / high / medium / low** risk; high-risk (safety, finance, employment, essential services) → formal assessment, documentation, human oversight, monitoring; align with **EU AI Act** (if EU scope), **NIST AI RMF** (Govern-Map-Measure-Manage), **ISO/IEC 42001** (management system). Prevent **shadow AI** by basic visibility even for low-risk tools.

## S5.5 An agent must change production infrastructure `[A:125-127]`
**Irreversible** → agent drafts and prepares; **human approves**; deterministic CI gate; canary monitoring; rollback within window; record artefact hash, lockfile, deployment manifest. **Incident response:** agent surfaces diagnostics and *drafts* remediation; **on-call human decides and executes**.

## S5.6 Governance drift — policies exist but nobody follows them (⚠ orig. 257-258)
Move controls into **shared infrastructure** (pipelines, platform harness, gateways), not individual laptops; add monitoring & logging of AI use `[A:124]`; map to NIST SSDF, AI RMF, OWASP LLM Top 10.

---

# 6. DEPLOYMENT / OPERATIONS SCENARIOS

## S6.1 Fast build, slow everything else `[A:55]`
Bottleneck moved to **plan, review/test, deploy**; weekly governance committees can't keep pace. **Fix:** automate hand-offs (AI-native SDLC `[A:56]`), AI-assisted review, **predictive CI/CD**, **continuous verification** with auto-rollback, risk-tiered approval (HITL only where irreversible) `[A:117]` (⚠ orig. 243).

## S6.2 A rollout degrades latency `[A:117]` (⚠ orig. 243)
**AI-assisted continuous verification**: compare live telemetry (logs/metrics/traces) with historical baselines during rollout and trigger **automated rollback**. **Edge:** false positives → tune thresholds; ensure rollback is reversible (DB migrations are irreversible → HITL).

## S6.3 Production incident at 2 a.m. `[A:10, A:127]`
**AIOps:** agent analyses logs, correlates with recent PRs, drafts fix/rollback and root-cause summary; **on-call engineer decides and executes** (agent in advisory mode). Feed the incident back to planning/backlog and SLO/error-budget review (⚠ orig. 267).

## S6.4 Model/data drift in an AI feature `[A:101]`
Monitor inputs/predictions vs training distribution; shadow-test new prompt/model versions; retrain or re-prompt; keep eval suite as a regression gate.

---

# 7. REQUIREMENTS & DESIGN SCENARIOS

## S7.1 Requirements ambiguous; AI turns them into wrong code `[A:64-65]`
Use **AI ambiguity/quality scoring** (INCOSE/EARS) in synthesis `[A:64]`; run `/clarify` before `/plan` (⚠ orig. 144); define **what the feature must not do** as guardrails `[A:65]`; human BA/PO review before sign-off `[⚠ orig. 140]`.

## S7.2 Scope creep during AI-accelerated sprints `[A:64]` (⚠ orig. 140)
AI **traceability + drift/scope-creep monitoring**; backlog prioritisation (MoSCoW); spec changes through version control; humans decide trade-offs.

## S7.3 "Code is cheap now — should we still care about architecture decisions?" `[A:77-78]`
Yes, but re-prioritise by **cost-of-change**: business logic/CRUD cheap; **invariants/state consistency high; data topology very high; non-deterministic AI boundaries new high cost**. Capture **invariants** and decide with **blast radius, auditability, latency budgets**; record ADRs.

---

# 8. NON-OBVIOUS EDGE CASES (examiners' favourites)
- **AI-generated tests for AI-generated code** → circular; fix with spec-grounded/mutation tests `[A:98]`.
- **Cheaper model but more rework** → compare cost per accepted outcome, not per token `[A:19, A:111]`.
- **Spec is wrong** → AI faithfully generates wrong code → plan/spec review gates and human validation `[A:69]`.
- **Agent loop never converges** → stop conditions, token/budget caps, escalate to human `[A:127]`.
- **Guardrail fails open** → defence in depth: input + context + output guardrails + action limits `[A:84]`.
- **High AI usage, low AI code share** → shallow adoption ⚠.
- **Healing test masks a real bug** → human review of healed tests `[A:95]`.
- **Fine-tuning vs prompt change** — prompt/context changes don't change weights (cheap, reversible); fine-tuning changes weights (costly, harder to roll back) ⚠ orig. 107.

---

# 9. RECOVERY PLAYBOOK (generic)
1. **Stop the bleeding** (revoke/disable/rollback). 2. **Preserve evidence** (audit trail, agent logs, diffs). 3. **Root-cause by layer** (spec/context/harness/loop/human). 4. **Fix and regenerate from the spec** with added sensors. 5. **Add a gate/guardrail + update constitution/ADR**. 6. **Measure** (turnover, CFR, rework, MTTR). 7. **Feed back** into context, standards, flywheel `[A:82]`.

---

# 10. WORKED CALCULATIONS (formulas as printed in the slides)

## 10.1 Code churn ⚠ (orig. 203)
- **Absolute churn = LOC added + LOC deleted + LOC modified.** *Example:* add 100, delete 30, modify 20 → **150 LOC**.
- **Churn rate (%) = (added + deleted + modified) ÷ total base LOC × 100.** *Example:* module of 1,000 LOC with 150 churned → **15%** (within the **15–20% acceptable** band). 150 on a 500-LOC module → **30%** → hotspot.

## 10.2 Cyclomatic complexity ⚠ (orig. 204)
**V = 1 + number of decision points** (`if`, `else if`, `while`, `for`, `case`, `catch`, `&&`, `||`).
*Example:* function with 1 `for`, 2 `if`, 1 `&&` → decision points = 4 → **V = 5** (<10 low risk). With 12 decision points → **V = 13** (moderate, 11–20). **>20 = high → refactor.**

## 10.3 Combining churn + complexity (defect hotspots) `[A:96-97]`
Rank files by (high churn, high complexity, defect history) → flag the **top-right quadrant** (e.g., churn rate 30%, V = 22) for extra review/tests.

## 10.4 ROI of AI-assisted development `[A:116]`
**ROI (%) = (Value − Investment) ÷ Investment**, where **Investment = direct hard costs (licences + AI infra/compute + training) + J-curve cost**, and **J-curve cost = staff size × fully-loaded salary × productivity drop × dip duration**.
*Illustration (made-up numbers):* 50 engineers, $120K loaded salary, 10% productivity dip for 0.25 year → 50 × 120,000 × 0.10 × 0.25 = **$150,000**. Licences/compute/training = $250,000 ⇒ Investment = **$400,000**. Value (time saved + new revenue + savings) = **$700,000** ⇒ ROI = (700−400)/400 = **75%**.

## 10.5 Net ROI with rework `[A:111]`
**Net ROI = (Productive value of time saved − Rework cost from code turnover) ÷ Total AI tool cost.**
*Illustration:* value of time saved $600K, rework cost $150K, total AI tool cost $200K → (600−150)/200 = **2.25×**. *Time-saved value = hours saved/engineer/week × loaded hourly cost.*

## 10.6 Tokens and cost `[A:28, A:19]`
- **1 token ≈ 4 characters ≈ 0.75 words → 100 tokens ≈ 75 words.** A 3,000-character prompt ≈ 750 tokens; a 750-word answer ≈ 1,000 tokens.
- 🆕 **Cost per query = (input tokens × input price) + (output tokens × output price)** (+ retrieval/vector-search cost); **gross margin impact = price − cost per query**. Context window limit applies to **input + output** `[A:31]`.
*Illustration:* 750 input tokens at $3/M + 1,000 output tokens at $15/M = $0.00225 + $0.015 = **≈ $0.0173 per query** → 1M queries/month ≈ **$17.3K**.

## 10.7 Test-time and CI savings `[⚠ orig. 202, 199]`
- Predictive TIA cuts run time **70–90%**: a 60-minute suite becomes **6–18 minutes**.
- Self-healing: 20–30% of a suite failing per release and **two engineer-days/sprint** lost; with 26 sprints/year ≈ **52 engineer-days/year** recovered per team (🆕 arithmetic).

## 10.8 Complexity-adjusted throughput (CAT) `[A:109]`
**CAT per engineer per week = Σ(complexity weight × delivered unit) across merged PRs.** *Example:* 3 PRs weights 1, 2, 5 → CAT = **8**, vs raw "3 PRs merged" — shows why raw counts mislead.
