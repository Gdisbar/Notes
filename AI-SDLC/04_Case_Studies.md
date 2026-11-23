# PART 4 — CASE STUDIES
*End-to-end analysis under real constraints → **a cohesive AI-augmented delivery & governance strategy tied to business goals***
REF format `[A:page]` → see `00_READ_FIRST.md`. Concepts: `01_`; scenarios: `02_`; decisions: `03_`. ⚠ = not in your condensed PDF.

---

# 1. THE AI-SDLC CASE-STUDY BLUEPRINT (write these headings first)

| # | Heading | What to write | Source |
|---|---|---|---|
| 1 | **Business goal & current state** | speed? cost? quality? compliance? where is the bottleneck today (plan/build/review/test/deploy)? | `[A:5, A:55]` |
| 2 | **Risk class & constraints** | regulatory tier (prohibited/high/medium/low), data sensitivity, brownfield vs greenfield, team maturity | `[A:119-123]` |
| 3 | **Target operating model** | **Spec → Harness → Loop**; AI-augmented vs AI-native; autonomy per stage (HITL/HOTL/autonomous) | `[A:48, A:54-56]` |
| 4 | **Phase-wise AI use** | requirements (intent-to-spec, ambiguity scoring), design (ADR, constraints), coding (patterns, guardrails), testing (self-healing, TIA, evals), review (AI reviewer + human), ops (AIOps) | `[A:6-10, A:63-65, A:79-86, A:94-102]` |
| 5 | **Context & harness design** | knowledge priming, `constitution`/steering files, MCP tools, guides (feedforward) + sensors (feedback), computational vs inferential | `[A:36-42, A:51, A:80-81]` |
| 6 | **Architecture & design principles** | deep modules/sinks, invariants, cost-of-change focus, ADRs, three-layer governance | `[A:71-78]` |
| 7 | **Verification strategy** | non-determinism handling, spec-grounded tests, mutation testing, evals for AI features, agent tests | `[A:89-102]` |
| 8 | **Governance & gates** | decision classes, reversible vs irreversible, two-approver paths, provenance, audit trail, kill switch | `[A:124-129]` |
| 9 | **Metrics & ROI** | DORA + rework rate, AI code share, code turnover, CAT, cost per query/outcome, J-curve expectation | `[A:103-116]` |
| 10 | **Economics** | token cost + verification cost vs human cost; pricing model | `[A:19-20]` |
| 11 | **Rollout plan** | pilot with control group → granular cost tracking → guardrail validation; enablement/training | `[A:115, ⚠ orig. 235]` |
| 12 | **Risks & trade-offs** | hallucinated dependencies, drift, rubber-stamp fatigue, IP, over-privilege → mitigation each | `[A:21, A:68-69, A:128-129]` |

**Tie-back template:** "Because the business needs **…**, we adopt **autonomy level/approach** …, controlled by **guardrail/gate** and measured by **metric**; the cost is **…**; unsuitable when …."

---

# 2. DOMAIN → TYPICAL AI-AUGMENTED APPROACH (use for unseen cases)

| Domain / context | Dominant concerns | Typical approach |
|---|---|---|
| **Fintech / payments** | irreversible actions, security, compliance | HOTL for tests/boilerplate; **two approvers + security reviewer on regulated paths**; guardrails stop agents deleting data/merging untested code `[A:83, A:126]`; provenance & audit trail; high-risk classification where decisions affect finance `[A:119]` |
| **Healthcare / public sector** | safety, bias, explainability, privacy | **high-risk AI controls**: risk assessment, human oversight, monitoring; NIST trustworthy-AI traits (⚠ orig. 253); ISO 42001 |
| **E-commerce / SaaS product** | speed, UI churn, cost per query | self-healing tests, predictive TIA, canary + continuous verification, **cost-per-query and outcome-based pricing** `[A:19, A:95, ⚠ orig. 243]` |
| **Large legacy/brownfield enterprise** | context loss, tech debt, upgrades | SDD with reference files (schemas/OpenAPI) `[A:69]`; **decomposed AI + static-analysis workflows** (library replacement) `[A:87-88]`; ADRs for decisions |
| **Platform/DevEx team** | scale, consistency | org-level harness (context lake, MCP gateway, skills marketplace) — Uber-style ⚠; Lattice skills `[A:85-86]` |
| **AI/LLM-powered product** | non-determinism, drift, hallucination | evals, LLM-as-judge, shadow testing, 3-tier agent tests, guardrails (input/context/output) `[A:84, A:99-102]` |
| **Start-up / prototype** | speed | prompt-first or spec-first; lighter gates; still guardrail irreversible actions |

---

# 3. WORKED CASES FROM THE COURSE

> Format: **Problem → Decisions → Controls/Metrics → Trade-offs → Business tie.**

## CASE 1 — Automating library replacement (SEI) `[A:87-88]` (⚠ orig. 183-187)
- **Problem:** DoD programs' open-source libraries are often **dozens of versions behind** (security and operational risk); manual upgrades are error-prone, labour-intensive and too slow.
- **Decisions:** **decompose the upgrade** instead of delegating it wholesale to an LLM.
  - **Static analysis (RevAPI)** formalises/detects **API breaking changes** between versions — *more complete and consistent than LLMs*.
  - **LLM** synthesises **structured transformation strategies** where no automated recipe exists.
  - **Automated tooling** applies the transformation across dependent projects (**"write once, roll out everywhere"**).
  - **Build system + unit tests** = **feedback sensor**; failures loop back.
- **Roles:** library providers (formalise breaking changes), library sustainers (LLM-generated strategies), application developers (apply, build, test).
- **Harness mapping `[A:88]`:** RevAPI = **computational sensor**; LLM = **inferential** generator; build/tests = feedback → **Spec → Harness → Loop** with **feedforward** (computed deltas) and **feedback** (test results) = "keep quality left".
- **Trade-offs:** needs good tests (else the loop converges on wrong code); upfront tooling; reduces **prompt creep and hallucination**.
- **Business tie:** reduces security exposure and tech-debt backlog at scale.

## CASE 2 — Uber's "software factory" (end-to-end AI SDLC) (⚠ orig. 212)
- **Lifecycle:** **ideation & alignment (inner loop)** → **autonomous execution & validation** → **outer loop & maintenance**.
- **Six foundational infrastructure pillars:** **model gateway**, **MCP gateway & tool optimisation**, **agentified DevPods**, **managed skills marketplace**, **AI context graph**, **AI assistant**.
- **Why it works:** shared **platform-level harness** (team/org/SDLC layer `[A:37]`) rather than per-developer setups; central governance and measurement.
- **Lessons for answers:** platform team owns the harness; skills are shared/version-controlled `[A:85]`; context is infrastructure `[A:40]`.

## CASE 3 — Adopting the AI-native SDLC playbook (Anthropic) `[A:54-56, A:60]`
- **Problem:** build now runs at agent speed; **plan, review/test, deploy** remain at human speed; weekly governance committees are bottlenecks.
- **Design:** replace **the line** with **the loop** `[A:56]`; six non-linear stages **Plan, Design, Build, Test, Deploy, Maintain**, each a set of **plays** (what changes, getting started, steps, governance, measurement) `[A:60]`; each stage **writes an artefact to version control and the next stage reads it** (.md early, code later) ⚠.
- **Controls:** commit chain = **audit trail** (who asked, what the agent produced, who approved); approval gates by decision class `[A:126-127]`.
- **Metrics:** DORA + rework rate; PR cycle time (verification tax) `[A:107]`.
- **Trade-offs:** requires strong automated verification before reclaiming cycle time; governance must become risk-tiered.

## CASE 4 — Greenfield feature with Spec-Driven Development (Spec Kit / Kiro) `[A:57-62, A:65-69]`
- **Flow (Spec Kit ⚠ orig. 144):** `/constitution` (principles: e.g., "prefer single relational tables over new microservices") → `/specify` → `/clarify` → `/plan` (**PR gate**) → `/checklist` → `/tasks` → implement. **Kiro:** requirements.md → design.md → tasks.md; tasks in **waves** by dependency `[A:59]`.
- **Risks & controls `[A:68-69]`:** API hallucination (reference contracts), shallow NFR coverage (explicit NFRs in spec), drift (**CI check**), cascading errors (senior review before `/tasks`), bloat (constitution rule).
- **Spec level:** **spec-anchored** for a long-lived product `[A:67]`.
- **Measure:** first-pass acceptance, rework, spec-to-code drift incidents.
- **Trade-offs:** upfront spec effort and markdown review fatigue vs less ambiguity and rework `[A:62]`.

## CASE 5 — Enterprise coding-assistant rollout: governance + ROI + J-curve `[A:115-116, A:111, A:124]`
- **Situation:** 200 engineers; leadership wants AI assistants.
- **Governance:** approved tools/models, repository boundaries, secure-coding policy, **SAST/SCA**, human review, provenance/audit logging linking AI code to developers `[A:124]` (⚠ orig. 260); guardrails (red-teaming, model allow-list).
- **Rollout:** **baseline & A/B** with control teams → **granular cost tracking** (per outcome) → **guardrail validation** (CI security/quality checks) ⚠.
- **Metrics:** adoption → engagement → productivity → quality → business value `[A:108]`; AI code share; **code turnover 30/90-day**; CAT.
- **Economics (illustration, made-up numbers):** 200 engineers × $120K loaded × 8% dip × 0.25 yr = **$480,000** J-curve cost; licences/compute/training $600,000 → investment **$1.08M**; value $2.0M → **ROI = (2.0 − 1.08)/1.08 ≈ 85%**; with rework: time-saved value $1.5M, rework $0.3M, tool cost $0.6M → **Net ROI = (1.5 − 0.3)/0.6 = 2.0×**.
- **Risks:** J-curve dip, verification tax, shallow adoption (high WAU, low code share), IP/licence exposure.
- **Boundary:** if turnover of AI code is much higher than human, pause expansion and fix context/standards first.

## CASE 6 — Self-healing test automation for a biweekly-release product `[A:94-95]`
- **Problem:** Selenium/Cypress suite fails **20–30%** after each release → **~2 engineer-days/sprint** on locator repair.
- **Solution:** AI-augmented **self-healing**: monitor DOM, detect element changes, repair locators using multiple fallback strategies, confirm before next run — as a **background process between deployments**; add **predictive TIA** to cut run time **70–90%** ⚠; **shift-left** scenario creation at PR phase `[A:94]`.
- **Controls:** human review of healed locators; behaviour-level assertions; periodic full regression; **mutation testing** to avoid circular validation `[A:98]`.
- **Metrics:** suite maintenance hours, flaky-test rate, escaped defects, CI duration.
- **Trade-off:** risk of masking regressions vs large maintenance savings.

## CASE 7 — AI coding agent on a payments application `[A:83-84, A:125-128]`
- **Problem:** the agent must modify a payment app without deleting production data, adding insecure code, or merging untested changes `[A:83]`.
- **Design:**
  - **Guardrails outside the model:** input (prompt-injection, PII redaction), context/RAG, output (grounding, schema) `[A:84]`; **action guardrails** (no destructive commands/prod infra).
  - **Decision classes:** regulated path (auth/payments/PII) → **two approvers incl. security reviewer**; prod deploy → deterministic CI gate + approval; **incident response human-only** `[A:126-127]`.
  - **Autonomy:** HOTL for tests/dev-branch bumps; **HITL for irreversible actions** `[A:128]`; scoped credentials, sandbox, allow-lists, kill switch ⚠.
  - **Verification:** spec-grounded tests, mutation testing, SAST/SCA, provenance.
- **Metrics:** CFR, MTTR, rework rate, review burden, % AI-assisted PRs reviewed by two people.
- **Trade-offs:** slower deployment on regulated paths vs reduced risk; mitigate rubber-stamp fatigue with escalation triggers.

## CASE 8 — AI customer-support agent with outcome-based pricing `[A:19-20, A:99-102]` (⚠ orig. 44)
- **Business model:** per-outcome pricing (e.g., **$2 per resolved ticket**) because seat licences collapse when agents do the work.
- **Economics:** unit cost per resolved ticket = (tokens + retrieval + human verification) → must be below price; design **cost efficiency per query**; model routing to cheaper models for simple tickets `[A:33]`.
- **Architecture:** RAG over the knowledge base `[A:45]`; guardrails (PII redaction, grounding, schema); escalation to human on low confidence/policy topics (**HITL triggers**).
- **Testing:** **evals** (similarity metrics, LLM-as-judge rubric, thresholds); **3-tier agent tests** (component → trajectory → end-to-end); **shadow testing** new prompts/models; **drift monitoring** `[A:99-102]`.
- **Governance:** classify risk tier (medium/high if affecting customers' rights); audit logs; NIST measure/manage.
- **Metrics:** resolution rate, escalation rate, cost per resolution, CSAT, drift alerts.

## CASE 9 — Modernising a brownfield system with AI `[A:62, A:69-70, A:77-78]`
- **Challenge:** SDD tools are hard to evaluate for brownfield; legacy context is lost (LLM context limits).
- **Approach:** build **reference files** (schema dumps, OpenAPI contracts) for planning `[A:69]`; extract/record decisions as **ADRs**; classify dimensions by **cost-of-change** — let AI refactor *business logic/CRUD* (low cost), keep **data topology and invariants** under architect control (high cost) `[A:78]`; staged data migration with human gates (irreversible) `[A:125]`; library upgrades via decomposed static-analysis + LLM workflow (Case 1).
- **Verification:** characterisation tests, mutation testing, canary releases with continuous verification ⚠.
- **Trade-offs:** slower start vs avoiding architectural drift and silent regressions.

## CASE 10 — Organisation-wide AI governance programme `[A:118-123, A:124]` (⚠ orig. 246-248, 257)
- **Starting point:** governance gap — traditional governance assumes humans at every decision (audit trail ends at the human boundary).
- **Programme:** **inventory** AI systems → **risk classification** (prohibited/high/medium/low) → proportional controls → **phased rollout from inventory to monitoring**; components: data governance, model governance, policies & ethics, operational governance, security & privacy `[A:118]`.
- **Frameworks together `[A:123]`:** **NIST AI RMF** (Govern-Map-Measure-Manage) for risk decisions + **ISO/IEC 42001** for the management system + **EU AI Act** compliance for EU scope.
- **SDLC integration:** guardrails in **shared infrastructure**; use-case controls (coding assistant, AI tests, autonomous agents); monitoring/logging; provenance `[A:129]`.
- **Metrics:** % AI systems inventoried/classified, audit findings, incident rate, policy exceptions.

## CASE 11 — A high-risk AI feature in a regulated sector (EU) `[A:119-122]`
- **Classification:** decisions on credit/employment/essential services → **high-risk**: formal risk assessment, enhanced documentation, **human oversight**, ongoing monitoring, **conformity assessment** `[A:119-120]`.
- **Engineering:** quality attributes of AI-enabled systems — **explainability, data centricity, verifiability** `[A:53]`; data lineage/ownership `[A:118]`; evals + drift monitoring; HITL for adverse decisions.
- **NIST traits (⚠ orig. 253):** valid & reliable, safe, secure & resilient, accountable & transparent, explainable, privacy-enhanced, fair with bias managed.
- **Evidence:** documentation, signed ADRs, audit trails, test/eval reports.

## CASE 12 — Fixing a metrics programme after AI adoption `[A:105-111]`
- **Symptoms:** deployment frequency ↑, CFR +20%, code churn doubled — DORA can't explain.
- **Redesign:** keep DORA (+ **rework rate**); add **AI code share, code turnover (30/90-day, AI vs human), CAT, PR cycle time, innovation rate**; instrument adoption/engagement; add qualitative DX surveys; compute **Net ROI** with rework; set J-curve expectations `[A:115]`.
- **Actions:** strengthen sensors and review capacity; fix context/priming; train; revisit use cases.

---

# 4. TEMPLATE FOR AN UNSEEN CASE (fill-in)

```
Business goal:     <speed | cost | quality | compliance> because <value>
Bottleneck now:    <plan | build | review/test | deploy>; brownfield/greenfield
Risk tier:         <prohibited | high | medium | low>; regulation: <EU AI Act / NIST / ISO 42001 / sector>
Operating model:   Spec → Harness → Loop; autonomy: <HITL | HOTL | autonomous> per stage
Phase plan:        Requirements: <AI elicitation/synthesis + ambiguity scoring>
                   Design: <constraints, ADRs, deep modules, invariants>
                   Coding: <5 patterns, guardrails, Lattice-style skills>
                   Testing: <spec-grounded tests, mutation, self-healing, TIA, evals>
                   Review/Deploy: <AI reviewer + human gate; continuous verification; rollback>
Harness:           guides = <constitution/AGENTS.md/spec>; sensors = <linters/tests (computational) + LLM judge (inferential)>
Governance:        decision classes + gates; reversible vs irreversible; provenance; audit; kill switch
Metrics:           DORA + rework; AI code share; code turnover; CAT; cost per outcome; J-curve
Economics:         inference + verification cost vs human cost; pricing model
Rollout:           pilot + control group → cost tracking → guardrail validation → scale
Risks:             hallucinated deps, drift, rubber-stamp, IP, over-privilege → mitigations
```

---

# 5. DIAGRAM CHEAT-SHEET
- **Traditional vs AI SDLC:** human-guided iteration through stages vs **continuous autonomous iteration** `[A:2]`; "the line vs the loop" `[A:56]`.
- **Agent loop:** Perceive → Reason → Act → Evaluate `[A:34]`.
- **Agent = Model + Harness:** guides (feedforward) → agent → sensors (feedback) `[A:36-38]`.
- **Three-layer operating model:** Spec → Harness → Loop; gates of autonomy `[A:48]`.
- **RAG:** ingest (chunk/embed) → retrieve → augment → generate `[A:45]`.
- **MCP:** host ↔ client ↔ server (JSON-RPC) ⚠.
- **Guardrail layers:** input → context/retrieval → output `[A:84]`.
- **Decision gate matrix:** reversible → PR review; irreversible → human approval gate `[A:125-127]`.
- **J-curve:** value dips then exceeds baseline `[A:115]`.
- **Governance stack:** NIST (risk) + ISO 42001 (system) + EU Act (law) `[A:123]`.

---

# 6. LIKELY CASE-STUDY QUESTIONS (inferred) → ANSWER FROM
| Likely question | Section |
|---|---|
| Propose an AI-augmented SDLC for an organisation/project | §1 blueprint, Case 3 |
| Automate a large-scale code migration/upgrade | Case 1, Case 9 |
| Platform for AI engineering across teams | Case 2 |
| SDD for a new feature; risks and controls | Case 4 |
| Roll out coding assistants with ROI & governance | Case 5, Case 12 |
| Reduce test maintenance/CI time | Case 6 |
| Safe agent on a critical (payments) system | Case 7 |
| AI product with outcome pricing and evals | Case 8 |
| Modernise legacy with AI | Case 9 |
| Enterprise AI governance / high-risk feature | Case 10, Case 11 |
