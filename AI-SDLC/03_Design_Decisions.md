# PART 3 — DESIGN-DECISION QUESTIONS
*Choose between options → **explicit trade-offs · selection criteria · boundaries (when NOT to use)***
REF format `[A:page]` → see `00_READ_FIRST.md`. Concepts in `01_Concepts.md` (**A…L**). ⚠ = not in your condensed PDF.

---

# 0. HOW TO ANSWER ANY "WHICH ONE / WHY" QUESTION

1. **Drivers first** — name 2-3 forces: *risk/reversibility, verification cost, speed, determinism needed, team maturity, regulatory exposure, cost per query.*
2. **Options** — 2-3 realistic candidates.
3. **Criteria table** — compare only on the drivers.
4. **Choose + because + at the cost of.**
5. **Boundary** — "this flips if the action becomes irreversible, if regulation applies, if the codebase is brownfield, if volume explodes…"

**Selection criteria menu for AI-SDLC:** reversibility/blast radius · determinism & verifiability · cost per outcome (tokens + human verification) · context needed · speed vs control · auditability/accountability · regulatory risk class · team skill & maturity · brownfield vs greenfield.

> **Golden rule of the course:** *AI collapses the cost of producing code, not of verifying it* `[A:105]` — so most decisions are about **where to place verification and human judgement**.

---

# 1. SDLC MODEL / DELIVERY APPROACH `[A:5, A:54-56]`

| Criterion | Waterfall | Agile/Scrum | DevOps/CI-CD | AI-native / agentic |
|---|---|---|---|---|
| Change velocity | low | medium (sprints) | high | **very high (hours)** |
| Main bottleneck | hand-offs, upfront docs | story writing, context switching | manual test/deploy | **verification & governance** |
| Human role | author | author + collaborator | operator | **orchestrator/reviewer** |
| Control model | phase gates | sprint reviews | pipeline gates | **spec → harness → loop; decision gates** |
| Best when | stable, regulated, well-known requirements | evolving requirements | frequent release | well-specified work with strong automated verification |
| Risk | late feedback | scope creep | pipeline fragility | hallucination, drift, accountability gap |
- **Boundary:** don't go AI-native without harness (tests, gates); AI-native is not "no process" — it re-places controls `[A:55]`.

---

# 2. HOW MUCH AUTONOMY: HITL vs HOTL vs AUTONOMOUS `[A:48, A:125-128]`

| | Human-in-the-loop | Human-on-the-loop | Autonomous (bounded) |
|---|---|---|---|
| Human does | **approves each action** (blocking) | monitors dashboards; intervenes via circuit breakers | sets constraints; audits after |
| Use for | **high-consequence, irreversible**: prod DB migrations, security perimeter, external financial transactions, incident response | bounded, **reversible**: unit-test generation, dev-branch dependency bumps | routine, well-harnessed tasks |
| Risk | slows delivery; **rubber-stamp fatigue** | missed signals | runaway actions, cost |
| Controls | two approvers on regulated paths; audit trail | alerts, kill switch | sandbox, scoped credentials, budget caps |
- **Decision rule:** classify the action **reversible vs irreversible** `[A:125]`: reversible → standard PR review/HOTL; **irreversible → the agent never holds write capability; human inside the gate.**
- **Boundary:** raise autonomy only as **harness quality** (sensors/tests) improves; add **escalation triggers** (low confidence, unfamiliar dependency, boundary violation) `[A:128]`.

---

# 3. WHO CONTROLS THE FLOW: AGENT HARNESS vs ORCHESTRATOR `[A:75]`

| | Agent harness (LLM owns control flow) | Orchestrator (code owns control flow) |
|---|---|---|
| Strength | flexibility; open-ended exploration | **deterministic reliability**; repeatable stages |
| Weakness | wanders; unpredictable cost/behaviour | rigid; can't handle novel situations |
| Use when | exploratory coding/debugging with human steering | structured, repeatable pipelines (release, migration recipes) |
- **Hybrid:** orchestrator for the stage graph + agent harness inside bounded steps (**graph engineering** `[A:47]`). *Prompt logic buys flexibility at the cost of reliability; code buys reliability at the cost of rigidity.*

---

# 4. COMPUTATIONAL vs INFERENTIAL CONTROLS (SENSORS) `[A:38, A:87-88]`

| | Computational (deterministic) | Inferential (LLM-based) |
|---|---|---|
| Examples | type checkers, linters, structural tests (ArchUnit), static analysis (RevAPI), build/unit tests | LLM-as-judge, AI code review, semantic checks |
| Speed/cost | fast, cheap, CPU | slower, token cost |
| Reliability | **repeatable, explainable** | probabilistic, can be wrong |
| Good at | syntax, structure, known rules, breaking-change detection | semantics, intent, style, ambiguity |
- **Choose:** put **computational sensors first** (cheap, trustworthy), add **inferential** for what rules can't capture; never use the *same model* to generate and judge without independent validation (**circular validation** `[A:98]`).
- **Pairing principle (SEI library replacement):** static analysis detects breaking changes completely and consistently; LLM synthesises transformations; build/tests give feedback `[A:87-88]`.

---

# 5. PROMPT vs CONTEXT/RAG vs FINE-TUNING `[A:42-45, A:49]` (⚠ orig. 107)

| | Prompt engineering | Context engineering / RAG | Fine-tuning |
|---|---|---|---|
| Changes model weights? | no | no | **yes** |
| Effort/cost | lowest, instant, reversible | medium (pipelines, vector DB) | highest (data, training, evaluation) |
| Solves | task phrasing, format, few-shot examples | **missing/outdated knowledge, hallucination**, private data | consistent style/behaviour, specialised domain |
| Limits | can't add knowledge beyond window | context window size, retrieval quality | stale again when knowledge changes; harder rollback |
- **Selection:** start with **prompt**; add **RAG/context** for knowledge and grounding; **fine-tune** only when persistent behaviour change is needed and data exists. **Boundary:** RAG can't fix reasoning errors; more context ≠ better (attention dilution `[A:32]`).

## 5.1 Prompting technique `[A:43]`
**Zero-shot** (simple, cheap), **few-shot** (format/style needed), **chain-of-thought** (multi-step reasoning, costs more tokens). Reasoning models already do hidden CoT `[A:26]` → pay for extra inference compute only when the task needs it.

## 5.2 Standard LLM vs reasoning model `[A:26-27]`
System 1 (fast, cheap, good for completion/boilerplate) vs System 2 (slow, more tokens, better for complex debugging, multi-step design). **Route tasks** accordingly to control cost (tokenomics `[A:33]`).

---

# 6. SPEC-DRIVEN DEVELOPMENT CHOICES

## 6.1 Prompt-first vs spec-driven `[A:57]`
Prompt-first = fast for simple tasks; **breaks as scope/complexity grows** (misses intent). SDD = upfront guardrails, requirements, acceptance criteria, edge cases → less rework, shared source of truth; **cost: upfront effort, markdown review overhead**. **Choose SDD** for features with multiple stakeholders/tests/compliance; prompt-first for throwaway prototypes.

## 6.2 Spec-first vs spec-anchored vs spec-as-source `[A:61, A:67]`
| Level | Spec afterwards | Pros | Cons | Choose when |
|---|---|---|---|---|
| **Spec-first** | may be discarded | cheap; initial clarity | drifts; docs rot | one-off features |
| **Spec-anchored** | maintained with code | traceability, safe evolution | double maintenance | long-lived products |
| **Spec-as-source** | the spec *is* the source; humans never edit code | max leverage, regenerate anywhere | immature tooling; false sense of control; hard in brownfield `[A:62]` | greenfield, highly regenerable components |

## 6.3 Tool choice `[A:58-60]`
| | GitHub Spec Kit | Kiro | Playbook approach |
|---|---|---|---|
| Nature | open-source CLI + templates (works with your coding agent) | AWS agentic IDE/CLI | process guidance (plays per stage) |
| Artefacts | constitution/spec/plan/tasks | requirements/design/tasks + steering files | markdown artefacts in version control |
| Strength | agent-agnostic, customisable | built-in workflow, task waves | stage-wise governance & metrics |
| Choose when | you want portable SDD with your own agent | you adopt the AWS ecosystem | you need an operating model across the org |

## 6.4 Where to put guardrails in SDD `[A:69]`
`constitution.md` (principles: simplicity), `plan.md` as **PR gate**, CI check for spec-to-code drift, reference files for legacy context. Trade-off: gates slow delivery; skip them only for low-risk reversible work.

---

# 7. DESIGN & ARCHITECTURE DECISIONS

## 7.1 Deep vs shallow modules; sinks vs pipes `[A:72]` (⚠ orig. 155)
Prefer **deep modules / sinks** — narrow interface, contained execution, small blast radius → **lower cognitive load for humans and agents (finite context window)**. Shallow/pipe designs force an agent to load and trace the downstream chain. **Cost:** careful interface design upfront; **boundary:** don't over-deepen where change is frequent and behaviours cross boundaries.

## 7.2 Which architectural decisions deserve human focus? (cost-of-change) `[A:77-78]`
| Dimension | Cost now | Decision |
|---|---|---|
| Business logic / CRUD | low | delegate to AI; verify by tests |
| System invariants & consistency | high/critical | **human-authored invariants**; formal tests |
| Data topology & storage model | very high | **architect decision + ADR**; staged migration |
| Non-deterministic (AI) boundaries | high (new) | define fallback, **latency budget**, guardrails |
- **Rule:** stop asking "how long to code?"; ask **blast radius, auditability, latency budget** (⚠ orig. 165).

## 7.3 Where to encode constraints: three-layer governance `[A:76]`
**Constraints** (cursor rules, ADLs) → **Conformance** (fitness functions, dependency rules, gates) → **Knowledge** (repository maps, decisions extracted into ADRs). Choose all three: constraints without conformance are ignored; conformance without knowledge goes stale.

## 7.4 Microservices vs consolidated design under cheap regeneration `[A:13, A:73]`
Disposable microservices become attractive (regeneration cheap) **but** every service adds interfaces and context to maintain; the *architecture is the prompt* — tangled modules defeat any model. Choose boundaries that are **explicit, honest and low-coupling**; avoid architectural bloat (**simplicity rule in constitution** `[A:69]`).

## 7.5 ADR or no ADR `[A:70, A:126]`
Write an ADR for decisions affecting ≥2 teams or reversible only at high cost; agent drafts, **human architecture forum decides & signs**. Skip for local, reversible choices.

---

# 8. CODING-ASSISTANT AND COLLABORATION CHOICES

## 8.1 Pattern selection `[A:79-82]`
| Symptom | Pattern |
|---|---|
| generic/ignores conventions | **Knowledge priming** |
| code before design | **Design-first** |
| long sessions lose decisions | **Context anchoring** |
| inconsistent quality across devs | **Encoded team standards** |
| same mistakes recur | **Feedback flywheel** |
- **Boundary:** overhead not justified for trivial one-off utilities (⚠ orig. 175).

## 8.2 Skills vs prompts `[A:85]`
Prompts are personal/ephemeral; **skills are shared, version-controlled, reviewable (PRs)**. Choose skills for team workflows (Lattice); prompts for one-off tasks.

## 8.3 Guardrail implementation `[A:84]` (⚠ orig. 179)
| Approach | Strength | Weakness |
|---|---|---|
| Deterministic rules (regex/blocklists) | fast, zero-cost, predictable | brittle, misses paraphrases |
| Model-based evaluation (small LLM/classifier) | semantic coverage | cost, can be wrong/bypassed |
| Middleware/hooks (LangChain, n8n) | enforced outside the model; central | engineering effort |
| Human-in-the-loop pause | safest for high-risk | latency, fatigue |
- **Choose a layered mix:** deterministic first, model-based second, HITL for high-risk. Cover **input, context/retrieval and output** layers `[A:84]`.

## 8.3 Measure coding assistant success `[⚠ orig. 169]`
Prefer **first-pass acceptance rate, iteration cycles, post-merge rework, review burden** over lines generated/time to first output.

---

# 9. TESTING DECISIONS

## 9.1 Test selection strategy `[A:94]` (⚠ orig. 202)
| | Full suite | Static rule-based | AI predictive TIA |
|---|---|---|---|
| Scope | 100% | file-path mappings | ML-selected subset |
| Feedback | hours | minutes | minutes |
| Misses | nothing (but slow) | indirect edge cases | rare cross-service failures |
| CI cost | very high | medium | low (70–90% shorter) |
- **Choose:** predictive TIA on PRs + **periodic full regression** as safety net.

## 9.2 Traditional vs self-healing UI tests `[A:95]`
Self-healing cuts the 20–30%-failing-suite maintenance load; risk = **masking regressions** → require review of healed locators and behavioural assertions.

## 9.3 How to validate AI-written code `[A:98]`
Specification-grounded tests, **TDD**, **mutation testing**, independent oracles. Same-model code+tests = circular → reject.

## 9.4 Evaluating LLM output `[A:99]`
| Method | Good for | Weakness |
|---|---|---|
| Similarity metrics (cosine/ROUGE/BLEU) | closeness to a reference | surface similarity ≠ correctness |
| **LLM-as-a-judge** with rubric | quality, safety, relevance | judge bias/cost; calibrate with humans |
| Assertion thresholds | pass/fail gates in CI | needs good thresholds |
| Production monitoring (drift, shadow tests) | real-world degradation | after-the-fact |

## 9.5 Agent testing depth `[A:102]`
Component (cheap) → trajectory (tool use/plan) → end-to-end (goal). Use all three; E2E alone hides where it failed.

---

# 10. METRICS & MEASUREMENT FRAMEWORK CHOICE `[A:103-114]`

| Need | Choose | Why |
|---|---|---|
| Baseline delivery health | **DORA (4 + rework rate)** `[A:104, A:107]` | shared language; but blind to AI |
| Is AI helping or hurting quality? | **Developer AI Impact Framework** (adoption, code share, CAT, **code turnover**, ROI) `[A:108-111]` | measures durability and cost |
| Organisation-wide scorecard | **AWS five categories** `[A:112]` / **DX Core 4** (speed, effectiveness, quality, business impact) `[A:113]` | holistic |
| Measure AI utilisation/impact/cost | **DX AI framework** `[A:114]` | three-dimension view |
| Auditable, tool-agnostic | **WAY** (⚠ orig. 222-223) | work-first, agnostic, yours |
- **Selection rule:** **pair throughput with stability/quality**; avoid vanity metrics (raw LOC, commits, deployments). Include **J-curve** expectations `[A:115]`.
- **Metric substitutions `[A:107]`:** deployment frequency → *TrueThroughput*; lead time → *PR cycle time*; CFR → *code durability / revert rate*; MTTR remains.

---

# 11. ECONOMIC DECISIONS

## 11.1 Build vs buy → token cost vs human labour `[A:20]`
Compare **(inference cost + human verification cost)** vs **full human execution cost**. AI wins when verification is cheap relative to production; loses when outputs need heavy rework or are low-volume/high-risk.

## 11.2 Pricing: per-seat vs outcome-based `[A:19, A:20]` (⚠ orig. 44)
Seats collapse when agents do the work; **outcome/usage pricing** ($2/resolved ticket) aligns revenue with value but needs **metering, cost per query control, and SLA on outcome**. Choose outcome pricing when AI replaces labour; seat pricing for tools that augment users.

## 11.3 Model routing `[A:33]`
Route simple tasks to cheaper models, complex reasoning to stronger ones; track **cost per outcome**. Trade-off: routing logic complexity vs spend.

---

# 12. GOVERNANCE FRAMEWORK & CONTROL CHOICES

## 12.1 NIST AI RMF vs ISO/IEC 42001 vs EU AI Act `[A:120-123]`
| | **NIST AI RMF** | **ISO/IEC 42001** | **EU AI Act** |
|---|---|---|---|
| Nature | voluntary guidance | auditable/certifiable **management system** | **legal obligations** (enforceable in EU scope) |
| Core | Govern, Map, Measure, Manage | policies, objectives, processes, continual improvement | risk classification, conformity assessment, documentation, human oversight |
| Use for | structuring risk decisions | institutionalising the operating model | compliance where EU users/markets |
- **Not either/or — complementary layers `[A:123]`.** Add **ISO 23053/23894** for ML framework/risk terms `[A:122]`.

## 12.2 Risk-tier controls `[A:119]`
Prohibited (don't deploy) · High (assessment, documentation, human oversight, monitoring) · Medium (documented controls, periodic review) · Low (basic visibility, usage policy). **Proportionality** saves effort; **don't ignore low-risk "shadow AI"**.

## 12.3 Approval gate design `[A:126-127]`
| Decision | Gate |
|---|---|
| Low-risk module change | one reviewer |
| Regulated path | two approvers incl. security-aware reviewer |
| Cross-team architecture | architecture forum + signed ADR |
| Prod deploy | deterministic CI gate + deployment approval |
| New dependency | security review + licence + owner sign-off |
| Cost-bearing agent run | budget approval + stop condition |
| Incident response | human-only execution |
- **Principle:** gate strength ∝ **irreversibility and blast radius**.

## 12.4 Where to put governance: policy vs platform `[⚠ orig. 257]`
Put controls in **shared infrastructure** (gateways, pipelines, platform harness) rather than written policy/individual laptops → enforceable and auditable.

---

# 13. INTEGRATION CHOICES

## 13.1 MCP vs custom integrations `[A:51]` (⚠ orig. 111-112)
**MCP** standardises tool/data connections (host–client–server, JSON-RPC) → less glue code, reusable across AI apps; **risk**: untrusted servers/excessive agency → allow-list, least privilege, review. Custom integrations give tighter control but cost more per tool. Choose MCP for breadth, custom for high-sensitivity paths.

## 13.2 Agent type selection `[A:35]`
| Need | Agent |
|---|---|
| fixed rules, immediate response | simple reflex |
| needs world state | model-based |
| reach a target (plan) | goal-based |
| trade off multiple objectives | **utility-based** |
| improve from feedback | learning |
- Coding agents are goal/learning-style loops with harness constraints.

---

# 14. LIKELY DESIGN-DECISION QUESTIONS (inferred) → ANSWER FROM
| Likely question | Section |
|---|---|
| Which SDLC approach for an AI-heavy project | §1 |
| How much autonomy; HITL vs HOTL; reversible vs irreversible | §2, §12.3 |
| Agent harness vs orchestrator; computational vs inferential sensors | §3, §4 |
| Prompt vs RAG vs fine-tuning | §5 |
| Prompt-first vs SDD; spec-first/anchored/as-source; Spec Kit vs Kiro | §6 |
| Where to invest design effort (cost-of-change); deep modules; ADRs | §7 |
| Collaboration patterns; guardrail design | §8 |
| Test selection; self-healing; validating AI code; evals; agent testing | §9 |
| Which metrics/framework | §10 |
| Build vs buy tokens; pricing models; routing | §11 |
| NIST vs ISO vs EU; risk tiers; gate design; MCP | §12-13 |
