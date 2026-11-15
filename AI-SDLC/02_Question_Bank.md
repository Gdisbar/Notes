# 02 — Question Bank (45 questions, 6 categories)

> Every answer is grounded strictly in CS8–CS14 of the merged deck and the two lab sheets. "(Sample Q#)" = adapted from your original 16-question bank. Attempt on paper before reading the model answer.

---

# CATEGORY A — Conceptual / descriptive (12 questions)

## A1. Non-determinism in AI-generated code (Sample Q13, re-grounded)

**Question:** AI coding assistants can produce different implementations for the same prompt across executions. Explain the determinism assumption, the type of non-determinism this represents, and how AI-generated code maps to the categorical sources of non-determinism.

**Concepts tested:** determinism assumption; actual vs apparent non-determinism; four categorical sources; emergent/concurrent/exceptional mapping. **Source:** CS11 deck 2.

**Model answer:**
1. **Determinism assumption:** testers assume identical inputs under identical preconditions always yield identical outputs and postconditions. When it breaks, bugs become **rare, intermittent, difficult to reproduce, isolate and diagnose**, and running a single test case once per requirement is no longer sufficient.
2. **Actual vs apparent:** *Actual* non-determinism = no theoretical way to predetermine behavior (e.g., quantum RNG). *Apparent* = no practical way for testers or test oracles to predict outcomes because of overwhelming complexity or inadequate controllability/observability. **AI-generated code and agentic development fall squarely under apparent non-determinism** — the process is complex and uncontrolled enough that outputs cannot be practically predicted.
3. **Four categorical sources:** Physical (environmental), Emergent (subsystem integration), Concurrent (multi-threading/distribution), Exceptional (fault/failure handling).
4. **AI code spans three (E-C-E):**
   - **Emergent (primary driver):** agents select libraries, write glue code, build integrations; minor prompt-wording or sampling variations introduce different architectural assumptions → unexpected emergent behaviors and hidden couplings at integration time.
   - **Concurrent:** agentic workflows use asynchronous tool calls, parallel worktrees, multi-step reasoning loops → race conditions, timing variations, non-deterministic tool-output ordering.
   - **Exceptional:** generated code works on the happy path but error-handling paths, boundary validations and resource cleanups behave inconsistently under rare or stress conditions.
5. **Implication:** validation cannot rely on one passing run — deterministic sensors (static + dynamic) and repeated/oracle-grounded checks are required (see Lab 2 harness).

**Memory hook:** "AI code is APPARENTLY random — Emergent, Concurrent, Exceptional."

## A2. Spec-Driven Development: what and why

**Question:** Explain Spec-Driven Development, why prompt-first workflows are insufficient as scope grows, and the benefits SDD provides.

**Concepts tested:** SDD definition; prompt-first limitations; four benefits. **Source:** CS8.

**Model answer:**
1. **Problem:** teams ship software that works but misses the original intent; prompt-first workflows work for simple tasks but struggle as scope and complexity increase.
2. **Definition:** SDD is a spec-first approach — teams define common guardrails, requirements, constraints, acceptance criteria and edge cases **up front**, then use AI to generate code, tests and supporting artifacts **from that shared context**. The spec becomes the **connective tissue across the lifecycle**.
3. **Benefits (A-A-F-P):** less **ambiguity** and rework (requirements clarified earlier); better **alignment** across product/engineering/test through a shared source of truth; **faster** implementation (AI generates against structured context); more **predictable** delivery (validation tied back to original intent).
4. **Supporting tools:** GitHub Spec Kit (Specify CLI + templates), Kiro (requirements → design → tasks), the AI-Native SDLC playbook (six stages, version-control handovers).

**Memory hook:** "Guardrails up front, generation from context, spec = connective tissue."

## A3. The requirements lifecycle and its artifacts

**Question:** Describe requirements elicitation, synthesis and management, contrasting traditional and agile artifacts, and name the prioritization technique used in management.

**Concepts tested:** elicitation techniques; BRD/SRS vs user stories/BDD; MoSCoW; backlog. **Source:** CS9 (+ CS8 technique list).

**Model answer:**
1. **Elicitation** = discovering, extracting and uncovering stakeholder needs, system constraints and business goals. It is one of the **least automated stages**. Techniques: interviews, workshops, observation, document analysis, surveys, process mapping, prototyping; agile adds user-story mapping (workflows, MVP, release slices), sprint reviews & demos, spikes & prototyping — validating assumptions with tangible software instead of predicting all needs up front.
2. **Synthesis** = converting raw elicitation data into structured, actionable artifacts. Traditional: **BRD/SRS** — comprehensive, hundreds of pages, the "what" and "how". Agile: **user stories** ("As a [role], I want [action], So that [benefit]"), **acceptance criteria** in BDD form (Given/When/Then), **themes & epics** decomposed into stories.
3. **Management** = tracking, prioritizing and controlling change. Prioritize by customer value, dependencies, risk; classify with **MoSCoW** (Must/Should/Could/Won't); single ordered **Product Backlog** re-prioritized by the Product Owner; sprint duration and capacity stay fixed while **scope flexes** through the backlog.
4. **AI role:** unstructured-data ingestion + synthetic personas (elicitation); automated artifact conversion + ambiguity/quality scoring against INCOSE/EARS (synthesis); bidirectional traceability + drift/scope-creep monitoring (management) — with humans still approving before sign-off.

**Memory hook:** "Discover → Structure → Control; stories say As-a-I-want-So-that; MoSCoW the backlog."

## A4. ADRs: definitions and value

**Question:** Define Architectural Decision, Architecturally Significant Requirement and ADR; explain five ways ADRs help and name the supporting tool.

**Concepts tested:** AD/ASR/ADR definitions; ADR benefits; adrkit. **Source:** CS9.

**Model answer:**
1. **AD** = a justified design choice addressing a functional or non-functional requirement that is architecturally significant.
2. **ASR** = a requirement with a measurable effect on the architecture and quality of a system.
3. **ADR** = captures a single AD and its rationale — reasons, trade-offs, consequences of a chosen decision.
4. **Five benefits (P-C-S-A-M):** preserving decision history; improving team communication; supporting maintenance; avoiding repeated discussions; managing change (reference when revisiting a decision).
5. **Tool:** the adrkit extension drafts ADRs from plans, retrieves existing decisions, and checks plans against them.
6. **In SDD:** ADRs are the knowledge layer's output — decisions extracted from agent reasoning traces become formal records, so agents with zero institutional memory inherit past rationale.

**Memory hook:** "One AD + why = ADR; history, alignment, maintenance, no re-debates, change reference."

## A5. Managing complexity: deep modules

**Question:** Explain why complexity is the primary problem of software design, the two ways design addresses it, and the difference between deep and shallow modules.

**Concepts tested:** complexity elimination/hiding; decomposition; deep vs shallow modules; information hiding. **Source:** CS10.

**Model answer:**
1. The primary problem is **managing complexity**. Good design addresses it by (a) **eliminating it entirely** (removing unnecessary special cases) or (b) **hiding it through modular design**; treat design as a **decomposition problem** ("Design It Twice" — Ousterhout).
2. **Deep module:** simple, **narrow interface** hiding **significant functionality** beneath the surface; maximizes leverage against complexity; imposes **minimal cognitive load on callers**.
3. **Shallow module:** interface **large relative to the small functionality** it hides; little leverage; **increases system-wide interface complexity**.
4. Deep modules revive classic **information hiding** — the same boundary/cohesion/coupling discipline that determines whether AI agents can work safely in the codebase.

**Memory hook:** "Narrow door, big room = deep; wide door, closet = shallow."

## A6. Guardrails in AI-augmented development

**Question:** Define a guardrail, give the four types with examples, and explain the layered architecture plus implementation approaches that enforce them.

**Concepts tested:** guardrail definition; action/quality/security/workflow types; pre-LLM/RAG/post-LLM layers; four implementation approaches. **Source:** CS11.

**Model answer:**
1. **Definition:** a rule, constraint or control mechanism that **prevents an AI system from performing unsafe, incorrect, unauthorized, or undesirable actions** (e.g., an agent on a payment app must not delete production data, introduce insecure code, or merge untested changes).
2. **Four types:** **Action** (block dangerous actions — deleting databases, modifying production infrastructure, arbitrary shell); **Code quality** (linting, type checks, coverage requirements, dependency policies, architectural constraints); **Security** (hard-coded secrets, unsafe dependencies, SQL injection, insecure authorization, sensitive-data exposure); **Workflow** (human approval before merge, deploy, or irreversible operations).
3. **Architecture principle:** guardrails must be **enforced by software outside the model**, not by asking the model to behave. Layers: **Input (pre-LLM)** — prompt-injection detection, PII redaction/tokenization, topic/policy filtering; **Context & retrieval (RAG)**; **Output (post-LLM)** — hallucination/grounding checks, format & schema validation.
4. **Implementation approaches:** deterministic rules (regex/keyword blocklists); model-based evaluation (secondary classifier/LLM for semantic safety); middleware & hooks (LangChain, n8n interceptors); human-in-the-loop (auto-pause on high-risk actions like large financial transactions or DB writes).

**Memory hook:** "A-Q-S-W outside the model; Input → RAG → Output; rules, judge, hooks, human."

## A7. The five collaboration patterns

**Question:** Name and explain the five patterns for effective AI collaboration and the failure mode each addresses, including the cost trade-off.

**Concepts tested:** five patterns; failure modes; shared mental model; trade-offs. **Source:** CS11.

**Model answer:** (K-D-C-E-F)
1. **Knowledge priming** — share curated project context before requesting code (stack versions, directory structure, naming rules, architecture, examples) as **reusable infrastructure**; addresses the AI defaulting to generic internet-average patterns.
2. **Design-first collaboration** — progress capabilities → components → interactions → contracts → implementation; resolve misunderstandings before code; ask AI to compare options and trade-offs; addresses premature implementation that solves the wrong problem.
3. **Context anchoring** — maintain a living feature document (decisions, constraints, open issues, current state) restored across sessions; addresses functional code that violates team expectations and needs heavy rework.
4. **Encoding team standards** — convert tacit senior-engineer judgment into versioned guidance (clean code, architecture, security, testing, error handling, dependencies) applied to **both generation and review**; addresses the AI forgetting or contradicting itself over long/multi-session conversations.
5. **Feedback flywheel** — capture recurring review findings, successful prompts, failure patterns; feed validated learning back into context/standards/workflows; addresses repeating the same mistakes without institutional memory.
6. **Together:** a **shared mental model** (unified vocabulary, architectural vision, quality baseline) shifts effort from vigilance/correction to intent/refinement. **Cost:** upfront time and discipline — unjustified for trivial one-off utilities; pays off for non-trivial features, long sessions, team coordination.

**Memory hook:** "Prime, design, anchor, encode, spin — same expectations in generation and review."

## A8. How AI is used in software testing

**Question:** List the six uses of AI in software testing and explain how AI augmentation changed the role of the QA engineer and regression strategy.

**Concepts tested:** six AI testing uses; traditional vs augmented comparison. **Source:** CS13.

**Model answer:**
1. **Six uses (T-R-S-D-V-L):** automated test generation; risk-based test prioritization; self-healing test scripts; defect prediction; visual AI & anomaly detection; log parsing & root-cause diagnostics.
2. **Changed areas:** maintenance (self-healing repairs locators automatically during execution); shift-left (scenario creation and requirement-ambiguity checks move to PR/design phase before execution); QA role shifts from manual scripting to **quality engineering — exploratory testing, risk evaluation, prompting, orchestrating AI agents**; regression becomes **impact-based selection** (run only what the commit impacts); entry barrier drops as NLP lets non-technical members write plain-language tests.

**Memory hook:** "Generate, prioritize, heal, predict, see, diagnose — QA becomes a quality engineer."

## A9. Code churn and cyclomatic complexity

**Question:** Define code churn (both measures, with the worked example) and cyclomatic complexity (decision points and risk thresholds), and explain why together they predict defects.

**Concepts tested:** absolute/relative churn; 15–20% rule; McCabe complexity; thresholds. **Source:** CS13.

**Model answer:**
1. **Code churn** = rate and volume of code modification in a window. **Absolute churn** = lines added + deleted + edited — adding 100, deleting 30, editing 20 gives **150 LOC**. **Relative churn (%)** normalizes by file/repo size so small and large modules compare fairly. **15–20% churn rate is acceptable.**
2. **Cyclomatic (McCabe) complexity** = number of **linearly independent paths** through the control-flow graph; each decision point adds a path: `if, else if, while, for, case, catch, &&, ||`. **Thresholds: <10 low risk; 11–20 moderate; >20 high risk → refactor.**
3. **Why they predict defects:** high churn marks **instability hotspots**; high complexity **exceeds human cognitive limits**; prediction systems ingest churn, complexity, author overlap and historical defect tags and output automated actions.

**Memory hook:** "150 LOC churn; 15–20% fine; complexity <10 / 11–20 / >20."

## A10. DORA metrics

**Question:** Name the four DORA metrics, what each measures, the assumption behind each, and the two performance families they form.

**Concepts tested:** DORA 4; throughput vs stability; availability. **Source:** CS14.

**Model answer:**
1. **Deployment Frequency** — how often code ships to production (assumption: more deploys = healthier team).
2. **Lead Time for Changes** — time from commit to production (shorter = better pipeline flow).
3. **Mean Time to Recovery** — how quickly incidents are resolved (faster = more resilient systems).
4. **Change Failure Rate** — % of deploys causing failures (lower = higher quality).
5. **DF + LT = throughput; CFR + MTTR = stability — do not trade one for the other.** Software delivery & operational performance adds **availability**. DORA has been the shared language of delivery performance for nearly a decade.

**Memory hook:** "D-L-C-M: two speed metrics, two safety metrics."

## A11. Self-healing tests and predictive defect analysis

**Question:** Explain how self-healing test automation reduces maintenance overhead and how predictive defect analysis identifies hotspots before execution.

**Concepts tested:** self-healing mechanics + numbers; defect-prediction inputs and models. **Source:** CS13.

**Model answer:**
1. **Self-healing:** a Selenium/Cypress team shipping biweekly expects **20–30% of the suite to fail** per release and burns **two engineer-days per sprint** on locator repairs. AI **monitors the DOM**, detects element changes, identifies the updated locator via **multiple fallback strategies**, repairs the test and **confirms the fix before the next run**; it runs as a **background process between deployments**.
2. **Predictive defect analysis:** finds high-risk **hotspots** (files/functions/modules with statistical bug probability) **before tests execute or code reaches production**. **Inputs:** code churn & commit metadata, complexity metrics, defect history. **JIT prediction:** supervised classifiers (Random Forests, XGBoost, GNNs mapped to ASTs) evaluate **every incoming PR**.

**Memory hook:** "20–30% broken, 2 days wasted → DOM-watch heals; churn+complexity+history → RF/XGBoost/GNN per PR."

## A12. Feedforward, feedback sensors, bounded repair, auditability

**Question:** Define feedforward controls and feedback sensors in a coding-agent harness, explain bounded repair and auditability, and state why generated artifacts remain untrusted until sensors pass.

**Concepts tested:** harness vocabulary; static vs dynamic sensors; bounded repair; audit logs. **Source:** Lab 2 (+ CS11 SEI mapping).

**Model answer:**
1. **Feedforward** = controls applied **before generation** to constrain what the agent produces: contract (`openapi.yaml`, single source of truth), style guide, harness rules, generation task. They shape behavior before any code exists.
2. **Feedback sensors** = deterministic checks **after generation**, outside the LLM: **static** (source-pattern checks — required files, exact assertions present, server exported, JSON header) and **dynamic** (actually execute `node --test`; pass = exit 0). Dynamic runs only after static is clean.
3. **Bounded repair:** only the machine-readable findings are fed back; the agent must preserve valid behavior; the loop **stops after a maximum number of attempts (two in the lab)** — preventing endless loops, cost blowup and sensor-overfitting; beyond the budget, escalate to a human.
4. **Auditability:** every attempt is logged (`sensor-report.json`, `attempt-log.json`) — what was generated, which findings fired, who/what repaired, final status; in the manual part a **human-review note** is recorded.
5. **Untrusted by default:** generated files are treated as untrusted until **both** sensors pass — and human review still happens after that.

**Memory hook:** "Feedforward constrains, sensors judge, findings-only repair, two tries, everything logged, trust only after green."

---

# CATEGORY B — Scenario-based application (10 questions)

## B1. AI-assisted requirements engineering from an unstructured client discussion (Sample Q3)

**Scenario:** A client gives a lengthy, unstructured business discussion mixing functional requirements, non-functional requirements and ambiguous statements. Design an AI-assisted requirements workflow: extraction, synthesis, risk control, HITL approval.

**Concepts tested:** elicitation; synthesis; hallucination risk; HITL; guardrails. **Source:** CS8 + CS9.

**Model answer:**
1. **Extraction (AI elicitation):** ingest the transcript as unstructured data (the same channel AI uses for tickets/logs/app reviews); run **synthetic persona + autonomous interviewing** follow-ups to surface implicit needs; prototype instantly to validate understanding. Output: candidate requirements tagged functional vs non-functional vs ambiguous.
2. **Synthesis:** automated artifact conversion into user stories ("As a… I want… So that…") with **BDD acceptance criteria (Given/When/Then)**; run **real-time ambiguity and quality scoring** against standards (**INCOSE guidelines / EARS syntax**) so each requirement is testable; organize into themes/epics; prioritize with **MoSCoW** into a product backlog.
3. **Risk — omitted or hallucinated requirements:** the model can invent requirements not said (hallucination) or silently drop rare-but-critical cases (shallow edge-case coverage, the same failure seen in SDD). Countermeasures: **traceability** (every requirement links to a transcript segment — bidirectional AI traceability), **ambiguity scoring**, and grounding generation in the source document rather than model memory.
4. **HITL approval:** business analysts and the Product Owner **review and approve every AI-generated artifact before sign-off** — strategic trade-offs, legal accountability and stakeholder relationships remain human decisions. Clarification questions (like `/speckit.clarify`) resolve ambiguities with the client; decisions are folded back into the spec baseline.
5. **Guardrails:** elicitation output defines **bounded guardrails — explicitly what the feature must NOT do** — so downstream agents cannot invent scope; spec lives in version control so later changes regenerate code + tests from the approved baseline.

**Memory hook:** "Ingest → score vs INCOSE/EARS → stories+GWT → human sign-off → must-not-do guardrails."

## B2. Feedforward context strategy for generating a production-grade API (Sample Q14, re-grounded)

**Scenario:** You must have a coding agent generate a production-grade Node.js REST endpoint plus tests. Design the context/feedforward strategy and the output-validation loop that makes the result trustworthy.

**Concepts tested:** knowledge priming; feedforward files; harness rules; schema validation; sensors. **Source:** Lab 2 + CS11 patterns.

**Model answer:**
1. **Prime before generating (pattern 1):** supply curated context — contract, stack version, style rules — as reusable infrastructure, not a one-off preamble. In the lab this is exactly four feedforward files: `openapi.yaml` (**single source of truth** for the endpoint: POST /items, 201 on valid `name`, 400 otherwise), `style-guide.md` (CommonJS, built-ins only, small functions, JSON errors, no secrets), `harness-rules.md` (exactly two artifacts, **exact assertions** `assert.strictEqual(response.status, 201/400)` — broad `>= 400` forbidden; response must be JSON with exactly `appJs`/`testJs`, no Markdown fences; **stop after 2 attempts**), `generation-task.md`.
2. **Constrain the output format:** the agent must return a schema-valid JSON object; **validate the schema before writing any file** (output guardrail — format & schema validation) so nothing malformed reaches the repo.
3. **Verify with deterministic sensors:** **static** sensor checks the artifacts reference the contract (POST /items exercised; exact 201/400 assertions present; server exported; JSON content type) and the **dynamic** sensor executes the test suite (`node --test`, exit 0 = pass). Dynamic runs only after static is clean.
4. **Repair loop:** on failure, return **only the coded findings** (e.g., `MISSING_400_ASSERTION`) plus prior artifacts; require repair to preserve valid behavior; **maximum two attempts**, all logged.
5. **Human checkpoint:** artifacts are untrusted until both sensors pass, and a human still reviews before acceptance — the same division as the SEI loop: computational sensors verify, the LLM infers, humans approve.

**Memory hook:** "4 feedforward files → schema-validated JSON → static then dynamic sensors → findings-only repair ≤2 → human accepts."

## B3. Validating non-deterministic AI output for production (Sample Q2)

**Scenario:** An assistant generates different implementations for the same requirement each run. Design a validation mechanism (prompt constraints, schema validation, unit + integration testing, static analysis, human approval, release quality gates) so behavior is production-safe.

**Concepts tested:** apparent non-determinism; guardrail layers; sensors; gates. **Source:** CS11 + CS13 + Lab 2.

**Model answer:**
1. **Why it happens:** AI code is **apparent non-determinism** — sampling and prompt interpretation vary; emergent (library/glue-code) assumptions differ run to run. You cannot force the model to be deterministic; you make the **pipeline** deterministic around it.
2. **Prompt constraints (feedforward):** contract as single source of truth; explicit style + harness rules; exact required output schema; bounded attempts. Constraints narrow, not eliminate, variability.
3. **Schema validation (output guardrail):** every response must validate (fields, types, no fences) before touching the repository — malformed generations are rejected mechanically.
4. **Unit + integration testing (dynamic sensor):** exact-assertion unit tests against the contract; integration tests cover the seams where **emergent** non-determinism hides (library selection, glue code). Because the tests come from the **spec, not the implementation**, they avoid circular validation.
5. **Static analysis (static sensor):** pattern/security/complexity checks (e.g., required exact assertions present; no secrets; cyclomatic complexity thresholds) run first; dynamic execution only after static is clean.
6. **Human approval (workflow guardrail):** merge/deploy of AI-generated code requires human review — auto-pause on high-risk actions.
7. **Release quality gates:** a change ships only when static + dynamic gates pass; gates embody "generated files are untrusted until both sensors pass", and every attempt is audited.

**Memory hook:** "You can't fix the model; fix the pipeline: constrain → validate → test from spec → static → human → gates."

## B4. A UI team drowning in broken test scripts

**Scenario:** A product ships every two weeks; after each release ~25% of the Selenium suite fails on locators; two engineer-days per sprint go to repairs. Design the AI-augmented fix and state what changes for the QA team.

**Concepts tested:** self-healing automation; impact-based selection; QA role shift. **Source:** CS13.

**Model answer:**
1. **Diagnosis:** this is the textbook self-healing use case — biweekly releases break **20–30%** of suites and consume **two engineer-days per sprint** on locator repair.
2. **Fix — self-healing test automation:** AI **monitors the application DOM**, detects element changes, identifies the updated locator using **multiple fallback strategies**, repairs the affected test, and **confirms the fix before the next run**; run it as a **background process between every deployment** so repair happens outside testing hours.
3. **Add impact-based test selection:** run only tests impacted by each commit (AI-driven TIA) — minutes-not-hours feedback, **70–90% CI run-time reduction**, and statistical isolation of flaky tests so real regressions aren't masked by retries.
4. **QA role shift:** engineers stop scripting repairs and move to **quality engineering** — exploratory testing, risk evaluation, prompting and orchestrating the AI agents; the freed two engineer-days go to new coverage.
5. **Guardrail:** repaired tests are still code — sensor-check repaired scripts (assertions unchanged in meaning) before accepting, keeping humans in the loop for semantic changes.

**Memory hook:** "DOM watch + fallback locators + background repair between deploys; TIA for selection; QA → orchestrator."

## B5. Rolling out just-in-time defect prediction on pull requests

**Scenario:** A team wants to catch defect-prone code before merge. Design a JIT defect-prediction capability: data, model, actions, and guardrails against false confidence.

**Concepts tested:** defect prediction inputs/models; churn/complexity thresholds; automated actions. **Source:** CS13.

**Model answer:**
1. **Goal:** identify **hotspots** — files/functions/modules with statistical bug probability — **before tests run or code hits production**.
2. **Data inputs:** code churn & commit metadata; code complexity metrics; defect history (plus author overlap).
3. **Model:** supervised classification for **JIT prediction — Random Forest, XGBoost, or GNN mapped to ASTs — evaluating every incoming PR**; risk features include churn rate (>**15–20%** is the warning band) and cyclomatic complexity (>**20** = refactor territory).
4. **Automated actions:** flag hotspot PRs for deeper review; trigger targeted test selection for impacted areas; propose refactoring when complexity exceeds threshold; feed recurring findings into the review checklist (feedback flywheel).
5. **Guardrails:** prediction is a prioritization signal, not a merge decision — a human reviewer still owns approval (workflow guardrail); monitor prediction quality over time (are flagged files actually failing?) to avoid learning from stale signals.

**Memory hook:** "Churn + complexity + history → per-PR classifier → flag, target tests, refactor — humans still merge."

## B6. Validating an AI-generated architecture before implementation (Sample Q4)

**Scenario:** An LLM proposes an initial architecture from business requirements. Design the validation you run before any code: ADRs, scalability, security/threat modelling, human review.

**Concepts tested:** risk-control activities; ADRs; authored constraints; conformance gates. **Source:** CS9 + CS10.

**Model answer:**
1. **Run the four risk-control activities of the SDD lifecycle:** **(1) requirements validation** (does the design trace to validated requirements?), **(2) architecture risk analysis** (failure modes, scalability limits), **(3) security and threat modelling** (trust boundaries, abuse cases, PII flows — invariants like "PII must never cross region boundaries"), **(4) task decomposition and dependency validation** (is the task graph real and complete?).
2. **Record decisions as ADRs:** every architecturally significant choice gets an **ADR — one decision + rationale + trade-offs/consequences**; the adrkit-style tool drafts ADRs from plans, retrieves existing decisions, and **checks the plan against prior decisions** so the agent cannot silently contradict the record.
3. **Author the constraints before the sprint:** versioning policy, behavioral constraints, dependency rationale, exception logic — categories the agent cannot infer and developers cannot reconstruct from code. Simplicity rules go in the constitution (e.g., "prefer single relational tables over new microservices") to counter **architectural bloat**.
4. **Human review as a hard gate:** treat the plan document as an **explicit PR gate with mandatory senior review before task generation** — this is the mitigation for cascading error propagation (an unchecked plan error fans out into every task).
5. **Conformance layer after approval:** architectural fitness functions, dependency rules, plan-build approval gates and linting evaluate generated code **before it propagates**; review shifts from catching violations to **confirming the system does what was declared**.

**Memory hook:** "RAST the design, ADR every choice, author the 4 constraints, plan-gate with seniors, then conformance checks."

## B7. Introducing AI into the CI/CD pipeline safely (Sample Q8, re-grounded)

**Scenario:** An enterprise wants AI-driven test selection and release automation in CI/CD. Design the workflow and decide which operations are autonomous vs approval-gated.

**Concepts tested:** TIA; guardrail validation phase; workflow guardrails; pipeline adaptation. **Source:** CS13 + CS14 + CS11.

**Model answer:**
1. **Workflow:** commit → **AI impact analysis** (TIA maps dynamic API dependencies and message contracts, selects the impacted ML-chosen subset — minutes feedback, 70–90% CI time reduction) → **automated security/quality checks** (static analysis, secret detection, dependency policy) → build → **staged release** → metrics to the delivery dashboard.
2. **Autonomous operations:** test selection/prioritization; flaky-test isolation; static analysis and linting; cost/attribution tracking. These are reversible, deterministic-verified and logged.
3. **Approval-gated operations (workflow guardrails):** merging to protected branches; **deploying to production**; any **irreversible operation** (data migration, infrastructure change); disabling a security check. AI proposes, humans approve.
4. **Quality gate before release:** both static and dynamic gates green (untrusted-until-verified principle); defect/coverage/churn baselines checked so velocity gains don't spike technical debt — exactly AWS's **guardrail-validation phase** of rollout.
5. **Pipeline adaptation (J-curve):** retune pipelines for the new throughput — **smaller batch sizes, stronger automated testing, tighter version control** — before trusting autonomous cadence; measure with DORA + rework rate to catch hidden churn.

**Memory hook:** "AI selects & analyzes autonomously; humans approve merge/deploy/irreversible; gates green before ship; retune the pipeline for the dip."

## B8. Building the economics case for AI assistants across 500 developers (Sample Q12)

**Scenario:** An enterprise considers deploying AI assistants to 500 developers. Lay out the cost model, value model, and the calculation that accounts for rework, including the adoption-dip budget.

**Concepts tested:** Total AI tool cost; time saved value; Net ROI; J-curve cost. **Source:** CS14.

**Model answer:**
1. **Investment (more than licenses):** **direct hard costs** — seat licenses, additional AI infrastructure/compute, training & enablement per employee — plus **implementation overhead** (admin, integration). Fully loaded: **Total AI Tool Cost per Engineer per month**.
2. **J-curve cost (the dip is budgetable):** modelled productivity dip = **staff size × fully loaded salary × productivity drop × dip duration** — treat the dip as a forecastable transition cost driven by the learning curve, the verification tax, and pipeline adaptation.
3. **Value:** hours saved per engineer per week × loaded engineering cost (**Time Saved Value**) + new revenues + cost savings.
4. **Correct for rework:** AI code can churn — use **Code Turnover Rate** (30/90-day, AI vs human) to price the rework: **Net ROI = (Productive Value of Time Saved − Rework Cost from Code Turnover) / Total AI Tool Cost**; plain **ROI(%) = (Value − Investment)/Investment**.
5. **Decision hygiene:** run **baseline & A/B testing** against control teams; track cost granularly (workspaces/IAM principals → true cost-per-outcome); require guardrail validation so quality doesn't degrade while the ROI looks good.

**Memory hook:** "Licenses + J-curve cost on one side; time saved + revenue on the other; subtract turnover rework before dividing."

## B9. Designing the measurement dashboard for an AI-assisted team (Sample Q11)

**Scenario:** Developers complete tasks 40% faster with AI assistants, but defect rates and production incidents rose. Design a measurement framework that shows whether AI augmentation is actually beneficial.

**Concepts tested:** DORA limits; hierarchy; five pillars; quality baselines. **Source:** CS14.

**Model answer:**
1. **Why the current story is incomplete:** "40% faster" is a level-3 signal measured with an **inflated raw-velocity metric**; DORA alone cannot attribute the CFR rise (dashboard shows CFR +20% but not whether AI code caused it), and doubled code churn is invisible when it doesn't break production.
2. **Measure across all five hierarchy levels (A-E-P-Q-B), not adoption only:** adoption (DAU/WAU/MAU, tool modes) → engagement (depth/mode mix) → productivity (**complexity-adjusted throughput**, cycle time split AI vs human) → **quality** → business value (ROI, not guesses).
3. **Quality pillar (the missing evidence):** **Code Turnover Rate at 30/90 days for AI vs human code** — durability answers "does it stick?"; plus defect rates, test coverage, churn patterns, new-to-maintenance ratio as baselines; **rework rate** (2025 DORA fifth metric) captures unplanned production fixes.
4. **Reliability signals:** keep **MTTR** (it stays honest), track **change failure rate with attribution context**, pair deployment metrics with **AI code share** (PR %, lines %, commits %) so throughput inflation by boilerplate is visible.
5. **Verdict framework:** AI is genuinely beneficial only when complexity-adjusted velocity rises **and** 30/90-day turnover, defect and rework rates hold steady — speed with durability; otherwise the verification tax is being paid later as defects.

**Memory hook:** "40% faster is level 3 of 5 — add durability (turnover 30/90), rework rate, AI code share before believing it."

## B10. Designing an automated AI code-review workflow (Sample Q16)

**Scenario:** Design an automated code-review workflow for AI-generated PRs that catches code smells, security vulnerabilities and logical defects — while preventing false positives and keeping review independence.

**Concepts tested:** 8-step review; guardrail layers; independence from generation; Lattice review molecule. **Source:** CS11 + CS13.

**Model answer:**
1. **Order the funnel like the 8-step checklist:** automated tests + static analysis **first** (functional gate); then context/intent fit against architecture; code-quality assessment; **dependency scrutiny** (invented/outdated/unnecessary packages = hallucination signal); AI-specific pitfalls; collaborative human review; automate everything repeatable; feed findings back into standards (step 8 = feedback flywheel).
2. **Independent reviewers:** the reviewer must not share the generator's assumptions — a **separate review pass** (Lattice's `review` molecule performs **independent code evaluations**) grounded in the **specification, not the implementation**, preventing circular validation where generated tests bless generated code.
3. **False-positive control (layered signals):** **deterministic rules** (regex/keyword — zero-cost, exact) for definite violations; **model-based evaluation** (secondary classifier/LLM) only for semantic judgments, with rubrics; findings must be **coded and machine-readable** so humans can triage quickly; tune rules via the flywheel instead of flooding reviewers.
4. **Security checks as guardrails:** secret detection, unsafe dependencies, SQL injection, insecure authorization, sensitive-data exposure — enforced **outside the model**; PII redaction on any context the reviewer model sees.
5. **Escalation:** high-risk findings (auth, payments, DB writes) auto-pause for **human-in-the-loop** review; everything logged for audit.

**Memory hook:** "Tests+static first, spec-grounded independent review, rules before judge, coded findings, auto-pause on risky code."

---

*(Categories C–F continue.)*

# CATEGORY C — Compare and justify (7 questions)

## C1. AI pair programming vs autonomous coding agents (Sample Q5)

**Question:** Compare AI pair programming (developers interactively accept/reject suggestions) and autonomous coding agents (independently perform tasks) across workflow, benefits, risks and control mechanisms; recommend when to use each.

**Concepts tested:** interaction modes; harness vs orchestrator; guiding rule; guardrails. **Source:** CS11 (modes) + CS10 (orchestrator vs harness).

**Model answer:**

| Axis | AI pair programming | Autonomous coding agent |
|---|---|---|
| Workflow | Developer drives; suggestions accepted/rejected inline (completion/chat mode) | Agent plans and executes multi-step tasks; human steers or reviews after |
| Control flow | Human owns every decision — highest oversight | LLM owns control flow in the loop (agent-harness style) or code does (orchestrator) |
| Benefits | Immediate feedback; developer learning; context supplied naturally; low blast radius | Frees humans; parallelism; consistent scaffolding; good for well-specified bulk work |
| Risks | Friction loop (generate → not aligned → regenerate) eats the speed gain | Wandering behavior, hidden couplings, non-deterministic outputs, compounding drift |
| Control mechanisms | Human accept/reject IS the guardrail; standards encoded for generation | Feedforward rules, deterministic sensors, bounded repair, workflow approval gates |

**Guiding rule (quote):** *the less human oversight involved, the more control flow must live in deterministic code* — pair programming keeps oversight high (prompt/LLM control flow is acceptable), autonomy demands orchestrator-style deterministic stages plus quality gates.

**Recommendation:** pair programming for exploratory/ambiguous work, learning contexts, and anything touching architecture (human judgment per line); autonomous agents for well-specified, repeatable, high-volume tasks (scaffolding, migrations, boilerplate, batch fixes) **inside a harness** — feedforward contract, static+dynamic sensors, ≤N attempts, human approval on merge. Never grant an autonomous agent unrestricted access to production systems or irreversible operations (workflow guardrails).

**Memory hook:** "Human-in-every-line vs agent-in-a-cage: oversight down → deterministic code up."

## C2. Spec-first vs spec-anchored vs spec-as-source

**Question:** Compare the three positions on the specification spectrum and justify which fits (a) a short prototype, (b) a long-lived enterprise feature, (c) an AI-native product team that wants specs to be the interface.

**Concepts tested:** specification spectrum; human-edit boundaries. **Source:** CS8 + CS9.

**Model answer:**
1. **Spec-first:** spec written before coding to guide initial implementation; **may or may not be maintained** afterwards — its value is initial clarity. Human edits code (spec optional).
2. **Spec-anchored:** spec **maintained alongside code** for the whole lifecycle; any behavior change updates **both** spec and code to stay synchronized. Two synchronized artifacts.
3. **Spec-as-source:** the spec is the **only artifact humans edit**; code is entirely generated and **never manually modified**; behavior change = edit spec → regenerate. Code becomes derivative.
4. **Control gradient:** human authority over code decreases left → right; automation and regeneration increase; so does the payoff for spec quality — and the risk (spec-to-code drift must be mechanically checked in anchored/as-source modes; CI fails commits touching architecture files without spec updates).
5. **Justification:** (a) prototype → **spec-first** (clarity now, no maintenance tax); (b) long-lived enterprise feature → **spec-anchored** (spec keeps evolving with code, synchronized, audit trail); (c) AI-native product team → **spec-as-source** (the spec is the team's interface to generation; "the spec declares intent, the code realizes it") — but only with drift-detection CI and reference files in context, and remember the critique: reviewing markdown can create a false sense of control, so keep conformance checks on the code itself.

**Memory hook:** "First = guide once; Anchored = sync forever; As-source = humans only write spec."

## C3. Deep vs shallow modules and sinks vs pipes (with agent impact)

**Question:** Compare deep and shallow modules, and pipes vs sinks; justify why this distinction becomes MORE important when AI agents modify code.

**Concepts tested:** module depth; sink/pipe; context limits; blast radius. **Source:** CS10.

**Model answer:**
1. **Deep module:** narrow interface hiding significant functionality — maximum leverage, minimal caller cognitive load. **Shallow module:** large interface over little functionality — system-wide interface complexity, little leverage.
2. **Pipe:** execution leaks across boundaries — cascading DB triggers, background queues, downstream side effects; almost always shallow. **Sink:** receives input, performs work, terminates entirely within its boundary; inherently deep. Sinks give **contained blast radius**.
3. **Agent impact:** human cognitive load translates into the **agent's finite context window and attention budget**. Meeting a **pipe**, the agent must load and trace the entire downstream dependency chain → context exhaustion, attention dilution, hallucinations. Meeting a **sink/deep module**, it holds only the immediate contract and local logic, tests against the explicit interface, and deploys with contained blast radius.
4. **Justification:** AI accelerates low-level code, not high-level design — so design quality is now the binding constraint; architectures must "tell the absolute structural truth about themselves" (boundaries, honest interfaces, minimal side effects, high cohesion, low coupling) before agents can work safely at scale.

**Memory hook:** "Pipes drain the context window; sinks keep the blast radius in the box."

## C4. Agent harness vs orchestrator (Sample Q15, re-grounded)

**Question:** Design an agentic workflow for a development task and explain the roles of planning, execution, feedback and verification; compare agent-harness and orchestrator control-flow styles and the risks of granting agents unrestricted access.

**Concepts tested:** agentic loop; control-flow ownership; guiding rule; unrestricted-access risks. **Source:** CS10 + Lab 2.

**Model answer:**
1. **Agentic workflow (harness-style):** plan (read contract/rules) → execute (generate artifacts) → feedback (deterministic sensors return coded findings only) → verify (static + dynamic pass) → bounded repair (≤2 attempts) → human approval; every step logged (audit trail).
2. **Agent harness (LLM owns control flow):** continuous loop where the model decides the next action (read files, run commands, terminate); best for **open-ended exploration with a human in the loop** steering wandering behavior.
3. **Orchestrator (code owns control flow):** code dictates execution across **structured, repeatable stages**; best for **autonomous, unattended workflows** (automated PR reviewers, batch pipelines) because **errors remain isolated and failure is predictable**.
4. **Trade-off:** prompt logic = flexibility at the expense of reliability; code = deterministic reliability at the expense of rigidity. **Guiding rule:** the less human oversight, the more control flow must live in deterministic code.
5. **Risks of unrestricted access:** unbounded repair loops and cost; prompt creep; wandering into irreversible actions (deleting data, deploying untested changes); hallucinated dependencies entering production. Controls: feedforward rules, deterministic sensors, attempt bounds, action/workflow guardrails, HITL pause on high-risk operations.

**Memory hook:** "Harness explores with a human; orchestrator runs alone; freedom down = deterministic code up."

## C5. Traditional vs AI-augmented testing, and the three selection strategies

**Question:** Compare traditional and AI-augmented testing across maintenance, shift-left, QA role, regression strategy and entry barrier; then compare full-suite, static rule-based, and AI-driven test impact analysis.

**Concepts tested:** 5-area comparison; TIA table. **Source:** CS13.

**Model answer:**
1. **Five-area comparison:** maintenance — manual locator repair vs **self-healing during execution**; shift-left — post-deployment testing vs **PR/design-phase ambiguity checks**; QA role — script writer vs **quality engineer (exploratory, risk, prompting, agent orchestration)**; regression — bloated full suites vs **impact-based selection**; entry barrier — Selenium/Cypress scripting vs **natural-language test creation**.
2. **Selection strategies:** full suite = 100% of tests, hours, blind retries of flaky failures, very high CI cost; static rule-based = explicit file-path mappings, fast but **misses indirect edge cases**, hardcoded cross-service links; **AI-driven TIA** = dynamic ML-selected subset, minutes with high accuracy, statistical separation of flakiness from true regressions, **70–90% CI run-time reduction**, AI maps dynamic API dependencies/message contracts.
3. **Justification:** full suites don't scale with AI-speed commits; static mappings break on refactors; TIA alone understands dynamic dependencies — but keep the full suite for periodic safety nets and treat TIA selection as a high-recall filter, with flaky statistics reviewed by humans.

**Memory hook:** "Hours → minutes; paths → learned dependencies; blind retries → statistical flaky isolation."

## C6. Static analysis vs LLMs: the division of labor

**Question:** Why does the SEI framework insist static analysis (RevAPI) identifies breaking changes while LLMs only synthesize transformation recipes? Compare what each does best and justify the pairing.

**Concepts tested:** computational vs inferential; RevAPI; recipes; cybernetic loop. **Source:** CS11.

**Model answer:**
1. **Static analysis (computational sensor):** identifies API breaking changes **more completely and consistently than LLMs** — RevAPI formalizes removals, signature changes, visibility changes **exhaustively and without hallucination**. Deterministic, repeatable, auditable.
2. **LLM (inferential synthesizer):** strongest at **generating code transformation strategies/recipes** for the identified breaking changes — translating formal diffs and release notes into contextual migrations. Probabilistic, semantic, creative.
3. **Automated application (deterministic gate):** transformation engines apply the recipes; **build + tests act as the final computational feedback sensor** — failures are injected back into the LLM context until convergence on a clean build (the cybernetic loop: feedforward → execution → feedback → converge).
4. **Why the pairing solves hallucination & prompt creep:** the LLM never decides *what* changed (facts come from the sensor), and the sensor never has to *write* migrations (semantics come from the LLM); a verified recipe becomes a **reusable deterministic asset** — "write once, roll out everywhere" across dependent projects **without re-running the LLM**.
5. **Roles:** library providers formalize breaking changes (static analysis); sustainers generate strategies with LLMs when recipes don't exist; application developers apply + build + test.

**Memory hook:** "Sensor finds facts, LLM writes plans, build/test judges — verify once, replay everywhere."

## C7. AI-era measurement frameworks: DORA vs Developer AI Impact vs WAY vs DX Core 4

**Question:** Compare the four frameworks and recommend an adoption sequence for an enterprise starting AI measurement from scratch.

**Concepts tested:** DORA limits; pillars; WAY layers; DX Core 4 dimensions. **Source:** CS14.

**Model answer:**
1. **DORA (D-L-C-M):** delivery outcomes; designed before AI assistants; MTTR holds, CFR partial, DF/LT misleading without context; can't attribute cause or see churn — necessary but no longer sufficient.
2. **Developer AI Impact Framework:** five levels (Adoption/Engagement/Productivity/Quality/Business Value) and five pillars (Adoption; AI Code Share; CAT velocity; Quality/durability; Cost & ROI with Net ROI formula) — strategy: **keep MTTR, evolve CFR → Code Turnover, add AI Code Share**; closes the quality and economics gaps DORA leaves open.
3. **WAY (Work-first, Agnostic, Yours):** meta-principles layered as **AI Adoption → Delivery Impact (DORA lives here) → Business Value (reclaimed hours)** — tells you HOW to measure honestly (what happened; across every tool; audit-survivable data).
4. **DX Core 4:** unifies **DORA + SPACE + DevEx** on four dimensions (speed, effectiveness, quality, business impact); adds direct/indirect impact metrics and developer-experience signals (DXI, change confidence).
5. **Recommended sequence:** start **WAY** as policy (agnostic + auditable), run **DORA** as the delivery baseline, layer the **Developer AI Impact pillars** for AI-specific quality/economics, and adopt **DX Core 4** when developer-experience and unified reporting are needed — "AI metrics measure what's happening; Core 4 confirms whether it's working."

**Memory hook:** "WAY = rules, DORA = baseline, Pillars = AI truth, Core 4 = unified view."

---

# CATEGORY D — Diagram / workflow (6 questions)

## D1. How AI changes the SDLC and its risks (Sample Q1)

**Question:** An organisation moves from traditional Agile to AI-augmented engineering. Show how AI changes the phases (as a flow), the benefits, the risks of higher velocity (technical debt, architectural drift), and the mechanisms that maintain quality.

**Concepts tested:** bottleneck shift; velocity risk; quality mechanisms. **Source:** CS8 (+ CS10/CS11 mechanisms).

**Model answer:**
1. **Flow — what changes:**
```
Traditional: Plan → Build → Review/Test → Deploy     (human speed end-to-end)
AI-native:   Plan ──► BUILD(agent speed) ──► Review/Test ──► Deploy
             automated handover between plays; artifact in version control
             passes from stage to stage; commits = audit trail
```
2. **Benefits:** build collapses to agent speed; faster implementation against structured context; predictable delivery when validation ties back to spec; automated handover removes manual phase transitions.
3. **Risks of velocity:** bottleneck moves to **plan, review/test, deploy** (still human speed); line-by-line review can't keep up with agent-written diffs; **controls stop matching reality**; governance costs rise (weekly/monthly committees vs agent-speed iterations); AI produces cheap code but **not cheap verification** → hidden churn (roughly doubled) and technical debt accumulate invisibly; **architectural drift** compounds as agents scaffold with varying assumptions (emergent couplings).
4. **Quality mechanisms:** spec-driven guardrails (requirements/constraints/acceptance criteria up front); deterministic sensors (static + dynamic) around generation; **deep modules/sinks** to contain blast radius and protect the context window; guardrails (action/quality/security/workflow) with human approval gates; conformance layer (fitness functions, dependency rules) before code propagates; measure with DORA **plus** rework rate, code turnover, AI code share so quality shifts are visible.

**Memory hook:** "Build got fast, everything around it didn't — so: specs up front, sensors around agents, humans at the gates, durability on the dashboard."

## D2. The AI-native SDLC playbook workflow

**Question:** Draw the playbook's stage flow with its handover mechanism, and explain how auditability is achieved.

**Concepts tested:** six stages; version-control handover; artifact types; audit trail. **Source:** CS8.

**Model answer:**
```
Plan ──► Design ──► Build ──► Test ──► Deploy ──► Maintain
  │         │        │        │         │          │
  └── each stage ENDS by writing an artifact to version control;
      the next stage BEGINS by reading it (automated handover + triggering)
```
1. **Six non-linear stages:** Plan, Design, Build, Test, Deploy, Maintain; each play documents what changes, getting started, concrete steps, governance considerations, and how to measure success.
2. **Artifacts:** early stages produce **.md files** (product owner and agent can both read and act on the same file); **from Build onward the artifact is code and its records**.
3. **Auditability:** the **chain of commits is the audit trail** — who asked for what, what the agent produced, who approved it.
4. **Why it matters:** manual handoffs were the friction; version-control artifacts make handover machine-triggerable while keeping humans at the approval points (governance considerations per play).

**Memory hook:** "Six stages, one version-control relay, markdown early → code later, commits = audit trail."

## D3. Circular validation: the broken loop and the fixed loop (Sample Q7)

**Question:** An LLM generates both production code and its unit tests. Explain why this is unreliable and draw the mechanism that prevents circular validation.

**Concepts tested:** circular validation; independent oracles; TDD; mutation testing. **Source:** CS13.

**Model answer:**
1. **Why unreliable:** when code and tests come from the **same internal model, prompt, or implementation assumptions**, they share the same blind spots — a wrong assumption is implemented AND asserted true; the suite passes while encoding the same misunderstanding (self-referencing feedback loop). Happy-path bias compounds it: generated code fails rare/stress conditions that generated tests never probe.
2. **Broken loop:**
```
Requirement → LLM → code ─┐
            └─► LLM → tests ┘ → tests "pass"  (same assumptions both sides)
```
3. **Fixed loop:**
```
Requirement/Spec (ground truth)
        │                                  LLM implementation
        ├──► tests derived FROM SPEC ──► verify code ──► pass/fail
        ├──► TDD: tests exist BEFORE code
        └──► computational verification gates + mutation testing:
             mutate code → suite MUST catch mutants → proves detecting power
             integration tests across module seams; human checkpoint at review
```
4. **Mechanism:** **(1) ground test generation in specifications, not implementation** (independent oracle); **(2) TDD** ordering; **(3) computational verification gates & mutation testing** to prove the suite can detect faults; plus integration testing and human checkpoints so no single model owns both sides of the equation.

**Memory hook:** "One brain writing both sides = echo chamber; spec writes the tests, mutation proves the tests."

## D4. The Lab 2 harness loop (feedforward → generate → sensors → bounded repair)

**Question:** Draw the coding-agent harness loop from the lab and annotate the control applied at each stage.

**Concepts tested:** feedforward files; static/dynamic sensors; findings-only repair; attempt bound; audit. **Source:** Lab 2.

**Model answer:**
```
FEEDFORWARD (before any code exists)
  openapi.yaml (single source of truth) + style-guide.md + harness-rules.md + generation-task.md
        │
        ▼
  Coding agent / LLM  ──►  app.js + app.test.js          (schema-validated JSON output)
        │
        ▼
  STATIC SENSOR  (reads source: files exist, POST /items exercised,
                  exact 201/400 assertions present, server exported, JSON header)
        │ fail → findings only (e.g. MISSING_400_ASSERTION) ──► repair prompt (≤2 attempts,
        │         preserve valid behavior) ──► back to generation
        ▼ pass
  DYNAMIC SENSOR (node --test app.test.js; exit 0 = pass)   [skipped until static clean]
        │ fail → findings only ──► bounded repair
        ▼ pass
  AUDIT LOG (sensor-report.json + attempt-log.json) ──► HUMAN REVIEW ──► trusted
```
1. **Feedforward constrains before generation**; sensors are deterministic and outside the LLM.
2. **Only observed findings** return to the agent — no conversation drift; findings are coded and machine-readable.
3. **Bounded repair:** two attempts maximum; unbounded loops risk cost blowup and sensor-overfitting.
4. **Trust rule:** artifacts are untrusted until both sensors pass, and human review is still recorded (auditability).

**Memory hook:** "4 files in, 2 files out, static → dynamic → log → human; findings only, twice max."

## D5. The Spec Kit command workflow

**Question:** Draw the Spec Kit command chain with its verification points and explain where human judgment enters.

**Concepts tested:** 9 commands; clarify/analyze/converge roles; artifact flow. **Source:** CS9 + Lab 1.

**Model answer:**
```
/speckit.constitution ──► /speckit.specify ──► /speckit.clarify ──► /speckit.plan
   principles              feature spec          resolve ambiguity       technical plan
                                                     │ answers folded back      │ (must comply with
                                                     ▼ into spec.md             │  constitution + spec)
                              /speckit.checklist ──► /speckit.tasks ──► /speckit.analyze
                               quality checklists      task graph          consistency check
                                                                              │ fix upstream if needed
                                                                              ▼
                                              /speckit.implement (increments) ──► /speckit.converge
                                              review each increment                gap report → fix → re-verify
```
1. **Constitution** defines project-wide principles first; every later artifact must comply.
2. **Clarify** kills ambiguity after specify; **analyze** checks requirement–plan–task consistency **before implementation** and re-runs after any upstream change; **converge** finds remaining gaps **during/after implementation**.
3. **Human judgment enters at:** answering clarify questions; approving the plan (senior review = mitigation for cascading errors); inspecting each implemented increment before acceptance; resolving gaps converge reports. Artifacts + git commits preserve the decision trail.

**Memory hook:** "Constitution → specify → clarify → plan → checklist → tasks → analyze → implement → converge; humans answer, approve, inspect."

## D6. The operations-to-development feedback loop (Sample Q10)

**Question:** Production failures recur despite green development stages. Design a feedback loop that uses telemetry and incident data to improve future AI-assisted development — without leaking secrets or feeding wrong learning signals.

**Concepts tested:** continuous observability; drift; shadow testing; feedback flywheel; security. **Source:** CS13 + CS11 + CS14.

**Model answer:**
1. **Why it recurs:** development gates verify the happy path; recurring failures signal what stages missed — so production observability must become a **continuous** input, not a one-time pre-release gate.
2. **Loop:**
```
Production telemetry + incidents
        │ (data & model drift monitoring; log parsing/root-cause diagnostics)
        ▼
Triage & generalize: incident → failed scenario → missing spec/acceptance criterion
        │
        ├──► Update spec + tests (acceptance criteria first — independent oracle,
        │     so fixes are grounded, not pattern-copied)
        ├──► Update guardrails/standards (new pitfall → encoded team standard;
        │     recurring finding → feedback flywheel / .lattice/ institutional memory)
        └──► Validate candidate changes safely: shadow mode / A-B trajectory
              comparison before full rollout
        │
        ▼
Next AI-assisted development cycle uses the enriched context
```
3. **Wrong learning signals — prevent:** never feed raw incidents as "truth" without root-cause analysis (a symptom can teach the wrong lesson); prefer spec/acceptance-criterion updates over prompt-by-prompt patching; validate prompt/model changes with **shadow testing** (real traffic, shadow mode, compare trajectory success rates) before adoption; monitor for **data/model drift** so stale lessons get retired.
4. **Security — prevent leaks:** **PII redaction & tokenization** on any telemetry entering AI context (input guardrail); topic/policy filtering; granular access so operational secrets don't ride along into prompts; audit every feedback artifact.
5. **Measure the loop:** MTTR (still honest), rework rate, defect recurrence — proving the loop actually reduces repeat failures.

**Memory hook:** "Telemetry → root-cause → spec/tests + standards → shadow-validate → redacted context only."

# CATEGORY E — Critical analysis (5 questions)

## E1. Governance and accountability when an autonomous agent ships a vulnerability (Sample Q9)

**Question:** An autonomous AI agent introduces a security vulnerability into a production application. Discuss the ownership and accountability model; explain the role of HITL approval, shift-left policy enforcement, dynamic quality gates and liability.

**Concepts tested:** governance; guardrail types; conformance gates; accountability. **Source:** CS11 + CS10 + CS13.

**Model answer:**
1. **Ownership:** accountability stays with **people and process, never the tool**. The agent is an instrument; ownership chains to (a) the **approver** who let the change ship (workflow guardrail failure — why was an irreversible production merge possible without human approval?), (b) the **pipeline owner** responsible for quality/security gates, (c) the **architect/team** who authored (or failed to author) constraints. An agent cannot hold liability — it has no duty, no licensure, and resets to zero institutional memory.
2. **Why it slipped through = a guardrail inventory failure:** action guardrails should have bounded what the agent could touch; security guardrails should have flagged the vulnerability class (hard-coded secrets, unsafe dependencies, SQL injection, insecure authorization, sensitive-data exposure); workflow guardrails should have required human approval before merge/deploy/irreversible operations.
3. **HITL approval:** the control point where a human accepts responsibility — auto-pause on high-risk actions (financial transactions, database writes); the approving human's review is the auditable moment accountability transfers.
4. **Shift-left policy enforcement:** catch the vulnerability **before it propagates** — conformance layer (fitness functions, dependency rules, plan-build approval gates, linting) evaluates code before it moves; security/threat modelling is one of the four SDD risk-control activities, and the cybernetic loop keeps quality left (sensors verify at generation time, not post-deployment).
5. **Dynamic quality gates:** gates that run continuously — static + dynamic sensors in CI, guardrail validation so "velocity gains do not spike technical debt or failure rates", monitored CFR/rework rate after release; a green build is never a moral guarantee, so gates + telemetry remain active in production.
6. **Liability:** documented controls (feedforward rules, sensor reports, attempt logs, human approvals) determine who is answerable — the audit trail (commit chain: who asked, what the agent produced, who approved) is the legal artifact. **Lesson:** the more autonomous the agent, the more the surrounding system must be deterministic (guiding rule) — an agent that ships alone must be fenced by code, not intentions.

**Memory hook:** "Humans own outcomes; guardrails are the evidence; shift-left gates before, dynamic gates after; the commit chain convicts or clears."

## E2. Is deployment frequency meaningless under AI? Critically assess DORA

**Question:** "DORA metrics still tell us everything we need." Critique this claim for AI-augmented teams.

**Concepts tested:** per-metric breakage; attribution failure; rework rate. **Source:** CS14.

**Model answer:**
1. **The claim is half-true at best.** DORA was designed when humans wrote all code; it measures delivery outcomes but is **blind to how AI shapes the code itself**.
2. **Per-metric status:** **MTTR holds up** (response to failure is honest regardless of authorship); **CFR has partial value** (only production failures — misses the massive hidden churn before deployment); **Deployment Frequency and Lead Time become misleading without context** — DF inflates via AI boilerplate, scaffolding and duplicated test files; lead time hides the bottleneck shift (fast commits offset by exploding PR review queues = the verification tax).
3. **Attribution failure:** the dashboard shows CFR rose 20% but cannot say whether AI-generated code caused it; DF may rise from AI, better process, or just micro-commits.
4. **The invisible regression:** code churn roughly **doubled** — DORA didn't flag it because rewritten code typically didn't cause production failures. Quality shifted; metrics stayed green.
5. **What DORA needs:** pairing with **AI Code Share, Code Durability/turnover, Complexity-Adjusted output (TrueThroughput), Innovation Rate**, and the 2025 fifth metric **rework rate** (unplanned fixes to production). Verdict: DORA is not meaningless — it is **necessary but insufficient**; deployment frequency specifically is "substantially less informative" and must be paired with new metrics.

**Memory hook:** "MTTR lives, CFR limps, DF/LT lie — add share, durability, rework before believing green."

## E3. The SDD critique: when specs create a false sense of control

**Question:** Critically evaluate spec-driven development with markdown artifacts (spec.md/plan.md/constitution.md): its risks across execution, architecture and governance, and how far the mitigations actually go.

**Concepts tested:** SDD challenges; 4 risks + mitigations; limits. **Source:** CS8 + CS9.

**Model answer:**
1. **General challenges:** SDD tools are **harder to evaluate on brownfield codebases**; they create a **false sense of control** (a tidy spec is not governed code); teams end up **reviewing markdown instead of reviewing code**.
2. **Execution/architecture risks:** **dependency & API hallucinations** (invented libraries); **shallow edge-case and non-functional coverage** (happy-path specs); **spec-to-code drift** (code changes without spec updates — the spec stops being the truth); **cascading error propagation** (one unchecked plan error fans out through every task).
3. **Mitigations and their limits:** simplicity rules in constitution.md counter architectural bloat — but only if the constitution is enforced, not aspirational; plan.md as a senior-review PR gate counters cascading errors — but depends on reviewers actually catching semantic errors in prose; CI checks that fail on architecture-file commits without spec updates counter drift — but can't detect semantic drift where both files changed cosmetically; reference files (schema dumps, OpenAPI contracts) counter legacy-context loss — bounded by context-window size.
4. **Governance risk:** spec-first/as-source concentrate authority in a document agents can satisfy superficially — so the conformance layer (fitness functions, dependency rules, automated linting) must verify the **code**, and the four risk-control activities (requirements validation, architecture risk analysis, security/threat modelling, task decomposition validation) must run on artifacts, not intentions.
5. **Balanced verdict:** SDD genuinely delivers earlier clarity, shared truth, faster generation and predictable validation on greenfield; its failure mode is treating the spec as the control instead of the **spec + sensors + human gates** system. Controls that aren't executable are eventually fiction.

**Memory hook:** "Markdown isn't governance: drift, hallucination, shallow coverage, cascade — mitigate with executable checks or the spec is theatre."

## E4. "The architecture is the prompt" — why prompting cannot rescue a bad codebase

**Question:** Evaluate the claim that prompt engineering, agentic frameworks and better models can compensate for poor architecture in AI-augmented development.

**Concepts tested:** architecture-is-the-prompt; context engineering; cost-of-change; non-deterministic surface area. **Source:** CS10.

**Model answer:**
1. **The claim fails:** *"No prompt engineering technique, agentic framework, or advanced foundation model can rescue an architecture composed of tangled modules and cascading side effects"* — **the codebase outweighs the prompt**. Agents with zero institutional memory must discover structure every session; a tangled architecture feeds them pipes: full downstream tracing, exhausted context, attention dilution, hallucination.
2. **What actually determines agent performance:** deep modules/sinks, low coupling/high cohesion, honest interfaces, minimal side effects — the same old remedies — because they cap how much context any step needs and keep blast radius contained.
3. **Context engineering reframed:** filling the context window with exactly the right information for the next step is a **system design problem and a first-class architecture discipline** — information must *exist, live in the repo, and be findable* (repo maps, steering files, specs, ADRs). That is architecture, not prompting.
4. **Cost-of-change lens:** AI made CRUD/business-logic changes cheap, but **invariants, data topology, and non-deterministic boundaries stayed expensive** — prompt tricks don't migrate terabytes or bypass ACID; they add a new architectural surface (fallback policies, latency budgets, guardrails) that must be designed.
5. **What changes for architects:** author constraints before the sprint (versioning policy, behavioral constraints, dependency rationale, exception logic); define invariants, not signatures; judge decisions by **blast radius, auditability, latency budgets, invariant protection**. Verdict: better models raise the ceiling of what agents can attempt; architecture decides whether attempting is survivable.

**Memory hook:** "Models get smarter, context stays finite — pipes poison prompts; only architecture caps the blast radius."

## E5. The verification tax and the J-curve: why "40% faster" precedes "defects up"

**Question:** An organisation reports 40% faster task completion but rising defects. Using the J-curve, explain the dynamics and critique the management response "roll back the AI tools".

**Concepts tested:** J-curve forces; verification tax; pipeline adaptation; metric blindness. **Source:** CS14.

**Model answer:**
1. **J-curve dynamics:** AI adoption follows a **temporary productivity dip and instability**, then recovery above the pre-adoption baseline. Three forces drive the trough: the **learning curve** (fluency has a measurable cost), the **verification tax** (reviewing AI output at AI volume takes longer than teams provisioned), and **pipeline adaptation** (pipelines built for lower velocity need smaller batches, stronger automated testing, tighter version control before absorbing new throughput).
2. **Why defects rise specifically:** AI collapses the cost of **producing** code, not **verifying** it — unreviewed volume ships; churn (~doubled) is invisible when it doesn't fail production; classic DORA shows green while durability decays. "40% faster" is raw velocity — an inflated metric exactly when complexity-adjusted truth is needed.
3. **Critique of "roll back":** rolling back treats a **forecastable transition cost** as a tool defect. The correct response is to manage the trough: provision review capacity (price the verification tax into planning — PR cycle time), retune pipelines, encode standards so the same mistakes stop recurring (feedback flywheel), and measure with the right instruments — complexity-adjusted throughput, 30/90-day code turnover, rework rate, MTTR — to see whether the recovery is actually arriving.
4. **When rollback IS right:** if 30/90-day turnover and change-failure attribution show AI code is durably worse after the trough should have ended (learning curve absorbed), then the economics (Net ROI including rework cost) justify retreat. The decision must come from durability + ROI evidence, not from the dip itself.
5. **Verdict:** the organisation is likely mid-trough, mis-measuring with pre-AI instruments; fix the measurement and the pipeline before fixing blame.

**Memory hook:** "Dip is a price tag, not a verdict: learning curve, verification tax, pipeline retune — measure durability before you roll back."

---

# CATEGORY F — Case-study / lab-based (5 questions)

## F1. Lab 1 case study: SDD with Spec Kit on the Task Manager app

**Question:** Walk through the Spec Kit lab: the artifact chain, how the constitution influenced downstream artifacts, one ambiguity clarify resolved, and how the implementation was verified against the spec. Contrast with building from a single prompt.

**Concepts tested:** Lab 1 workflow; artifacts; verification; SDD vs prompt-first. **Source:** Lab 1 (+ CS9 commands).

**Model answer:**
1. **Setup & artifact chain:** `specify init --here --integration copilot --script ps` scaffolds the project; then `constitution → specify → clarify → plan → checklist → tasks → analyze → implement → converge`, committing milestones to git (audit trail), excluding secrets/env files.
2. **Constitution's influence:** principles like React+Node/Express, modular architecture, RESTful APIs with proper status codes, input validation + error handling + tests for critical logic, "never expose secrets", "prefer simple maintainable solutions" became **constraints the plan had to satisfy** — e.g., the plan's endpoint design (status codes), the test strategy (critical logic), and the refusal of unnecessary complexity all trace to constitution clauses; two principles that touch technical decisions: validation/testing and the secrets rule.
3. **Clarification example:** ambiguities such as whether due dates are optional, whether completed tasks can be edited, deletion confirmation, default sorting — each answer was **folded back into spec.md**. If unresolved: inconsistent overdue logic, contradictory filter behavior, and every downstream artifact inheriting the same hidden assumption.
4. **Verification:** acceptance criteria mapped to implementation; manual user journeys (create/edit/complete/filter/delete-confirm/overdue/empty state/invalid data/API-unavailable) plus automated tests (Vitest + RTL frontend, Node backend framework); `/speckit.converge` reported remaining gaps → fix → re-verify; `/speckit.analyze` re-run whenever upstream artifacts changed.
5. **Incremental implementation discipline:** implement small task groups (backend setup + data model first), run relevant tests, **never approve changes without inspection**.
6. **Contrast with a single prompt:** no shared artifacts → no constitution constraints, no clarify step, no consistency analysis; correctness rests entirely on one generation + spot checks; drift and missing requirements surface late (or never), and nothing in version control explains why decisions were made. SDD trades upfront discipline for traceable, verifiable alignment.

**Memory hook:** "Constitution constrains, clarify disambiguates, analyze aligns, converge closes gaps — increments, inspected, committed."

## F2. Lab 2 case study: the harness's static sensor and bounded repair

**Question:** In the lab the sensor rejected the first simulated attempt with `MISSING_400_ASSERTION`. Explain why checking an assertion in source differs from executing the test, why findings must be machine-readable, what could go wrong with unlimited repair attempts, and what still needs human review after all sensors pass.

**Concepts tested:** static vs dynamic; coded findings; bounded repair; human review; 409 extension. **Source:** Lab 2.

**Model answer:**
1. **The defect:** the simulated attempt 1 used the broad assertion `assert.ok(response.status >= 400)` instead of the required exact `assert.strictEqual(response.status, 400)`; the static sensor flagged `MISSING_400_ASSERTION` and **skipped the dynamic stage until static was clean**.
2. **Static vs dynamic:** checking the assertion in source proves the test **states the required exact oracle** — a static, input-quality property of the test suite itself (a broad assertion could hide wrong behavior like 500s "passing" as >= 400). Executing the test proves the **implementation actually satisfies that oracle** at runtime. One audits the test; the other audits the code. Both are needed: a correct oracle never run proves nothing; a green run with a weak oracle proves less.
3. **Machine-readable findings:** coded findings (`MISSING_400_ASSERTION`) are pasted directly into the repair prompt — the loop repairs **without human interpretation**, repairs stay minimal ("preserve valid behavior, fix only the findings"), and the attempt log stays deterministic and auditable.
4. **Unlimited repair risks:** infinite loops on the same failure; exploding token/API cost; **prompt creep** (repair prompts growing until they override the original rules); **overfitting to the sensor** (making findings disappear instead of making the code correct); masking the need to escalate. The two-attempt bound forces the loop to stop and hand a curated evidence trail to a human.
5. **Still human after green:** sensors verify contract conformance, not intent — a human still reviews whether the implementation **fits purpose and architecture**, whether the contract itself was right, and whether the tests assert what the business meant; the audit note records exactly this review. (Extension case: adding HTTP 409 = update `openapi.yaml` first, then harness rules + static-sensor patterns, then regenerate — contract and sensors move together.)

**Memory hook:** "Static audits the oracle, dynamic audits the code; findings are data, budget is two; green ≠ approved."

## F3. Legacy modernization case: safe library replacement / migration with agents (Sample Q6)

**Question:** An enterprise must migrate a large codebase off an outdated library (monolith-to-modular refactor of dependencies) using AI. Design a safe migration workflow covering dependency identification, generation, guardrails, compatibility, verification, and human approval.

**Concepts tested:** SEI framework; decomposition; RevAPI+LLM division; cybernetic loop; write-once rollout. **Source:** CS11 (+ CS13 verification).

**Model answer:**
1. **Never delegate the whole migration to an LLM.** Decompose into discrete sub-tasks and **pair generative AI with deterministic tools** (SEI shift-left framework). Context: outdated libraries sit **dozens of versions behind** (FY22 NDAA Section 835 findings) → security and operational risk; manual refactors are too slow and error-prone.
2. **Dependency identification (computational sensor):** static analysis — **RevAPI** for Java — formalizes API **breaking changes** (removals, signature changes, visibility changes) **completely, consistently, without hallucination**. The machine, not the model, owns facts.
3. **Code generation/refactoring (inferential synthesizer):** LLM generates **transformation strategies and recipes** for the identified changes; feedforward = formal diffs + upgrade guidelines as bounded context; guardrails keep the LLM from touching anything outside the mapped breaking changes.
4. **Deterministic verification (actuator + feedback sensor):** automated transformation engines **apply** the recipes; then **build + test suite** run — failures are injected back into the LLM context to refine the recipe **until convergence on a clean build**; exact contract assertions (not broad ones) keep oracles honest and avoid circular validation.
5. **Compatibility & version management:** recipes carry upgrade guidelines; API-compatibility is enforced by the sensor layer (tests + static checks), not by the model's confidence.
6. **Write once, roll out everywhere:** once verified on one repository, the recipe is a **reusable deterministic asset** executed across dozens of dependent projects **without re-running the LLM** — eliminating per-project hallucination risk.
7. **Human approval:** sustainers review generated recipes before publication; application developers review applied changes before merge (workflow guardrail); audit logs record each rollout.

**Memory hook:** "Sensor maps the damage, LLM writes the recipe, engine applies, build/test judges, verified recipe replays everywhere."

## F4. Uber's Software Factory as an end-to-end blueprint

**Question:** Describe Uber's Software Factory lifecycle and its infrastructure pillars; what should a mid-size organisation adopt first and why?

**Concepts tested:** inner/outer loop; six pillars; adoption sequencing. **Source:** CS13.

**Model answer:**
1. **Lifecycle:** **Ideation & Alignment (inner loop)** — humans and AI align on intent (the spec-first moment); **Autonomous Execution & Validation** — agents implement with automated verification; **Outer Loop & Maintenance** — production feedback, drift handling, ongoing improvement.
2. **Six infrastructure pillars:** Model Gateway (controlled model access), MCP Gateway & Tool Optimization (managed tool calling), Agentified DevPods (execution environments), Managed Skills Marketplace (shared, versioned skills — the Lattice "skills over prompts" idea at platform scale), AI Context Graph (institutional knowledge as queryable context), AI Assistant (the developer-facing surface).
3. **What to adopt first:** the **context + gateway layers** (Model Gateway, AI Context Graph) — they deliver guardrailed access and shared context, the prerequisites for everything else; then **managed skills** (stop re-typing standards), then **DevPods** for safe autonomous execution with sensors; the outer loop (telemetry → context) last, once artifacts exist to update.
4. **Why this order:** without context infrastructure, agents hallucinate (zero institutional memory); without gateways, there is no audit or cost control (Utilization/Impact/Cost measurement becomes impossible); validation loops only pay off when context and skills make their findings actionable.

**Memory hook:** "Align inside, execute validated, maintain outside — gateway and context graph before autonomy."

## F5. Reproducibility vs variability: Lab 2 Parts A/B/C compared

**Question:** Compare the manual, simulated, and live-LLM runs of the harness (generation, feedforward, feedback, repair, attempt limits, audit evidence) and analyze what Part B proves that Part C cannot, and vice versa.

**Concepts tested:** verification matrix; determinism vs sampling; schema validation; audit evidence. **Source:** Lab 2.

**Model answer:**

| Control | Part A — manual | Part B — simulation | Part C — live LLM |
|---|---|---|---|
| Generation | VS Code chat agent | Predefined strings, **no LLM** | External Gemini call (temperature 0.1) |
| Feedforward | Files attached manually | Same artifacts represented | Files read + concatenated into prompt |
| Feedback | sensor.js run manually | sensor.js invoked by demo.js | sensor.js invoked by demo1.js |
| Repair | Student submits findings | Predetermined corrected output | Findings sent to the LLM |
| Attempt limit | Student stops after 2 | 2 | 2 |
| Evidence | manual-attempt-log.json | demo-output/attempt-log.json | live-output/attempt-log.json |

1. **What B proves:** the **harness logic itself** is sound and fully **reproducible** — same inputs, same findings, same outcome every run; it isolates sensor correctness from model noise (attempt 1 fails `MISSING_400_ASSERTION`; attempt 2 repairs; SUCCESS), with zero network/API dependencies.
2. **What C adds:** proof the loop survives a **probabilistic generator** — variability handled by feedforward + **schema validation before any file is written** (exactly `appJs`/`testJs`, no fences), temperature lowered (0.1) but not eliminated, transient API failures (429/503) retried up to three times then **recorded separately from code-validation failures**. A first-attempt success is valid; repair happens **only** when deterministic findings exist.
3. **Vice versa:** B cannot show how a real model behaves (malformed JSON, fence-wrapped output, creative violations) — that is C's evidence; C cannot separate harness bugs from model noise — that is B's.
4. **Shared invariants across all three:** findings-only repair, two-attempt bound, static-then-dynamic ordering, audit logs — demonstrating that the **control architecture, not the model, is what makes the workflow trustworthy**.

**Memory hook:** "A shows the human the loop, B proves the loop deterministic, C proves the loop tames the model."

---

# Index — mapping your original 16 questions to this bank

| Original sample question | New ID | What changed |
|---|---|---|
| Q1 Evolution of SDLC under AI | **D1** | Answer grounded in CS8 bottleneck shift + CS10/CS11 quality mechanisms |
| Q2 Non-deterministic outputs + validation | **B3** | Grounded in apparent non-determinism (CS11) + harness/guardrails (Lab 2, CS11) |
| Q3 AI-assisted requirements engineering | **B1** | Grounded in CS8/CS9 elicitation-synthesis-management, INCOSE/EARS, HITL |
| Q4 AI-generated architecture + ADRs | **B6** | Grounded in CS9 risk-control activities + CS10 authored constraints/governance |
| Q5 Pair programming vs autonomous agents | **C1** | Grounded in CS11 modes + CS10 harness-vs-orchestrator guiding rule |
| Q6 Legacy refactoring with agents | **F3** | Grounded in CS11 SEI library-replacement case (RevAPI + LLM + loop) |
| Q7 Circular validation | **D3** | Grounded in CS13 (specs-as-oracle, TDD, gates + mutation testing) |
| Q8 AI-driven CI/CD | **B7** | Re-grounded: TIA selection, workflow guardrails, AWS guardrail-validation phase, J-curve pipeline adaptation |
| Q9 Governance and accountability | **E1** | Grounded in CS11 guardrails + CS10 conformance + audit trail |
| Q10 Operational feedback loops | **D6** | Grounded in CS13 drift/shadow testing + CS11 flywheel + PII redaction |
| Q11 Measuring AI SDLC effectiveness | **B9** | Grounded in CS14 hierarchy, pillars, turnover, rework rate |
| Q12 Cost and economics | **B8** | Grounded in CS14 ROI/Net ROI/J-curve cost formulas |
| Q13 Foundation models & transformers | **A1** | **Rewritten**: transformer/tokenisation are outside the supplied files; same concept-verification pattern applied to non-determinism (CS11) |
| Q14 Prompt engineering for production APIs | **B2** | **Re-grounded** as feedforward/context strategy (Lab 2 + knowledge priming) |
| Q15 Agentic workflows | **C4** | Grounded in CS10 harness/orchestrator + Lab 2 loop |
| Q16 AI-assisted code review | **B10** | Grounded in CS13 8-step review + CS11 guardrail layers + Lattice review molecule |

**Coverage check:** every question in this bank cites CS8–CS14 or Lab 1/Lab 2 only. 45 questions = A12 + B10 + C7 + D6 + E5 + F5.
