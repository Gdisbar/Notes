# AI-Augmented SDLC (SEZG534) — EC3 Study Notes

**Prepared for:** Comprehensive Exam (EC3), BITS WILP — used as closed-book preparation notes
**Sources:** Course handout (8 modules), CS9 Requirements & Design deck, CS14 Software Delivery Metrics deck, Lab Sheet 2 (Coding Agent Harness), Sample Questions paper, plus your 16-question analysis bank (expanded to 49 questions with model answers).

---

## 1. What is in this pack

| File | Contents | Use it for |
|---|---|---|
| `01_Concept_Notes.md` | Part A: all 8 modules, one concept per section, each ending with an exam-ready skeleton; Part B: 8 memorizable text flows | Building understanding + memorizing answer skeletons |
| `02_Question_Bank.md` | 49 questions in your 6 categories, all with full model answers | Practising reproduction of answers under time pressure |
| `03_Last_Day_Cheat_Sheet.md` | Definitions, mnemonics, formulas, number anchors, dispatch table | Final 24 hours before the exam |
| `EC3_All_In_One.md` | Everything above in one file | Searching across the whole pack |

---

## 2. Your question-pattern analysis → where it is covered

| Question category (your analysis) | What examiners ask | Covered in |
|---|---|---|
| Conceptual / descriptive | SDLC evolution, AI fundamentals, prompting | QBank Category A; Modules 1–2 |
| Scenario-based application | Apply AI tools to a given SDLC scenario | QBank Category B; Modules 3–6 scenarios |
| Compare and justify | Compare AI workflows, tools, validation strategies | QBank Category C; comparison tables in every module |
| Diagram / workflow | AI-augmented SDLC workflow, HITL, CI/CD | QBank Category D; Part B flows (arrow notation only) |
| Critical analysis | Governance, hallucinations, circular validation | QBank Category E; Modules 5, 7, 8 |
| Case-study based | Legacy modernization, AI-assisted development labs | QBank Category F; Module 4 + Lab Sheet 2 section |

**Pattern insight from the sample paper:** every question is a short scenario (4–6 lines) followed by 2–3 directive verbs (Evaluate / Analyze / Explain / Identify), each worth ~3 marks. Your answers must therefore: (1) name the concept, (2) define it in one line, (3) apply it to the scenario's specifics, (4) give one example or number. Three sentences per mark is a safe density.

---

## 3. Closed-book strategy (because you are effectively writing it closed-book)

1. **Memorize skeletons, not paragraphs.** Every concept section in `01_Concept_Notes.md` ends with an "Exam skeleton" — a numbered list of 5–8 memory points. In the exam, reproduce the skeleton first, then expand each point into 2–3 sentences.
2. **Memorize numbers as anchors.** Numbers make answers concrete and are hard to invent under pressure: 4 DORA metrics + rework rate, 5 pillars of the Developer AI Impact Framework, 2 generation attempts in the harness, 30/90-day code turnover windows, 3 J-curve forces, temperature 0.1 in the live harness. Full list in the cheat sheet.
3. **Memorize one comparison table per module.** Compare-and-justify questions are almost guaranteed (your pattern analysis says so). Each module has one master table; if you can reproduce the table you can answer any comparison question.
4. **Diagrams: write flows, do not draw.** Per your instruction, all diagrams are text-only arrow flows (A → B → C). In the exam, write the flow on one line, then describe each stage in words. That earns diagram-question marks without drawing.
5. **Practise the 3-mark rhythm.** Sample paper parts are ~3 marks each: definition (1) + application to scenario (1) + example/justification (1). Practise writing 120–180 word answers against the clock.
6. **Use the dispatch table** (cheat sheet, last section): "If the question says X → open with Y." It is your fastest route into any answer when the clock starts.

---

## 4. Suggested 10-day revision plan

| Days | Task |
|---|---|
| 1–2 | Modules 1–2 concept notes + QBank A |
| 3–4 | Modules 3–4 + QBank B1–B5, C1–C3 |
| 5 | Modules 5–6 + QBank E1–E3, D2 |
| 6 | Modules 7–8 + QBank E4–E8, C4–C7 |
| 7 | Part B flows + QBank D (all workflow questions) |
| 8 | QBank F case studies + Lab Sheet 2 concepts |
| 9 | Attempt 6 random questions from the bank, closed-book, timed (25 min each) |
| 10 | Cheat sheet only; rewrite all skeletons from memory once |

---

## 5. Syllabus coverage check (against the course handout)

- Module 1 (Evolution, velocity, risk, economics) — Sections 1.1–1.6, QBank A1, B1, B8, E4, E5
- Module 2 (AI fundamentals, prompting, agents) — Sections 2.1–2.7, QBank A2–A7, B4, B10
- Module 3 (Requirements, SDD, ADRs, threat modeling) — Sections 3.1–3.9, QBank B2, B3, B9, B11, C2, C3, D7, F7
- Module 4 (Coding, refactoring, guardrails) — Sections 4.1–4.6, QBank A8, B10, C1, C5, D6, E6, F1–F4
- Module 5 (Testing, QA, circular validation) — Sections 5.1–5.6, QBank E1, E3, C6, C7, F2
- Module 6 (CI/CD, release) — Sections 6.1–6.5, QBank D2, D5, B7
- Module 7 (Governance, HITL, liability) — Sections 7.1–7.4, QBank E2, D5, F3
- Module 8 (Feedback loops, metrics, health) — Sections 8.1–8.7, QBank B5, B6, C4, D3, E5, E8, F6
- Lab Sheet 2 harness — Part A sections 2.5, 4.5, 4.6 + Flows F3/F4 + QBank D4, F2, E7
- Sample paper questions — QBank E4 (Q1) and B8 (Q2)
# PART A — CONCEPT NOTES (One Concept Per Section)

> How to use Part A: read the concept, then memorize the **Exam skeleton** at the end of each section. In the exam, write the skeleton first and expand each point into 2–3 sentences.

---

# MODULE 1 — FOUNDATIONS AND PARADIGM SHIFT: EVOLUTION, VELOCITY, RISK

## 1.1 Evolution of SDLC models: Waterfall → Agile/DevOps → Continuous AI-Augmentation

**Waterfall (1970s–1990s).** Linear, sequential phases: Requirements → Design → Implementation → Testing → Deployment → Maintenance. Each phase is signed off before the next begins. Strengths: documentation, predictability, contract-friendly. Weaknesses: late feedback (testing happens last), change is expensive, customers see software only at the end.

**Agile (2001 onwards).** Iterative and incremental: short sprints, working software every 1–4 weeks, continuous customer feedback, embrace change. Scrum/Kanban ceremonies replaced phase gates. Weaknesses addressed: late feedback. New weakness: architecture can erode under constant refactoring ("emergent design" needs discipline); delivery still gated by manual release processes.

**DevOps / Continuous Delivery (2010s).** Extends Agile past "done in the sprint" to "running in production": CI/CD pipelines, infrastructure as code, automated testing, deployment frequency as a first-class goal. DORA metrics (2018, *Accelerate*) gave it a measurement language.

**Continuous AI-Augmentation (now).** AI assistants and agents are embedded in every phase simultaneously — requirements drafting, design proposals, code generation, test generation, review support, deployment automation, incident triage. The lifecycle stops being a relay race between phases and becomes a **continuous loop** where AI compresses each hand-off.

**What AI actually changes per phase (the "AI-Augmented Lifecycle Phases"):**

| Phase | Traditional | AI-augmented | New risk |
|---|---|---|---|
| Requirements | Interviews, BRD/SRS documents | AI extracts requirements from transcripts, tickets, telemetry; drafts user stories; scores ambiguity | Omitted or hallucinated requirements |
| Design/Architecture | Human architects, review boards | AI proposes architecture, drafts ADRs, prototypes | Architectural drift, over-complex designs |
| Coding | Developers write code | Pair programming, autonomous agents, code generation | Non-determinism, hallucinated APIs, technical debt |
| Testing | Manual + automated suites | AI generates tests from specs, predicts defects | Circular validation (AI testing AI) |
| Review | Peer review | AI-assisted automated review | False positives, over-reliance |
| Deployment | Scripted pipelines | AI dependency resolution, smart test selection, canary analysis | Automated errors at scale |
| Operations | Monitoring, on-call | Self-healing, AI triage, telemetry feedback loops | Wrong learning signals |

**Exam skeleton (evolution):**
1. Waterfall = sequential, phase-gated, documentation-heavy, late feedback.
2. Agile = iterative sprints, continuous feedback, embraces change.
3. DevOps = extends Agile to production via CI/CD; DORA metrics measure it.
4. AI-augmentation = AI embedded in every phase; lifecycle becomes a continuous loop, not a relay.
5. AI compresses cost of *producing* artifacts (code, tests, docs) but not the cost of *verifying* them.
6. Each phase gains speed but also a new AI-specific risk (hallucination, drift, circular validation).

**Memory hook:** the through-line of the whole course: *AI collapses the cost of producing code; it does not collapse the cost of verifying it.*

## 1.2 AI-driven change velocity and amplification of engineering risk

AI accelerates change velocity: more code, more commits, more PRs, more deploys per unit time. Velocity itself is neutral — the danger is that **risk scales with velocity while human verification capacity does not**. The handout names four amplified risks (mnemonic **D-HAT**):

1. **Technical debt (D).** AI generates code fast and in volume; if review, refactoring and test investment do not scale equally, shortcuts accumulate silently. Debt also becomes invisible: churn is hidden as rewrites that never reach production.
2. **Hallucination risks (H).** Models generate plausible-but-wrong content: non-existent APIs, invented library functions, wrong business rules, confident security holes. Hallucinations look exactly like correct code, so they pass casual inspection.
3. **Architectural drift (A).** Many small AI-generated changes each locally sensible, none globally checked → the system's actual structure diverges from its intended architecture: duplicated patterns, inconsistent layering, new implicit dependencies, "architectural bloat" (LLM tendency toward maximal complexity).
4. **Automated errors (aT).** Errors are now produced *by automation at machine speed*: a bad prompt template, a flawed agent tool, or a wrong AI-generated pipeline change propagates to hundreds of artifacts before a human notices. Errors also compound: AI code → AI tests → AI review → AI deployment, each stage trusting the previous one.

**Why traditional gates fail under AI velocity.** Manual code review and manual QA were designed for a human code volume. When PR volume rises 40%+ overnight, queues explode (the "verification tax"), reviewers rubber-stamp, and gates degrade from *verification* to *formality*.

**Mechanisms to maintain quality at high velocity (this is a guaranteed question):**
- **Deterministic verification gates**: schema validation, static analysis, exact-assertion test suites, contract tests (see the Lab harness, Part A §9).
- **Bounded automation**: agents may attempt repairs only N times (lab uses 2) before escalating to a human.
- **Shift-left policy enforcement**: security/quality policies checked at commit time, not release time.
- **Human-in-the-loop (HITL) approval points** for high-risk changes (auth, payments, data migration).
- **Feedback loops from operations**: telemetry and incidents feed back into prompts, specs and test suites.
- **Balanced metrics**: track quality (code turnover, change failure rate) alongside throughput, so debt is visible.

**Exam skeleton (velocity and risk):**
1. AI raises change velocity; risk scales with velocity, human verification does not.
2. D-HAT: Debt, Hallucination, Architectural drift, automated errors at machine speed.
3. "AI collapses the cost of producing code, not of verifying it" — the verification tax.
4. Traditional manual gates (review, QA) overload and degrade under AI PR volume.
5. Quality mechanisms: deterministic gates, bounded repair loops, shift-left enforcement, HITL points, ops feedback, balanced metrics.
6. Conclusion line: velocity is safe only when verification capacity grows with it.

## 1.3 Inversion of Engineering Value

**Definition (memorize this):** When code generation becomes nearly free, engineering value **inverts** — it moves *away from producing code* and *toward specifying intent, verifying outputs, and governing change*. Specifications become the source of truth; code becomes a generated, disposable, verified artifact. Spec-driven development "inverts the traditional workflow by treating specifications as the source of truth and code as a generated artifact."

**Value before inversion:** writing code, refactoring by hand, tribal knowledge of implementation.
**Value after inversion:** specification quality (clear, testable, complete specs); verification infrastructure (tests, sensors, gates); governance (ADRs, HITL, audit trails); context engineering (feeding the right knowledge to AI).

**Consequences:**
- The scarce skill is no longer typing code; it is writing unambiguous specifications and designing verification that catches wrong code.
- "Disposable code" becomes rational *only if* verification is stronger than generation — regenerate-from-spec works only when the spec captures all behaviour and a deterministic harness proves equivalence.
- Teams that invert value correctly invest in: spec repositories, ADRs, test oracles, evaluation harnesses, guardrails.
- Teams that misread the inversion ("code is free, so stop testing") destroy the mechanism that makes code disposable — you cannot safely throw away code you cannot verify.

**Exam skeleton (inversion):**
1. Definition: value moves from code production → specification, verification, governance.
2. Spec = source of truth; code = generated, disposable artifact (spec-as-source end of the spectrum).
3. New scarce skills: spec authoring, context engineering, verification design, governance.
4. Disposable code is safe ONLY with deterministic verification proving spec conformance.
5. Misreading the inversion ("stop writing tests") removes the safety net that justifies disposability.

## 1.4 Local optimization vs systemic enablers (the PR-throughput scenario)

**Local optimization:** improving one stage of the pipeline (coding) while ignoring the whole flow. A 40% jump in PR throughput with unchanged 4-week lead time means the *constraint moved elsewhere* — review queues, QA environments, deployment configuration, meetings. Queuing theory logic: the fastest stage in a pipeline with a slow downstream stage just builds inventory (unread PRs).

**Why it happens:** AI licenses were bought per-developer (local), but lead time is a *system* property — it includes waiting time in review, environment provisioning, and release. Developers "write code faster but spend afternoons waiting" — classic symptom that coding is no longer the bottleneck.

**Systemic enablers:** organization-wide capabilities that raise flow for everyone rather than output for one person:
- Platform engineering / internal developer platform: self-service environments, golden paths, paved-road pipelines.
- Automated verification infrastructure: fast CI, exact-assertion suites, contract tests, smart test selection.
- Spec and context infrastructure: OpenAPI contracts, ADRs, internal RAG knowledge bases that make AI output *fit* the system.
- Flow management: WIP limits, PR pickup-time and review-turnaround metrics, queue monitoring.
- DORA's finding: AI **amplifies** existing system properties — good foundations accelerate; weak foundations amplify friction.

**Two technical engineering foundations to prioritize (3-mark favourite):**
1. **Automated CI/CD and testing infrastructure** (deterministic gates, environments-as-code, smart test selection) — absorbs higher throughput safely.
2. **Specification and context infrastructure** (specs/contracts, ADRs, RAG-indexed internal knowledge) — makes AI-generated work conform to system intent instead of drifting.

**Exam skeleton (local vs systemic):**
1. Define local optimization: improving one stage; lead time is a system property.
2. Scenario diagnosis: constraint moved from coding to review/environments/deployment; PRs pile up as inventory.
3. AI amplifies the system: strong foundations → acceleration; weak foundations → friction amplification.
4. Systemic enablers: platform engineering, automated verification, spec/context infrastructure, flow metrics.
5. Two foundations: (a) CI/CD + automated testing infrastructure, (b) specification + context infrastructure.

## 1.5 Cost and economics of AI augmentation

**Cost side (per engineer, per month — "fully loaded" means more than licence):**
- Licence/seat cost; token/usage (inference) costs; AI infrastructure and compute; training and enablement time; admin/integration overhead; **review overhead** (the verification tax — humans reviewing machine-generated volume); **defect remediation** cost (AI code that must be rewritten); security-risk cost (vulnerabilities shipped, incidents).

**Benefit side:**
- Time saved per engineer per week × loaded engineering cost; faster time-to-market (revenue earlier); quality gains if verification improves; reclaimed hours redirected to higher-value work; developer satisfaction/retention.

**Two formulas (memorize exactly):**
- **ROI (%) = (Value − Investment) / Investment × 100**
- **Net ROI = (Productive Value of Time Saved − Rework Cost from Code Turnover) / Total AI Tool Cost**
- **J-curve dip cost = staff size × fully loaded salary × productivity drop × dip duration**

**The AI adoption J-curve (CS14):** productivity temporarily *dips* after AI adoption, then recovers above baseline. Three forces drive the trough (mnemonic **L-V-P** — "Learn, Verify, Pipe"):
1. **Learning curve** — fluency with tools/workflows has a real ramp cost.
2. **Verification tax** — reviewing AI output volume takes longer than teams provisioned for.
3. **Pipeline adaptation** — pipelines built for lower change volume need retuning (smaller batches, stronger automated tests, tighter version control) before they absorb new throughput.
Management implication: **treat the dip as a forecastable transition cost**, not as evidence AI has failed; do not judge ROI in week 4.

**Exam skeleton (economics):**
1. Fully-loaded costs: licences, tokens, training, review overhead, defect remediation, security risk.
2. Benefits: time saved × loaded cost, faster TTM, quality, retention.
3. Formulas: ROI %, Net ROI (net of rework), J-curve dip cost formula.
4. J-curve: dip then recovery; forces = Learning curve, Verification tax, Pipeline adaptation.
5. Judge over quarters, not weeks; measure rework, not just throughput.

---

# MODULE 2 — FUNDAMENTALS OF AI AND PROMPT ENGINEERING

## 2.1 Foundation models and types

**Definition:** a foundation model is a large model trained on broad data at scale, adaptable by fine-tuning/prompting to a wide range of downstream tasks — it is a *foundation* on which many applications are built (Chip Huyen, *AI Engineering*, the course's R2 reference).

**Key properties:** emergent capabilities (skills not explicitly trained); task-agnostic; prompt-adaptive; scale-driven performance.

**Types relevant to software engineering:**
- **LLMs (text)** — requirements drafting, documentation, review comments.
- **Code models** — trained heavily on source code (Copilot-style completion, agents).
- **Multimodal models** — text + images (UI mockups to code), audio.
- **Embedding models** — text → vectors; power retrieval (RAG) and semantic search over code/docs.
- **Small/local models (e.g., Llama-3 via Ollama)** — data-sensitive environments; lower cost, lower capability.

**Training pipeline (one line each):** pretraining (predict next token on massive corpora) → fine-tuning (task specialization) → RLHF/instruction tuning (align to human preferences). Know that this is why models are *probabilistic next-token predictors*, which explains non-determinism (§2.5).

**Risks:** hallucination, bias, prompt injection, data leakage, knowledge cutoff (stale APIs), cost of inference.

**Exam skeleton:**
1. Foundation model = large, broad-training, adaptable base model for many tasks.
2. Types: LLM, code, multimodal, embedding, small/local.
3. Pipeline: pretraining → fine-tuning → RLHF; output is probabilistic next-token prediction.
4. Properties: emergent capability, prompt adaptability, scale.
5. Risks: hallucination, bias, injection, leakage, staleness.

## 2.2 Transformer architecture (software-engineering relevance)

**Why it matters:** all modern coding assistants are transformers; understanding them explains context windows, cost, and non-determinism.

**Pipeline (memorize in order):**
1. **Tokenization** — text split into tokens (§2.3).
2. **Embedding** — each token mapped to a high-dimensional vector capturing meaning.
3. **Positional encoding** — order information added (transformers have no recurrence).
4. **Self-attention** — every token attends to every other token; attention weights decide which tokens influence which. This is the core innovation: **long-range dependencies in one hop**. For code: a variable declared at the top "attends to" its use 200 lines later; attention is how the model links an API call to its definition.
5. **Feed-forward layers + residual connections + layer normalization** — per-token transformation, stacked N times (deep stack = "layers of reasoning").
6. **Output head** — probability distribution over the next token; sampling picks one; decoding repeats (autoregressive generation).

**SE-relevant consequences:**
- Attention → in-context learning: give good examples in the prompt and behaviour changes without retraining (few-shot prompting works *because* of attention).
- Attention is quadratic in sequence length → context windows are finite and costly → context management strategies exist (§2.4).
- Sampling from a probability distribution → same prompt can yield different outputs (§2.5).

**Exam skeleton:**
1. Pipeline: tokenize → embed → position-encode → self-attention → feed-forward blocks → next-token probability.
2. Self-attention = every token attends to every other token; captures long-range dependencies (variable ↔ usage, call ↔ definition).
3. Attention enables in-context learning → few-shot prompting.
4. Quadratic attention cost → finite, expensive context windows.
5. Autoregressive sampling → non-determinism.

## 2.3 Tokenization and its impact

**Definition:** splitting text/code into subword units (tokens) that map to model vocabulary entries; typically via byte-pair encoding (BPE). Rule of thumb: ~4 characters or ~0.75 words per English token; code often tokenizes *worse* (identifiers split: `getCustomerBalance` → `get`, `Customer`, `Balance` or worse).

**Impacts on software engineering:**
1. **Context budget:** everything in the prompt (specs, files, conversation) consumes tokens; hit the limit and the model truncates or forgets → silent quality loss.
2. **Cost:** APIs price per token; large-context agent loops are expensive; cost scales with prompt size, not task size.
3. **Correctness edge cases:** unusual identifiers, non-ASCII text, whitespace-heavy code tokenize unpredictably; long tokens can break completion quality.
4. **Non-determinism tie-in:** tiny token-level ambiguity + sampling = divergent outputs across runs.
5. **Impact of truncation on agents:** an agent whose context overflows loses harness rules or spec sections — exactly the failure the lab's feedforward-files approach mitigates by keeping inputs small and pointed.

**Exam skeleton:**
1. Tokenization = subword splitting (BPE); code tokenizes worse than prose.
2. Impacts: context budget limits, per-token cost, unpredictable splits on identifiers/unicode.
3. Truncation silently drops rules/specs → degradation agents must guard against.
4. Management: measure tokens, chunk context, keep prompts minimal (spec files, not whole repos).

## 2.4 Context windows

**Definition:** the maximum number of tokens the model can consider at once — the model's entire working memory (system prompt + rules + files + conversation + output).

**Why it matters for AI-assisted engineering:**
- Determining *what* fits in context is the practical skill of "context engineering": openapi.yaml + style-guide + harness-rules + task (the lab's exact four files) is a deliberate context design.
- **Relevance beats volume:** stuffing a whole repository degrades output ("lost in the middle" — models attend less to middle-of-context content); targeted extracts outperform dumps.
- **Context rot in agent loops:** long multi-turn sessions accumulate stale, contradictory context; fix by summarizing or restarting with a curated context.

**Management strategies (memorize 5):** chunking and retrieval (RAG) instead of full dumps; summarization of long histories; hierarchical context (constitution → spec → plan → task); explicit reference files (schema dumps, OpenAPI contracts); context budgets per agent step.

**Exam skeleton:**
1. Context window = model's full working memory in tokens.
2. Finite and costly (quadratic attention) → context engineering is a core skill.
3. More content ≠ better: relevance beats volume; middle-of-context loss.
4. Strategies: RAG retrieval, summarization, hierarchy, reference files, budgets.
5. The lab harness is a context-design example: 4 small files instead of a repo dump.

## 2.5 Non-deterministic AI outputs in deterministic software systems

**Why outputs differ across runs (memorize 4 sources):**
1. **Sampling:** the model outputs a probability distribution; temperature > 0 samples from it; temperature 0 is argmax (near-deterministic, but not guaranteed across infra changes).
2. **Server-side state:** provider-side batching, model version updates, floating-point non-associativity on different hardware.
3. **Prompt sensitivity:** tiny prompt/context changes (even token order) change the distribution.
4. **Agent stochasticity:** tool results, retries, and timeouts differ per run, so the *conversation* diverges even with the same model.

**Design a validation mechanism (this is QBank B4 — memorize the 6-layer stack):**
1. **Prompt constraints (feedforward):** temperature 0–0.1 for code; explicit output schema; fixed rule files; "return only JSON" instructions (lab harness rules).
2. **Schema validation:** validate the *model response* first (exact keys `appJs`/`testJs`, non-empty strings, no Markdown fences) before touching the repo.
3. **Static analysis (deterministic sensors):** linters, pattern checks, required exact assertions — checks that pass/fail identically on every run.
4. **Unit and integration testing:** run the generated tests against the contract; assertions must be *exact* (status 201/400), not broad (`>= 400`).
5. **Human approval:** human review is the final non-determinism firewall for high-risk artifacts.
6. **Release quality gates:** CI gates that block merge/deploy until static + dynamic + review all pass; bounded repair (max attempts) with audit logs.
Result: each AI run may differ, but **what enters production is deterministic** — every artifact passes the same deterministic checkpoints. Freeze behaviour at deterministic checkpoints: inputs (frozen prompts/specs), process (bounded loop), outputs (validated artifacts).

**Exam skeleton:**
1. Non-determinism sources: sampling/temperature, provider state, prompt sensitivity, agent tool variability.
2. Principle: allow variability in generation, forbid it in verification — deterministic sensors, not AI judges.
3. Six layers: prompt constraints → schema validation → static sensors → dynamic tests → human approval → release gates.
4. Bounded repair (N attempts) + audit log; escalate to human after bound.
5. One-line conclusion: "the pipeline is deterministic even when the model is not."

## 2.6 Prompting strategies for software creation

**Principles (memorize as: ROLE–TASK–CONTEXT–CONSTRAINTS–FORMAT):**
1. **Role/persona:** "You are a Node.js coding agent operating inside a controlled harness" — primes domain conventions.
2. **Task decomposition:** break big requests into small, verifiable steps (generate function → generate tests → repair findings). Smaller tasks = higher correctness + easier validation.
3. **Context management:** attach the minimum sufficient, authoritative context (contracts, style guides) — not whole repos.
4. **Constraints:** enumerate what is forbidden (no new dependencies, no hardcoded secrets, no broad assertions); forbidden lists reduce hallucinated APIs.
5. **Output format specification:** exact schema ("JSON with keys appJs, testJs; no Markdown fences") makes output machine-checkable.
6. **Few-shot examples:** show one ideal input→output pair; attention does the rest.
7. **Chain-of-thought / plan-then-code:** ask for a plan or reasoning for complex tasks; but for *machine-consumed* outputs, ask for JSON only.
8. **Iterative refinement with feedback:** feed back only observed findings (the lab repair prompt) — not the whole history.

**Anti-patterns:** vague instructions ("make it good"), context dumping, accepting the first draft, no format contract, mixing multiple tasks in one prompt.

**Exam skeleton:**
1. Structure prompts: role, decomposed task, minimal authoritative context, explicit constraints, exact output format.
2. Few-shot examples exploit attention; chain-of-thought for reasoning-heavy tasks.
3. Constraints (forbidden lists) are hallucination guardrails.
4. Feedback prompts should carry only sensor findings (lab pattern).
5. Anti-patterns: vagueness, context dumps, multi-task prompts, no schema.

## 2.7 Agentic workflows

**Definition:** an LLM-driven loop in which the model *plans, acts through tools, observes results, and iterates* toward a goal — as opposed to single-shot prompting.

**Loop (memorize): Goal → Plan → Act (tools) → Observe → Reflect/Verify → repeat until verified or budget exhausted.**
**Components:** LLM core (reasoning); **tools** (file read/write, shell, test runner, web search, linters — the sensor.js of the lab); **planning** (task decomposition, ordering); **memory** (short-term context, long-term stores/scratchpads); **feedback/verification** (tests, sensors, human checkpoints); **guardrails** (permissions, attempt bounds, allow-lists).

**Multi-agent patterns:** orchestrator–worker; generator–critic; reviewer agents. Note each agent adds non-determinism and error-compounding risk.

**Risks of granting autonomous agents unrestricted access (memorize 6):**
1. Destructive actions (deleting files, force-pushes, dropping tables).
2. Security exposure (secrets in context, installing malicious/hallucinated dependencies, shell injection via crafted content — prompt injection through repo files or web results).
3. Infinite/expensive loops (cost runaway).
4. Silent drift from requirements (confidently solving the wrong problem).
5. Data exfiltration (code/IP sent to external APIs).
6. Compounding errors across agent hand-offs.

**Controls:** least-privilege tool access; sandboxing; allow-listed commands; attempt bounds (lab: 2); human approval for irreversible actions; full audit logs; deterministic sensors as the agent's "reality check".

**Exam skeleton:**
1. Agent = LLM + tools + planning + memory + feedback loop (Goal→Plan→Act→Observe→Verify).
2. Tools give agency; verification closes the loop; sensors must be deterministic.
3. Unrestricted access risks: destruction, injection, runaway cost, drift, exfiltration, error compounding.
4. Controls: least privilege, sandbox, bounds on attempts, HITL for irreversible steps, audit trail.
5. Lab mapping: generation task = plan; sensor = feedback; repair prompt = bounded re-plan; attempt log = audit.

---

# MODULE 3 — REQUIREMENTS AND DESIGN

## 3.1 Requirements engineering baseline: elicitation, synthesis, management (from CS9)

**Elicitation** = discovering, extracting and uncovering stakeholder needs, system constraints and business goals. Agile techniques (memorize 3): **User story mapping** (visual journeys → MVP → release slices), **Sprint reviews & demos** (feedback on working increments), **Spikes & prototyping** (short investigations to de-risk ambiguity).

**Synthesis** = converting raw elicitation data into structured, actionable artifacts. Traditional: BRD/SRS (hundreds of pages). Agile: **User stories** ("As a [role], I want [action], so that [benefit]"), **Acceptance criteria** (BDD: "Given [context], When [action], Then [outcome]"), **Themes & epics** decomposed into stories.

**Management** = tracking, prioritizing, controlling change: prioritization by value/dependency/risk, **MoSCoW** (Must/Should/Could/Won't), **Product Backlog** owned and continuously re-ordered by the Product Owner (fixed time/capacity, scope flexes).

**Exam skeleton:**
1. Elicitation = discovering needs; techniques: story mapping, reviews/demos, spikes.
2. Synthesis = raw input → structured artifacts: stories, BDD acceptance criteria, epics.
3. Management = prioritization (MoSCoW) + backlog control by PO.
4. AI touches all three (next section) but must not replace human approval.

## 3.2 AI-driven requirements elicitation (CS9 slide 6 — three capabilities)

AI transforms elicitation from a manual discovery exercise into a **continuous, data-driven engine** that surfaces explicit *and implicit* needs:

1. **Implicit need & telemetry analysis** — AI ingests unstructured data (support tickets, logs, app-store reviews, transcripts) and mines patterns users never stated ("users always retry the failed payment twice" → implicit reliability requirement).
2. **Synthetic user personas & autonomous interviewing** — AI simulates persona populations and can auto-interview (chatbots) to explore need spaces cheaply and at scale.
3. **Instant functional prototyping** — AI generates throwaway prototypes within hours to *validate assumptions with tangible software* instead of predicting all needs up front.

**Exam skeleton:** three capabilities (telemetry mining, synthetic personas/interviewing, instant prototyping); benefit = continuous discovery of explicit + implicit needs; risk = inferred needs are statistical, not confirmed — human validation still required.

## 3.3 AI-driven requirements synthesis and management (CS9 slides 7–8)

**Synthesis capabilities:**
1. **Automated artifact conversion** — transcripts/tickets → user stories, acceptance criteria, SRS drafts.
2. **Real-time ambiguity & quality scoring** — models pre-trained on standards (INCOSE guidelines, **EARS syntax**: Easy Approach to Requirements Syntax — "When [trigger], the system shall [response]") score each requirement for ambiguity, testability, and standard compliance.
3. **Live bidirectional AI traceability** — every requirement links to code/tests and back; gaps (requirement without test, code without requirement) are flagged continuously.

**Management capabilities:** proactive **drift & scope-creep monitoring** — AI watches backlog vs delivered behaviour and raises divergence alerts.

**Human boundary (memorize the sentence):** strategic trade-offs, legal accountability and stakeholder relationships still require human BAs and Product Owners to **review and approve AI-generated artifacts before sign-off**.

**Exam skeleton:** artifact conversion; ambiguity/quality scoring vs INCOSE/EARS; bidirectional traceability; drift/scope-creep monitoring; humans approve before sign-off.

## 3.4 Risks in AI-assisted requirements engineering + guardrails

**Risks (memorize 4):**
1. **Omissions** — requirements silently missing because the source transcript never mentioned them or AI missed implicit needs; downstream they surface as defects.
2. **Hallucinated requirements** — AI invents plausible requirements/stakeholders/features that were never asked for (scope pollution).
3. **Data leaks** — feeding confidential transcripts, customer PII, or pricing data into external LLM APIs; retention/training-use risk.
4. **Compliance issues** — generated requirements may breach regulatory constraints (GDPR/DPDP-style privacy, sector rules) because the model has no knowledge of the org's obligations.

**Guardrails (memorize the pair):**
- **RAG (Retrieval-Augmented Generation):** ground generation in an authoritative, curated corpus (contracts, past SRS, compliance rules) so outputs cite retrievable sources instead of inventing; enables access control on what the model can see.
- **HITL sign-off:** mandatory BA/PO review checklist — completeness check against source, ambiguity score review, compliance checklist, stakeholder confirmation — before a requirement enters the backlog; approval is recorded (audit trail).

**Exam skeleton:** 4 risks (omissions, hallucinations, leaks, compliance); guardrail pair = RAG grounding + HITL sign-off with recorded approvals; note RAG reduces but does not eliminate hallucination — hence the human gate.

## 3.5 Spec-Driven Development (SDD) and the Specification Spectrum

**Core sentence (CS9):** *The spec declares intent; the code realizes it.* In SDD, specifications live in version control alongside/above the code. When business needs change, developers **update the specification**, and AI refactors the codebase and regenerates tests to match the new spec baseline. Elicitation feeds directly into **bounded guardrails**: the spec explicitly defines what the feature *must not do*, preventing agents from adding unnecessary dependencies.

**The Specification Spectrum (memorize all three + one-line differences):**
1. **Spec-first** — spec written before coding to guide initial implementation; afterwards spec may or may not be maintained. Value = initial clarity.
2. **Spec-anchored** — spec maintained *alongside* code for the whole lifecycle; behaviour changes require updating **both** spec and code, kept synchronized.
3. **Spec-as-source** — spec is the **only artifact humans edit**; code is fully generated and never manually modified; any behaviour change = change the spec and regenerate. (This is the "disposable code" end — see Sample Q1 / QBank E4.)

**Trade-off chain:** control & safety (spec-first) → consistency & traceability (spec-anchored) → maximum automation but total dependence on spec completeness + verification strength (spec-as-source).

**Exam skeleton:** spec = intent, code = realization; specs live in version control; change the spec, regenerate code + tests; spectrum = spec-first (clarity) / spec-anchored (synchronized) / spec-as-source (humans edit only specs); must-not-do sections bound the agents.

## 3.6 GitHub Spec Kit (know the command table)

| Command | Purpose | When |
|---|---|---|
| `/speckit.constitution` | Define project-wide principles | Project start |
| `/speckit.specify` | Create feature specification | Before technical design |
| `/speckit.clarify` | Identify and resolve ambiguities | After initial spec |
| `/speckit.plan` | Create technical plan | After requirements clear |
| `/speckit.checklist` | Generate quality checklists | Before implementation |
| `/speckit.tasks` | Create actionable tasks | After planning |
| `/speckit.analyze` | Check artifact consistency | Before implementation / after big changes |
| `/speckit.implement` | Execute implementation tasks | During development |
| `/speckit.converge` | Identify remaining gaps vs artifacts | During/after implementation |

Order mnemonic: **Constitution → Specify → Clarify → Plan → Checklist → Tasks → Analyze → Implement → Converge** ("CCP CCT AIC").

## 3.7 SDD risks and mitigations (CS9 slide 14 — reproduction table)

| Risk | Root cause | Mitigation |
|---|---|---|
| Architectural bloat | LLM tendency toward maximal complexity | Simplicity rules in constitution.md (e.g., "prefer single relational tables over new microservices") |
| Cascading errors | Unchecked transition from /plan to /tasks | Treat plan.md as an explicit PR gate: mandatory senior human review before /tasks |
| Spec-to-code drift | Unlinked code changes during /implement | CI checks fail if code commits touch architecture without updated specs; /speckit validation |
| Legacy context loss | LLM context limitations | Include explicit reference files (schema dumps, OpenAPI contracts) in .specify/ context during /plan |
| Dependency & API hallucinations | Model invents packages/endpoints | Lockfiles, SCA scanning, contract-first design, allow-listed dependencies |
| Shallow edge-case / NFR coverage | Specs focus on happy path | Non-functional requirement checklists; adversarial test generation |

Also know the four **risk-control activities across the SDD lifecycle** (CS9 slide 17): (1) requirements validation, (2) architecture risk analysis, (3) security & threat modeling, (4) task decomposition & dependency validation.

**Exam skeleton:** reproduce the 4-row table (bloat/cascading/drift/context-loss) + name dependency hallucination and shallow NFR coverage; close with the 4 risk-control activities.

## 3.8 Architecture Decision Records (ADRs)

**Definitions (exact):**
- **Architectural Decision (AD):** a justified design choice addressing a functional or non-functional requirement that is architecturally significant.
- **Architecturally Significant Requirement (ASR):** a requirement with a measurable effect on the architecture and quality of the system.
- **ADR:** captures a single AD and its rationale — the reasons, trade-offs and consequences.

**Standard ADR structure:** Title/number, Status (proposed/accepted/superseded), Context (problem + forces), Decision, Consequences (positive/negative/risks). Optionally Options considered + Alternatives.

**Why ADRs matter more in the AI era (5 benefits — CS9 slide 16):** preserve decision history; improve team communication; support maintenance (future developers — *and future AI agents* — understand why choices were made); avoid repeated discussions; manage change (reference when revisiting a decision). AI angle: ADRs are machine-readable organizational memory — feed them as context so agents stop re-proposing settled architecture (adrkit extension drafts ADRs from plans, retrieves existing decisions, checks plans against them).

**Validating AI-generated architecture before implementation (QBank B3 — memorize 6 steps):**
1. Requirements validation — traceability from spec to design; ASRs covered.
2. Architecture risk analysis — scalability, performance, failure modes vs expected load.
3. Security & threat modeling (§3.9) — trust boundaries, STRIDE per element.
4. ADR review — every significant choice recorded with alternatives + consequences; senior architect approval (HITL).
5. Architecture prototyping — spike/PoC of riskiest interactions; measure, don't assume.
6. Consistency checks — /speckit.analyze or CI checks; constitution/simplicity rules enforced.

**Exam skeleton:** 3 definitions (AD, ASR, ADR); ADR structure; 5 benefits; AI-era role = context for agents (adrkit); validation = requirements→risk→threat→ADR review→prototype→consistency.

## 3.9 Threat modeling for AI-generated designs

**Definition:** structured identification of threats/vulnerabilities per element of a design before implementation. Do it *after* AI proposes architecture and *before* code generation.

**Method (STRIDE per trust boundary):**
- **S**poofing (fake identity) → authentication
- **T**ampering (data/code modified) → integrity (signing, checksums)
- **R**epudiation (deny action) → audit logging
- **I**nformation disclosure → encryption, access control
- **D**enial of service → rate limiting, capacity design
- **E**levation of privilege → authorization, least privilege

**AI-specific additions:** prompt injection through data the AI reads; AI-generated code embedding secrets; hallucinated dependencies as supply-chain attack surface; overly permissive default AI designs. Deliverable: trust-boundary diagram (text arrows), STRIDE table per boundary, mitigations mapped into ADRs and tests.

**Exam skeleton:** threat modeling = pre-implementation, structured (STRIDE) analysis at trust boundaries; output feeds ADRs + tests; AI-specific threats = injection via data, secrets in code, hallucinated deps; shift-left = do it at design time.

---

# MODULE 4 — CODING AND REFACTORING

## 4.1 Developer–AI interaction patterns: pair programming vs autonomous agents

**AI pair programming:** developer drives; AI suggests inline (completions) or via chat; human accepts/rejects/edits each suggestion in real time. Human remains the decision-maker per change; tight feedback loop; context = open files + chat.

**Autonomous coding agent:** human issues a *task* ("implement POST /items per openapi.yaml"); agent plans, edits multiple files, runs tools/tests, iterates; human reviews the resulting PR, not each keystroke. Control shifts from per-line to per-task (gates, bounds, audit).

| Dimension | AI pair programming | Autonomous coding agent |
|---|---|---|
| Initiative | Human drives, AI suggests | Agent plans and executes a task |
| Control point | Every accepted suggestion | Task definition + PR review + gates |
| Best for | Implementation inside known design, boilerplate, exploration | Well-specified bounded tasks: migrations, test generation, scaffolding |
| Risk profile | Low per change; slow scaling; subtle over-reliance | High autonomy risk: compounding errors, destructive actions |
| Verification | Human reads each diff | Deterministic sensors + tests + human PR review |
| Failure mode | Bad suggestion accepted unnoticed | Wrong solution delivered confidently |
| Cost per task | Low | Higher (tokens, retries) but amortized over larger tasks |

**Recommendation logic (memorize):** use **pair programming** when design context lives in the developer's head and changes are small/frequent; use **agents** when tasks are well-specified, verifiable by deterministic checks, and batch-like (tests, refactors, migrations, scaffolding). Any agent use requires: bounded attempts, least-privilege tools, deterministic gates, HITL on merge.

**Exam skeleton:** definitions; control point shift (line → task/gate); comparison table (initiative, control, best-for, risk, verification); recommendation rule; agent guardrails.

## 4.2 AI for specific programming tasks

- **Code generation** — from spec/prompt to implementation (works best contract-first: OpenAPI → handler code). Risk: hallucinated APIs, missing edge cases.
- **Code completion** — inline next-line/next-block suggestions; fastest interaction loop; risk: thoughtless acceptance ("AI says yes" bias), drift from intent.
- **Bug detection & debugging** — AI explains stack traces, hypothesizes root causes, proposes fixes; pairs well with deterministic reproduction (failing test first). Risk: plausible-but-wrong diagnosis; always require a reproducing test before accepting a fix.
- **Code documentation** — generate docstrings/READMEs from code; keep human review for accuracy of *intent* descriptions; docs drift → regenerate on spec/code change.
- **Versioning & dependency management (AI-assisted)** — semantic versioning discipline, lockfiles, pinning; AI helps upgrade paths and changelogs; SCA tools + AI analyze vulnerable/transitive dependencies; never let agents unpinned-add packages (supply chain).

**Exam skeleton:** name the 5 task families; one strength + one risk each; the recurring mitigation: contract-first inputs, exact tests, human review of intent.

## 4.3 Code smells and AI-assisted refactoring

**Definition:** a code smell is a surface symptom in code that usually indicates a deeper design problem (Fowler). Not a bug — a maintainability risk.

**Common smells (memorize 8 + one-line each):** Long Method; Large Class (God class); Duplicate Code; Long Parameter List; Feature Envy (method obsessed with another class's data); Primitive Obsession (strings/ints where domain types belong); Shotgun Surgery (one change → many files); Magic Numbers/Strings.

**AI workflow for smell detection & refactoring:** scan → rank smells by risk → propose refactorings (extract method/class, rename, introduce parameter object) → **verify with tests before and after** (behaviour preserved) → bounded iteration → human review. For legacy code with no tests: write **characterization tests** first (pin current behaviour), then refactor.

**Exam skeleton:** definition; 8 smells with one-liners; AI refactoring loop = detect→rank→propose→verify→review; no-tests legacy → characterization tests first.

## 4.4 Legacy code modernization (monolith → microservices, safe migration)

**Goal:** decompose a legacy monolith into services using AI agents *without breaking production*. The safe migration workflow (QBank D6/F1 — memorize 8 stages):
1. **Inventory & dependency identification** — map modules, data coupling, shared databases; AI builds the call graph; identify seams (natural service boundaries).
2. **Characterization tests** — pin current behaviour (golden master tests) before touching anything.
3. **Extract bounded contexts** — pick the lowest-risk, highest-value seam first (strangler-fig: new service grows around the monolith; route traffic gradually).
4. **Contract-first interfaces** — define OpenAPI contracts per service; API compatibility = versioned, backward-compatible contracts; contract tests in CI.
5. **AI-assisted code generation under guardrails** — feed the contract + style guide + harness rules (never the whole monolith); forbid new dependencies, forbid secrets.
6. **Deterministic verification** — static sensors + exact-assertion tests + contract tests + integration tests per extraction; parity checks (old vs new output).
7. **Version & dependency management** — pin, lockfiles, canary rollout, rollback plan per step.
8. **Human approval gates** — senior review at each extraction; audit log per step; keep reverse path (feature-flagged routing back to monolith).

**API compatibility rules:** don't break consumers — additive changes only; version endpoints; deprecate with notice; contract tests catch accidental breaks.

**Exam skeleton:** inventory/AI dependency map → characterization tests → strangler-fig seam extraction → contract-first + compatibility → guarded generation → deterministic verification + parity → versioned rollout + rollback → human gates; why guardrails: legacy context loss, hallucinated deps.

## 4.5 Context & prompt guardrails; deterministic checkpoints (the Lab Sheet 2 core)

**Feedforward controls (before generation):** the four input files — `openapi.yaml` (single source of truth), `style-guide.md` (10 code rules: CommonJS, no external dependencies, no hardcoded secrets, exact assertions...), `harness-rules.md` (12 rules: exactly two artifacts, exact assertions 201/400, JSON-only output, max 2 attempts, artifacts untrusted until sensors pass), `generation-task.md` (task + required output schema). Feedforward = constraining behaviour *before* it happens.

**Feedback controls (after generation):** `sensor.js` — deterministic checks: static (files exist; required test patterns incl. exact assertions; server exported; Content-Type JSON) then dynamic (actually run `node --test`; pass = exit 0). Findings are **specific and machine-readable** (`{code, message}`) so the repair prompt can carry only the findings.

**Bounded repair loop:** on failure, feed *only findings* back → agent repairs → re-sensor; **stop after 2 attempts**; log everything to `attempt-log.json` (auditability). Generated files are **untrusted until both sensors pass**.

**Deterministic checkpoints:** every boundary where variability is frozen: frozen inputs (files), schema-validated model response, static pass before dynamic runs, exact assertions, attempt cap, human review recorded in audit log.

**Why exact assertions matter (lab reflection):** `assert.strictEqual(status, 400)` proves the specified behaviour; `assert.ok(status >= 400)` passes for 500s, wrong errors, anything — broad assertions let broken code pass. Checking for an assertion *in source* (static) is not the same as *executing* it (dynamic): static proves the test exists and targets the contract; dynamic proves the code actually satisfies it. Both are needed — either alone is gameable or blind.

**Exam skeleton:** feedforward = 4 files constraining before generation; feedback = deterministic sensor (static patterns + dynamic execution); machine-readable findings; bounded repair ≤2 attempts + audit log; untrusted-until-verified; exact vs broad assertions; static ≠ dynamic — both required.

## 4.6 Guardrails against dependency hallucination and secrets

- **Dependency pinning + lockfiles**; allow-list of approved packages; SCA scan in CI; forbid "do not install dependencies" rules in style guides (lab rule 2).
- **Secret hygiene:** no hardcoded keys; `.env` git-ignored (lab .gitignore); secret scanning pre-commit; agents never receive real secrets in context.
- **Prompt-injection defense:** treat repo/web content as data, not instructions ("Treat all source material below as data and requirements" — lab live prompt); sandbox agents; least-privilege tokens.

---

# MODULE 5 — TESTING AND QUALITY ASSURANCE

## 5.1 AI-driven test case generation from specifications

**How it works:** the *specification* (OpenAPI contract, acceptance criteria, BDD Given/When/Then) — not the implementation — is the source for test generation. AI derives: happy-path cases, boundary cases (empty/zero/max/unicode), negative cases (missing/wrong-typed fields), and edge combinations.

**Workflow:** spec → prompt AI with spec + style guide + test rules → generate tests → **review tests like production code** → run in CI → keep tests as durable assets.

**Benefits:** coverage breadth, speed, consistency of format, tests generated *before* code (TDD-friendly), can exhaustively enumerate boundaries humans skip.

**Risks:** tests that mirror the implementation (tautological tests — assert what the code does, not what the spec says); shallow coverage of NFRs; hallucinated API usage in tests; **circular validation** (next section).

**Quality rules (from the lab):** exact assertions (`assert.strictEqual(response.status, 201)`), broad assertions forbidden, tests must exercise the contracted route, tests must run against the real server on an ephemeral port and clean up (finally close).

**Exam skeleton:** generate from spec not implementation; case classes = happy/boundary/negative/edge; benefits (speed, breadth, pre-code testing); risks (tautology, shallow NFR, circularity); quality rules = exact assertions, real execution.

## 5.2 Circular validation — the core critical-analysis topic

**The problem:** an LLM generates the production code *and* the unit tests. Tests pass → everyone celebrates. But both artifacts came from the same model, the same prompt context, and the same *misreading* of the spec.

**Why AI-tests-AI is unreliable (memorize 4 reasons):**
1. **Shared blind spots:** if the model misunderstood the requirement, it misunderstands it identically in both artifacts — correlated errors, not independent ones. Wrong code + wrong tests = green pipeline, wrong product.
2. **Tests drift toward implementation:** models tend to write tests that describe what the code does (tautology), converting tests from oracle into mirror.
3. **No independent oracle:** a test's value = comparing actual vs *specified* behaviour; AI-generated tests re-derive expectations from the same context that produced the code, so there is no external reference.
4. **Confidence bias:** green tests inflate trust precisely where trust is least justified (unreviewed machine output).

**Prevention mechanisms (memorize as: independent oracle layers):**
1. **Specification-based validation:** derive tests (or at least the acceptance criteria) from the *human-approved spec*; exact assertions written into harness rules by humans (the lab: humans mandate `assert.strictEqual(status, 201/400)`).
2. **Independent test oracles:** golden-master/characterization tests from the legacy system; property-based tests (invariants: idempotence, monotonicity); contract tests generated independently of implementation; mutation testing — inject mutants, tests must kill them (tests that can't fail are worthless).
3. **Structural independence:** validation rule engine checks test code is *structurally sound and independent* of production code (lab experiment 3) — e.g., tests generated from spec artifacts, not from reading the implementation.
4. **Integration/system testing across boundaries:** real HTTP calls, ephemeral server, end-to-end flows — different level, different failure modes.
5. **Human checkpoints:** human-authored acceptance criteria; human review of test intent (not just green); human approves merge.
6. **Separate generation contexts:** generate tests in a different session with *only the spec* as input (code never shown) — context separation breaks correlation.

**Exam skeleton:** define circular validation; 4 failure reasons (shared blind spots, tautology, no oracle, confidence bias); prevention = spec-derived tests + independent oracles (golden master, property, contract, mutation) + structural independence + integration level + human checkpoints + separate contexts; closing line: "tests are only evidence if the tester is independent of the accused."

## 5.3 Automated code review (AI-assisted)

**Workflow:** PR opened → AI reviewer analyzes diff + context → findings categorized (smells, security vulnerabilities, logic defects, style) → posted as review comments → human reviewer adjudicates → CI gates still apply.

**What AI catches well:** code smells, missing input validation, obvious injection vectors (SQLi/XSS patterns), hardcoded secrets, error-handling gaps, inconsistent naming, missing tests for changed paths.

**False-positive prevention (QBank B7):** ground review in repo conventions (constitution/style guide as context); require explanation + concrete fix per finding; confidence thresholds; learning loop — humans mark findings valid/invalid, templates are tuned; route uncertain findings to "question" not "defect". **Review independence:** the AI that *generated* code must not be the sole reviewer of it — use a different model/context or human review; AI review complements, never replaces, human accountability (the approving human owns the merge).

**Exam skeleton:** workflow steps; strengths (smells/security/validation gaps); false-positive controls (grounding, explanation required, thresholds, feedback loop); independence rule (generator ≠ reviewer; human owns merge).

## 5.4 Predictive bug detection

**Concept:** mine history (defect datasets, churn hotspots, complexity metrics, commit patterns) to score which files/changes are likely defective, so review and testing concentrate there. AI improves it: semantic change analysis, historical defect correlation, "files touched by AI-generated churn get elevated risk scores".

**Uses:** targeted test selection; review routing (risky PRs → senior humans); early warnings in CI.

**Limits:** biased by past practices; spurious correlations; must feed *causes* (churn from AI rewrites) not just locations.

**Exam skeleton:** definition (risk-score code from history + semantics); uses (targeted testing, review routing); caution (bias, feedback loops — predicted-risky files get more review, find more defects, look riskier).

## 5.5 QA strategies and human checkpoints

**Layered QA (test pyramid, AI-augmented):** unit (AI-generated, exact assertions) → integration/contract (spec-derived, independent) → E2E (few, human-critical flows) → production (canary analysis, telemetry, self-healing).

**Human checkpoints (memorize the 4 placements):** (1) acceptance criteria approval before generation; (2) test-intent review after generation; (3) merge approval with AI findings adjudicated; (4) release approval for high-risk changes (payments, auth, migrations). Each checkpoint has recorded sign-off (accountability, Module 7).

**Exam skeleton:** pyramid with AI at unit/contract layers; humans at 4 checkpoints with recorded sign-off; gates block, humans approve — automation proposes, humans dispose.

## 5.6 Self-healing automation (dev + ops view)

**In testing:** auto-fix flaky tests (retries with quarantine, selector repair for UI tests), auto-repair broken builds via bounded agent loops with deterministic verification (the harness pattern generalized to CI).
**In operations:** anomaly detection → diagnosis → runbook execution (restart, scale, failover, rollback) → verify recovery → record. Guardrails: only *pre-approved* runbooks automated; irreversible actions need human approval; all actions logged (audit).

**Exam skeleton:** definition (detect→diagnose→repair→verify→record); dev-side (flaky tests, builds) vs ops-side (runbooks, rollbacks); guardrails = pre-approved runbooks only, HITL for irreversible, full audit.

---

# MODULE 6 — DEPLOYMENT, CI/CD AND RELEASE MANAGEMENT

## 6.1 AI-generated artifacts and declarative code

**Declarative code** states *desired end-state* (Kubernetes YAML, Terraform, CloudFormation) rather than imperative steps — ideal for AI generation because intent is explicit, output is verifiable against schema/plan, and diffs are reviewable.

**Benefits:** IaC generated from natural-language intent + validated by `plan`/dry-run; consistency via modules; reviewable PRs of infrastructure.

**Risks:** AI-generated IaC can expose resources publicly (wrong security groups), hallucinate providers/resources, hardcode secrets. Mitigations: schema + policy-as-code (OPA-style) gates, plan review by human, secret scanning, least-privilege defaults.

**Exam skeleton:** declarative = intent-stating, machine-verifiable; AI generates IaC well because plan/dry-run gives deterministic feedback; risks = misconfig exposure, hallucinated resources; gates = schema, policy-as-code, human plan review.

## 6.2 AI-driven dependency resolution and security analysis

- **AI-driven dependency resolution:** choose compatible versions, propose upgrade paths, resolve conflicts — but always through lockfiles + registry allow-lists + SCA (software composition analysis).
- **Automated security patch analysis:** AI reads CVE advisories + your usage, judges exploitability *in context* (do you call the vulnerable function?), prioritizes patches, drafts upgrade PRs.
- **Predictive vulnerability scanning:** learns from history/patterns to flag likely-vulnerable new code before scan signatures exist.
**Guardrail:** all dependency changes via versioned, reviewed, scanned PRs; agents never auto-apply unpinned updates.

**Exam skeleton:** three capabilities (resolution, patch analysis, predictive scanning); AI adds context-judgment on top of SCA; controls = allow-lists, lockfiles, reviewed PRs, no auto-unpinned updates.

## 6.3 CI/CD pipeline optimization: smart test selection and impact analysis

**Problem AI solves:** full suites are slow; under AI velocity, running everything on every commit is impossible.
**Test impact analysis (TIA):** map code changes → affected tests via dependency graph + historical pass/fail correlation + semantic similarity → run the minimal high-value subset per commit; full suite nightly/before release.
**Smart selection inputs:** changed files, call graph, code-coverage mapping, defect history (predictive detection), flakiness scores.
**Pipeline flow (see Part B F5):** commit → static analysis → schema/lint gates → selected tests → AI security analysis → build → canary → full suite pre-release.
**Risk:** selection misses → escape defects; mitigation: periodic full runs, coverage guard (selected tests must cover changed lines), escape-rate monitoring.

**Exam skeleton:** definition of TIA + smart selection inputs; pipeline placement; risk = missed coverage → guard rails (coverage check, nightly full suite, escape-rate tracking).

## 6.4 Automated release management and deployment strategies

**Strategies (one-liners):** blue-green (two environments, instant switch, fast rollback, double cost); canary (route small % to new version, analyze metrics, ramp or abort); feature flags (decouple deploy from release; kill-switch); rolling (incremental pod replacement).
**AI in release management:** release-note drafting; risk scoring of the release (change size, components touched, recent incident correlation); canary metric analysis (statistical comparison vs baseline); go/no-go recommendations — **recommendation only, human approves releases for high-risk products**.
**Automated self-healing & rollbacks:** define SLO-based triggers (error rate, latency, saturation) → automatic abort/rollback on breach → incident record → feed post-mortem into specs/tests (Module 8 loop). Rule: rollback is cheaper than debugging in production; automate rollback, never auto-forward-fix.

**Exam skeleton:** 4 strategies with trade-offs; AI roles (risk scoring, canary analysis, notes) with human approval boundary; rollback triggers from SLOs; automate rollback, not forward-fixes.

## 6.5 What can be autonomous vs what needs approval (the Module 6/7 boundary table)

| Operation | Autonomy | Reason |
|---|---|---|
| Lint, schema validation, static analysis | Fully autonomous | Deterministic, reversible, low risk |
| Smart test selection & run | Autonomous | Accelerates, no production effect |
| Dependency minor patch with green tests + SCA | Autonomous merge possible | Low risk, auditable |
| Dependency major upgrade | Human approval | Compatibility risk |
| Canary deploy to internal env | Autonomous | Isolated |
| Production canary ramp | Autonomous *within* policy (abort on SLO breach) | Guarded by automatic rollback |
| Full production rollout | Human approval | Blast radius |
| Schema/data migrations | Human approval (always) | Irreversible |
| Rollback on SLO breach | Autonomous (pre-authorized) | Recovery speed > review speed |
| Security patch to production | Fast-track human approval | Urgency but accountability needed |

**Memory hook:** *"Autonomous for reversible and verifiable; approval for irreversible, external-facing, and production data."*

---

# MODULE 7 — GOVERNANCE, CONTROL POINTS AND HUMAN-IN-THE-LOOP

## 7.1 Autonomous limits and shift-left policy enforcement

**Autonomous limits:** explicit, written boundaries on what agents may do without a human: max repair attempts; allow-listed tools/commands; forbidden operations (production data, secrets, irreversible infra); spend caps. Limits are *configuration, not culture* — enforced by the harness/platform, not by hope.

**Shift-left policy enforcement:** move quality/security policy checks from release time to commit/design time: pre-commit secret scans; lint/schema at commit; policy-as-code in CI; threat modeling at design; dependency scanning at PR. Rationale: under AI velocity, end-of-pipeline inspection is too late — defects replicate faster than reviews.

**Exam skeleton:** define autonomous limits (attempt bounds, allow-lists, forbidden ops, spend caps, platform-enforced); shift-left = policy checked at commit/design time; why: velocity outpaces end-stage review.

## 7.2 Dynamic quality gates and non-deterministic validation

**Quality gate:** an automated checkpoint that blocks progression until criteria pass. **Dynamic** = criteria adapt to change risk: bigger/riskier change → stricter gate set (full suite + security scan + senior review); trivial docs change → minimal gate.
**Non-deterministic validation problem:** AI outputs vary per run, so gates must evaluate the *artifact deterministically* (schema, static, tests) — never "ask another LLM if the code looks right" as the sole judge. Where AI judgment is unavoidable (review comments, canary analysis), it is advisory: a *deterministic* condition still decides pass/fail.
**Gate stack (memorize 6):** schema validation → static analysis/security → unit+integration tests (exact assertions) → coverage/diff guards → canary/SLO checks → human approval for high-risk classes.

**Exam skeleton:** gate = blocking checkpoint; dynamic = risk-adaptive criteria; rule: deterministic gates for non-deterministic outputs (AI judgment only advisory); 6-stage gate stack.

## 7.3 Human-in-the-loop decision points across the lifecycle

| Phase | HITL decision | Why human |
|---|---|---|
| Requirements | Sign-off of AI-synthesized requirements | Business accountability, legal liability |
| Design | ADR/architecture approval; threat-model review | Trade-offs are strategic, not statistical |
| Coding | Merge approval (PR) | Ownership of what ships |
| Testing | Test-intent review; acceptance criteria ownership | Prevents circular validation |
| Release | Go/no-go for high-risk changes | Blast radius, compliance |
| Operations | Post-incident learning approval | Prevents wrong learning signals |

**Design principles for HITL:** checkpoint at *irreversible/expensive-to-reverse* steps only (too many gates → rubber-stamping); give humans evidence (sensor reports, diffs, scores) not just verdicts; record sign-offs (who/what/when — auditability); escalation path when agents hit bounds.

**Exam skeleton:** table of 6 phase checkpoints; principles: gate irreversibility, evidence-based approval, recorded sign-off, escalation on bounds; note over-gating causes rubber-stamp decay.

## 7.4 Ownership, accountability and liability

**Core principle (memorize verbatim-ish):** *AI is a tool, not a legal person. Accountability always resides with the humans and the organization that deploy it.* "The agent did it" is not a defence; the developer who merges, the approver who signs off, the org that configured the agent own the outcome.

**Accountability model (QBank E2):** RACI over AI-augmented work — **Author/agent-operator** (developer deploying the agent) owns the change; **Reviewer** owns catching what sensors miss; **Approver** (senior/lead) owns release for high-risk classes; **Platform/governance team** owns harness, gates, policies; **Organization** owns liability to customers/regulators.
**Mechanisms that make accountability real:** recorded HITL sign-offs; audit logs (prompts, outputs, sensor reports, attempt counts — the lab's attempt-log.json); traceability from requirement→commit→release; incident post-mortems referencing the audit trail; clear policy: what each autonomy level means and who approved its limits.
**Liability:** contracts and regulation attach to the org (data protection, sector compliance); insurance/indemnity for AI tool vendors does not transfer developer accountability; negligence = deploying AI output without the mandated gates.

**Exam skeleton:** principle (tool not person); RACI mapping; evidence mechanisms (audit logs, sign-offs, traceability); liability sits with deploying organization; negligence = bypassing gates.

---

# MODULE 8 — OPERATIONAL FEEDBACK LOOPS AND LIFECYCLE HEALTH

## 8.1 DevOps-to-Dev feedback loop

**Definition:** a closed loop that carries production knowledge (incidents, telemetry, support tickets, performance data) back into development so specs, code, tests and AI prompts improve continuously.

**Flow (memorize):** Detect (monitoring/SLO breach) → Triage (incident) → Resolve (rollback/hotfix) → Post-mortem (blameless) → Extract learnings (new edge cases, spec gaps, test gaps) → Inject into dev assets (spec "must-not" sections, regression tests, guardrail rules, AI context) → Verify (the fix and the loop itself).

**Why it matters more with AI:** AI velocity amplifies whatever the loop feeds back — if incidents teach the system, defects fall; if incidents are ignored, AI reproduces the same mistakes faster.

**Exam skeleton:** definition; 6-step flow; AI angle = feedback quality determines whether velocity amplifies learning or error; close the loop into *specs and tests*, not just tickets.

## 8.2 Feeding telemetry back to AI tools — safely

**Useful feedback channels (memorize 4):** (1) incident summaries → RAG corpus for coding agents ("known failure modes"); (2) production error patterns → regression test generation; (3) hotspots → predictive risk scores for review routing; (4) usage telemetry → requirements discovery (Module 3 link).

**Without compromising security (memorize 5 controls):** sanitize/scrub PII and secrets before telemetry enters any AI context; aggregate/anonymize rather than raw dumps; access-controlled retrieval (RAG with permissions); keep sensitive data in on-prem/private endpoints; audit what entered which context.
**Avoiding wrong learning signals:** don't feed unreviewed post-mortems as facts (hypotheses ≠ causes); separate correlation from cause (hot file ≠ bad file — it may just be busy); human curator approves what enters the knowledge base; version the knowledge corpus so bad lessons can be rolled back; measure whether feedback improved outcomes (escape rate trend).

**Exam skeleton:** 4 channels; 5 security controls (sanitize, aggregate, RAG permissions, private endpoints, audit); wrong-signal risks (unreviewed post-mortems, correlation-as-cause, no versioning) + curator + measure improvement.

## 8.3 DORA metrics and how AI breaks them (CS14 core table)

**The four (Accelerate, 2018) + assumptions:**

| Metric | Measures | Assumption |
|---|---|---|
| Deployment Frequency | How often code ships | More deploys = healthier team |
| Lead Time for Changes | Commit → production | Shorter = better flow |
| Mean Time to Recovery (MTTR) | Incident resolution speed | Faster = more resilient |
| Change Failure Rate (CFR) | % deploys causing failure | Lower = higher quality |

Plus availability/focus; **2025 DORA report adds a fifth: rework rate** (unplanned fixes pushed to production — the blind spot that matters when AI writes much of the code).

**How AI breaks each (memorize the pairing):**
- **Deployment Frequency** — inflated by AI boilerplate/scaffolding/micro-commits → pair with **TrueThroughput** (complexity-adjusted) and **AI Code Share**.
- **Lead Time** — bottleneck shifted to review queues ("verification tax") → pair with **PR Cycle Time / review turnaround**.
- **Change Failure Rate** — misses *hidden churn* (rewrites before production) → pair with **Code Durability / Turnover / PR Revert Rate**.
- **MTTR** — **holds up well**: measures team response regardless of who wrote the code; keep it, pair with adoption metrics.
- **Code churn roughly doubled** post-AI — DORA didn't flag it because rewritten code typically didn't fail in production.

**Key line:** *DORA metrics were designed for a world where humans wrote all the code; they measure delivery outcomes but stay blind to how AI shapes the code itself.*

**Exam skeleton:** 4 metrics + assumptions; 2025 fifth = rework rate; break-down table (DF inflated → TrueThroughput; LT → PR cycle time; CFR → durability/turnover; MTTR survives); churn doubled unseen; DORA insufficient alone — pair, don't discard.

## 8.4 AI-era metrics and frameworks (CS14 slides 8–13)

**Four new metric families (CS14 slide 8):**
1. **AI Code Share** — % of committed code AI-generated/assisted (PR %, line %, commit %).
2. **Code Durability** — does code stick or churn? (CFR misses quiet rewrites).
3. **Complexity-Adjusted Output** — all deploys are not equal (auth system ≠ README edit).
4. **Innovation Rate** — new features vs maintenance/config work.

**Developer AI Impact Framework — 5 levels (memorize exactly):**
| Level | Name | Core question |
|---|---|---|
| 1 | Adoption | Are developers using AI tools? |
| 2 | Engagement | How deeply (modes: inline → chat → agentic)? |
| 3 | Productivity | Is AI accelerating *real* output? |
| 4 | Quality | Is AI code maintainable — does it stick? |
| 5 | Business Value | Is the investment paying off? |

Most orgs measure level 1 and stop; quality (4) is "almost never measured".

**Its 5 pillars (mnemonic: "All Coders Value Quality & ROI"):**
1. **Adoption:** DAU/WAU/MAU active-user rate; adoption by mode (inline/chat/agent); power vs casual distribution.
2. **AI Code Share:** AI-assisted PRs %, lines %, commits % (70% WAU can coexist with 10% code share = shallow usage).
3. **Velocity:** **CAT** (Complexity-Adjusted Throughput) per engineer per week; AI vs human delivery volume; cycle time split by origin.
4. **Quality:** **Code Turnover Rate** at 30-day (early warning) and 90-day (deep quality) windows, AI vs human; innovation rate.
5. **Cost & ROI:** fully-loaded AI cost per engineer (licence + tokens + training + admin); time-saved value (hours × loaded cost); **Net ROI = (Productive Value of Time Saved − Rework Cost from Code Turnover) / Total AI Tool Cost**.

Qualitative layer: developer-experience surveys → composite "AI Value Realization Score".

**Other frameworks (one line each):**
- **WAY** (Work-first, Agnostic, Yours): measure what happened (not self-report); across *all* tools or the picture lies; data must survive an audit. Layers: AI Adoption, AI Impact, AI ROI.
- **AWS 5 categories:** deployment velocity, code quality, operational efficiency, team productivity & satisfaction, business impact — with rollout in 3 phases: **Baseline & A/B testing → Granular cost tracking → Guardrail validation**.
- **DX Core 4:** speed, effectiveness, quality, business impact; unifies DORA + SPACE + DevEx.
- **DX AI measurement framework:** 3 dimensions — **Utilization** (usage, AI-assisted PR %, agent tasks), **Impact** (direct: hours saved, satisfaction, HEH; indirect: PR throughput, DXI, maintainability, change-fail %), **Cost** (spend & ROI). Line: *AI metrics measure what's happening; Core 4 metrics confirm whether it's working.*

**Exam skeleton:** 4 new metric families; 5-level hierarchy with "orgs stop at 1, skip 4"; 5 pillars + Net ROI formula; WAY/AWS/DX one-liners; AWS 3-phase rollout.

## 8.5 AI adoption J-curve (detailed)

Shape: adoption → temporary productivity dip + instability → recovery → outperform baseline. **Three trough forces (L-V-P):** learning curve, verification tax, pipeline adaptation. Implications: forecast the dip (staff × salary × drop × duration); don't judge ROI in weeks; provision review capacity *before* scaling licences; retune pipelines (smaller batches, stronger automated tests) to absorb throughput.

**Exam skeleton:** draw-as-text: Productivity vs Time — dip then rise above baseline; 3 forces; management response (forecast cost, provision verification, retune pipeline).

## 8.6 Developer cognitive load and lifecycle health

**Cognitive load** = total mental burden (intrinsic: task complexity; extraneous: tooling friction, context switching, waiting; germane: learning). AI changes it: reduces typing/tedium but *adds* verification burden, prompt crafting, agent supervision, review queues. Symptoms of overload: rubber-stamp reviews, acceptance-without-reading, meeting-afternoon complaints (Sample Q2!), rework spikes.
**Measure via:** SPACE/DXI surveys (satisfaction, flow, focus time), PR pickup & review turnaround, after-hours commits, context-switch counts.
**Lifecycle health = balanced scorecard across productivity + quality + reliability** (velocity never alone): throughput (CAT) + durability/turnover + CFR/MTTR + rework rate + cognitive load — reviewed together, trended over quarters.

**Exam skeleton:** 3 load types; AI's double effect (relieve production, add verification); overload symptoms incl. the sample scenario; health = balanced multi-dimensional scorecard, not speed alone.

## 8.7 Measuring whether AI augmentation actually works (the 40%-faster-but-defects-up design)

**Measurement framework (QBank B5 — memorize 6 steps):**
1. **Baseline before/parallel control:** measure pre-AI metrics; use A/B teams where possible (AWS phase 1).
2. **Throughput, complexity-adjusted:** CAT, not PR counts; split AI vs human.
3. **Quality counter-metrics:** 30/90-day code turnover (AI vs human), defect density, change failure rate, rework rate, escaped defects.
4. **Reliability:** MTTR, incident trends correlated with AI Code Share of affected modules.
5. **Human layer:** cognitive-load/DX surveys; review turnaround (verification tax evidence).
6. **Business layer:** Net ROI formula; J-curve-adjusted evaluation horizon (quarters).
Verdict logic: *faster delivery is real value only if quality/reliability metrics hold stable or improve and turnover doesn't reveal churn-faking velocity.*

---

# PART B — KEY FLOWS (TEXT-ONLY; WRITE THE ARROW LINE, THEN DESCRIBE EACH STAGE)

**F1. AI-augmented SDLC end-to-end with control points**
Spec (intent + must-not) → AI plan/design → HITL ADR & threat-model review → AI code+tests (guardrails) → Deterministic sensors (schema/static/tests) → HITL merge → AI-assisted CI (security, smart tests) → Deploy (canary) → SLO watch → Telemetry feedback → back to Spec.
Prose: humans own intent (spec), trade-offs (ADR), and release; AI owns drafts; deterministic sensors own verification; production teaches the loop.

**F2. HITL approval gate**
AI artifact → evidence pack (diff + sensor report + risk score) → classify risk (reversible/irreversible; blast radius) → low risk: auto-proceed with log → high risk: named human reviews evidence → approve (recorded sign-off) or reject with findings → findings return to AI (bounded loop) → escalate if bound hit.
Prose: gates sit at irreversible steps; humans get evidence not verdicts; every sign-off is recorded.

**F3. Coding-agent harness loop (Lab Sheet 2)**
Feedforward files (openapi.yaml + style-guide + harness-rules + generation-task) → Agent generates JSON {appJs, testJs} → Schema validation → Static sensor (patterns/exact assertions) → Dynamic sensor (run tests) → pass: write audit log, human review → fail: findings-only repair prompt → max 2 attempts → untrusted-until-both-pass.
Prose: control before (feedforward), sensing after (deterministic sensors), bounded repair, machine-readable findings, audit log at every attempt.

**F4. Circular validation prevention**
Approved spec → [branch A] AI generates implementation → sensors; [branch B, separate context, spec-only] AI generates tests → human reviews test intent → independent oracles (golden master / property / contract / mutation) → integration level → merge gate requires all three sources green.
Prose: independence = different source (spec not code), different context, human intent review, external oracles.

**F5. AI-driven CI/CD pipeline**
Commit → lint/schema/static → AI security & dependency analysis (SCA) → smart test selection (impact analysis) → build → deploy to staging (autonomous) → canary release with SLO analysis → auto-abort/rollback on breach (pre-authorized) → human approval for full rollout → release notes drafted by AI → telemetry captured.
Prose: autonomous up to production edge; rollback pre-authorized; rollout human-approved; every stage emits audit evidence.

**F6. Legacy monolith → microservices migration**
Inventory + AI dependency map → characterization tests pin behaviour → pick seam (strangler-fig) → define OpenAPI contract per service → AI generates service under harness guardrails → sensors + parity checks (old vs new) → contract tests in CI → canary route % traffic → human gate per extraction → next seam → retire monolith slice.
Prose: behaviour pinned before change; contracts as source of truth; verification at every step; traffic shifting is the rollback lever.

**F7. AI-assisted requirements workflow**
Raw transcript/tickets → PII scrub → AI extraction (explicit + implicit needs) → synthesis to stories/criteria (EARS-scored) → traceability matrix → drift/scope-creep monitor → BA/PO checklist review → recorded sign-off → backlog (MoSCoW).
Prose: grounding via RAG against curated corpus; ambiguity scoring vs INCOSE/EARS; humans approve before backlog entry.

**F8. DevOps-to-Dev feedback loop**
SLO breach/incident → telemetry + post-mortem → sanitized, curated learning (RAG corpus entry) → spec "must-not" update + regression test generation → agent guardrail rule update → next AI-assisted change verified against new tests → escape-rate trend confirms loop works.
Prose: production knowledge re-enters through *specs and tests*, sanitized and human-curated, so AI velocity amplifies learning, not error.
# PART C — EXPANDED QUESTION BANK (49 QUESTIONS, FULL MODEL ANSWERS)

> Sources: **[YB]** = your 16-question bank; **[SP]** = official Sample Questions paper; **[N]** = new question. Marks are estimates based on the sample paper's 3-mark-per-part pattern. Every answer is written to be reproduced under exam conditions: definition → application → example/justification.

## CATEGORY A — CONCEPTUAL / DESCRIPTIVE

---

### A1. [N] Trace the evolution of SDLC models from Waterfall to Continuous AI-Augmentation. How does each model respond to change? (6 marks)

**Concept tags:** Module 1; SDLC evolution; change management.

**Model answer:**

**Waterfall** is a linear, sequential lifecycle — Requirements → Design → Implementation → Testing → Deployment → Maintenance — where each phase is signed off before the next begins. It assumes change is expensive and rare; feedback arrives only at final testing, so errors discovered late are costliest. It suits fixed, contract-driven systems.

**Agile** replaced phase gates with short iterations (sprints), working software every 1–4 weeks, and continuous stakeholder feedback (sprint reviews, story mapping, spikes). It *embraces* change: scope flexes through the Product Backlog while time and team size stay fixed. Its residual weakness is that delivery past "sprint done" — integration, testing depth, release — often remained manual.

**DevOps/Continuous Delivery** extended Agile to production: CI/CD pipelines, infrastructure as code, automated test and deployment, measured by DORA metrics (deployment frequency, lead time, MTTR, change failure rate). Change flows continuously and safely to production.

**Continuous AI-Augmentation** embeds AI assistants and agents in *every* phase simultaneously: AI drafts requirements from transcripts, proposes designs and ADRs, generates code and tests, reviews PRs, optimizes pipelines, and triages incidents. The lifecycle stops being a relay of phases and becomes a **continuous loop** — the spec declares intent, AI generates artifacts, deterministic sensors verify them, humans approve at irreversible points, and production telemetry feeds back into specs and tests.

The through-line: each era shortened the feedback cycle (months → weeks → days → minutes), and each moved the constraint: from writing code (Waterfall) to coordinating teams (Agile) to operating pipelines (DevOps) to **verifying machine-generated work** (AI era). AI collapses the cost of producing code but not the cost of verifying it — which is why the AI era's defining discipline is deterministic verification plus human-in-the-loop governance rather than faster typing.

---

### A2. [YB-Q13a] What are foundation models? Explain their types and why they matter for software engineering. (5 marks)

**Concept tags:** Module 2; foundation models; model types.

**Model answer:**

A **foundation model** is a large AI model trained on broad data at scale so that it can be adapted — by prompting or fine-tuning — to a wide range of downstream tasks, rather than being built for one task (Chip Huyen, *AI Engineering*). Its performance emerges from scale; its adaptability comes from in-context learning.

**Types relevant to software engineering:**
1. **Large Language Models (text)** — requirements drafting, documentation, code explanation, review comments.
2. **Code models** — pretraining weighted toward source code; power completion tools (Copilot-style) and coding agents.
3. **Multimodal models** — accept text plus images/audio; enable UI-mockup-to-code and design review.
4. **Embedding models** — map text/code to vectors; the engine behind semantic search and RAG retrieval over documentation, tickets and codebases.
5. **Small/local models** (e.g., Llama-3 served via Ollama) — run on-premise for data-sensitive or cost-constrained use, trading capability for control.

**Training pipeline:** pretraining (next-token prediction over massive corpora) → fine-tuning (task specialization) → RLHF/instruction tuning (alignment to human preferences). This is why outputs are *probabilistic*, which drives non-determinism and hallucination risk.

**Why they matter:** they are the substrate of every AI-augmented SDLC tool — assistants, agents, review bots, test generators. Their properties (emergent capability, prompt adaptability) and risks (hallucination, knowledge cutoff, bias, data leakage, injection) define the course's central problem: integrating probabilistic generators into deterministic engineering systems with appropriate guardrails.

---

### A3. [YB-Q13b] Explain the transformer architecture and the roles of tokenization, attention and context windows in software-engineering applications. (6 marks)

**Concept tags:** Module 2; transformer; attention; tokens; context.

**Model answer:**

A transformer processes input in a fixed pipeline: (1) **tokenization** — text is split into subword tokens; (2) **embedding** — each token becomes a high-dimensional vector; (3) **positional encoding** adds order information; (4) stacked blocks of **self-attention + feed-forward layers** transform the representations; (5) an output head produces a **probability distribution over the next token**, sampled repeatedly (autoregressive generation).

**Tokenization:** tokens are the unit of cost and memory. Code tokenizes worse than prose (an identifier like `getCustomerBalance` splits into several tokens), so a "small" file can consume a large context budget. Tokenization determines how much of a spec, codebase or conversation fits in the prompt, and APIs bill per token — so context management is an economic and correctness constraint on AI-assisted engineering.

**Attention:** self-attention lets every token attend to every other token, capturing long-range dependencies in one hop. In software terms, attention is how the model links a variable's declaration to its distant usage, or an API call to its definition, or a harness rule to the code that must obey it. Attention also enables **in-context learning**: examples placed in the prompt change behaviour without retraining — the mechanism behind few-shot prompting and the lab's feedforward rule files.

**Context window:** the finite token budget the model can attend to at once — its entire working memory. Consequences: (a) context must be *engineered* — the lab feeds four small authoritative files (OpenAPI contract, style guide, harness rules, task) rather than a whole repository; (b) middle-of-context degradation means relevance beats volume; (c) long agent sessions accumulate stale context ("context rot") and need summarization or restart; (d) truncation silently drops rules — a key agent failure mode.

Together these explain the three practical levers of AI-assisted coding: control *what* enters the context (feedforward), control *how* output is sampled (temperature), and control *what* is verified afterwards (deterministic sensors) — because the model itself is a probabilistic sequence generator.

---

### A4. [N] Why do large language models produce non-deterministic outputs? List the sources of variability and how each is controlled. (5 marks)

**Concept tags:** Module 2; non-determinism; temperature; sampling.

**Model answer:**

LLMs are next-token probability generators: at each step they output a distribution over the vocabulary and a sample is drawn. Variability arises from four sources:

1. **Sampling parameters.** Temperature > 0 samples from the distribution (higher temperature = more randomness); top-p/top-k add further variability. *Control:* temperature 0–0.1 for code generation (the lab's live harness uses **temperature 0.1**), fixed top-p, and pinned model versions.
2. **Prompt/context sensitivity.** Tiny changes in wording, token order or attached files shift the distribution. *Control:* freeze inputs — versioned rule files and contracts as feedforward (the lab's four files), exact output-schema instructions.
3. **Provider-side state.** Model version updates, batching, and floating-point non-determinism on different hardware can change results even at temperature 0. *Control:* pin model versions; do not rely on "same prompt = same bytes".
4. **Agent environment stochasticity.** Tool results, retries, timeouts and file states differ per run, so an agent's *trajectory* diverges even with a frozen prompt. *Control:* bounded attempt counts (lab: max 2), deterministic sensors, audit logs per attempt.

**The engineering principle:** you cannot make the generator deterministic, so you make the *system* deterministic around it — allow variability at generation, forbid it at verification. Every candidate artifact passes the same deterministic checkpoints: schema validation → static analysis → exact-assertion tests → human approval → release gates. The conclusion line worth writing: *"the pipeline is deterministic even when the model is not — verified artifacts, not raw generations, enter production."*

---

### A5. [YB-Q14] Design a prompt strategy for generating production-grade Python APIs. Explain how task decomposition, context management, constraints and output validation improve the result. (6 marks)

**Concept tags:** Module 2; prompt engineering; production-grade code.

**Model answer:**

A production-grade prompt strategy has five layers:

**1. Role and framing.** "You are a senior Python engineer implementing a FastAPI service against the attached OpenAPI contract" — primes conventions (typing, error handling) and anchors the model to the contract as the single source of truth.

**2. Task decomposition.** Never "build the API". Decompose: (a) data models/schemas from the contract; (b) one endpoint at a time; (c) validation and error responses; (d) tests per endpoint; (e) documentation. Small tasks raise per-step correctness and make each step independently verifiable — a failure is localized instead of poisoning the whole generation.

**3. Context management.** Attach the minimum authoritative context: the relevant contract section, the org style guide, one exemplar handler (few-shot — attention imitates structure). Do not paste whole repositories; irrelevant context dilutes attention and costs tokens, and middle-of-context content degrades.

**4. Constraints (feedforward guardrails).** Enumerate forbidden behaviours explicitly: "no external dependencies beyond the allow-list; no hardcoded secrets; return RFC-style error JSON; use exact assertions in tests; do not invent endpoints that are not in the contract." Forbidden lists are the cheapest hallucination guardrail, because models comply better with explicit negatives than with silence.

**5. Output validation (feedback guardrails).** Demand a machine-checkable output contract (exact JSON schema; no Markdown fences). Then validate mechanically: schema check of the response → static analysis (lint, type check) → run the generated tests with exact assertions (`assert response.status_code == 400`) → integration test against a real server → human review before merge. Feed back *only the findings* on failure (the lab's repair-prompt pattern), bound the repair attempts, and log each attempt.

**Why each layer improves the result:** decomposition localizes errors; curated context grounds generation in truth; constraints prevent known failure classes before they occur; validation converts a probabilistic output into a verified artifact. The strategy is exactly the harness pattern from the lab generalized to a production language stack.

---

### A6. [YB-Q15] Explain agentic workflows. Describe the roles of tools, planning, execution, feedback and verification, and the risks of granting agents unrestricted access. (6 marks)

**Concept tags:** Module 2; agentic workflows; autonomy risks.

**Model answer:**

An **agentic workflow** is an LLM-driven loop in which the model plans, acts through tools, observes results and iterates toward a goal: **Goal → Plan → Act (tools) → Observe → Verify → repeat until verified or budget exhausted.** Unlike single-shot prompting, the agent's own actions generate new context that steers subsequent steps.

- **Tools** give the agent agency: file read/write, shell commands, test runners, linters, web search. Tool quality determines the ceiling of agent capability — a test-runner tool turns "hope the code works" into "know whether it works".
- **Planning** is task decomposition and sequencing (what to build first, where seams are). Weak planning is the root of cascading errors, which is why spec-driven workflows review the *plan* artifact before execution.
- **Execution** is the edit/build/run cycle, ideally sandboxed with least-privilege credentials.
- **Feedback** is what the environment returns — test failures, sensor findings, compiler errors. The lab pattern matters here: return *only specific, machine-readable findings*, not vague prose, so repairs are targeted.
- **Verification** closes the loop and must be **deterministic** (schema validation, static analysis, exact-assertion tests) rather than the agent's self-assessment, because models are overconfident judges of their own output.

**Risks of unrestricted access (six):** (1) destructive actions — deleting files, force-pushing, dropping tables; (2) security exposure — secrets in context, installing hallucinated/malicious dependencies, shell or prompt injection through repo or web content; (3) cost/runaway loops without attempt bounds; (4) silent drift — confidently solving the wrong problem; (5) data exfiltration of proprietary code to external endpoints; (6) compounding errors across multi-agent hand-offs.

**Controls:** least-privilege tool grants; sandboxing; allow-listed commands; bounded attempts (the lab stops after 2 generation attempts); human approval for irreversible actions; full audit logging (attempt-log.json). The design conclusion: agents earn autonomy in proportion to the determinism of their verification and the reversibility of their actions.

---

### A7. [N] What is RAG (Retrieval-Augmented Generation) and why is it essential for enterprise AI-assisted engineering? (5 marks)

**Concept tags:** Module 2/3; RAG; grounding; guardrails.

**Model answer:**

**Definition.** RAG is a pattern in which the model's answer is *grounded* in documents retrieved at query time: retrieve relevant chunks from an indexed corpus (contracts, past SRS, ADRs, compliance rules, incident learnings) and place them in the context so the generation cites or uses retrieved facts instead of parametric memory alone.

**Why enterprises need it — four reasons:**
1. **Currency.** Models have a knowledge cutoff and hallucinate APIs. RAG injects the *actual* current contract, schema or library docs, cutting dependency/API hallucination.
2. **Organizational truth.** Company-specific rules (style guides, architecture constitutions, MoSCoW priorities) exist nowhere on the internet; RAG makes them enforceable in every prompt.
3. **Access control.** Retrieval is permission-checked per user/team, so the model only sees what the requester may see — a data-leak guardrail impossible with raw prompts.
4. **Traceability and audit.** Answers can cite retrieved sources, so reviewers verify claims (and auditors trace which knowledge produced a decision).

**In the course's workflows:** RAG grounds requirements synthesis (compliance rules, past requirements — Module 3 guardrail against omissions/hallucination), powers context management (retrieve the right spec chunk instead of dumping repos — Module 2), and carries the DevOps-to-Dev feedback loop (curated incident learnings as a retrieval corpus — Module 8).

**Limits (worth one line):** RAG reduces but does not eliminate hallucination (retrieval can miss or mis-rank), so it is always paired with HITL sign-off and deterministic validation — grounding is a guardrail layer, not a replacement for verification.

---

### A8. [N] Define code smells. List common smells and explain how AI assists detection and refactoring, including the verification requirement. (5 marks)

**Concept tags:** Module 4; code smells; refactoring.

**Model answer:**

A **code smell** is a surface symptom in source code that usually indicates a deeper design problem — not a bug (the code works), but a maintainability risk that makes future change expensive and defect-prone.

**Common smells (with one-line meaning):**
1. **Long Method** — one function doing many jobs; hard to test and reuse.
2. **Large Class / God class** — a class that knows and does too much.
3. **Duplicate Code** — the same logic in several places; fixes get applied inconsistently.
4. **Long Parameter List** — unwieldy signatures signalling missing abstractions.
5. **Feature Envy** — a method obsessed with another class's data; belongs elsewhere.
6. **Primitive Obsession** — raw strings/ints where domain types belong (e.g., raw phone-number strings).
7. **Shotgun Surgery** — one conceptual change forces edits across many files.
8. **Magic Numbers/Strings** — unexplained literals scattered through logic.

**AI assistance:** AI scans large codebases quickly, ranks smells by risk, and proposes standard refactorings (extract method/class, rename, introduce parameter object, replace magic literals with named constants). It converts smell remediation from an occasional cleanup into a continuous, automated review finding.

**The mandatory verification requirement:** refactoring must preserve behaviour. Therefore: (1) tests must exist *before* refactoring — for legacy code without tests, write **characterization tests** that pin current behaviour first; (2) run tests before and after each refactoring step; (3) bound AI repair iterations (the harness's 2-attempt pattern) and require human review of the final diff; (4) keep refactoring PRs separate from feature PRs so review attention stays focused. Without this, AI-assisted refactoring becomes AI-assisted behaviour change — the exact defect class sensors exist to catch.

---

### A9. [N] Explain the AI adoption J-curve, its three driving forces, and how management should respond. (5 marks)

**Concept tags:** Module 1/8; J-curve; verification tax.

**Model answer:**

DORA frames AI adoption as a **J-curve**: after rollout, productivity temporarily *dips* and instability rises (the trough), then recovers and eventually **exceeds the pre-adoption baseline** (the long arm of the J). Writing it as text: *Productivity vs Time — baseline → adoption → dip → recovery → above-baseline.*

**Three forces drive the trough (L-V-P):**
1. **Learning curve** — teams need time to build fluency with tools and new workflows; the ramp has real, measurable cost.
2. **Verification tax** — reviewing AI-generated output *at the volume AI makes possible* takes longer than teams provisioned for; PR review queues become the new bottleneck.
3. **Pipeline adaptation** — existing pipelines were built for lower change volume and must be retuned (smaller batch sizes, stronger automated testing, tighter version control) before they can safely absorb the new throughput.

**Management response:**
- **Treat the dip as a forecastable transition cost**, not as evidence AI failed — model it as *staff size × fully loaded salary × productivity drop × dip duration* and include it in the ROI calculation.
- Do not judge ROI in week 4; evaluate over quarters (baseline → trough → recovery).
- **Provision verification capacity before scaling licences** — the tax is a capacity-planning item, not a surprise.
- Retune pipelines early (automation, smaller batches) so throughput lands on a system that can absorb it.

The J-curve explains the classic scenario (Sample Q2): a 40% throughput jump with unchanged lead time is the trough visible in the data — the code-production stage accelerated while verification and pipeline stages had not yet adapted.

---

## CATEGORY B — SCENARIO-BASED APPLICATION

---

### B1. [YB-Q1] An organisation transitions from traditional Agile to AI-augmented software engineering. (a) How does AI change the SDLC phases? (b) Benefits? (c) Risks of increased velocity? (d) Why do technical debt and architectural drift grow? (e) Mechanisms to maintain quality? (10 marks)

**Concept tags:** Module 1; end-to-end synthesis question.

**Model answer:**

**(a) Phase changes.** Requirements move from interviews/BRDs to AI mining of transcripts, tickets and telemetry, with stories and acceptance criteria auto-drafted and scored for ambiguity (INCOSE/EARS). Design moves from architect-only boards to AI-proposed architectures and auto-drafted ADRs, validated by threat modeling. Coding shifts from writing to reviewing — pair programming and autonomous agents generate most code. Testing becomes AI generation from specifications, with exact-assertion suites run by deterministic sensors. Deployment gains AI dependency resolution, smart test selection, canary analysis, self-healing and rollbacks. Overall the lifecycle becomes a continuous loop instead of a phase relay.

**(b) Benefits.** Compressed cycle time and lead time; higher throughput (complexity-adjusted); broader test coverage; consistent documentation; instant prototyping for requirement validation; reclaimed developer hours redirected to design and verification; 24/7 agent availability for batch work (migrations, dependency upgrades).

**(c) Risks of velocity.** Verification becomes the bottleneck (the "verification tax"): AI collapses the cost of producing code, not of verifying it. Review queues explode; reviewers rubber-stamp under volume; defects and insecure code ship at machine speed; automated errors replicate across artifacts before humans notice. Early adoption dips productivity (J-curve trough: learning curve, verification tax, pipeline adaptation).

**(d) Debt and drift.** **Technical debt** grows because generation outpaces refactoring/test investment — code is cheap, so shortcuts accumulate invisibly, and churn (rewrites) hides debt from failure-based metrics. **Architectural drift** grows because thousands of locally-sensible AI changes lack global architectural review: duplicated patterns, inconsistent layering, implicit new dependencies, and LLM "architectural bloat" (tendency toward maximal complexity) pull the real structure away from the intended one.

**(e) Quality mechanisms.** (1) Deterministic verification gates — schema validation, static analysis, exact-assertion tests, contract tests (the harness pattern). (2) Bounded autonomy — agents stop after N attempts and escalate. (3) Shift-left policy enforcement — security/quality checks at commit time. (4) HITL approval at irreversible steps (merge, release, migrations) with recorded sign-off. (5) ADRs and spec-driven development to anchor intent and prevent drift. (6) Operational feedback loops — telemetry/incidents update specs, tests and guardrails. (7) Balanced metrics (turnover, CFR, rework rate alongside throughput) so debt stays visible. Conclusion: velocity is safe only when verification capacity and governance grow with it.

---

### B2. [YB-Q3] A client sends a lengthy, unstructured business discussion containing functional requirements, non-functional requirements and ambiguities. Design an AI-assisted requirements engineering workflow covering extraction, synthesis, hallucination/omission risks, HITL approval, and RAG guardrails. (8 marks)

**Concept tags:** Module 3; requirements; RAG; HITL.

**Model answer:**

**Step 1 — Preprocess and protect.** Scrub PII/confidential data from the transcript; classify sensitivity; keep processing inside approved endpoints (data-leak guardrail). Split the discussion into topical chunks for retrieval.

**Step 2 — AI-assisted extraction (elicitation).** Prompt the model to extract: explicit functional requirements; implicit needs (e.g., "users always retry failed payments twice" → reliability requirement); non-functional requirements (performance, security, availability, compliance); and open ambiguities as a *question list* for the client. Ground the extraction with **RAG**: retrieve the org's requirement templates, past SRS, compliance rules so drafts follow house standards and nothing forbidden is proposed.

**Step 3 — Automated synthesis.** Convert extracted needs into structured artifacts: user stories ("As a [role], I want [action], so that [benefit]"), acceptance criteria in BDD form ("Given… When… Then…"), NFRs with measurable thresholds. Run **ambiguity and quality scoring** against INCOSE guidelines/EARS syntax; every requirement must be singular, unambiguous, testable.

**Step 4 — Traceability and drift monitoring.** Build a bidirectional traceability matrix: client statement → requirement → acceptance criterion → (later) test. Flag requirements with no source (possible hallucination) and source statements with no requirement (possible omission). Set up drift/scope-creep monitoring against the backlog.

**Step 5 — Risks and countermeasures.** *Omissions* — countered by the traceability gap report and a client confirmation round. *Hallucinated requirements* — countered by source-link rules: a requirement without a retrievable source is rejected. *Data leaks* — the Step 1 scrubbing plus private endpoints. *Compliance* — a compliance-rule RAG corpus checked at generation time plus a legal/compliance checklist in review.

**Step 6 — HITL sign-off.** The BA/PO reviews with a fixed checklist: completeness vs source, ambiguity scores, compliance items, stakeholder confirmation of implicit-needs inferences. Approval is recorded (who/what/when) before items enter the backlog under MoSCoW prioritization.

**Why the design is sound:** AI does the volume work (extraction, drafting, scoring) while humans retain strategic trade-offs, accountability and sign-off; RAG keeps generation grounded and permissioned; traceability converts "trust me" into checkable evidence.

---

### B3. [YB-Q4] A team uses an LLM to generate an initial software architecture from business requirements. Explain how you would validate it before implementation — ADRs, scalability, security, threat modeling, human review. (6 marks)

**Concept tags:** Module 3; architecture validation; ADR; threat modeling.

**Model answer:**

Treat the generated architecture as an *unverified proposal* and run six validation activities before any implementation:

1. **Requirements/traceability validation.** Every architecturally significant requirement (ASR) maps to a design element, and every element maps back to a requirement — elements with no requirement are scope pollution to remove; ASRs with no element are gaps to fix.
2. **Architecture risk analysis (scalability/performance).** Identify expected load and growth; examine statelessness, data partitioning, caching, and failure modes; require the AI's assumptions (users, TPS, data volume) to be stated and checked against real business numbers — LLMs default to plausible generic patterns, not your load profile.
3. **Security and threat modeling.** Draw trust boundaries in text form (client → API → services → data), then apply **STRIDE** per boundary: Spoofing → authentication; Tampering → integrity controls; Repudiation → audit logs; Information disclosure → encryption/access control; DoS → rate limiting; Elevation of privilege → least-privilege authorization. Add AI-specific threats: secrets in generated config, hallucinated dependencies as supply-chain surface.
4. **ADR capture and review.** Every significant choice becomes an ADR (Title, Status, Context, Decision, Consequences, Alternatives). This forces the LLM's choices to expose trade-offs, gives future agents organizational memory (they stop re-proposing settled questions), and gives architects something reviewable.
5. **Architecture prototyping.** Build a spike of the riskiest interaction (e.g., highest-traffic path) and *measure* — prototyping replaces plausible-sounding claims with evidence.
6. **Human review gate.** A senior architect reviews the evidence pack (traceability report, risk analysis, threat model, ADRs, prototype results) and approves — recorded sign-off. Optionally enforce simplicity rules from a constitution ("prefer single relational tables over new microservices") to counter LLM architectural bloat.

Only after all six pass does implementation begin — typically spec-driven, so the approved architecture becomes the anchor the code must converge to.

---

### B4. [YB-Q2] An AI coding assistant generates different implementations for the same requirement across runs. Explain why this happens and design a validation mechanism (prompt constraints, schema validation, unit/integration testing, static analysis, human approval, release gates). (8 marks)

**Concept tags:** Module 2; non-determinism; deterministic checkpoints.

**Model answer:**

**Why outputs differ.** The model is an autoregressive *probability* generator: at each step it samples from a distribution, so temperature > 0 produces different token sequences per run. Variability is amplified by prompt sensitivity (tiny context changes shift the distribution), provider-side state (model version updates, batching, hardware floating-point differences), and agent-environment stochasticity (tool results and retries differ). Non-determinism is therefore intrinsic — the engineering task is not to eliminate it but to prevent it from reaching production.

**Validation mechanism (six layers; "allow variability in generation, forbid it in verification"):**

1. **Prompt constraints (feedforward).** Freeze inputs: versioned OpenAPI contract as single source of truth; style-guide and harness-rule files; temperature 0–0.1; exact output schema ("return only JSON with keys appJs and testJs, no Markdown fences"); explicit forbidden behaviours (no new dependencies, no secrets). Frozen, versioned prompts make runs comparable.
2. **Schema validation.** Validate the *model response itself* before it touches the repository: exactly the expected keys, non-empty strings, no code fences (the lab's `validateModelObject`). A malformed response never enters the loop at all.
3. **Static analysis (deterministic sensors).** Linters, type checks, security scanners, and required-pattern checks — e.g., tests must contain the exact assertions `assert.strictEqual(response.status, 201)` and `(..., 400)`. Static sensors fail identically on every run, whatever the model produced.
4. **Unit and integration testing (dynamic sensors).** Execute the tests against the real artifact: unit tests with exact assertions (broad assertions like `status >= 400` are forbidden because they pass broken code), integration tests against an ephemeral server covering the full contract. Dynamic execution proves behaviour, not just presence.
5. **Human approval.** The developer reviews the diff and sensor report; for high-risk modules (auth, payments) a second reviewer. Humans own the merge decision — recorded sign-off.
6. **Release quality gates.** CI blocks merge until static + dynamic + review all pass; production release gates add canary/SLO checks. Repair loops are **bounded** (max 2 attempts in the lab) with an audit log per attempt, then escalate to a human.

**Result:** runs differ, but what ships is deterministic — every artifact, from any run, must pass the same fixed checkpoints. As the one-line conclusion: *"the pipeline is deterministic even when the model is not."*

---

### B5. [YB-Q11] Developers are 40% faster with AI assistants, but defect rates and production incidents rose. Design a measurement framework to evaluate whether AI augmentation is actually beneficial. (8 marks)

**Concept tags:** Module 8; DORA; Developer AI Impact Framework; balanced metrics.

**Model answer:**

The scenario is the classic AI-era trap: **local throughput up, system health down**. A sound framework measures all five levels of the Developer AI Impact Framework, not just speed.

1. **Baseline and controls.** Capture 2–3 pre-AI quarters for every metric below; where possible run A/B teams (AI-enabled vs control) across multiple release cycles (AWS rollout phase 1), because org-wide events can fake causality.
2. **Throughput — complexity-adjusted.** Replace raw PR counts with **CAT (Complexity-Adjusted Throughput) per engineer**, split AI-assisted vs human-only, plus cycle time by origin. The "40% faster" claim must survive complexity adjustment — it may be boilerplate volume, not value.
3. **Quality counter-metrics.** **Code Turnover Rate at 30 and 90 days, segmented AI vs human** (is AI code sticking or being rewritten?); defect density; **rework rate** (2025 DORA's fifth metric); change failure rate; escaped-defect count. Expectation if augmentation is healthy: turnover for AI code converges toward human turnover.
4. **Reliability.** MTTR and incident frequency correlated with the AI Code Share of affected modules — the scenario's rising incidents must be attributable (which modules, who/what wrote them, which gate missed them).
5. **Human layer.** Review turnaround and PR pickup time (evidence of the verification tax), developer cognitive-load/DX surveys (rubber-stamping shows up here before it shows in defects).
6. **Economics.** **Net ROI = (Productive Value of Time Saved − Rework Cost from Code Turnover) / Total AI Tool Cost**, evaluated on a J-curve-adjusted horizon (quarters, not weeks) and including the forecasted dip cost.

**Verdict logic:** AI augmentation is genuinely beneficial only when (a) complexity-adjusted throughput rises, (b) quality/reliability metrics stay flat or improve, (c) AI-vs-human turnover converges, and (d) Net ROI is positive after rework. If speed rises but turnover and incidents rise with it, the organization is buying velocity with debt — the response is to provision verification capacity (review staffing, deterministic gates, smart test selection) and re-measure, not to abandon or blindly expand AI.

---

### B6. [YB-Q12] An enterprise is deploying AI coding assistants to 500 developers. Explain how to evaluate the economic benefits and costs. (6 marks)

**Concept tags:** Module 1/8; economics; ROI.

**Model answer:**

**Cost side — fully loaded, per engineer per month:** (1) seat licences; (2) token/inference usage costs (agents multiply this through retries and long contexts); (3) AI infrastructure/compute and integration; (4) training and enablement time; (5) **review overhead** — the verification tax of humans reviewing machine-generated volume; (6) **defect remediation** — the cost of rewriting churned AI code; (7) security-risk cost — incidents and patching from AI-introduced vulnerabilities.

**Benefit side:** time saved per engineer per week × fully loaded engineering cost (salary + benefits + overhead); faster time-to-market (revenue earlier); quality gains where deterministic gates improve defect catch rates; developer satisfaction/retention effects.

**Formulas:** ROI (%) = (Value − Investment)/Investment × 100; **Net ROI = (Productive Value of Time Saved − Rework Cost from Code Turnover) / Total AI Tool Cost**; J-curve dip cost = staff size × fully loaded salary × productivity drop × dip duration. For 500 developers, the dip cost alone is material — a 10% productivity drop for one quarter across 500 loaded salaries must be modelled.

**Process:** (1) pilot with a control group and baseline metrics (AWS phase 1: baseline & A/B testing); (2) attribute costs granularly — licences, tokens by workspace/project — to get true cost-per-outcome (phase 2); (3) enforce guardrail validation so velocity gains don't spike debt or failure rates (phase 3); (4) measure Net ROI quarterly, segmented by team and use-case (agents vs completions), and reallocate to where ROI is proven.

**Decision framing:** the licence fee is the *smallest* line item — the economic verdict hinges on review overhead and rework cost, which are governed by verification infrastructure. Organizations that provision deterministic gates convert rework cost into cheap automated checking; organizations that don't pay the tax twice (slow reviews + churned code).

---

### B7. [YB-Q16] Design an automated code-review workflow that identifies code smells, security vulnerabilities and potential logic defects. How do you prevent false positives and ensure review independence? (6 marks)

**Concept tags:** Module 5; automated review; false positives.

**Model answer:**

**Workflow (per PR):** (1) PR opened; diff computed with repo/spec context attached (style guide, constitution, OpenAPI contract — grounding prevents style-guessing). (2) Static tier runs deterministically: linters, type checks, secret scanning, SCA — these are *gates*, not opinions. (3) AI review tier analyzes the diff semantically: code smells (long methods, duplication), security patterns (injection, missing validation, hardcoded secrets, unsafe deserialization), logic risks (unhandled errors, boundary conditions, missing tests for changed paths). (4) Findings are normalized: each carries severity, file/line, explanation, and a *concrete suggested fix*; uncertain items are posted as questions, not defects. (5) Deterministic CI gates still decide merge-blocking; AI findings are advisory input to the human reviewer. (6) Human reviewer adjudicates — accepts, rejects with reason, or requests changes; the approving human owns the merge (accountability).

**False-positive prevention (four controls):** (a) **grounding** — review against the repo's own conventions and specs, not generic best practice; (b) **evidence requirement** — a finding without a concrete explanation/fix template is suppressed; (c) **confidence thresholds + routing** — low-confidence items go to a "question" bucket; (d) **feedback loop** — human valid/invalid marks tune rules and prompts over time, so the reviewer learns the codebase's real standards.

**Review independence (the critical rule):** the AI (or human) that *generated* the code must not be the sole reviewer of it — use a different model/context for review, keep AI review complementary to (never a replacement for) human review, and separate concerns across tiers: deterministic sensors prove facts, AI reviews flag risks, humans own the decision. This preserves both technical independence and the accountability chain (recorded sign-off per merge).

---

### B8. [SP-Q2] A financial platform buys AI coding licences for 200 developers. PR throughput rises 40%, but lead time stays 4 weeks and developers spend afternoons in meetings, waiting for QA environments, and fighting deployment configurations. (a) Why did throughput not translate into time-to-market? (b) How does the org shift from "local optimization" to "systemic enablers"? (c) Identify two technical engineering foundations to prioritize. (9 marks)

**Concept tags:** Module 1/8; sample paper; flow; platform engineering.

**Model answer:**

**(a) Why throughput ≠ time-to-market.** Lead time is a *system* property — the end-to-end flow from commit to production — while PR throughput measures a single stage. The scenario shows the bottleneck **moved, not disappeared**: code production accelerated 40%, but review queues, QA environment availability, and deployment configuration remain slow, so finished PRs pile up as *inventory* (in queuing terms: the fastest station in a slow pipeline just builds a queue). This is textbook **local optimization**: improving one stage while the constraint sits elsewhere. It is also the visible J-curve trough — the verification tax (reviews of AI-generated volume) and pipeline adaptation (environments, deployment config built for lower volume) had not been provisioned. Developers "writing code faster but spending afternoons waiting" is the classic symptom that coding is no longer the constraint.

**(b) From local optimization to systemic enablers.** Stop measuring individual output and start measuring *flow*: lead time, PR pickup time, review turnaround, environment wait time, deployment failure rate. Then remove the systemic constraints: (1) **platform engineering** — an internal developer platform providing self-service, ephemeral QA environments and paved-road deployment pipelines (ends "fighting deployment configurations"); (2) **automated verification infrastructure** — deterministic gates, smart test selection, contract tests that absorb AI-generated PR volume without queueing humans; (3) **WIP management** — limit concurrent PRs so quality gates don't degrade into rubber stamps; (4) **spec/context infrastructure** so AI output conforms to system intent. DORA's finding applies: AI *amplifies* existing system properties — the org's weak foundations are being amplified as friction, and fixing foundations converts the same 40% into actual market speed.

**(c) Two technical engineering foundations (3 marks each in spirit):**
1. **CI/CD and automated testing infrastructure** — deterministic quality gates (static analysis, exact-assertion suites, contract tests), environments-as-code for instant QA provisioning, smart test selection so longer verification isn't needed per PR. This removes the review/environment bottleneck and makes higher throughput *safe*.
2. **Specification and context infrastructure** — versioned specs/contracts (OpenAPI), ADRs, and an internal RAG knowledge base, so AI assistants generate code that fits the intended architecture instead of drifting, and reviews verify against intent rather than reviewer memory.

Together they shift the constraint from human verification queues to automated, deterministic checks — which is the precondition for AI throughput to become customer-visible delivery.

---

### B9. [N] A team feeds raw client call transcripts (containing customer PII and confidential pricing) into an external LLM to draft requirements. Identify the risks and design the guardrails. (5 marks)

**Concept tags:** Module 3; data leaks; compliance; guardrails.

**Model answer:**

**Risks.** (1) **Data leakage/exfiltration** — PII and confidential pricing leave the organization's trust boundary; depending on the provider's terms, prompts may be retained, logged, or used for training. (2) **Compliance breach** — privacy regulation (GDPR/India DPDP-type obligations) restricts processing and cross-border transfer of personal data; leaking client data can breach both contract and law. (3) **Hallucinated/omitted requirements** — the model may invent requirements not in the transcript or silently drop implicit needs; compliance-relevant obligations may be missed entirely. (4) **Access control collapse** — anyone with the transcript output sees data they were never entitled to.

**Guardrails (layered):** (1) **Pre-process:** scrub/mask PII and commercially sensitive fields before any model call; chunk transcripts; classify sensitivity. (2) **Endpoint control:** use enterprise/private endpoints (or on-prem models like local Llama-3) with no-training-retention contracts; keep a data-flow record for audit. (3) **RAG grounding with permissions:** requirements are generated against a permission-checked corpus of templates, past SRS and compliance rules — the model cites retrievable sources, and requirement-without-source is rejected (anti-hallucination rule). (4) **HITL sign-off:** BA/PO checklist — completeness vs source transcript, ambiguity scores (INCOSE/EARS), compliance checklist, client confirmation of implicit needs — with recorded approval before backlog entry. (5) **Traceability:** bidirectional matrix from transcript statement → requirement → acceptance criterion, so omissions surface as gaps and hallucinations surface as orphans.

**Conclusion line:** the pattern is "sanitize at the boundary, ground with RAG, approve by human" — AI accelerates synthesis, but data protection and accountability stay inside the organization.

---

### B10. [N] A developer accepts an AI pair-programming suggestion that calls a library function that does not exist, using a deprecated package version. The code passes build but fails in staging. Diagnose what happened and design prevention. (5 marks)

**Concept tags:** Module 2/4; dependency hallucination; guardrails.

**Model answer:**

**Diagnosis.** This is a **dependency/API hallucination**: the model generated a plausible call signature that fits the intent ("makes sense to a human reader") rather than a verified API. Root causes: (1) the model's training data is stale (knowledge cutoff — deprecated package) or the API was simply invented; (2) no grounding — the prompt did not include the project's actual dependency manifest or the library's real interface; (3) the acceptance loop failed — the developer accepted without verifying the API exists (over-trust of suggestions), and the build passed only because the call is late-bound or the package exposes a permissive surface; (4) no deterministic gate caught the mismatch before staging.

**Prevention (layered):** (1) **Feedforward grounding** — attach `requirements.txt`/lockfile and relevant interface definitions to context; rule files say "use only dependencies in the manifest; do not invent APIs; if unsure, say so". (2) **Allow-list + pinning** — dependencies pinned with lockfiles; style guide forbids new packages without review; SCA validates every dependency change. (3) **Static gates** — type checkers/linters that resolve imported symbols catch non-existent functions at commit time, not staging time. (4) **Dynamic gates** — the module's tests must actually exercise the new call with exact assertions (execution beats plausibility). (5) **Human verification habit** — pair-programming suggestions are accepted per-diff; a quick "does this symbol resolve?" check is part of acceptance discipline; unfamiliar APIs require doc verification. (6) **Feedback into guardrails** — add the hallucinated API to a block-list corpus so the pattern is caught next time.

**Lesson:** hallucination risk scales with context vacuum — the cure is grounding (real manifest, real interfaces) plus deterministic symbol-resolution gates, never reviewer optimism.

---

### B11. [N] A team wants AI to convert 3,000 backlog tickets into user stories with acceptance criteria. Design the workflow, including quality scoring and traceability. (5 marks)

**Concept tags:** Module 3; synthesis; EARS; traceability.

**Model answer:**

**Workflow.** (1) **Preprocess:** deduplicate, scrub PII, cluster tickets by theme/epic (AI clustering with human naming). (2) **Batch conversion with grounded prompts:** per cluster, prompt with the team's story template ("As a [role], I want [action], so that [benefit]") and BDD acceptance-criteria template ("Given [context], When [action], Then [outcome]"); forbid inventing scope beyond ticket text — anything uncertain becomes an explicit open question. (3) **Automated quality scoring:** each generated requirement scored against **INCOSE guidelines / EARS syntax** for singularity, unambiguity, testability; sub-threshold items are auto-flagged for rewrite rather than shipped. (4) **Traceability:** every story links back to its source ticket IDs; run gap analysis — tickets with no story (omission) and stories with no ticket (possible hallucination) are reviewed explicitly. (5) **HITL sign-off:** PO reviews by exception — the scoring and gap reports let a human approve 3,000 conversions in hours by checking flags, samples and edge clusters; approvals recorded. (6) **Backlog integration:** stories enter under MoSCoW priority; a drift monitor compares backlog growth against new ticket inflow (scope-creep watch).

**Why it works:** the design converts an unreviewable volume problem into an exception-driven review problem — quality scores and traceability gaps tell humans exactly where to look, so speed comes from AI volume while correctness comes from deterministic scoring plus targeted human sign-off.

---

## CATEGORY C — COMPARE AND JUSTIFY

---

### C1. [YB-Q5] Compare AI pair programming with autonomous coding agents — workflows, benefits, risks, control mechanisms — and recommend when each should be used. (8 marks)

**Concept tags:** Module 4; interaction patterns.

**Model answer:**

**Workflows.** *AI pair programming:* the developer drives; suggestions arrive inline (completions) or in chat; the human accepts, rejects or edits each suggestion in real time; the context is the open files plus conversation. *Autonomous agents:* the human issues a bounded task ("implement POST /items per this OpenAPI contract"); the agent plans, edits multiple files, runs tools and tests, and iterates within guardrails; the human reviews the resulting PR and sensor evidence rather than each line.

**Benefits.** Pair programming: near-zero setup, immediate feedback, keeps design thinking human, low per-change risk, excellent for learning and exploration. Agents: parallelism (multiple tasks at once), works well for well-specified batch work (migrations, test generation, scaffolding, dependency upgrades), amortizes cost over large tasks, enforces spec-driven discipline because tasks must be well-defined.

**Risks.** Pair programming: subtle over-acceptance ("AI says yes" bias), unverified APIs in accepted suggestions, skill atrophy, suggestion-chasing that fragments design intent. Agents: compounding errors across an unattended run, destructive actions, hallucinated dependencies, prompt injection through repository content, runaway cost, and confident delivery of the *wrong* solution.

**Control mechanisms.** Pair programming: human judgment per diff, small diffs keep review cheap, normal PR gates. Agents: bounded attempts (the lab's 2-attempt cap), least-privilege tools, sandboxing, deterministic sensors before human review, recorded audit logs per attempt, human approval at merge, escalation path when bounds are hit.

**Recommendation.** Use **pair programming** when design context lives in the developer's head, the change is small or exploratory, and per-change human judgment is cheap. Use **agents** when the task is well-specified, verifiable by deterministic checks, and batch-like in nature — with governance: frozen specs as feedforward, bounded repair, human merge approval. The deciding variable is not "how much AI" but *how verifiable the task is*: verifiable tasks can be delegated; unverifiable ones need the human in the loop per change.

---

### C2. [N] Compare spec-first, spec-anchored and spec-as-source development. When is each appropriate? (6 marks)

**Concept tags:** Module 3; specification spectrum.

**Model answer:**

**Spec-first:** a specification is written *before* coding to guide initial implementation; once code exists the spec may or may not be maintained — its value is initial clarity. Lowest process overhead; risks spec rot. *Appropriate for:* one-off prototypes, short-lived services, teams new to SDD who need the clarity without the maintenance contract.

**Spec-anchored:** the spec is maintained *alongside* the code for the system's whole life; behaviour changes require updating both, kept synchronized. Delivers traceability and drift control at moderate cost. *Appropriate for:* most production enterprise systems — long-lived, multi-team, audit-sensitive; the sweet spot of the spectrum.

**Spec-as-source:** the spec is the *only artifact humans edit*; code is entirely generated and never manually modified; any behaviour change means changing the spec and regenerating ("disposable code"). Maximum automation; total dependence on spec completeness and verification strength. *Appropriate for:* narrow, contract-heavy domains with strong deterministic verification — APIs generated from OpenAPI, infrastructure from IaC templates — where a sensor can fully prove spec conformance.

**Justification logic (the comparison axis):** as you move along the spectrum, (1) human control shifts from code to spec, (2) the verification burden grows because humans no longer read code as the backstop, (3) the cost of an incomplete spec grows from "confusion" (spec-first) to "wrong production behaviour" (spec-as-source). Therefore the choice is governed by one question: *can behaviour be fully and deterministically verified against the spec?* If yes, spec-as-source is rational; if behaviour exceeds what the spec can express (tacit performance behaviour, edge-case workarounds), stay spec-anchored. This is exactly the Sample Q1 trap: "disposable microservices" claims spec-as-source benefits while dismantling the verification (no refactoring, no comprehensive tests) that makes disposability safe.

---

### C3. [N] Compare RAG, fine-tuning and long-context prompting as grounding strategies for enterprise AI-assisted engineering tools. Which would you choose and why? (6 marks)

**Concept tags:** Module 2/3; grounding strategies.

**Model answer:**

**RAG (Retrieval-Augmented Generation):** at query time, retrieve relevant chunks from a curated, permissioned corpus and place them in context. *Strengths:* always current (update the corpus, not the model); access-controlled per team; traceable (answers cite sources); cheapest per knowledge update. *Weaknesses:* retrieval quality is a new failure mode; only grounds what was indexed.

**Fine-tuning:** train the base model further on organizational data. *Strengths:* bakes in style/conventions; no runtime retrieval infrastructure. *Weaknesses:* expensive and slow to update (every corpus change needs retraining); risks memorizing sensitive data into weights (leakage via extraction); no per-user access control; opaque provenance — poor auditability.

**Long-context prompting:** stuff maximal context into one prompt. *Strengths:* simplest to build; everything visible in one session. *Weaknesses:* cost scales with tokens; middle-of-context degradation ("lost in the middle"); no permissions (everything in the prompt is exposed to the session); context rot in long sessions; silently truncated past the limit.

**Choice and justification:** choose **RAG as the default enterprise grounding** — it uniquely satisfies the four enterprise constraints simultaneously: currency (update documents, not weights), access control (permission-checked retrieval), auditability (cited sources), and cost (retrieval is cheap). Layer the others narrowly: fine-tune (or parameter-efficient adapters) only for stable, non-sensitive stylistic conventions; use long-context prompting for single-session deep analysis of one bounded artifact (e.g., reviewing one service against its full spec). In course terms: RAG is the guardrail layer that makes requirements synthesis and incident-feedback loops safe (Modules 3 and 8), which is why the handout lists "RAG/Human-in-the-Loop sign-off" as the required guardrail pair — retrieval grounds the content, the human owns accountability.

---

### C4. [N] Compare Waterfall, Agile and AI-augmented lifecycles on control points, risk profile and measurement. Which controls survive the AI era unchanged? (6 marks)

**Concept tags:** Module 1/7; lifecycle comparison.

**Model answer:**

| Dimension | Waterfall | Agile | AI-augmented |
|---|---|---|---|
| Control points | Phase gates (document sign-offs) | Sprint boundaries, DoD, backlog prioritization | Deterministic sensors + dynamic gates + HITL at irreversible steps |
| Where risk lives | Late (testing at the end) | Distributed (per sprint) | Concentrated in verification: unreviewed machine output |
| Change cost | Very high | Low within sprint | Very low to generate, high to verify |
| Primary artifact | Contracts/documents | Working software | Spec + verified artifacts + audit logs |
| Measurement | Plan adherence | Velocity, burndown | DORA + AI-era metrics (turnover, AI code share, CAT, rework rate) |

**Narrative:** Waterfall controls risk with *upfront documents*; Agile controls it with *short iterations*; the AI lifecycle controls it with *deterministic verification* — because generation cost collapsed, the scarce resource moved to validation, so control points migrate from "approval to proceed" to "proof before merge/release".

**Controls that survive unchanged — and why.** (1) **MTTR** among metrics: it measures how a team responds to failure regardless of who wrote the code; it remains the honest DORA signal. (2) **Human accountability structures**: sign-offs, ownership and liability were never about typing speed — the human who approves still owns the outcome (Module 7). (3) **Specification discipline**: clear, testable requirements mattered in Waterfall's BRDs, matter in Agile's acceptance criteria, and matter *more* in SDD because the spec is now the source of truth. (4) **Defense-in-depth testing**: the test pyramid survives; AI shifts where tests come from, not the need for independent oracles. What genuinely dies is *manual inspection as the primary gate* — human review alone cannot scale to machine-generated volume; it survives only as the top layer above deterministic sensors.

---

### C5. [N] Compare traditional peer code review with AI-augmented code review. What does each catch, what fails, and how should they be combined? (6 marks)

**Concept tags:** Module 5; code review.

**Model answer:**

**Traditional peer review** catches: design misalignment with system intent, domain-logic errors, architectural violations, maintainability concerns — everything requiring *organizational memory and judgment*. It fails at: scale (queues explode under AI volume), consistency (quality varies with reviewer fatigue — "surface-level comments under time pressure"), and coverage (nothing forces checking every path).

**AI-augmented review** catches: mechanical and pattern-level issues at 100% coverage — code smells, missing input validation, injection patterns, hardcoded secrets, error-handling gaps, missing tests for changed paths, style drift. It fails at: system-level intent ("is this the right change?"), novel domain logic, and it hallucinates or over-flags (false positives) when ungrounded in repo conventions.

**Combination (the justified design):** layer them. Tier 1 — deterministic static gates (lint, type, secret scan, SCA) as merge blockers. Tier 2 — AI review for breadth: every finding carries explanation + concrete fix; uncertain findings posted as questions; humans mark valid/invalid to tune precision over time. Tier 3 — human review for judgment: intent, architecture, domain logic; the human approves and owns the merge. Two independence rules complete the design: the generator of the code must not be its only reviewer (different model/context for AI review), and AI findings never auto-merge code — automation proposes, humans dispose, with recorded sign-off for accountability.

**Justification:** each layer covers the other's blind spots — deterministic gates give consistency, AI gives coverage and tireless breadth, humans give intent and accountability. Removing any layer reproduces a known failure: no Tier 1 → inconsistent style/security enforcement; no Tier 2 → queues explode (Sample Q2's review bottleneck); no Tier 3 → nobody is accountable for what ships.

---

### C6. [N] Compare deterministic CI gates with AI-judged ("LLM-as-judge") quality gates. Where is AI judgment acceptable, and where must the gate be deterministic? (6 marks)

**Concept tags:** Module 5/7; quality gates; non-deterministic validation.

**Model answer:**

**Deterministic gates** evaluate artifacts against fixed rules: schema validation, linters, type checks, exact-assertion tests, contract tests, coverage thresholds, policy-as-code. Properties: identical verdict for identical input every run; explainable failure (which rule, which line); gameable only by changing the artifact. They are the only *safe* authority over non-deterministic AI outputs, because they don't inherit the model's error distribution.

**AI-judged gates** use a model to evaluate quality ("does this PR look secure?", "is this plan reasonable?"). Properties: flexible, catches fuzzy issues rules can't express; but the verdict is itself sampled (non-deterministic), inherits the judge model's blind spots, can be prompted/injected, and cannot be audited as evidence.

**Where AI judgment is acceptable:** advisory roles — code-review *comments*, release-risk *recommendations*, canary-analysis *summaries*, plan review *suggestions*, prioritizing findings by likely severity. In all of these a human or a deterministic condition still makes the final call.

**Where the gate must be deterministic:** merge blocking, security policy enforcement, contract conformance, test pass/fail, production rollout authorization. Rationale (three reasons worth writing): (1) **auditability** — a gate decision must be reproducible evidence for the accountability chain; (2) **stability** — a gate that flips between runs either blocks good work randomly or passes bad work randomly; (3) **independence** — an AI judge shares training-data blind spots with the AI generator, so AI-judging-AI reproduces circular validation.

**Combined pattern (the justified answer):** AI judgment ranks and explains; deterministic rules decide. E.g., AI triages 200 findings into "likely-real / likely-noise" order, and the CI gate still passes/fails on the fixed rule set. This preserves throughput gains without sacrificing reproducibility or accountability.

---

### C7. [N] Compare human-written tests, AI-generated tests and independent oracles as validation mechanisms. Why can't AI-generated tests alone validate AI-generated code? (6 marks)

**Concept tags:** Module 5; test oracles; circular validation.

**Model answer:**

**Human-written tests:** derive expectations from the approved spec and domain judgment. Strength: an *independent* oracle — the author's understanding is not the generator's. Weakness: slow, limited coverage breadth, and they age.

**AI-generated tests:** derive cases from a spec (or worse, from the code). Strength: breadth (boundaries, negative cases), speed, format consistency. Weakness: when generated in the same context as the code, they share its misunderstanding — expectations are re-derived from the same premises as the implementation, so there is effectively **no oracle**, only a mirror.

**Independent oracles:** validation sources that cannot inherit the generator's error — *golden-master/characterization tests* (pin the legacy system's behaviour), *property-based invariants* (idempotence, monotonicity, round-trip laws), *contract tests* (generated from the OpenAPI contract independently of implementation), and *mutation testing* (inject mutants; tests that cannot kill mutants are proving nothing).

**Why AI-tests-AI fails alone (the four reasons):** (1) shared blind spots — if the model misread the requirement, code and tests misread it identically, producing a green pipeline around a wrong product (correlated, not independent, errors); (2) tautological drift — models tend to assert what the code does rather than what the spec requires; (3) no external reference — test value comes from comparing against an oracle; same-context generation eliminates the independence that defines an oracle; (4) false confidence — green results inflate trust precisely where output is least reviewed.

**Justified combination:** human-owned acceptance criteria → AI generates the test *scaffold* from the spec in a **separate context** (implementation never shown) → exact assertions mandated by human harness rules → independent oracles layered on (golden master + property + contract + mutation) → integration level with real execution → human test-intent review. The comparison's conclusion: tests are evidence only when the tester is independent of the accused — hence Lab 3's "validation rule engine ensuring AI-generated test code is structurally sound and independent of AI-generated production code."

---

## CATEGORY D — DIAGRAM / WORKFLOW (write the arrow line, then describe each stage)

---

### D1. [YB-Q1-ext] Draw/describe the end-to-end AI-augmented SDLC workflow with human-in-the-loop control points. (6 marks)

**Concept tags:** Module 1/7; master workflow.

**Flow (write this line first):**
Spec (intent + must-not rules) → AI plan/design → **HITL: ADR + threat-model review** → AI code + tests (guardrails active) → Deterministic sensors (schema → static → tests) → **HITL: merge approval** → AI-assisted CI (security scan, smart test selection) → Deploy (canary) → SLO watch → Telemetry feedback → back to Spec.

**Describe each stage (the prose that earns the marks):**
1. **Spec:** humans author the specification and its "must not do" rules — the source of truth bounding every downstream agent.
2. **AI plan/design:** agents propose architecture and ADR drafts grounded in retrieved org context (RAG).
3. **HITL gate 1:** senior review of ADRs, risk analysis and STRIDE threat model; recorded sign-off; this is where trade-offs are owned.
4. **Generation under guardrails:** feedforward rules (style guide, harness rules, forbidden dependencies), bounded attempts.
5. **Deterministic sensors:** schema validation of outputs, static analysis, exact-assertion unit/integration tests — identical verdicts every run; failures return as machine-readable findings for bounded repair.
6. **HITL gate 2:** human merge approval with the sensor evidence pack; the approver owns the change.
7. **CI/CD:** AI-driven dependency/security analysis, test impact analysis selecting the minimal test set; staging deploy autonomous.
8. **Canary + SLO watch:** progressive rollout with automatic abort/rollback on breach (pre-authorized), full rollout human-approved.
9. **Feedback:** sanitized telemetry and post-mortems update specs, regression tests and guardrail rules — closing the loop.

**Closing line:** humans own intent, trade-offs and release; AI owns drafts; deterministic sensors own verification; production owns learning. Control points sit exactly at the irreversible steps.

---

### D2. [YB-Q8] An enterprise wants to introduce AI into its CI/CD pipeline. Design the workflow (dependency resolution, security analysis, smart test selection, impact analysis, release management, rollbacks/self-healing) and state which operations can be autonomous vs approval-required. (8 marks)

**Concept tags:** Module 6/7; CI/CD; autonomy boundary.

**Flow:**
Commit → static gates (lint/schema/secrets) → AI dependency resolution + SCA → AI security/patch analysis → test impact analysis → selected test run → build → staging deploy (autonomous) → canary with SLO analysis → breach? → auto-rollback (pre-authorized) : ramp → **HITL: full production rollout** → AI release notes → telemetry captured → feedback to specs/tests.

**Stage descriptions:**
1. **Dependency resolution (AI):** proposes compatible versions and upgrade PRs — but through lockfiles, registry allow-lists and SCA; minor patches with green tests can auto-merge; major upgrades need approval.
2. **Security/vulnerability analysis (AI):** reads CVEs *in context* (do we call the vulnerable function?), prioritizes, drafts patch PRs; exploitability judgment is advisory, gate logic is deterministic.
3. **Smart test selection / impact analysis:** dependency graph + coverage map + defect history select the minimal high-value test set per commit; a coverage guard (selected tests must cover changed lines) plus nightly full suite prevents escapes.
4. **Release management:** AI drafts release notes and scores release risk (change size, components, recent incidents); canary ramp decisions follow SLO statistics.
5. **Self-healing/rollback:** SLO breach → automatic abort/rollback (pre-authorized because recovery speed beats review speed) → incident record → post-mortem feeds the learning loop. Rollbacks are automated; forward-fixes never are.

**Autonomy vs approval table (the marks are here):**
- Fully autonomous: lint/schema/static, test selection and execution, staging deploys, canary *analysis*, auto-abort/rollback on SLO breach, minor dependency patches with green tests + SCA.
- Approval-required: production rollout, major dependency upgrades, schema/data migrations (always), security patches to production (fast-track human approval), any change to the pipeline itself.

**Justification line:** autonomous for reversible and verifiable steps; approval for irreversible, external-facing or data-touching steps — with every autonomous action logged for audit.

---

### D3. [YB-Q10] A production app has recurring failures despite passing dev and test stages. Design a DevOps-to-Dev feedback loop using telemetry and incident data to improve future AI-assisted development, without compromising security or introducing wrong learning signals. (8 marks)

**Concept tags:** Module 8; feedback loop; telemetry safety.

**Flow:**
SLO breach / incident → telemetry + blameless post-mortem → sanitize (PII/secrets scrub, aggregate) → human curator approves learning → knowledge corpus (RAG) entry + spec "must-not" update + regression test generation → guardrail rule update → next AI-assisted change verified against new tests → escape-rate trend measured.

**Stage descriptions (the five marks hide here):**
1. **Detect and capture:** monitoring alerts, incident timelines, error patterns, affected modules — raw material for learning.
2. **Post-mortem:** blameless analysis separates facts (what happened) from hypotheses (why); only *verified causes* may feed the loop.
3. **Sanitize before any AI context:** scrub PII/secrets, aggregate where possible, keep sensitive telemetry inside approved/private endpoints; access-controlled RAG ensures the corpus is permissioned.
4. **Human curation = wrong-signal prevention:** a curator rejects correlation-as-cause ("hot file ≠ bad file"), rejects unverified post-mortem claims, and version-controls the corpus so a bad lesson can be rolled back.
5. **Inject into development assets:** (a) RAG corpus entries so agents retrieve "known failure modes" while coding; (b) new regression tests with exact assertions; (c) spec updates — the failed behaviour becomes a "must not" clause; (d) guardrail/sensor rule updates if a gate missed the defect class.
6. **Close and measure:** track escape rate and repeat-incident rate — the loop is working only if recurrence falls.

**Why failures repeated despite passing stages:** the dev/test stages verified against *yesterday's* knowledge; the loop's job is to make today's production knowledge part of tomorrow's verification. Security note to close: production data never flows raw into AI prompts — it flows through sanitization, curation and permissioned retrieval.

---

### D4. [N — Lab Sheet 2] Describe the feedforward–feedback harness loop for coding agents. Explain each stage and the role of the audit log and attempt bound. (8 marks)

**Concept tags:** Module 4; lab harness; deterministic checkpoints.

**Flow:**
Feedforward files (openapi.yaml + style-guide.md + harness-rules.md + generation-task.md) → coding agent generates JSON {appJs, testJs} → schema validation of the response → static sensor (required patterns, exact assertions) → dynamic sensor (execute node --test) → pass → audit log + human review; fail → findings-only repair prompt → bounded (≤2) attempts.

**Stage descriptions:**
1. **Feedforward:** four small authoritative files constrain behaviour *before* generation — the contract is the single source of truth; the style guide forbids external dependencies and hardcoded secrets; harness rules mandate exact output format, exact assertions (`assert.strictEqual(response.status, 201)` / `(400)`), forbid broad assertions, and declare generated files untrusted until both sensors pass.
2. **Generation + schema validation:** the model must return exactly two keys with non-empty strings and no Markdown fences — malformed responses are rejected before touching the repo.
3. **Static sensor (deterministic):** checks file presence, required test patterns (route exercised, exact 201/400 assertions), server export, JSON content-type. Presence checks that pass/fail identically every run.
4. **Dynamic sensor:** *executes* the tests against the artifact — static ≠ dynamic: static proves the test exists and targets the contract; dynamic proves the code actually satisfies it. Both are needed; either alone is gameable or blind.
5. **Bounded repair:** on failure, only the machine-readable findings array is fed back ("repair findings without removing valid behaviour"); the loop stops after **2 attempts**, then escalates to the human.
6. **Auditability:** every attempt is logged (attempt-log.json: mode, findings, pass/fail, model) — evidence for review, debugging and governance.

**Interpretation line:** first-attempt success is valid; repair happens only on deterministic findings; API errors are logged separately from validation failures. The harness embodies the course thesis: control the input (feedforward), sense the output (deterministic sensors), bound the loop (attempts), record everything (audit).

---

### D5. [N] Design the human-in-the-loop approval workflow across the AI-augmented lifecycle. Where are the gates, what evidence does each use, and what prevents rubber-stamping? (6 marks)

**Concept tags:** Module 7; HITL; gates.

**Flow:**
AI artifact → evidence pack (diff + sensor report + risk score) → risk classification (reversible? blast radius? data?) → low risk: auto-proceed with log → high risk: named human reviews evidence → recorded approve/reject → reject findings return to AI (bounded) → bound exceeded: escalate.

**Gates and their evidence (the table earns the marks):**
| Gate | Evidence pack | Approver |
|---|---|---|
| Requirements sign-off | Traceability matrix, ambiguity scores, open questions | BA / Product Owner |
| Architecture/ADR | Traceability, risk analysis, threat model, prototype results | Senior architect |
| Merge | Diff, static + dynamic sensor report, AI review findings | Developer/reviewer |
| Test intent | Test-purpose summary, coverage of acceptance criteria | QA owner |
| Release | Canary/SLO results, risk score, rollback plan | Release manager |

**Anti-rubber-stamping controls (why the design holds under load):** (1) gate only *irreversible* steps — over-gating every change is what creates rubber stamps; (2) evidence packs are pre-digested (sensor verdicts, scores), so review effort stays small; (3) risk classification is policy-based, so the same change class always gets the same treatment; (4) sign-offs are recorded (who/what/when) — visible accountability disciplines approvals; (5) escalation on attempt-bound exhaustion prevents "approve to unblock" shortcuts; (6) DORA-style monitoring of review turnaround exposes when the tax is overloading approvers.

**Closing line:** HITL exists to keep humans accountable for what ships and to insert judgment at irreversible steps — evidence-based, recorded, and minimal by design.

---

### D6. [YB-Q6-flow] Design the safe workflow for migrating a legacy monolith to microservices using autonomous coding agents (dependency identification, generation/refactoring, guardrails, API compatibility, version/dependency management, deterministic verification, human approval). (8 marks)

**Concept tags:** Module 4; legacy modernization. *(Case-study depth appears in F1; here the workflow itself is the answer.)*

**Flow:**
Inventory + AI dependency/call-graph map → characterization tests pin behaviour → choose seam → OpenAPI contract per service (contract-first) → AI generates service under harness guardrails → sensors + parity checks (old vs new) → contract tests in CI → version + lockfile management → canary traffic shift (strangler-fig) → **HITL gate per extraction** → next seam → retire monolith slice.

**Stage descriptions (one per mark):**
1. **Dependency identification:** AI maps the call graph, data coupling and shared databases to find natural seams (bounded contexts); humans validate the map — AI output is a hypothesis to check, not a fact.
2. **Characterization tests first:** pin the monolith's *current* behaviour (including its bugs) so every later change is measured against reality; without this, "migration" is indistinguishable from silent re-implementation.
3. **Contract-first boundaries:** define OpenAPI contracts before code; contracts become the feedforward input and the compatibility oracle.
4. **Guarded generation:** the agent receives contract + style guide + harness rules — never the whole monolith (context limits, legacy context loss); rules forbid new unpinned dependencies and secrets.
5. **API compatibility:** additive-only changes, versioned endpoints, deprecation policy; contract tests in CI fail the build on accidental breaking changes.
6. **Version/dependency management:** lockfiles, pinned versions, semantic versioning per service; each extraction is independently deployable and revertible.
7. **Deterministic verification:** static sensors + exact-assertion tests + *parity checks* comparing old-path and new-path outputs on golden requests; canary shifts a small traffic percentage with SLO watching.
8. **Human approval gates:** senior review at each seam extraction with the evidence pack (sensor reports, parity results, rollback plan); the strangler-fig routing flag is the rollback lever — traffic can return to the monolith at any point.

**Closing line:** the workflow's safety comes from pinning behaviour before changing it, verifying deterministically at every step, and keeping every extraction reversible until a human approves it.

---

### D7. [N] Design the AI-assisted requirements workflow from raw client transcript to approved backlog, showing where RAG and HITL sit. (5 marks)

**Concept tags:** Module 3; requirements workflow.

**Flow:**
Raw transcript → PII scrub + sensitivity classification → RAG-grounded AI extraction (explicit + implicit needs, NFRs, ambiguity question-list) → automated synthesis (stories + BDD criteria, INCOSE/EARS scoring) → traceability matrix (statement ↔ requirement ↔ criterion) → gap report (omissions = unsourced statements; hallucinations = orphan requirements) → BA/PO checklist review → recorded sign-off → backlog (MoSCoW).

**Descriptions:**
1. **Scrub first:** PII and confidential fields are removed before any model call (data-leak guardrail); processing on approved endpoints.
2. **RAG grounding:** retrieval injects house templates, past requirements and compliance rules, so drafts follow standards and cite sources; requirement-without-source is rejected.
3. **Synthesis with scoring:** stories in the role/action/benefit form, criteria in Given/When/Then form; every item scored for singularity, unambiguity, testability; low scores loop back for rewrite.
4. **Traceability as the proof layer:** the matrix converts "trust me" into checkable evidence — omissions appear as unsourced statements, hallucinations as orphan requirements.
5. **HITL sign-off:** the BA/PO reviews the gap report and flagged items (exception-driven review of a large corpus), confirms implicit-need inferences with the client, and records approval before backlog entry.

**Closing line:** AI does volume; RAG keeps it grounded and permissioned; the traceability matrix makes errors visible; the human sign-off keeps accountability where the law and the business expect it.

---

## CATEGORY E — CRITICAL ANALYSIS

---

### E1. [YB-Q7] An LLM generates both production code and unit tests for a requirement. Explain why validating AI-generated code using AI-generated tests alone is unreliable, and design a mechanism to prevent circular validation (independent oracles, specification-based validation, integration testing, human checkpoints). (8 marks)

**Concept tags:** Module 5; circular validation — the course's flagship critical-analysis topic.

**Model answer:**

**Why it is unreliable.** When the same model, in the same context, produces both the implementation and its tests, the validation loop is *circular*: the tests validate the code against the model's own understanding rather than against the requirement. Four concrete failure mechanisms follow. (1) **Shared blind spots:** if the model misreads the requirement, it misreads it identically in both artifacts — the errors are correlated, not independent, so tests are green precisely where the code is wrong. (2) **Tautological drift:** models tend to write tests that describe what the code does, converting the test suite from an oracle into a mirror. (3) **No independent oracle:** a test's evidential value comes from comparing actual behaviour against an *externally derived* expectation; same-context generation removes the externality, so a passing suite proves only internal consistency ("the model agrees with itself"). (4) **False confidence:** green pipelines inflate trust exactly where output is least reviewed — the failure ships with maximum credibility.

**Prevention mechanism (layered):**
1. **Specification-based validation:** derive expectations from the *human-approved spec*, never from the code. The lab demonstrates this: humans write the harness rules that mandate exact assertions (`assert.strictEqual(response.status, 201)` / `(400)`), so the contract — not the model — dictates what must be asserted; broad assertions (`status >= 400`) are forbidden because they cannot falsify wrong behaviour.
2. **Independent test oracles:** layer oracles the generator cannot have contaminated — golden-master/characterization tests pinning known-good behaviour; property-based invariants (idempotence, round-trip); contract tests derived from the OpenAPI document independently of implementation; mutation testing to prove the tests can fail (a suite that kills no mutants detects nothing).
3. **Context separation:** generate tests in a separate session whose input is *only the spec* — the implementation is never shown, breaking the correlation at its source.
4. **Integration testing at a different level:** real HTTP execution against an ephemeral server exercises the full stack, catching what unit-level mirroring hides.
5. **Human checkpoints:** humans own acceptance criteria before generation and review test *intent* after generation; the human approves merges; Lab 3 formalizes this as a validation rule engine ensuring AI-generated tests are "structurally sound and independent of the AI-generated production code".

**Judgment to close:** AI-generated tests are valuable scaffolding, but they are evidence only when their expectations trace to the approved specification and to oracles the generator never touched. Independence is not a nice-to-have — it is the definition of a test.

---

### E2. [YB-Q9] An autonomous AI agent introduces a security vulnerability into a production application. Discuss the ownership and accountability model: HITL approval, shift-left policy enforcement, dynamic quality gates, and liability. (8 marks)

**Concept tags:** Module 7; governance; liability — sample-paper style critical analysis.

**Model answer:**

**The principle.** AI is a tool, not a legal or moral person: *accountability always resides with the humans and the organization that configured and deployed it.* "The agent did it" is not a defence — the developer who merged, the approver who signed off, and the organization that set the agent's autonomy limits own the outcome. The correct post-incident question is never "who wrote it, the human or the model?" but "which control point should have caught it, and why was it missing or bypassed?"

**Accountability model (RACI over AI-augmented work).** The *operator* (developer running the agent) owns the change it produced; the *reviewer* owns adjudicating AI findings; the *approver* (senior engineer/release manager) owns high-risk releases; the *platform/governance team* owns the harness — gates, policies, allowed autonomy; the *organization* owns liability to customers and regulators. Blameless post-mortems attribute to the *system* (which gate failed) while accountability stays with the roles.

**Why the vulnerability shipped — the four controls that should have caught it:**
1. **HITL approval:** an autonomous agent's change to production-touching code must pass a human merge/release gate with an evidence pack (diff, sensor report, risk score). If the vulnerability came from an agent change, either the gate was absent for that change class or the evidence pack was too thin to reveal the issue — both are governance defects, fixable by policy.
2. **Shift-left policy enforcement:** security policy must be enforced at commit/PR time — secret scanning, SCA, static security analysis, policy-as-code — not at release time. Under AI velocity, end-of-pipeline inspection is structurally too late: defects replicate faster than reviews. The vulnerability proves the shift-left net had a hole.
3. **Dynamic quality gates:** gates must be risk-adaptive — an agent-authored change to auth code should trigger the strict gate set (full suite + security scan + senior review), not the trivial one. The rule that closes this incident: *agent-generated changes get deterministic verification before human eyes, never after.*
4. **Liability:** contract and regulatory liability (data protection, sector compliance) attaches to the deploying organization; vendor indemnities do not transfer developer accountability. Negligence is defined operationally: deploying AI output *without the mandated gates*. Hence the audit trail matters — prompts, outputs, sensor reports, attempt logs and recorded sign-offs establish exactly where the process failed and who was responsible for that step.

**Judgment to close:** the incident indicts the pipeline, not the technology. The remedy is to strengthen the four controls — gates at irreversible steps, shift-left enforcement, risk-adaptive verification, recorded accountability — and to treat the agent's autonomy as a *granted, revocable privilege* whose limits are written, enforced by the harness, and audited.

---

### E3. [N] Analyse hallucination as a risk in AI-augmented SDLC: types, causes, why review alone fails, and the layered mitigation strategy. Can hallucination ever be fully eliminated? (8 marks)

**Concept tags:** Module 1/2/5; hallucination — guaranteed critical-analysis topic.

**Model answer:**

**Definition and types.** Hallucination is the generation of confident, plausible, false content. In software engineering it appears as: (1) **API/dependency hallucination** — invented functions, packages, endpoints (the deprecated-package staging failure); (2) **requirement hallucination** — invented features/stakeholders polluting scope; (3) **code-logic hallucination** — subtly wrong business rules that look correct; (4) **security hallucination** — confident code with injection vectors or hardcoded secrets; (5) **documentation/test hallucination** — misleading docs and tautological tests that hide the above.

**Causes.** The model is a probabilistic sequence generator with no truth mechanism: it predicts plausible next tokens from training-data patterns (stale by the knowledge cutoff), under prompt pressure it prefers fluent completion over refusal, and finite context means it may never see the authoritative contract. Plausibility is optimized, not truth.

**Why human review alone fails.** Review capacity is finite while AI output volume scales — queues explode (the verification tax), reviewers fatigue into rubber-stamping, and hallucinated code is *designed by statistics to look correct*, attacking exactly the reviewer's trust heuristic. Sample Q2's scenario is this failure in action: code faster than humans can verify.

**Layered mitigation (map to the guardrail stack):** (1) **Feedforward grounding** — RAG over authoritative corpora (real manifests, contracts, compliance rules); explicit forbidden lists ("no new dependencies; say 'unsure' rather than inventing"). (2) **Schema validation** — machine-check the response format before it touches the repo. (3) **Deterministic sensors** — static analysis and symbol resolution (a nonexistent function fails type checking), exact-assertion dynamic tests (execution falsifies plausible fiction). (4) **Independent oracles** — golden masters, property tests, contract tests, mutation testing so validation does not share the generator's blind spots. (5) **HITL at irreversible steps** — humans approve with evidence packs, not vibes. (6) **Feedback** — every caught hallucination becomes a block-list corpus entry and a regression test.

**Can it be fully eliminated?** No — and the honest answer earns the marks. Grounding and gates reduce the *rate* and, more importantly, shift hallucination from production to the sandbox, but retrieval can mis-rank, sensors only check what they encode, and oracles only cover specified behaviour. The engineering goal is therefore not elimination but **containment**: no hallucination reaches production *unverified and unowned*. The system-level conclusion — "the pipeline is deterministic even when the model is not" — is the course's central answer to this risk.

---

### E4. [SP-Q1] An IT department adopts a "Specification-Driven, Disposable Microservices" lifecycle: no refactoring, no comprehensive manual unit tests; instead they update design specs and have an AI agent regenerate services from scratch. (a) Evaluate this strategy in light of the "Inversion of Engineering Value". (b) Analyze the downstream maintainability risks. (6 marks)

**Concept tags:** Module 1/3; sample paper Q1; spec-as-source.

**Model answer:**

**(a) Evaluation against the Inversion of Engineering Value.** The inversion says: when code generation becomes nearly free, engineering value moves from *producing code* to *specifying intent and verifying outputs* — specifications become the source of truth and code a generated artifact (spec-as-source is the extreme end of the specification spectrum). The department has understood **half** of the inversion. They correctly relocate value to specifications (updating specs rather than hand-editing code is exactly the spec-as-source pattern, and regeneration-from-spec is rational *when* verification is strong). But they have misread the other half: the inversion transfers investment from writing code to **verification and governance** — it never licenses *removing* verification. Their declaration "no refactoring, no comprehensive unit tests" dismantles the deterministic checkpoints that make disposable code safe. Disposable artifacts are only disposable when a harness can prove each regeneration conforms to the spec (schema validation, exact-assertion tests, parity checks, bounded repair, human gates). Without that harness, "disposable" means "unverifiable and unreviewable at production speed". Verdict: the strategy is directionally consistent with the inversion but operationally self-defeating — it keeps the cheap half (generation) and deletes the valuable half (verification).

**(b) Downstream maintainability risks.** (1) **Spec completeness problem:** real services encode tacit knowledge — edge cases, performance workarounds, retry semantics, undocumented integrations — that no spec captures; each regeneration silently drops whatever the spec never stated, so behaviour *erodes* across regenerations. (2) **Loss of the safety net:** without comprehensive tests, regenerations cannot be proven behaviour-preserving; regressions surface in production rather than CI. (3) **Spec-to-code drift and orphans:** hotfixes applied directly to running services (they will happen under incident pressure) desynchronize deployed behaviour from spec, so the next regeneration overwrites emergency fixes. (4) **Institutional memory loss:** refactoring and reading code is how engineers learn a system; with no durable code to study and tests as documentation, onboarding and debugging rely entirely on spec quality. (5) **Hallucination exposure at scale:** every regeneration re-rolls the dice on dependency/API hallucinations with no test net; a "cheap" regeneration is actually a fresh supply-chain and correctness risk. (6) **Auditability and accountability gaps:** changing behaviour by regenerating without recorded verification weakens traceability (which spec version produced which deployed behaviour?) — a governance and compliance liability. (7) **Architectural drift:** regeneration decisions made agent-by-agent without ADR discipline accumulate inconsistent patterns.

**Remedy (one line to close):** the inversion done right = invest the refactoring/test hours into *spec quality + a deterministic verification harness + human gates*; keep regeneration, never surrender verification.

---

### E5. [N] "Deployment frequency is now meaningless as a metric." Critically evaluate this claim in the AI era. Which DORA metrics survive, which degrade, and what replaces them? (6 marks)

**Concept tags:** Module 8; DORA under AI — CS14 core analysis.

**Model answer:**

**The claim is half-right.** AI makes DORA metrics *substantially less informative alone*, but not meaningless — the correct response is pairing, not discarding. The root cause: DORA was designed for a world where humans wrote all the code; it measures delivery **outcomes** but is blind to **how AI shapes the code itself**. AI collapses the cost of producing code, not of verifying it, which decouples the metrics' assumptions from reality.

**Metric-by-metric analysis:**
- **Deployment Frequency — most degraded.** AI-generated boilerplate, scaffolding and micro-commits inflate counts ("vanity inflation"); a deploy shipping an authentication system counts the same as a README update. Not meaningless, but uninterpretable without **AI Code Share** (what fraction is AI-generated?) and **TrueThroughput** (complexity-adjusted) and **Innovation Rate** (features vs maintenance).
- **Lead Time for Changes — misleading.** Extremely fast commit-to-merge is offset by exploding PR review queues — the bottleneck silently moved to verification (the verification tax). Replacement/companion: **PR Cycle Time** that explicitly isolates review time, pricing the tax into capacity planning.
- **Change Failure Rate — partial value.** It catches production failures but misses the massive *hidden churn* — AI code quietly rewritten within weeks that never "failed" in production. Companion metrics: **Code Durability / Code Turnover Rate** (30/90-day windows, AI vs human) and **PR Revert Rate**; the 2025 DORA report's fifth metric, **rework rate**, targets exactly this blind spot.
- **MTTR — survives.** It measures how a team *responds* to failure, which remains honest regardless of who wrote the code. Keep it; pair it with AI-adoption context.

**Judgment to close:** "meaningless" overstates; "insufficient alone" is precise. The evidence that DORA alone misleads is concrete: post-AI, code churn roughly doubled while DORA dashboards stayed green, because rewritten code typically did not cause production failures. A truthful AI-era dashboard = DORA four + rework rate + AI Code Share + turnover/durability + complexity-adjusted throughput + review cycle time — measuring throughput, quality and verification load together instead of speed alone.

---

### E6. [N] Architectural drift accelerates under AI augmentation. Explain the causes, how to detect it, and how to prevent it. (6 marks)

**Concept tags:** Module 1/3/4; architectural drift.

**Model answer:**

**Definition:** the real structure of the system progressively diverges from its intended architecture — not through one bad decision, but through thousands of locally-sensible changes no one sees together.

**Why AI accelerates it (causes):** (1) **Volume without synthesis** — agents produce changes faster than any human reviews them *in architectural aggregate*; each diff is fine, the sum is not. (2) **LLM complexity bias ("architectural bloat")** — models default to maximal solutions: new microservices where a table would do, new libraries where a function would do. (3) **Context blindness** — an agent sees one task's context, never the whole architecture, so it re-invents patterns that already exist (duplicated utilities, parallel conventions). (4) **Weakened gatekeepers** — review queues under AI volume degrade into rubber stamps (Sample Q2), so the last human architectural checkpoint stops functioning. (5) **Spec-code drift** — code changed without spec updates makes the "intended architecture" a fiction.

**Detection:** (1) **architecture conformance checks** in CI — automated fitness functions (layering rules, dependency direction, allowed-dependency lists) fail builds that violate the intended structure; (2) **/speckit.analyze-style artifact consistency checks** and CI rules failing commits that touch architecture without spec updates; (3) **dependency-graph diffing** per release — new edges are reviewed as architectural events; (4) **duplicate-pattern detection** (AI finding AI's own duplications); (5) **periodic human architecture reviews** of trends, not individual PRs; (6) telemetry: rising cross-service call complexity and churn concentration as drift symptoms.

**Prevention:** (1) **constitution-style rules as feedforward** — simplicity mandates in the spec layer ("prefer single relational tables over new microservices") constrain every agent prompt; (2) **ADRs as organizational memory** fed to agents (adrkit pattern) so settled questions are not re-litigated per task; (3) **plan-as-gate** — treat plan artifacts as explicit PR gates with senior human review before task execution (blocks cascading architectural errors); (4) **bounded agent scope** — least-privilege file scopes so agents cannot sprawl across layers; (5) **balanced metrics** — track duplication, dependency counts, and complexity-adjusted throughput so drift is *measured*, not discovered at the rewrite.

**Closing line:** drift is a systems problem created by speed without synthesis — the cure is making the intended architecture machine-checkable (conformance gates + ADR context) so every agent change is verified against the whole, not just its task.

---

### E7. [N — Lab Sheet 2] What risks would arise if an LLM coding agent were allowed unlimited repair attempts against a failing sensor? Justify the bounded-loop design (max 2 attempts) and explain what still requires human review after all sensors pass. (6 marks)

**Concept tags:** Module 4/5/7; lab reflection Q5 + Q6 — classic lab-based analysis.

**Model answer:**

**Risks of unlimited attempts.** (1) **Reward-hacking the sensor:** with infinite tries the agent optimizes for *passing the checks*, not for correct behaviour — broadening assertions, deleting the failing path, or special-casing the test's exact inputs. A sensor is a proxy; unbounded optimization finds the proxy's loopholes (this is exactly why the harness forbids broad assertions like `status >= 400` — they are the sensor's known loophole). (2) **Repair-drift:** each repair consumes the previous output as context; error compounds — attempt 7 may "fix" the finding while removing valid behaviour (the rules explicitly demand repairs "without removing valid behavior", which only a bound makes enforceable). (3) **Cost runaway:** token and time costs scale with attempts while success probability does not. (4) **Masking real spec defects:** endless repairs can hide the truth that the *contract or rules are contradictory* — the correct response is human escalation, not attempt 9. (5) **Audit noise:** an unbounded log is undebuggable; a bounded log (2 attempts, findings recorded) is reviewable evidence.

**Why max 2 is justified:** two attempts preserve the useful signal — attempt 1 catches generation noise, attempt 2 catches findings-repair — while making failure *informative*: a twice-failing artifact is evidence of a spec/sensor/harness problem and must go to a human, which converts the bound from a limitation into an escalation trigger. The bound also enforces honesty: no opportunity to grind the sensor into submission.

**What still requires human review after all sensors pass (reflection Q6):** (1) **intent conformance** — sensors verify the contract's letter; only a human can judge whether the artifact serves the actual business intent and nothing extra; (2) **specification gaps** — behaviour the contract never stated (missing NFRs, unhandled edge cases) is invisible to every sensor by construction; (3) **security beyond patterns** — logic-level vulnerabilities and novel injection forms evade regex/pattern checks; (4) **the sensor itself** — a passing run proves conformance to the *current* rules; if the rules are wrong or incomplete, green is meaningless (human owns harness rules); (5) **accountability** — the recorded sign-off is what makes ownership real (Module 7): automation proposes, humans dispose.

**Closing line:** the bounded loop embodies "agents earn autonomy in proportion to verification determinism and action reversibility" — and the human after the sensors exists because sensors verify the spec, while humans own the intent.

---

### E8. [N] Production telemetry is proposed as direct input for fine-tuning the organization's coding agents. Critically analyse this: what could go wrong, and what design makes feedback safe? (6 marks)

**Concept tags:** Module 8; telemetry feedback; learning signals.

**Model answer:**

**What could go wrong (the critical half).** (1) **Security/confidentiality:** raw telemetry contains PII, credentials in logs, proprietary payloads; fine-tuning *bakes data into weights* — unlike prompts, you cannot easily revoke or permission weights, and extraction attacks can regurgitate memorized sensitive data. (2) **Wrong learning signals:** telemetry shows *what* happened, not *why* — training on "files touched during incidents are bad" teaches the model to avoid busy files (which are often core files); training on post-mortem hypotheses propagates unverified causes as facts. (3) **Feedback loops and drift:** the model's own past outputs are in production; training on that output risks degenerate self-reinforcement (the model learns its own failures as style). (4) **Correlation-causation confusion at scale:** incident-correlated patterns are confounded by process (hot files get more review, therefore more findings). (5) **Compliance:** privacy regulation constrains repurposing operational data for training; consent and purpose-limitation apply.

**The safe design (the constructive half).** (1) **Curate, don't dump:** a human curator transforms raw telemetry into *verified lessons* — sanitized incident summaries, confirmed root causes, explicit "must-not" rules — before anything enters a corpus. (2) **Prefer RAG to fine-tuning:** retrieval gives currency, revocability (delete the document), permissioning per team, and cited provenance; fine-tune only on stable, non-sensitive, human-approved style knowledge, if at all. (3) **Sanitization pipeline:** PII/secret scrubbing and aggregation before any storage; private/on-prem endpoints for sensitive corpora; audit of what entered which context. (4) **Separate signals:** keep *hypotheses* out of training data; version the knowledge corpus so a bad lesson can be rolled back like code. (5) **Measure the loop:** track escape rate and repeat-incident rate after each feedback wave — the loop is justified by falling recurrence, not by "more data trained".

**Closing judgment:** telemetry is high-value, high-risk fuel: as curated, permissioned, revocable RAG context it accelerates learning; as raw fine-tuning input it converts operational data into permanent, unauditable model behaviour — the exact opposite of accountable engineering.

---

## CATEGORY F — CASE-STUDY BASED

---

### F1. [YB-Q6] An enterprise wants to modernize a legacy monolithic application into microservices using autonomous coding agents. Design a safe migration workflow covering dependency identification, generation/refactoring, prompt and context guardrails, API compatibility, version/dependency management, deterministic verification, and human approval. (10 marks)

**Concept tags:** Module 4; legacy modernization — the flagship case study. *(Workflow diagram version = D6; this answer adds management depth.)*

**Model answer:**

**Stage 1 — Inventory and dependency identification.** AI builds the call graph, data-coupling map and shared-database inventory; humans validate it (AI maps are hypotheses). Identify seams = bounded contexts with low coupling. Classify risk per candidate service (traffic, data sensitivity, change frequency). Output: a migration backlog ordered by risk/value — start low-risk, high-value.

**Stage 2 — Pin current behaviour.** Write characterization/golden-master tests on the monolith before any change — including its accidental behaviours, because downstream consumers may depend on them. Without this stage, "migration" cannot be distinguished from "silent re-implementation"; parity checking later depends on it.

**Stage 3 — Contract-first boundaries.** For the chosen seam, author the OpenAPI contract before code: endpoints, schemas, error responses. The contract becomes (a) feedforward input to the agent, (b) the API-compatibility oracle in CI, (c) the consumer-facing commitment. Compatibility policy: additive changes only; versioned endpoints; deprecation windows; no breaking schema changes without a consumer-migration plan.

**Stage 4 — Guarded generation by agents.** The agent receives exactly: the service's contract, the relevant legacy module extracts, the style guide, harness rules (forbid new unpinned dependencies, forbid secrets, exact assertions, JSON-only outputs), and the generation task. It never receives the whole monolith — both for context limits and legacy-context-loss reasons. Attempts are bounded (2) with findings-only repair prompts and an audit log per attempt.

**Stage 5 — Deterministic verification.** Sensor stack: schema validation of agent output → static analysis/security scan → exact-assertion unit tests → contract tests against the OpenAPI document → **parity checks**: identical golden requests against monolith path and new service, diffing responses (respecting intended behaviour changes). All deterministic; identical verdicts every run.

**Stage 6 — Version and dependency management.** Lockfiles and pinned versions per service; semantic versioning; the service deploys independently; dependency upgrades arrive as reviewed PRs from the SCA pipeline, never agent-unpinned. Feature-flag/strangler-fig routing controls traffic: 0% → 5% canary → ramp, with instant flag-flip back to the monolith as the rollback lever.

**Stage 7 — Human approval gates.** Senior review per extraction with an evidence pack: sensor reports, parity results, contract diff, rollback plan. Production rollout of each service is a human decision; agents may do everything up to the production edge (staging deploys, canary analysis) but never the final flip.

**Stage 8 — Iterate and retire.** Repeat per seam; monitor cross-service telemetry (the new failure surface: network, latency, partial failure); after cutover confidence, retire the monolith slice. Record every extraction as an ADR so the architecture's evolution stays explainable.

**Why the design is safe (the justification line examiners want):** behaviour is pinned before change; every change is verified deterministically before human eyes; every step is reversible until approved; and the agents' autonomy is bounded by contracts, rules and attempt caps — autonomy proportional to verification, exactly as the course prescribes.

---

### F2. [N — Lab Sheet 2] Using the coding-agent harness case (OpenAPI contract → agent → sensors → bounded repair), explain: (a) how feedforward rules constrained the code before generation; (b) why checking for an assertion in source differs from executing the test; (c) why sensor findings must be specific and machine-readable; (d) how the audit trail supports governance. (8 marks)

**Concept tags:** Lab Sheet 2; harness concepts — expect lab-derived questions.

**Model answer:**

**(a) Feedforward constraints.** Before any generation, four files defined the solution space: `openapi.yaml` as the single source of truth (exact endpoints, status codes 201/400, response shapes); `style-guide.md` (CommonJS, no external dependencies, no hardcoded secrets, exact assertions in tests, JSON content-type); `harness-rules.md` (exactly two artifacts, exact assertion syntax mandated, broad assertions forbidden, JSON-only response with keys `appJs`/`testJs`, no Markdown fences, max two attempts); `generation-task.md` (the task plus required output schema). Effect: the model never *chooses* conventions — style, security and verification requirements are fixed inputs, so output variance collapses to the task itself, and forbidden failure classes (secrets, dependency injection, broad assertions) are excluded before generation rather than cleaned after.

**(b) Static check vs dynamic execution.** Checking for `assert.strictEqual(response.status, 400)` *in the source* proves the test *targets the specified behaviour* — it is a conformance check on the test as an artifact, deterministic and instant. *Executing* the test proves the *code actually satisfies that behaviour* — it is a truth check on the implementation. Neither subsumes the other: without the static check, a broken test file or broad assertion could pass execution while proving nothing; without execution, a perfectly-written assertion could sit over failing code. The deliberate lab defect made this visible: replacing the exact 400 assertion with `assert.ok(response.status >= 400)` produced `MISSING_400_ASSERTION` from the static sensor and *skipped* the dynamic stage — the harness treats a weakened oracle as a defect even before running anything. Verification therefore needs both: presence-of-correct-oracle (static) and oracle-passes (dynamic).

**(c) Specific, machine-readable findings.** The sensor emits `{code, message}` pairs (`MISSING_400_ASSERTION`, `SERVER_NOT_EXPORTED`, `JSON_HEADER_NOT_VISIBLE`). Because they are structured, the repair prompt can carry *only the findings* — no conversation history, no vague prose — so repairs are targeted and cheap; findings can be asserted upon (the harness itself can test that a repair addressed each finding); and they are auditable data, comparable across attempts. Vague findings ("the test seems weak") would force the model to guess, inviting reward-hacking edits.

**(d) Audit trail and governance.** Every attempt is logged (`attempt-log.json` / `manual-attempt-log.json`): mode, feedforward files, per-attempt findings, pass/fail, repair finding, final status, and the human-review note. Governance value: (1) **accountability** — the recorded `humanReview` field evidences Module 7's ownership chain; (2) **traceability** — which spec version and rules produced which artifact; (3) **improvement data** — recurring finding codes reveal weak spots in prompts/rules; (4) **bounded-autonomy proof** — the log shows the loop never exceeded two attempts, evidencing that autonomy limits were enforced, not merely intended.

---

### F3. [N] Case: a payments team must let coding agents modify a card-processing module under PCI-DSS-style regulation. Design the governance plan. (8 marks)

**Concept tags:** Modules 3/5/6/7 integration — compliance-heavy scenario.

**Model answer:**

**Frame:** regulated, high-blast-radius domain — the design principle is *minimal autonomy, maximal verification, recorded accountability*.

**1. Scope and classification.** Tag the card-processing module as high-risk in the platform (policy-as-code): every change touching it automatically receives the strict gate set, regardless of author (human or agent).

**2. Specification and guardrails.** Contract-first (OpenAPI) with compliance rules compiled into "must-not" spec clauses (never log PAN data, never store secrets, TLS everywhere). Feedforward files for agents: contract + compliance rule pack + style guide + harness rules (no new dependencies, no secrets in context). RAG corpus is permissioned: agents retrieve compliance rules but never raw cardholder data; no production data ever enters prompts (Module 8's sanitization rule).

**3. Verification (deterministic, deepened for the domain).** Schema validation → static security analysis → SCA on every dependency → exact-assertion unit tests → contract tests → penetration-style pattern tests for the specific threats (injection, log-leakage) → parity checks against the current behaviour. AI-judged quality checks are advisory only; every blocking gate is deterministic (Module 7's non-deterministic-validation rule).

**4. HITL gates.** Merge requires senior reviewer + security champion sign-off (evidence pack: diff, sensor reports, SCA results); release of the module requires change-advisory approval with rollback plan; data-schema changes are human-only, always. Every sign-off recorded (who/what/when) — this is the audit evidence regulators ask for.

**5. Deployment controls.** Blue-green/canary with automatic abort on SLO breach (pre-authorized rollback); no direct-to-production agent path; deployment config itself generated declaratively and policy-checked (IaC gates) so agents cannot widen exposure.

**6. Monitoring and feedback.** Telemetry watches security events (auth failures, unusual payloads); incidents feed sanitized learnings back into the compliance RAG corpus and regression tests; escape-rate tracked per gate.

**7. Accountability.** RACI documented; agent runs logged (prompts, outputs, findings, attempt counts); the org owns liability; negligence is defined as bypassing any mandated gate.

**Closing judgment:** in regulated domains the agents' speed is still usable — but only because every autonomy grant is bounded by contracts, deterministic gates, human sign-offs and an audit trail that can reconstruct any change end-to-end.

---

### F4. [N] Case: a 10-year-old Java monolith has no automated tests. The team must modernize it with AI assistance within a quarter. Plan the first 90 days. (8 marks)

**Concept tags:** Module 4; pragmatic modernization plan.

**Model answer:**

**Days 1–15 — Understand and pin.** (1) AI-assisted code archaeology: dependency/call-graph maps, module clustering, hotspot analysis (churn + complexity) — AI accelerates mapping, humans validate. (2) Build **characterization tests** on the highest-traffic paths first: capture current outputs as golden masters (including bugs, intentionally) — this creates the safety net that makes every later AI change verifiable. (3) Stand up CI with static gates (lint, build, secret scan) and lock/pin the dependency tree.

**Days 16–30 — Guardrails and pilot.** (4) Author the feedforward pack: architecture constitution (simplicity rules), style guide, harness rules (exact assertions, no unpinned dependencies, no secrets), and a RAG corpus of the system's real interfaces (schema dumps, contracts) to combat hallucination. (5) Pilot the harness loop on one low-risk module: agent generates refactoring/tests → deterministic sensors → bounded repair (2 attempts) → human review. Measure: sensor pass rate, review time, defect escapes. This pilot *is* the verification-infrastructure investment — do not scale licences before it works.

**Days 31–60 — Extract first seams.** (6) Choose 1–2 low-risk, high-value seams; contract-first (define OpenAPI per extraction); agents generate the service under harness guardrails; parity checks compare old/new on golden requests; strangler-fig routing with canary traffic shift; human gate per extraction. (7) In parallel, AI-assisted smell remediation *inside* the monolith for paths that stay — always test-before-refactor.

**Days 61–90 — Scale what worked.** (8) Repeat the extraction pattern on the next seams using the audited, tuned harness; (9) establish the metrics baseline for the AI era (turnover, escape rate, review turnaround, MTTR) so speed claims are evaluable; (10) retrospection: update guardrail rules from every caught hallucination (block-list corpus), decide agent-vs-pair-programming per task class from pilot evidence.

**Risks to name (shows judgment):** J-curve dip is expected — plan capacity for the verification tax; legacy context loss means agents need explicit reference files, never memory; never let agents touch data migrations without human-only gates; keep every extraction reversible via routing flags.

**Closing line:** the quarter's real deliverable is not the migrated services — it is the *verified migration capability* (tests, harness, guardrails, gates) that makes all future AI-assisted change safe.

---

### F5. [N] Case: a startup proposes "spec-as-source" development — humans write only specs, agents write all code, no human reads code at all. Evaluate feasibility and define the minimal guardrail set you would demand. (6 marks)

**Concept tags:** Module 3; specification spectrum extreme; risk judgment.

**Model answer:**

**Feasibility — bounded yes, unconditionally no.** Spec-as-source is legitimate at the spectrum's extreme *when two conditions hold*: (1) the domain is fully expressible as machine-checkable specifications (APIs, infrastructure, CRUD services — where OpenAPI/JSON-schema/IaC give complete observable contracts), and (2) a deterministic verification harness exists that can prove conformance end-to-end (schema, static, exact-assertion, contract, parity tests — the lab harness generalized). The startup's claim "no human reads code" can therefore be *operationally* true for narrow, contract-heavy domains — but "no human reads code" must not become "no human verifies anything", which is the failure mode Sample Q1 punishes.

**Where it breaks (name the domains to show judgment):** tacit performance behaviour, security-sensitive logic beyond pattern checks, regulatory obligations not expressible as tests, UX judgment, and novel integration behaviour — all exceed what specs capture, and every unexpressed behaviour is silently regenerated-away at each spec change (the spec-completeness problem).

**Minimal guardrail set (the demanded list):** (1) **constitution-level simplicity and dependency rules** as feedforward (no unpinned dependencies; allow-listed packages; no secrets); (2) **deterministic sensor suite** before anything merges: schema validation of agent output, static analysis + symbol resolution (kills API hallucinations), exact-assertion tests derived from the spec (broad assertions banned), contract tests, parity/golden-master tests where legacy behaviour exists; (3) **bounded repair loops** (2 attempts) with machine-readable findings and audit logs; (4) **independent oracles** — tests generated in spec-only context, property invariants, mutation testing — to break circular validation; (5) **HITL at the edges humans can't automate**: acceptance-criteria approval, security review for trust-boundary code, release approval, and *spec review itself* — in spec-as-source, the spec is the entire system, so spec review *is* code review; (6) **drift detection** — CI fails any behaviour change without a spec change, and periodic "regeneration diff audits" sample regenerated output for human spot-review; (7) **recorded accountability** — every artifact traceable to spec version + verification run + approver.

**Closing judgment:** approve the model for their contract-heavy core; refuse it for the money paths and trust boundaries until the sensor suite provably covers those classes. Feasibility is not a boolean — it is a function of how much of the system the spec plus sensors can *actually see*.

---

### F6. [N] Case: six months after AI adoption, an org finds code churn has doubled — much AI code is rewritten within 30 days — while DORA dashboards show no change. Diagnose root causes and prescribe remedies. (6 marks)

**Concept tags:** Module 8; churn/durability diagnosis — CS14 scenario.

**Model answer:**

**Why the dashboards stayed green:** the four DORA metrics measure delivery outcomes; rewritten-but-never-failing code is invisible to them. Change Failure Rate only catches production failures; code quietly rewritten within weeks never "fails" — it churns. That is precisely why 2025 DORA added **rework rate** and why AI-era frameworks add **Code Turnover Rate** (30/90-day windows, segmented AI vs human).

**Root-cause hypotheses (diagnosis before remedy):** (1) **Spec/understanding gap** — agents produced plausible code that missed intent, discovered late and rewritten (tests passed, intent didn't); (2) **context starvation** — agents lacked repo conventions/contracts, so style and structure mismatch forced rewrites; (3) **hallucination cleanup** — invented APIs or shallow edge-case handling caught in later development, not review; (4) **verification tax shortage** — reviews rubber-stamped, defects surfaced downstream as rework (Sample Q2 pattern); (5) **acceptance-without-reading** — developers accepted suggestions uncritically (pair-programming bias); (6) **task mismatch** — agents used for work needing design judgment (their churn is naturally higher); (7) **metric-gaming possibility** — throughput pressure encouraging fast AI output regardless of durability.

**Remedies mapped to causes:** (1) strengthen feedforward — contracts, style guides, RAG grounding per task; (2) make review *effective* under volume — evidence packs, risk-adaptive gates, senior review for high-churn modules (predictive bug detection routes them); (3) enforce exact-assertion, spec-derived tests + independent oracles so intent errors surface *before* merge; (4) instrument 30/90-day turnover by origin as the early-warning metric with a review trigger threshold; (5) retrain interaction patterns — pair-programming for design-heavy tasks, agents for verifiable batch tasks (C1's rule); (6) management signal: judge on complexity-adjusted throughput *and* durability together; a CAT score without turnover is vanity.

**Closing line:** doubled churn is the system reporting that generation outran understanding — the cure is feedforward grounding plus verification capacity, measured by durability metrics that DORA alone cannot see.

---

### F7. [N] Case: a 12-person team adopts GitHub Spec Kit for all features. Lay out the command workflow they should follow, and the two risks most likely to bite them with mitigations. (5 marks)

**Concept tags:** Module 3; SDD tooling — CS9 slides 12–14.

**Model answer:**

**Workflow (order matters — memorize):** `/speckit.constitution` (project principles: simplicity rules, allowed dependencies — once, at project start) → `/speckit.specify` (feature spec before technical design) → `/speckit.clarify` (resolve ambiguities — do not skip; ambiguity here becomes cascading error later) → `/speckit.plan` (technical plan once requirements are clear) → **human plan review as an explicit gate** → `/speckit.checklist` (quality checklist before implementation) → `/speckit.tasks` (actionable tasks) → `/speckit.analyze` (artifact consistency check before implementation and after significant changes) → `/speckit.implement` (execute tasks under CI guards) → `/speckit.converge` (identify remaining gaps during/after implementation).

**The two risks most likely to bite a small team (from the CS9 risk table):**
1. **Cascading errors from an unchecked /plan → /tasks transition.** A twelve-person team has no dedicated architect; a flawed plan flows into tasks and multiplies silently across artifacts. *Mitigation:* treat `plan.md` as a formal PR gate — mandatory senior review before `/tasks` is ever run; add `/speckit.analyze` after every significant change.
2. **Spec-to-code drift.** Developers hotfix code directly under delivery pressure; specs stop matching behaviour; the next spec-driven regeneration overwrites the hotfix. *Mitigation:* CI checks that fail any commit touching behaviour without the corresponding spec update; hotfixes must land as spec amendments (even small ones), keeping the spec the single source of truth.

*(Mention-worthy second tier if asked for more:* architectural bloat — countered by constitution simplicity rules like "prefer single relational tables over new microservices"; and legacy context loss — countered by attaching schema dumps/OpenAPI contracts to `.specify/` context during `/plan`.*)*

**Closing line:** for a small team, Spec Kit's value is *discipline*: the commands encode the review points that AI velocity would otherwise erode.

---

## Answer-mapping index (your original 16 + the 2 sample questions)

| Your bank | This bank | Your bank | This bank |
|---|---|---|---|
| Q1 SDLC evolution | B1 | Q9 governance/accountability | E2 |
| Q2 non-determinism | B4 (+A4) | Q10 feedback loops | D3 (+E8) |
| Q3 requirements | B2 (+D7) | Q11 measurement | B5 |
| Q4 architecture/ADR | B3 | Q12 economics | B6 |
| Q5 pair vs agents | C1 | Q13 foundation models | A2, A3, A4 |
| Q6 legacy migration | F1 (+D6) | Q14 prompting | A5 |
| Q7 circular validation | E1 (+C7) | Q15 agentic workflows | A6 |
| Q8 CI/CD | D2 | Q16 code review | B7 (+C5) |
| Sample Q1 disposable microservices | E4 | Sample Q2 PR throughput | B8 |
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
