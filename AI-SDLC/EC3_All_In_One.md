# EC3 Revision Notes — AI-Augmented SDLC (SEZG534)

**Scope (strict):** These notes are built **only** from the course files you supplied for sessions 8–14:

| Source file | What it covers | Where it lands in the notes |
|---|---|---|
| `Merged_Presentation_2026-11-15.pptx` | CS8 AI-Native SDLC, CS9 Spec-Driven Development, CS10 Design & Architecture, CS11 Coding (two decks), CS13 Software Testing, CS14 Software Delivery Metrics | `01_Concept_Notes.md` Parts I–VI |
| `Spec_Driven_Development_GitHub_Spec_Kit_Lab_Sheet.pdf` (Lab 1) | SDD hands-on with GitHub Spec Kit in VS Code | `01_Concept_Notes.md` Part VII |
| `Coding Agent_Harness_Lab_Sheet.pdf` (Lab 2) | Feedforward/feedback harness for coding agents | `01_Concept_Notes.md` Part VII + question bank case studies |
| Your 16-question analysis | Question **pattern** only (6 categories) | `02_Question_Bank.md` — every question re-grounded in the files above |

Nothing from sessions 1–7 (SDLC evolution history, AI/ML fundamentals, prompting theory, RAG internals, DevOps basics, etc.) is included. Where a sample question touched an out-of-scope topic (e.g., "foundation models and transformer architecture"), the question was **rewritten to the same pattern using only in-scope content** (e.g., non-determinism and context windows), and the mapping index at the end of the question bank shows exactly what happened to each of your 16 questions.

## Files

| File | Use it for |
|---|---|
| `01_Concept_Notes.md` | First pass. 7 parts, every concept with a definition, mechanism, and an **Exam skeleton** you must be able to reproduce from memory. |
| `02_Question_Bank.md` | Second pass. 45 questions in 6 categories (A–F), each with a full model answer. Write answers by hand before reading the model. |
| `03_Last_Day_Cheat_Sheet.md` | Day-before and morning-of. One-liners, mnemonics, formulas, number anchors, dispatch table. |
| `EC3_All_In_One.md` | Single file with everything, for search/phone. Do not study from it linearly. |

## Closed-book strategy (even though it is nominally open-book)

1. **Skeleton-first, not prose-first.** Every concept ends with a 5–8 line skeleton. If you can reproduce the skeleton from memory, you can rebuild the paragraph around it in the exam. Prose is decoration; the skeleton is the answer.
2. **Active recall loops.** For each part: read once → close notes → write the skeleton → check → mark misses. A miss gets a red star; red-starred skeletons are your day-before list.
3. **Question-bank discipline.** Do not read model answers first. Attempt, then compare, then note which skeleton lines you missed. 45 questions × ~10 minutes each ≈ 8 sessions.
4. **Anchor on numbers and named artifacts.** Examiners reward specifics: "15–20% churn acceptable", "max 2 repair attempts", "Wave 1 = no dependencies", "`requirements.md / design.md / tasks.md`". The cheat sheet collects every number once.
5. **Flows as text arrows.** Every workflow answer must be written as `A → B → C` first, then explained line by line. Practise redrawing the six master flows: bottleneck shift, Spec Kit command chain, harness loop (Lab 2), cybernetic loop (SEI), circular-vs-independent validation, operations feedback loop.

## Ten-day plan (≈ 3 h/day; compress to 6 days at 5 h if needed)

| Day | Work | Output you must be able to do |
|---|---|---|
| 1 | Part I–II (CS8+CS9) | Draw spec spectrum; list 9 Spec Kit commands in order; 4 SDD risks + mitigations |
| 2 | Part III (CS10) | Deep vs shallow, sink vs pipe, harness vs orchestrator, 3-layer governance — from memory |
| 3 | Part IV first half (CS11: patterns, guardrails, Lattice) | 5 patterns + failure modes; 4 guardrail types; I-C-O layers |
| 4 | Part IV second half (CS11: SEI, non-determinism, review) | Cybernetic loop; actual vs apparent; P-E-C-E → E-C-E; 8 review steps |
| 5 | Part V (CS13) | 6 testing uses; churn & complexity numbers; circular validation remedies; agent-testing triad |
| 6 | Part VI (CS14) | DORA 4 + status under AI; hierarchy A-E-P-Q-B; 5 pillars; ROI + J-curve formulas |
| 7 | Part VII (Labs) | Lab 1 artifact chain; Lab 2 feedforward files, sensor order, bounded repair; verification matrix |
| 8 | Question bank categories A, B (22 Qs) | Attempt 11, mark misses |
| 9 | Question bank categories C, D, E, F (23 Qs) | Attempt all; redraw every flow by hand |
| 10 | Cheat sheet + red-star skeletons + 5 hardest questions re-attempted | Recite all 50 one-liners |

## Question-category → study-section map

| Category (your pattern) | Weight in bank | Primary sections |
|---|---|---|
| A. Conceptual / descriptive | 12 Qs | Parts I–VII (all skeletons) |
| B. Scenario-based application | 10 Qs | Part I–II (requirements), IV (guardrails/patterns), V (testing), VI (metrics), Lab 2 |
| C. Compare and justify | 7 Qs | Spec spectrum (II), deep/shallow + sink/pipe + harness/orchestrator (III), testing comparisons (V), frameworks (VI) |
| D. Diagram / workflow | 6 Qs | Bottleneck shift (I), Spec Kit chain (II), validation loop (V), harness loop (VII) |
| E. Critical analysis | 5 Qs | Governance (III–IV), SDD critique (II), DORA critique + J-curve (VI) |
| F. Case-study / lab | 5 Qs | Lab 1 (VII), Lab 2 (VII), SEI case (IV), Uber case (V) |

## Exam answer template (works for every category)

1. **Define** (1–2 lines, exact terms from the notes).
2. **Flow** (the `A → B → C` arrow line for whatever is asked).
3. **Mechanism** (3–5 numbered points, one named artifact or number each).
4. **Risk / trade-off** (2–3 failure modes with the named guardrail or metric that controls them).
5. **Verdict / recommendation** (one sentence that answers the exact verb in the question: recommend, justify, critique).

If a question says "compare", write a two-column mini-table in the answer script; if it says "design", the flow line comes before any prose.


---

# 01 — Concept Notes

> Every section ends with an **Exam skeleton** — the minimum you must reproduce from memory. Sources are tagged [CS8], [CS9], [CS10], [CS11], [CS13], [CS14], [Lab1], [Lab2] against the merged deck and lab sheets.

---

# PART I — CS8: AI-Native SDLC

## 1.1 Why the traditional SDLC breaks when AI writes the code

The traditional SDLC is **process-heavy**: each step exists to guarantee accountability and control when humans do all the work. AI does not speed every stage equally — it collapses only the **build** stage to agent speed.

```
Traditional:   Plan ──► Build ──► Review/Test ──► Deploy     (all at human speed)
AI-era:        Plan ──► BUILD ──► Review/Test ──► Deploy
                 ▲        ▲            ▲              ▲
             human speed  agent speed  human speed    human speed
             (new bottleneck moves HERE)
```

Four consequences once build collapses:

1. **Bottleneck moves** to the stages on either side of build — mainly **plan, review/test, and deploy** — which still run at human speed.
2. **Controls stop matching reality** and become intractable. Line-by-line human review made sense when a person wrote the code; it cannot keep up once **agents write most of the diff**.
3. **Governance costs rise** because exceptions still route through meetings and committees that meet **weekly or monthly** — far slower than agent-speed iteration.
4. Result: the fastest part of the pipeline is gated by its slowest human parts.

**Exam skeleton [CS8]**
- Traditional SDLC = process-heavy for accountability + control
- Build collapses to agent speed; plan / review-test / deploy stay human speed
- Bottleneck moves to the stages AROUND build
- Hand line-by-line review can't scale to agent-written diffs
- Governance costs rise: weekly/monthly committee cadence vs agent-speed iteration

## 1.2 The AI-native SDLC: automated handover

The AI-native SDLC answers the manual-handoff problem: it promotes **automated handover and triggering of subsequent plays** — when one phase finishes, the next phase starts automatically from the artifact the previous phase wrote, instead of waiting for a human to carry context across a meeting.

Contrast: traditional SDLC fixes hand-offs with humans and process; AI-native SDLC fixes them with artifacts + automation, so the whole chain can eventually run at machine speed except where humans deliberately gate it.

**Exam skeleton [CS8]**
- AI-native SDLC = automated handover + triggering of subsequent plays
- Replaces manual handoff between phases
- Handover happens via artifacts in version control (see 1.4 playbook)

## 1.3 Spec-Driven Development (SDD): definition and benefits

**Problem SDD solves:** teams often ship software that *works* but still *misses the original intent*. **Prompt-first** workflows are fine for simple tasks but struggle as **scope and complexity increase**.

**Definition:** SDD is a **spec-first approach** — teams define **common guardrails, requirements, constraints, acceptance criteria, and edge cases up front**, then use AI to generate code, tests, and supporting artifacts **from that shared context**. The spec becomes the **connective tissue across the lifecycle**.

**Four benefits (memorise as A-A-F-P):**
1. **A**mbiguity down — less ambiguity and rework because requirements are clarified earlier.
2. **A**lignment up — better alignment across product, engineering, and test through a **shared source of truth**.
3. **F**aster implementation — AI can generate against structured context instead of guessing.
4. **P**redictable delivery — validation is tied back to the original intent.

**Exam skeleton [CS8]**
- Prompt-first OK for simple tasks; breaks as scope/complexity grow; works-but-misses-intent
- SDD = guardrails + requirements + constraints + acceptance criteria + edge cases UP FRONT
- AI generates code/tests/artifacts FROM shared context
- Spec = connective tissue across lifecycle
- Benefits: less ambiguity, alignment via shared truth, faster generation, predictable delivery

## 1.4 SDD tooling: Spec Kit, Kiro, AI-Native SDLC Playbook

The ideas only matter if teams can apply them consistently day to day. Three tools:

**(a) GitHub Spec Kit** (open-source, created by Microsoft). Two key components:
1. **Specify CLI** — bootstraps a project for SDD: downloads official templates for the coding agent/platform of your choice and sets up SDD scaffolding the agent can iterate on.
2. **A set of templates + helper scripts** — defines what a spec looks like, what a technical plan encompasses, and how work breaks into tasks an AI agent can pick up and execute.

**(b) Kiro** (agentic AI by AWS; desktop IDE, CLI, web app, mobile app, personal AI agent "Crew").
- Workflow: **Requirements → Design → Tasks**.
- Every spec generates three files:
  - `requirements.md` (or `bugfix.md`) — user stories, acceptance criteria, or bug analysis in structured notation
  - `design.md` — technical architecture, sequence diagrams, implementation considerations
  - `tasks.md` — detailed implementation plan with discrete, trackable tasks
- **Dependency graph → waves:** Wave 1 = all tasks with no dependencies (run **concurrently**); Wave 2 = tasks whose dependencies were satisfied by Wave 1; ... Wave N. **Waves execute sequentially; tasks within a wave execute concurrently.**
- Two spec types: **Feature Specs** (new features) and **Bugfix Specs** (systematic diagnosis, surgical fixes, regression prevention).
- **Steering files:** Kiro scans the repository and generates persistent context files — `product.md`, `tech.md`, `structure.md` — so the AI permanently knows project conventions without repeated chat explanations.

**(c) The AI-Native SDLC Playbook** (Anthropic). Plays grouped into **six non-linear stages: Plan, Design, Build, Test, Deploy, Maintain**. Each play covers: what changes; getting started; concrete implementation steps; governance considerations; how to measure whether it worked.
- Each stage **ends by writing an artifact to version control**; the next stage **begins by reading it**.
- Early stages: `.md` files dominate (a product owner and an agent can both read and act on the same file). **From Build onward**, the artifact is code and its records.
- **The chain of commits is the audit trail**: who asked for what, what the agent produced, who approved it.

**Exam skeleton [CS8]**
- Spec Kit = Specify CLI + templates/scripts (Microsoft, open-source)
- Kiro: Requirements → Design → Tasks; files requirements.md / design.md / tasks.md
- Kiro waves: within = concurrent, across = sequential; Feature vs Bugfix specs; steering = product/tech/structure.md
- Playbook: 6 stages P-D-B-T-D-M; stage ends writing VC, next starts reading it
- Early stages = .md artifacts; Build onward = code; commit chain = audit trail

## 1.5 The Specification Spectrum (SDD approaches)

Three positions on how much authority the spec keeps over time [CS8 + CS9, arXiv 2602.00180]:

```
Spec-First ──────────► Spec-Anchored ──────────► Spec-as-Source
(guide once)           (kept in sync)             (only artifact humans edit)
Human edits: code      Human edits: spec + code   Human edits: spec ONLY
```

| Mode | What happens | After code exists |
|---|---|---|
| **Spec-first** | Spec written **before coding** to guide initial implementation | Spec **may or may not** be maintained; value was initial clarity |
| **Spec-anchored** | Spec **maintained alongside code** for the whole lifecycle; behavior changes require updating **both** spec and code to stay synchronized | Spec stays alive |
| **Spec-as-source** | Spec is the **only artifact humans edit directly**; code is **entirely generated** from the spec and never manually modified; any behavior change = change spec + regenerate | Code is derivative |

**Exam skeleton [CS8/CS9]**
- Spectrum: Spec-First → Spec-Anchored → Spec-as-Source
- First: guide initial build; may lapse. Anchored: spec+code updated together. As-source: human edits only spec; code regenerated
- Human control over code decreases left → right

## 1.6 Challenges with SDD tools

1. **Brownfield problem** — harder to evaluate SDD usefulness for existing (brownfield) codebases; the tools shine on greenfield.
2. **False sense of control** — having a spec feels like governance, but the spec may not match what the agent actually did.
3. **Reviewing markdown instead of code** — teams end up reviewing documents over reviewing the actual code the agent wrote.

**Exam skeleton [CS8]**
- Brownfield hard to evaluate; false sense of control; markdown reviewed instead of code

## 1.7 Requirements elicitation: still the least automated stage

**Definition:** requirements elicitation = the process of **discovering, understanding and documenting what stakeholders need** from a proposed system. It remains **one of the least automated stages** of software delivery.

**Common elicitation techniques (7):** stakeholder interviews; workshops; observation; document analysis; surveys; process mapping; prototyping.

**Exam skeleton [CS8]**
- Elicitation = discover + understand + document stakeholder needs
- Least automated stage of delivery
- 7 techniques: interviews, workshops, observation, document analysis, surveys, process mapping, prototyping

---

# PART II — CS9: Spec-Driven Development (deep dive)

## 2.1 The requirements lifecycle: elicitation → synthesis → management

**Phase 1 — Elicitation** (discover needs). Agile flavour: validate assumptions quickly with **tangible software** rather than predicting all needs up front. Agile techniques:
- **User Story Mapping** — visual mapping of user journeys to find core workflows, MVP, release slices.
- **Sprint Reviews & Demos** — direct feedback from end-users/stakeholders on working increments each sprint.
- **Spikes & Prototyping** — short technical/design investigations that clarify ambiguous requirements before committing implementation effort.

**Phase 2 — Synthesis** (convert raw elicitation data into structured, actionable artifacts for dev + test teams).
- Traditional: **BRD** (Business Requirements Document) & **SRS** (Software Requirements Specification) — comprehensive, highly detailed, often hundreds of pages, define the "what" and "how".
- Agile:
  - **User Stories** — short-form, end-user perspective: *"As a [role], I want [action], So that [benefit]."*
  - **Acceptance Criteria** — conditions for story completion, frequently in **BDD** form: *Given [context], When [action], Then [outcome].*
  - **Themes & Epics** — high-level abstractions for large bodies of work, progressively **decomposed** into smaller stories.

**Phase 3 — Management** (track, prioritize, control changes across the lifecycle).
- **Prioritization** by customer value, dependencies, risk; Agile teams classify with **MoSCoW** (Must / Should / Could / Won't have).
- **Product Backlog** — a single ordered list, continuously re-prioritized by the **Product Owner**.
- Fixed sprint duration and team capacity; **scope is the variable** — items move in/out of the backlog.

**Exam skeleton [CS9]**
- Elicitation (Agile: story mapping, sprint reviews/demos, spikes/prototypes)
- Synthesis (BRD/SRS vs user stories "As a…I want…So that"; BDD Given/When/Then; themes→epics→stories)
- Management (value/dependency/risk; MoSCoW; backlog owned by PO; time fixed, scope flexes)

## 2.2 Where AI plugs into each requirements phase

**AI-driven elicitation** — turns manual discovery into a **continuous, data-driven engine** surfacing explicit AND implicit needs:
1. **Implicit Need & Telemetry Analysis** — AI ingests unstructured data (tickets, logs, transcripts, app reviews) to surface implicit needs.
2. **Synthetic User Personas & Autonomous Interviewing** — simulated personas interviewed at scale.
3. **Instant Functional Prototyping** — requirements validated as working prototypes immediately.

**AI-driven synthesis** — an intelligent authoring partner that automates formal drafting, detects ambiguity, and ensures standards compliance:
1. **Automated Artifact Conversion** — raw discussion → structured artifacts.
2. **Real-Time Ambiguity & Quality Scoring** — models pre-trained on standards such as **INCOSE guidelines** or **EARS syntax** score each requirement.

**AI-driven management:**
1. **Live Bidirectional AI Traceability** — requirement ↔ code/test linkage maintained live in both directions.
2. **Proactive Drift & Scope-Creep Monitoring** — flags when implementation or scope wanders from the baseline.
3. **Human sign-off stays mandatory** — strategic trade-offs, legal accountability, and stakeholder relationships still require human business analysts and Product Owners to review and approve AI-generated artifacts before sign-off.

**Exam skeleton [CS9]**
- Elicitation AI: unstructured ingestion (tickets/logs/transcripts/reviews), synthetic personas + autonomous interviews, instant prototypes
- Synthesis AI: artifact conversion; ambiguity/quality scoring vs INCOSE / EARS
- Management AI: bidirectional traceability; drift + scope-creep monitoring
- Humans still approve: trade-offs, legal accountability, stakeholder trust

## 2.3 How SDD transforms the requirements lifecycle

Two core ideas:
1. **"The spec declares intent; the code realizes it."** Specifications live **in version control** alongside/above the codebase. When business needs change, developers **do not edit raw source code** — they update the **specification file**, and AI refactors the codebase and **regenerates tests** to match the new spec baseline.
2. **Elicitation feeds directly into bounded guardrails.** Elicitation explicitly defines what the feature **must NOT do**, which prevents AI agents from generating unnecessary dependencies. (Negative requirements are guardrails.)

**Exam skeleton [CS9]**
- Spec declares intent, code realizes it
- Specs live in version control; change = edit spec → AI regenerates code + tests
- Elicitation defines must-not-do → bounded guardrails for agents

## 2.4 The Spec Kit command chain (memorize the order)

| Command | Purpose | When to use |
|---|---|---|
| `/speckit.constitution` | Define project-wide principles | Beginning of a project |
| `/speckit.specify` | Create a feature specification | Before technical design |
| `/speckit.clarify` | Identify and resolve ambiguities | After initial specification |
| `/speckit.plan` | Create the technical plan | After requirements are sufficiently clear |
| `/speckit.checklist` | Generate quality checklists | Before implementation |
| `/speckit.tasks` | Create actionable implementation tasks | After planning |
| `/speckit.analyze` | Check artifact consistency | Before implementation or after significant changes |
| `/speckit.implement` | Execute implementation tasks | During development |
| `/speckit.converge` | Identify remaining gaps against artifacts | During and after implementation |

Flow: `constitution → specify → clarify → plan → checklist → tasks → analyze → implement → converge` (converge feeds gaps back: fix → re-verify).

**Exam skeleton [CS9/Lab1]**
- 9 commands in order; clarify = ambiguity kill-step after specify
- analyze = cross-artifact consistency BEFORE implement
- converge = gap-finder DURING/AFTER implementation

## 2.5 Risks of Markdown-first SDD (Spec Kit) + mitigations

Shifting architecture and design to Markdown artifacts (`spec.md`, `plan.md`, `constitution.md`) introduces risks across **execution, architecture, and governance**. Four architectural/technical risks:

1. **Dependency & API hallucinations** — the agent invents libraries/APIs that do not exist.
2. **Shallow edge-case & non-functional coverage** — specs cover the happy path; NFRs get thin treatment.
3. **Spec-to-code drift** — code changes without the spec being updated (they diverge silently).
4. **Cascading error propagation** — one wrong assumption early (e.g., in plan.md) propagates into every downstream task.

Mitigation table (risk → root cause → practical mitigation):

| Risk | Root cause | Practical mitigation |
|---|---|---|
| Architectural bloat | LLM tendency toward maximal complexity | Mandate **simplicity rules in constitution.md** (e.g., "prefer single relational tables over new microservices") |
| Cascading errors | Unchecked transition from `/plan` to `/tasks` | Treat **plan.md as an explicit PR gate** requiring mandatory **senior human review** before running `/tasks` |
| Spec-to-code drift | Unlinked code changes during `/implement` | Require speckit validation or **CI checks that fail if code commits touch architecture files without updated specs** |
| Legacy context loss | LLM context limitations | Include **explicit reference files** (schema dumps, OpenAPI contracts) in `.specify/` context during the `/plan` prompt |

**Exam skeleton [CS9]**
- 4 risks: hallucinated deps/APIs; shallow edge/NFR coverage; spec-to-code drift; cascading errors
- Fixes: simplicity rules in constitution; plan.md as senior-review PR gate; CI fails drift; reference files in context
- Risks live across execution + architecture + governance

## 2.6 Architecture Decision Records (ADR)

Definitions (exact):
- **Architectural Decision (AD):** a **justified design choice** that addresses a functional or non-functional requirement that is **architecturally significant**.
- **Architecturally Significant Requirement (ASR):** a requirement that has a **measurable effect** on the architecture and quality of a software and/or hardware system.
- **ADR:** captures **a single AD and its rationale** — the reasons for a chosen decision plus its **trade-offs and consequences**.

Five things ADRs help with (P-C-S-A-M):
1. **Preserving decision history** — why a technology/design was selected.
2. **Improving team communication** — developers, architects, stakeholders aligned.
3. **Supporting maintenance** — future developers understand existing choices.
4. **Avoiding repeated discussions** — prior decisions + rationale are accessible.
5. **Managing change** — a reference when a decision must be revisited.

Tool note: the **adrkit** extension **drafts ADRs from plans, retrieves existing decisions, and checks plans against them**.

**Exam skeleton [CS9]**
- AD = justified choice for architecturally significant requirement; ASR = requirement with measurable architectural effect; ADR = one AD + rationale + trade-offs
- 5 benefits: history, communication, maintenance, no repeated debates, change management
- adrkit: drafts from plans, retrieves decisions, checks plans against them

## 2.7 Risk-control activities across the SDD lifecycle (4)

1. **Requirements validation** — are the requirements consistent, complete, testable?
2. **Architecture risk analysis** — where can the generated architecture fail?
3. **Security and threat modelling** — abuse cases, attack surface, trust boundaries.
4. **Task decomposition and dependency validation** — is the task graph correct and are dependencies real?

**Exam skeleton [CS9]**
- RAST: Requirements validation → Architecture risk analysis → Security/threat modelling → Task decomposition + dependency validation

## Part I–II combined memory map

```
CS8/CS9 spine:
Bottleneck shift ─► AI-native handover ─► SDD (guardrails up front)
   ├─ Tools: Spec Kit (CLI+templates) | Kiro (R→D→T, waves) | Playbook (6 stages, VC handover)
   ├─ Spectrum: First / Anchored / As-Source
   ├─ Requirements: Elicitation → Synthesis → Management  (+ AI in each; MoSCoW, BDD)
   ├─ Commands: constitution→specify→clarify→plan→checklist→tasks→analyze→implement→converge
   ├─ Risks: hallucinated deps, shallow NFR, drift, cascade  (+4 mitigations)
   ├─ ADR: AD/ASR/ADR; 5 benefits; adrkit
   └─ RAST risk-control activities
```

# PART III — CS10: Design and Architecture

## 3.1 Zero institutional memory: the new intern vs the agent

- A human new developer constructs a **shared mental model** over weeks and months.
- An **AI agent resets every hour or day** and **never accumulates understanding** — *zero institutional memory*.
- A chronic gap separates the **shared mental model** (the conceptual design living in engineers' heads) from the **concrete dependency structure buried in code** — and **agents live in that gap**.

Consequence: anything that exists only in heads or chat history is invisible to every future agent session. It must be encoded in the repository (specs, steering files, ADRs, standards) or it is lost.

**Exam skeleton [CS10]**
- Humans build shared mental model over weeks; agents reset hourly — zero institutional memory
- Agents live in the gap between mental model and dependency structure in code
- Cure: encode constraints/artifacts in the repo, not in chat

## 3.2 The primary problem is managing complexity

Good design attacks complexity in **two ways**:
1. **Eliminate it entirely** (e.g., remove unnecessary special cases).
2. **Hide it through modular design**.

Treat software design as a **decomposition problem**; Ousterhout's *Philosophy of Software Design* adds **"Design It Twice"** — consider (at least) two alternative designs before committing.

**Exam skeleton [CS10]**
- Complexity = primary problem; fight it by eliminating or hiding it
- Design = decomposition problem; "Design It Twice"

## 3.3 Deep modules vs shallow modules

| | **Deep module** | **Shallow module** |
|---|---|---|
| Interface | Simple, **narrow** | Relatively **large** |
| Functionality hidden | **Significant**, complex beneath the surface | **Small** relative to interface size |
| Leverage vs complexity | **Maximizes leverage**; minimal cognitive load on callers | Little leverage; **increases system-wide interface complexity** |
| Design lineage | Revives classic **information hiding** | "Shallow webs" |

**Exam skeleton [CS10]**
- Deep = narrow interface, big hidden functionality, low caller cognitive load
- Shallow = big interface, little hidden work, system-wide interface bloat
- Deep modules revive information hiding

## 3.4 Sinks, not pipes — and contained blast radius

- A **pipe** continually **leaks execution across boundaries** — triggering cascading database triggers, background queues, downstream side effects. It never "finishes".
- A **sink** **receives an input, performs its work, and terminates entirely within its boundary**.
- **A sink is inherently a deep module; a pipe is almost always a shallow module.**
- Benefit: **contained blast radius** — a change to a sink cannot ripple outward.

**Exam skeleton [CS10]**
- Pipe = execution leaks across boundaries (triggers, queues, side effects)
- Sink = input → work → terminate inside boundary
- Sink = deep module; pipe = shallow; sinks give contained blast radius

## 3.5 Cognitive load vs agent context limits (why deep modules matter MORE now)

The primary benefit of deep modules was always reducing **human cognitive load**: callers interact using **only the interface documentation**, never internals. In an AI-augmented SDLC, human cognitive load translates directly into an **agent's finite context window and attention budget**:

```
Agent meets a PIPE:
  must load + trace entire downstream dependency chain (files, services)
  → exhausts context limits → attention dilution → hallucinations

Agent meets a SINK (deep module):
  treats component as true black box
  holds only immediate contract + local logic in context
  tests against explicit interface → changes deploy with contained blast radius
```

**Exam skeleton [CS10]**
- Deep modules cut human cognitive load; for agents the equivalent budget = context window + attention
- Pipe: agent traces whole downstream chain → context exhaustion, attention dilution, hallucination
- Sink: black box — contract + local logic only; contained blast radius

## 3.6 The same old story: the remedies never changed

- **Low coupling and high cohesion** (drift compounds; under AI its cost compounds too).
- **Progressive disclosure of complexity** via **graduated layers of abstraction**.
- Remedies remain foundational: **explicit boundaries, honest interfaces, minimal side effects, high cohesion, low coupling**.
- Key line to quote: *"AI reaches its true potential not when models grow arbitrarily smarter, but when software architectures finally tell the absolute structural truth about themselves."*

**Exam skeleton [CS10]**
- Low coupling + high cohesion; compounding drift; compounding cost under AI
- Progressive disclosure; graduated abstraction layers
- Remedies: boundaries, honest interfaces, minimal side effects, cohesion, coupling

## 3.7 The codebase outweighs the prompt — context engineering as architecture

- **"The Architecture is the Prompt."** No prompt-engineering technique, agentic framework, or advanced foundation model can rescue an architecture of tangled modules and cascading side effects. **The codebase outweighs the prompt.**
- The answer is not new: well-defined module boundaries, clear interfaces, minimal side effects, high cohesion, low coupling.
- **Context engineering** — filling the context window with **just the right information for the next step** — is now a **first-class architecture discipline, not a prompting technique**; it is a **system design problem** (what info exists, where it lives, how the agent finds it).

**Exam skeleton [CS10]**
- Architecture is the prompt; codebase outweighs the prompt; no prompt trick fixes a tangled architecture
- Context engineering = right information for the next step = first-class architecture discipline, a system design problem

## 3.8 The architect authors the constraint

Some knowledge **cannot be recovered later**: agents cannot infer it, developers cannot reliably reconstruct it from code, and it **disappears entirely if the architect does not encode it before the sprint begins**. Four such categories:
1. **Versioning policy**
2. **Behavioral constraints**
3. **Dependency rationale**
4. **Exception logic**

Effect on review: authoring the constraint **does not eliminate the review** — it changes what the review is *for*: **from catching violations → to confirming the system is doing what it was declared to do**.

**Exam skeleton [CS10]**
- Architect authors constraints BEFORE the sprint: versioning policy, behavioral constraints, dependency rationale, exception logic
- Review shifts: catching violations → confirming declared behavior

## 3.9 Orchestrators vs agent harnesses (control-flow ownership)

Trade-off: **prompt logic = flexibility at the expense of reliability; code = deterministic reliability at the expense of rigidity.**

| | **Agent Harness** | **Orchestrator** |
|---|---|---|
| Control flow owned by | **The LLM** | **Deterministic code** |
| Shape | Continuous loop: model decides next action (read files, run commands, terminate) | Code dictates execution path across **structured, repeatable stages** |
| Best for | **Open-ended exploration** with a **human in the loop** steering wandering behavior | **Autonomous, unattended** workflows (automated PR reviewers, batch pipelines) |
| Failure behavior | Can wander; needs steering | **Errors remain isolated; fails predictably** |

**Guiding rule (quote it):** *The less human oversight involved, the more control flow must live in deterministic code.*

**Exam skeleton [CS10]**
- Harness: LLM owns control flow; exploration; human steers
- Orchestrator: code owns control flow; unattended; isolated + predictable failure
- Rule: less human oversight → more control flow in deterministic code
- Prompt = flexible/unreliable; code = reliable/rigid

## 3.10 A three-layer governance framework for AI coding agents

1. **Constraints Layer** — formal boundaries using tools like **cursorrules** and **Architecture Description Languages (ADLs)**.
2. **Conformance Layer** — evaluates generated code against **architectural fitness functions** and **dependency rules** (plan-build approval gates, automated linting) **before code propagates**.
3. **Knowledge Layer** — feeds architectural context back to the agent via **repository maps**, and automatically extracts decisions from agent reasoning traces into formal **ADRs**.

**Exam skeleton [CS10]**
- Constraints (cursorrules, ADLs) → Conformance (fitness functions, dependency rules, gates, linting) → Knowledge (repo maps, ADRs from traces)

## 3.11 Cost-of-change mapping: what got cheap, what did not

Classic definition of architecture: *"a set of decisions that are difficult to change, and hence we should get them right."* Is that still true when dev effort is dirt cheap? The cost-of-change table answers it:

| Architectural dimension | Traditional cost | AI-augmented cost | Why the shift |
|---|---|---|---|
| Business logic & CRUD APIs | High (manual coding/testing) | **Low** | LLMs generate boilerplate, tests, standard transformations instantly |
| System invariants & state consistency | High | **High / Critical** | AI cannot bypass distributed state locking, ACID guarantees, race conditions |
| Data topology & storage model | High | **Very high** | Migrating multi-terabyte production data stays high-risk however fast code is written |
| Non-deterministic boundaries | Non-existent | **High (new cost)** | Fallback policies, latency budgets, guardrails for AI components add structural complexity |

**Exam skeleton [CS10]**
- CRUD/business logic: high → LOW
- Invariants/state consistency: HIGH/CRITICAL (ACID, locking, races — AI can't bypass)
- Data topology: VERY HIGH (data migration risk unchanged)
- Non-deterministic boundaries: NEW cost (fallbacks, latency budgets, guardrails)

## 3.12 Architecture is not dead; its coordinates have moved

- Architects spend **less time defining interface method signatures** and **more time defining system invariants** (e.g., *"Account balance can never drop below zero"*, *"PII must never cross region boundaries"*).
- New job: **managing the non-deterministic surface area**; **spec as the new codebase**.
- Stop framing architectural decisions around **"how long it takes to code"**; start framing them around **blast radius, auditability, latency budgets, and invariant protection**.

**Exam skeleton [CS10]**
- Signatures → invariants ("balance never negative", "PII never crosses regions")
- Manage non-deterministic surface area; spec = new codebase
- New decision axes: blast radius, auditability, latency budgets, invariant protection

## Part III memory map

```
CS10 spine:
Agents live in the gap ─► complexity is the problem ─► deep modules / sinks
   ├─ Pipe exhausts context → hallucination; Sink = black box
   ├─ Old remedies: boundaries, interfaces, cohesion, coupling
   ├─ Architecture IS the prompt; context engineering = system design
   ├─ Architect authors constraints (versioning/behavior/dependency/exception)
   ├─ Harness vs Orchestrator (+ guiding rule)
   ├─ Governance: Constraints → Conformance → Knowledge
   └─ Cost-of-change: CRUD cheap, invariants/data/not-det boundaries expensive
```

# PART IV — CS11: Coding (both decks)

## 4.1 From autocomplete to structured AI collaboration

Six AI coding modes to name in an answer: **pair programming; autonomous coding agents; code generation; completion; bug detection; debugging.**

### The Frustration Loop

```
Generate ──► Review ──► Not aligned ──► Regenerate ──► Heavy edit / abandon
    ▲                                                    │
    └────────────────────── (loop repeats) ◄─────────────┘
```

Three causes of friction:
1. The model **defaults to generic patterns when project context is absent**.
2. Output may be **syntactically correct yet architecturally misaligned**.
3. **Correction effort can consume the apparent speed gain**.

### The Speed Trap: measure the right outcome

The goal is **usable, aligned code — not merely fast output**.

| Misleading metric | More useful alternative |
|---|---|
| Time to first output | **First-pass acceptance rate** |
| Lines of code generated | **Iteration cycles per task** |
| Tasks completed | **Post-merge rework required** |
| Generation speed | **Review burden compared to manual writing** |

**Exam skeleton [CS11]**
- 6 modes; frustration loop: generate → review → not aligned → regenerate → heavy edit/abandon
- Causes: no project context → generic patterns; syntactically fine but architecturally wrong; corrections eat the speed gain
- Speed trap: measure first-pass acceptance, iteration cycles, post-merge rework, review burden

## 4.2 Five patterns for effective AI collaboration

Mnemonic **"K-D-C-E-F"**: Knowledge priming, Design-first, Context anchoring, Encoding team standards, Feedback flywheel.

| # | Pattern | What you do | Failure mode it addresses |
|---|---|---|---|
| 1 | **Knowledge Priming** | Share **curated project context before** requesting code: stack versions, directory structure, naming rules, architecture, representative examples. Treat context as **reusable project infrastructure**, not a one-off preamble | AI defaulting to **generic, internet-average patterns** due to missing context |
| 2 | **Design-First Collaboration** | Move progressively: capabilities → components → interactions → contracts → implementation. Resolve misunderstandings **before** code exists; ask the AI to **compare options and surface trade-offs** | AI prematurely jumping to implementation, **solving the wrong problem** |
| 3 | **Context Anchoring** | Maintain a **living feature document**: decisions, constraints, open issues, current state. Restore context across long conversations and session boundaries; update it when decisions change | AI writing functional code that **violates team expectations**, heavy post-generation rework |
| 4 | **Encoding Team Standards** | Convert **tacit senior-engineer judgment into reusable, versioned guidance**: clean-code, architecture, security, testing, error handling, dependency rules. **Same expectations for generation and review** | AI **forgetting earlier decisions or contradicting itself** as conversations lengthen or span sessions |
| 5 | **Feedback Flywheel** | Capture **recurring review findings, successful prompts, failure patterns**; feed validated learning back into context, standards, workflows | Teams **repeating the same AI mistakes** without accumulating institutional knowledge |

**Tool-to-teammate shift mapping** (life-cycle event → pattern): Onboarding → knowledge priming; Whiteboarding → design-first; Review norms → encoded team standards; Shared decisions → context anchoring; Retrospective → feedback flywheel.

**Shared mental model:** together the five patterns establish a **unified vocabulary, architectural vision, and quality baseline** between humans and AI — shifting developer effort **from constant vigilance and correction to expressing intent and refining output**.

**Trade-offs and costs:** maintaining priming documents and structured design discussions requires **upfront time and discipline**. Not warranted for **trivial, one-off utility functions**; pays off for **non-trivial features, long sessions, team-wide coordination**.

**Exam skeleton [CS11]**
- K-D-C-E-F; each pattern has a named failure mode (generic patterns / wrong problem / team-expectation violations / self-contradiction / repeated mistakes)
- Context supplied deliberately; standards apply to generation AND review
- Cost: upfront time; skip for trivial one-offs

## 4.3 Guardrails: definition and four types

**Definition:** a guardrail is a **rule, constraint, or control mechanism that prevents an AI system from performing unsafe, incorrect, unauthorized, or undesirable actions**. Example: an agent modifying a payment application must be prevented from **deleting production data, introducing insecure code, or merging untested changes**.

| Type | Stops |
|---|---|
| **Action guardrails** | Dangerous actions: deleting a database, modifying production infrastructure, arbitrary shell commands |
| **Code quality guardrails** | Enforce linting, type checking, test coverage requirements, dependency policies, architectural constraints |
| **Security guardrails** | Hard-coded secrets, unsafe dependencies, SQL injection, insecure authorization, sensitive-data exposure |
| **Workflow guardrails** | Require **human approval** before merging code, deploying to production, or **irreversible operations** |

**Exam skeleton [CS11]**
- Definition + payment-app example
- A-Q-S-W: Action / Quality / Security / Workflow (workflow = human approval gates)

## 4.4 Guardrail architecture: enforced OUTSIDE the model

Principle: a robust guardrail **should not depend only on telling the AI to behave responsibly** — it must be **enforced by software outside the model whenever possible**.

**Three core layers:**
1. **Input guardrails (Pre-LLM):** prompt-injection detection; **PII redaction & tokenization**; topic & policy filtering.
2. **Context & retrieval guardrails (RAG):** control what the model can retrieve and ground on.
3. **Output guardrails (Post-LLM):** **hallucination & grounding checks**; **format & schema validation**.

**Four practical implementation approaches:**
1. **Deterministic rules** — fast, zero-cost keyword matching and regex for explicit blocklists.
2. **Model-based evaluation** — a secondary smaller LLM or specialized classifier scores semantic safety, tone, toxicity.
3. **Middleware & hooks** — interceptor hooks around the agent execution pipeline (e.g., LangChain, n8n).
4. **Human-in-the-loop** — **pause agent execution automatically** on high-risk actions (large financial transactions, database writes).

(LangChain offers built-in guardrails — PII detection, HITL — plus middleware for custom ones.)

**Exam skeleton [CS11]**
- Enforced outside the model, not by instructions alone
- Layers: Input (injection, PII, policy) → Context/RAG → Output (grounding, schema)
- Implementations: deterministic rules | model-based eval | middleware/hooks | HITL pause

## 4.5 Lattice: operationalizing the five patterns

**Lattice** = an **installable, repository-based framework** that operationalizes and sustains the five collaboration patterns across teams and projects.

**Three design principles:**
1. **Skills over Prompts** — prompts are personal, ephemeral, die with the chat; skills are **shared, version-controlled in the repository, evolved via pull requests**.
2. **Composability over Monoliths** — small single-purpose skills (**atoms**) combine into broader lifecycle workflows (**molecules**) instead of one unwieldy instruction prompt.
3. **Living Context over Static Config** — the **`.lattice/` directory** accumulates institutional memory (project architecture, standards, operational learnings) that refines itself through each feature cycle.

**Core building blocks:**
- **Atoms** — modular reusable skills encoding principles: knowledge-priming, design-first, context-anchoring, clean-code, domain-driven-design, secure-coding, test-quality.
- **Molecules** — composed multi-step workflows for the feature pipeline: **design-blueprint** (progressive design), **code-forge** (implementation within domain boundaries), **review** (independent code evaluations).
- **Refiners** — guided interviews tailoring defaults to a codebase: **Overlay mode** (project rules added on top of defaults) vs **Override mode** (defaults fully replaced).
- **Collaborative Judgment** — trains the AI to recognize genuine architectural ambiguity and **present structured options with trade-offs** instead of guessing silently.

**Exam skeleton [CS11]**
- Lattice = installable repo-based implementation of the 5 patterns
- Skills over prompts | composability (atoms→molecules) | living context (.lattice/)
- Blocks: atoms, molecules (design-blueprint/code-forge/review), refiners (overlay vs override), collaborative judgment

## 4.6 SEI Shift-Left case: automating library replacement

**Problem:** outdated libraries & technical debt. DoD program data and the **FY22 NDAA Section 835 Technical Debt Study** show open-source libraries often **dozens of versions behind** → major security vulnerabilities and operational risk. **Manual upgrades** are error-prone, labor-intensive, too slow across many programs.

**Solution shape:**
- **Targeted task decomposition** — never delegate the entire upgrade to an LLM; decompose into discrete sub-tasks and **pair generative AI with established deterministic tools**.
- **"Write once, roll out everywhere"** — solve a library upgrade once; roll the transformation out automatically across all dependent projects.

**Division of labor (static analysis vs LLM):**
- **Identifying breaking changes → static analysis.** Tools like **RevAPI** (Java API analysis/change tracking) detect API breaking changes **more completely and consistently than LLMs**.
- **Synthesizing transformation strategies → LLMs.** Generate code transformation recipes for the identified breaking changes.
- **Automated application → transformation engines**, then final build + test.

**Stakeholder roles (P-S-A):**
1. **Library Providers & Developers** — determine + formalize API breaking changes (static analysis).
2. **Library Sustainers & Tooling** — use LLMs to generate structured transformation strategies when recipes don't exist.
3. **Application Developers** — locate + apply strategies to their codebases; automated build/test with the updated library.

**Exam skeleton [CS11]**
- FY22 NDAA 835; libraries dozens of versions behind; manual upgrades fail
- Decompose + pair LLM with deterministic tools; write once roll out everywhere
- RevAPI finds breaking changes; LLM writes recipes; engines apply; build+test verify
- Roles: Providers formalize → Sustainers synthesize → App developers apply

## 4.7 Harness engineering mapping + the cybernetic loop

The SEI pairing mirrors the **computational vs inferential sensor architecture** of harness engineering — and solves the "prompt creep" & hallucination dilemma:

| Harness layer | SEI workflow | Function |
|---|---|---|
| **Computational sensor** (deterministic, rule-based) | RevAPI static analysis | Detect breaking changes consistently, exhaustively, **without hallucination** |
| **Inferential synthesizer** (probabilistic, semantic) | Generative AI / LLM | Translate formal diffs + release notes into contextual migration recipes |
| **Computational actuator & verifier** (deterministic gate) | Code transformation tools + test runners | Apply the recipe; run compiler + tests to verify correctness |

**Cybernetic loop ("keeping quality left", Spec → Harness → Loop):**

```
FEEDFORWARD:  RevAPI computes exact deltas between versions, formalizes them
      │
EXECUTION:    LLM generates migration pattern → automated tooling applies it
      │
FEEDBACK SENSOR: build system + unit tests = computational feedback sensor
      │            build breaks / test fails → error output injected back
      │            into LLM context → refine recipe
      └──► loop until clean build (converge)
WRITE ONCE, ROLL OUT EVERYWHERE: verified recipe = reusable deterministic asset
      executed across dozens of dependent projects WITHOUT re-running the LLM
```

**Exam skeleton [CS11]**
- Computational sensor (RevAPI) | inferential synthesizer (LLM) | computational actuator+verifier (tools+tests)
- Loop: feedforward deltas → LLM recipe → apply → build/tests as feedback sensor → errors re-injected → converge
- Verified recipe = deterministic asset reused without re-running the LLM

## 4.8 Testing in a non-deterministic world (CS11 deck 2)

**The assumption that breaks:** software behaves **deterministically** — identical inputs under identical preconditions yield identical outputs and postconditions. In modern systems tests do not always yield the same results → bugs become **rare, intermittent, hard to reproduce, isolate, diagnose**. Running a **single test case once per requirement is insufficient** under non-determinism.

**Pre-AI technological drivers (5):** Agile & DevOps (automated regression assumes repeatable behavior); multicore & multithreading (race conditions, timing); cyber-physical & autonomous systems (sensor/environment inputs); ML & AI (adaptive logic changes with data); systems-of-systems & cloud/VMs (distributed, unpredictable interactions).

**Actual vs apparent non-determinism:**
- **Actual** — no **theoretical** way to predetermine exact behavior (e.g., quantum-based random number generation).
- **Apparent** — no **practical** way for testers or test oracles to predict outcomes (overwhelming complexity, inadequate controllability/observability).
- **AI-generated code and agentic development fall squarely under Apparent non-determinism.**

**Four categorical sources: P-E-C-E** — Physical (environmental), Emergent (subsystem integration), Concurrent (multi-threading/distribution), Exceptional (fault/failure handling).

**AI-generated code spans three of them: E-C-E**
- **A. Emergent (primary driver):** agents select libraries, write glue code, build integrations; **minor prompt-wording or sampling variations → different architectural assumptions → unexpected emergent behaviors and hidden couplings** at integration.
- **B. Concurrent:** agentic workflows use asynchronous tool calls, parallel worktree executions, multi-step reasoning loops → **race conditions, timing variations, non-deterministic tool-output ordering**.
- **C. Exceptional:** models produce code that works on the **happy path** but fails under rare/stress conditions → **error-handling paths, boundary validations, resource cleanups behave inconsistently** on unhandled exceptions.

**Exam skeleton [CS11]**
- Determinism assumption; bugs become intermittent, unreproducible; one-run testing insufficient
- Actual (no theoretical prediction) vs Apparent (no practical prediction); AI code = APPARENT
- P-E-C-E four sources; AI spans E-C-E (emergent = primary; concurrent = agent loops; exceptional = happy-path-only code)

## 4.9 Reviewing AI-generated code: the 8-step checklist

1. **Start with functional checks** — always run automated tests and static analysis **first**.
2. **Verify context and intent** — does the code fit the purpose and architecture of the project?
3. **Assess code quality.**
4. **Scrutinize dependencies** (invented, outdated, unnecessary).
5. **Spot AI-specific pitfalls.**
6. **Use collaborative reviews.**
7. **Automate what you can.**
8. **Keep improving your workflow.**

Mnemonic: **"First Check Quality, Don't Push Careless AI In"** (F-C-Q-D-P-C-A-I).

**Exam skeleton [CS11]**
- Tests + static analysis FIRST; then intent/architecture fit; quality; dependencies; AI pitfalls; collaborative; automate; iterate

## Part IV memory map

```
CS11 spine:
Frustration loop ─► 5 patterns (K-D-C-E-F) ─► Shared mental model
   ├─ Guardrails: A-Q-S-W types; layers Input→RAG→Output; rules/eval/middleware/HITL
   ├─ Lattice: skills>prompts, atoms→molecules, .lattice/ living context
   ├─ SEI: RevAPI (compute) + LLM (infer) + engines/tests (verify) = cybernetic loop
   ├─ Non-determinism: Apparent; E-C-E for AI code
   └─ Review: 8 steps, tests+static first
```

# PART V — CS13: Software Testing

## 5.1 Six ways AI is used in software testing

Mnemonic **"T-R-S-D-V-L"**:
1. **Automated Test Generation**
2. **Risk-Based Test Prioritization**
3. **Self-Healing Test Scripts**
4. **Defect Prediction**
5. **Visual AI and Anomaly Detection**
6. **Log Parsing and Root-Cause Diagnostics**

**Exam skeleton [CS13]**
- T-R-S-D-V-L, six uses; be ready to describe any one in one line

## 5.2 How AI augmentation changed testing (traditional vs AI-augmented)

| Area | Traditional | AI-augmented |
|---|---|---|
| **Test maintenance** | QA time burned updating broken locators/obsolete scripts | **Self-healing frameworks repair script paths automatically during execution** |
| **Shift-left testing** | Testing starts after code is written/deployed to staging | Scenario creation + requirement-ambiguity checks happen **early at PR/design phase** |
| **Role of QA engineer** | Manual script creation, repetitive execution, log scanning | Quality engineering: exploratory testing, risk evaluation, **prompting, orchestrating AI agents** |
| **Regression strategy** | Entire bloated suite every release | **Smart test-suite optimization; impact-based selection** runs only what the commit impacts |
| **Barrier to entry** | Required scripting skills (Selenium, Cypress, Playwright) | **NLP lets non-technical members create tests in plain language** |

**Exam skeleton [CS13]**
- Maintenance: self-healing replaces locator repair
- Shift-left: ambiguity checks at PR/design, before execution
- QA → Quality Engineering (exploratory, risk, prompting, agent orchestration)
- Regression: impact-based selection; entry barrier falls via natural-language tests

## 5.3 Self-healing test automation (use case)

- **Problem:** UI updates break automated test scripts continuously. A Selenium/Cypress team shipping **every two weeks** expects **20–30% of the suite to fail** after each release; **two engineer-days per sprint** disappear into locator repairs before any new testing starts.
- **How AI helps:** AI **monitors the application DOM**, detects element changes, **identifies the updated locator using multiple fallback strategies**, repairs the affected test, **confirms the fix before the next run**.
- **Runs as a background process between every deployment.**

**Exam skeleton [CS13]**
- 20–30% suite failures per biweekly release; 2 engineer-days/sprint on locators
- DOM monitoring → fallback locator strategies → repair → verify before next run
- Background process between deployments

## 5.4 Test selection / prioritization: full suite vs static rules vs AI-driven TIA

| Feature | Full Suite Regression | Static Rule-Based Selection | AI-Driven Predictive TIA |
|---|---|---|---|
| Selection scope | 100% of tests | Explicit file-path mappings | **Dynamic, ML-selected subset** |
| Feedback loop | Hours (slow) | Fast, but **misses indirect edge cases** | **Minutes (high accuracy)** |
| Handling flaky tests | Retries all failed tests blindly | Excludes test paths manually | **Statistical failure patterns isolate flakiness from true regressions** |
| CI compute cost | Very high | Medium | **Low (reduces run times 70–90%)** |
| Cross-service tracking | Rare/complex | Hardcoded static links | **AI maps dynamic API dependencies and message contracts** |

TIA = Test Impact Analysis. Quote **70–90% CI run-time reduction**.

**Exam skeleton [CS13]**
- Three strategies; TIA = dynamic ML subset, minutes feedback, separates flaky from real regressions
- 70–90% CI time reduction; AI maps dynamic cross-service API dependencies

## 5.5 Predictive defect analysis (defect forecasting)

- Goal: find high-risk **"hotspots"** — files, functions, or modules with a **statistical probability of containing bugs — before tests run or code hits production**.
- **Data inputs (3):** code churn & commit metadata; code complexity metrics; defect history.
- **JIT (Just-In-Time) defect prediction:** supervised classification models — **Random Forests, XGBoost, Graph Neural Networks mapped to ASTs** — evaluate **every incoming PR**.

**Exam skeleton [CS13]**
- Hotspots predicted pre-execution; inputs: churn+commits, complexity, defect history
- JIT models: RF / XGBoost / GNN-on-ASTs classify each PR

## 5.6 Measuring churn and complexity (the two hotspot predictors)

**Code churn** — rate and volume of code modification in a window (commits, PRs, sprints, releases):
- **Absolute churn** = lines added + deleted + edited. Example: +100, −30, edited 20 → **150 LOC**.
- **Relative churn (rate %)** = churn normalized by file/repo size, enabling fair comparison across small/large modules.
- **15–20% churn rate is acceptable.**

**Code complexity** — how hard a program's logic is to read, test, maintain, execute safely.
- **Cyclomatic complexity (McCabe)** = number of **linearly independent paths** through the control-flow graph.
- Decision points that each add a path: `if, else if, while, for, case, catch, &&, ||` (counting rule: paths ≈ decision points + 1).
- **Thresholds: < 10 low risk; 11–20 moderate; > 20 high risk → refactor.**

**How they drive prediction:** high churn identifies **"instability hotspots"**; high complexity **exceeds human cognitive limits**; systems ingest churn, complexity, author overlap, and historical defect tags to output **automated actions**.

**Exam skeleton [CS13]**
- Absolute churn = added+deleted+edited (150 LOC example); relative = %; 15–20% OK
- Cyclomatic = independent paths; decision points if/elseif/while/for/case/catch/&&/||
- <10 / 11–20 / >20 thresholds; churn = instability; complexity = cognitive limit

## 5.7 Circular validation (major exam topic)

**Definition:** circular validation occurs when an AI system generates **both the production code and its accompanying test suite from the same internal model, prompt, or implementation assumptions**. The tests share the code's blind spots, so a wrong assumption passes both sides — a **self-referencing feedback loop**.

**How to break it (3 remedies):**
1. **Ground test generation in specifications, not implementation** — tests come from the spec/acceptance criteria, an independent ground-truth source.
2. **Test-Driven Development (TDD)** — tests exist before code; code is written to satisfy them.
3. **Computational verification gates & mutation testing** — deterministic gates plus mutation testing (perturb the code; a good suite must catch the mutations) to prove the suite has real detecting power.

**Exam skeleton [CS13]**
- Same model/prompt/assumptions → code + tests share blind spots
- Break the loop: specs as independent oracle | TDD | verification gates + mutation testing

## 5.8 Testing AI-enabled software: from unit assertions to evals

Traditional testing verifies **deterministic** software; AI software is **probabilistic, dynamic, state-dependent**. Instead of exact string/number matches, use **Model Evaluations (Evals)**:
1. **Distance & similarity metrics** — **Cosine similarity, ROUGE, BLEU** between expected and actual outputs.
2. **LLM-as-a-Judge** — a **stronger, deterministic evaluator LLM** grades quality, safety, relevance, factual correctness against **structured rubrics**.
3. **Assertion thresholds** — output must fall in a **statistical confidence band (e.g., 95%+ semantic alignment)** instead of 100% literal agreement.

**Exam skeleton [CS13]**
- Deterministic vs probabilistic/state-dependent
- Cosine/ROUGE/BLEU | LLM-as-judge with rubrics | thresholds (95%+ semantic alignment)

## 5.9 Testing AI agents (autonomous workflows & tool use)

QA must evaluate the **step-by-step reasoning chain (ReAct loop) and tool usage**, not just the final text:
1. **Tool-Calling Validation** — correct API tool chosen, **valid JSON parameters**, graceful handling of API failure responses.
2. **Goal & Plan Execution** — multi-step objective completed **within an allowed trajectory length**, without **infinite execution loops** (e.g., "book a flight and schedule a calendar invite").
3. **Context & Memory Drift** — long-horizon memory retrieval: agent keeps context across multi-turn sessions without losing focus or **hallucinating state changes**.

**Three core agent-testing strategies (IBM):**
1. **LLM-as-a-judge** with a **three-tiered approach**: **component tests → trajectory tests → end-to-end testing**.
2. **Simulating environments** (mock tools/APIs so trajectories are testable safely).
3. **Maintaining human-in-the-loop practices.**

**Exam skeleton [CS13]**
- Judge the ReAct chain + tool use, not just output
- T-G-M: Tool-calling / Goal-plan-execution / Memory drift
- Strategies: judge + 3 tiers (component, trajectory, E2E) | simulated environments | HITL

## 5.10 Data-centric testing and continuous monitoring

- AI models **degrade as real-world input changes** → testing shifts from a **one-time pre-release gate** to **continuous production observability**.
- **Data & model drift:** monitor production inputs and predictions; detect when incoming user data **diverges from the training distribution**.
- **Shadow testing / A-B trajectory analysis:** route real user traffic **concurrently to newer prompts or model versions in shadow mode**; compare **trajectory success rates** without impacting end users.

**Exam skeleton [CS13]**
- One-time gate → continuous observability; drift detection vs training distribution
- Shadow mode compares trajectory success rates safely

## 5.11 Case: Uber's Software Factory (end-to-end SDLC in action)

- Three loops: **Ideation & Alignment (Inner Loop)** → **Autonomous Execution & Validation** → **Outer Loop & Maintenance**.
- **Six foundational infrastructure pillars:** **Model Gateway; MCP Gateway & Tool Optimization; Agentified DevPods; Managed Skills Marketplace; AI Context Graph; AI Assistant**.

**Exam skeleton [CS13]**
- Inner loop (ideation/alignment) → autonomous execution & validation → outer loop (maintenance)
- 6 pillars: Model Gateway, MCP Gateway & Tool Optimization, Agentified DevPods, Skills Marketplace, AI Context Graph, AI Assistant

## Part V memory map

```
CS13 spine:
6 uses (T-R-S-D-V-L) ─► self-healing (20–30%, 2 days) ─► TIA (70–90% CI saving)
   ├─ Hotspots: churn + complexity + history → RF/XGBoost/GNN per PR
   ├─ Churn 15–20% OK; cyclomatic <10 / 11–20 / >20
   ├─ Circular validation → specs-as-oracle, TDD, gates+mutation
   ├─ Evals: cosine/ROUGE/BLEU, LLM-judge, 95% thresholds
   ├─ Agents: tool-calling, goal/plan, memory drift; component→trajectory→E2E
   └─ Continuous: drift monitoring, shadow testing; Uber 3 loops + 6 pillars
```

# PART VI — CS14: Software Delivery Metrics

## 6.1 DORA fundamentals

DORA metrics have been the **shared language of software delivery performance for nearly a decade** (Ref: *Accelerate: State of DevOps 2018*). Two families: **throughput and stability — do not trade one for the other**; software delivery and operational (SDO) performance adds **availability**.

| Metric | What it measures | The assumption |
|---|---|---|
| **Deployment Frequency** | How often code ships to production | More deploys = healthier, more capable team |
| **Lead Time for Changes** | Time from commit to production | Shorter = better flow through the pipeline |
| **Mean Time to Recovery (MTTR)** | How quickly incidents are resolved | Faster recovery = more resilient systems |
| **Change Failure Rate (CFR)** | Percentage of deploys causing failures | Lower rate = higher code quality |

**Exam skeleton [CS14]**
- D-L-C-M + assumptions; throughput (DF, LT) vs stability (CFR, MTTR); availability added for SDO performance

## 6.2 Why AI breaks DORA

Core one-liner: **"AI collapses the cost of producing code. It does not collapse the cost of verifying it."** This pulls **throughput and stability apart**. DORA was designed for a world where **humans wrote all the code**, so it measures delivery outcomes but stays **blind to how AI shapes the code itself**.

- A dashboard can show **CFR rose 20% last quarter** but **cannot tell whether AI-generated code caused it**.
- Deployment frequency may rise because of AI, better processes, or just **more micro-commits** — attribution is impossible from DORA alone.
- **Code churn has roughly doubled**; DORA did not flag the quality shift because the rewritten code **typically did not cause production failures**.

**Not all metrics break equally:**
- **MTTR holds up well** — measuring response to failure stays honest regardless of who wrote the code.
- **CFR has partial value** — only tracks production failures, misses hidden churn before deployment.
- **Deployment Frequency and Lead Time become misleading without additional context** — inflated by boilerplate/scaffolding/duplicated test files; lead time hides the bottleneck shift (fast commits offset by exploding PR review queues).

Is deployment frequency now meaningless? **Not meaningless, but substantially less informative — pair it with new metrics.**

**Exam skeleton [CS14]**
- Producing got cheap, verifying did not → throughput vs stability pulled apart
- MTTR holds; CFR partial; DF + LT misleading without context
- Churn ~doubled but invisible (no production failures); attribution impossible

## 6.3 AI-era evolution of each metric

| Traditional metric | Challenge in the AI era | Modern AI-era evolution |
|---|---|---|
| Deployment Frequency | Inflated by AI-generated boilerplate, scaffolding, duplicated test files | **TrueThroughput** — adjusts throughput for actual code complexity, preventing vanity inflation |
| Lead Time for Changes | Hides bottleneck shift: fast commits offset by exploding PR review queues | **PR Cycle Time** — explicitly isolates review time to price the **"verification tax"** into capacity planning |
| Change Failure Rate | Only tracks production failures; misses hidden churn before deployment | **Code Durability & PR Revert Rate** — tracks discarded/rewritten/reverted AI code |
| Mean Time to Recovery | Holds up well — an honest signal regardless of code authorship | **MTTR remains valid**, increasingly paired with AI-adoption metrics |

**Exam skeleton [CS14]**
- DF → TrueThroughput (complexity-adjusted); LT → PR Cycle Time (verification tax); CFR → Code Durability & PR Revert Rate; MTTR stays + adoption pairing

## 6.4 What DORA doesn't measure (and now needs)

1. **AI Code Share** — what % of committed code is AI-generated vs human-written?
2. **Code Durability** — does code stick or churn? CFR catches code that **breaks production**; it does not catch code **quietly rewritten, refactored, or replaced within weeks**.
3. **Complexity-Adjusted Output** — DORA treats all deployments as equal: a deploy shipping a new authentication system counts the same as a README update.
4. **Innovation Rate** — new features vs maintenance? DORA can't distinguish a new product capability from a dependency patch.

**Plus adoption signals to capture alongside DORA:** **cycle time** (start→finish), **PR pickup time** (wait before review begins), **review turnaround time**; and code-quality baselines: **defect rates, test coverage, churn patterns, ratio of new code to maintenance work** — baselines reveal whether AI **accelerates technical-debt accumulation or genuinely improves quality**.

**Rework rate (2025 DORA report):** introduced as a **fifth metric** — tracks how often you push **unplanned fixes to production**; a blind spot of the original four that becomes critical when AI generates a significant portion of the codebase.

**Exam skeleton [CS14]**
- S-D-C-I: Share / Durability / Complexity-adjusted / Innovation
- Adoption signals: cycle time, PR pickup, review turnaround; quality baselines: defects, coverage, churn, new:maintenance
- Rework rate = 5th metric (2025 report): unplanned production fixes

## 6.5 Emerging AI-era measurement frameworks

- **WAY Framework** (Work-first, Agnostic, Yours) — meta-framework with three measurement layers: **AI Adoption → Delivery Impact (where DORA lives) → Business Value (reclaimed hours)**. Way principles: **Work-first** = measure what happened, not what people say happened; **Agnostic** = measure across every tool or the picture lies; **Yours** = the data must survive an audit.
- **Developer AI Impact Framework** — keep what works (**MTTR**), evolve what is broken (**CFR → Code Turnover**), add what is new (**AI Code Share**).
- **AWS 5-I Framework** — holistic enterprise approach balancing deployment velocity, code quality, operational efficiency, business impact.
- **DX Core 4** — unified framework encapsulating **DORA, SPACE, DevEx**; four dimensions: **speed, effectiveness, quality, business impact**.

**Exam skeleton [CS14]**
- WAY = Work-first/Agnostic/Yours; layers Adoption → Impact → ROI
- Dev AI Impact: keep MTTR, CFR→Code Turnover, + AI Code Share
- AWS 5-I; DX Core 4 = DORA+SPACE+DevEx (speed/effectiveness/quality/business impact)

## 6.6 The AI Impact Hierarchy (5 levels)

| Level | Name | Core question | What most orgs do |
|---|---|---|---|
| 1 | **Adoption** | Are developers using AI tools? | Measure this **and stop** |
| 2 | **Engagement** | How deeply are they using AI? | Rarely measured |
| 3 | **Productivity** | Is AI accelerating real output? | Measured with **inflated metrics** |
| 4 | **Quality** | Is AI code maintainable? | **Almost never measured** |
| 5 | **Business Value** | Is the AI investment paying off? | **Guessed, not measured** |

Mnemonic: **A-E-P-Q-B** — "All Engineers Produce Quality, Boss".

**Exam skeleton [CS14]**
- Adoption → Engagement → Productivity → Quality → Business Value
- Failure pattern: stop at level 1; quality almost never measured; ROI guessed

## 6.7 The five pillars of the Developer AI Impact Framework

Mnemonic **A-C-V-Q-R**.

**Pillar 1 — AI Adoption.** Whether developers actually use AI tools, how frequently, and how usage distributes. Metrics: **AI Active User Rate (DAU/WAU/MAU)**; **Tool Adoption by Type** — inline completions (Copilot tab, Cursor autocomplete) vs chat (Copilot Chat, Claude) vs **agentic workflows (Claude Code, Cursor Composer)** — deeper modes = deeper integration; **Adoption Distribution** — power users (daily, multi-mode) / casual (weekly, single-mode) / non-users.

**Pillar 2 — AI Code Share.** % of committed code generated or substantially assisted by AI, at **line, commit, and PR level**. Answers what adoption cannot: **how much of the codebase is AI-produced** (a team can have 70% WAU but only 10% AI-assisted code = shallow usage). Metrics: **AI-Assisted PRs %**, **AI-Assisted Lines %**, **AI-Assisted Commits %** (via tool telemetry and attribution signals).

**Pillar 3 — Velocity (Complexity-Adjusted Throughput).** Raw velocity (PRs merged, commits, LOC) is now an **unreliable proxy**. **CAT weights each unit of work by complexity**; **CAT per engineer per week** is the primary velocity metric — value delivered, not volume. Metrics: **CAT per engineer (weekly)**; **Delivery Volume (AI vs Human)** — is AI applied to complex work or only easy tasks?; **Cycle Time (commit→deploy)** tracked **separately for AI-assisted vs human-only PRs**.

**Pillar 4 — Quality.** The **durability** of AI-generated code: does it **persist, or get rewritten/reverted/deleted shortly after merge**? The pillar most orgs miss — they measure output volume and speed but never ask *"does it stick?"*. Primary metric: **Code Turnover Rate** — % of committed code substantially rewritten or deleted within **30 or 90 days** of merge, **tracked separately for AI vs human code**. 30-day = early-warning metric; 90-day = deeper issues (passes initial review, breaks later development). Also **Innovation Rate** (new features vs bug fixes vs infra/config work).

**Pillar 5 — Cost & ROI.** Financial return accounting for **value generated AND hidden rework costs**. Metrics: **Total AI Tool Cost per Engineer (monthly)** — fully loaded: seat licenses + token/usage + implementation overhead (training, admin, integration), "not just the license fee — the actual spend"; **Time Saved Value** — hours saved/engineer/week × loaded engineering cost (salary + benefits + overhead); **Net ROI Multiplier**:

```
Net ROI = (Productive Value of Time Saved − Rework Cost from Code Turnover) / Total AI Tool Cost
```

**Qualitative layer:** Developer-experience surveys combine telemetry + survey data into a composite **AI Value Realization Score**.

**Exam skeleton [CS14]**
- A: DAU/WAU/MAU; inline vs chat vs agentic; power/casual/non-users
- C: PR %, lines %, commits %; 70% WAU vs 10% share = shallow
- V: CAT/eng/week; AI vs human volume; cycle time split by origin
- Q: durability — turnover 30/90-day AI vs human; "does it stick?"
- R: fully-loaded cost; time saved × loaded cost; Net ROI = (time-saved value − turnover rework cost) / total AI cost

## 6.8 AWS: five categories of metrics

1. **Deployment velocity** — time to market; sprint velocity (story points); code commit frequency; PR resolution time; release velocity (releases per quarter/year).
2. **Code quality** — defect density (reduction in bugs); code coverage (up); technical debt (down); static code analysis scores (up).
3. **Operational efficiency**.
4. **Team productivity and satisfaction**.
5. **Business impact**.

(AWS expects metrics to capture immediate efficiency/productivity gains AND long-term quality, team satisfaction, business value.)

**AWS rollout in three phases:**
1. **Baseline & A/B Testing** — compare AI-enabled teams against control groups across multiple release cycles.
2. **Granular Cost Tracking** — attribute inference and tool expenses via project workspaces and IAM principals → true cost-per-outcome.
3. **Guardrail Validation** — enforce automated CI/CD security and quality checks so velocity gains don't spike technical debt or failure rates.

**Exam skeleton [CS14]**
- 5 categories: deployment velocity / code quality / operational efficiency / team productivity+satisfaction / business impact
- Rollout: baseline+A/B → granular cost tracking (IAM, cost-per-outcome) → guardrail validation in CI/CD

## 6.9 DX AI measurement framework (utilization / impact / cost)

A structured system for tracking, interpreting, acting on signals when evaluating AI-assisted engineering; organizes metrics across three dimensions connected to developer conditions, engineering performance, business outcomes:
- **Utilization:** DAUs/WAUs; % PRs AI-assisted; % committed code AI-generated; tasks assigned to agents.
- **Impact:** *Direct* metrics (immediate tool-effectiveness signals) — AI-driven time savings (developer hours/week), developer satisfaction, **Human-Equivalent Hours (HEH)** completed by agents; *Indirect* metrics (regression/longitudinal analysis of Core 4 measures; surface long-term benefits and hidden risks) — PR throughput, perceived rate of delivery, **DXI** (Developer Experience Index), code maintainability, change confidence, change-fail percentage.
- **Cost:** is AI spend and ROI optimal?

Key line: **"AI metrics measure what's happening. Core 4 metrics confirm whether it's working."**

**Exam skeleton [CS14]**
- Utilization / Impact / Cost; direct (time saved, satisfaction, HEH) vs indirect (throughput, DXI, maintainability, change confidence/fail %)

## 6.10 The AI adoption J-curve

DORA frames AI adoption as a **J-curve**: a **temporary productivity dip and period of instability** early, followed by a recovery that **eventually outpaces the pre-adoption baseline**.

**Three forces drive the trough (L-V-P):**
1. **The learning curve** — teams need time to develop fluency with new tools/workflows; the ramp has a real, measurable cost.
2. **The verification tax** — reviewing AI-generated output at the volume AI makes possible **takes longer than most teams initially provision for**.
3. **Pipeline adaptation** — deployment pipelines built for lower volume/velocity need retuning (**smaller batch sizes, stronger automated testing, tighter version control**) before they can safely absorb the new throughput.

Management posture: **treat the dip as a forecastable cost of the transition.**

**Exam skeleton [CS14]**
- Dip → recovery above baseline; treat dip as forecastable cost
- L-V-P: learning curve / verification tax / pipeline adaptation (smaller batches, stronger automated testing, tighter VC)

## 6.11 ROI of AI-assisted software development

**ROI (%) = (Value − Investment) / Investment**

- **Value** = new revenues + cost savings = total value of AI-assisted development.
- **Investment is more than license cost** — a defensible model includes:
  - **Direct hard costs:** licenses, additional AI infrastructure and compute, training and enablement per employee.
  - **J-curve cost:** the modelled productivity dip = **staff size × fully loaded salary × productivity drop × dip duration**.

**Exam skeleton [CS14]**
- ROI% = (Value − Investment)/Investment; Value = new revenue + savings
- Investment = direct hard costs + J-curve cost (staff × loaded salary × productivity drop × dip duration)

## Part VI memory map

```
CS14 spine:
DORA 4 (D-L-C-M) ─► AI breaks DF/LT (churn invisible), MTTR holds ─► evolution table
   ├─ Missing: S-D-C-I + rework rate (5th, 2025)
   ├─ Frameworks: WAY / Dev-AI-Impact / AWS 5-I / DX Core 4
   ├─ Hierarchy A-E-P-Q-B; Pillars A-C-V-Q-R; Net ROI formula
   ├─ AWS 5 categories + 3-phase rollout
   ├─ J-curve: learning curve, verification tax, pipeline adaptation
   └─ ROI% = (Value − Investment)/Investment; J-curve cost formula
```

# PART VII — The Labs (Lab Sheet 1 + Lab Sheet 2)

## 7.1 Lab 1: Spec-Driven Development with GitHub Spec Kit (VS Code)

**Purpose:** hands-on SDD — from requirements to implementation and verification using **GitHub Spec Kit + GitHub Copilot** in VS Code (`specify init --here --integration copilot --script ps`).

**Scenario:** a Task Manager web app for individual users — create tasks (title, description, priority, due date), view, edit, mark completed, delete after confirmation, filter All/Pending/Completed, show overdue tasks, empty state. **Scope exclusions:** authentication, collaboration, notifications, recurring tasks.

**Artifact chain produced (memorize this order):**

```
constitution  ─►  spec.md  ─►  (clarify decisions folded back into spec)
     ─►  plan.md (+ research/data model)  ─►  checklist  ─►  tasks.md (+ dependencies)
     ─►  analyze (cross-artifact consistency)  ─►  implement in increments
     ─►  run + verify vs acceptance criteria  ─►  converge (gap report → fix → re-verify)
     ─►  git commits at each milestone (audit trail)
```

Key mechanics:
- **Constitution first:** establishes measurable project principles — e.g., React + Node/Express, modular architecture, separation of concerns, RESTful APIs with proper status codes, validation, error handling, tests for critical logic, **never expose secrets**, prefer simple maintainable solutions. The constitution **constrains every downstream artifact** (it shapes the technical plan and implementation decisions).
- **Clarify:** the agent asks targeted questions (optional due dates? editing completed tasks? deletion confirmation? default sorting?); answers are **reflected back into the spec** — ambiguity killed before design.
- **Plan:** technology constraints given (React+Vite, Node/Express, REST, MongoDB+Mongoose, Vitest + React Testing Library); the agent produces architecture, project structure, endpoints, data model, components, error handling, testing strategy; **plan must comply with the constitution and approved spec**.
- **Tasks + analyze:** task list must cover project setup, data model + DB connection, CRUD API + validation, frontend components/filters, frontend–backend integration, automated tests, acceptance-criteria verification; `/speckit.analyze` finds **inconsistencies and missing links among requirements, plan, tasks** — if you change an upstream artifact, **update dependents and analyze again**.
- **Implement in increments:** implement a small group of tasks first (e.g., backend setup + data model only), run relevant tests, report completed/remaining — **never approve changes without inspection**.
- **Verify:** manual user journeys (create valid task, view/edit, mark completed, filters, delete confirmation, overdue display, empty state, invalid data → error handling, API/DB unavailable behavior) **plus** automated tests; `/speckit.converge` reports remaining gaps vs the spec.
- **Git milestones:** commit approved artifacts, backend, frontend, verification fixes; **exclude secrets, local env files, machine-specific state**.

**Reflection questions the lab drills (be ready to answer any):**
1. How did the constitution influence the plan/implementation? (Principles became constraints — e.g., "never expose secrets", "input validation + tests" appear as plan sections and code.)
2. Name one ambiguity clarified and its risk if unresolved (e.g., optional due dates unresolved → inconsistent overdue logic and wrong filters).
3. One requirement translated into implementation tasks (e.g., "delete after confirmation" → confirmation dialog component + DELETE endpoint + test).
4. Did plan/code omit anything the spec required? Detect via `/speckit.analyze` + converge + acceptance-criteria checklist.
5. Which parts needed the most human review? (Clarification answers, plan approval, generated code inspection before accepting.)
6. SDD vs single-prompt build: SDD = shared artifacts, staged verification, traceable decisions; single prompt = no shared context, no consistency checks, drift undetected.

**Exam skeleton [Lab1]**
- init → constitution → specify → clarify → plan → checklist → tasks → analyze → implement → converge (+ git)
- Constitution constrains everything downstream; clarify folds answers into spec
- Incremental implementation; never approve without inspection
- Verify = manual journeys + automated tests + convergence gap report
- Analyze re-run after any upstream change

## 7.2 Lab 2: Feedforward & feedback harness for coding agents

**Scenario:** generate a small **Items API** (Node.js) from an OpenAPI contract. The coding agent receives **contract + feedforward guidance**, creates `app.js` and `app.test.js`, then **deterministic sensors** check the artifacts. If a sensor fails, **only the observed findings** are returned to the agent. The loop stops after a configured maximum of attempts.

**Master flow:**

```
FEEDFORWARD INPUTS                    GENERATION                 FEEDBACK
openapi.yaml  ─┐                                          ┌── repair feedback
style-guide.md ─┼──►  Coding agent / LLM  ──►  app.js + app.test.js
harness-rules.md┤                                          │
generation-task┘                                          ▼
                                   Static sensor ──► Dynamic test execution ──► Audit log
                                        │ failure        │ pass
                                        └─ findings only ┘
```

**Feedforward = controls applied BEFORE generation** (they constrain the code before it exists):
1. `openapi.yaml` — the **contract, single source of truth** (POST /items; 201 with `id`+`name`; 400 with `error` when `name` missing/empty/non-string).
2. `style-guide.md` — style constraints (CommonJS; Node built-ins only, no dependencies; small functions; `const` by default; JSON responses; `{ "error": "..." }` shape; no secrets; export the server; `node:test` + `node:assert/strict`; comments only where intent is not obvious).
3. `harness-rules.md` — the loop contract: treat OpenAPI as single source of truth; generate exactly two artifacts; exact assertions required (`assert.strictEqual(response.status, 201)` / `(…, 400)`); **broad assertions like `status >= 400` are forbidden**; response must be valid JSON with exactly two string properties `appJs` and `testJs`; no Markdown fences; **a repair must address findings without removing valid behavior**; **the automated loop stops after two generation attempts**; **generated files are untrusted until both sensors pass**.
4. `generation-task.md` — the task statement + required output JSON schema.

**Feedback sensors (deterministic, outside the LLM):**
- **Static sensor** (reads source, no execution): required files exist; contract-completeness patterns — test visibly exercises POST /items; **exact 201 and 400 assertions present**; server exported (`module.exports =`); JSON Content-Type set. Findings are **coded + machine-readable**, e.g. `MISSING_400_ASSERTION`.
- **Dynamic sensor** (real execution): runs `node --test app.test.js`; passes only on exit 0. **Dynamic checks are skipped until static findings are resolved.**
- Both write `sensor-report.json`; overall pass = static clean AND dynamic green.

**Bounded repair:** on failure, the repair prompt contains **only the findings array** plus previous artifacts and rules; the agent must **preserve valid behavior** and fix only the findings; **maximum 2 attempts**; everything logged.

**Auditability:** `sensor-report.json` (what failed, when) + `attempt-log.json` (attempts, findings, final status) + `manual-attempt-log.json` in Part A (includes a human-review note). Generated files stay **untrusted until both sensors pass** — and human review is still recorded even after sensors pass.

**Why "checking an assertion in source" differs from "executing the test" (classic viva/exam line):** the static sensor proves the test **states the required exact expectation** (an input-quality check on the test suite itself — a broad assertion could pass while hiding wrong behavior); the dynamic sensor proves the **implementation actually satisfies that expectation at runtime**. Static = did you write the right oracle; dynamic = does the code meet it.

**Why findings must be specific and machine-readable:** they are injected directly into the repair prompt; coded findings (`MISSING_400_ASSERTION`) let the loop repair **without human interpretation**, keep repairs minimal, and make the audit log deterministic.

**Why bounded (2) attempts:** unlimited repair invites **looping on the same failure, cost blowup, prompt creep, and overfitting to the sensor instead of the spec**; a bounded budget forces escalation to a human instead.

**Extending the harness for a new 409 response:** update `openapi.yaml` (contract first) → add the exact 409 assertion rule to harness-rules.md → extend the static sensor's required-test patterns with a 409 exact assertion → regenerate/re-run the loop → both sensors re-verify. Contract and sensors change together; the agent never decides semantics alone.

**The three parts (verification matrix):**

| Control | Part A (manual) | Part B (simulation) | Part C (live LLM) |
|---|---|---|---|
| Generation | VS Code chat agent | Predefined strings (no LLM) | External Gemini call |
| Feedforward | Files attached manually | Same artifacts in simulation | Files read + concatenated into prompt |
| Feedback | `sensor.js` run manually | `sensor.js` invoked by `demo.js` | `sensor.js` invoked by `demo1.js` |
| Repair | Student submits findings | Predetermined corrected output | Findings sent to LLM |
| Attempt limit | Student stops after 2nd | 2 | 2 |
| Audit evidence | `manual-attempt-log.json` | `demo-output/attempt-log.json` | `live-output/attempt-log.json` |

**Reproducibility vs variability (Part B vs C):** Part B is fully **deterministic/reproducible** (same inputs → same outputs; proves the harness logic itself); Part C adds a **probabilistic model** (temperature 0.1 lowers but does not eliminate variability; malformed JSON is rejected by schema validation; API 429/503 retried up to 3 times then recorded as failure) — the harness contains the variability, evidence shows which attempt came from where.

**Response-schema validation in Part C (output guardrails in action):** model response must be a JSON object with **exactly** `appJs` and `testJs` (both non-empty strings), **no Markdown fences** — validated **before any file is written**.

**Exam skeleton [Lab2]**
- Feedforward 4 files: openapi.yaml (source of truth), style-guide, harness-rules, generation-task
- Sensors: static (pattern/exact-assertion checks) → dynamic (node --test); findings only fed back; dynamic skipped until static clean
- Bounded repair: 2 attempts max; preserve valid behavior; findings machine-readable (MISSING_400_ASSERTION)
- Audit: sensor-report.json + attempt-log.json; untrusted until both sensors pass; human review still required
- A = manual loop; B = deterministic simulation, no LLM; C = live LLM, temperature 0.1, schema-validated JSON, 3 retries on 429/503

## Part VII memory map

```
Labs spine:
Lab1: constitution → spec → clarify → plan → tasks → analyze → implement → converge
Lab2: 4 feedforward files → agent → static sensor → dynamic sensor
        └─ findings-only repair (≤2 attempts) → audit logs → human review
Static = right oracle written; Dynamic = code meets oracle
```


---

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


---

# 03 — Last-Day Cheat Sheet

## 1. Fifty one-liners (recite until fluent)

**CS8 — AI-Native SDLC**
1. Traditional SDLC is process-heavy for accountability and control at each step.
2. AI collapses build to agent speed; plan, review/test, deploy stay human speed — the bottleneck moves there.
3. Line-by-line review can't scale once agents write most of the diff.
4. Governance costs rise: exceptions still route through weekly/monthly committees.
5. AI-native SDLC = automated handover + triggering of subsequent plays.
6. SDD = guardrails, requirements, constraints, acceptance criteria, edge cases defined up front, then AI generates from shared context.
7. The spec is the connective tissue across the lifecycle.
8. SDD benefits: less ambiguity, shared-truth alignment, faster generation, predictable delivery.
9. Spec Kit = Specify CLI + templates/scripts (Microsoft, open-source).
10. Kiro: Requirements → Design → Tasks → `requirements.md`, `design.md`, `tasks.md`.
11. Kiro waves: sequential across waves, concurrent within a wave.
12. Kiro steering files: `product.md`, `tech.md`, `structure.md`.
13. Playbook: six stages Plan-Design-Build-Test-Deploy-Maintain; each stage ends writing version control, next begins reading it.
14. Early stages use .md artifacts; from Build onward the artifact is code; the commit chain is the audit trail.
15. SDD challenges: brownfield hard, false sense of control, reviewing markdown over code.

**CS9 — Spec-Driven Development**
16. Elicitation → synthesis → management; elicitation is the least automated stage.
17. User story: "As a [role], I want [action], So that [benefit]"; acceptance criteria: Given/When/Then (BDD).
18. MoSCoW = Must/Should/Could/Won't; backlog re-prioritized by the Product Owner; time fixed, scope flexes.
19. AI synthesis scores ambiguity against INCOSE guidelines and EARS syntax.
20. Spec declares intent; code realizes it; edit the spec, AI regenerates code and tests.
21. SDD risks: dependency/API hallucinations, shallow edge/NFR coverage, spec-to-code drift, cascading errors.
22. Mitigations: simplicity rules in constitution; plan.md as senior-review PR gate; CI fails drift; reference files in context.
23. AD = justified choice for an architecturally significant requirement; ASR = requirement with measurable architectural effect; ADR = one AD + rationale.
24. ADR benefits: history, communication, maintenance, no repeated debates, change management; adrkit drafts/retrieves/checks.
25. Risk-control activities: requirements validation, architecture risk analysis, security & threat modelling, task decomposition + dependency validation.

**CS10 — Design & Architecture**
26. Agents have zero institutional memory — they reset every hour/day and live in the gap between mental model and code.
27. Deep module = narrow interface, big hidden functionality, minimal caller load; shallow = large interface, little hidden work.
28. Pipe leaks execution across boundaries; sink receives, works, terminates within its boundary; sink = deep module.
29. Pipe → agent traces the whole chain → context exhaustion, attention dilution, hallucination; sink → black box, contained blast radius.
30. The architecture is the prompt; the codebase outweighs the prompt; context engineering is a system design problem.
31. The architect authors the constraint: versioning policy, behavioral constraints, dependency rationale, exception logic.
32. Review changes from catching violations to confirming declared behavior.
33. Harness: LLM owns control flow (exploration + human steering); orchestrator: code owns it (unattended, predictable failure).
34. Guiding rule: the less human oversight, the more control flow must live in deterministic code.
35. Governance layers: Constraints (cursorrules/ADLs) → Conformance (fitness functions, gates, linting) → Knowledge (repo maps, ADRs from traces).
36. AI made CRUD cheap; invariants, data topology, non-deterministic boundaries stayed expensive.
37. Frame architecture decisions by blast radius, auditability, latency budgets, invariant protection — not coding time.

**CS11 — Coding**
38. Frustration loop: generate → review → not aligned → regenerate → heavy edit/abandon; caused by missing context and generic patterns.
39. Five patterns K-D-C-E-F: knowledge priming, design-first, context anchoring, encoding standards, feedback flywheel → shared mental model.
40. Guardrails: action / code-quality / security / workflow, enforced by software outside the model; layers Input → RAG → Output; implementations: rules, model-judge, middleware, HITL.
41. Lattice: skills over prompts, atoms→molecules, .lattice/ living context; refiners overlay/override; collaborative judgment presents trade-offs.
42. SEI: RevAPI (static) finds breaking changes; LLM writes recipes; engines apply; build+tests feed back until clean build; write once, roll out everywhere.
43. AI code = apparent non-determinism; spans Emergent (primary), Concurrent, Exceptional of the four sources P-E-C-E.
44. Review AI code in 8 steps — tests and static analysis first, verify intent, quality, dependencies, AI pitfalls, collaborative, automate, improve.

**CS13 — Testing**
45. Six AI testing uses T-R-S-D-V-L: test generation, risk-based prioritization, self-healing, defect prediction, visual AI, log parsing.
46. Churn: absolute = added+deleted+edited; 15–20% acceptable. Cyclomatic: independent paths; <10 low, 11–20 moderate, >20 refactor.
47. Circular validation = code + tests from the same assumptions; break it with spec-grounded tests, TDD, verification gates + mutation testing.
48. Evals: cosine/ROUGE/BLEU, LLM-as-a-judge with rubrics, thresholds (95%+ semantic alignment); agents judged on tool-calling, goal/plan execution, memory drift (component → trajectory → E2E tiers).
49. Continuous observability: data/model drift monitoring; shadow testing compares trajectory success rates safely.
50. DORA: deployment frequency, lead time, MTTR, change failure rate; AI-era: MTTR holds, CFR partial, DF/LT misleading; add rework rate (2025 fifth metric), AI code share, durability, complexity-adjusted output, innovation rate.

## 2. Mnemonics

| Mnemonic | Expands to |
|---|---|
| **A-A-F-P** | SDD benefits: Ambiguity, Alignment, Faster, Predictable |
| **P-D-B-T-D-M** | Playbook stages: Plan, Design, Build, Test, Deploy, Maintain |
| **R-D-T** | Kiro files: requirements, design, tasks |
| **P-T-S** | Kiro steering: product, tech, structure |
| **First / Anchored / Source** | Spec spectrum — human code-editing power decreases → |
| **C-S-C-P-C-T-A-I-C** | Spec Kit chain: constitution, specify, clarify, plan, checklist, tasks, analyze, implement, converge |
| **MoSCoW** | Must / Should / Could / Won't |
| **RAST** | Risk-control: Requirements, Architecture risk, Security/threat modelling, Task decomposition |
| **P-C-S-A-M** | ADR benefits: Preserve, Communicate, Support, Avoid, Manage |
| **K-D-C-E-F** | Five patterns: Knowledge priming, Design-first, Context anchoring, Encoding standards, Feedback flywheel |
| **A-Q-S-W** | Guardrail types: Action, Quality, Security, Workflow |
| **I-C-O** | Guardrail layers: Input (pre-LLM), Context (RAG), Output (post-LLM) |
| **Rules · Judge · Hooks · Human** | Guardrail implementation approaches |
| **P-S-A** | SEI roles: Providers formalize, Sustainers synthesize, App developers apply |
| **P-E-C-E → E-C-E** | Non-determinism sources; AI spans Emergent, Concurrent, Exceptional |
| **F-C-Q-D-P-C-A-I** | 8 review steps: Functional, Context, Quality, Dependencies, Pitfalls, Collaborative, Automate, Improve |
| **T-R-S-D-V-L** | AI testing uses |
| **T-G-M** | Agent testing: Tool-calling, Goal/plan, Memory drift |
| **C-T-E** | Agent test tiers: Component, Trajectory, End-to-end |
| **D-L-C-M** | DORA: Deployment frequency, Lead time, Change failure rate, MTTR |
| **S-D-C-I** | What DORA misses: Share, Durability, Complexity-adjusted, Innovation |
| **A-E-P-Q-B** | Hierarchy: Adoption, Engagement, Productivity, Quality, Business value |
| **A-C-V-Q-R** | Pillars: Adoption, Code share, Velocity (CAT), Quality, ROI |
| **L-V-P** | J-curve forces: Learning curve, Verification tax, Pipeline adaptation |
| **U-I-C** | DX AI measurement: Utilization, Impact, Cost |

## 3. Formula box

```
Absolute churn      = lines added + lines deleted + lines edited
                      (add 100, delete 30, edit 20 → 150 LOC)
Relative churn (%)  = churn / file-or-repo size × 100        [15–20% acceptable]

Cyclomatic (McCabe) = number of linearly independent paths through the CFG
Counting rule       ≈ decision points + 1
                      (if, else if, while, for, case, catch, &&, ||)
Risk bands          <10 low | 11–20 moderate | >20 high → refactor

ROI (%)             = (Value − Investment) / Investment
Value               = new revenues + cost savings
Investment          = direct hard costs (licenses, infra/compute, training)
                      + J-curve cost
J-curve cost        = staff size × fully loaded salary × productivity drop × dip duration

Net ROI multiplier  = (Productive Value of Time Saved − Rework Cost from Code Turnover)
                      ────────────────────────────────────────────────────────────
                                        Total AI Tool Cost
Time Saved Value    = hours saved/engineer/week × loaded engineering cost
Total AI Tool Cost  = seat licenses + token/usage + implementation overhead
                      ("not just the license fee — the actual spend")
```

## 4. Number anchors

| Number | Anchor |
|---|---|
| **15–20%** | Acceptable churn rate |
| **<10 / 11–20 / >20** | Cyclomatic complexity risk bands |
| **20–30%** | Selenium/Cypress suite failing per biweekly release |
| **2 engineer-days/sprint** | Locator repair burn before self-healing |
| **70–90%** | CI run-time reduction with AI-driven TIA |
| **95%+** | Semantic alignment threshold example for eval assertions |
| **2** | Max generation attempts in the Lab 2 harness |
| **0.1** | Temperature in the live harness call; 3 retries on HTTP 429/503 |
| **30 / 90 days** | Code-turnover measurement windows (AI vs human) |
| **~2× (roughly doubled)** | Code churn growth in the AI era |
| **20%** | CFR-rise example the DORA dashboard can't attribute |
| **5** | Metrics: DORA's 4 + rework rate (2025 DORA report) |
| **70% WAU / 10% share** | Shallow-adoption example (high usage, low AI code share) |
| **Dozens of versions** | How far behind OSS libraries sit (FY22 NDAA Section 835 study) |
| **6 / 4 / 3 / 2** | Playbook stages / guardrail types / guardrail layers / max repair attempts |
| **9** | Spec Kit commands |
| **5** | Hierarchy levels = pillars = AWS metric categories = DX Core 4 dimensions |
| **Wave 1 = no dependencies** | Kiro concurrency rule (waves sequential, tasks within wave concurrent) |

## 5. "See the question, write the skeleton" dispatch table

| If the question says… | Write first |
|---|---|
| "How does AI change the SDLC / velocity risks" | Bottleneck-shift flow (D1) → plan/review-test/deploy human speed → churn/debt/drift → spec + sensors + gates + DORA-plus metrics |
| "Requirements from unstructured input / elicitation" | Elicit → synthesize (INCOSE/EARS, stories+GWT) → manage (MoSCoW, traceability) → HITL sign-off → must-not-do guardrails |
| "Validate an AI-generated architecture" | RAST activities → ADRs → authored constraints → plan PR gate with senior review → conformance layer |
| "Compare two AI workflows/tools" | 2-column table → control-flow ownership → guiding rule (oversight ↓ ⇒ deterministic code ↑) → recommendation |
| "Spec vs code authority" | Spectrum First/Anchored/Source → who edits what → drift risk + CI check |
| "Agent generates code AND tests" | Circular validation: same-assumption loop → spec-oracle, TDD, gates+mutation → independent integration tests + human checkpoint |
| "Harness / feedforward / sensors" | 4 feedforward files → static → dynamic → findings-only repair ≤2 → audit → human review |
| "Test maintenance pain / flaky suites" | Self-healing (DOM, fallback locators, background) + TIA (70–90%) + QA-role shift |
| "Defect prediction" | Inputs churn+complexity+history → RF/XGBoost/GNN per PR → actions → humans still merge |
| "DORA under AI / 40% faster but defects up" | Producing≠verifying → MTTR holds/CFR partial/DF+LT mislead → add S-D-C-I + rework rate → durability evidence |
| "ROI / economics of AI" | Cost (licenses+overhead) + J-curve cost formula vs time-saved value + revenue → Net ROI subtracting turnover rework |
| "Governance / who is liable" | Humans own outcomes → guardrail inventory (A-Q-S-W) → HITL pause points → shift-left conformance → audit trail |
| "Legacy migration / library upgrade" | Decompose → RevAPI facts → LLM recipes → engine applies → build/test loop → write once roll out everywhere → approvals |
| "Non-deterministic outputs" | Apparent non-determinism → E-C-E mapping → deterministic pipeline: constrain, schema-validate, spec-grounded tests, static+dynamic, human, gates |
| "Anything from the labs" | Artifact chain + exact command/file names + verification step + audit evidence |

## 6. Lab evidence checklist (if asked to "design/justify a harness")

- [ ] Contract first: `openapi.yaml` as single source of truth
- [ ] Feedforward: style-guide.md + harness-rules.md + generation-task.md
- [ ] Exact assertions required; broad (`>= 400`) forbidden
- [ ] Output schema validated before files written (exactly `appJs`/`testJs`, no fences)
- [ ] Static sensor → dynamic sensor (dynamic skipped until static clean)
- [ ] Findings-only, coded, machine-readable repair input
- [ ] Max 2 attempts; escalate to human
- [ ] Audit: sensor-report.json + attempt-log.json (+ manual-attempt-log.json in Part A)
- [ ] Untrusted until both sensors pass; human review still recorded
- [ ] Extend = contract → rules → sensor patterns change together
