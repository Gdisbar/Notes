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
