# PART 1 — CONCEPTS
*What it is · Why we need it · How it works · Where it fits (and what it costs) · Minimal example*
REF format `[A:page]` → see `00_READ_FIRST.md`. **⚠ NOT-CARRIED (orig. slide N)** = not in your condensed PDF.

**Contents:** A SDLC & AI impact · B Paradigm shifts, economics, risks · C AI fundamentals · D Agents, harness, context, prompt, RAG, MCP · E AI-native SDLC & spec-driven development · F Requirements · G Design & architecture · H Coding & collaboration · I Testing · J Delivery metrics · K Deployment & governance · L Likely questions

---

# A. SDLC AND THE IMPACT OF AI

## A1. SDLC phases and roles `[A:1]`
| Phase | Roles | Key tasks / artefacts |
|---|---|---|
| Planning & feasibility | PM, engineering lead, stakeholders | scope, budget, resources, timelines |
| Requirements analysis | BA, Product Owner | user stories, **SRS**, acceptance criteria |
| Design & architecture | solution architect, UI/UX | architecture diagrams, DB schemas, API specs, wireframes |
| Implementation | software engineers | production code, unit tests, peer review |
| Testing & QA | QA / automation testers | functional, integration, regression, performance tests; bug tracking |
| Deployment | DevOps / SRE | CI/CD, production infra, releases, cut-over |
| Maintenance & support | SRE, support, operations | monitoring, security patches, incidents, feature requests |

## A2. Evolution of SDLC models `[A:5]` (+ ⚠ orig. 16-20, 23)
| Era | Paradigm | Iteration | Bottleneck | AI transformation |
|---|---|---|---|---|
| **Waterfall** (1970s-90s) | sequential, document-heavy | months–years | rigid handoffs, upfront planning, late feedback | AI turns unstructured documentation into specs, code and test cases |
| **Agile/Scrum** (2000s-10s) | iterative, customer feedback | weeks (2-4-week sprints) | communication overhead, context switching, manual story writing | AI generates user stories, estimates, initial PRs |
| **DevOps/CI-CD** (2010s-20s) | automated pipelines, IaC, continuous delivery | days → hours | operational handoffs, manual testing & deployment risk | AI automates pipeline failure diagnostics, auto-remediation, predicts deployment risk |
| **AI-native / agentic** (2020s+) | autonomous agents, intent-driven design | real-time / on-demand | **human verification, architectural governance, hallucination** | human shifts from author/builder to **orchestrator/reviewer** (human-in-the-loop) |
- ⚠ *Other models (orig. 17-20):* **V-model** (1980; test phase paired with each development phase), **Iterative/Incremental** (1975/78; build a subset then enhance), **DevOps** = CI/CD bridging dev and ops.
- ⚠ *Drivers of evolution (orig. 23):* faster delivery/time-to-market, continuous customer feedback, lower risk and cost, productivity & collaboration, quality & security, no-downtime updates, automation, skill shortage, new tech (hybrid cloud, containers, low-code).

## A3. Traditional vs AI-driven SDLC `[A:2, A:3-4]`
- Traditional: progress moves **stage by stage, paced by human review** (human-guided iteration). AI-driven: **progress happens continuously as systems work across stages** (autonomous iteration). `[A:2]`
- **New challenges `[A:3]`:** technical, organisational and cultural constraints → two questions: *how AI is reshaping every stage* and *how to adapt to AI-driven delivery*.

## A4. AI impact across SDLC phases `[A:6-10]`
| Phase | Traditional | AI-accelerated |
|---|---|---|
| **1. Requirements & architecture** `[A:6]` | PM writes PRD → architect designs schema/APIs over weeks | **intent-to-spec** (transcripts/prompts → OpenAPI specs & schemas); **architectural simulation** (agents test topology for concurrency bottlenecks) |
| **2. Development** `[A:7]` | manual boilerplate, doc lookups, hand-written unit tests | **copilots → autonomous agents** modifying dozens of files from one task prompt |
| **3. Testing & QA** `[A:8]` | brittle Selenium/Cypress scripts need constant updates | **self-healing suites** (agents repair locators in CI), **autonomous edge-case generation** from code branches |
| **4. Code review & governance** `[A:9]` | peer review takes 24–72 h; fatigue misses flaws | **static + semantic AI reviewers** flag OWASP-Top-10 issues, performance regressions, architecture violations before humans look |
| **5. Operations (AIOps)** `[A:10]` | 2 a.m. log-searching by hand | **autonomous remediation**: analyse logs, correlate with recent PRs, auto-generate fix/rollback, give SRE a root-cause summary |
- ✍ *Answer line:* "AI compresses the *build* step, so the bottleneck moves to **plan, review/test and deploy**, which still run at human speed" `[A:55]`.

---

# B. PARADIGM SHIFTS, ECONOMICS AND RISKS

## B1. Five shifts in software engineering `[A:11-15]` (+ ⚠ orig. 30)
| # | Shift | Old assumption → new reality | Example |
|---|---|---|---|
| 1 | **Dev effort is no longer the bottleneck** | scarcity drove MVP scope-cutting, UX-first, rigorous prioritisation → if implementation ≈ free, think about *what is worth building and verifying* | parallel exploratory prototypes instead of one MVP; design docs become **machine specifications** not human-coordination docs |
| 2 | **Roles are less siloed** | PM → dev → QA → SRE hand-offs blur | PM/dev blur; dev/QA/SRE blur (spec author, reviewer, operator overlap) |
| 3 | **Decisions are less "hard to change"** | architecture as one-way door because refactoring took months → regenerating code is cheap | **disposable microservices**; code as abundant commodity |
| 4 | **Accountability is foggier** | "you build it, you run it" → when AI writes ~80% of code/tests/PRDs, who is accountable? | security vulnerability / supply-chain attack; compliance & algorithmic auditability |
| 5 | **Outcomes are probabilistic, not guaranteed** | deterministic input→output → models are non-deterministic | static unit tests → **continuous LLM evaluation suites (evals)** |

## B2. Value inversion and new competencies `[A:16-18]`
- **Enduring value of engineers:** shifts from *writing* software to **designing, orchestrating, verifying and governing** intelligent systems (⚠ orig. 36-37).
- **Diagnostic criteria `[A:17]`:** Traditional → Agentic AI era: *primary artefact* **laborious scarcity → disposable abundance**; *engineer's role* **manual authorship → strategic orchestration**; *bottleneck* **writing syntax → verifying semantics**; *success* **lines of code/PR throughput → decision velocity & system reliability**; *pedagogy* **language memorisation → architectural trade-offs**.
- **Four new core competencies `[A:18]`:** intent articulation & architectural control · systematic verification & QA · multi-agent orchestration · human judgement & accountability.
- ⚠ *Transition needs four dimensions (orig. 41):* **education** (syntax mastery → systems curation), **tooling** (orchestration & verification infrastructure), **processes** (verification-first, human-in-the-loop lifecycles), **professional practice** (roles, metrics, governance).

## B3. Shifts in software economics `[A:19-20]` (+ ⚠ orig. 44)
1. **End of zero-marginal-cost software:** AI features carry **variable compute cost per transaction** (inference tokens, API calls, vector DB searches) → engineers must design for **cost efficiency per query**; one feature can destroy gross margin. `[A:19]`
2. **Build vs buy → token cost vs human labour:** old = developer salary vs vendor subscription; new = **inference cost + human verification cost vs full human execution cost**. `[A:20]`
3. ⚠ **Pricing: per-seat → outcome/usage-based:** AI agents do the work, so seat licences collapse; pricing moves to **per outcome/usage** (e.g., $2 per resolved ticket, $10 per automated deployment); architects must meter outcomes.

## B4. Risks across the SDLC `[A:21]`
- **Code quality & security:** plausible-but-insecure code; **hallucinated package dependencies (supply-chain attacks)**; missing edge cases in generated tests; architectural drift / unchecked tech debt ("**code slop**"); code bloat; **context blindness**.
- **Human factors:** reviewer fatigue; atrophy of fundamental skills.
- **Legal/IP/compliance exposure**; **economic & operational** risks.

---

# C. AI FUNDAMENTALS

## C1. AI, ML and capabilities `[A:22-23]` (+ ⚠ orig. 48-54)
- **AI** = systems that perform tasks needing human intelligence. Four capabilities: **learning, reasoning, perception, problem-solving/action**.
- **ML** = statistical models that train on data and improve **without explicit programming**: **supervised** (labelled data), **unsupervised** (clusters/patterns, e.g., recommendations), **reinforcement** (trial-and-error with rewards). *All ML is AI; not all AI is ML* (rule-based/symbolic AI uses if-then rules).
- ⚠ *Capabilities in detail:* **reasoning** — symbolic/knowledge-graph logic, probabilistic inference, chain-of-thought; **problem-solving** — search/optimisation (A*, Monte-Carlo Tree Search), agentic planning; **perception** — computer vision (CNNs, vision transformers), speech/audio.
- ⚠ *Evolution:* symbolic logic → traditional ML (1990s-2010s) → deep learning & foundation LLMs (2017-2023) → **test-time-compute reasoning models (2024-)**.

## C2. LLMs, foundation models, transformers `[A:24-27]`
- **Language model** encodes statistical information about language (how likely a token is in a context); the basic unit is the **token**; trained by **self-supervision** (no labelled data) which made scaling possible; "large" = measured by number of parameters. `[A:24]`
- **Foundation model** = massive, general-purpose model trained on multimodal data by self-supervised learning; the "base". `[A:25]`
- **Transformer** (2017) — **self-attention** lets the model weigh different tokens at different moments (⚠ orig. 57).
- **Reasoning model `[A:26]`:** an LLM fine-tuned for multi-step problem solving that generates **hidden intermediate "reasoning traces"** (chain of thought) before the answer; spends **extra compute at inference** to test hypotheses and verify logic.
- **System 1 vs System 2 `[A:27]`** (Kahneman): fast, automatic, intuitive vs slow, deliberate, effortful → standard LLMs ≈ System 1; reasoning models ≈ System 2. ⚠ *S2A (System 2 Attention)*: an early inference-scaling method that rewrites the prompt before answering.
- ⚠ *Use case (orig. 61, code bug fixing):* traditional ML can *locate* a likely buggy file but can't fix; a standard LLM writes a plausible patch in one pass but may miss race conditions/breaking API changes; a **reasoning model** deliberates, tests hypotheses and checks edge cases.

## C3. Tokens and tokenization `[A:28-30]` (+ ⚠ orig. 64-71)
- **Token** = character, word or part of a word; **tokenization** = splitting text into tokens. Rule of thumb: **1 token ≈ 4 characters ≈ 0.75 words → 100 tokens ≈ 75 words**; short common words = 1 token; rarer words split ("token"+"ization"); punctuation/whitespace also cost tokens. `[A:28]`
- **Why tokens (not words/characters):** meaningful sub-parts, **smaller vocabulary** (efficiency), handles **unknown words**. Techniques: word, character, **subword (BPE, SentencePiece)** — modern standard. ⚠
- **How the model "sees" code `[A:29]`:** common words = one token; **custom identifiers split** (e.g., `getUserPaymentHistory`); **cryptic identifiers = high token cost**. Because models operate on subword IDs, tokenization causes **subtle failure modes** in generation/analysis (character-level tasks, whitespace). ⚠ *Code-aware tokenizers* (GPT-4, Llama 3, Claude 3) treat code structure better than older ones (≈10 vs ≈19 tokens in the deck's example).
- **System-level interventions `[A:30]`:** **code interpreters** (model writes code, runs it in a sandbox, reads the answer — delegate character-level/computational tasks) and **constrained decoding / structured-output enforcers** (**mask invalid tokens** so output is valid JSON/syntax).

## C4. Context window and tokenomics `[A:31-33]`
- **Context window** = the amount of text (tokens) the model can consider at once; **input + output together** must fit under the limit. Counts: system prompt/custom instructions, conversation history, current prompt (+ files/images), **and the response being generated**. `[A:31]` (⚠ orig. 74)
- **Failures `[A:32]`:** **silent degradation**, **attention dilution ("lost in the middle")**, **inconsistent behaviour**, **cascading failures** → *"don't build agents without tracking token consumption."*
- **Tokenomics `[A:33]`:** the "black box" of AI billing — hidden costs beyond the token, **inability to calculate ROI**, **inefficient model routing and waste** (Linux Foundation's Tokenomics Foundation).

---

# D. AGENTS, HARNESS, CONTEXT, PROMPTS, RAG, MCP

## D1. AI agents `[A:34-35]`
- **Agent** = autonomous software entity that **perceives** (inputs/sensors), **reasons/decides**, and **acts via tools** to achieve goals, in a **Perceive → Reason → Act → Evaluate loop**, with autonomy **inside defined boundaries**; unlike static rule software or a plain LLM that merely answers a prompt. `[A:34]` *Example:* SWE-agent — environment = computer/terminal; actions = navigate repo, search, view, edit lines (⚠ orig. 79).
| Type `[A:35]` | Mechanism | Example |
|---|---|---|
| Simple reflex | IF/THEN on current percept | thermostat |
| Model-based reflex | internal state/world model | automated braking on wet road |
| Goal-based | planning toward a target | GPS navigation |
| Utility-based | multi-objective utility trade-off | ride-share dispatcher (wait time vs distance) |
| Learning | feedback loop adapts behaviour | Netflix/Spotify recommendations |

## D2. Harness engineering `[A:36-39]`
- **Agent = Model + Harness.** The model supplies general intelligence; the **harness** supplies what the model cannot do alone — **tools, state/execution, context, permissions and validation**. Harness engineering = building the environment that makes models reliable, autonomous agents. `[A:36]`
- **Three harness layers `[A:37]`:** **Coding harness** (runtime built into the agent: tool use, system prompt, sub-agents, code search, context mgmt — controlled by the agent/SDK maker); **User harness** (convention files, MCP servers, eval loops, custom skills, `CLAUDE.md` — controlled by the deploying team); **Team/Org/SDLC harness** (context lake, integrations, agent registry, permissions, orchestration, governance, HITL, measurement — controlled by the platform team).
- ⚠ **Guides & sensors (orig. 84):** harness = a "cybernetic governor". **Guides (feedforward)** steer *before* action (system prompts, specs, `AGENTS.md`/`CLAUDE.md`); **sensors (feedback)** observe and enable self-correction *after* action (linters, tests).
- **Computational vs inferential controls `[A:38]`:** *computational* = fast, **deterministic**, CPU-run (type checkers, linters, pre-commit structural tests); *inferential* = semantic, **probabilistic**, GPU/NPU (**LLM-as-judge**, autonomous review). ⚠ Examples (orig. 86): coding conventions = feedforward/inferential; code mods (OpenRewrite) = feedforward/computational; ArchUnit structural tests = feedback/computational; review instructions = feedback/inferential.
- **Where harness fits in the SDLC `[A:39]`:** Plan → context management decides what the agent sees; Build → tools, orchestration, guardrails limit actions; Review → sensors (tests, linters) catch problems; Deploy → **approval gates** decide what runs automatically vs waits for a human.

## D3. Context engineering `[A:40-42]` (+ ⚠ orig. 89, 91-92)
- **What:** deliberately **designing, structuring and optimising the context** given to an LLM so outputs are accurate, relevant, reliable — "filling the context window with just the right information at each step". Karpathy: **LLM = CPU, context window = RAM** (limited working memory). `[A:40]`
- **Why `[A:41]`:** LLMs/agents have **no long-term memory**; they answer only from the context at inference time → context structure drives output quality.
- **Process:** selection, structuring, prompt design, **compression**, sequencing, tool & memory integration; or retrieval, generation, processing, management. ⚠
- **Relation `[A:42]`:** context engineering **includes** prompt engineering and **RAG**; prompt engineering is a narrower subset (ignores system prompts and other inference-time information). Harness engineering uses context engineering to make guides and sensors available.

## D4. Prompt engineering and RAG `[A:43-45]`
- **Prompt** = instruction to a model; prompt engineering matters because it shapes quality, relevance, accuracy; *the problem is when it is the only skill people know*. System prompt = developer's instructions; user prompt = user's. ⚠
- **Techniques `[A:43]`:** **zero-shot** (no examples), **few-shot** (a few examples), **chain-of-thought** (ask for step-by-step reasoning) (+ role/format specification 🆕).
- **Prompt vs context engineering `[A:44]`:** early LLM work was mostly prompting; now context engineering is the bigger component.
- **RAG `[A:45]`:** retrieve from an **authoritative external knowledge base** before generating. Steps: **Ingestion** (chunk documents → embed as vectors → store in vector DB) → **Retrieval** (search by meaning) → **Augmentation** (add retrieved chunks to the question) → **Generation**. Reduces hallucination and stale knowledge without retraining.

## D5. Maturity: prompt → context → harness → loop → graph `[A:46-48]`
| Phase `[A:46]` | Period | Discipline | Focus | Output |
|---|---|---|---|---|
| 1 | 2022-23 | **Prompt engineering** | how we talk to the model (syntax/phrasing) | code snippets, autocomplete |
| 2 | 2024-25 | **Context engineering** | what the model knows (relevance & memory) | feature logic, context-aware updates |
| 3 | 2026 | **Harness engineering** | how the model is allowed to act and self-correct (autonomy & control) | end-to-end task execution and verification |
- **Two more `[A:47]`:** **Loop engineering** — design automated cycles where an agent acts, tests, adjusts until a goal is met; **Graph engineering** — design the graph the agentic system executes (nodes that do work, who checks whom, what runs when).
- **Three-layer operating model `[A:48]`: Spec** (human-authored intent & constraints) → **Harness** (mechanised validation: tests, security gates) → **Loop** (agent execution until convergence). **Three gates of autonomy:** **Human-in-the-loop** (manual review of every action; high-risk), **Human-on-the-loop** (agent runs, humans monitor dashboards), **Autonomous** (bounded, routine, well-harnessed tasks).

## D6. Prompt vs context vs harness; AI vs ML engineering `[A:49-50]`
| | Prompt | Context | Harness |
|---|---|---|---|
| What | the task/question | reference material | testing/validation machinery |
| Audience | LLM | LLM | AI engineer |
| Lives in | user input / app logic | vector DB / system instructions | CI/CD, test suite |
| Goal | right answer now | prevent hallucination | prove it works at scale |
- **Prompt engineering vs fine-tuning `⚠ orig. 107`:** prompting adapts a model **without updating weights**; fine-tuning **updates weights**.
- **AI vs ML engineering `[A:50]`:** ML = data → train model → product last; AI engineering with ready models = **build the product first**, invest in data/models only when it shows promise; ML often has disjoint model/product development.

## D7. Model Context Protocol (MCP) `[A:51]` (+ ⚠ orig. 111-112)
- **What:** open standard for connecting AI apps to external systems — data sources (files, DBs), tools (search, calculators), workflows. "**USB-C port for AI**".
- ⚠ **Architecture:** client-server — **MCP Host** (AI app coordinating clients), **MCP Client** (maintains connection to a server, gets context), **MCP Server** (provides context/tools). Transport: **JSON-RPC 2.0** messages (stdio / HTTP). Flow: LLM recognises it can't query a DB/send email → client **discovers tools** on servers → LLM requests tool call → server executes → result returned.
- **Governance link:** MCP servers are **untrusted tools / excessive agency** risk `[A:124]`.

## D8. AI stack and AI-enabled systems `[A:52-53]`
- **Three layers:** **application development** (interface, prompt/context engineering, evals), **model development** (inference optimisation, dataset engineering, training), **infrastructure** (compute, data management, serving, monitoring). `[A:52]`
- **Engineering AI-enabled systems `[A:53]`:** adding ML makes the lifecycle **data-centric** and introduces **probabilistic uncertainty**; classic design techniques remain starting points; the new emphasis is on quality attributes such as **explainability, data centricity, verifiability**, plus security/usability/privacy.

---

# E. THE AI-NATIVE SDLC AND SPEC-DRIVEN DEVELOPMENT (SDD)

## E1. Traditional vs AI-native SDLC `[A:54-56]`
- **Traditional** is **process-heavy** for accountability; *before agents* every stage runs at human speed, with **Build** the long block. *After agents* **build runs at agent speed** and the saved cycle time is "reclaimed" — plan/design (requirements), test (review), deploy/maintain (release) remain. `[A:54]`
- **When code is no longer the bottleneck `[A:55]`:** the bottleneck moves to plan, review/test and deploy; **controls stop matching reality** (line-by-line review can't keep up once agents write most of the diff); **governance costs rise** (weekly/monthly committees); *build collapses to agent speed, everything else stays human speed*.
- **AI-native `[A:56]`:** **automated hand-over and triggering** of the next "play"; removes manual hand-offs. Traditional = *the line* (one slow loop back = new release cycle); AI-native = *the loop* (hours not weeks; humans above the loop instigating, directing, governing).

## E2. Spec-Driven Development `[A:57]` (+ ⚠ orig. 121)
- **Problem:** teams ship software that works but **misses original intent**; prompt-first workflows break down as scope grows.
- **SDD:** a **spec-first approach** — teams define guardrails, requirements, constraints, acceptance criteria, edge cases **up-front**; AI then generates code, tests and artefacts from that shared context.
- **Benefits (⚠ orig. 121):** less ambiguity/rework; shared source of truth across product/engineering/test; faster implementation against structured context; more predictable delivery (validation tied to intent).
- **Three SDD levels `[A:61, A:67]`:** **Spec-first** (written before coding; may be dropped afterwards) → **Spec-anchored** (kept and synced with code throughout the lifecycle) → **Spec-as-source** (spec is the main source; humans edit only the spec, never the code).
- **Challenges `[A:62]`:** hard to evaluate for **brownfield** code; **false sense of control**; reviewing markdown instead of code.

## E3. Tools `[A:58-60]`
- **GitHub Spec Kit** (open source, Microsoft): **Specify CLI** (bootstraps SDD scaffolding, downloads templates for your coding agent) + **templates & helper scripts** (what a spec/plan/task breakdown looks like). `[A:58]` ⚠ *Commands (orig. 144):* `/speckit.constitution` (project principles, at the start) → `/specify` (feature spec) → `/clarify` (resolve ambiguity) → `/plan` (technical plan) → `/checklist` (quality checks) → `/tasks` (actionable tasks) → implement.
- **Kiro** (AWS): workflow **Requirements → Design → Tasks**; each spec yields `requirements.md` (or `bugfix.md`), `design.md`, `tasks.md`; **feature specs** vs **bugfix specs**; **steering files** (`product.md`, `tech.md`, `structure.md`) give persistent project context; ⚠ tasks are grouped into **waves** by dependency graph (independent tasks in a wave run concurrently; waves run sequentially). `[A:59]`
- **AI-Native SDLC playbook (Anthropic) `[A:60]`:** *plays* grouped in six non-linear stages — **Plan, Design, Build, Test, Deploy, Maintain**; each play states what changes, getting started, steps, governance, how to measure. ⚠ *Each stage ends by writing an artefact to version control and the next begins by reading it; early artefacts are `.md` files, later code; the chain of commits is the audit trail* (who asked, what the agent produced, who approved).

---

# F. REQUIREMENTS IN THE AI ERA

## F1. Elicitation, synthesis, management — traditional vs AI `[A:63-64]` (+ ⚠ orig. 133, 135-137, 140)
| Stage | Traditional | AI-driven |
|---|---|---|
| **Elicitation** (discover needs) | interviews, workshops, observation, document analysis, surveys, process mapping, prototyping; agile: user-story mapping, sprint demos | AI ingests **unstructured data** (tickets, logs, transcripts, reviews), **implicit-need & telemetry analysis**, **synthetic user personas & autonomous interviewing**, **instant functional prototyping** `[A:63]` |
| **Synthesis** (structure it) | BRD/SRS documents; user stories "As a …, I want …" | **automated artefact conversion**, **real-time ambiguity & quality scoring** against standards (INCOSE, EARS syntax) `[A:64]` |
| **Management** (track/prioritise/control change) | MoSCoW, product backlog | **live bidirectional traceability**, **proactive drift & scope-creep monitoring**; humans still approve (trade-offs, legal accountability, stakeholder relationships) |
- *Elicitation is one of the least automated stages of delivery* (⚠ orig. 133).

## F2. How SDD transforms the requirements lifecycle `[A:65]`
- The **spec declares intent; code realises it**. Specs live in version control beside/above code. When business needs change, **edit the spec** and let AI refactor code and regenerate tests. Elicitation feeds **bounded guardrails** — what the feature **must not** do — to restrain the AI.

## F3. Risks of SDD and their mitigations `[A:68-69]`
- **Risks `[A:68]`:** moving architecture/design into markdown (`spec.md`, `plan.md`, `constitution.md`) brings: **dependency & API hallucinations**, **shallow edge-case and non-functional coverage**, **spec-to-code drift**, **cascading error propagation**.
| Problem `[A:69]` | Root cause | Mitigation |
|---|---|---|
| **Architectural bloat** | LLM tends to maximal complexity | **simplicity rules in `constitution.md`** ("prefer single relational tables over new microservices") |
| **Cascading errors** | unchecked step from `/plan` to `/tasks` | **treat `plan.md` as an explicit PR gate**; mandatory senior review before `/tasks` |
| **Spec-to-code drift** | unlinked code changes during implement | **CI check** that fails if code touches architecture files without spec updates |
| **Legacy context loss** | LLM context limits | include **reference files** (schema dumps, OpenAPI contracts) in the planning context |
- ⚠ *Risk-control activities across SDD (orig. 149):* requirements validation · architecture risk analysis · security & threat modelling · task decomposition & dependency validation.

## F4. ADR — Architecture Decision Records `[A:70]`
- **AD** = justified design choice addressing an **architecturally significant requirement (ASR)**; **ADR** = record of **one** decision and its rationale/trade-offs. Benefits (⚠ orig. 148): preserve decision history, improve communication, support maintenance, avoid repeated debates, manage change. In AI era: **agents draft ADRs; humans decide and sign** `[A:126]`.

---

# G. DESIGN AND ARCHITECTURE IN THE AI ERA

## G1. Agent vs intern; managing complexity `[A:71-72]` (+ ⚠ orig. 152, 154-157)
- **The new intern vs the agent `[A:71]`:** a human builds a **shared mental model** over weeks; an **AI agent resets every hour/day and never accumulates understanding** — zero institutional memory. A chronic gap separates the **conceptual design (in engineers' heads)** from the **dependency structure in the code** — *agents live in that gap*.
- **Design = managing complexity** by *eliminating* it or *hiding* it through modular design (Ousterhout: "design it twice"). ⚠
- **Deep vs shallow modules `[A:72]`:** **deep** = simple narrow interface hiding substantial functionality (high leverage, low cognitive load); **shallow** = big interface, little functionality hidden (little leverage). Revives classic **information hiding**.
- ⚠ **Sinks vs pipes:** a *pipe* keeps leaking execution across boundaries (cascading triggers, queues, side-effects); a *sink* takes an input, does its work, and **terminates inside its boundary** → a sink is a deep module; **contained blast radius**.
- ⚠ **Cognitive load → context limits:** human cognitive load becomes the agent's **finite context window/attention budget**; when an agent meets a pipe it must load and trace the whole downstream chain. The remedies are the "same old story": low coupling, high cohesion, explicit boundaries, honest interfaces, minimal side effects, graduated layers of abstraction.

## G2. The architecture is the prompt `[A:73-76]`
- **Codebase outweighs the prompt:** no prompting technique, framework or model can rescue **tangled modules and cascading side-effects**; well-defined boundaries, clear interfaces, minimal side effects, high cohesion, low coupling. ⚠ *Context engineering is now a first-class architecture discipline.* `[A:73]`
- **Governance through authored constraints `[A:74]`:** architect **authors the constraints** agents cannot infer (versioning policy, behavioural constraints, dependency rationale, exception logic). Review changes purpose: from **catching violations to confirming the system does what the constraint intended**.
- **Orchestrator vs agent harness `[A:75]`:** *Agent harness* — **LLM owns control flow** (loop; model decides next step; best for open-ended exploration with human steering). *Orchestrator* — **code owns control flow** (structured, repeatable stages; deterministic reliability but rigid). *Prompt logic = flexibility at the cost of reliability; code = reliability at the cost of rigidity.*
- **Three-layer governance framework `[A:76]`:** **Constraints layer** (cursor rules, Architecture Description Languages), **Conformance layer** (architectural **fitness functions**, dependency rules, plan-build approval gates, linting *before* code propagates), **Knowledge layer** (repository maps; decisions extracted from agent reasoning traces into ADRs).

## G3. Cost-of-change mapping (activity) `[A:77-78]`
> *Classic definition:* architecture = decisions that are **difficult to change**. Is that still true when code is cheap?
| Architectural dimension | Traditional cost | AI-augmented cost | Why |
|---|---|---|---|
| **Business logic & CRUD APIs** | High (manual coding/testing) | **Low** | LLMs generate boilerplate, tests, standard transformations |
| **System invariants & state consistency** | High | **High / critical** | AI cannot bypass distributed locking, ACID guarantees, race conditions |
| **Data topology & storage model** | High | **Very high** | migrating multi-TB production stores stays risky however fast code is written |
| **Non-deterministic boundaries** (AI components) | Non-existent | **High (new cost)** | fallback policies, latency budgets, guardrails add structural complexity |
- ⚠ **Shift in architects' activities (orig. 165):** less time on method signatures, **more on system invariants** ("balance never below zero", "PII never crosses region"); manage the **non-deterministic surface area**; **spec as the new codebase**; frame decisions around **blast radius, auditability, latency budgets** rather than "how long to code". *Architecture is not dead — its coordinates have moved.*

---

# H. CODING AND COLLABORATION PATTERNS

## H1. From autocomplete to structured collaboration `[A:79]` (+ ⚠ orig. 167-169, 171)
- **Friction:** the **frustration loop** — generate → review → *not aligned* → regenerate → heavy edit/abandon; the model defaults to **generic internet-average patterns** without project context; output may be syntactically right but **architecturally misaligned**; correction can consume the speed gain.
- **Speed trap — measure the right outcome:** use **first-pass acceptance rate** (not time to first output), **iteration cycles per task** (not lines generated), **post-merge rework**, **review burden vs manual** (not generation speed).
- **Tool-to-teammate shift (onboarding → knowledge priming; whiteboarding → design-first; review norms → encoded standards; shared decisions → context anchoring; retrospective → feedback flywheel).**

## H2. Five patterns of effective AI collaboration `[A:79-82]`
| Pattern | What to do | Failure mode it addresses |
|---|---|---|
| **1 Knowledge priming** `[A:80]` | share curated project context first (stack versions, structure, naming rules, architecture, examples) as reusable infrastructure | AI defaults to generic patterns |
| **2 Design-first collaboration** `[A:80]` | move progressively: capabilities → components → interactions → contracts, *before* code | code written before the design is agreed |
| **3 Context anchoring** `[A:81]` | keep a **living feature document** (decisions, constraints, open issues, state) to restore context across long sessions | functional code that violates team expectations; lost context |
| **4 Encoding team standards** `[A:81]` | convert tacit standards into explicit artefacts the AI follows | inconsistent style/quality, rework |
| **5 Feedback flywheel** `[A:82]` | capture review findings, good prompts, failure patterns; feed back into context/standards | repeating the same AI mistakes |
- **Shared mental model & trade-offs (⚠ orig. 175):** the patterns shift effort from constant vigilance to expressing intent; cost = upfront discipline and maintenance of priming docs — **not worth it for trivial one-off utilities**.

## H3. Guardrails `[A:83-84]` (+ ⚠ orig. 177, 179)
- **Guardrail** = rule/constraint/control that **prevents an AI system from unsafe, incorrect, unauthorised or undesirable actions** (e.g., stop a coding agent on a payment app from deleting production data, adding insecure code, merging untested changes). Types: **action guardrails** (prevent dangerous actions) and content/behaviour guardrails.
- **Principle:** don't rely on telling the AI to behave — **enforce outside the model** wherever possible.
- **Core layers `[A:84]`:** **1 Input** (pre-LLM: prompt-injection detection, PII redaction/tokenisation, topic & policy filtering) → **2 Context & retrieval** (RAG) → **3 Output** (post-LLM: hallucination & grounding checks, format/schema validation).
- ⚠ **Implementation:** deterministic rules (regex/blocklists), **model-based evaluation** (smaller LLM/classifier scoring safety/tone), **middleware & hooks** (LangChain, n8n), **human-in-the-loop** pause for high-risk actions.

## H4. Lattice `[A:85-86]`
- **Lattice** = installable, repository-based framework that **operationalises the five patterns**. Principles: **skills over prompts** (skills are shared, version-controlled, evolve via PRs; prompts are personal and ephemeral), … Building blocks: **Atoms** (reusable skills: knowledge-priming, design-first, context-anchoring, clean-code, DDD, secure-coding, test-quality), **Molecules** (composed workflows: design-blueprint, code-forge, review), **Refiners** (guided interviews that tailor standards to the team).

## H5. Use case — automating library replacement (SEI) `[A:87-88]` (+ ⚠ orig. 183-185, 187)
- **Problem:** open-source libraries are often dozens of versions behind (security and operational risk); manual upgrades are slow, error-prone.
- **Solution:** **decompose the task** and pair gen-AI with **deterministic tools** — **static analysis (e.g., RevAPI)** identifies API **breaking changes** completely and consistently; **LLMs** synthesise **transformation strategies** where no automated recipe exists; build + unit tests act as final check. Roles: library providers formalise breaking changes; sustainers use LLMs for transformation strategies; application developers apply them and run build/test. `[A:87]`
- **Mapping to harness engineering:** RevAPI = **computational sensor**, LLM = **inferential** generator, build/tests = **feedback sensor** → a complete **feedforward–feedback loop** (Spec → Harness → Loop): "write once, roll out everywhere". `[A:88]`

---

# I. TESTING

## I1. Testing in a non-deterministic world `[A:89-92]` (+ ⚠ orig. 191)
- Assumption "same input → same output" no longer holds; bugs become **rare, intermittent, hard to reproduce**; **one test run per requirement is insufficient**. `[A:89]` Drivers: agile/DevOps regression, multicore/multithreading races, cyber-physical sensors, **ML/AI adaptive systems**. ⚠
- **Actual vs apparent non-determinism `[A:90]`:** *actual* — no theoretical way to predict (quantum RNG); *apparent* — **no practical way to predict due to complexity or poor controllability/observability**. **AI-generated code and agentic development = apparent non-determinism.**
- **Four categorical sources `[A:91-92]`:** physical, **emergent**, **concurrent**, **exceptional**. AI-generated code spans three: **Emergent** (primary: agents choose libraries/glue code; small prompt/sampling changes → different architectural assumptions → hidden couplings), **Concurrent** (agent loops, parallel tool calls → race conditions, ordering variations), **Exceptional** (hallucinated edge cases and fault handling that don't match reality).

## I2. Reviewing AI-generated code `[A:93]`
Eight steps: **1** functional checks (run tests + static analysis first) → **2** verify context & intent/architecture fit → **3** assess quality → **4** **scrutinise dependencies** → **5** spot **AI-specific pitfalls** → **6** collaborative reviews → **7** automate what you can → **8** keep improving the workflow.

## I3. How AI is used in testing `[A:94-98]` (+ ⚠ orig. 197, 202)
- **Uses (⚠ 197):** automated test generation, **risk-based prioritisation**, **self-healing scripts**, **defect prediction**, visual AI/anomaly detection, log parsing & root-cause diagnostics.
- **Traditional vs AI-augmented `[A:94]`:** maintenance (hand-fixing locators → **self-healing**); timing (after code → **shift-left**, checks at PR/design phase); QA role (manual scripts → **quality engineering**, exploratory testing, orchestrating agents); regression (full suite → **impact-based selection**); barrier (Selenium/Cypress skills → **natural-language test creation**).
- **Self-healing `[A:95]`:** UI changes break scripts — a team shipping every two weeks may see **20–30% of the suite fail per release**, costing about **two engineer-days per sprint** in locator repairs; AI watches the DOM, finds the new locator with **multiple fallback strategies**, repairs and confirms the fix between deployments.
- **Defect prediction `[A:96]`:** identify high-risk **hotspots** (files/functions/modules) **before** tests run, from **code churn & commit metadata, complexity metrics, defect history**; **just-in-time (JIT)** supervised models (random forest, XGBoost, graph neural nets on ASTs) score every PR.
- ⚠ **Test selection/prioritisation (orig. 202):** *full suite* = 100% tests, hours, very high CI cost; *static rule-based* = file-path mappings, minutes but misses indirect edge cases; **AI-driven predictive Test Impact Analysis** = ML-selected subset, uses failure statistics to separate flaky from real regressions, maps cross-service/API dependencies, **cuts run time 70–90%**.
- ⚠ **Code churn (orig. 203):** Absolute churn = LOC added + deleted + modified (add 100, delete 30, edit 20 ⇒ **150 LOC**); **Churn rate % = (added+deleted+modified) ÷ total base LOC × 100**; **15–20% considered acceptable**.
- ⚠ **Cyclomatic complexity (orig. 204):** **V = 1 + number of decision points** (if, else-if, while, for, case, catch, &&, ||); **< 10 low risk; 11–20 moderate; > 20 high — refactor.**
- **How churn & complexity drive prediction `[A:97]`:** high churn = **instability hotspots**; high complexity **exceeds human cognitive limits**; systems ingest churn, complexity, author overlap, defect tags to output actions (extra review, extra tests…).
- **Circular validation `[A:98]`:** the same AI writes **both code and tests from the same assumptions**, so tests pass without proving anything → **ground tests in specifications, not implementation; use TDD; computational verification gates and mutation testing**; validate against independent ground truth.

## I4. Testing AI-enabled software `[A:99-102]` (+ ⚠ orig. 207, 212)
- **Shift:** deterministic *(input → logic → output)* vs **probabilistic** *(input + context + model parameters → output)* `[⚠ orig. 207]`.
- **Model evals `[A:99]`:** instead of exact matches use **similarity metrics** (cosine similarity, ROUGE, BLEU), **LLM-as-a-judge** against structured rubrics, **assertion thresholds**.
- **Testing agents `[A:100]`:** evaluate the **reasoning chain (ReAct loop)** and **tool use** — tool-calling validation (right tool, valid JSON params, graceful API-failure handling) and **goal & plan execution** of multi-step objectives.
- **Continuous monitoring `[A:101]`:** testing becomes **continuous production observability** — **data & model drift** detection, **shadow testing / A/B trajectory analysis** (route real traffic to new prompts/models in shadow mode).
- **Three agent-testing strategies `[A:102]`:** LLM-as-judge · **three-tier approach** (**component → trajectory → end-to-end**) · **simulated environments** — all with **human-in-the-loop practices**.
- ⚠ **Uber's software factory (orig. 212):** inner loop (ideation & alignment) → autonomous execution & validation → outer loop & maintenance; six infrastructure pillars: **model gateway, MCP gateway & tool optimisation, agentified DevPods, managed skills marketplace, AI context graph, AI assistant**.

---

# J. SOFTWARE DELIVERY METRICS

## J1. DORA `[A:103-107]` (+ ⚠ orig. 214, 217, 220)
| Metric `[A:104]` | Measures | Assumption |
|---|---|---|
| **Deployment frequency** | how often code ships to production | more = healthier |
| **Lead time for changes** | commit → production | shorter = better flow |
| **Mean time to recovery (MTTR)** | incident resolution speed | faster = more resilient |
| **Change failure rate (CFR)** | % of deploys causing failure | lower = higher quality |
- Principle: **throughput and stability — don't trade one for the other** (⚠ orig. 214; "SDO performance: adding availability").
- **AI impact `[A:105]`:** DORA was designed for human-written code. **AI collapses the cost of producing code, not of verifying it**, so throughput and stability pull apart. The dashboard may show CFR up 20% but **can't say whether AI caused it**; deployment frequency may rise from micro-commits. **MTTR holds up well; CFR partial value; deployment frequency and lead time become misleading.** Code churn roughly **doubled** without DORA flagging it (rewritten code didn't fail in production).
- **What DORA doesn't measure `[A:106]`:** **AI code share**, **code durability** (does it stick?), **complexity-adjusted output** (auth system vs README deploy), **innovation rate** (new features vs maintenance).
- **Evolved metrics `[A:107]`:** **rework rate** added as a **fifth DORA metric (2025)** (unplanned production fixes); *TrueThroughput* (complexity-adjusted deployment frequency), **PR cycle time** (isolates review time — the "verification tax"), **code durability & PR revert rate**, **MTTR** still valid, paired with adoption metrics. ⚠ *Also capture:* cycle time, **PR pickup time**, review turnaround, defect rates, test coverage, churn patterns, new-vs-maintenance ratio (orig. 220).

## J2. Emerging AI-era frameworks `[A:108-114]` (+ ⚠ orig. 222-223, 225-226, 230, 232-235)
- **AI impact hierarchy `[A:108]`:** **1 Adoption** (are devs using it?) → **2 Engagement** → **3 Productivity** → **4 Quality** (is AI code maintainable?) → **5 Business value** (is it paying off?). Most orgs stop at 1 or measure 3 with inflated metrics.
- **Developer AI Impact Framework — five pillars:** **1 AI adoption** (active-user rate DAU/WAU/MAU, tool adoption by type) ⚠ · **2 AI code share** (AI-assisted PR %, % committed code AI-generated; 70% WAU but 10% share = shallow use) ⚠ · **3 Velocity = complexity-adjusted throughput (CAT)** per engineer per week, AI vs human `[A:109]` · **4 Quality = code turnover rate** (% of code rewritten/deleted within **30/90 days**, AI vs human) `[A:110]` · **5 Cost & ROI** `[A:111]`: **Net ROI = (Productive value of time saved − Rework cost from code turnover) ÷ Total AI tool cost**; total tool cost = seats + tokens/usage + implementation overhead; time-saved value = hours saved × loaded cost. Plus a qualitative layer: **developer-experience surveys** → *AI Value Realization Score* (telemetry + survey) ⚠.
- **WAY framework (⚠ orig. 222-223):** **W**ork-first (measure what happened, not what people say), **A**gnostic (across every tool), **Y**ours (survives an audit); layers AI adoption → AI impact → AI ROI.
- **AWS five categories `[A:112]`:** **deployment velocity** (time to market, sprint velocity, commit frequency, PR resolution time), **code quality**, **operational efficiency**, **team productivity & satisfaction**, **business impact**. ⚠ *Rollout (orig. 235):* **baseline & A/B** vs control teams → **granular cost tracking** (per-outcome) → **guardrail validation** (CI/CD security/quality checks so velocity doesn't spike debt/failure).
- **DX Core 4 `[A:113]`:** unifies **DORA + SPACE + DevEx** into four dimensions — **speed, effectiveness, quality, business impact**. **DX AI measurement framework `[A:114]`:** three dimensions — **utilisation, impact, cost**.

## J3. J-curve and ROI `[A:115-116]`
- **J-curve `[A:115]`:** AI adoption brings a **temporary productivity dip and instability** then recovery beyond baseline. Three trough forces: **learning curve**, **verification tax** (reviewing AI volume takes longer than provisioned), **pipeline adaptation**.
- **ROI `[A:116]`:** **ROI (%) = (Value − Investment) ÷ Investment.** Investment = **direct hard costs** (licences, AI infra/compute, training/enablement) **+ J-curve cost = staff size × fully loaded salary × productivity drop × dip duration**; value = new revenue + cost savings.

---

# K. DEPLOYMENT AND GOVERNANCE

## K1. Deployment vs release `[A:117]` (+ ⚠ orig. 243-244)
- **Deployment** = place software in a specific environment and make it operational (provision infra, configure, runtime). **Release** = make it available to end users (testing, QA, bug-fixing, GA).
- ⚠ **AI-assisted release management:** **GitOps** (version control as single source of truth for code and declarative infrastructure), automated semantic versioning, **SLSA provenance**, drift detection; **AI-assisted deployment verification** (compare live telemetry with historical baselines and **auto-rollback** on anomalies); **predictive CI/CD** (test-impact selection, predictive build-failure analysis). *Automation vs AI:* automation = known action (approval arrives → trigger); AI = **interpretation** (summarise, spot patterns, estimate risk).

## K2. AI governance `[A:118-123]` (+ ⚠ orig. 246-248, 253, 255)
- **What:** framework of **policies and oversight** directing how AI is built, approved, used and monitored — connects risk management, compliance, security and operations; "the operating system for responsible decisions, not a policy binder". ⚠ *Governance gap:* traditional governance assumes **human actors at every decision** — audit trails end at the human boundary, approval workflows assume human logic, security boundaries target human threat models.
- **Core components `[A:118]`:** **data governance** (lineage, ownership, access), **model governance** (purpose, training sources, thresholds, update cycles, drift), **policies & ethical guidelines**, **operational governance** (intake, approval checkpoints, review cycles), **security & privacy controls** (monitoring, vulnerability testing, privacy impact assessments).
- **Risk classification `[A:119]`:** **prohibited** (unacceptable) · **high-risk** (safety, finance, employment, essential services → formal risk assessment, documentation, human oversight, monitoring) · **medium-risk** (limited direct impact; documented controls, periodic review) · **low-risk** (basic visibility, usage policy; prevent **shadow AI**). Proportional controls.
- **EU AI Act `[A:120]`:** first comprehensive legal framework; **risk-based classification**, documentation & transparency, human oversight & risk management, **conformity assessments** and enforcement.
- **NIST AI RMF `[A:121]`:** voluntary; four functions — **Govern** (structures, accountability), **Map** (identify systems, context, impacts), **Measure** (evaluate bias, performance limits, security), **Manage** (prioritise, mitigate, monitor). ⚠ *Seven trustworthy-AI characteristics:* **valid & reliable** (foundation), **safe**, **secure & resilient**, **accountable & transparent** (cross-cutting), **explainable & interpretable**, **privacy-enhanced**, **fair with harmful bias managed**.
- **ISO/IEC `[A:122]`:** **23053** (framework for ML-based AI systems; concepts/terminology), **23894** (AI risk management guidance), **42001** (AI **management system** standard — establish, implement, maintain, improve; auditable/certifiable).
- **How they fit `[A:123]`:** complementary layers — **NIST = risk thinking (Govern-Map-Measure-Manage; flexible practices)**, **ISO 42001 = management system (policies, objectives, processes, continual improvement; auditable)**, **EU AI Act = legal obligations (scope, classification, actor duties, conformity; enforceable in EU)**. *Use NIST to structure risk decisions, ISO 42001 to institutionalise the operating model, and the EU Act for legal compliance.*

## K3. Governance inside the SDLC `[A:124-129]` (+ ⚠ orig. 257-258, 260, 266-268)
- **AI coding-assistant governance `[A:124]`:** visibility and control — track AI usage, define policies, enforce standards across teams/repos; **monitoring & logging link AI code to developers, inputs and decisions**; guardrails (red-teaming, model whitelisting…).
- ⚠ **SDLC-specific failure modes (orig. 257):** **insecure code & hidden defaults, hallucinated dependencies, data & secret leakage, excessive agency & untrusted tools (MCP), governance drift & rubber-stamp fatigue** → embed guardrails in **shared infrastructure**, not only written policy. *Standards (orig. 258):* NIST SSDF (SP 800-218/218A), NIST AI RMF + GenAI profile (AI 600-1), OWASP Top 10 for LLM & agentic apps.
- ⚠ **Use-case controls (orig. 260):** *coding assistant* — risks IP leakage, insecure code, hallucinated APIs → approved tool/model, repo boundaries, secure-coding policy, human review, SAST/SCA, provenance (NIST MAP/MEASURE, ISO 42001); *AI-generated tests* — false confidence, weak coverage → coverage thresholds, mutation testing, privacy-safe data, independent test approval; *autonomous agent* — excess privilege, prompt injection, cascading changes → scoped credentials, sandbox, allow-lists, transaction limits, dual approval, immutable logs, **kill switch**.
- **Architecture decision matrix `[A:125]`:** **reversible actions** (local diffs, dev docs) → delegate with standard PR review; **irreversible actions** (production deploy, schema migration, external library adoption) → **agent must never hold write capability; a human stays inside the approval gate**.
- **Decision classes & gates `[A:126-127]`:**
| Decision class | Agent role | Human role / gate | Reversibility |
|---|---|---|---|
| Low-risk module code change | author diff, run tests, open PR | review & merge — **one reviewer** | reversible (revert) |
| Regulated-path code (auth, payments, PII) | draft diff, cite constraint | **two approvers**, security-aware reviewer | reversible at code level; downstream data writes may not be |
| Architecture decision affecting ≥2 teams | draft **ADR** | **architecture review forum**; named owner signs | reversible only by another ADR |
| Production deploy | build artefact, run pre-deploy checks | **deterministic CI gate + deployment approval**; monitor canary | rollback within window |
| External dependency adoption | identify candidate, lockfile diff, supply-chain summary | **security review**, license check; library owner signs | only by removal + refactor |
| Cost-bearing agent run | declare model tier, per-run cost, stop condition | approve install & run; budget | spend not recoverable |
| Production incident response | surface diagnostics, **draft** remediation | **on-call engineer decides & executes; agent never executes** | human-only |
- **HITL vs HOTL `[A:128]`:** **HITL** (blocking approval) for high-consequence, irreversible actions (prod DB migration, perimeter change, external financial transaction); **HOTL** (monitor & intervene via dashboards and circuit breakers) for bounded reversible tasks (test generation, dev-branch dependency bumps). **Prevent rubber-stamp fatigue:** mandatory **escalation triggers** (low confidence, unfamiliar legacy dependencies, architectural-boundary violations) and **cryptographic audit trails** linking agent actions to human approvers.
- **Ownership & liability `[A:129]`:** **code provenance** = traceability of code origins. AI-code IP challenges: **opacity of training data, risk of code overlap, licence conflicts**. ⚠ *Why it matters:* IP/licensing, supply-chain security, AI governance/audit, quality & maintainability; *how to track:* signed commits, **DCO Signed-off-by**, **SLSA** build records, IDE/agent metadata (typed vs pasted vs generated).
- ⚠ **Feedback loop (orig. 267):** route production incidents, performance bottlenecks, **error budgets (SLOs)** and security telemetry back to backlog; automated **RCA and self-healing**; standardised **OpenTelemetry GenAI semantic conventions (`gen_ai.*`)**. ⚠ **Lifecycle health (orig. 268):** measure **DORA** (speed & stability), **SPACE** (developer experience & flow), quality, and business dimensions.

---

# L. LIKELY CONCEPT QUESTIONS (inferred — no PYQ archive) → ANSWER FROM

| Likely question | Answer from |
|---|---|
| Compare SDLC models and how AI changes each; AI impact on each phase | A2, A4 |
| Five shifts in software engineering; new competencies; economics | B1-B3 |
| Risks of AI in the SDLC | B4 |
| LLM, transformer, reasoning model, System 1/2; tokens, context window, tokenomics | C |
| Agent types; harness engineering (layers, guides/sensors, computational vs inferential) | D1-D2 |
| Context vs prompt vs harness engineering; RAG; MCP | D3-D7 |
| Spec-driven development; Spec Kit/Kiro; spec-first/anchored/as-source; ADRs | E, F4 |
| AI-driven requirements engineering; SDD risks & mitigations | F1-F3 |
| Deep modules; architecture is the prompt; cost-of-change mapping | G |
| Five collaboration patterns; guardrails; Lattice; library replacement | H |
| Non-determinism; reviewing AI code; AI in testing; churn/complexity; circular validation; testing AI/agents | I |
| DORA & AI; Developer AI Impact Framework; J-curve; ROI | J |
| AI governance; EU AI Act/NIST/ISO; decision gates; HITL vs HOTL; provenance | K |
