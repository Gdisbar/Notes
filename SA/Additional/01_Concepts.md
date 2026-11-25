# PART 1 — CONCEPTS (lean version)
*What it is · Why it exists · How it works · Where it fits (and what it costs) · A minimal example*

**You carry every slide, so this file is not a copy of the slides.** It gives the *idea* behind each topic, how topics connect, the trade-offs, and the words to use in an answer. Use the page references `[M:…]` to look up the *lists* (tactic names, checklist questions, step-by-step procedures, definitions' exact wording). Page numbers are for your merged PDF (see `00_READ_FIRST.md`).

**Contents:** A Foundations · B Quality-attribute framework · C The seven attributes (+ others) · D ASR & design strategy · E Agility · F Documentation & views · G Layered techniques · H Evaluation, conformance, reconstruction · I Patterns · J Cloud · K Mobile · L Big data & NoSQL · M Security technologies · N ML systems · O Management & governance · P Economics (CBAM) · Q Likely questions

---

# A. FOUNDATIONS

## A1. What is software architecture?
- **Idea:** the set of **structures** needed to reason about a system — elements, relations, and properties of both. The course prefers "structures" over "early/major decisions" because structures are identifiable and are what you analyse. `[M:12-13]`
- **Why it matters:** structures let you predict qualities (performance, availability, modifiability…) *before* building. A structure is "architectural" only if it supports such reasoning. `[M:14]`
- **Judgement:** there is **no inherently good or bad architecture** — only *fit for purpose*. Say this in any evaluation answer. `[M:20-21]`
- ✍ *Terms:* structure, element, relation, reasoning, fit-for-purpose, stakeholder concern.

## A2. Three kinds of structure; view vs structure
- **Module** (what the code units are, who may use whom) · **Component-and-connector** (what runs, how it communicates) · **Allocation** (how software maps to hardware, files, teams). Each answers a different stakeholder question and supports a different QA. `[M:13, M:16-18]`
- The "relations and useful-for" tables are the lookup you need: `[M:92-93]`.
- **View = a documented representation of a structure.** Architects *design* structures and *document* views. `[M:15]` **Modules ≠ components** — don't expect one-to-one mapping. `[M:19, M:21]`
- ✍ *Sentence:* "Layers (module) → portability; deployment (allocation) → availability; concurrency (C&C) → contention."

## A3. Patterns vs tactics
**Pattern = molecule, tactic = atom.** A tactic influences one QA response; a pattern bundles several and brings side-effects that need more tactics. Catalogues exist because building every design from first principles is slow. `[M:19-20, M:25, M:203-206]`

## A4. What makes an architecture good
Themes (details on the slides): **conceptual integrity** (one architect/small team), **QAs prioritised and written as scenarios**, **information hiding / separation of concerns**, **no dependence on a specific tool version**, **evaluated** with a method, supports **incremental delivery** (skeletal system). Link each rule to a QA when you quote it. `[M:20-21]`

## A5. Why architecture matters (skim)
One message: **the architecture enables or inhibits the QAs**; everything else (managing change, early prediction, communication, early hard-to-change decisions, constraints on implementation, shaping the organisation, enabling prototyping, estimates, reuse) follows from that. [M:1-6]

## A6. Contexts and influences (skim)
Four contexts — technical, project life-cycle, business, professional — and the **architecture influence cycle** (the system changes the environment which influences the next architecture). Business context is the one to quote: *every QA should trace to a business goal.* [M:6-11]

## A7. Requirements: functional, QA, constraint
Functionality says *what*; a QA *qualifies* it (how fast, how often available, how easy to learn); a constraint is a decision with zero freedom. **Functionality and QAs are orthogonal — functionality does not determine architecture, QAs do.** Old QA talk fails because definitions are untestable and attributes overlap; **QA scenarios** fix this. `[M:23-24]`

---

# B. QUALITY-ATTRIBUTE FRAMEWORK

## B1. Quality-attribute scenario (QAS)
Six parts: **source · stimulus · environment · artifact · response · response measure**. The measure makes it testable. *General* scenario = the template for an attribute; *concrete* = for your system. `[M:24-25]`
✍ *Template:* "[Source] [stimulus] on [artifact] in [environment]; the system [response] within [measure]."

## B2. Tactics
A design decision affecting a single QA response. Choose by **trade-offs and cost**; use them to adapt a pattern or to design where no pattern fits. `[M:25, M:28]`

## B3. The seven categories of design decisions — and the checklists
Allocation of responsibilities · coordination model · data model · management of resources · mapping among elements · binding time · choice of technology. `[M:26-28]`
**How to use the checklists (your 49 pages):** for *any* scenario, walk the seven categories and ask the attribute-specific question for each. It shows completeness and finds tactics you would otherwise miss. Checklists: availability `[M:33-34]`, performance `[M:38-39]`, usability `[M:43-44]`, security `[M:51-53]`, modifiability `[M:56-58]`, interoperability `[M:61-63]`, testability `[M:67-68]`.
✍ *Pattern for an answer:* "Allocation: who detects/logs the fault? Coordination: guaranteed delivery? Data model: what can be rolled back? Mapping: can elements be remapped? Resources: enough left to recover? Binding: does late binding bypass the strategy? Technology: does the tech itself fail?"

---

# C. THE SEVEN QUALITY ATTRIBUTES

> For each: **essence → how to reason → tactic logic → trade-offs → example.** The tactic *names* are on the slides; learn the *logic*.

## C1. Availability `[M:29-35]`
- **Essence:** be ready when needed; **minimise outage time by handling faults.** A *fault* is a cause; a *failure* is what users observe. Tactics keep faults from becoming failures or limit their effect.
- **Fault types to name:** omission, crash, incorrect timing, incorrect response. Response measure: availability %, time to detect, time to repair, degraded-mode period.
- **Tactic logic:** *Detect* (monitor, heartbeat, ping/echo, voting…) → *Recover* (**prepare and repair**: redundancy, rollback, retry, degradation, reconfiguration; **reintroduce**: shadow, state resync, escalating restart) → *Prevent* (removal from service, transactions, prediction).
- **Key reasoning:** the **redundancy spectrum** — hot (active) = fastest failover, highest cost; warm (passive) = state catch-up delay; cold = cheapest, slowest, data-loss window. Choose by tolerated downtime.
- 🆕 *Numbers:* 99.9% ≈ 8.8 h/yr; 99.99% ≈ 53 min/yr; 99.999% ≈ 5 min/yr. Availability ≈ MTBF ÷ (MTBF + MTTR): raise it by failing less *or* repairing faster.
- **Trade-offs:** redundancy ↑ availability but ↑ cost/complexity; ping/echo ↑ detection but ↓ performance and is DoS-able.

## C2. Performance `[M:35-40]`
- **Essence:** about *time* — responding to events within timing constraints. **Latency = processing time + blocked time** (contention, waiting on resources/dependencies). Measures: latency, throughput, deadline, jitter, miss rate.
- **Tactic logic:** only two levers — **control resource demand** (limit/prioritise/bound/sample/reduce overhead) or **manage resources** (more resources, more concurrency, copies of computation/data, bounded queues, scheduling).
- **Trade-offs:** removing intermediaries speeds things up but hurts modifiability; concurrency hurts testability (non-determinism); caching risks stale data; sampling loses fidelity.
- ✍ *Example:* Aadhaar registration drop-downs slow → cache static lists (copies of data) + AJAX incremental load (reduce demand). `[M:113]`

## C3. Usability `[M:40-44]`
- **Essence:** ease of accomplishing a task and support for the user; limit the impact of errors. Measures: task time, error count, satisfaction, data lost on error.
- **Tactic logic:** **support user initiative** (cancel, pause/resume, undo, aggregate) vs **support system initiative** (maintain task / user / system models so the system can anticipate and give feedback).
- **Trade-offs:** undo needs stored state (memory/consistency); user models add complexity and privacy concerns.

## C4. Security `[M:48-53]`
- **Essence:** protect data while still serving authorised users. **CIA** — confidentiality, integrity, availability — supported by authentication, authorisation, non-repudiation. An *attack* is an action meant to harm.
- **Mental model:** physical security → **detect, resist, react, recover.** Resist (authenticate, authorise, limit exposure, encrypt, separate…) is cheapest *before* an attack; recovery includes **audit** to trace the attacker.
- **Reasoning:** *no tactic is foolproof* → **defence in depth**; minimise **attack surface**.
- **Trade-offs:** every layer costs performance and usability (encryption overhead, extra logins).

## C5. Modifiability `[M:54-58]`
- **Essence:** cost and risk of change. Three questions: **what can change, how likely, when and by whom?**
- **Tactic logic:** reduce **module size**; increase **cohesion**; reduce **coupling** (encapsulate, intermediary, restrict dependencies, refactor, abstract common services); **defer binding**.
- **Reasoning:** cost of change grows with module size and coupling; later binding makes *each* change cheaper but the mechanism costs more to build. **Defer binding only for changes that are likely and frequent.**
- **Trade-offs:** intermediaries/late binding ↓ performance, ↑ complexity.
- ✍ *Example:* hospital fee change in a day without code change → configuration/rules engine. `[M:83]`

## C6. Interoperability `[M:59-63]`
- **Essence:** exchanging *meaningful* information between systems (shades, not yes/no). 🆕 Syntactic = can parse; semantic = agree on meaning.
- **Tactic logic:** systems must **know about each other** (locate/discover) and **exchange information correctly** (manage interfaces: orchestrate, tailor interface).
- **Costs:** security exposure, load from external callers, versioning; binding to unknown systems needs a policy to reject and log.

## C7. Testability `[M:64-68]`
- **Essence:** probability that a fault will show up on the next test. Needs **controllability** (set state/inputs) and **observability** (see outputs/state).
- **Tactic logic:** *control and observe state* (special interfaces, record/playback, localised state, abstract data sources, sandbox, assertions) and *limit complexity* (limit structural complexity, limit non-determinism).
- **Link to architecture:** design for test (switchable data sources, simulators for payment gateways/sensors) and use the utility tree to prioritise tests. `[M:129-130]`

## C8. Other attributes `[M:45-48]`
Scalability (**horizontal = add servers; vertical = bigger machine**), portability, variability, deployability, mobility, monitorability, safety, conceptual integrity, marketability, quality in use; standard lists (ISO 25010). For an unfamiliar "X-ability": **model it, assemble tactics, build design checklists.** `[M:47]`

---

# D. ASR & DESIGN STRATEGY

## D1. Architecturally significant requirement (ASR)
A requirement with a **profound, measurable effect on the architecture** — remove it and the design would look very different. Indicators: high business value or technical risk, important to an influential stakeholder, unique, differs from what the evolving design provides, previously caused trouble. You cannot design for everything — **find the few that drive the structure.** `[M:78-79]`

## D2. Finding ASRs
Requirements documents/stories · leads from the seven decision categories · stakeholder workshops (**QAW**) · **business goals** (write as "X to Y by when") · the **utility tree**. `[M:79-81]`

## D3. Utility tree — how to reason
Root utility → quality attributes → refinements → **scenarios tagged (importance, difficulty) = (H/M/L, H/M/L)**. **(H,H) first** — important *and* hard. It is reused in ATAM (step 5) and to prioritise tests. `[M:82-83, M:122]`
✍ *Skeleton:* identify QAs from business goals → refine → write measurable scenarios → rate → design for top-right first.

## D4. Attribute-Driven Design (ADD) — standard method, slide is short
Pick an element → identify its ASRs → choose design concepts (patterns/tactics) → instantiate elements and interfaces → sketch views and record decisions → analyse → iterate. 🆕 (See the ADD heading slide: [M:69-70].)

---

# E. ARCHITECTURE & AGILITY `[M:69-76]`
- **Not a conflict:** the question is *how much* architecture up-front and *how much* documentation.
- **Sweet spot reasoning:** up-front architecture/risk work costs schedule; **rework** from later changes costs schedule; they trade off. Boehm & Turner: ~10 KSLOC → little up-front; ~100 KSLOC → ~20%; ~1000 KSLOC → ~40%. `[M:73, M:84]`
- **Advice:** large, complex, stable or distributed → much up-front; large with unstable requirements → a candidate architecture first, elaborate later; small/uncertain → agree only the major patterns, use **spikes**; **refactor when technical debt gets high**. `[M:74-76]`
- ✍ *Terms:* BDUF, spike, technical debt, incremental commitment, skeletal system.

---

# F. DOCUMENTATION & VIEWS
## F1. Principle
Document only what communicates the design and mitigates risk. `[M:86-93, M:103]`
## F2. Kruchten 4+1
One diagram cannot serve every stakeholder, so use several views: **Logical** (end user; functions), **Process** (integrators; concurrency, scalability), **Development** (programmers/managers; module organisation), **Physical** (system engineers; hardware mapping), plus **Scenarios** to tie them together. **Choose views by dominant concern** (complex UI → logical; high traffic → process; distributed team → development; cloud/security → physical). `[M:95-103]`

---

# G. LAYERED ARCHITECTURE — WHAT EACH LAYER'S TECHNIQUES ARE FOR `[M:105-112]`
- **Presentation:** speed and reach — caching (client-side for per-user data; server-side for shared), AJAX, responsive design.
- **Business:** modifiability and hiding complexity — façade, session management, workflow/rules engine, aspect-oriented concerns, logging vs instrumentation.
- **Data:** efficiency and safety — pooling, copies of data, transactions, ORM, stored procedures, **parameterised SQL against injection**.
- **Service:** exposing capabilities safely — must handle **duplicate requests, out-of-sequence messages and communication failure (retry or queue)**.
- **Practice (answers on slides):** four exercises `[M:112-114]` — see `02_Scenario_Based.md` §10 for how to attempt them.

---

# H. EVALUATION, CONFORMANCE, TESTING, RECONSTRUCTION
## H1. Evaluate early
"Fail early": find risks and trade-offs before building. Three forms — **by the designer** (the "test" in generate-and-test), **peer review**, **by outsiders**; choose by importance of the decision, artifacts available, who sees the results. `[M:117-118]`

## H2. ATAM — the reasoning
- Evaluates **the most important QA scenarios only**, not the whole architecture; needs no built system. Produces **risks and non-risks, sensitivity points, trade-off points, risk themes**, mapped to business goals. `[M:119-120]`
- 🆕 **Sensitivity point** = a property critical to achieving a QA response; **trade-off point** = a sensitivity point for more than one QA. *Example:* heartbeat interval affects time to detect a failure → sensitivity point; too slow = risk. `[M:122]`
- **Flow:** present ATAM → business drivers → architecture → approaches → **utility tree** → analyse approaches → brainstorm scenarios → analyse again → present results (nine steps; phases 0-3). `[M:120-123]`
- **Cost reasoning:** full ATAM ≈ 20-30 person-days for large risky projects; **lightweight** (½-1 day, internal) otherwise. `[M:124]`

## H3. Conformance — keep code aligned
Drift examples: skipping layers, bypassing the data layer, not using pub-sub or common logging. Techniques: architecturally evident coding style, frameworks, code templates, updated documentation (plus reviews and education). `[M:125-128]`

## H4. Architecture & testing `[M:129-130]`
Utility tree → test priorities; module dependencies → integration plan; design for testability.

## H5. Reconstruction `[M:131-135]`
When no documentation exists or you must migrate: **extract raw views → build a database → fuse views (static + trace + expert grouping) → analyse conformance.**

---

# I. PATTERNS — THE IDEA, WHEN TO USE, WHAT TO WATCH
> A pattern = **context + problem + solution**. Structure, steps and variants are on the slides; here is the *selection logic*. `[M:137-141]`

| Pattern | Idea in one line | Use when | Watch out for | Pages |
|---|---|---|---|---|
| **Layers** | unidirectional "allowed-to-use" groupings of cohesive services | portability, independent evolution, reuse | efficiency, cascading change, granularity | `[M:141-146]` |
| **Pipes & Filters** | data flows through independent incremental transformations | stream transformation, reuse, recombination | shared state, error handling, transform overhead | `[M:147-150]` |
| **Blackboard** | knowledge sources cooperate through a shared store under a control component | no deterministic algorithm (vision, speech) | control complexity, testing | `[M:151-153]` |
| **Broker** | a broker locates servers and forwards requests; proxies hide distribution | distributed components needing location transparency | broker = bottleneck/SPOF, efficiency | `[M:155-162]` |
| **MVC** | separate model, view(s), controller(s); change propagation | UIs that change often, many views of one model | complexity, excess updates, view-controller coupling | `[M:165-171]` |
| **PAC** | hierarchy of cooperating agents (presentation, abstraction, control) | distributed/concurrent interactive subsystems | complex control component | `[M:171-178]` |
| **SOA** | services behind published contracts; ESB/registry | interoperability across platforms | complexity, no control of service evolution | `[M:184-185]` |
| **Microkernel** | minimal core + plug-in servers; **mechanism vs policy** | adapt to changing platforms / product lines | performance, design complexity | `[M:187-191]` |
| **Reflection** | meta level changes base-level behaviour at run time (MOP) | run-time adaptation | damage from meta changes, overhead | `[M:192-195]` |
| **Client-Server / P2P / Pub-Sub / Shared-Data / Map-Reduce / Multi-Tier** | SEI C&C and allocation patterns | see `03_Design_Decisions.md` §2 | see slides | `[M:196-203]` |

**Reasoning to quote:** a pattern is *under-specified* for a real system — it needs tactics to patch its weaknesses (e.g., Broker + replicated brokers for availability). **Every tactic creates a new concern** (ping/echo → overhead, DoS risk), and the chain stops when side-effects are small enough. `[M:203-206]`

---

# J. CLOUD ARCHITECTURE
- **NIST view:** five essential characteristics; **SaaS / PaaS / IaaS** by who the consumer is; private, public, community, hybrid. `[M:207-208]`
- **Why it pays:** virtualisation raises utilisation; **multi-tenancy** gives one version and lower support cost; but sharing brings isolation risks. `[M:209]`
- **Design-for-failure:** with thousands of servers **failure is normal** — the application must detect and recover; auto-scaling may be too slow, so applications should be self-aware. `[M:216-217]`
- **Why NoSQL:** web-scale sequential data, relational overhead, CAP; **trade-off: no transactions, schemas, joins — eventual consistency.** `[M:214-215]`
- **HDFS write path and its three failure cases** (client, NameNode, DataNode) are a favourite scenario: `[M:212]`.
- **CAP:** only two of consistency, availability, partition tolerance; partitions happen, so **choose C or A *per service*** — e.g. cart = available, order submission = consistent. `[M:223-224, M:236]`
- **Multi-tenancy degree and per-tier choices:** `[M:237-239]`. **Cloud-native:** `[M:239]`.

# K. MOBILE ARCHITECTURE `[M:226-234, M:240]`
- **Constraints drive design:** simple UI, compact code, few layers, **store locally and sync later**.
- **Native vs mobile-web/PWA vs hybrid:** performance/UX vs reach/cost vs middle path — see decision table in `03_Design_Decisions.md` §6.
- **Android components** map to responsibilities: Activity (UI), Service (background work), Content provider (data), Broadcast receiver (event reaction). Use these to answer component-identification exercises `[M:231-232]`.

# L. BIG DATA & NoSQL `[M:242-262]`
- **Hadoop:** HDFS (blocks, replication, NameNode metadata, heartbeats) + MapReduce; good for batch/log analysis, not low-latency. `[M:245-248]`
- **Spark / streaming:** disk-based MapReduce is too slow for real-time (e.g., fraud) → in-memory or stream processing (Storm + Kafka). `[M:248-251, M:258-262]`
- **NoSQL types and when:** key-value, document, column, graph, in-memory — pick by access pattern; all trade away ACID/joins. `[M:252-257]`

# M. SECURITY TECHNOLOGIES `[M:263-269]`
Know *what problem each solves*: **TLS** (privacy and integrity of the channel), **OpenID** (authentication), **OAuth** (authorisation of an app to your data), **LDAP** (fast directory reads), **IAM** (roles and compliance), **firewall + DMZ** (perimeter), logging, AJAX.

# N. ML SYSTEMS `[M:271-274]`
Supervised vs unsupervised vs reinforcement; steps to build a model (problem → features → model → train-test-validate → improve); an event-driven architecture example (image stored → trigger → ML service → store result → notify).

# O. MANAGEMENT & GOVERNANCE `[M:276-285]`
Planning: **top-down** (to convince management, inherently inaccurate) vs **bottom-up** (architecture team, more accurate) → **reconcile**. The PM–architect relationship is the most important on the team. PMBOK = process groups × knowledge areas. Trade-offs: architect trades QAs; PM trades features-schedule-quality; control **scope creep**. Governance = controls, compliance, processes, accountability.

# P. ECONOMICS — CBAM `[M:287-297]`
- **Idea:** choose strategies by **value for cost**, not by benefit or cost alone.
- **Core calculation:** per scenario benefit `b = U_expected − U_current`; total `B = Σ b × weight`; **`VFC = B ÷ C`** — rank by VFC. `[M:288]`
- **Nine steps** (collate scenarios → refine → prioritise by votes → assign utility → map strategies → interpolate → total benefit → cost and VFC → confirm with intuition). `[M:290-292]` Worked tables: `[M:295-297]`.

---

# Q. LIKELY CONCEPT QUESTIONS (inferred — no PYQ archive)
| Likely question | Where |
|---|---|
| Define architecture; structure vs view; module vs C&C vs allocation | A1-A2 |
| QA scenario; write one for X | B1 + the attribute in C |
| Tactics for availability/performance/security/modifiability/testability/usability/interoperability | C1-C7 |
| Seven decision categories / checklists | B3 |
| ASR; utility tree; finding ASRs | D |
| Agile vs architecture | E |
| 4+1 views | F2 |
| Layered-architecture techniques | G |
| ATAM; sensitivity vs trade-off; lightweight evaluation | H2 |
| Drift; reconstruction | H3, H5 |
| Explain pattern X with benefits and liabilities; patterns vs tactics | I |
| Cloud/NIST; multi-tenancy; CAP; HDFS | J |
| Mobile; Hadoop/Spark; NoSQL; TLS/OAuth/OpenID; CBAM; PM and governance | K-P |
