# Software Architectures (SE ZG651 / SS ZG653) — EC3 Exam Notes

**BITS Pilani WILP · Comprehensive Examination (EC-3) · 40% weight · 2.5 hours · All topics (Contact Sessions 1–16)**

---

## How to use these notes (read this first)

You will sit EC-3 **as a closed-book paper** even though it is officially open book. That changes how you should study:

1. **You cannot look anything up.** Every definition, list of pattern elements, tactic name, CBAM step and formula in this file must be retrievable from memory. The notes are therefore written in *retrievable* form: numbered lists, fixed templates, comparison tables and mnemonics — not flowing prose.
2. **EC-3 questions are scenario-based and ask for analysis, design decisions, comparisons and justification — not recall.** The exam pattern (from the 2025-26 paper) is **four questions of 10 marks each**, each opening with "Suppose you are tasked with..." Every sub-part uses command words like *propose, analyse, explain, discuss trade-offs, justify*.
3. **Weight of study.** The paper is comprehensive, but the biggest scoring surface is the material *not* covered in the closed-book mid-sem (EC-2 covered Sessions 1–8). These notes put the deepest coverage on Sessions 9–16: **architectural patterns, security patterns, cloud & integration, mobile, big data/MapReduce/NoSQL, and CBAM economics**. Sessions 1–8 (foundations, quality attributes, tactics, ASR/utility tree, views, ATAM, layered guidelines) appear here as compact refreshers because scenarios constantly borrow from them.
4. **A model answer pattern is baked into every section** — see the Answer-Writing Guide in the QA file. Roughly: *state assumptions → name the attribute/driver → select pattern/tactic → map scenario elements onto it → state gain AND price → resolve side effects with a tactic → conclude with a decision and the condition that would change it.*
5. **Budget:** 150 minutes / 40 marks ≈ **3.75 minutes per mark**, and 60–80 words per mark. A 10-mark question ≈ 600–800 words including tables and a labelled diagram. Practice with the mock paper at the end of the QA file, timed.

**What the last real EC-3 paper looked like (2025-26, all scenario-based):**

| Q | Topic | What was actually asked |
|---|-------|-------------------------|
| Q1 (10m) | Pipes & Filters + MVC | (a) Recommend a pipes-and-filters architecture for a personalised recommendation workflow; configure per data type; scale it; limits in real-time use + remedies. (b) Apply MVC to an e-commerce platform; benefit for payment-gateway and order-workflow changes; price-change example that leaves UI untouched. |
| Q2 (10m) | Security patterns + Map-Reduce | (a) Privilege separation in C&C design + a real system (OpenSSH) + challenges/strategies. (b) Secure Logger + Clear Sensitive Information: objectives, applicability, trade-offs, combined use in a payment system. (c) Map-Reduce on social-media user logs; batch vs real-time trade-off. |
| Q3 (10m) | Mobile | (a) Cross-platform vs native for a GPS+camera app across six named attributes + justified decision with frameworks. (b) Responsive design case study: loading time, navigation, interaction + the role of responsive design. |
| Q4 (10m) | Economics (CBAM) | (a) Scalability-first vs speed-first strategy under opportunity cost; stakeholder factors. (b) Evaluating security/scalability/cost trade-offs; role of utility-response curves with numeric example. |

Expect the same *shape* of question, different scenarios. The 2026-27 paper draws on **all topics**, so treat cloud, microservices, NoSQL, ATAM and pattern comparisons as equally live.

---

# PART A — Core Refresher (Sessions 1–8, compressed)

> This part is deliberately compact. It exists so that any scenario answer can lean on definitions, scenarios, tactics, utility trees, views, ATAM and layered techniques from memory.

## A1. What software architecture is — the one definition you must quote

**Definition (Bass/Clements/Kazman):** *The software architecture of a system is the set of structures needed to reason about the system, which comprise software elements, relations among them, and properties of both.* A structure is a set of elements held together by a relation.

Unpacking, word by word (each phrase earns marks):

| Phrase | What it commits the architect to |
|---|---|
| "the set of structures" | Architecture is **plural** — many structures at once; none alone is "the" architecture |
| "needed to reason about the system" | A structure is architectural only if it supports reasoning (response time, cost of change, failure impact) |
| "software elements" | Modules, components, services, data stores, layers |
| "relations among them" | Calls, uses, data flow, containment, allocation — a parts list is not an architecture |
| "properties of both" | Responsibility, interface, protocol, timing, synchrony — what makes it analysable |

**Three consequences (frequent short questions):**

1. **Architecture is an abstraction** — it keeps detail that has ramifications outside a single element and omits element-private detail (internal algorithms, helper functions).
2. **Every system has an architecture** (documented or not). The architecture belongs to the system; documentation is a *representation* of it. An undocumented architecture cannot be communicated, analysed or checked, and must be rediscovered at high cost.
3. **Behaviour counts when it influences other elements or the acceptability of the system.** A box-and-line drawing is *not* an architecture by itself: it names elements but states no relations or properties, so readers imagine behaviour from names, and that may be wrong.

**Why "the early, major design decisions" is NOT the definition:** many architectural decisions are made late (a cache added in year four); many early decisions are not architectural (a naming convention); and "major" is hard to judge. Structures, in contrast, are identifiable.

### The three families of structure (one of the most reusable lists)

| Family | Elements | Question answered | Useful structures | Reasoning supported |
|---|---|---|---|---|
| **Module** | Units of implementation (modules, classes, layers) | How is the system partitioned as code to be built/procured? | Decomposition (is-a-submodule-of), Uses (uses), Layer (allowed-to-use), Class (inherits-from), Data model (entity relations) | Modifiability, extension, subsetting, portability, reuse |
| **Component-and-Connector (C&C)** | Runtime components + connectors | What runs, and how do the pieces interact at runtime? | Service structure, Concurrency structure; relation = attachment | Performance, security, availability, parallelism |
| **Allocation** | Software + environment elements | Where does software live and who builds it? | Deployment (allocated-to), Implementation (mapped-to files), Work assignment (assigned-to teams) | Performance, availability, security, build & team organisation |

**Structure vs view:** a *structure* is the set of elements and relations as they exist in software/hardware; a *view* is a documented representation of one or more structures, written and read by stakeholders in a chosen notation. **Architects design structures and document views.** Mappings between structures are many-to-many: one Client module may appear as ten runtime components.

**Layer vs tier (classic confusion):** a *layer* is a module-structure unit — cohesive services through a managed interface, governed by allowed-to-use; a *tier* is an allocation-structure unit — a hardware/software subset to which components are allocated. A three-layer code base can deploy in one tier or several; the two decisions are independent.

**Pattern families by structure type:** Module → Layered. C&C → Shared-data, Client-server, Broker, Pipes-and-Filter, MVC, P2P, SOA, Publish-Subscribe. Allocation → Multi-tier, Map-Reduce.

## A2. Quality attribute scenarios — the six-part form

Every scenario answer in EC-3 should be writable in this exact six-part form. Memorise the six part-names: **source, stimulus, artifact, environment, response, response measure**.

| Part | Meaning | Example (availability) |
|---|---|---|
| Source | Where the stimulus comes from | External payment gateway |
| Stimulus | The event arriving | Gateway stops responding (crash/omission fault) |
| Artifact | Which part of the system is affected | Payment-processing component |
| Environment | Conditions at the time | 08:00 reservation peak, overload |
| Response | What the system does | Detect fault, notify operator, queue/redirect payments, degrade gracefully |
| Response measure | Numbers that make it testable | Fault detected < 5 s; service restored < 30 s; no confirmed booking lost |

**Why scenarios instead of adjectives:** "reliable", "fast", "secure" are untestable and ambiguous. A scenario with numeric response measures is designable, testable and evaluable (it is also the unit used by the utility tree, ATAM and CBAM).

### The attributes most often named in scenarios

| Attribute | One-line definition | Typical response measure |
|---|---|---|
| Availability | Probability the system is operational to deliver requested services | % uptime, downtime/year, MTTF/MTBF/MTTR |
| Performance | Responsiveness under events: latency, throughput, capacity | Response time s/ms, TPS, concurrent users |
| Security | Resistance to unauthorised use while still serving legitimate users | % attacks blocked, disclosure probability, time to detect |
| Modifiability | Ease/cost of making a specific change | # modules touched, person-days, # tests affected |
| Scalability | Ability to handle growth in load by adding resources | Peak users supported, degradation curve |
| Usability | Ease of task completion, error recovery, learnability | Task time, # user errors, recovery steps |
| Interoperability | Ability of systems to exchange information and use it | % requests interoperable, protocol conformance |
| Testability | Ease of demonstrating faults | % statements executed in a test, time to write a test |
| Reliability | Continuity of correct service | Probability of failure-free operation for t |

**Availability arithmetic (memorise — appears in calculations):**

- `A = MTTF / (MTTF + MTTR)` (MTTF = mean time to failure, MTTR = mean time to repair)
- Series: `A_system = A1 × A2 × A3` (downtime multiplies)
- Active redundant pair: `A = 1 − (1−A1)²`; Downtime/year = (1 − A) × 8760 hours; a 99.95% promise = 4.38 h/year; in a 30-day month = 0.05% × 720 h = **0.36 h ≈ 21.9 minutes**.

## A3. Tactics catalogues (compact — quote by category)

A **tactic** is a single design decision influencing one quality-attribute response. A **pattern** is a molecule of tactics (see B13). The categories below follow *Software Architecture in Practice* (T1).

**Availability** — three sub-categories:
- *Detect fault:* ping/echo, heartbeat, monitoring, timestamp, sanity checking, condition monitoring, voting.
- *Recover from fault:* active redundancy (hot spare), passive redundancy (warm spare), spare, exception handling, rollback, graceful degradation, retry, ignore faulty behaviour, shadow operation, reconciliation, removal from service.
- *Prevent fault:* transactions, preventive maintenance, increased resources.

**Performance** — two sub-categories:
- *Manage resource demand:* reduce computational overhead, manage event rate, control frequency of sampling, limit event response, prioritise events, reduce overhead elsewhere, bound execution times, bound queue sizes.
- *Manage resources:* increase computational resources, introduce concurrency, maintain multiple copies (of computations or data), scheduling policy (FIFO, earliest-deadline-first…).

**Security** — four sub-categories:
- *Detect attacks:* detect intrusion, detect service denial, verify message integrity.
- *Resist attacks:* identify actors, authenticate actors, authorise actors, limit access, limit exposure, separate entities, validate inputs, encrypt data.
- *React to attacks:* revoke access, lock computer, inform actors.
- *Recover:* maintain audit trail, restore (as availability recover tactics).

**Modifiability** — two sub-categories:
- *Localise change:* increase semantic coherence, anticipate change, generalise the module, restrict dependencies.
- *Prevent ripple effects:* encapsulate, use an intermediary, restrict dependencies, defer binding (configuration time, deployment time, run time), refactor, abstract common services.

**Usability** — support user initiative: undo, cancel, aggregate; support system initiative: reify user context, maintain task model, maintain working-session history, interface scaffolding; also retrieve forgotten password, scaffolding.

**Side-effect discipline (used in every "analyse" answer):** each tactic has pluses and minuses. Example: ping/echo improves fault detection but raises security questions (ping flood), performance overhead (extra traffic) and modifiability questions (adding it to the architecture). Resolve with further tactics (increase resources → scheduling policy → intermediary → restrict communication paths) and stop when the remaining side effect is small.

## A4. ASRs, QAW, utility tree, ADD (compressed)

**Architecturally Significant Requirement (ASR):** a requirement that has a profound effect on the architecture — most are quality-attribute requirements plus a few functional ones that shape structure.

**Five routes to ASRs:** (1) mining the requirements documents (look for: many constraints, high-change areas, externals and boundaries, prototypes, new technologies, service-level agreements, early architectural proofs); (2) the **Quality Attribute Workshop (QAW)** with stakeholders — brainstorm scenarios, prune, prioritise, refine (QAW steps: present QAW/roles, present business drivers/plan, brainstorm scenarios, consolidate, prioritise, refine); (3) business goals (PALM: **P**urpose, **A**nticipated benefits, **L**osses/risks, **M**etrics); (4) capturing in a **utility tree**; (5) from design decisions and risks already made.

**Utility tree (memorise the four levels):** root = *utility* → level 2 = *quality attributes* (performance, availability, security…) → level 3 = *attribute refinements* (latency, throughput; confidentiality, integrity…) → level 4 = *scenarios* (six-part, with numbers), each leaf rated **(importance, difficulty)** = (H/M/L, H/M/L). Analyse (H,H) and (H,M) first. Purpose: concretise requirements and rank them; the tree is the bridge between requirements and evaluation (ATAM) and economics (CBAM).

**Attribute-Driven Design (ADD), three steps:** (1) choose the module to design (start with all); (2) identify the ASRs for it; (3) generate design — choose tactics/patterns satisfying the ASRs, document and iterate. Agile context: do "just enough" architecture up front, iterate, use spikes for technical risk; the sweet spot sits between *no architecture* and *over-engineering* (cost of architecture vs cost of not having it — document minimal architecture via views stakeholders need).

## A5. Views & the 4+1 model (compressed)

**4+1 views (Kruchten):**

| View | Audience/concern | Elements |
|---|---|---|
| Logical | Functionality for end users | Classes/objects (class & object diagrams) |
| Process | Runtime performance, concurrency | Processes/threads |
| Development | Software organisation for programmers | Modules/packages |
| Physical | Deployment, topology, scalability | Nodes, allocation |
| Scenarios (+1) | Tie the other four together | Use-case traces verified across views |

The "+1" checks consistency: walk one important use case through all four views. Documentation rule: document each chosen view (primary presentation, element catalogue, context, variability, rationale), combine related views when useful, and include quality-attribute views (security view, communication view, reliability view) when those qualities matter.

## A6. ATAM — Architecture Tradeoff Analysis Method (compressed but exact)

**Purpose:** find how well an architecture satisfies quality goals, expose **risks** and **trade-offs** before it is too late to change them. It is scenario-driven and *not exhaustive* — it analyses only what the most important scenarios touch.

**Participants:** evaluation team (leader, evaluation leader, scenario scribe, proceedings scribe, questioner), project decision makers, stakeholders.

**Four phases:**

| Phase | What | Participants | Duration |
|---|---|---|---|
| 0 | Partnership & preparation | Team leadership + decision makers | weeks, informal |
| 1 | Evaluation steps 1–6 | Team + decision makers | 1–2 days, then 2–3 week hiatus |
| 2 | Evaluation steps 7–9 | Team + decision makers + stakeholders | 2 days |
| 3 | Follow-up: report | Team + client | ~1 week |

**Nine steps (memorise as: *Present–Present–Present, Identify–Tree–Analyse, Brain–Analyse–Report*):**
1. Present the ATAM
2. Present business drivers
3. Present the architecture
4. Identify architectural approaches (patterns/tactics)
5. Generate the quality attribute utility tree
6. Analyse architectural approaches (against high-ranked scenarios) → findings
7. Brainstorm and prioritise scenarios (with full stakeholders)
8. Analyse architectural approaches again (new scenarios from step 7)
9. Present results.

**The four finding types (this exact vocabulary earns marks):**
- **Risk:** a potentially problematic decision not yet proven.
- **Nonrisk:** a decision proven safe by analysis.
- **Sensitivity point:** a decision that markedly affects a particular quality attribute response (e.g., number of DB clients → throughput).
- **Tradeoff point:** a sensitivity point affecting more than one attribute, where improving one may degrade another (e.g., heartbeat frequency ↑ detection but ↓ performance).

**Outputs:** architecture presentation, business goals, prioritised scenarios in six-part form, risks & nonrisks, risk themes, decision-to-requirements mapping, sensitivity & tradeoff points.

**Lightweight evaluation:** when a full ATAM is unaffordable — smaller scenario set, fewer steps (present architecture → utility tree → analyse highest-priority leaves → report), or ATAM-style review in agile sprints.

## A7. Layered architecture guidelines (Sessions 7 — presentation/business/data/service)

**Why layers:** separates concerns, eases team work, isolates change, promotes portability and reuse. Cost: up-front complexity and a performance penalty (every request hops through layers).

| Layer | Responsibility | Key techniques |
|---|---|---|
| Presentation | User interaction, display | MVC/MVP in the UI, rich vs thin clients, avoid business logic in the UI, use UI patterns (action bars, large buttons, discoverable controls), responsive layout |
| Business | Domain logic, workflows | Facade pattern to hide subsystems, workflow engines for multi-step processes, business rules externalised/configurable, session state handling for scale-out (sticky sessions vs stateless + shared store) |
| Data | Persistence, queries | DAO/repository abstraction, connection pooling, caching (distributed cache), transaction management, partitioning/sharding, avoid SQL string concatenation (injection) |
| Service | Cross-cutting integration | Service interfaces (SOAP/REST), API gateway, service layer for orchestration, versioned contracts |

**Two exam-favourite techniques:** (1) *Facade* — a single interface in front of a subsystem, reducing coupling and client knowledge; (2) *workflow/work-process layer* for long multi-step business processes whose order changes with rules. Security add-ons: SSL/TLS at transport between layers across machines; IPSec at network layer for site-to-site.
---

# PART B — Architectural Patterns (EC-3 Question 1 territory — study deepest)

## B1. The pattern concept — definition and the four-part solution

**Formal definition:** An architectural pattern establishes a relationship between:
- a **Context** — a recurring, common situation in the world that gives rise to a problem;
- a **Problem** — the problem, appropriately *generalised*, that arises in that context;
- a **Solution** — a successful architectural resolution, appropriately *abstracted*.

**The solution is described by exactly four things (write all four in any "define pattern" answer):**
1. a set of **element types** (repositories, processes, objects…);
2. a set of **interaction mechanisms / connectors** (method calls, events, message bus, pipes);
3. a **topological layout** of the components;
4. a set of **semantic constraints** covering topology, element behaviour and interaction.

**Why patterns are valuable (seven properties — quote 3–4 in an exam answer):** proven design experience made reusable; they identify and specify abstractions above single classes; they provide a common vocabulary ("MVC" says everything); they document the architecture's vision for maintainers; they support construction with defined properties; they serve as building blocks for complex heterogeneous architectures; they help manage complexity.

**Pattern ≠ prefabricated library:** a pattern is a *scheme*, completed by the architect per problem; a library is finished code reused as-is. Applying the same pattern twice yields systems alike in broad structure but different in detail.

**Four pattern categories (POSA):**

| Category | Purpose | Patterns |
|---|---|---|
| From Mud to Structure | Give a sea of components a controlled decomposition | Layers, Pipes & Filters, Blackboard |
| Distributed Systems | Infrastructure for distributed applications | Broker |
| Interactive Systems | Structure human-computer interaction | MVC, PAC |
| Adaptable Systems | Support extension and adaptation | Reflection, Microkernel |

**By structure family:** Module → Layers; C&C → Pipes & Filters, Broker, MVC, Client-Server, P2P, SOA, Publish-Subscribe, Shared-Data; Allocation → Map-Reduce, Multi-Tier.

**How to answer "apply pattern X to scenario Y" (the reusable template):**
1. State assumptions the scenario leaves open.
2. Map the scenario's own components onto the pattern's **element types** — name each one explicitly ("the fraud-checker service is a filter").
3. Name the **connectors** between them.
4. Show the **topology** (a labelled diagram or walk-through).
5. State the **constraints** the pattern imposes and verify they hold.
6. State benefits *for the named quality in the scenario* and the *liabilities you accept*, then resolve or acknowledge.

## B2. The Layers pattern

**Context:** complex systems need portions to develop and evolve independently; developers need a clean separation of concerns.

**Problem:** segment software so modules develop/evolve separately with little interaction, supporting **portability, modifiability, reuse**.

**Solution:** divide software into **layers** — groupings of modules, each offering a cohesive set of services. The **allowed-to-use** relation is unidirectional (a layer uses only layers below it). Layers completely partition the software; each partition is exposed through a public interface.

**Constraints:** every piece of software allocated to exactly one layer; at least two layers (usually three or more); no circular use.

**Benefits:** change isolation (swap a whole layer), portability (replace the platform layer), reuse of standard layers, support for team organisation (work assignment follows layers).
**Weaknesses:** added up-front cost and complexity; performance penalty (calls hop through every layer); change cascades may still cross layers; hard to decide layer count.

**When to choose:** multi-team development, portability needs, clear abstraction hierarchy (UI / business / data). When not: hard real-time or performance-critical paths where hop overhead is unacceptable; systems whose dependencies are inherently non-linear.

**Course example:** presentation–business–persistence–database stack; a change to the database vendor touches only the persistence layer.

## B3. The Pipes and Filters pattern (deep — was EC-3 Q1a)

**Context:** systems that transform **streams of discrete data items** from input to output; the same transformations recur, so build them as independent reusable parts.

**Problem:** divide such systems into **reusable, loosely coupled components with simple generic interaction mechanisms**, flexible to combine — because (1) several developers build it, (2) the task decomposes naturally into stages, (3) requirements change so steps must be exchangeable/reorderable.

**Forces (each maps to a solution feature):**

| Force | Resolution in the pattern |
|---|---|
| Change by exchange/recombination | Small self-contained filters; uniform pipes |
| Small steps | Filters are small and single-purpose; non-adjacent filters share nothing |
| Varied inputs/outputs | Source and sink are plug-in ends of the flow |
| Intermediate results | Live inside pipes, not in user-managed files |
| Parallelism | Filters are independent → run concurrently |

**Solution — the elements (the exact vocabulary):**
- **Filter** — a *component* that transforms data read on its input port(s) into data written on its output port(s). Filters consume and deliver **incrementally** (never wait for all input) → **low latency** and **real parallelism**.
- **Pipe** — a *connector* conveying data from one filter's output to the next filter's input: **single source, single target, preserves order, never alters data**.
- **Data source / data sink** — the ends of the flow (input file, network feed; output store, terminal, API).
- **Processing pipeline** — the whole chain.

**Constraints:** pipes connect output ports to input ports; connected filters agree **only on the data format** of the pipe — everything else is hidden. That single agreement is what makes filters exchangeable and recombinable.

**Pipe annotations (engineering content of the connector):** capacity (bounded buffer → a fast producer is slowed to the consumer's speed; on full, producer blocks and retries), end-of-data marker (EOT / empty record), when-full and when-empty behaviour (block N seconds and retry). A pipe is a connector carrying a *protocol*, not just data.

**Variants:** *tee and join* — filters with several inputs/outputs form a directed graph, possibly with feedback loops; such graphs need formal analysis to prove termination. Another variant embeds a pipeline inside call-based software (compiler front end calls scanner/parser; a UNIX pipe hands off to the interpreter).

**Implementation steps (6):** (1) divide task into processing stages; (2) **define the data format** on each pipe (the contract); (3) decide pipe implementation (concurrent hand-off vs direct call); (4) build filters as incremental, single-responsibility components; (5) design error handling (pipes carry no error information); (6) set up the pipeline (fixed main program → shell/scriptable → visual pipeline builder).

**Known uses:** UNIX shells and tool chains; CMS Pipelines (record-based); LASSP numerical/graphics toolsets; compiler pipelines; modern data pipelines (ETL, streaming analytics).

**Benefits:** no intermediate files (but files can be inserted for debugging); flexibility by filter exchange; flexibility by recombination; reuse of small generic filters; rapid prototyping; efficiency by parallel processing.

**Liabilities (memorise — real questions ask for limits + remedies):**
1. **Sharing state is expensive** — non-adjacent filters share nothing; shared state must travel in the stream or live in a separate store.
2. **Parallel gain is often an illusion** — data transfer cost, context switches and pipe synchronisation can exceed the gain; one slow filter sets the pace; some filters need all input first.
3. **Data transformation overhead** — common pipe format forces parse/print conversions per filter.
4. **Error handling is awkward** — pipes carry data, not error information; fault detection and recovery across stages need extra machinery.

**Scaling a slow filter (the standard remedy):** partitioner filter → replicated filter instances (by key, e.g. customer ID) → merger filter. Bounded pipes apply back-pressure. Ultimate limit: slowest filter + inter-filter transfer.

**When to choose:** known, ordered transformation chain on a stream; ETL; compilers; shell tools; media processing. **When not:** heavy shared state; strict low-latency transaction handling; highly interactive control flow; when errors need rich cross-stage handling.

## B4. The Blackboard pattern

**Context:** an **immature domain in which no deterministic solution strategy is known** (speech recognition, image recognition, surveillance, damage assessment). Complete search of the solution space is infeasible; different algorithms (with different representations) solve partial problems; uncertain data and approximate solutions; experiments needed.

**Solution — three components:**
- **Blackboard** — the central data store holding the *solution space* (hypotheses at levels of abstraction) plus *control data*.
- **Knowledge sources** — separate, independent subsystems each solving one aspect; they **never call each other**; each evaluates its own applicability (execCondition), computes a result, and updates the blackboard (execAction).
- **Control component** — runs a loop monitoring blackboard changes and schedules which knowledge source acts next (**opportunistic, data-directed problem solving**).

**Variants:** production system (condition-action rules, conflict resolution); repository (same shared store, but control comes from user input or an external program instead of the state of the data).

**Known uses:** HEARSAY-II (speech recognition), HASP/SIAP (submarine detection from sonar), CRYSALIS (protein structure from X-ray), SUS (software understanding).

**Consequences:** easy experimentation and exchange of knowledge sources; but the shared structure needs synchronisation, testing/debugging is harder (no fixed sequence), and no guarantee of a solution.

**Blackboard vs Pipes and Filters (one-line contrast):** pipeline order is fixed by the pipes and data flows between adjacent steps; blackboard order is decided at runtime by the state of shared data visible to all specialists. Pipeline = known recipe; blackboard = emerging solution.

## B5. The Broker pattern (deep)

**Context:** a distributed, possibly heterogeneous environment of independent cooperating components.

**Problem:** decoupled components give flexibility, maintainability, distributability, scalability — but if every component handles its own IPC, it becomes tied to one network technology, one location map, often one language. **How can distributed software be structured so users of services need not know the nature or location of providers, and bindings can change dynamically?**

**Solution:** insert a **broker** between clients and servers. Servers **register** with the broker; clients send requests **via** the broker; the broker **locates the server, forwards the request, and transmits results and exceptions back**.

**Six component types (memorise):**

| Component | Responsibility |
|---|---|
| Client | Implements user functionality; sends requests through the client-side proxy |
| Server | Implements services; registers with the local broker; replies via server-side proxy |
| Broker | Registers/unregisters servers, offers APIs, transfers messages, locates servers, error recovery, interoperates with remote brokers through bridges |
| Client-side proxy | Hides IPC + marshalling; makes the remote object look local |
| Server-side proxy | Unmarshals incoming requests, calls the service, marshals results/exceptions |
| Bridge (optional) | Hides network/OS specifics between two brokers (interoperability) |

**Supporting notions:** *name service* (maps names to objects relative to a name space); *marshalling* (machine-independent encoding, e.g. ASN.1/XDR; unpacking = unmarshalling).

**Request path (be able to trace it):** client call → client-side proxy (pack_data, send_request) → broker (find_server, activate if inactive, forward_request) → server-side proxy (unpack_data, call_service) → server → response flows back through the same five hand-offs. **Every exchange passes at least five hand-offs — that is the cost; neither end knows the other's location, mechanism or language — that is the benefit.**

**Variants (named):** Direct Communication Broker (broker only locates; proxies then talk directly), Message Passing Broker (messages routed by type), Trader System (advertised service offers matched to requests), Adapter Broker (adapter layer handles registration/activation/dispatch), Callback Broker (event-driven, no fixed client/server split).

**Known uses:** CORBA, IBM SOM/DSOM, Microsoft OLE 2.x, the World Wide Web (browser = client, httpd = server, Internet gateway/DNS = broker-ish locator, URL = identifier).

**Benefits:** location transparency; changeability/extensibility (servers move/add/remove without touching clients); portability; interoperability between broker systems via bridges; reusability; easy isolated testing (each side tests against interfaces).

**Liabilities:** restricted efficiency (indirection, marshalling, hand-offs → latency); lower fault tolerance (broker or server failure halts dependents); testing/debugging the *assembled* system is demanding; the broker can be a bottleneck, a single point of failure, an attack target, and hard to test.

**Repair tactics:** performance bottleneck → increase resources / replicate broker (multiple copies); single point of failure → maintain multiple copies of the broker; bypass risk → restrict communication paths through the intermediary.
## B6. The Model-View-Controller pattern (deep — was EC-3 Q1b)

**Context:** interactive applications whose **user-interface software is the most frequently modified portion**; users want the same data seen from different perspectives (bar chart AND pie chart), both reflecting the current state.

**Problem (two pressures):** (1) the interface is a moving target — new functions, customer adaptations, platform ports, window-system releases each imply UI changes; (2) different users make conflicting demands on interaction style. If UI is interwoven with the functional core, flexibility is expensive, changes spread across many modules, and several systems may need maintaining. *How can UI functionality be kept separate from application functionality yet stay responsive to input and data changes? And how can several views be created, maintained and coordinated when the data changes?*

**Solution:** divide the application into **three components**:
- **Model** — the functional core: encapsulates data + application functionality; knows nothing about screens or input; maintains the **change-propagation registry** (attach/detach/notify).
- **View** — displays information obtained from the model; implements an **update procedure**; registers with the change-propagation mechanism; creates its controller (view : controller = 1 : 1); offers display-manipulation functions for non-model operations (e.g. scrolling).
- **Controller** — accepts user input as events, translates events into **service requests on the model** or **display requests on the view**; selects which view answers (view selection); registers with the change-propagation mechanism when its behaviour depends on model state (e.g. enabling a menu item).

**The five interactions (name them — exam gold):**

| Interaction | From → To | Kind |
|---|---|---|
| State query | View → Model | method invocation |
| State change | Controller → Model | method invocation |
| Change notification | Model → View (registered components) | event |
| User gestures | View → Controller | event |
| View selection | Controller → View | method invocation |

**Change propagation (the heart of the answer to "how does a business-logic change leave the UI untouched"):** user action → controller translates → service call on model → model changes state → `notify()` fires → every registered view calls back (`update()`) → each view re-fetches data and redraws → registered controllers adjust their own behaviour. Views are synchronised; the model never calls display code — it only announces change.

**Implementation items (10, quote 5–6):** separate HCI from core functionality; implement change-propagation (registry with attach/detach/notify + update interface); design views; design controllers; design view-controller relationship (1:1, view creates controller); implement set-up (model first, then views, then controllers); dynamic view creation at runtime; **pluggable controllers**; hierarchical views (composite); further decoupling from platform dependencies.

**Variant — Document-View:** merges view and controller into one component (fits platforms where display and event handling are interwoven, e.g. X Window, MFC); price = losing pluggable/exchangeable controllers. Known uses: Smalltalk VisualWorks (full MVC), MFC Document-View, ET++.

**As a C&C pattern:** elements = model (data/state + logic), view (UI component), controller (mediates, translates user actions); relation = **notifies** (relevant state changes); constraints = at least one instance each of model, view, controller; **the model should not interact directly with the controller**; weaknesses = complexity may not pay for simple UIs; the three abstractions may not fit some UI toolkits.

**Benefits (analyse, don't list):**
1. **Multiple views of one model** — model independent of representation.
2. **Synchronised views** — change propagation brings every registered view in line.
3. **Pluggable views and controllers** — replacement without touching the model.
4. **Exchangeability of look-and-feel** — porting touches only views + controllers.
5. **Framework potential** — regular enough to be packaged (Spring MVC, Struts).

**Liabilities:** increased complexity (registry + protocol); potentially excessive updates (every change notifies every view); intimate view-controller pairing; views/controllers coupled to the model's interface (interface change reaches all); inefficient view data access (call per value per redraw); porting still rewrites views/controllers; modern toolkits don't separate cleanly.

**The e-commerce answer skeleton (rehearse this):** *Assumption* → separation (catalogue/stock/pricing/orders/workflow/payment-gateway interface in Model; product/cart/status/report Views; "add to cart"/"pay" Controllers) → adaptability (new payment gateway = change one model component; order workflow change = confined to model's order logic; same model serves a new mobile view+controller pair) → price example (discount algorithm replaced inside model's pricing rules; cart controller calls the same price service; model notifies; views re-fetch; no view/controller code touched) → maintenance (each change has one home; UI not retested for business-rule changes) → honest cost (extra complexity + update protocol). Note: *scalability is NOT an MVC benefit* — hedge: an interface-independent model can in principle be hosted/replicated separately.

## B7. The PAC pattern (Presentation-Abstraction-Control)

**Idea:** for interactive systems made of **many cooperating agents**, each agent = **Presentation** (its own UI) + **Abstraction** (its own data/logic) + **Control** (mediates between its own P and A, and communicates with *other agents' Controls* only).

- **Top-level agent:** the system's core data + global UI (e.g. main window/menu bar).
- **Intermediate agents:** composite sub-features (e.g. a document editor's spell-check panel).
- **Bottom-level agents:** atomic self-contained elements (e.g. a single figure, a spreadsheet cell).

**Hierarchy:** a tree of agents; **agent-to-agent communication travels only via Controls** (P and A never talk across agents). Supports multiple simultaneous user hierarchies (unlike MVC's single triad), e.g. process control, air-traffic control, network management.

**PAC vs MVC (frequent comparison):**

| Aspect | MVC | PAC |
|---|---|---|
| Composition | One triad per application (repeated for same data) | Many hierarchical agents, each with own P-A-C |
| Mediator | Controller mediates model↔view; model notifies views | Control mediates within agent AND between agents (tree of controls) |
| Coupling | Views depend on the model's interface | Abstractions of different agents fully decoupled; change via Control coordination |
| Suitability | One dataset, many presentations | Complex composite UIs, independent sub-tasks |
| Cost | Simpler for small apps | More components; coordination overhead |

**Liability of PAC:** many agents → overhead and complexity; coordination logic can become intricate.

## B8. The Service-Oriented Architecture pattern (SOA) — and microservices

**Formal definition:** SOA describes a collection of **distributed components that provide and/or consume services**. Consumers understand and use services **without any detailed knowledge of implementation** — the only dependency is the **published service description/interface**.

**Context/problem:** interoperability among components on different platforms, in different languages, owned by different organisations, distributed across the Internet.

**Five component types:** service **provider** (published interfaces), service **consumer** (invokes directly or via intermediary), **ESB** (routes and transforms messages), **registry of services** (providers register; consumers discover at runtime), **orchestration server** (coordinates multi-service business processes/workflows). Providers may also be consumers.

**Three connectors:** SOAP (synchronous web-service calls, typically over HTTP), REST (HTTP request/reply verbs), asynchronous messaging (point-to-point or publish-subscribe via a messaging system — decouples in time).

**Constraints:** consumers connect to providers; intermediaries (ESB, registry, orchestration) optional. **Weaknesses:** complex to build; independent services evolve outside your control; middleware overhead; services can become bottlenecks without performance guarantees.

**Course example (Adventure Builder / OPC):** browser → Consumer Web site (HTTP/HTTPS) → Order Processing Centre via two published endpoints (purchase-order, order-tracking) → SOAP calls through a web-service broker to Bank CreditCardService, Airline/Lodging/Activity PO services; OPC also e-mails the user (SMTP) and consults a service registry. Shows cross-organisation published interfaces + registry + broker.

**SOA vs Microservices (both EC-3 topics — memorise this table):**

| Aspect | SOA (classic) | Microservices |
|---|---|---|
| Scope | Enterprise-wide integration, often across organisations | One product's decomposition into small services |
| Granularity | Coarse-grained services | Fine-grained, single-bounded-context services |
| Middleware | Heavy: ESB central to routing/transformation | Smart endpoints, dumb pipes: lightweight (HTTP/REST, messaging) |
| Data | Often shared databases via services | Database **per service**, private |
| Governance | Central contracts/registry | Decentralised; teams own their services ("you build it, you run it") |
| Coupling | Service contracts + shared data can couple | Independent deployability; API versioning |
| Failure handling | Depends on ESB design | Circuit breakers, bulkheads, graceful degradation designed per service |

**Microservices in practice (supporting tactics):** independent deployment; decentralised data → **eventual consistency**; anti-corruption layer at legacy boundaries; API gateway as single entry (auth, routing, aggregation); circuit breaker to fail fast on a downed dependency; service discovery; containerisation for density and CI/CD. Known case: Danske Bank moved from a monolith to microservices for independent release cycles; Netflix for resilience (replication + graceful degradation).

## B9. Four C&C interaction patterns — Client-Server, Peer-to-Peer, Publish-Subscribe, Shared-Data

### Client-Server
- **Context:** shared resources/services needed by many distributed clients, with access/QoS control.
- **Solution:** clients request services of servers via **request/reply** connectors; servers may be clients of other servers; one or more (distributed) servers.
- **Weaknesses:** server = performance bottleneck; server = single point of failure; client/server functionality split is costly to change later.
- **Example:** ATM system — ATM main process (client) → transaction authoriser, monitoring server; monitoring station (client) → monitoring + reconfiguration servers; TCP socket connectors.

### Peer-to-Peer
- **Context:** distributed entities, **each equally able to initiate interaction and each providing its own resources**, cooperating for a user community.
- **Solution:** components interact **directly as peers**; no peer (or group) is critical to system health; request/reply without the asymmetry.
- **Constraints:** may limit number of attachments per peer, search hops, who-knows-whom; star topologies with **supernodes** exist.
- **Weaknesses:** managing security, consistency, availability, backup, recovery is harder without a centre; small P2P systems may miss performance/availability goals.
- **Example:** Gnutella — leaf peers attach to ultrapeers; queries flow via Gnutella protocol; the file itself transfers peer-to-peer over HTTP.

**Client-Server vs P2P:** asymmetric roles vs symmetric; server can be SPOF vs no critical peer; fixed attachments vs attachments changing at runtime; bottleneck concern vs management complexity.

### Publish-Subscribe
- **Context:** independent producers and consumers of data whose **number and nature are not predetermined**.
- **Solution:** components interact through **announced events**; publishers announce on the bus; the connector delivers events to **registered subscribers**. Elements: component (with publish/subscribe ports), pub-sub connector (announce + listen roles), attachment relation. All components attach to an **event distributor** (bus or component).
- **Weaknesses:** increased latency; weaker scalability/predictability of delivery time; **less control over ordering; delivery not guaranteed** (publishers can't wait for every subscriber's ack).
- **Example:** Eclipse-based ArchE tool — UI event manager + observers registered on in-memory fact data + notifications to/from the Jess rule engine.

### Shared-Data (repository)
- **Context:** components must share and manipulate large amounts of **persistent data** that belongs to no single one.
- **Solution:** interaction dominated by exchange of persistent data between **data accessors** and at least one **shared-data store**; connector = data read/write; either side may initiate; accessors interact **only with the store**.
- **Weaknesses:** store = bottleneck; store = single point of failure; producers/consumers **tightly coupled through the data model** (a schema change is a change to every accessor).
- **Example:** security-administration system around one central security realm serving authentication, web sign-on, password reset, audit, provisioning.

**Pub-Sub vs Shared-Data:** events vs persistent data; event distributor vs store; delivery guarantees vs durability; latency/ordering concern vs data-model coupling.

## B10. Adaptable-system patterns (brief) — Microkernel & Reflection

**Microkernel:** a **minimal core** (mechanisms) providing inter-process communication + system resources, with **internal servers, external servers, adapters and clients** built around it. Mechanism = atomic core service; policy = complex function built on mechanisms. Examples: OS kernels (Mach), workstation window systems. Benefits: portability, adaptability, separation of mechanism/policy. Liabilities: performance overhead of crossing the kernel boundary; complexity of designing the minimal core.

**Reflection:** structure with a **meta level** (metaobjects = self-representation of the system, e.g. its types, operations, state access) and a **base level** (application logic); the two are linked so changing metaobjects (via a **metaobject protocol**) changes base-level behaviour at runtime. Purpose: run-time adaptation (adding operations, changing implementation strategy) without recompiling. Liabilities: complexity, meta-level changes can break invariants, performance overhead.

## B11. The Map-Reduce pattern (deep — was EC-3 Q2c)

**Context:** businesses must analyse enormous data volumes (petabyte scale) quickly.
**Problem:** efficiently perform a **distributed, parallel sort-and-analyse** of a large dataset, with a **simple way for the programmer to specify** the analysis.
**Solution — three parts:**
1. **Infrastructure** — allocates software to nodes of a massively parallel environment, sorts/shuffles data as needed, detects and recovers from failure.
2. **Map** (programmer-written) — filters/extracts the items to combine; produces `<key, value>` pairs; performs the **extract + transform** of ETL.
3. **Reduce** (programmer-written) — combines all values sharing a key; performs the **load** portion.

**Elements/relations:** map (multiple instances across processors), reduce (single or multiple instances), infrastructure (instantiate-monitor-control), deploy-on relation.

**Constraints (write all three):** input must exist **as a set of files**; **map functions are stateless and never communicate with each other**; the **only communication** between map and reduce is the emitted `<key, value>` pairs.

**Weaknesses (write all three):** overhead not justified for small datasets; **uneven partitions destroy the parallel gain** (one straggler sets finish time); **multi-stage reduces are complex to orchestrate**.

**Data flow:** input file portions → map instances (each writes partitioned output) → **shuffle/merge** gathers each key's pairs to one reducer (this is the "distributed sort") → reduce instances → outputs.

**Worked micro-example (rehearse; shows map/shuffle/reduce concretely):** log of users u1,u2,u1,u3,u2,u1 split across two mappers → mappers emit `<u,1>` per line → shuffle groups by key → reducers output `<u1,3>`, `<u2,2>`, `<u3,1>`; check 3+2+1=6. Hashtag variant: mappers scan posts and emit `(#tag, 1)` per occurrence; reducer sums; final `#Tech: 5` etc. Weighted-average variant: map emits `<page, (avg, count)>`; reduce computes Σ(avg×count)/Σcount — carrying the count is essential because simple averages cannot be combined.

**Where computation goes:** map runs **where the data already is** (data locality); only compact intermediate pairs cross the network.

**Trade-off table — batch vs real-time (the 2-mark Q2c answer core):**

| Feature | MapReduce (batch/Hadoop) | Stream processing (Spark Streaming/Flink) |
|---|---|---|
| Latency | High — job completes first, results after | Milliseconds–seconds as data arrives |
| Fault tolerance | Excellent — failed task re-run elsewhere, data from files | Complex — checkpoints + state management |
| Volume | Massive, petabyte-scale, cost-efficient | High but memory-intensive, always-on cost |
| Model | Rigid Map/Reduce logic | Flexible windows, joins, stateful logic |
| Best for | Historical/whole-dataset insight, daily reports | Trending topics, fraud *during* the transaction, dashboards |

**Recommendation template:** use batch for complete, cheap historical analysis; add a streaming path when seconds-level detection matters (or a **lambda** design: batch recomputes accurate results; stream serves fast approximations).

## B12. The Multi-Tier pattern

**Context:** distributed deployment needs the infrastructure split into distinct subsets.
**Solution:** organise execution structures as **logical groupings of components called tiers**; relations: *is-part-of*, *communicates-with*, *allocated-to* (when tiers map to platforms). **Constraint: every component belongs to exactly one tier.** Weakness: substantial up-front cost and complexity.
**Example:** client tier (browser) → web tier (servlets/JSP) → EJB tier (session-bean facades) → back end (database via JDBC; OPC application via SOAP).
**Layers vs tiers:** layers organise *code* (module structure, allowed-to-use); tiers organise *execution/deployment* (allocation structure). A system may use both simultaneously.

## B13. Patterns and tactics — molecule and atom

**Formal relationship:** a pattern is a **package of design decisions** found repeatedly in practice, with known properties; a tactic is a **single design decision** influencing one quality response. *A pattern is a molecule, a tactic is an atom; patterns are built from tactics.*

**Patterns are built from tactics — MVC decomposes into four modifiability tactics:** increase semantic coherence (data in Model, display in View, input in Controller); encapsulation (each part behind an interface); use an intermediary (the Controller); run-time binding (views bind to the model at runtime). Publish-Subscribe = intermediary + run-time binding.

**Tactics augment patterns (repair the weaknesses):** Broker's bottleneck → increase resources; Broker's SPOF → maintain multiple copies. Client-Server SPOF → replicate behind a request distributor. Shared-Data SPOF → replicate the store. Pub-Sub weak ordering → choose a connector with stronger delivery guarantees. Map-Reduce uneven partitions → re-partition to similar sizes.

**Every tactic has side effects — the cascade:** ping/echo → performance overhead → increase resources → cost + resource-utilisation → scheduling policy → modifiability (add/modify policy) → use an intermediary → ensure usage → restrict communication paths → performance overhead of the intermediary (recursive!). The process **ends** when remaining side effects are small enough to ignore.

**Patterns are underspecified:** they are the *beginning* of a design; augmentation with tactics stops when the system's requirements are met.

## B14. Pattern master comparison + selection decision rules

### Master table (condense everything above into this one revision table)

| Pattern | Type | Core idea | Best for | Key liabilities |
|---|---|---|---|---|
| Layers | Module | Unidirectional allowed-to-use groupings | Modifiability, portability, team work | Performance hops, up-front cost |
| Pipes & Filters | C&C | Filters transform; pipes convey; incremental streaming | Stream transformation, ETL, compilers | Shared state, transfer overhead, error handling, weak real-time |
| Blackboard | C&C | Shared board + opportunistic specialists | No deterministic strategy (recognition, fusion) | Hard to test, no guarantee, synchronisation |
| Broker | C&C | Mediator locates/forwards between clients & servers | Location transparency, heterogeneous distributed systems | Latency, broker SPOF, attack target |
| MVC | C&C (interactive) | Model + Views + Controllers, change propagation | UI variability, multiple views | Update storms, view-controller pairing, model-interface coupling |
| PAC | C&C (interactive) | Hierarchical P-A-C agents | Complex composite UIs | Coordination overhead |
| Client-Server | C&C | Asymmetric request/reply | Centralised shared services | Bottleneck, SPOF |
| Peer-to-Peer | C&C | Symmetric direct interaction | Availability without centre (file sharing) | Security/consistency/backup complexity |
| Publish-Subscribe | C&C | Event bus, announce/listen | Decoupled many-to-many integration | Latency, ordering/delivery guarantees |
| Shared-Data | C&C | Repository of persistent data | Many components + shared persistent state | Store SPOF, data-model coupling |
| SOA | C&C | Published services + ESB/registry/orchestration | Cross-organisation interoperability | Build complexity, middleware overhead, service evolution |
| Microservices | C&C (modern SOA) | Small independently deployable services, DB per service | Large team agility, independent releases | Distributed-system complexity, eventual consistency |
| Microkernel | Module (adaptable) | Minimal core + plug-in servers | Product lines, OS kernels | Hop overhead, core design difficulty |
| Reflection | Module (adaptable) | Meta level modifies base level | Runtime adaptation | Complexity, invariant risk |
| Map-Reduce | Allocation | Stateless map + key-grouped reduce on infrastructure | Petabyte batch analysis | Small data, straggler partitions, multi-stage jobs |
| Multi-Tier | Allocation | Logical deployment groupings | Web systems, scale-out | Up-front complexity |

### Selection decision rules (write these from memory)

- Transformation chain, order known, stream of items → **Pipes & Filters**.
- No deterministic strategy; specialists + evolving partial solution → **Blackboard**.
- Many UIs over one dataset; UI changes often → **MVC** (single triad) or **PAC** (hierarchical agents).
- Heterogeneous distributed services; location transparency → **Broker** (within one system) or **SOA/microservices** (across organisations).
- Strict change isolation + portability → **Layers**; deployment scaling → add **Multi-Tier** on top.
- Producers/consumers unknown in advance → **Publish-Subscribe**.
- Persistent data shared by many components → **Shared-Data**.
- Availability without a centre → **Peer-to-Peer**.
- Massive parallel batch analysis → **Map-Reduce** over **HDFS** (Part F).
---

# PART C — Security Patterns for Component-and-Connector Design (EC-3 Q2 territory)

## C1. Least privilege and where security enters a C&C view

**Principle of least privilege (Saltzer & Schroeder):** *every program and every user should operate using the least amount of privilege necessary to complete the job.* A **privilege** is a security attribute required for certain operations (not unique — several entities may hold it).

**Two key terms:**
- **Trust boundary** — a line in the architecture across which the level of trust in data or in the sender changes (public internet → DMZ, service → service).
- **Attack surface** — the set of points through which untrusted input can reach a component, plus the code that handles that input.

**Security questions per C&C element (a table worth quoting):**

| Element | Security question | Typical design response |
|---|---|---|
| Component | What authority does this process hold; what does an attacker gain by controlling it? | Minimal privilege; restricted file-system view; split by privilege |
| Connector crossing a trust boundary | Who is on the other end; may the data be believed? | Authenticate peer, validate every input, encrypt channel |
| Data store | Who may read/write; what need not be stored? | Access control, encryption, minimal retention, clearing after use |
| Cross-cutting record (logs/audit) | Can it be read to plan attacks; edited to hide attacks? | Protected log store; no secrets in logs |

**Two kinds of privilege escalation:**
1. **Remote, unauthenticated** — anonymous network party → most privileged local user (e.g. integer overflow in packet parsing of a root-running service). Most severe: no credential needed.
2. **After authentication** — ordinary authenticated user → privileged local user (e.g. bug in a post-login handler running at full privilege).

**Why least privilege is a design rule, not a prevention:** defects are inevitable (own code + third-party libraries); design must assume some component will eventually be controlled by an adversary. Least privilege limits *damage*, not occurrence. A process running as superuser converts any exploitable error into total machine control; a privilege-less process converts the same error into control of an identity that can do almost nothing. **Gap:** a service that legitimately needs a high privilege for a few operations still holds it for its whole lifetime — privilege separation fixes exactly this.

**Pattern ↔ tactics mapping (quote in answers):**

| Pattern | Tactics it realises (Chapter 3) | Served characteristic |
|---|---|---|
| Privilege separation | Separate entities; limit access; limit exposure; authorise actors | Contain a compromise (confidentiality, integrity) |
| Secure Logger | Audit trail (recover); encrypt data; limit access; verify message integrity | Non-repudiation, integrity of the record |
| Clear Sensitive Information | Limit exposure (over time); encrypt data as companion | Confidentiality of data in reusable resources |

**Abstraction levels (SEI):** architectural-level patterns (privilege separation, distrustful decomposition, defer-to-kernel) vs implementation-level patterns (secure logger, clear sensitive information) — the latter sit inside components, tied to secure-coding guidance.

## C2. Privilege Separation (the deep one — was EC-3 Q2a)

**Formal definition:** a generic approach that lets **parts of one application run with different privilege levels**, so errors in unprivileged parts can no longer be abused to gain privileges. Principle: **reduce the amount of code running with special privilege without limiting functionality.** Privileged part = **monitor**; unprivileged parts = **slaves**; a slave must ask the monitor for anything privileged, and **the monitor validates each request** (in the SEI catalogue, PrivSep is a specialisation of **Distrustful Decomposition**).

**Participants:**

| Participant | Role | Privilege |
|---|---|---|
| Privileged monitor | Fields initial requests, spawns the slave, validates and serves its requests, spawns later user-privileged child | Root/administrator |
| Unprivileged slave | Handles the untrusted interaction (network conversation, parsing) | Unused user ID, empty root directory (chroot) |
| User-privileged child | Handles the session after authentication | The authenticated user's privileges, nothing more |
| System user | The requester reaching the server over IPC (socket) | None until authenticated |

**Two phases:**
- **Pre-authentication:** slave has no process privilege and no file-system right; it handles network data and asks the monitor for privileged results.
- **Post-authentication:** after successful authentication the **old slave is terminated**; it **exports its state** (keys, counters, buffered network data, compression state — dynamic state via shared memory + marshalling) and the **monitor starts a new slave under the user's identity**. Rare privileged operations still go through the monitor.

**The monitor is a finite-state machine:** the set of valid requests changes with phase (early: few; after key exchange: user validation; after validation: authentication requests). **Any request the current state does not permit → terminate.** Keeping valid requests few and state-driven shrinks the attack surface left to a compromised slave.

**Three request types the slave may make:**

| Type | Slave obtains | Example |
|---|---|---|
| Informational | A privileged piece of information | Signature over exchanged values (host private key never leaves the monitor); password-check result |
| Capability | A file descriptor it couldn't open | Pseudo-terminal creation |
| Change of identity | Right to continue as the authenticated user | Pre-auth → post-auth transition |

**Interface rule:** the monitor must never export secrets (e.g. the private key itself) — it exports *services over the secret* (request-a-signature, not give-me-the-key).

**Implementation mechanics (Unix):** monitor forks slave → slave `chroot` to an empty directory → `setuid`/`setgid` to unused IDs → IPC over a socket pair established before fork. No kernel support needed.

**OpenSSH (the canonical example — rehearse the numbers):** before separation, the per-connection daemon needed superuser throughout (key exchange signatures, password database, pseudo-terminals, spawning the user's session) → any packet-parsing bug = remote root. After: ~**two thirds of code** (including third-party crypto + compression libraries) runs without privilege (~75% of OpenSSH's own code); ~**2% code change** (~950 of 44,000 lines) + ~3,000 new monitor/marshalling lines; historical vulnerabilities (packet integer overflow → remote root; channel off-by-one → post-auth root; compression-library double-free) would all have been **contained**; measured login/transfer times showed **no significant penalty**. Related uses: OpenBSD daemons, Postfix/qmail (distrustful decomposition), Chromium sandbox (broker + restricted renderers — same structure: small trusted mediator, large exposed worker assumed hostile).

**Challenges → strategies (each pair = a mark):**

| Challenge | Strategy |
|---|---|
| No portable way for an unprivileged process to change identity | Export slave state; monitor spawns new process under user ID; marshalling + shared memory |
| Monitor interface is itself an attack surface | Small request set per state; strict validation; terminate on invalid; never export secrets; audit the interface |
| Code changes across process boundary | ~2% refactoring + monitor code; interfaces got cleaner as a side effect |
| State continuity (compression streams) | Shared-memory-backed allocation so state can be handed over |
| Platform differences (Windows) | Process-creation call plays fork's role; degrade gracefully if descriptor passing/shared memory missing |
| Residual abuse (resource exhaustion, signalling) | Per-slave distinct UID; read-only empty root; process/descriptor limits; monitor watches CPU |
| Administration overhead | Accepted cost (new UIDs, monitoring) |

**Benefits vs liabilities:** contained compromise, concentrated audit effort, third-party-library defects contained — vs admin overhead, extra processes + protocol complexity, monitor must be correct (it holds the privilege), and **denial of service within the slave's rights is not prevented**.

**Do not confuse:** privilege separation (splits *one application*, monitor mediates) vs sandbox/confinement (OS restricts the whole app externally; cannot know internal state) vs layering (abstraction ordering; security only a side effect).

## C3. The Secure Logger pattern

**Intent:** (1) prevent an attacker from **gathering useful information** from logs; (2) prevent an attacker from **hiding actions by editing** logs — while keeping logs available to authorised readers. All log output passes through a **secure logging subsystem** that stores records so unauthorised users cannot read them and modifications are **detectable**.

**Applicability (three conditions):** the system logs to a file/subsystem; log contents could help an attacker plan attacks; logs are used to detect and diagnose attacks (this is why the integrity half exists — an editable log is useless as evidence).

**Participants:** Application (generates events) → **Secure Logger** (masks sensitive fields; protects each record — encrypt/sign/chain; appends to a protected store) → **Log Reader** (special mechanism; standard text editors won't work) → **Log Viewer** (authorised user's application). The store's access is limited to the logging subsystem.

**Dynamics:** event → logger filters/masks → protects record → appends → authorised reader/viewer decrypt + verify; a failed verification reveals modification/removal.

**Implementation guidance:** one logging interface for the whole system (use an abstract interface + Abstract Factory so dev-time simple logger swaps to secure logger for release); **never write to stdout/stderr in a delivered system** (that escapes protection); prefer an existing secure logging facility (syslog-ng, encrypted-log libraries, filesystem/volume encryption).

**What must never be logged (OWASP/CWE-532):** session identifiers, access tokens, passwords, DB connection strings, encryption keys/primary secrets, cardholder/account data, sensitive personal data, source code. Remove debug logs before production; change logging config between debug and production. **Arithmetic of exposure:** a log is copied (collectors, backups, analytics, support) — every copy widens the audience; a card number vaulted correctly but logged once has leaked.

**Making the log trustworthy (integrity half):** tamper detection (chain each record with a hash/MAC over content + previous record — any edit breaks all later links; signing with a key held off-host defeats forgery); ship to read-only media / central log server promptly (an attacker with full host control can still delete everything); separate partition + write-only account; restrict and monitor log access; syslog over TLS across networks; validate/sanitise events from other trust zones (neutralise CR/LF/delimiters → defeats **log injection**); logging must not exhaust resources or stop the application.

**Trade-offs:** protected store → operators need the reader/viewer too (slower diagnosis); tamper detection is detection, not prevention; single logging interface needs discipline (stray writes bypass); central storage aids correlation but is a high-value target, SPOF and adds per-record cost; masking must balance evidence vs secrecy; under-logging leaves no evidence, over-logging creates noise ("alarm fog").

## C4. The Clear Sensitive Information pattern

**Problem:** sensitive data does **not disappear when the program finishes with it** — freed buffers, caches and files keep contents until overwritten; any later reader (another process, a core dump, reused memory) can recover them.

**Intent:** ensure sensitive data is **erased from reusable resources before (or immediately after) the program finishes with it**.

**Applicability:** wherever sensitive data (passwords, keys, card numbers, tokens) lives in reusable resources — memory buffers, caches, temp files.

**Mechanics:** overwrite before release (in destructors/finalisers **so error paths are covered too**); prefer writing over secrets in ways that can be purged; beware **optimising compilers/dead-store elimination** removing "useless" overwrites (use platform secure-zeroing APIs); clear caches of sensitive entries; delete/shred temp files; keep sensitive data in fewer places (less clearing to guarantee).

**Trade-offs:** overwriting costs time; you must find *every* sensitive item and *every* release point; compilers may elide clears; clearing cannot recall copies made elsewhere (logs, backups, network traces) — which is why it pairs with the Secure Logger.

## C5. Combining the two in a payment-processing system (was EC-3 Q2b — rehearse)

**Secure Logger = the persistent, widely read path (audit trail). Clear Sensitive Information = the transient path (buffers).** One transaction trace to quote:
1. Service reads the card number into buffers (transient secret).
2. Tokenises it (token stored; PAN not) and obtains authorisation.
3. Logs **one event** containing: timestamp, transaction ID, amount, outcome, **token reference, last four digits** — no PAN, no track data (matches card-industry rules: no authentication data retained after authorisation; no readable account numbers in logs).
4. Buffers overwritten in a destructor that also runs on error paths (Clear Sensitive Information).
5. Log record signed/chained and shipped to the central log server.

**Benefits:** both leak paths closed (log reveals nothing, memory holds no residue); disputes resolvable from the audit record; compliance aligned. **Challenges:** debugging uses tokens/IDs, not raw requests; the logger's own queues are buffers needing clearing; error paths and debug settings are where secrets leak; both patterns add latency and operational cost.

## C6. Web & identity security from the technology module (TLS, OpenID, OAuth, DMZ, LDAP)

- **TLS/SSL (transport layer):** encrypts the channel between client and server; server authenticates via certificates; protects data in transit. Place at every hop crossing a trust boundary (browser↔web tier, service↔service). IPSec protects at network layer (site-to-site VPNs).
- **OpenID = authentication ("who are you?"):** delegated identity — the app redirects the user to a trusted identity provider which asserts identity; the app never handles the password. (AuthID concept: verifying the claimed identity.)
- **OAuth = authorisation ("what may I access on your behalf?"):** the user grants a client limited access to a resource server via **access tokens**, without sharing credentials; scopes limit what the token can do; refresh tokens renew. Typical pairing: OpenID Connect for login, OAuth for API access.
- **DMZ (demilitarised zone):** a perimeter sub-network between two firewalls where public-facing servers (web, reverse proxy) live; internal network unreachable directly from outside; traffic rules restrict DMZ→internal flows.
- **Firewalls / next-generation firewalls:** packet filtering → stateful inspection → application awareness (control by application, user, content). Place: perimeter + internal segmentation.
- **LDAP / directory services:** hierarchical directory of users/groups used for centralised authentication & authorisation lookups; IAM builds on directories (provisioning, entitlements).
- **Password handling rule:** never store plaintext — store salted hashes; never log secrets (C3).
---

# PART D — Integration Strategies & Architecting for the Cloud (Sessions 11–12)

## D1. Integration strategies — five ways two systems exchange data

| Strategy | Mechanism | Strengths | Weaknesses | Best when |
|---|---|---|---|---|
| **File transfer** | Export to files (fixed/CSV/XML) at intervals; other side reads | Simple, decoupled in time, works with legacy | Latency = schedule; format agreement needed; no feedback; error handling manual (retry whole file) | Bulk periodic exchange, batch reporting, legacy systems |
| **Shared database** | Both systems read/write one DB (cf. Shared-Data pattern) | Immediate access to same data; no format translation | Tight coupling through schema; one DB is SPOF + bottleneck; ownership conflicts | Tightly related internal applications needing same data |
| **Remote Procedure Call (RPC)** | Client invokes a procedure on the server over the network; stubs/proxies marshal | Natural programming model; strong typing | Tight temporal coupling (both must be up); location/knowledge coupling; brittle across versions | Coordinated internal services needing immediate answers |
| **Messaging** | Producer posts messages to queues/topics (pub-sub); consumer reads asynchronously | Temporal decoupling (receiver may be down), buffering, guaranteed/retryable delivery, pub-sub fan-out | Needs broker infrastructure; ordering/consistency semantics; message design effort | Decoupled services, event-driven architectures, load buffering |
| **WebSockets** | Full-duplex persistent TCP connection after an HTTP upgrade handshake | Real-time bidirectional push; low per-message overhead; works through browsers | Connection state on server (scaling harder); firewall/proxy handling; own auth on reconnect | Live dashboards, chat, collaboration, streaming updates to browsers |
| **API Gateway** | Single entry point in front of internal services: routing, auth, rate limiting, aggregation, versioning | Hides internal topology; central cross-cutting policy; protocol translation | Gateway can be bottleneck/SPOF; can grow into a mini-monolith | Microservices exposed to clients; mobile back ends |

**Choosing rule of thumb:** immediate consistency inside your own system → RPC/REST; decoupling, fan-out, resilience → messaging; bulk, periodic, legacy → file transfer; browser push → WebSockets; many clients over many services → API gateway. Combine: gateway front, REST for synchronous reads, messaging for events, file drop for legacy.

## D2. Cloud definitions (NIST) — characteristics, service models, deployment models

**Seven characteristics (quote six to be safe):** on-demand self-service; ubiquitous network access; resource pooling; location independence; rapid elasticity; measured service; multi-tenancy. (NIST's published five drop location independence and multi-tenancy as separate items — say "seven in the course, five in NIST".)

**Service models — who is the consumer, what does he get:**

| Model | Consumer | Gets | Controls |
|---|---|---|---|
| SaaS | End user | Finished applications (mail, storage) | Nothing below the app; data/settings |
| PaaS | Developer/admin | Platform: runtime, languages, tools (e.g. LAMP; App Engine) | Application code only |
| IaaS | Developer/admin | Processing, storage, network; deploy arbitrary OS + software | Everything above the hardware |

**Deployment models:** private (one organisation), public (sold to general public), community (shared by organisations with common concerns), hybrid (composition of two or more; remain unique entities). **Service model and deployment model are independent axes** (public cloud can host SaaS; private cloud can offer IaaS).

## D3. Why the cloud is cheaper — three economic reasons (also CBAM-friendly)

1. **Economies of scale** — large data centres (100k+ servers) vs small (<10k): power is 15–20% of operating cost (bulk rates, cheap geography, renewables); shared racks/switches; hardware discounts up to 30%; administrators: small DC ≈ 150 servers per admin, large ≈ 1,000+ (10,000 servers need ~67 admins; 100,000 need ~100 → ~6.7× fewer per 10k); fixed security/redundancy investment amortised over more servers.
2. **Higher utilisation** — virtualisation co-locates applications with different load shapes: random-access smoothing; time-of-day (office email peaks 10:00, video peaks evening); time-of-year; resource-pattern complementarity (CPU-heavy beside I/O-heavy); uncertainty spikes. Filling each other's gaps = fewer servers per user.
3. **Multi-tenancy** — one application instance serves many consumers (salesforce.com style): help-desk savings, single upgrade for all, one version to develop/maintain. Price = shared resources → new security concerns (D6).

## D4. Base mechanisms — hypervisor, VM, HDFS, network

**Hypervisor:** software layer on the host that **creates, schedules and isolates virtual machines**. **Virtual machine:** isolated address space; looks like bare metal to the application; has its own IP address; runs any OS/software the host processor can execute. **Mechanism:** each VM has its own page table; the **host page table points to the VM's page tables** → every address generated by a VM passes a table the hypervisor controls → VMs cannot reach each other's memory → the hypervisor is security-critical (it is the only barrier between tenants).

**HDFS (Hadoop Distributed File System) — the cloud file system:**
- Elements: application processes (with client code), **NameNode**, many **DataNodes**.
- Connectors: client↔NameNode = **RPC**; NameNode↔DataNodes = **RPC**; client↔DataNodes = **streaming protocol**.
- Each block held by **three DataNodes** (replication default 3).
- **Write path (8 steps, sunny day):** app writes unaware → client buffers to block size (64 KB in the course example; block sizes like 128 MB in Hadoop) → client asks NameNode for a new block → NameNode returns 3 DataNodes → client sends block to first, tells it the other two → pipeline: DN1 writes & forwards to DN2 → DN2 to DN3 → each DataNode acks → client commits to NameNode after all three ack.
- Failure handling: **heartbeats** from DataNodes detect failure; NameNode's metadata re-replicates lost blocks; **rebalancer** evens distribution; pipelines move data between nodes.
- Properties: petabyte capacity, distributed, replicated, fault-tolerant/self-healing. **Metadata lives in NameNode memory** → thousands of small files are a known weakness.

**Network:** gateway property — IP addresses can be reassigned when a VM instance fails (service continues at a new location); underpins IaaS's automatic reallocation.

## D5. IaaS / PaaS structure and multi-tenancy designs

**IaaS managers:** virtual resource manager → persistent object manager; cluster manager (one cluster); file system manager (network-wide FS); node manager (one server). **Two services to note:** automatic IP reallocation on instance failure; **automatic scaling** (VMs created/deleted with load — practical rapid elasticity).

**PaaS:** integrated stack (LAMP: Linux, Apache, MySQL, Python); developer writes code, platform assigns to layers; cost = loss of control over those layers.

**Multi-tenancy designs (the Tailspin survey example):**
- **Multi-instance, single-tenant:** separate instance per client — strong isolation, easy customisation, higher cost.
- **Single-instance, multi-tenant:** one shared instance — economy of scale, but isolation/customisation risk.
- **Tenancy degree per tier:** decide web / app / data tiers independently — heavy processing justifies a dedicated tier (one tenant's load can't slow another); confidential data justifies a dedicated data tier; light processing + non-confidential data → share for economy. Cross-industry utilitarian workloads (email, expense, survey) suit high tenancy; strategic custom apps suit low.
- **Security issues:** inadvertent sharing (same table shared by orgs distinguished only by a tenant column — any query omitting the filter leaks rows); VM escape through the hypervisor. **Defences:** enforce tenant filter in one shared data-access layer; row-level security by tenant; separate schemas/databases for stricter tenants; per-tenant keys; patched hypervisor.

## D6. Distributed cache, containers, serverless

- **Distributed cache:** an in-memory cluster (e.g. Redis/Memcached) between services and the database. Gains: latency, DB load reduction, session state for stateless scale-out. Decisions: eviction policy, TTL, cache-aside vs write-through, **hot keys**, coherence (invalidation on write). Placement: web-tier output caching, data-tier query caching, session caching. Risk: staleness — pair with bounded staleness windows or invalidation.
- **Containers:** OS-level virtualisation — processes in isolated namespaces/cgroups sharing the host kernel; image = application + dependencies. vs VMs: VMs virtualise hardware (each its own OS; stronger isolation; heavier, slower start); containers start in seconds, run many per host, ideal for microservices + CI/CD; isolation is weaker (shared kernel) → image hygiene + runtimes matter. Orchestration (Kubernetes) schedules, scales, heals containers.
- **Serverless (FaaS):** functions managed by the provider, scale to zero, event-triggered, billed per invocation; no server administration. Benefits: no capacity planning, cost for spiky workloads, faster iteration. Limits: execution time limits, cold starts, state must live outside (DB/cache), harder end-to-end testing, tighter provider coupling. Typical design: API gateway → functions → managed DB/queue.

## D7. Failure management in distributed systems

**Assume failure is routine** (thousands of commodity servers). Principles:
- **Detect:** heartbeats/ping-echo between nodes; timeouts on every call; checksums on data.
- **Recover:** retries (idempotent operations!), replication (active pair or warm spare), failover routing, state checkpoints, graceful degradation (shed optional features, keep core), load shedding/bounding queues at overload.
- **Prevent:** transactions for atomic multi-step changes; no SPOF (replicate every tier); bulkheads (isolate resource pools so one failing component can't exhaust everything).
- **Self-healing design loop:** monitor → detect anomaly → automatic recovery action → verify → report. (Microsoft "self-healing" guidance.)
- Two generals/coordination reality: cross-node transactions are expensive — prefer **eventual consistency + compensating actions** where the business allows (links to CAP below).

## D8. The CAP theorem (memorise precisely)

**Statement:** a distributed web service cannot simultaneously guarantee all three of:
- **C — Consistency:** all nodes see the same data at the same time;
- **A — Availability:** node failures do not prevent surviving nodes from operating;
- **P — Partition tolerance:** the system continues to operate despite network partitions.

**Any two, never all three.** Since a network partition cannot be ruled out across machines, the real choice during a partition is **C vs A**.

| Pair kept | Called | Behaviour during partition | Examples (course) |
|---|---|---|---|
| A + P | **AP** — not consistent | Node in one partition returns its older copy while another partition writes | CouchDB, Cassandra, ScyllaDB |
| C + P | **CP** — not available | The node with the inconsistent copy refuses/is unavailable until the partition heals | MongoDB, Redis |
| C + A | **CA** | Requires permanent contact between nodes — exactly what a partition denies; **theoretical only** | No practical CA database |

**Micro-example (be able to run it):** N1 and N2 both hold X=5; partition; client writes X=7 at N2; another client reads X at N1 → AP answers 5 (stale but available); CP refuses until healed (consistent but unavailable). **Clarify:** CAP availability ≠ the availability quality attribute (uptime %); CAP consistency ≠ ACID consistency.

**Design consequence:** for a mobile/offline or geo-distributed system, pick per operation — payments → CP; social feeds → AP with eventual consistency. For an SLA like 99.95% (4.38 h/year), design for AP + conflict resolution (Part E/E5).

## D9. Cloud best practices (AWS + Azure + Microsoft guides)

**AWS — four differences cloud makes:** IT assets become programmable resources; global availability, effectively unlimited capacity; higher-level managed services; security built in.
**AWS ten design principles (quote 5+):** scalability (add resources, don't replace); **disposable resources instead of fixed servers**; automation (script provisioning/recovery); **loose coupling**; services not servers; choose the right database; remove single points of failure; optimise for cost (match demand, metered billing); **caching**; security at every layer.

**Azure multi-tenant guidance:** single vs multi-tenant decision; multi-tenant data architectures (shared schema w/ tenant ID vs schema-per-tenant vs DB-per-tenant); partitioning; maximise availability/scalability/elasticity; securing multi-tenant apps; managing/monitoring.

**Microsoft App Architecture Guide:** layered guidelines (presentation/business/data/service), quality attributes, cross-cutting concerns, physical tiers & deployment.

**Cloud strategy — begin with service definition:** (1) identify the cloud-related IT services to offer/procure; (2) document internal processes affected; (3) map applications/workloads to services. Value rises from **IT task automation** (provisioning, patching — simple, fixed, technical) to **IT service automation** to **business service automation** (cross-silo, situational, coordination-oriented) — define cloud services at the service level, not just tasks.

## D10. Qualities that change in the cloud

- **Security:** multi-tenancy → inadvertent information sharing (residual data on reallocated disks); VM escape (breaking the hypervisor — "so far purely academic"); side-channel attacks (cache monitoring — academic); denial-of-service by one tenant. Risk decision: which applications may go multi-tenant at all.
- **Performance:** auto-scaling is not instant — **a new VM takes time to start**; during that window existing instances carry the load. Consequences: know your resource requirements; build that knowledge into applications; make applications **self-aware/proactive** (predictive scaling, queue-depth signals) rather than reactive after the spike.
- **Availability:** individual instances WILL fail — build in detection + correction (HDFS is the worked example). **Cloud availability ≠ application availability:** the platform's SLA covers the platform; a single-instance app dies with its instance whatever the SLA says.
---

# PART E — Architecting for Mobile (Session 13 — was EC-3 Q3)

## E1. What mobile design means + the five design considerations

**Definition:** mobile application architecture = the structure of the **device component** (UI, background processing, storage, event handling) **together with its connection to the server-side component**.

**Five design considerations (each maps to a technique below):**

| Consideration | What it demands | Treated in |
|---|---|---|
| Simple user interface | Easy typing, large buttons, minimal features/menus | E4 |
| Responsive design | Adaptation to screen sizes and orientations | E4 |
| Compact code | Less CPU, memory, storage | E6/E7 |
| Few layers | Performance — every layer is a hop and a memory cost | E6/E7 |
| Connectivity | **Store data locally, synchronise later** when the connection is poor | E5 |

**Constraint → attribute → response (expanded table for "analyse" answers):**

| Constraint | Quality attribute affected | Typical architectural response |
|---|---|---|
| Limited CPU/memory/storage | Performance | Compact code, few layers, heavy work moved to server |
| Battery is finite | Energy efficiency | Batch network calls, no continuous polling, platform-scheduled background work |
| Intermittent connectivity | Availability | Local store, deferred synchronisation, retry |
| Small touch screens, many sizes | Usability | Simple interface, large controls, responsive layout |
| Many device models/OS versions | Modifiability, portability | Shared common code, platform adapters, device-matrix testing |
| Device lost/stolen; hostile owner | Security | Authenticate users, encrypt stored data, keep secrets on server |
| Updates only when installed (native) vs instantly (web) | Modifiability | Choice of app type; server-driven configuration |

## E2. Types of mobile application (native / mobile web / hybrid) + the two axes

**Definitions:** *Native* — built for one platform with its language/tools (Swift/Objective-C for iOS; Java/Kotlin for Android). *Mobile web* — HTML5/CSS3/JavaScript in a browser (progressive web apps deliver app-like behaviour). *Hybrid* — a web app (HTML/CSS/JS) inside a **native shell** that exposes camera, accelerometer, GPS. *Cross-platform* — the same code runs on multiple platforms (a property overlapping hybrid, also provided by non-web toolkits).

**Two axes (draw this quadrant):** capability used (full vs partial) × platform reach (single vs multiple): Native = full/single; Hybrid = full/multiple; HTML5 = partial/multiple; the fourth quadrant is empty.

**Pros/cons table (quote exactly):**

| Type | Pros | Cons |
|---|---|---|
| Native | High performance; superior UX; access to all OS features | One platform only; special language per platform; version updates needed |
| Web app | Easy deployment of new versions; common code base | Little device-hardware access; lower UX; must be found via browser |
| Hybrid | No browser needed; single code base; device-hardware access via shell | Slower (extra layer) |

**Tools table:** Native — iPhone SDK / Android SDK / Windows Phone SDK (Objective-C/Java/.NET). Cross-mobile ("build once, deploy as native apps") — RhoMobile (Ruby), Titanium Appcelerator, PhoneGap, Worklight (JS/HTML). Mobile web — Sencha, jQuery Mobile (JS/HTML). Also named: **Xamarin** (C#/.NET). Modern landscape: React Native (JS bridging to native widgets), Flutter (Dart, renders its own UI), Kotlin Multiplatform (shared logic, native UI), .NET MAUI (successor of Xamarin), Ionic/Capacitor (web-view family).

**Examples (name them):** native — Google Maps, Facebook, LinkedIn; web — Flipkart, BookMyShow, MakeMyTrip; hybrid — Twitter, Uber, Instagram.

**Architecture of mobile web apps:** mobile clients (Android/iPhone/Windows) send HTTPS requests — HTTP verb (GET/POST/HEAD/PUT/DELETE) + URL — to **RESTful web services**, receiving **JSON**; services run behind web servers in an enterprise data centre or in a cloud with cloud storage.

## E3. Native vs cross-platform — the six-attribute decision (was Q3a — rehearse)

| Attribute | Native | Cross-platform |
|---|---|---|
| **Performance** | Compiled for the platform; direct API calls; best for continuous camera processing, sustained tracking, graphics-heavy work | Extra layer (web view/bridge/engine) adds start-up + call overhead; fine for forms/lists/maps; weakest at the hardware-intensive edge |
| **Security** | Full platform facilities: secure key storage, biometrics, certificate pinning, obfuscation; platform documentation | Same facilities via plug-ins/bridges; web code in a shell is easier to tamper with; each third-party plug-in widens supply-chain risk |
| **User experience** | Platform look, gestures, accessibility, new OS features immediately | Near-native on common screens; conventions must be reproduced; new OS features wait for the framework |
| **Development efficiency** | Two code bases, two skill sets, two test cycles | One team, one code base for most of the app, shared tests; some platform-specific work remains |
| **Code reusability** | Almost none between iOS and Android (beyond contracts) | High for business logic and screens; native modules for hardware/special behaviour |
| **Platform-specific optimisation** | Unrestricted (graphics APIs, background rules, power management) | Only via native modules — which reintroduces per-platform code |

**Decision template (rehearse):** *Assumption:* GPS + camera on both Android and iOS; limited team/budget. GPS capture and photo capture-and-upload are served well by hybrid/cross-platform plug-ins; **real-time camera analysis or continuous background tracking favours native**. Decision: **cross-platform (React Native/Xamarin) for screens and logic + native modules for the two hardware components**; if the requirement is real-time analysis, go native for those parts or wholly. **Method:** a short **spike** of the GPS+camera feature in the chosen framework on both platforms, measured against the performance target, decision recorded, agile iterations with device testing in the pipeline. A defensible decision names *the hardware requirement that decides it*, never a general preference.

## E4. Responsive design & mobile UI patterns (was Q3b — rehearse)

**Definitions:** **Responsive design** — build ONE website for laptop/mobile/tablet whose rendering adapts to screen size and orientation. **Mobile-optimised site** — a separate, reduced-feature site (second product to build and maintain). **Flexi grids** — divide the screen into columns; assign elements to columns; choose layout by screen size.

**Three ingredients (all CSS/HTML):** (1) **flexible grids** — media queries detect screen size; grid width is a **percentage of the screen**; (2) **flexible images** — image width a percentage of its column; (3) **layout choice** — per screen size, columns sit side by side or stack. Classic example: **Boston Globe** — desktop three columns; tablet aside drops below; phone single vertical column.

**Mechanics (Beyond-the-course detail that strengthens answers):** viewport meta tag (without it mobile browsers render desktop width and shrink); media queries with **breakpoints**; mobile-first base style (single column, queries add columns as screens widen); flexbox/CSS grid; responsive images (`srcset`/`<picture>`, container-capped width); touch targets ~44–48 units; **limit:** CSS-hiding content doesn't stop the browser downloading it — desktop-sized images to a phone save layout effort, not bandwidth.

**Five mobile UI patterns:** action bars (frequent actions); social login (Facebook/Google instead of local ID/password); large buttons; notifications (recent activity); discoverable controls (appear when an item is selected — WhatsApp Forward button).

**Q3b answer table (loading / navigation / interaction):**

| Factor | Improvement | Responsive mechanism |
|---|---|---|
| Page loading time | Images scale to the column, not fixed desktop size; secondary content hidden/moved on small screens; no separate mobile site or redirect; one code base cached once | Flexible images; hide/move content; single website |
| Navigation | Multi-column desktop page becomes one vertical column; compact menu; large tap targets; no sideways scroll/pinch-zoom | Flexible grid; layout chosen by screen size |
| Overall interaction | Readable without zooming; finger-sized controls; layout follows rotation portrait↔landscape | Media queries on size and orientation |

Outcome line: the reader sees the **full site content** (not a reduced version) and the publisher maintains **one site**.

## E5. Connectivity — Store locally, synchronise later (the named mobile pattern)

**Definition:** the mobile application records data in **local storage on the device** and transmits to the server **when connectivity permits**.

**Course example (Project Smart Health):** responsive UI (Bootstrap + Durandal) on laptop/tablet/phone → **SQLite** local storage for offline use → **REST APIs** (when connected) → web server → **MySQL** server database (loaded from a pipe-separated file). Doctors enter patient data on devices; synchronisation happens later.

**Architectural effect:** the **UI depends on the local store, not the network** — application availability no longer equals connection availability. **Price:** two copies of the data → consistency becomes a design problem.

**Synchronisation decisions (the analysis marks):** a **queue of pending changes** preserves order; transmissions must be **idempotent** (safe to repeat — a connection may fail after the server applied the change but before the phone learns it); a **conflict rule** is required because two devices can edit the same record offline — *last-write-wins, server-wins, field-level merge, or manual resolution*; operations that cannot tolerate conflicts (a payment) are **not queued** — refused until connectivity returns. This is the CAP choice at device level: favour availability for notes/lists; favour consistency for money.

## E6. Android application architecture (components + intents — decompose any app)

**Four component types:**

| Component | Role | Example |
|---|---|---|
| **Activity** | The user interface (one screen) | Booking screen |
| **Service** | Background process, no screen | Download in progress; location sender |
| **Content provider** | Storage access path (SQLite, files) — the API to data, not the data itself | Calendar DB provider |
| **Broadcast receiver** | Reacts to system/app events | SMS arrival, high-pulse event |

**Intents** wire them: **directed intents** (to one named component) and **broadcast intents** (to every registered receiver). A broadcast receiver only reacts; the work is done by the Service/Activity it starts.

**Three canonical mini-architectures (rehearse one in full):**
1. *Activity + content provider:* calendar screen never touches the database — asks the SQLite provider.
2. *Background service:* Google Play screen (Activity) starts a download Service that talks to the server and writes via the File-API provider; the user can leave the screen.
3. *Broadcast chain (patient monitoring):* pulse-monitor Service reads device → emits "high pulse" **broadcast intent** → two receivers fire independently (Alarm Service sounds; Call-doctor Service) → Call-doctor asks File-API provider for doctor's number → calls. Property to note: the pulse monitor doesn't know who listens — a nurse-notification receiver can be added without changing it (**publish-subscribe inside one device**).

**App decomposition exercise (rehearse):** *Uber* — UI Activity = book-a-cab screen; Broadcast receiver = receives cab location from backend, supplies UI; Service = sends cab location to backend during the journey; Database = config file with user data. *Courier delivery* — Activity shows package list from SQLite provider; "mark delivered" hands data to a Service that syncs to the central server via a REST API call (and, per E5, writes locally first).

**Confusion guard:** an Android *Service* is a no-UI background component — not an SOA service; a *content provider* is the access path, not the database.

## E7. Mobile platform architecture (the stack under the app)

**Layered stack:** Hardware → Kernel & device drivers → Core → Framework → Applications. Typical form splits the framework/applications into a **native side** (native framework + native apps) and a **web side** (web framework + web apps). The **HAL (Hardware Abstraction Layer)** exposes device capabilities to the higher-level API framework.

**Hardware platform:** CPU, GPU, DSP, multimedia (audio/video/gestures), connectivity (Wi-Fi/LTE), location, display/LCD, camera, sensors.

**Platforms named:** iOS (Apple, multitasking), Windows Phone (Metro UI — discontinued), Android (Google, Linux kernel, multitasking, software stack), Tizen (open-source, Linux Foundation, multiple profiles). Tizen layers: web apps / native apps → web framework (W3C/HTML5 + device APIs: BT, Call, NFC) → native framework (social/content, locations, media, graphics/UI…) → core (app framework, multimedia, security…) → Linux kernel + drivers.

**Android stack (top-down):** System apps (dialer, email, camera…) → **Java API framework** (content providers, view system, activity/location/package/notification/resource/telephony/window managers) → **Native C/C++ libraries** (WebKit, OpenGL ES, Libc, media framework) → **Android Runtime (ART)** → Kernel & drivers (HAL between runtime and kernel).

**Why this matters for cross-platform:** a hybrid tool inserts itself *at the framework layer* — it reaches camera/GPS through the platform's native APIs/HAL, which is why basic capture works cross-platform but continuous processing pays the bridge overhead.

## E8. Cloud back ends for mobile + tenancy degree

- Most mobile apps lean on a cloud back end: REST APIs, cloud storage, push notification services, identity services (cf. OpenID/OAuth in C6).
- Vendor availability promises (99.95% = 4.38 h/year) still require the **app side** to survive disconnection → E5's store-locally pattern is the mobile half of the CAP choice (availability for notes; consistency for payments).
- **Tenancy for the back end (Tailspin):** multi-instance single-tenant vs single-instance multi-tenant; choose per tier (E/D5): heavy processing → dedicated tier; confidential data → dedicated data tier; utilitarian cross-industry workloads tolerate high tenancy.
---

# PART F — Big Data, Hadoop/MapReduce, NoSQL and New Technologies (Session 14)

## F1. Big data — definition, 3Vs, uses

**Definition:** a collection of data sets so large and complex that they become difficult to process **using on-hand database management tools or traditional data-processing applications**.

**Three dimensions (Volume–Variety–Velocity) — each pushes architecture a different way:**

| Dimension | Range | Effect on architecture |
|---|---|---|
| **Volume** | MB → GB → TB → PB → EB → ZB → YB | Forces **distribution across many machines** (one box can't hold it) |
| **Variety** | Structured/unstructured: text, clickstream, logs, sensors, images, audio, video | Defeats a **fixed relational schema** → flexible data stores |
| **Velocity** | batch → daily → hourly → near-real-time → real-time | Forces the **batch vs stream** choice (F3/F6) |

**Why now:** proliferation of the Internet, social media, e-commerce — ordinary activity became recorded data. Scale snapshots (for colour in answers): YouTube > 500 h of video uploaded per minute; Meta sharing in billions of items/day; genome sequencing from 13 years to under 24 hours — volume growth went with processing-tool growth.

**Uses (name a few):** fraud detection (banking), customer targeting (retail), sentiment analysis (social media), call-centre/service analytics, industrial equipment monitoring (power plants), weather forecasting for fishermen/farmers, recommendation systems.

**Opening move for any big-data scenario answer:** state the volume, the variety of formats and the velocity, then link each to a design decision.

## F2. Hadoop and HDFS (know both as separate answers)

**Hadoop definition:** an open-source Apache framework that **processes large heterogeneous datasets in a distributed fashion across clusters of commodity hardware, using a simplified programming model**; provides reliable shared storage + analysis. **Two components:** **HDFS** (storage) + **MapReduce** (computation). (Hadoop ≠ HDFS ≠ MapReduce — a standard confusion trap.)

**HDFS architecture:** **NameNode** holds **metadata** (files, blocks, block→node mapping) **in memory**; **DataNodes** hold replicated **blocks** (commonly 128 MB; default replication factor 3); **heartbeats** from DataNodes let the NameNode detect failures and re-replicate; **pipelines** move data between nodes; a **rebalancer** evens distribution. Properties: petabyte scale, distributed, replicated, fault-tolerant/self-healing. HDFS is also the cloud file system (D4 — write path in 8 steps).

## F3. MapReduce in application (the 2-mark formula + worked examples)

**Definition:** a pattern for analysing a large distributed dataset by **executing in parallel on many nodes**; **map performs extract + transform and produces <Key, Value> pairs; reduce summarises** the transformed data; parallelism → low latency and high availability.

**Phases (name all three + infrastructure):** *Map* (each server processes its local share, emits pairs) → *Shuffle and sort* (framework collects all pairs with the same key onto one node — the distributed sort) → *Reduce* (aggregates each key's values into the insight). The infrastructure deploys instances, shepherds data, recovers failures.

**Worked example A — average time per web page (rehearse the arithmetic):** servers hold click logs; each map emits `<page, (avg time, visit count)>`; reduce computes the **weighted average** Σ(avg×count)/Σ(count). Example: /home = (40×3 + 50×3 + 30×2)/8 = 41.25 s. The count is essential: plain averaging of per-server averages (40+50+30)/3 = 40 s is **wrong** because servers saw different visit numbers. Check: weighted result equals the unpartitioned computation.

**Worked example B — hashtags on social-media logs (the actual Q2c shape):** posts split into shards → mappers emit `(#tag, 1)` per occurrence → shuffle groups (e.g. #Tech:[1,1,1,1,1]) → reducers sum (#Tech: 5, #Music: 2, #AI: 3, #Sports: 2) → check Σ = number of pairs emitted (no loss, no double count). Optional optimisation: a **combiner** (local mini-reduce on the mapper) cuts network traffic — valid because sums are associative.

**Constraints/weaknesses (from B11 — reuse here):** files-based input; stateless maps; only <key,value> communication; not justified on small data; uneven partitions → straggler; multi-stage reduces awkward.

## F4. When to use Hadoop — and when NOT

**Good fits:** analytics; search; data retention; log-file processing; text/image/audio/video analysis; recommendation systems — large volumes processed as a whole, minutes-to-hours latency acceptable.

**Poor fits (with reasons — quote these):**

| Scenario | Reason |
|---|---|
| Low-latency / near-real-time access | Processing model is batch |
| Very many small files | NameNode holds all metadata in memory → grows with file count |
| Multiple/arbitrary writes, file-to-file writes | Design favours bulk read + process of large files |

**Telecom-vs-bank contrast (memorise):** a telco analysing a month of call records to reprice plans → Hadoop fits; a bank deciding *before approving* whether a card payment is fraudulent → does NOT fit (needs real-time analytics).

**Batch vs stream vs MPP (the comparison table):**

| Feature | MapReduce/Hadoop (batch) | Stream (Spark Streaming, Flink) | MPP/SQL (Snowflake, BigQuery) |
|---|---|---|---|
| Latency | High (job-sized) | ms–s | Medium; fast queries on structured data |
| Fault tolerance | Excellent (re-run tasks) | Complex (checkpoints, state) | Managed by engine |
| Data volume | Petabyte-scale, cheap | High but memory-intensive | High; cost grows with storage+compute |
| Model | Rigid Map/Reduce | Flexible windows/joins/state | Declarative SQL |
| Choose when | Whole-history insight, daily reports, cheap | Seconds-level detection | Simple behavioural metrics, SQL skills |

**Lambda-style recommendation:** batch path recomputes accurate results over all history; streaming path serves fast, approximate results meanwhile.

## F5. NoSQL databases — why they exist and what they trade away

**Definition:** a non-relational database designed to be **scalable, fast, available, and tolerant of semi-structured/unstructured data**, usually **without ACID transactions and without joins**. ACID = Atomicity, Consistency, Isolation, Durability. **Sharding** distributes one database over several machines; **replication** keeps copies on several.

**Why they arose (three reasons):** massive web data processed sequentially (RDBMS overhead in creation/maintenance); CAP says all three guarantees cannot hold; the relational model is not the best fit for some applications.

**Five strengths with mechanisms + two omissions:**

| Property | Mechanism | | Omission | Why |
|---|---|---|---|---|
| Scalable | Sharding (capacity grows by adding nodes) | | **No ACID transactions** | Atomic, isolated coordination across machines needs locking/agreement — the work sharding avoids |
| Fast | In-memory operation | | **No joins** | Joined rows may live on different shards; instead, store read-together data together |
| Available | Replication (another copy answers) | | | |
| Semi/unstructured tolerance | Records not forced into a predefined table layout | | | |
| Rapid adaptation | Record structure can change without DB-wide migration | | | |

**Illustrated advantages (the USERS example):** *flexibility* — iteration 2 adds a Twitter handle: relational rejects until schema change; document store accepts the new field at once; *simplicity* — one user with skills + experience = one document vs three normalised tables joined on user ID; *cost* — RDBMS cost curve climbs steeply (bigger single machine) vs NoSQL near-linear (add commodity nodes).

**Single-node RDBMS vs CAP (three configurations):**

| Configuration | Gets | Loses |
|---|---|---|
| One node reads+writes | Consistent | Not partition-tolerant; availability compromised during writes |
| Two machines | High availability; partition tolerant | No (strict) consistency |
| One machine + backup/mirror | Consistent; partition tolerant | Not highly available |

**Who uses NoSQL (name 2–3):** Tesco (millions of products/promotions), Ryanair (mobile app, 3M+ users), Marriott (reservation system booking $38B/yr), Gannett/USA Today (Presto CMS), GE (Predix industrial IoT). Shared traits: volume, record-shape variety, speed.

## F6. The four NoSQL data models + choosing SQL vs NoSQL

| Model | Unit of storage | Access | Examples | Suits |
|---|---|---|---|---|
| **Document** | Self-describing documents (JSON/BSON/XML), nested | Key → whole document; queried inside | MongoDB, CouchDB | Profiles, catalogues, content management — read-one-entity workloads |
| **Key-value** | Opaque value per key | Hash-based get/put; value not interpreted | Redis, Riak; HBase (key+timestamped values, no transactions) | Sessions, caches, shopping carts — simple lookups at scale |
| **Column-oriented** | Values of one field of many records stored together | Column-range scans | Cassandra, HBase (column-family) | Analytics over few columns, time-series, write-heavy logs |
| **Graph** | Nodes + directed edges, both with properties | Traversals | Neo4j | Social networks, recommendations, fraud rings |

**Note the course quirk:** HBase is listed as key-value in the cloud module but is more commonly a column-family store modelled on Bigtable — say "column-family (Bigtable-style)" and you're safe.

**SQL vs NoSQL decision rule:** need ACID across records (bank ledger: debit+credit as one unit), complex joins, mature tooling → **relational**; need horizontal scale, flexible record shape, speed on huge simple workloads, availability during partitions → **NoSQL** (then pick the model by access pattern). Hybrid is normal: relational for money + NoSQL for catalogue/streams.

**Shard-key warning:** a key that spreads load evenly keeps nodes busy; a key concentrating popular items creates a **hot spot** no added node relieves.

## F7. Beyond batch — Spark, streaming, IoT (brief but exam-usable)

- **Spark:** in-memory processing engine — MapReduce-class batch plus interactive queries, streaming (micro-batches), and MLlib; avoids Hadoop's disk round-trips; belongs to the same ecosystem (reads/writes HDFS).
- **Real-time analytics:** process events or small windows as they arrive (trending topics, abuse detection within seconds); trade-offs vs batch = lower latency but complex state handling, weaker ordering/exactly-once guarantees, always-on cost.
- **IoT architecture sketch:** sensors/devices → gateway → ingestion bus (messaging) → stream processing for alerts + storage (time-series/NoSQL) → analytics/ML → dashboards. Qualities: massive device counts (scale-out ingestion), intermittent connectivity (store-and-forward at the gateway), security (device identity, encrypted channels).

## F8. AI/ML architecture (pipeline, not magic)

**Pipeline (name the stages):** data collection (batch + streams) → preparation/cleaning/feature engineering → **training** (choose algorithm, fit model on training data, validate on held-out data) → evaluation (accuracy/recall vs business metric) → **deployment** (model served via API or batch scoring) → monitoring + **retraining** as data drifts. Architectural consequences: training is compute-heavy (cluster/GPU, cf. Hadoop/Spark), serving is latency-sensitive (cache features, scale serving tier separately), models are **artifacts with versions** (reproducibility), feedback loops need data capture. Steps commonly cited: collect → prepare → choose algorithm → train → evaluate → deploy → monitor/retrain.

## F9. Blockchain (one-paragraph answer)

A **distributed, append-only ledger** replicated across nodes: transactions are grouped into blocks; each block carries a hash linking it to the previous block (tamper-evident chain); consensus among nodes replaces a central authority (public permissionless, e.g. Bitcoin; or permissioned consortium, e.g. trade finance). Architectural properties: integrity/auditability without a trusted centre, but **throughput limits, latency, and the cost of consensus**; store data *on-chain only when* it must be trusted jointly — big blobs belong off-chain with on-chain hashes. Use cases from the course list: supply-chain provenance (IBM–Maersk shipping), finance settlement.
---

# PART G — Economics of Architecture: CBAM (Session 15 — was EC-3 Q4)

## G1. The decision-making context and the two central terms

Business goals → architectural strategies → each strategy affects **several quality attributes at once**, all of which feed **benefit**; the strategy also carries a **cost**. No strategy is free; none improves only one attribute.

- **Utility:** the benefit stakeholders derive from the quality-attribute responses an architectural decision achieves — the **common currency** that lets unlike attributes (seconds, %, person-days) be compared.
- **Value for Cost (VFC):** a function of the costs, benefits and uncertainty of a portfolio of architectural investments: `VFC = Benefit / Cost`. It **informs but does not make** the decision — the decision stays with stakeholders (with their risk aversion).
- Economic analysis is reserved for the **most basic, strategy-setting decisions**; fine-grained design choices don't justify the elicitation effort.

## G2. Utility-response curves

**Definition:** the graph of the relationship between a scenario's **response measures** and the corresponding **utility values**. Fix the anchors: **best case = 100** (level above which no further utility is foreseen — e.g. 0.1 s response perceived as instantaneous), **worst case = 0** (minimum threshold; below it the system is useless). (In practice the tables sometimes deviate — worst 5, best 85 — the anchors are a convention applied loosely.)

**Why curves, not single targets:** stakeholders can name "99.999% available", but the economic question is how much they'd value 99.99% — if almost as good, its lower cost may win. The curve shows **where improvement is worth buying**.

**Five curve shapes (sketch + read):**

| Shape | Meaning | Example |
|---|---|---|
| (a) steep then flat | Early improvement valued; later adds little | Performance from 12 s → 2 s matters hugely; 2 s → 1.5 s little |
| (b) flat then steep | Value concentrated at highest levels | Certifications requiring near-max compliance |
| (c) S-shaped | Little value below threshold; rapid gain in a band; saturation | Security: little value until a compliance threshold, then rapid |
| (d) straight line | Every increment worth the same | Cost: each unit of spend lowers utility equally (falling line) |
| (e) steps | Value arrives only at defined levels | Capacity tiers |

**Reading rule:** the same horizontal distance (a→b) is worth a lot or nothing depending on where it falls on the curve — a cheap a→b move is excellent on shape (a) at low values, nearly worthless at high values.

**Response vs utility (clarify in answers):** "0.5 s" is a response; "utility 80" is the stakeholder valuation. Interpolation moves response→utility, never backwards; utilities of different scenarios can be summed because they share the 0–100 scale; responses cannot be summed.

## G3. Weighting scenarios and the benefit formula

Scenarios are weighted because optimising a scenario nobody cares about wastes effort.

**Formulas (memorise both):**
- Benefit in one scenario: `b(i,j) = U_expected − U_current` (can be **negative** — a strategy that harms a scenario counts against itself)
- Total benefit of strategy i: `B(i) = Σ_j ( b(i,j) × W(j) )` over all scenarios j.

**How weights W(j) arise:** in the nine-step method each stakeholder distributes **100 votes** among scenarios (based on *desired* response); highest-rated scenario gets **W = 1.0**, others weighted **relative** to it. (Alternative: rank weighting — 1, (N−1)/N, …)

**Mini worked example (rehearse):** scenarios A, B, C ranked 1st/2nd/3rd → weights 1, 2/3, 1/3. Strategy X: gains 30, 10, −10 → B = 30×1 + 10×(2/3) + (−10)×(1/3) = **33.33**. Strategy Y: gains 10, 40, 0 → B = 10 + 26.67 + 0 = **36.67**. Y has the larger benefit — but see G4 before deciding.

## G4. Value for cost and the practicalities of cost

`VFC(i) = B(i) / C(i)`; rank strategies by VFC, **not** by raw benefit.

**Where C(i) comes from:** architectural cost models are scarce (most cost models measure whole-system size/function points, not "what does publish-subscribe cost here"); in practice: corporate models from similar past architectures + senior-architect judgement. **Relative estimates are "enormously helpful"**: "if A costs $x, B looks like $2x, C like $0.5x" — a coarse ranking suffices to rank strategies; absolute figures are unnecessary.

**Mini worked example (continuing):** X costs 100, Y costs 200 → VFC_X = 0.33, VFC_Y = 0.18 → **X ranks first** despite the smaller benefit: 10% more benefit is not worth twice the cost. This is exactly the case the ratio exists to expose.

**Common confusion:** VFC is benefit ÷ cost — **not** net benefit (B − C) and not a financial return with cancellable units; it has meaning only *within one analysis*. (The NASA case calls the same ratio "ROI".)

## G5. The nine steps of CBAM (memorise the list verbatim)

1. **Collate scenarios** — gather, let stakeholders add; prioritise informally (high/medium/low) against business goals; **keep the top one-third**.
2. **Refine scenarios** — stimulus-response form; elicit **worst-case, current, desired, best-case** response levels.
3. **Prioritise scenarios** — each stakeholder distributes **100 votes** (based on desired response); **keep the top 50%**; highest gets weight 1.0, others relative; also list the quality attributes of concern.
4. **Assign utility** — utility for each response level of each surviving scenario (table: one row per scenario, four columns).
5. **Map architectural strategies to scenarios** — for each strategy × affected scenario, record the **expected response level**.
6. **Determine the utility of expected responses by interpolation** — read utilities off the curve:
   `y = y_a + (y_b − y_a) × (x − x_a)/(x_b − x_a)`
   Worked: points (0.5 s, 80) and (1.5 s, 50); expected 0.7 s → y = 80 + (50−80)×(0.7−0.5)/(1.5−0.5) = 80 − 6 = **74**.
7. **Calculate total benefit per strategy** — `b = U_expected − U_current`, normalise by votes, sum across scenarios (G3 formula; negative gains included).
8. **Choose strategies by VFC within constraints** — estimate cost & schedule; VFC = B/C; rank; **take the top-ranked until budget or schedule is exhausted** (ties broken by stakeholder judgement = step 9).
9. **Confirm results with intuition** — check chosen set against business goals; if misaligned, iterate the analysis.

**Three filters (clarify which step filters what):** business-goal ranking filters scenarios (top 1/3, step 1); votes filter scenarios (top 50%, step 3); **VFC ranking filters strategies** (step 8). Only the third filters strategies.

## G6. The NASA ECS case (the numbers to reproduce if asked)

**Setting:** EOSDIS Core System — hundreds of GB/day of satellite data, 250 standard products, thousands of GB archived at 8 data centres; project manager had a limited annual budget; used CBAM for an ROI decision.

- Step 1: **ten scenarios** (reduce hung/lost distribution requests, failed/hung/lost orders, tracking without spreadsheets, user information on failures, artificial size/number limits, notification flood, process 50 GB in a day / 1 TB in a week).
- Step 2: refined with worst/current/desired/best (e.g. #1: 10%→5%→1%→0% hung).
- Step 3: close-knit team used **consensus instead of individual voting**; 100 votes total (10, 15, 15, 10, 15, …).
- Step 4: utility tables; e.g. scenario 8 current utility 20 at "30% limited", reaching 100 only at "0% limited"; scenarios already at 5% hung have utility 80 and gain only 15 by reaching 1% (existing level acceptable).
- Step 5: **ten strategies** (e.g. order persistence on submission; order retry without resubmission; cancel orders at any stage…), each with expected response per affected scenario.
- Step 6: interpolation for each expected response.
- Step 7: weighted benefits (e.g. raw benefit on a 5-vote scenario ×5; on a 10-vote scenario ×10 — **the weight belongs to the scenario, not the strategy**).
- Step 8 (the table to memorise):

| Strategy | Cost | Total benefit | VFC | Rank |
|---|---|---|---|---|
| 1 Order persistence | 1200 | 950 | **0.79** | 1 |
| 2 | 400 | 200 | 0.5 | 3 |
| 3 Order retry without resubmission | 400 | 275 | **0.69** | 2 |
| 4 | 200 | 100 | 0.5 | 3 |
| 5 | 400 | 120 | 0.3 | 7 |
| 6 | 200 | 50 | 0.25 | 8 |
| 7 | 200 | 70 | 0.35 | 6 |
| 8 | 300 | 150 | 0.5 | 3 |
| 9 | 1000 | 225 | **0.22** (lowest) | 10 |
| 10 | 100 | 25 | 0.25 | 8 |

Budget example: **1600 units** → take rank 1 (cost 1200, remaining 400) → rank 2 = strategy 3 (cost 400) → budget exhausted → chosen pair delivers **950 + 275 = 1225 benefit points**. Ties (0.5 three times) would need stakeholder judgement (step 9). Lesson the case validates: the VFC ranking roughly followed the order strategies were proposed — the stakeholders' intuition.

## G7. What CBAM delivers (and its limits)

**Delivers:** (1) a **VFC ordering of architectural strategies**; (2) social/cultural benefit — structured discussion: scenarios clarified in advance, utilities assigned to specific response goals, priorities made explicit (architecture discussions are normally unstructured); (3) utility as the **common coin** across attributes.
**Limits:** output only as good as the stakeholder consensus; ranking sensitive to utility values; **numbers support but do not replace** the intuition check of step 9.

## G8. Opportunity cost (the Q4a lens)

**Definition (economics):** opportunity cost = **the value of the best alternative forgone** by a choice. It is not a second payment; it is the benefit that would have been obtained elsewhere. **Sunk costs** (already spent, unrecoverable) are **not** opportunity costs.

**Applied to the scalability-first (S) vs speed-first (F) choice:**
- S's opportunity cost = the features / earlier release the time would have bought.
- F's opportunity cost = the scalability utility surrendered + rework if load grows.
- Analysis: express as scenarios (peak users; release date); elicit four response levels; utilities 0–100; vote-weights; B = Σ(b×W); compare VFC = B/C with relative costs ("if S costs x, F ≈ 0.6x").
- **Factors under limited resources/tight timeline:** probability & timing of growth (growth within months favours S); cost of adding scalability later (cheap if the design stays modular → favours F); penalty for a missed deadline; reversibility; stakeholder risk tolerance; **the hybrid** — F plus cheap scalability-enabling tactics (stateless, separable components).
- Conclusion template: demand uncertain + firm deadline → F-with-scalability-path likely wins VFC; growth certain & imminent → S wins. Check against intuition (step 9); the decision rests with stakeholders.

---

# PART H — Recent Developments & Emerging Trends (Session 16 — short answers)

- **WebAssembly (Wasm):** a portable binary instruction format executing at near-native speed in browsers (and on servers via runtimes); language-agnostic compilation target (C/C++/Rust/Go); sandboxed. Architectural use: performance-critical client modules (image/video editing, games), plugin sandboxes, edge functions; complements rather than replaces JavaScript/DOM.
- **Service mesh:** a dedicated infrastructure layer handling **service-to-service communication** (east-west traffic): sidecar proxies intercept calls and provide mTLS encryption, retries/timeouts, circuit breaking, traffic shifting, observability — moving these from application code into the mesh control plane. Use when microservice fleets make uniform security/reliability policy in code painful; cost = operational complexity + sidecar latency.
- **Edge computing:** run computation near data sources/users (CDN nodes, cell towers, on-prem gateways) instead of a central cloud. Motivations: latency (AR/industrial control), bandwidth cost (filter/aggregation before backhaul), data locality/compliance, resilience offline. Architecture pattern: tiered processing — edge for immediate response, cloud for heavy analytics/training (ties to IoT in F7 and to cloud elasticity in D).
- **Forward engineering for OO and client-server systems (course topic):** mapping design artifacts forward into implementation — from UML/domain models to code and database schema; for client-server: partitioning classes into client-side presentation and server-side business/data tiers, generating interface stubs and persistence mappings. (Concept-level awareness is sufficient.)

---

# PART I — Mnemonics & Memory Hooks (closed-book essentials)

**Scenario parts (6):** **S-S-A-E-R-R** — *Source, Stimulus, Artifact, Environment, Response, Response measure.* ("Students Study All Eight Real Requirements.")

**Utility tree levels (4):** **U-A-R-S** — *Utility → Attributes → Refinements → Scenarios* ("Under All Roofs, Shelter").

**ATAM 9 steps (order):** *3 Presentations, then 3+3 analysis beats:* **P-B-P, I-U-A, B-A-P** → Present ATAM, Present Business drivers, Present architecture; Identify approaches, Utility tree, Analyse approaches; Brainstorm scenarios, Analyse again, Present results. ("Please Bring Pizza: I Usually Ask Before Adding Pepperoni.")

**ATAM findings (4):** **R-N-S-T** — Risk, Nonrisk, Sensitivity point, Tradeoff point ("Real Navies Sense Threats").

**Cloud characteristics (7):** **O-U-R-L-E-M-M** — On-demand, Ubiquitous network, Resource pooling, Location independence, Elasticity, Measured service, Multi-tenancy ("Our Users Rarely Lose Electricity; Machines Multiply").

**CAP (3, keep 2):** Consistency, Availability, Partition tolerance — AP = stale but up (Cassandra), CP = right but refuses (MongoDB/Redis), CA = theoretical.

**CBAM 9 steps:** **C-R-P-U-M-U-B-V-C** — Collate, Refine, Prioritise, Utility, Map strategies, Utility-by-interpolation, Benefit, VFC-choice, Confirm intuition ("Careful Reviewers Prefer Using Meaningful Utility Benefits; Very Considerate").

**Pipes & Filters elements:** **F-P-S-S** — Filter, Pipe, Source, Sink ("Fresh Pipes Serve Streams"). Filter transforms; pipe only conveys.

**MVC interactions (5):** Query, Change, Notify, Gesture, Select — *views query, controllers change, model notifies, gestures go to controller, controller selects the view.*

**Broker 6 participants:** **C-S-B-P-P-B** — Client, Server, Broker, Client-proxy, Server-proxy, Bridge ("Clients Send Bribes; Proxies Pass Buckets").

**MapReduce constraints (3):** **F-S-K** — Files input, Stateless maps, <Key,value> only ("Files Ship Keys").

**NoSQL omissions (2):** **A-J** — ACID, Joins ("All Joins abandoned").

**Security pattern rows:** Privilege separation = *component* row; Secure Logger = *record* row; Clear Sensitive Information = *data* row.

**Mobile 5 considerations:** **U-R-C-L-C** — UI simple, Responsive, Compact code, Few Layers, Connectivity/store-locally ("Users Rarely Carry Large Chargers").

**Android 4 components:** **A-S-C-B** — Activity, Service, Content provider, Broadcast receiver ("Apps Span Calendar + Broadcast"), wired by **intents** (directed vs broadcast).

**Availability maths:** series multiply; active pair `1−(1−A)²`; `A = MTTF/(MTTF+MTTR)`; 99.95% = 4.38 h/year.

**CBAM maths:** `b = U_expected − U_current`; `B = Σ(b×W)`; `VFC = B/C`; interpolation `y = y_a + (y_b−y_a)(x−x_a)/(x_b−x_a)`.

---

# PART J — Decision Trees (which choice when — quote these rules)

## J1. Which architectural pattern?

```
What is the system's dominant organising problem?
|
+-- Transform a stream of items, steps known in order?
|     +-- yes, and steps are independent, reusable → PIPES & FILTERS
|     +-- yes, but shared state/interactive control → reconsider (layers + components)
|     +-- no deterministic strategy; specialists contribute to a shared solution
|           → BLACKBOARD
|
+-- UI over data, UI changes often / multiple views?
|     +-- one dataset, one triad per view → MVC
|     +-- hierarchical composite UIs, independent agents → PAC
|
+-- Distributed services?
|     +-- within one system, location transparency → BROKER
|     +-- across organisations, published contracts → SOA
|     +-- many small independently deployable services, per-service DB
|           → MICROSERVICES (+ API gateway + messaging)
|     +-- central shared resources, simple request/reply → CLIENT-SERVER
|     +-- no central authority required → PEER-TO-PEER
|     +-- producers/consumers unknown in advance → PUBLISH-SUBSCRIBE
|     +-- persistent data shared by many components → SHARED-DATA
|
+-- Deployment?
      +-- code organisation → LAYERS ; deployment groupings → MULTI-TIER
      +-- petabyte batch analysis → MAP-REDUCE over HDFS
```

## J2. Cloud / data store choices

```
Need to scale beyond one machine?
+-- Strict ACID, complex joins, moderate scale → RELATIONAL DB (scale up)
+-- Huge volume, flexible schema, availability under partitions → NoSQL
|     +-- read/serve whole entity (profiles, catalogues) → DOCUMENT
|     +-- simple key lookups at massive scale (sessions, carts) → KEY-VALUE
|     +-- analytics over few columns, time-series, write-heavy → COLUMN
|     +-- relationships are the data (social, fraud, recommendations) → GRAPH
+-- Consistency vs availability during partitions:
      +-- payments, inventory → CP (MongoDB, Redis)
      +-- feeds, availability under SLA → AP + eventual consistency (Cassandra, CouchDB)
+-- Whole-history batch insight, cheap → Hadoop/MapReduce
+-- Seconds-level reaction → stream processing (Spark/Flink)
+-- Spiky event-driven workloads, no server admin → serverless
+-- Latency/bandwidth/locality near users → edge computing
```

## J3. Mobile decisions

```
Device hardware needed?
+-- Heavy/continuous hardware use (real-time camera, sustained GPS) → NATIVE
+-- Standard capture + shared screens/logic, limited team → CROSS-PLATFORM
      (React Native/Xamarin/Flutter) + NATIVE MODULES for the hardware parts
Distribution / update model?
+-- instant updates, widest reach → mobile web (responsive)
+-- store presence + hardware access, one code base → hybrid
Connectivity poor? → STORE LOCALLY, SYNC LATER
      +-- notes/lists → availability, conflict rules (LWW/server-wins/merge)
      +-- payments → consistency (refuse offline)
```

## J4. Economics / evaluation choices

```
Which method?
+-- Evaluate an existing/planned architecture's quality risks → ATAM
      (utility tree → analyse top scenarios → risks, sensitivity, tradeoffs)
+-- Choose among architectural strategies by value → CBAM
      (scenarios → 4 response levels → votes → utilities → B=Σ(b×W) → VFC=B/C
       → select within budget → confirm with intuition)
Choosing one big strategy under scarcity → frame BOTH options as
opportunity costs (best forgone alternative), then apply CBAM-style VFC.
```
---

# PART K — Worked Mini-Cases with Examiner Commentary (practise writing these)

> Each mini-case shows: the scenario trigger, the model answer skeleton, and the *examiner lens* — what earns and loses marks. Rehearse until you can reproduce the skeleton unaided.

## K1. Pipes & Filters — capacity arithmetic and back-pressure

**Trigger:** "Filter A produces 100 records/s; filter B consumes 40/s; pipe capacity 50. What happens, and what do you do?"

**Answer.** The pipe fills in 50/(100−40) ≈ 0.83 s; thereafter filter A **blocks on when-full** (e.g. 2 s retry) and its effective rate drops to B's 40/s — bounded memory, throughput = min of stages. Remedies: (1) **replicate B** behind a partitioner keyed on the record key, so 3 replicas consume ~120/s > 100/s; (2) increase B's resources; (3) batch records to cut per-item overhead. The lasting rule: **throughput of a pipeline = throughput of its slowest stage**; end-to-end latency ≈ sum of stage latencies for the first item, but steady-state latency ≈ slowest stage if filters run concurrently.

**Examiner lens:** naming *when-full block-and-retry* and *bounded queue* earns the connector-protocol mark; computing the fill time shows you understand capacity. Losing marks: saying "the pipe speeds B up" (a pipe conveys, never processes) or ignoring the blocking behaviour.

## K2. MVC — tracing a change to prove isolation

**Trigger:** "Show that adding a bulk-discount rule does not touch the UI."

**Answer.** Locate the rule's home: pricing logic in the **Model**. Sequence: (1) developer edits model's pricing service (add bulk tier); (2) controllers unchanged — they call the same `getPrice()` interface; (3) when prices are next computed the model **notifies** registered views via the change-propagation registry; (4) each view's `update()` re-fetches the affected values and redraws; (5) no view/controller source changes → no UI regression testing for the rule. The proof is the *interface invariance*: the model's public interface did not change (only its internal rule), so dependents are untouched. If the interface itself must grow (new query), then views/controllers *may* change — say so; honesty earns marks.

**Examiner lens:** the trace (controller → model → notify → update → redraw) is the money diagram; the liability sentence ("views coupled to the model's interface — interface changes propagate") shows both-side analysis.

## K3. Broker — the five hand-offs and where latency lives

**Trigger:** "Why is my brokered call 8× slower than a local call?"

**Answer.** Count the path: client → client-side proxy (marshal) → broker (locate, maybe activate) → server-side proxy (unmarshal) → server, and back — **at least five hand-offs each way**, each adding serialisation, a context switch and a network hop; plus broker registry lookup. Mitigations in order: direct-communication variant (broker locates once, proxies talk directly), batch requests, cache the registry result, replicate the broker for throughput (copies also fix its SPOF role). Never "fix" by letting clients hold addresses — that reintroduces the coupling the pattern exists to remove.

**Examiner lens:** drawing the path with operation names (pack_data/send_request → find_server → forward_request → unpack_data/call_service) is the high-mark version; the closing sentence about *why* the cost is accepted (location transparency, dynamic re-binding) shows purpose, not complaint.

## K4. Privilege separation — the state machine on paper

**Trigger:** "List what the monitor accepts in each phase."

**Answer.** Pre-authentication phase, valid requests only: (1) key-exchange group selection (slave cannot read the moduli file), (2) signature over the exchange (host private key never leaves the monitor), (3) user validation (password database unreadable by slave), (4) password/public-key authentication result. Post-authentication phase: (5) key renewal signature, (6) pseudo-terminal creation (capability: file descriptor passed back), plus rare privileged file operations. Anything else → **terminate the session**. The identity transition is not a request the slave "makes happen" — the old slave exports state and exits; the monitor spawns the new slave under the user's UID.

**Examiner lens:** the phrase *state machine of allowed requests* plus *terminate on invalid* is the core; candidates who say "the monitor elevates the slave's privileges" lose the mark — privilege never flows up; a new process starts at the right level.

## K5. CAP — writing the partition paragraph

**Trigger:** "Your replicated inventory service splits into two halves during a data-centre link failure. What does each variant do?"

**Answer.** AP (Cassandra-style): both halves keep serving; each may sell against a stale copy; when the link heals, replicas reconcile — over-sells possible, resolved by compensating actions (cancel + refund). CP (MongoDB-style): the minority partition's nodes refuse/unavailable for writes until quorum is reachable — consistent stock, but users in that region see errors. The impossible CA option would need continuous contact — the link failure itself denies it. Answer the business question: which is worse for *this* system — briefly wrong counts or briefly unavailable checkout? For ticketing, usually CP for the final seat commit, AP for browsing/search.

**Examiner lens:** one concrete two-node example (X=5, write X=7 elsewhere, read returns 5) is the fastest full-credit path; quoting product examples per side adds the last mark.

## K6. HDFS / MapReduce — the weighted average in 90 seconds

**Trigger:** "Average session length per page across 3 servers holding partial logs."

**Answer.** Map emits `<page, (avg, count)>` per server; reduce computes Σ(avg×count)/Σ(count). Guard sentence: "averages are not associative — the count must travel with the average; a plain mean of server averages weights a 2-visit server equally with a 2000-visit server." Verification habit: recompute one page from raw totals to show equality. If asked for the *why it scales* sentence: maps are stateless, run on local data, only the compact pairs cross the network (data locality), failures re-run tasks.

**Examiner lens:** full marks need the three phases *by name* (map; shuffle and sort; reduce), the pair example, and the count-carrying insight — the most commonly dropped mark in the 2025-26 paper's Q2(c).

## K7. Mobile decision — the six-row table under time pressure

**Trigger:** any "native vs cross-platform" question.

**Answer.** Write the assumption line, then the six rows (performance / security / UX / dev efficiency / reuse / platform-specific optimisation) **as a table**, each cell one clause; then the hardware paragraph (GPS/camera feasible via shell plug-ins; continuous processing → native); then the decision + spike method. Total ≈ 300 words for 5 marks. The table format halves your writing time and doubles marking legibility.

**Examiner lens:** questions name the six attributes — *answer under those headings, in that order*. A free essay that covers all six but unordered wastes marker goodwill and your minutes.

## K8. CBAM — the full 9-step run on a fresh scenario (template)

**Trigger:** "Choose between two strategies given utilities and votes."

**Answer template (fill numbers, show all four calculation lines):**
1. Scenarios kept (top third) — name them. 2. Four response levels each. 3. Votes → weights (top = 1.0). 4. Utility per level. 5. Strategy → expected response per affected scenario. 6. Interpolate: `y = ya + (yb−ya)(x−xa)/(xb−xa)` — **state the two bracketing points you used**. 7. `b = U_exp − U_cur` per scenario (negatives allowed); `B = Σ b×W`. 8. `VFC = B/C`; rank; take top-ranked until budget exhausted; ties → stakeholder judgement. 9. Intuition check + iterate condition.

**Examiner lens:** marks sit in (i) bracketing points stated, (ii) negative gains included, (iii) the budget stop rule, (iv) one sentence of step 9. A bare number with no formula line forfeits the calculation mark.

## K9. ATAM in 200 words (when the scenario asks "how would you evaluate?")

**Answer.** Assemble team + decision makers + stakeholders. Present ATAM; present business drivers; architect presents architecture (views, patterns, tactics). Identify approaches; **build the utility tree** (attributes → refinements → scenarios, ranked H/M/L × H/M/L); analyse the highest-ranked scenarios against the architecture: probe which decisions serve each scenario, tagging findings — risks, nonrisks, sensitivity points, tradeoff points. Brainstorm scenarios with the wider stakeholders, prioritise, analyse again against the newly found ones. Present results: risks → mitigation plan; risk themes; decision-to-requirement mappings. Close with a lightweight variant note: same skeleton, fewer scenarios/steps, when budget forbids the full 4-phase method.

## K10. The "compare X and Y" machine (any pair, any marks)

**Answer.** Decide the **criteria first** (5–7 rows: element types, relation, constraints, best quality, key liability, typical use), then fill both columns row by row, then one closing sentence: "the two complement: X organises __ while Y organises __, and systems commonly use both (e.g., layers inside tiers, MVC views fed by pub-sub events)." The closing sentence is the difference between a list and an analysis — always write it.

---

# PART L — Common Confusions Master Table (the traps examiners set)

| Trap pair | The distinction in one line |
|---|---|
| Architecture vs detailed design | Architecture = structures + properties; detailed design = inside one element |
| Structure vs view | Structure exists in the system; view is its documented representation |
| Module vs component | Module = implementation unit (written); component = runtime entity (runs) |
| Uses vs calls | Uses = static correctness dependency between modules; call = runtime interaction |
| Layer vs tier | Layer = code grouping (module structure); tier = deployment grouping (allocation) |
| Pattern vs tactic | Pattern = molecule (package of decisions, class of architectures); tactic = atom (single decision, one quality) |
| Pattern vs library | Pattern = scheme you complete; library = finished code reused as-is |
| Pipe vs filter | Pipe conveys (never alters); filter transforms |
| Pipe-and-filter vs blackboard | Fixed order, data between adjacent steps vs data-directed opportunistic order on a shared board |
| Broker vs ESB | Broker locates/forwards for registered servers; ESB routes/transforms/orchestrates between published services |
| SOA vs microservices | Coarse services + ESB + shared data vs fine-grained, DB-per-service, lightweight pipes |
| MVC controller vs PAC control | Within-triad mediation vs hierarchical agent-to-agent-only mediation |
| MVC model vs shared-data store | Model = logic + data + change propagation; store = passive persistence (no notifications by pattern definition) |
| Publish-subscribe vs shared-data | Events via distributor vs persistent data via store |
| Client-server vs P2P | Asymmetric roles, possible SPOF vs symmetric peers, no critical peer |
| Availability (QA) vs CAP availability | Uptime proportion vs "survivors keep operating despite partitions" |
| Consistency (CAP) vs ACID consistency | Replicated-copy agreement vs transaction atomicity guarantees |
| Hadoop vs HDFS vs MapReduce | Framework vs its storage component vs its computation component |
| Map vs reduce | Map = extract/transform → <key,value> per record, stateless; reduce = combine all values of one key |
| Combiner vs reducer | Optional local pre-aggregation on the mapper vs global per-key aggregation |
| NoSQL "schema-free" vs "schemaless" | Records self-describing; queries still assume structure — the app must enforce |
| Sharding vs replication | Splitting one dataset across nodes vs copying the same data across nodes |
| Eventual consistency vs no consistency | Copies agree after delay under no partitions vs unbounded divergence |
| Multi-instance single-tenant vs single-instance multi-tenant | Isolation+cost vs economy+shared-risk |
| Auto-scaling vs elasticity | Mechanism (create/delete instances on load) vs the platform property enabling it |
| Serverless vs containers | Function-per-event, scale-to-zero, provider-managed vs packaged processes you orchestrate |
| Authentication vs authorisation | Who you are (OpenID) vs what you may do (OAuth scopes) |
| Least privilege vs privilege separation vs sandbox | Principle vs in-app monitor/slave split vs external OS confinement |
| Risk vs tradeoff point | Unproven problematic decision vs one decision affecting 2+ attributes oppositely |
| Sensitivity vs tradeoff point | Marked effect on one attribute vs on several |
| Utility vs response measure | Stakeholder valuation (0–100) vs physical measure (s, %, users) — interpolation goes response→utility only |
| Benefit vs VFC | Weighted utility sum vs benefit ÷ cost (rank by VFC, not benefit) |
| Opportunity cost vs sunk cost | Best forgone alternative vs already-spent unrecoverable money |
| MTTF/MTBF/MTTR | Mean time to failure / between failures / to repair — A uses MTTF/(MTTF+MTTR) |
| Open book vs closed book (this exam!) | You must now memorise — hence these notes' lists, tables and mnemonics |
---

# PART M — Formula Sheet & Calculation Drills (all arithmetic you can be asked)

## M1. Every formula in the syllabus

| # | Formula | Use |
|---|---|---|
| F1 | `A = MTTF / (MTTF + MTTR)` | Availability of one element |
| F2 | `A_series = A1 × A2 × … × An` | Elements in series (all must work) |
| F3 | `A_pair = 1 − (1−A)²` | Active redundant pair (either suffices) |
| F4 | `Downtime/year = (1 − A) × 8,760 h` | Convert availability to hours down |
| F5 | `Downtime/month(30d) = (1 − A) × 720 h` | SLA arithmetic (99.95% → 0.36 h) |
| F6 | `VFC = B / C` | CBAM strategy ranking |
| F7 | `B = Σ_j ( U_expected − U_current ) × W_j` | CBAM total benefit (negative gains included) |
| F8 | `y = y_a + (y_b − y_a) × (x − x_a) / (x_b − x_a)` | Utility interpolation (response→utility) |
| F9 | `Weight(rank) = 1.0 for top; others relative to top votes` | CBAM weighting |
| F10 | `Reduce_avg = Σ(avg_i × count_i) / Σ(count_i)` | MapReduce weighted average |
| F11 | `Scale ratio: admins_small ≈ servers/150; admins_large ≈ servers/1000` | Cloud economies of scale |
| F12 | `Blocks stored = files × ceil(file/block_size) × replication` | HDFS capacity/metadata intuition |

## M2. Ten drill calculations (do them with the cover sheet; solutions below)

**D1.** Three tiers: web 99.9%, app 99.8%, DB 99.5%. System availability and yearly downtime?
**D2.** Same system; DB duplicated as an active pair. New availability and downtime?
**D3.** Same single-DB system, but MTTR cut from 5 h to 0.5 h (MTTF 995 h). Availability now?
**D4.** A vendor promises 99.99%. Downtime per year and per 30-day month?
**D5.** Strategy X: utility gains (30, 10, −10) on scenarios weighted (1, 2/3, 1/3); cost 100. Compute B and VFC.
**D6.** Strategy Y (same scenarios): gains (10, 40, 0); cost 200. B, VFC, and which strategy ranks first?
**D7.** Interpolate utility of a 0.7 s response between (0.5 s, 80) and (1.5 s, 50).
**D8.** MapReduce: 6 log lines u1,u2,u1,u3,u2,u1 split over 2 mappers. Give each mapper's output, the shuffle, the reduce results, and the check.
**D9.** MapReduce weighted average: /home records (20,40,60) on server 1; (30,50,70) on server 2; (30,30) on server 3. Compute each map output and the reduce result; show the naive mean is wrong.
**D10.** Cloud admin arithmetic: 10,000-server small DC vs 100,000-server large DC — administrators per 10k servers and the ratio.

### Solutions (show formula → substitution → result)

- **D1.** F2: 0.999 × 0.998 = 0.997002; × 0.995 = **0.992017 (99.20%)**. F4: 0.007983 × 8,760 ≈ **69.9 h/year**.
- **D2.** F3: DB pair = 1 − 0.005² = 0.999975. System = 0.997002 × 0.999975 = 0.996977 (99.70%); downtime ≈ 0.003023 × 8,760 ≈ **26.5 h/year**.
- **D3.** F1: 995/(995+0.5) = 0.999498. System = 0.997002 × 0.999498 = 0.996501 (99.65%); ≈ **30.6 h/year**. Note: duplication (26.5 h) beats faster repair (30.6 h) here; both leave ≈26 h from the web+app tiers — further gains need redundancy there.
- **D4.** 0.0001 × 8,760 = **0.876 h/year**; 0.0001 × 720 = **0.072 h ≈ 4.3 min/month**.
- **D5.** F7: B = 30×1 + 10×(2/3) + (−10)×(1/3) = 30 + 6.67 − 3.33 = **33.33**. F6: VFC = 33.33/100 = **0.33**.
- **D6.** B = 10×1 + 40×(2/3) + 0 = **36.67**; VFC = 36.67/200 = **0.18**. **X ranks first** (0.33 > 0.18) despite Y's larger benefit — 10% more benefit for twice the cost.
- **D7.** F8: y = 80 + (50−80)×(0.7−0.5)/(1.5−0.5) = 80 − 30×0.2 = **74**.
- **D8.** Mapper i: ⟨u1,1⟩,⟨u2,1⟩,⟨u1,1⟩; mapper j: ⟨u3,1⟩,⟨u2,1⟩,⟨u1,1⟩. Shuffle groups u1:[1,1,1] → **⟨u1,3⟩**; u2:[1,1] → **⟨u2,2⟩**; u3:[1] → **⟨u3,1⟩**. Check 3+2+1 = 6 = input lines.
- **D9.** Maps: S1: /home (40,3), /products (120,2), /checkout (90,1); S2: /home (50,3), /products (60,3); S3: /home (30,2), /checkout (60,3). Reduce /home = (40×3+50×3+30×2)/8 = 330/8 = **41.25 s**; /products = (120×2+60×3)/5 = **84 s**; /checkout = (90×1+60×3)/4 = **67.5 s**. Naive mean of averages = (40+50+30)/3 = 40 s ≠ 41.25 s — wrong because it ignores visit counts.
- **D10.** Small: 10,000/150 ≈ 67 admins. Large: 100,000/1,000 = 100 → per 10k servers = 10. Ratio ≈ **67:10 ≈ 6.7× fewer per 10k servers**.

## M3. Three sentences that must appear in any calculation answer

1. "Formula: …" (name it). 2. "Substitution: …" (numbers in). 3. "Result and interpretation: … X ≈ …, meaning …" — interpretation earns the final mark, not the arithmetic.

---

# PART N — Glossary (compact A–Z, closed-book recall)

**API Gateway** — single entry point fronting services: routing, auth, rate limiting, aggregation, versioning.
**ASR (Architecturally Significant Requirement)** — requirement with a profound effect on the architecture; usually a quality-attribute scenario.
**ATAM** — Architecture Tradeoff Analysis Method: scenario-driven evaluation producing risks/nonrisks/sensitivity/tradeoff points.
**Availability** — proportion of time the system is operational; A = MTTF/(MTTF+MTTR).
**Blackboard** — pattern for problems without deterministic strategies: shared board + independent knowledge sources + opportunistic control.
**Broker** — intermediary that locates servers and forwards requests/results between clients and servers (CORBA-style).
**CAP theorem** — consistency, availability, partition tolerance: at most two.
**CBAM** — Cost Benefit Analysis Method: nine steps ranking architectural strategies by value for cost.
**Circuit breaker** — fails fast to a fallback after repeated downstream failures; probes to restore.
**Combiner** — optional local pre-aggregation of a mapper's output before the shuffle.
**Content provider (Android)** — component exposing stored data (SQLite/files) as an access API.
**Container** — OS-level isolation sharing the host kernel; image packages app + dependencies.
**CBAM utility** — stakeholder benefit from quality-attribute responses, anchored worst=0/best=100.
**Direct Communication Broker** — variant where the broker only locates; proxies then exchange directly.
**DMZ** — perimeter sub-network between firewalls hosting public-facing servers.
**Distributed cache** — in-memory cluster (Redis/Memcached) reducing DB load and latency; hot-key and staleness risks.
**Document database** — NoSQL model storing self-describing JSON/BSON documents (MongoDB, CouchDB).
**Elasticity** — rapid provisioning/deprovisioning of resources (cloud characteristic enabling auto-scaling).
**Eventual consistency** — guarantee that replicas agree after some delay (AP systems).
**Filter** — pipes-and-filters component that incrementally transforms its input stream.
**Graph database** — NoSQL model of nodes + directed edges with properties (Neo4j).
**HDFS** — Hadoop Distributed File System: NameNode metadata in memory + replicated blocks on DataNodes + heartbeats.
**Hypervisor** — software creating/scheduling/isolating VMs; host page tables isolate VM memory.
**Interoperability** — ability of two systems to exchange information and use it (SOA's raison d'être).
**Intents (Android)** — messages wiring components; directed (one target) vs broadcast (all registered receivers).
**IaaS/PaaS/SaaS** — infrastructure / platform / software as a service; consumer controls progressively less upward.
**Key-value store** — NoSQL model of opaque values addressed by hashed keys (Redis).
**LDAP** — hierarchical directory protocol used for centralised authentication/authorisation lookups.
**Lambda architecture** — batch path recomputes accurate results; stream path serves fast approximations.
**Layer** — module-structure unit: cohesive services behind an interface, unidirectional allowed-to-use.
**MapReduce** — pattern/framework: stateless map emits <key,value>; shuffle groups by key; reduce aggregates.
**MVC** — model (core data+logic) / view (display) / controller (input→requests), joined by change propagation.
**MTTF / MTTR** — mean time to failure / to repair.
**Multi-tenancy** — one application instance serving multiple consumers (cost vs isolation trade).
**NameNode / DataNode** — HDFS metadata master / block-storage workers.
**OAuth** — authorisation framework: delegated access via scoped tokens without sharing credentials.
**OpenID** — authentication delegation to a trusted identity provider.
**Opportunity cost** — value of the best forgone alternative.
**PAC** — pattern of hierarchical agents, each Presentation+Abstraction+Control; controls mediate between agents.
**Peer-to-peer** — symmetric distributed pattern; no peer critical to system health.
**Pipe** — connector conveying an ordered stream between filters, never altering data.
**Publish-subscribe** — components announce events; a distributor delivers to registered subscribers.
**PWA** — progressive web app: web technology delivering app-like experiences in the browser.
**Registry (SOA)** — providers register services; consumers discover them at runtime.
**Risk / nonrisk (ATAM)** — potentially problematic unproven decision / decision proven safe by analysis.
**RPC** — remote procedure call: network invocation of a server procedure via stubs.
**Sensitivity point** — decision markedly affecting one quality attribute response.
**Serverless/FaaS** — provider-managed functions, event-triggered, scale-to-zero, per-invocation billing.
**Service mesh** — sidecar-proxy layer providing mTLS, retries, traffic policy, observability for east-west traffic.
**Sharding** — partitioning one dataset across nodes (shard-key choice is the design risk).
**Shared-data pattern** — accessors exchange persistent data only through a store.
**Sidecar proxy** — helper process beside each service instance handling mesh concerns.
**SOAP / REST** — synchronous web-service protocol over HTTP / HTTP-verb request-reply style.
**Spike (agile)** — short time-boxed experiment to retire a technical risk before committing.
**Tier** — allocation-structure unit: logical grouping of components possibly mapped to platforms.
**TLS** — transport-layer encryption with certificate-based server (and optional client) authentication.
**Tradeoff point** — decision affecting two or more attributes oppositely.
**Trust boundary** — line across which trust in data/sender changes; attacks surface at crossings.
**Utility tree** — hierarchy utility→attributes→refinements→ranked scenarios; the bridge between requirements, ATAM, CBAM.
**VFC (value for cost)** — benefit ÷ cost; ranks strategies, not an absolute financial return.
**WebAssembly** — portable sandboxed binary format running at near-native speed in browsers/runtimes.
**Webhook/WebSocket** — HTTP callback on events / persistent full-duplex browser connection for push.
---

# PART O — EC2-Overlap Topics at EC3 Depth (sessions 1–8: needed because EC-3 is comprehensive)

> These are the topics the mid-sem covered. EC-3 can still ask them, and scenario questions constantly borrow their vocabulary. This part gives each one enough depth to answer a 5–6 mark sub-part without the mid-sem notes.

## O1. Availability — general scenario, tactics in action, and the arithmetic

**Six-part scenario (fill every row when asked to "construct a scenario"):**

| Part | Value |
|---|---|
| Source | Internal/external hardware, software, or a fault generator |
| Stimulus | Fault: omission, crash, timing, response fault |
| Artifact | System or a specific component |
| Environment | Normal operation, degraded mode, or overloaded |
| Response | Detect, log, notify, take corrective action, continue in degraded mode, repair |
| Response measure | Probability of availability (99.9x%), downtime window, time to detect/repair |

**Tactic-to-scenario mapping (quote two or three):** a crash fault detected by **heartbeat/monitoring** → recovered by **active redundancy** (hot spare takes over, availability preserved during repair) or **passive redundancy** (warm spare, less cost, slower switch) → repaired by **rollback/retry**; **transactions** prevent half-completed state; **graceful degradation** keeps the core selling while optional features drop. **Side effects:** heartbeats cost bandwidth (performance), replication costs consistency effort, degradation changes user-visible behaviour (specify the degraded UX).

**Layered calculation drill (memorise D1–D4 in Part M):** series tiers multiply unavailabilities; the **database or weakest tier dominates**; duplicating it as an active pair removes most downtime; faster repair (lower MTTR) helps but less than removing the SPOF; after fixing the weakest tier, the *next* weakest tier sets the floor — availability work is always "find the dominant term" arithmetic.

## O2. Performance — the two resource laws

Performance = the response to events: **latency** (time per event), **throughput** (events per time), **capacity** (events holdable). Two levers only: **manage resource demand** (fewer/lighter/slower-arriving events: reduce computational overhead, manage event rate, control sampling frequency, limit event response, prioritise events, bound queues/execution times) and **manage resources** (more of them: increase resources, introduce concurrency, maintain multiple copies, scheduling policy FIFO/EDF). Scenario measure examples: "process N transactions per second", "respond within 0.5 s at 99th percentile", "handle 2,000 concurrent users".
**Exam-shaped insight:** auto-scaling supplies "more resources" **eventually, not instantly** (D5/D6) — so predictable spikes need *proactive* scaling signals (queue depth, request rate), not reactive CPU alarms.

## O3. Security — attributes, the scenario, and the tactic ladder

Security = resistance to unauthorised use while still serving legitimate requests; sub-properties: **confidentiality, integrity, availability, authentication, non-repudiation, authorisation, accountability/audit**.
**Scenario:** source = external attacker/insider; stimulus = unauthorised attempt to display/modify/kill/DOS/change access rights; artifact = system services/data; environment = online/under attack/offline; response = authenticate, block, report to authority, recover; measure = probability of attack success, % attacks detected, time to detect, data/services compromised.
**Tactic ladder (resist → detect → react → recover):** identify/authenticate/authorise actors → limit access, limit exposure (fewer ports, shorter windows, less data), separate entities, validate inputs, encrypt data → detect intrusion, detect service denial, verify message integrity → revoke access, lock → audit trail + restore. Patterns from Part C bundle these: privilege separation (separate entities + limit access), Secure Logger (audit + encrypt + verify integrity), Clear Sensitive Information (limit exposure over time).

## O4. Modifiability — cost of change is the whole point

**Scenario:** stimulus = a change request (function, platform, environment, quality); artifact = code/module/system; response = locate, modify, test, deploy; measure = **number of modules affected, person-days, number of tests affected**.
**Which module structure answers:** *decomposition* (where the change lands), *uses* (what breaks), *generalisation* (parametrise instead of copy). Tactics: increase semantic coherence (one purpose per module), anticipate change (isolate predicted volatility), generalise, encapsulate, use an intermediary, restrict dependencies, **defer binding** (config time: parameters; deployment time: containers/DI; run time: plugins/dynamic binding).
**Cost-of-change reasoning (works in any justify answer):** a change is cheap when (1) it lands in one module (coherence), (2) its blast radius is small (restricted dependencies), (3) it needs no redeployment of unaffected parts (encapsulation + intermediaries). MVC, layers, microservices are all packages of exactly these tactics.

## O5. Interoperability and testability (short but askable)

**Interoperability** — two systems exchange information and **use** it. Location of concern: the **connector**. Tactics: *locate the service* (registry/discovery, broker), *manage the interface* (publish a stable, versioned contract; orchestrate multi-service flows). Scenario measure: "% of requests interoperating successfully", "one protocol or data-format change touches ≤ N components". SOA/microservices are the pattern answer; standards (SOAP/REST/JSON) are the connector answer.
**Testability** — ease of demonstrating faults. Responds to: *control* (inputs controllable, isolated components, sandbox) and *observe* (outputs observable, monitors, built-in diagnostic interfaces). Tactics: input/output capture, **specialised interfaces** for testing, sandbox, abstract data sources (inject fakes), executable assertions, record/replay. Scenario: "% statements executed per test", "time to write a test", "one defect found by N tests". Architecture connection: a pattern with clean interfaces (broker, MVC) is testable per component; a monolith with shared state is not — say this in testability answers.

## O6. Documentation — the views in depth (beyond 4+1)

Document each chosen view with: **primary presentation** (the diagram), **element catalogue** (elements + their properties), **context diagram** (what's outside), **variability guide** (what can vary), **rationale** (why this design). Combine views when elements correspond closely (e.g., deployment + implementation for ops teams). **Quality-attribute views** cut across: a *security view* (trust boundaries, auth flows), a *communication view* (protocols, encryption points), a *reliability view* (replication, failover paths) — write one when a single attribute needs its own stakeholder audience. Documentation rule: **write for the stakeholders' questions** — a view nobody reads is cost without benefit (the "sweet spot" of effort).

## O7. Conformance, drift and reconstruction (Session 8 — increasingly asked in EC3-style "how do you keep it true" sub-parts)

**Drift:** each change is made by someone solving an immediate problem, so code gradually departs from the documented architecture (ad-hoc logging, a UI class calling the DB, a new path bypassing the intermediary). Drift is not malice — it is the absence of a mechanism.

**Four conformance techniques (name, mechanism, example, limit):**

| Technique | Acts on | Example | Limit |
|---|---|---|---|
| Architecturally evident coding style | Readability of intent | Naming/annotation/package convention marking layer or role (a class marked "presentation" calling the DB stands out in review) | Relies on convention being followed |
| Frameworks | Code structure | **Spring MVC** forces every screen through View/Controller/Model; Hibernate for persistence; JMS for pub-sub; Log4J removes ad-hoc logging; AUTOSAR in automotive | Binds the system to the framework; code that doesn't fit is awkward |
| Code templates | Shape of one kind of component | Fault-tolerant process template with mandatory sections: normal processing (send state to backup), state update, switch-over — the developer cannot omit state transfer | Covers only templated patterns |
| Documentation update | The record | Mark stale parts "No longer applicable"; synchronise doc with each release | Depends on process discipline |

Plus three habits: educate new team members; code reviews; **folder structure mirrors the architecture**. Detection: reconstruction tools + violation queries (who-calls-whom rules).

**Architecture reconstruction** — recovering the architecture from artefacts that cannot lie (code, execution traces, build scripts). *Purposes:* (1) understand an undocumented system; (2) **migrate** old technology to new; (3) identify **reusable components**. *Phases (four + iterate):*
1. **Raw view extraction** — collect uninterpreted facts: classes, files used, caller-callee, global data accessed (typed relations: file includes file; file contains function; module uses interface…).
2. **Database construction** — put extracted data into a common format.
3. **View fusion** — merge static views + dynamic (trace) views + **expert grouping** into consolidated views of the architecture.
4. **Architecture analysis** — validate fused elements against rules; display for human analysis; **iterate until trustworthy**.
(Broad 3-phase reading: identify components+relations → aggregate into abstract components → analyse.) **Case (Vanish/ARMIN):** extraction produces "white noise" of raw elements; grouping rules aggregate; analysis validates. *Testing tie-in:* architecture conformance is tested by rule queries over the reconstructed view (e.g., "no presentation module imports data-layer") — reconstruction is to maintenance what the utility tree is to evaluation.

## O8. Real-time architectures (one-paragraph insurance)

Hard/soft real-time systems add **timing correctness** to the quality set: worst-case execution time, scheduling (rate-monotonic/EDF), priority inversion handling, watchdogs. Architectural habits: few layers on the hot path (every hop adds jitter), statically allocated resources where possible, ping/echo + watchdog between critical components, and a design that separates the *fast loop* (deterministic) from the *slow loop* (configuration, HMI). Connect to mobile's "few layers" and to performance tactics — the same physics.
---

# PART P — Technology Topics Expanded (Session 14 addenda: analytics, ML, blockchain, IoT)

## P1. Analytics landscape — data mining vs data analysis vs visualisation

- **Data analysis** = answering stated questions over data (reports, aggregations, dashboards); **data mining** = discovering *unknown* patterns (clustering customers, association rules like market baskets, anomaly/fraud detection, prediction). One line for exams: analysis answers questions you have; mining finds questions you didn't know to ask.
- **Visualisation** turns both into human-decidable form: dashboards (operational KPIs), charts of distributions/trends, geo-maps; architectural placement: reads from the served warehouse/marts, never from the transactional DB directly (separation of the serving and transactional paths — same bulkhead logic as X6).
- **Real-time analytics** (Session 14 topic): compute metrics on arriving events within seconds — architectures: stream processing (Flink/Spark Streaming) over a message bus, windowed aggregation, alert rules; serves trending dashboards and operational alerts. Batch analytics over HDFS serves history. The two together = lambda (F4/K6).

## P2. AI/ML architecture — the full pipeline with architectural consequences

**Pipeline stages (write as numbered steps):**
1. **Data collection** — batch sources (warehouse) + streams (events); capture feedback labels for supervised tasks.
2. **Preparation** — cleaning, deduplication, feature engineering (transforms must be *reusable code*, not notebook magic, so training/serving stay consistent).
3. **Algorithm selection** — classify/regress/cluster/recommend per the decision the business wants.
4. **Training** — compute-heavy: clusters/GPUs (cf. Hadoop/Spark infrastructure); produce a **versioned model artifact**.
5. **Evaluation** — accuracy/recall/precision or business metric (revenue per user) on held-out data; compare against the baseline.
6. **Deployment/serving** — model served via API (real-time scoring) or batch scoring jobs; A/B deployment for cautious rollout.
7. **Monitoring + retraining** — data drift degrades models; monitor input distributions and outcomes; schedule retraining; capture new labels from production.

**Architectural consequences (the marks beyond the pipeline):** (1) *training and serving have opposite shapes* — training is throughput-oriented batch, serving is latency-critical online → separate tiers, separate scaling; (2) *feature consistency* — the same feature definitions must run in both paths (train/serve skew is the classic bug); (3) *models are versioned artifacts* with lineage (data version + code version → model version) for audit; (4) *feedback loops* — user interactions become next quarter's training data, so the serving tier must log features (privacy-aware — links to Part C masking rules); (5) the platform story: cloud GPUs/managed notebooks; at petabyte scale, feature jobs run on the same Hadoop/Spark ecosystem as analytics.

## P3. Blockchain — architecture-level answer

**What it is:** a **distributed, append-only ledger** replicated across nodes; transactions grouped into blocks; each block carries a hash of its predecessor (tamper-evident chain); **consensus** (proof-of-work in public networks; voting/BFT-style in permissioned ones) replaces a central authority. **Properties:** integrity and auditability without a trusted centre; disintermediation between parties that do not trust each other. **Costs:** consensus throughput limits (public chains ~ tens of transactions/s vs card networks' tens of thousands), latency to finality, energy (proof-of-work), and immutability cutting both ways (mistakes are permanent).
**Architectural decision rule:** put data **on-chain only when multiple mutually distrusting parties must jointly trust it**; everything else stays off-chain with on-chain **hashes/anchors** (e.g., document hash + signature on-chain, document in your own store). **Course cases:** IBM–Maersk trade logistics (shipping events shared across carriers, customs, ports), finance settlement, provenance tracking. **Pattern connection:** each node = peer in a P2P pattern with a consensus connector; permissioned blockchains reintroduce a membership layer (closer to a broker-governed federation).

## P4. IoT reference architecture (name the five planes)

1. **Things/devices** — sensors/actuators; constraints: power, connectivity, compute → lightweight protocols (MQTT/CoAP), store-and-forward buffering.
2. **Gateway/edge** — protocol translation, filtering/aggregation, local rules, offline operation, device security anchor (ties to edge computing, Part H).
3. **Ingestion/transport** — messaging bus (pub-sub) absorbing device scale; per-device identity and auth (certificates, not passwords).
4. **Processing/analytics** — stream processing for immediate alerts + batch/ML for models; cold path to time-series/NoSQL stores.
5. **Application/management** — dashboards, device twins/registry, OTA updates, fleet monitoring.
**Qualities:** scale-out ingestion (millions of devices), availability under intermittent links (store-and-forward = the mobile store-locally pattern at device scale), security end-to-end (device identity → encrypted channels → least-privilege topic ACLs → audit).

## P5. Emerging trends — one-liners with an architectural hook

- **WebAssembly:** portable, sandboxed, near-native binary target; hook: performance-critical client modules and plugin isolation without OS trust.
- **Service mesh:** east-west communication policy (mTLS, retries, traffic shifting) as infrastructure; hook: it *is* the "use an intermediary" tactic productised — with the intermediary's classic costs (hop, operational complexity).
- **Edge computing:** computation near data/users for latency, bandwidth and locality; hook: tiered processing (edge decides now, cloud learns slowly) — the CAP/availability choices repeat at the edge.
- **Serverless:** event-driven, scale-to-zero functions; hook: elastic economics for spiky workloads (ties to CrowdCast/agri-spike cases) with state-externalisation as the price.
- **Micro-frontend:** decompose the UI into independently built/deployed fragments; hook: MVC/layer thinking applied to the browser; cost: duplicated tooling, consistent look-and-feel effort.

---

# PART Q — Case-Study Cards (the course's named cases, in recall form)

| Case | The architecture story in six lines |
|---|---|
| **Netflix** | Monolith → cloud microservices after a DB outage; **replication across zones** (no SPOF), **distributed cache/CDN** for catalogue and artwork, **circuit breakers + graceful degradation** (recommendations may fail, playback must not), chaos engineering validates failure paths, auto-scaling for evening peak; lesson: availability is architected, not bought |
| **Salesforce.com** | The canonical **multi-tenant SaaS**: one application + one shared schema with tenant IDs, metadata-driven customisation per tenant, single upgrade for all; lesson: tenancy degree is an economic decision (D5) with security engineering (tenant filters, row-level security) |
| **SOA at CIGNA** | Enterprise integration of insurance silos via services + ESB/registry; published interfaces let mainframe and web-era systems interoperate; lesson: SOA's interoperability benefit is real, but ESB governance and overhead are the price |
| **SOA at TripAdvisor** | Public-facing site consumes internal and external services (airline/hotel providers) through service endpoints; orchestration composes results; lesson: providers may be consumers; published contracts enable cross-organisation composition |
| **Microservices at Danske Bank** | Monolith constrained release cycles; decomposition into microservices gave independent deployability per business capability, DB-per-service, CI/CD; lesson: the driver is *organisational speed of change*, not fashion |
| **Project Smart Health** | Mobile field app with **store-locally/sync-later** (SQLite → REST → MySQL); UI depends on local store; lesson: availability ≠ connectivity (Part E5) |
| **Tailspin Surveys** | Cloud SaaS with two tenancy options (multi-instance/single-tenant vs single-instance/multi-tenant) chosen per workload value, volatility and security; lesson: tenancy may differ per tier |
| **OpenSSH** | Privilege separation: monitor + slave, state-machine interface, state-export respawn; ~2/3 code unprivileged; lesson: least privilege applied *inside* the application (Part C2) |
| **Boston Globe** | Responsive design landmark: one content set, three layouts via flexible grids/images/media queries; lesson: one site beats two (Part E4) |
| **NASA ECS** | CBAM with ten scenarios/ten strategies/100 votes; strategies ranked by benefit ÷ cost; budget spent in VFC order; lesson: economics is a method, not a spreadsheet (Part G6) |
---

# PART R — Pattern Cards (uniform one-page recall format — rehearse one card per day)

> Every card: **Definition → Elements → Relations → Constraints → Implementation → Variants → Known uses → Benefits → Liabilities → Exam sentences.** If you can reproduce a card cold, you can answer any application question on that pattern.

## R1. LAYERS

- **Definition:** divide software into groupings of modules offering cohesive services; usage is unidirectional (allowed-to-use, downward only); complete, exclusive partition.
- **Elements:** layer (a kind of module). **Relations:** allowed-to-use. **Constraints:** every unit in exactly one layer; ≥2 layers; no circular use.
- **Implementation:** define layer contents + usage rules; design interfaces between layers; decide strictness (strict = only adjacent; relaxed = any lower layer); assign modules; document exceptions.
- **Variants:** strict layering; relaxed layering; layers with Facades; shared-data layer exceptions.
- **Known uses:** OSI stack; OS kernels; three/four-tier enterprise apps (presentation/business/data).
- **Benefits:** change isolation; portability; reuse; team organisation; support for incrementality.
- **Liabilities:** up-front cost/complexity; performance penalty through hops; change may still cascade; layer count bikeshedding.
- **Exam sentences:** "The layer isolates the change: a database-vendor swap touches only the persistence layer, because the business layer depends only on its published interface." / "The cost of the isolation is hop overhead, which is why mobile architectures keep the stack shallow."

## R2. PIPES AND FILTERS

- **Definition:** stream transformations as filter components joined by pipes; filters transform incrementally; pipes convey in order without altering.
- **Elements:** filter, data source, data sink. **Connectors:** pipe (single source/target, order-preserving, annotated: capacity, end-of-data, when-full/when-empty).
- **Constraints:** pipes join output ports to input ports; neighbours agree only on data type/format.
- **Implementation (6 steps):** divide task into stages → define the pipe data format → choose pipe implementation (concurrent vs call) → build incremental single-responsibility filters → design error handling (pipes carry no errors) → set up the pipeline (main program / shell / visual builder).
- **Variants:** tee-and-join (directed graphs, feedback — needs formal analysis); piped call structure inside one program (compiler front end).
- **Known uses:** UNIX tools; CMS Pipelines; LASSP; compilers; ETL pipelines.
- **Benefits:** no intermediate files; exchange flexibility; recombination; reuse; prototyping; parallelism.
- **Liabilities:** shared state expensive; parallel gain often illusory (transfer/sync cost, slowest filter); format-conversion overhead; awkward error handling.
- **Exam sentences:** "Filters consume and deliver incrementally, so the first output leaves before the last input arrives — latency stays low and filters genuinely run in parallel." / "Scaling a slow filter means partitioning its input by key and running copies between a partitioner and a merger; the bounded pipe then applies back-pressure instead of exhausting memory."

## R3. BLACKBOARD

- **Definition:** for problems with no deterministic solution strategy: specialised subsystems (knowledge sources) assemble a possibly partial solution on a shared blackboard, scheduled opportunistically by a control component.
- **Elements:** blackboard (solution space + control data), knowledge sources (independent; execCondition/execAction), control (loop, nextSource).
- **Constraints:** sources never call each other; the state of the board drives the schedule.
- **Variants:** production system (condition-action rules, conflict resolution); repository (store without self-directed control).
- **Known uses:** HEARSAY-II, HASP/SIAP, CRYSALIS, SUS.
- **Benefits:** experiment-friendly; exchangeable specialists; partial/incremental solutions.
- **Liabilities:** no termination guarantee; testing/debugging harder; synchronisation on the board.
- **Exam sentences:** "Unlike a pipeline, the order of steps is decided at run time by the state of the shared data — that is opportunistic problem solving." / "Choose it when no deterministic strategy exists (recognition, fusion, assessment), not when a known recipe exists."

## R4. BROKER

- **Definition:** a broker mediates communication between clients and servers: locates the server, forwards requests, returns results/exceptions.
- **Elements:** client, server, broker, client-side proxy, server-side proxy, bridge (optional). **Supporting:** name service, marshalling.
- **Constraints:** clients attach only via broker (through client proxy); servers register with the broker.
- **Implementation (6):** define object model → choose interop mechanism (IDL vs binary standard) → specify broker APIs → proxies for transparency → design broker in parallel → build IDL compilers.
- **Variants:** direct communication; message passing; trader; adapter; callback.
- **Known uses:** CORBA, IBM SOM/DSOM, OLE 2.x, WWW.
- **Benefits:** location transparency; changeability/extensibility (dynamic add/move); portability; inter-broker interoperability; reusability; isolated testing.
- **Liabilities:** indirection latency; broker SPOF; more failure points; hard end-to-end debugging; attack target.
- **Exam sentences:** "Every exchange crosses five hand-offs — client, client proxy, broker, server proxy, server — which is the price of neither end knowing the other's location or language." / "A new district's service simply registers with the broker; no client changes — the pattern's dynamic-binding force in action."

## R5. MODEL-VIEW-CONTROLLER

- **Definition:** model = functional core (data + logic + change-propagation registry); views display; controllers translate input into model/view requests; views:controllers = 1:1.
- **Five interactions:** view→model state query (call); controller→model state change (call); model→views notification (event); view→controller gesture (event); controller→view selection (call).
- **Constraints (C&C form):** ≥1 of each; model does not interact directly with the controller; relation = notifies.
- **Implementation highlights:** change-propagation (attach/detach/notify + update interface); set-up order (model → views register → controllers created); pluggable controllers; dynamic views; composite views.
- **Variant:** Document-View (merged view+controller; loses controller exchangeability) — MFC, ET++.
- **Known uses:** Smalltalk VisualWorks; MFC; ET++; Spring MVC (framework enforcement).
- **Benefits:** multiple views; synchronised views; pluggable components; look-and-feel exchange; framework potential.
- **Liabilities:** complexity (registry+protocol); update storms; view-controller intimacy; model-interface coupling to all views/controllers; per-value view queries; porting rewrites views/controllers; toolkit mismatch.
- **Exam sentences:** "A business-rule change lands in the model alone: controllers keep calling the same service, the model notifies views, and no UI code is retested." / "The model never calls display code — it announces change; each view decides how to redraw."

## R6. PRESENTATION-ABSTRACTION-CONTROL

- **Definition:** hierarchical agents, each owning Presentation (its UI), Abstraction (its data/logic) and Control (mediates internally and is the only channel between agents).
- **Levels:** top (system core + global UI), intermediate (composite features), bottom (atomic elements).
- **Benefits:** multi-hierarchy UIs; full decoupling of agent abstractions; multi-user/cooperative UIs.
- **Liabilities:** many components; coordination overhead; control complexity.
- **Exam sentences:** "MVC has one triad per dataset; PAC has a tree of agents for composite UIs — cross-agent talk travels only through Controls." / "Extending the console with a new panel = adding one bottom agent under the relevant intermediate Control, touching no sibling."

## R7. SERVICE-ORIENTED ARCHITECTURE / MICROSERVICES

- **Definition:** distributed components provide/consume services through published descriptions; consumers need no implementation knowledge.
- **Elements:** provider, consumer, ESB, registry, orchestration server. **Connectors:** SOAP, REST, async messaging.
- **Weaknesses:** build complexity; service evolution outside consumer control; middleware overhead; no performance guarantees.
- **Microservices deltas:** fine-grained, DB-per-service, lightweight pipes, decentralised governance, independent deployability; failure machinery (circuit breaker, bulkhead) is part of the pattern in practice.
- **Exam sentences:** "The only dependency is the published contract — a provider may move platforms, language or even organisation without touching consumers." / "Microservices buy independent releases with eventual consistency as the currency; decompose along bounded contexts, not layers."

## R8. CLIENT-SERVER / PEER-TO-PEER / PUBLISH-SUBSCRIBE / SHARED-DATA (mini-cards)

- **Client-Server:** clients request, servers provide via request/reply; servers may be clients. Weaknesses: bottleneck, SPOF, costly role split. Exam: "centralising shared services buys control and modifiability at the price of a critical server — replicate it."
- **Peer-to-Peer:** symmetric peers, direct interaction, no critical peer; constraints on attachments/hops/knowledge; supernode stars. Weaknesses: security/consistency/backup complexity; small nets underperform. Exam: "P2P trades the centre's risks for protocol complexity — consistency and identity become protocol problems."
- **Publish-Subscribe:** announce/listen on an event distributor; consumers unknown in advance. Weaknesses: latency, ordering/delivery guarantees, scalability of the distributor. Exam: "Publishers cannot wait for every subscriber's ack — that is why guarantees weaken; durable queues and per-key ordering restore them."
- **Shared-Data:** accessors exchange persistent data only with the store; either side may initiate. Weaknesses: bottleneck, SPOF, tight coupling through the data model. Exam: "Accessors are decoupled in identity but married to the schema — a data-model change is a change to everyone."

## R9. MAP-REDUCE

- **Definition:** infrastructure + stateless map (extract/transform → <key,value>) + reduce (combine per key); parallel analysis of file-based datasets.
- **Constraints:** input as files; maps stateless, non-communicating; only <key,value> between map and reduce.
- **Weaknesses:** small-data overhead; stragglers from uneven partitions; multi-stage reduces complex.
- **Flow:** map (data-local) → shuffle/sort (distributed sort; each key to one reducer) → reduce → output.
- **Exam sentences:** "Adding nodes adds map capacity — maps never wait on each other, so scalability is linear until partitions skew." / "Averages must travel with their counts; reduce computes the weighted mean, or servers with different visit volumes are counted equally — the classic silent error."

## R10. MICROKERNEL / REFLECTION (mini-cards)

- **Microkernel:** minimal core of mechanisms + IPC/resources; internal servers, external servers, adapters, clients around it. Benefits: portability, adaptability, mechanism/policy separation. Liabilities: hop overhead; minimal-core design difficulty. Uses: OS kernels, window systems.
- **Reflection:** meta level (metaobjects = system's self-representation) + base level; metaobject protocol changes base behaviour at runtime. Benefits: runtime adaptation, transparent extension. Liabilities: complexity; invariants at risk; overhead. Exam: "Reflection changes *how* the system works; microkernel changes *what plugs in*."

## R11. MULTI-TIER

- **Definition:** logical groupings of components (tiers) for deployment; is-part-of, communicates-with, allocated-to; one component = one tier.
- **Example:** browser → web tier → EJB tier → back end (JDBC + SOAP).
- **Weakness:** up-front distribution cost/complexity.
- **Exam sentences:** "Layers organise code; tiers organise execution — the same three-layer code base may deploy into one tier or four." / "Tenancy may also differ per tier: heavy processing dedicated, standard processing shared, confidential data dedicated."
---

# PART S — General Scenario Templates per Attribute (SAIP six-part grids — fill-and-adapt in exams)

> When a question says "construct a scenario", reproduce the relevant grid with the scenario's own values. Each grid: the general scenario, then one filled course-style example.

## S1. Availability

| Part | General possibilities |
|---|---|
| Source | Internal/external hardware; internal/external software; fault generator |
| Stimulus | Fault: omission, crash, timing, response |
| Artifact | System; processor; process; component |
| Environment | Normal operation; degraded; overloaded |
| Response | Detect; log; notify; take corrective action; continue degraded; repair |
| Measure | Availability %; downtime window; time to detect; time to repair |

*Filled example:* external payment gateway (source) stops responding — omission fault (stimulus) — payment-processing component (artifact) — 08:00 booking peak (environment) — detect within 5 s, queue payments, degrade to seat-hold-only, restore standby gateway (response) — no confirmed booking lost; detection < 5 s; restoration < 30 s (measure).

## S2. Performance

| Part | General possibilities |
|---|---|
| Source | External/internal user; scheduled event; system overload |
| Stimulus | Event arrives: periodic, sporadic, stochastic |
| Artifact | System; component; network |
| Environment | Normal; overloaded; overloaded with faults |
| Response | Process; measure latency/deadline/jitter |
| Measure | Response time; throughput; capacity; deadline miss rate; jitter |

*Filled example:* 200,000 passengers (source) submit reservations at 08:00 (stimulus) — booking pipeline (artifact) — overloaded (environment) — each request processed within deadline (response) — 99th percentile ≤ 2 s; throughput ≥ 4,000 TPS; zero deadline misses on payment calls (measure).

## S3. Security

| Part | General possibilities |
|---|---|
| Source | External attacker; internal user; compromised process |
| Stimulus | Display/modify/deny/change-access data or service |
| Artifact | System services; data; specific process |
| Environment | Online; offline; under attack; disconnected |
| Response | Authenticate; block; notify authority; recover; audit |
| Measure | Attack-success probability; % detected; time to detect; % recovered |

*Filled example:* anonymous network party (source) attempts to alter a transaction record (stimulus) — ledger service (artifact) — online, under attack (environment) — reject + lock account + alert SOC + append tamper-evident audit record (response) — attacks blocked ≥ 99.9%; detection ≤ 1 s; audit chain verifies (measure).

## S4. Modifiability

| Part | General possibilities |
|---|---|
| Source | Developer; end user (via config) |
| Stimulus | Change request: function, platform, environment, quality |
| Artifact | Code; module; system; UI; deployment |
| Environment | Design time; build time; run time |
| Response | Locate; modify; compile/test; deploy; verify |
| Measure | Modules touched; person-days; tests affected; downtime to deploy |

*Filled example:* product team (source) replaces the discount algorithm (stimulus) — pricing module (artifact) — design time (environment) — change lands in one module; regression = pricing tests only (response) — ≤ 1 module, ≤ 2 person-days, zero UI tests (measure).

## S5. Usability

| Part | General possibilities |
|---|---|
| Source | End user |
| Stimulus | Wants to: learn; use efficiently; minimise errors; recover; recall |
| Artifact | System; part of system |
| Environment | Run time; configuration time |
| Response | Support task; undo/cancel; correct error; remember preferences |
| Measure | Task time; error count; satisfaction; recovery steps |

*Filled example:* first-time mobile user (source) books a cab with one thumb while walking (stimulus) — booking UI (artifact) — run time (environment) — three taps, confirm screen, undo within 10 s (response) — booking ≤ 3 taps; recovery ≤ 2 steps (measure).

## S6. Testability

| Part | General possibilities |
|---|---|
| Source | Developer; tester; automated suite |
| Stimulus | Test executed; fault indicated |
| Artifact | Component; subsystem; whole system |
| Environment | Design time; development; production diagnostics |
| Response | Control inputs; capture outputs; isolate fault; replay |
| Measure | % statements covered; tests to find a fault; time to write a test; replay capability |

*Filled example:* regression suite (source) runs fraud-rule tests nightly (stimulus) — fraud filter (artifact) — development (environment) — inject synthetic streams, capture decisions per rule (response) — 100% rule branches covered; any rule reproduced by replaying its recorded stream (measure).

## S7. Interoperability

| Part | General possibilities |
|---|---|
| Source | System A requesting; system B responding |
| Stimulus | Request at a documented interface |
| Artifact | Connectors; service interfaces |
| Environment | Run time; integration time |
| Response | Mutual exchange; translation; graceful reject |
| Measure | % successful exchanges; protocol/format mismatches; time to add a partner |

*Filled example:* new district ambulance service (source) joins the dispatch platform (stimulus) — broker + service interfaces (artifact) — run time (environment) — registers, serves dispatch calls through the standard contract (response) — zero client changes; first successful call within one day (measure).

## S8. Scalability (as an aspect of performance/modifiability)

| Part | General possibilities |
|---|---|
| Source | Load generator; growth forecast |
| Stimulus | Load multiplies (2×, 10×); data grows |
| Artifact | Service tier; data tier |
| Environment | Peak; growth period |
| Response | Add instances/nodes; redistribute; hold response |
| Measure | Response at new load; linear capacity per added node; cost per added node |

*Filled example:* exam-week traffic 10× baseline (stimulus) — read tier + cache (artifact) — predictable seasonal peak (environment) — auto-scale from 4 to 40 instances; cache absorbs reads (response) — 99th percentile stays ≤ 2 s; capacity scales near-linearly per instance (measure).
---

# PART T — Integration Strategy Cards (each strategy as a full card)

## T1. FILE TRANSFER

- **Mechanism:** system A exports data to files (fixed-width/CSV/XML/JSON) on a schedule; system B reads and imports them.
- **Strengths:** simplest to build; decoupled in time (B may be down at write time); works with any legacy that can read files; natural audit artefact (the file itself).
- **Weaknesses:** latency = schedule interval; format must be agreed and versioned; no per-record feedback (errors surface at import — reconcile whole files); duplicates data (storage + consistency of copies).
- **Design details that earn marks:** batching window; checksum + record counts for integrity; a quarantine area for rejected rows with a repair-and-replay job; schema versioning (add columns, never repurpose); delivery = SFTP/object-store with completion markers.
- **Use when:** nightly bulk exchange, legacy ERP/bank connectivity, regulatory submissions.
- **Exam sentence:** "File transfer buys simplicity and time-decoupling at the price of latency and whole-file error semantics — reject rows go to a quarantine sink and are replayed, not lost."

## T2. SHARED DATABASE (integration by data)

- **Mechanism:** two applications read/write one database (the Shared-Data pattern used as an integration strategy).
- **Strengths:** immediate consistency; no format translation; rich queries across both applications' data.
- **Weaknesses:** strongest coupling available — schema changes hit both; ownership disputes; the DB is a shared bottleneck/SPOF; one side's load spikes degrade the other.
- **Use when:** tightly related internal applications with one owning team; analytical reads via replicas.
- **Exam sentence:** "Accessors are decoupled in identity but married to the schema — when the two systems have different teams and release cycles, prefer a service contract over a shared table."

## T3. RPC (REMOTE PROCEDURE CALL)

- **Mechanism:** client invokes a procedure on a remote server via stubs/proxies (gRPC-style or SOAP); request/reply with typed contracts.
- **Strengths:** natural programming model; strong typing and tooling; immediate answers; efficient for chatty internal calls (binary protocols).
- **Weaknesses:** temporal coupling (both must be up); location/knowledge coupling (need discovery); version discipline required; a slow server consumes client threads unless timeouts/async are used.
- **Design details:** deadlines on every call; idempotency for retries; circuit breaker at the client; versioned interfaces.
- **Use when:** internal synchronous coordination needing an immediate result (validation, price lookup).
- **Exam sentence:** "RPC is the most convenient and most coupled option: the caller waits on the callee's health — every RPC boundary needs a timeout, a retry policy and a breaker."

## T4. MESSAGING (QUEUES AND TOPICS)

- **Mechanism:** producers post messages to queues (point-to-point) or topics (pub-sub) on a broker; consumers process asynchronously.
- **Strengths:** temporal decoupling; buffering against bursts; redelivery/at-least-once with acks; fan-out to unknown consumer sets; load levelling.
- **Weaknesses:** broker infrastructure to operate; ordering limited to partitions; duplicate delivery (consumers must be idempotent); message-design effort; eventual consistency in pipelines.
- **Design details:** durable subscriptions; per-key ordering (per account/device); poison-message handling (dead-letter queue + replay); message schema versioning; visibility of lag (queue depth as a scaling signal).
- **Use when:** event-driven architectures, decoupling microservices, absorbing spikes, integration across teams.
- **Exam sentence:** "The queue converts 'the consumer is down' from an error into a delay — the price is duplicates and reordering, which consumers must resolve by idempotency and per-key ordering."

## T5. WEBSOCKETS

- **Mechanism:** after an HTTP upgrade handshake, a persistent full-duplex TCP channel between browser and server; either side may push.
- **Strengths:** real-time bidirectional push; low per-message overhead (no HTTP request per update); works through standard browser APIs.
- **Weaknesses:** server-side connection state (horizontal scaling needs sticky sessions or a pub/sub backplane); proxies/firewalls must permit upgrade; reconnection + re-authentication logic; no built-in message semantics.
- **Design details:** heartbeat/ping to detect half-open connections; resume with last-received-id replay; backplane (pub-sub cluster) behind the gateway; fall back to polling where upgrades fail.
- **Use when:** live dashboards, chat, collaborative editing, driver/vehicle maps.
- **Exam sentence:** "WebSockets remove the per-update request overhead but introduce connection state — scale them by pairing a stateless auth/routing tier with a pub-sub backplane."

## T6. API GATEWAY

- **Mechanism:** a single entry tier in front of internal services: routing, authentication/authorisation, rate limiting, response aggregation, protocol translation, versioning.
- **Strengths:** hides internal topology (services can move/split freely); one place for cross-cutting policy; simplifies clients (one endpoint, aggregated payloads — crucial for mobile).
- **Weaknesses:** must never contain business logic (it becomes a monolith-in-front); it is a bottleneck/SPOF to replicate; aggregation latency.
- **Design details:** stateless, horizontally scaled; per-route rate limits; circuit breakers toward services; cache policy per route; versioned routes.
- **Use when:** exposing microservices to browsers/mobile/partners; mobile back ends (payload shaping saves round trips).
- **Exam sentence:** "The gateway is policy, not logic: routing, auth and shaping live there so that every service behind it can change without breaking a single client."

## T7. CHOOSING — THE EXAM-READY RULE

Couple by need: **immediate answer inside your trust boundary → RPC/REST; decouple in time, absorb bursts, fan out → messaging; bulk/period/legacy → file transfer; shared truth for one team → shared database; browser push → WebSockets; many clients over many services → API gateway. Real systems compose them (gateway front, REST for reads, events for state changes, files for legacy) — name the composition and why each boundary earned its choice.**

---

# PART U — Microservice Machinery (the named mechanisms behind "microservices" answers)

- **API gateway** — see T6; also the natural point for token validation and tenant routing.
- **Service discovery** — services register (or are found via DNS/k8s services) so callers resolve current instances; replaces the broker's registry for lightweight stacks.
- **Circuit breaker** — counts failures per dependency; open (fail fast) → half-open (probe) → closed; prevents a slow dependency from exhausting caller resources.
- **Bulkhead** — separate pools (threads, connections, memory) per dependency so one flooding dependency cannot starve the rest.
- **Retry + idempotency** — retries are safe only when the operation is idempotent; add jittered exponential backoff; deduplicate by request ID.
- **Saga / compensating actions** — replace cross-service distributed transactions: each step publishes an event; failures trigger compensating steps (refund after failed fulfilment). Two styles: choreography (services react to events) vs orchestration (a coordinator directs steps).
- **Event sourcing / CQRS (awareness level)** — store state as events; separate write models from read models; powerful for audit (the log *is* the state) but adds complexity — mention as an option for audit-heavy domains, not a default.
- **Anti-corruption layer** — a translation boundary that keeps a legacy model from leaking into the new domain model.
- **Strangler pattern** — incrementally route capabilities from the monolith to new services behind a façade until the monolith strangles; the migration method of C4.
- **Sidecar/service mesh** — infrastructure-level mTLS, retries, traffic policy per instance (Part H); adopt when per-service code for these concerns diverges.
- **Contract tests + consumer-driven contracts** — each consumer publishes the contract it depends on; CI fails when a provider breaks it — the modifiability guarantee behind independent deployment.
- **Observability** — correlation IDs propagated across calls; distributed tracing; health endpoints (ping/echo at service scale); SLO dashboards. Exam line: "In a distributed system, debugging is an architectural feature — build correlation and tracing in from day one."

---

# PART V — Calculation Practice Set B (availability + economics, with solutions)

**V1.** A service runs on two independent instances behind a balancer; each has MTTF 2,000 h, MTTR 2 h. Availability of one instance and of the pair? (One suffices to serve.)
**V2.** Adding a shared database (99.9%) in series with the pair — system availability and yearly downtime?
**V3.** The team can either raise each instance's MTTF to 4,000 h (same MTTR) or cut MTTR to 0.5 h. Which yields higher pair availability?
**V4.** CBAM: three scenarios with weights (1.0, 0.5, 0.25). Strategy P gains utilities (20, 30, −8); cost 60. Strategy R gains (35, 5, 0); cost 90. Compute B and VFC for both; which do you select first under a budget of 100?
**V5.** Interpolate utility of a 99.92% availability response between current (99.5%, utility 40) and desired (99.9%, utility 85).

### Solutions

- **V1.** A₁ = 2000/(2000+2) = 0.999. Pair = 1 − (0.001)² = **0.999999 (99.9999%)**.
- **V2.** System = 0.999999 × 0.999 = 0.998999 (≈ **99.90%**); downtime = 0.001001 × 8,760 ≈ **8.8 h/year** — the shared DB now dominates: fix the tier with the largest unavailability term first.
- **V3.** Raise MTTF: A₁ = 4000/4002 = 0.9995005…; pair = 1 − (0.0004995)² ≈ 0.99999975. Cut MTTR: A₁ = 2000/2000.5 = 0.9997501; pair = 1 − (0.0002499)² ≈ 0.99999994. **Cutting MTTR wins** — halving repair time beats doubling time-to-failure here (both beat V1's pair).
- **V4.** B_P = 20×1 + 30×0.5 + (−8)×0.25 = 20 + 15 − 2 = **33**; VFC_P = 33/60 = **0.55**. B_R = 35×1 + 5×0.5 + 0 = **37.5**; VFC_R = 37.5/90 ≈ **0.42**. Select **P first** (higher ratio), then with the remaining 40 of the 100 budget, R (90) does not fit → **P alone**.
- **V5.** y = 40 + (85−40) × (99.92−99.5)/(99.9−99.5) = 40 + 45×(0.42/0.40) → **clamp**: the expected response *exceeds* the desired level, so utility = **85** (no extrapolation beyond the elicited curve — interpolate only between bracketing points; state the clamp).
---

# PART X — Most-Likely Sub-Questions per Topic (self-interrogation checklist)

> For each EC3 topic: the sub-questions that fit the paper's scenario style, each with a one-line pointer to the answer. If you can answer every row without the notes, you are exam-ready.

## X.1 Pipes & Filters / MVC / patterns general
1. "Propose an architecture for processing X into Y" → source→filters→sinks diagram + walk-through + pipe annotations (capacity/EOT/when-full). [B3]
2. "How do you scale stage S?" → partition by key → replicate → merge; bounded back-pressure; limit = slowest stage + transfer cost. [B3/K1]
3. "Three limitations of the pattern here + remedies" → shared state, transfer overhead, error handling, real-time latency — with side pipes/metadata service/replicas/lambda. [B3]
4. "Why not pipes-and-filters for this UI/system?" → no natural stream; tight shared state; interaction control (then name the right pattern). [B14]
5. "Apply MVC: where does change C land?" → model component; controller unchanged; notify → views re-fetch; no UI regression. [B6/K2]
6. "MVC weaknesses for our toolkit?" → update storms, toolkit binding, model-interface coupling, cost for simple UIs. [B6]
7. "MVC vs PAC" → triad vs agent tree; cross-agent talk via Controls only. [B7]
8. "Why a broker here?" → heterogeneous distributed services, dynamic add/remove, location transparency + the five hand-offs cost. [B4/K3]
9. "SOA vs microservices for decomposition?" → granularity, ESB vs lightweight pipes, shared DB vs DB-per-service, governance. [B8]
10. "Which pattern for unknown producers/consumers?" → publish-subscribe; liabilities: ordering/delivery, latency; restore with durable queues + per-key order. [B9/Q2.7]

## X.2 Security patterns
1. "Contain a parser-bug compromise" → privilege separation: monitor/slaves, chroot+setuid, narrow state-machine interface, respawn-on-auth. [C2/K4]
2. "Real system example" → OpenSSH: ~2/3 unprivileged, ~2% change, historical bugs contained, no significant slowdown. [C2]
3. "Challenges + strategies for privilege separation" → identity change (state export/respawn); interface attack surface (small per-state set, terminate); platform gaps (degrade); residual DoS (limits). [C2]
4. "Secure Logger objectives/applicability/trade-offs" → unreadable+tamper-evident logs; three applicability conditions; slower ops, high-value store. [C3]
5. "What never enters a log?" → OWASP list; remove debug logs; one logging interface; no stdout/stderr. [C3]
6. "Clear Sensitive Information trade-offs" → every item/release point; compiler elision; copies already made. [C4]
7. "Combine both in one payment transaction" → tokenise → minimal log event (token + last 4) → destructor overwrite incl. error paths → ship signed record. [C5]
8. "OpenID vs OAuth" → who vs what; tokens+scopes; pairing in mobile. [C6]
9. "Where does the trust boundary change?" → internet→DMZ→internal; every crossing: authenticate, validate, encrypt. [C1]

## X.3 Cloud & integration
1. "Financial case for cloud" → scale economies (power/1,000-admin ratio/30% hardware), utilisation via co-location, elasticity+measured service for spiky workloads. [D3]
2. "IaaS vs PaaS vs SaaS for this app" → control vs speed of delivery; public/private per compliance. [D2]
3. "Multi-tenant risks + defences" → leakage/escape/side-channel/DoS; tenant filters in one layer, row-level security, per-tenant keys. [D5/D10]
4. "Auto-scaling gap" → VM start latency; proactive signals (queue depth), warm pools, bounded queues. [D10]
5. "99.95% means?" → 4.38 h/year; 0.36 h/30-day month; app must survive it offline-first. [D8/E8]
6. "Pick the integration strategy for each partner" → file/RPC/messaging/WebSocket/gateway rules; compose. [T1–T7/W1]
7. "CAP for this operation?" → payments CP, feeds AP, CA theoretical; two-node example. [D8/K5]
8. "Cache placement + failure modes" → boundaries; staleness, hot keys, stampede, PII in cache. [W3]
9. "Serverless vs containers here?" → latency × load shape × state locality; warm/provisioned middle. [W4]
10. "Failure-management architecture" → timeouts, breakers, bulkheads, retries+idempotency, degradation, replication, transactions for money. [Q4.5/U]

## X.4 Mobile
1. "Native vs cross-platform (six attributes + decision)" → table + hardware caveat + spike method. [E3/Q5.1]
2. "Responsive design case: loading/navigation/interaction" → Boston Globe table + mechanisms + honest bandwidth limit. [E4/Q5.2]
3. "Offline design + conflicts" → SQLite primary; idempotent queue; LWW vs server-wins vs refuse-for-money. [E5/Q5.3]
4. "Decompose app X into Android components" → Activity/Service/Provider/BroadcastReceiver + directed/broadcast intents. [E6/Q5.4]
5. "Why not run sync in the receiver?" → receivers react, no lifetime; Service owns background work. [Q3 mock d]
6. "Where does the cross-platform framework sit?" → framework layer above HAL; hardware via native APIs; bridge overhead. [Q5.6]
7. "m.site vs responsive" → second-product cost vs one-site economy; per-segment decision. [Q5.5/C5]

## X.5 Big data & NoSQL
1. "Apply MapReduce to logs" → map/shuffle/reduce with <key,1> example + verification; batch vs stream trade-off. [F3/Q6.1/Q6.2]
2. "Weighted average job" → carry counts; Σ(avg×count)/Σ(count); naive mean wrong. [F3/M-D9]
3. "When is Hadoop wrong?" → low latency, many small files (NameNode memory), arbitrary writes. [F4]
4. "SQL vs NoSQL for workload X" → ACID/joins need vs scale/flexibility; per data class; ledger stays relational. [F6/X6.4? use Q6.4]
5. "Four NoSQL models + workloads" → document/key-value/column/graph + shard-key risk. [F6/Q6.5]
6. "HDFS write path + fault tolerance" → 8 steps; heartbeats/re-replication; tiny-file problem. [F2/M, Q6.6]
7. "ML pipeline architecture" → 7 stages; training vs serving shapes; versioned models; drift. [P2]
8. "Blockchain: when on-chain?" → only jointly-trusted facts; hashes off-chain; throughput/latency costs. [P3]

## X.6 CBAM & evaluation
1. "Run the 9 steps on two strategies" → collate→refine→vote→utility→map→interpolate→benefit→VFC+budget→intuition; show formulas. [G5/K8]
2. "Interpolate utility" → bracketing points; y = ya+(yb−ya)(x−xa)/(xb−xa); clamp at curve ends. [G5/V5]
3. "Classify the finding" → risk/nonrisk/sensitivity/tradeoff with reason lines. [Y2]
4. "Utility curve shapes and their lesson" → five shapes; where marginal improvement is worth buying. [G2]
5. "Opportunity cost of each strategy" → best forgone alternative; hybrid option. [G8]
6. "ATAM 3-day plan + outputs" → phases/steps compressed; 7 outputs; utility-tree fragment. [Y1]
7. "What CBAM delivers/limits" → VFC order + structured discussion; consensus-sensitive, intuition-checked. [G7/Q7.4]

## X.7 Foundations
1. "Define architecture + unpack" → structures; elements/relations/properties; abstraction; every system has one. [A1/Z1]
2. "Structure vs view; module vs component; layer vs tier" → the pairs table. [A1/L]
3. "Three structures to document first" → decomposition/uses/data model with reasons. [Z2]
4. "Construct an availability scenario + calculate downtime" → six-part grid + series/pair arithmetic. [S1/M2]
5. "Tactics for attribute X + one side effect" → category lists; ping/echo cascade example. [A3/B13]
6. "Conformance: how do you keep code true?" → evident coding, frameworks, templates, doc sync + reconstruction. [O7]
7. "Reconstruction on the legacy system" → purposes + 4 phases + migration feed. [O7/Y3]
8. "Why document views?" → communicable/analysable/checkable; template; security view example. [O6/Y4]
---

# PART Y — Comparison Compendium (beyond the pattern master table)

## Y1. Middleware identities: Broker vs ESB vs API Gateway vs Message Broker vs Service Mesh

| Dimension | Broker (POSA) | ESB (SOA) | API Gateway | Message Broker (MQ) | Service Mesh |
|---|---|---|---|---|---|
| Level | In-app distributed middleware | Enterprise integration | Client edge | Transport between apps | East-west infra per service |
| Core act | Locate + forward request/reply | Route, transform, orchestrate | Policy at entry: auth, rate, aggregation | Store-and-forward, pub-sub | mTLS, retries, traffic policy via sidecars |
| Coupling reduced | Location | Interface + protocol | Topology | Time + identity | Operational policy |
| Failure role | SPOF unless replicated | Central hub risk | Entry bottleneck risk | Buffered (survives consumer faults) | Per-hop resilience |
| Exam pairing | CORBA/WWW | CIGNA-style integration | Microservice/mobile front | Event-driven systems | Kubernetes-era fleets |

One-liner: all five are "use an intermediary" at different altitudes — name the altitude and the coupling each removes, and the question is answered.

## Y2. Redundancy spectrum

| Kind | Sync | Takeover speed | Cost | Use |
|---|---|---|---|---|
| Active (hot) pair | Both live, state synchronised | Immediate | Highest (2× live capacity) | Payment/checkout paths |
| Passive (warm spare) | Standby updated periodically | Seconds–minutes (state catch-up) | Medium | Databases, services where 1-min loss is tolerable |
| Cold spare | Off; started on failure | Minutes+ (boot + data restore) | Lowest | Non-critical internal tools |
| Shadow operation | Parallel run without serving | N/A (verification) | Medium | Migration/cutover validation |

## Y3. Consistency mechanisms across the syllabus

| Mechanism | Where it appears | Guarantee |
|---|---|---|
| ACID transactions | Relational DBs; payment paths | Atomicity + isolation |
| Two-phase commit (mention only) | Cross-node relational | Strong but costly/fragile — why sagas replace it |
| Saga/compensating actions | Microservices | Eventual; business-level reversal |
| Eventual consistency | AP NoSQL, replicas | Copies agree after delay |
| Last-write-wins / server-wins / merge | Mobile sync | Conflict policy choice |
| Bounded staleness window | Caches, read replicas | Fresh within N seconds |
| Read-your-own-writes routing | Sessions after a write | Per-user consistency illusion |

## Y4. Evaluation methods: ATAM vs lightweight vs CBAM

| Dimension | ATAM | Lightweight evaluation | CBAM |
|---|---|---|---|
| Question | Does the architecture meet quality goals? What are the risks? | Same, cheaper | Which strategy is worth its cost? |
| Unit | Scenarios vs decisions → findings | Fewer scenarios, fewer steps | Strategies × scenarios → utilities → VFC |
| Participants | Team + decision makers + stakeholders | Smaller set | Stakeholders who can speak to utility |
| Output | Risks/nonrisks/sensitivity/tradeoffs, risk themes | Short risk list | Ranked strategies within budget |
| Use | Before build / major release | Agile cadence | Portfolio budget decisions |

## Y5. Data-tier growth options (recap with the trigger points)

| Option | Trigger | Buys | Costs |
|---|---|---|---|
| Vertical scale (bigger node) | CPU/RAM pressure, moderate growth | Simplest | Ceiling; SPOF; cost curve steepens |
| Read replicas | Read-heavy load | Read throughput, HA-ish | Replication lag; writes unchanged |
| Caching layer | Hot reads | Latency + DB offload | Staleness/hot keys |
| Sharding | Write saturation | Linear write scale | Shard-key design; cross-shard queries |
| NoSQL offload | Flexible/appended data at scale | Schema freedom, AP scale | No ACID/joins |
| Warehouse/lake + batch | Analytics at petabyte scale | Cheap scans | Latency (batch) |

## Y6. Mobile app-type decision by segment (the C5 rule)

| Segment trait | Choose | Why |
|---|---|---|
| Occasional task, payment, search-findable | Responsive web | No install friction; one code base |
| Daily task, offline, camera/GPS, push | App (native modules as needed) | Device APIs + background work + store trust |
| Composite UI, many agent screens | Native/PAC thinking | Complexity of coordinated panels |
| Content-only readers | Responsive web | Zero-install, SEO |

## Y7. The "two axes" cheat cards

- **CAP axes:** consistency vs availability during partitions (P fixed).
- **App-type axes:** device capability (full/partial) × platform reach (single/multiple) → native / hybrid / HTML5 quadrants.
- **Latency vs completeness axes:** batch (complete, late) vs stream (fast, approximate) → lambda = both.
- **Tenancy axes:** isolation/cost × customisation → per-tier decision.
- **Automation axes:** IT task → IT service → business service (value rises upward).
---

# PART Z — Verbatim Definition Sheet (write these word-perfect)

**Software architecture:** the set of structures needed to reason about the system, which comprise software elements, relations among them, and properties of both.

**Structure:** a set of elements held together by a relation.

**View:** a representation of a coherent set of architectural elements, as written by and read by system stakeholders — a representation of one or more structures.

**Architectural pattern:** a relationship between a context (a recurring situation giving rise to a problem), a problem (appropriately generalised) and a solution (appropriately abstracted) — the solution described by element types, connectors, topological layout and semantic constraints.

**Tactic:** a design decision that influences a single quality-attribute response; a pattern is a molecule built of tactic-atoms.

**Filter:** a component that transforms data read on its input port(s) into data written on its output port(s), consuming and delivering incrementally.

**Pipe:** a connector conveying data from one filter's output to another filter's input — single source, single target, order-preserving, never altering data.

**MVC:** divides an interactive application into a model (core functionality and data), views (display) and controllers (input handling), with a change-propagation mechanism ensuring consistency between the model and its views.

**Broker:** an intermediary component that coordinates communication between clients and servers — locating servers, forwarding requests, transmitting results and exceptions.

**SOA:** a pattern describing a collection of distributed components that provide and/or consume services; consumers understand and use services without detailed knowledge of their implementation.

**Blackboard:** several specialised subsystems assemble a possibly partial solution on a central shared data store (the blackboard), activated opportunistically by a control component.

**Least privilege:** every program and every user should operate using the least amount of privilege necessary to complete the job.

**Privilege separation:** letting parts of an application run with different privilege levels so errors in unprivileged parts cannot be abused to gain privileges — a privileged monitor plus unprivileged slaves communicating over a narrow, validated interface.

**Secure Logger:** all log output passes through a secure logging subsystem that stores records so unauthorised users cannot access them and modifications are detectable by authorised users.

**Clear Sensitive Information:** erase sensitive data from reusable resources (buffers, caches, files) before or immediately after the program finishes with it.

**Trust boundary:** a line in the architecture across which the level of trust in data or in the sender changes.

**Attack surface:** the set of points through which untrusted input can reach a component, together with the code that handles that input.

**Cloud (characteristics):** on-demand self-service; ubiquitous network access; resource pooling; location independence; rapid elasticity; measured service; multi-tenancy.

**Hypervisor:** the software layer on a host server that manages virtualisation — creating, scheduling and isolating virtual machines on the host's hardware.

**Virtual machine:** an isolated address space that looks like a bare-metal machine to its applications, has its own IP address, and runs any OS the host processor can execute.

**HDFS:** a distributed file system comprising a NameNode (metadata, in memory), many DataNodes (replicated blocks), connected by RPC and streaming protocols, self-healing via heartbeats and re-replication.

**MapReduce:** a pattern for analysing a large distributed dataset by executing in parallel on many nodes; map performs extract/transform producing <key, value> pairs; reduce summarises pairs sharing a key; the infrastructure deploys instances and recovers failures.

**CAP theorem:** it is impossible for a web service to provide consistency, availability and partition tolerance simultaneously; any two, never all three.

**NoSQL database:** a non-relational database designed to be scalable, fast, available and tolerant of semi-structured data, usually without ACID transactions and without joins.

**Sharding:** distributing one database's data over several machines. **Replication:** keeping copies of the same data on several machines.

**Multi-tenancy:** one application instance serving multiple consumers who are unaware of each other.

**Utility:** the benefit stakeholders derive from the quality-attribute responses an architectural decision achieves (scale: worst acceptable = 0, best useful = 100).

**Utility-response curve:** the graph relating a scenario's response measures to corresponding utility values.

**Value for cost:** VFC = total benefit ÷ cost; used to rank-order architectural strategies.

**CBAM:** a nine-step method selecting architectural strategies by value for cost — collate, refine, prioritise, assign utility, map strategies, interpolate, compute benefit, choose by VFC within constraints, confirm with intuition.

**Opportunity cost:** the value of the best alternative forgone by a choice; sunk costs are not opportunity costs.

**ASR:** a requirement that has a profound effect on the architecture.

**Utility tree:** hierarchical representation — root utility → quality attributes → attribute refinements → ranked scenarios (importance, difficulty).

**ATAM:** a scenario-driven evaluation method producing risks, nonrisks, sensitivity points and tradeoff points through nine steps in four phases.

**Risk / nonrisk:** a potentially problematic, unproven decision / a decision proven safe by analysis.

**Sensitivity point:** a decision with a marked effect on a particular quality-attribute response. **Tradeoff point:** a sensitivity point affecting more than one attribute where improving one may degrade another.

**Store locally, synchronise later:** the mobile pattern in which the application records data in local storage and transmits to the server when connectivity permits.

**Responsive design:** building one website whose rendering adapts to screen size and orientation via flexible grids, flexible images and media queries.

---

# PART Z2 — Ready-Made Scenario Library (memorise two per attribute; adapt in the hall)

**Availability scenarios (pick per system):**
1. *Rail booking:* external payment gateway stops responding at 08:00 peak → detect < 5 s, queue payments, hold seats, degraded mode (no payment) until restore < 30 s → no confirmed booking lost.
2. *Streaming:* an instance of the recommendation service crashes → balancer removes it < 10 s, cached "trending" grid serves, full service restored < 60 s → ≥ 99.95% uptime monthly.
3. *IoT:* a region's ingestion node fails → gateway buffers device events (store-and-forward), alerts switch to secondary region < 15 s → data loss ≤ 30 s of telemetry.

**Performance scenarios:**
1. *Results evening:* 1 M students fetch marks → page render ≤ 2 s at 99th percentile; ≥ 4,000 TPS; CDN/cache absorbs 90% of reads.
2. *MapReduce report:* monthly billing over 40 PB → job completes < 6 h; no straggler > 1.5× median task time.
3. *Ride ETA:* ETA on driver move → ≤ 150 ms compute; ≤ 400 ms end-to-end at 3,000 events/s.

**Security scenarios:**
1. *Payments:* attacker attempts transaction alteration → authenticated+authorised only; % blocked ≥ 99.9%; every attempt audited (tamper-evident) within 1 s.
2. *Hospital:* unauthorised record read attempted → RBAC denies; audit record appended; detected ≤ 90 s.
3. *Portal:* SQL-injection attempt → parameterised queries reject; intrusion detector logs signature; alert ≤ 5 s.

**Modifiability scenarios:**
1. *E-commerce:* add a payment provider → change confined to the payment adapter module → ≤ 1 module, ≤ 2 person-days, zero UI regression.
2. *Tax platform:* change assessment-rule order → workflow configuration edit, no code deploy → ≤ 1 working day.
3. *Mobile:* add a new notification channel → subscribe new consumer to the bus → no producer changes.

**Usability / testability / interoperability one-liners:**
- Usability: first-time user books a cab in ≤ 3 taps; undo ≤ 2 steps.
- Testability: every fraud rule reproducible by replaying its recorded input stream; 100% rule-branch coverage nightly.
- Interoperability: a new district service registers and serves dispatch calls in ≤ 1 day with zero client changes.

**How to use the library:** an exam scenario rarely matches a stored scenario exactly — reuse the *shape* (source/stimulus/environment/response/measure) and swap the domain nouns. Store the response measures: they are the hardest part to invent under time pressure.

---

# PART Z3 — The Assumption Bank (20 reusable opening lines)

1. "≈2,000 concurrent users at peak; 40,000 TPS off-peak baseline of 200 TPS." *(size)*
2. "Three data feeds arrive as streams keyed by customer ID." *(shape)*
3. "The external bureau SLA is 99% with multi-minute outages twice a quarter." *(dependency)*
4. "Reads outnumber writes 20:1." *(read/write mix)*
5. "Peak lasts 12 hours per day during exam fortnight; near-zero otherwise." *(spiky)*
6. "Field workers operate with connectivity available ≤ 50% of the working day." *(offline)*
7. "Team of six developers skilled in JavaScript; two have mobile experience." *(capacity)*
8. "Data is confidential (cardholder/personal) and regulated; retention ≤ 30 days for raw documents." *(compliance)*
9. "Growth forecast: 3× users in 12 months, 10× in 3 years." *(horizon)*
10. "The platform must run on both Android and iOS from day one." *(reach)*
11. "Existing monolith releases quarterly; two squads need weekly." *(org pain)*
12. "A new VM instance takes 2–5 minutes to become ready." *(elasticity lag)*
13. "The cache cluster is Redis-class: < 1 ms reads, replicated, ~64 GB." *(cache)*
14. "Message ordering is guaranteed only per key; delivery at-least-once." *(bus)*
15. "Budget: 60 person-months; deadline is fixed by the board." *(economics)*
16. "Weights from stakeholders: response time 1.0, availability 0.6." *(CBAM)*
17. "Illustrative costs: if strategy A costs x, B ≈ 2x, C ≈ 0.6x." *(relative costs)*
18. "Each inverter reports every 30 s; 200,000 devices." *(IoT scale)*
19. "Users are on 3G/4G networks with 1–3 s page-load tolerance." *(mobile net)*
20. "Audit trail must be tamper-evident and retained 7 years." *(audit)*

*Rationale:* a specific assumption licenses specific design numbers later; examiners read "assumptions stated clearly at the beginning" as the first scored instruction of the paper.
---

# PART AA — MASTER RAPID REVISION SHEET (the whole syllabus compressed — read thrice on the last day)

## Foundations
- Architecture = structures (elements + relations + properties) to reason about a system. Plural. Abstraction omits element-private detail. Every system has one. Box-and-line ≠ architecture. "Early/major" rejected as definition.
- Families: **Module** (decomposition, uses, layer, class, data model) / **C&C** (service, concurrency; attachment) / **Allocation** (deployment, implementation, work assignment).
- Structure exists; view is its documented representation. Mappings many-to-many. Layer (code) ≠ tier (deployment).
- Patterns: context + generalised problem + abstracted solution (element types, connectors, topology, constraints). Categories: Mud-to-Structure / Distributed / Interactive / Adaptable. Pattern = scheme, not library. Molecule of tactics.

## Quality attributes & tactics
- Scenario: **source, stimulus, artifact, environment, response, response measure**.
- Availability: detect (ping/echo, heartbeat, monitor, voting) / recover (active+passive redundancy, retry, degradation, rollback) / prevent (transactions, increased resources). `A = MTTF/(MTTF+MTTR)`; series multiply; pair `1−(1−A)²`; 99.95% = 4.38 h/yr.
- Performance: manage demand (reduce overhead, event rate, limit response, bound queues) / manage resources (increase, concurrency, multiple copies, scheduling).
- Security: detect (intrusion, DoS, message integrity) / resist (identify-authenticate-authorise, limit access/exposure, separate entities, validate, encrypt) / react / recover (audit trail).
- Modifiability: localise (coherence, anticipate, generalise) / prevent ripples (encapsulate, intermediary, restrict dependencies, defer binding).
- Side effects cascade (ping/echo → resources → scheduling → intermediary → restrict paths → recursion); stop when small.

## ASR & evaluation
- ASRs via requirements, QAW (brainstorm→prioritise→refine), PALM, design decisions, utility tree.
- Utility tree: utility → attributes → refinements → scenarios (H/M/L × H/M/L); analyse (H,·) first.
- ADD: choose module → ASRs → generate design (tactics/patterns) → iterate. Agile: just enough, spikes, sweet spot.
- ATAM: 4 phases, 9 steps (P-B-P, I-U-A, B-A-P); findings = risk, nonrisk, sensitivity, tradeoff; 7 outputs; scenario-driven, not exhaustive. Lightweight for agile.

## Patterns (one-liners)
- Layers: unidirectional allowed-to-use; isolation vs hop cost.
- Pipes&Filters: filter transforms incrementally, pipe conveys; forces: exchange/recombination, small steps, no intermediate files, parallelism; liabilities: state, transfer cost, format overhead, errors; scale = partition/replicate/merge.
- Blackboard: board + independent knowledge sources + opportunistic control; no deterministic strategy; repository/production variants.
- Broker: locate+forward; client/server/broker/2 proxies/bridge; 5 hand-offs; transparency vs latency/SPOF; CORBA/WWW.
- MVC: model+views+controllers; 5 interactions; change propagation; multiple views, pluggability vs update storms, model-interface coupling; Document-View variant.
- PAC: agent tree (P+A+C each); controls are the only cross-agent channel.
- SOA: providers/consumers/ESB/registry/orchestration; SOAP/REST/messaging; complexity + overhead. Microservices: fine-grained, DB-per-service, lightweight pipes, breakers/bulkheads, independent deploys.
- Client-Server (bottleneck/SPOF) / P2P (symmetric, no critical peer, hard ops) / Pub-Sub (announce-listen; latency, weak guarantees) / Shared-Data (store SPOF; schema coupling).
- Microkernel (mechanism/policy core) / Reflection (meta level modifies base).
- MapReduce: files in, stateless maps, <key,value> only; shuffle = distributed sort; weaknesses: small data, stragglers, multi-stage; weighted averages carry counts.
- Multi-tier: logical deployment groupings; one component one tier.
- Tactics↔patterns: patterns built from tactics (MVC = coherence+encapsulation+intermediary+run-time binding); tactics repair weaknesses (broker copies); patterns underspecified.

## Security patterns
- C&C security rows: component (privilege), connector (authenticate/validate/encrypt), store (access control/clearing), logs (protected store).
- Privilege separation: monitor (state machine, terminate-on-invalid) + slaves (chroot, unused UID); informational/capability/identity requests; state-export respawn; OpenSSH: 2/3 unprivileged, 2% change, bugs contained, no slowdown; DoS remains.
- Secure Logger: one interface; mask OWASP secrets; encrypt/sign/chain; append-only + central ship; reader/viewer; costs: ops convenience, per-record cost, high-value target.
- Clear Sensitive Information: overwrite incl. error paths; compiler elision; can't recall copies.
- Payment combo: tokenise → minimal signed log event → destructor clear → ship.
- TLS channel; OpenID who; OAuth what; DMZ; LDAP directories.

## Cloud & integration
- 7 characteristics; SaaS/PaaS/IaaS control ladder; private/public/community/hybrid.
- Cheap because: scale (power 15–20%, 1,000 vs 150 servers/admin, 30% hardware), utilisation (co-located load shapes), multi-tenancy (support/upgrades/one version).
- Hypervisor isolates via page tables; HDFS (NameNode memory metadata, 3 replicas, heartbeats, tiny-file weakness); IaaS managers; PaaS = LAMP; KV/document stores (no transactions/joins/schemas/joins).
- Cloud qualities: multi-tenant security risks; auto-scaling lag → proactive; instances WILL fail.
- AWS principles: disposable resources, automation, loose coupling, services not servers, caching, security everywhere. Begin with service definition (task → service → business automation).
- CAP: pick two; AP Cassandra/CouchDB, CP MongoDB/Redis, CA theoretical; two-node example; per-operation choice.
- Integration: file (bulk/legacy), shared DB (one team), RPC (immediate, coupled), messaging (decouple, idempotent consumers), WebSockets (browser push), API gateway (policy edge). Compose them.
- Containers (kernel-shared, fast) vs VMs; serverless (scale-to-zero, cold starts, state outside); distributed cache (staleness, hot keys); self-healing (detect-recover-verify); bulkheads; breakers.

## Mobile
- 5 considerations: simple UI, responsive, compact, few layers, store-locally.
- Native (fast, full access, one platform) / web (instant, reach, weak hardware) / hybrid (shell, one code base, slower); quadrant: capability × reach.
- Cross-platform decision: 6 attributes + hardware caveat (continuous camera/GPS → native modules); React Native/Xamarin(.NET MAUI)/Flutter/KMP; spike + measure + record.
- Responsive: flexible grids (media queries, %), flexible images, per-size layout; Boston Globe; 5 UI patterns (action bars, social login, large buttons, notifications, discoverable controls).
- Store-locally: SQLite primary, sync service, idempotent queue (UUID dedup), conflict rules (LWW/server-wins/merge/refuse-for-money).
- Android: Activity (UI) / Service (background) / Content provider (storage API) / Broadcast receiver (events); directed + broadcast intents; patient-monitoring chain; Uber decomposition.
- Stack: apps → native+web frameworks → core → kernel (HAL); hardware = CPU/GPU/DSP/radios/sensors; framework layer is where cross-platform tools live.
- Back end: REST/JSON; tenancy per tier; 99.95% = 0.36 h/month → design offline-first.

## Big data & technologies
- 3Vs: volume → distribute; variety → schema-flexible; velocity → batch vs stream.
- Hadoop = HDFS + MapReduce. Map (local, stateless) → shuffle/sort (per key) → reduce. Combiner optional. Verification by totals.
- Hadoop right: bulk analytics, logs, recommendations; wrong: low latency, many small files, arbitrary writes. Batch vs stream vs MPP table; lambda = batch truth + stream speed.
- NoSQL: scalable (shard), fast (memory), available (replication), flexible; minus ACID and joins; shard key = hot-spot risk; four models (document/key-value/column/graph) by access pattern; ledger stays relational.
- ML pipeline: collect→prepare→algorithm→train→evaluate→deploy→monitor/retrain; training vs serving shapes; versioned models.
- Blockchain: append-only hash-chained ledger, consensus; on-chain only jointly-trusted facts; IoT: devices→gateway→bus→stream+batch→apps.

## CBAM & economics
- Utility = common currency (0–100; worst 0, best 100); curves: diminishing, value-near-top, S, linear, step; response→utility by interpolation `y = ya+(yb−ya)(x−xa)/(xb−xa)`, clamp.
- Weights: 100 votes, top = 1.0. b = U_exp − U_cur (negatives count); B = Σ b×W; VFC = B/C; rank; budget stop; step 9 intuition.
- 9 steps: collate (top ⅓) → refine (4 levels) → prioritise (top 50%) → utility → map → interpolate → benefit → VFC select → confirm. Filters: 2 for scenarios, 1 for strategies.
- NASA: 10 scenarios/strategies, votes 100, strategy 1 (0.79) & 3 (0.69) chosen within 1600; consensus inputs; sensitive to utilities; numbers support intuition.
- Opportunity cost = best forgone alternative (sunk ≠). S vs F: features/earlier-release vs scalability utility + rework; factors: growth timing, later-cost, deadline penalty, reversibility, risk tolerance, hybrid.

## Trends
- Wasm (portable sandbox near-native); service mesh (sidecar mTLS/retries/traffic); edge (latency/bandwidth/locality; tiered processing); micro-frontend; serverless economics.

## Exam craft
- 3.75 min/mark; 60–80 words/mark; assumptions first; map onto elements; both sides; tactic-repairs; decision + flip condition; formula→substitution→result→interpretation; tables for comparisons; one number per benefit claim; 5-minute final check.
---

# PART AB — THE 150-MINUTE HALL PLAN

- **Minutes 0–5:** read all four questions once; mark command words; start with your strongest question (confidence + momentum); write its assumptions immediately.
- **Minutes 5–40 (Q1), 40–75 (Q2), 75–110 (Q3), 110–145 (Q4):** ~35 minutes each. Inside each: 3 min re-read + assumptions; 5 min outline (bullets + diagram boxes); 25 min writing to the marks (sub-part order as printed, never skip a sub-part — a 1-mark sub-part is one sentence); 2 min re-read against the command words.
- **Minutes 145–150:** sweep — every sub-part numbered? every calculation has formula→substitution→result→interpretation? every conclusion consistent with your stated assumptions? every diagram labelled?
- **If a question stalls:** write the skeleton (assumptions + attribute + candidate pattern/tactic + both sides) and move on — returning with 5 minutes of fresh eyes beats polishing one answer to 12 marks-worth of words.
- **Marks discipline:** sub-parts printed with [2] get two ideas; [1.5] gets one idea + a labelled sketch; never exceed the printed structure — the marker's rubric follows the paper's numbers, so should your paragraphs.
- **Never leave a 10-mark question with white space after the assumptions** — a structured attempt with mapped elements earns half even when the design is imperfect; blank earns zero.
