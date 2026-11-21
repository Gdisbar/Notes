# PART 1 — CONCEPTS
*What it is · Why we need it · How it works · Where it fits (and what it costs) · Minimal example*
REF format `[file:page]` → see `00_READ_FIRST.md`.

**Contents:** A Foundations · B Quality-attribute framework · C The seven quality attributes (+ others) · D ASR & design strategy · E Architecture & agility · F Documentation & views · G Layered-architecture techniques · H Evaluation, conformance, reconstruction · I Patterns catalogue · J Cloud · K Mobile · L Big data & NoSQL · M Security technologies · N ML systems · O Management & governance · P Economics (CBAM)

---

# A. FOUNDATIONS

## A1. What is software architecture?
- **What:** The set of **structures** needed to reason about a system — its **elements**, the **relations** among them, and the **properties** of both 🆕. *The course stresses structures, not "early/major decisions"*: many decisions are early but not architectural, and "major" is hard to judge, while structures are easy to identify. `[1:1-2]`
- **Why:** Structures let you **reason about system properties** (performance, availability, modifiability…) that matter to a stakeholder. A structure is *architectural* only if it supports such reasoning. `[1:4]`
- **How:** Software has **many structures**; **no single structure is "the" architecture**. `[1:2]`
- **Cost/limits:** Every system has an architecture, but it may be undocumented. There is **no inherently good/bad architecture — only more/less fit for a purpose**, judged against stated goals. `[1:19, 1:24]`
- ✍ *Terms:* structure, element, relation, reasoning, fit-for-purpose, stakeholder concern.

## A2. Three categories of structures `[1:3, 1:6-8, 1:24]`
| Category | Elements | Answers questions like | Example structures `[1:9-14]` |
|---|---|---|---|
| **Module** | code/data units (classes, layers, subsystems) | What is each module responsible for? What may it *use*? Who inherits from whom? | Decomposition (is-a-submodule-of → modifiability, project org); **Uses** (requires correct presence of → subsets/incremental dev); **Layers** (virtual machine, allowed-to-use → portability); **Class/generalisation** (inherits-from → reuse) |
| **Component-and-connector (C&C)** | runtime components (services, clients, servers, filters) + connectors (call-return, pipes, protocols) | What runs, what is replicated, what shares data, what runs in parallel, how does data flow? | Service structure, **Concurrency** (parallelism/contention), shared-data, client-server, process |
| **Allocation** | software ↔ non-software (CPU, file system, teams) | Which processor runs it? where stored? which team builds it? | **Deployment** (allocated-to, migrates-to → performance/availability/security), implementation, work assignment |

- **Table to quote in exams:** relations and "useful for" for each structure → `[2:17-19]`.
- **View vs structure:** *structure* = the elements as they exist; *view* = a **representation** of a structure written for stakeholders. **Architects design structures and document views.** `[1:5]`
- **Modules ≠ components:** don't expect a one-to-one mapping (one module may become many runtime instances). `[1:15, 1:23]`
- ✍ *Example sentence:* "For a payment system, the layer structure (module) tells us portability, the deployment structure (allocation) tells us availability, and the concurrency structure (C&C) shows contention on the ledger."

## A3. Architectural patterns and tactics (preview) `[1:16-18, 1:33-34]`
- **Pattern** = packaged, proven composition of element types and interactions that solves a recurring problem (Layered; Shared-data, Client-server; Multi-tier, Competence-centre, Platform).
- **Tactic** = primitive design decision affecting **one** quality-attribute response. **Patterns are packages of tactics** — molecule vs atom. `[1:43, 3:106]`
- *Why tactics:* patterns are complex bundles that must be adapted; if no pattern exists, build a fragment from first principles; cataloguing makes design systematic. Choice depends on trade-offs and cost. `[1:34]`

## A4. What makes an architecture "good"? — rules of thumb `[1:19-23]`
- **Process rules:** single architect/small team with a leader (conceptual integrity — also in agile/open source); prioritised list of QA requirements; documented with views; evaluated with a formal method (ATAM); supports incremental implementation (a "skeletal system").
- **Product rules:** information hiding + separation of concerns (modules encapsulate what is likely to change); use well-known tactics/patterns; **never depend on a particular version of a commercial tool**; separate data producers from consumers; few ways for components to interact; small, specified set of resource-contention areas; processes re-assignable to processors.
- ✍ Link each rule to a QA: *information hiding → modifiability; few interaction mechanisms → understandability, reliability; separated producers/consumers → modifiability.*

## A5. ⚠ NOT-CARRIED — Why architecture is important (original Part 1, slides 4-21)
Eight reasons you can list: ① **inhibits or enables quality attributes** (most important message of the course) ② **reason about and manage change** — ~80% of cost is after deployment; changes are *local*, *non-local* or *architectural* ③ **early prediction** of qualities through analysis ④ **communication** among stakeholders ⑤ **earliest design decisions** (hardest to change) ⑥ **constraints on implementation** ⑦ **influences organisation structure** (work-breakdown) ⑧ **enables evolutionary prototyping** (skeletal system), better **cost/schedule estimates**, **transferable/reusable model**, use of **independently developed components**, **restricted design vocabulary**, **basis for training**.

## A6. ⚠ NOT-CARRIED — Contexts and influences (original Part 1, slides 24-43)
- Four contexts: **Technical** (QAs, current practice), **Project life-cycle** (waterfall/iterative/agile/MDD all include architecture activities), **Business** (QAs must trace to a business goal — "why do you want fast response?"), **Professional** (architect also manages stakeholders, needs soft skills).
- **Architecture Influence Cycle:** requirements + business + technical + project + professional influences → architect → architecture → system → which feeds back and changes the environment that influences the *next* architecture.
- **Stakeholder** = anyone with a stake in success; each has different concerns; identify and engage early.

## A7. Requirements: functional, quality attributes, constraints `[1:25-29]`
- **Functional** = what the system must do. **Quality attribute (QA)** *qualifies* a functional requirement (how fast, how resilient, how easy to learn). **Constraint** = a design decision with **zero degrees of freedom**.
- **Functionality does not determine architecture** — infinite architectures can satisfy the same function; functionality and QAs are **orthogonal**. `[1:26]` *Example:* "press green button → Options dialog": performance = how quickly it appears; availability = how often it fails/how fast repaired; usability = how easy to learn. `[1:27]`
- *Three problems with old QA talk:* untestable definitions ("modifiable"), wasted time arguing which attribute a concern belongs to (a DoS attack hits availability/performance/security/usability?), separate vocabularies. **Fix = quality-attribute scenarios** + a short discussion of each attribute's underlying concerns. `[1:28-29]`

---

# B. QUALITY-ATTRIBUTE FRAMEWORK

## B1. Quality attribute scenario (QAS) `[1:30-32]`
Six parts — *use this to turn any vague requirement into a testable one*:
1. **Source of stimulus** (human, system, actuator) 2. **Stimulus** (condition requiring response) 3. **Environment** (normal, overload, startup, degraded…) 4. **Artifact** (what is stimulated) 5. **Response** (activity undertaken) 6. **Response measure** (how it is measured so it can be tested).
- **General scenario** = attribute-wide template; **concrete scenario** = specific to the system. Example availability: *"Heartbeat monitor finds the server non-responsive during normal operation; system informs operator and continues with no downtime."* `[1:46]`
- ✍ *Template sentence:* "[Source] [stimulus] on [artifact] in [environment]; the system [response] within [measure]."

## B2. Tactics `[1:33-34, 1:43]`
Design decision that influences a single QA response. Used to adapt patterns, build from first principles, and make design systematic. Pick on **trade-offs among QAs and cost**.

## B3. Seven categories of design decisions `[1:35-43]`
| # | Category | Typical decisions |
|---|---|---|
| 1 | **Allocation of responsibilities** | identify responsibilities (functions, infrastructure, QA support); assign to modules/components/connectors `[1:36]` |
| 2 | **Coordination model** | who coordinates; properties (timeliness, currency, completeness, correctness, consistency); mechanisms: stateful/stateless, sync/async, guaranteed/non-guaranteed delivery, throughput/latency `[1:37]` |
| 3 | **Data model** | major abstractions, operations, create/persist/translate/destroy, metadata, relational vs objects `[1:38]` |
| 4 | **Management of resources** | which resources, limits, who manages, arbitration under contention, behaviour at saturation `[1:39]` |
| 5 | **Mapping among architectural elements** | module↔runtime element, element↔processor, data↔store, element↔delivery unit `[1:40]` |
| 6 | **Binding time** | when is a choice bound (build, startup, runtime) — e.g. parameterised build, runtime protocol negotiation, plug-ins, app store `[1:41]` |
| 7 | **Choice of technology** | availability of tools, team familiarity/support, side effects (forces a coordination model), compatibility with stack `[1:42]` |

**Design checklists:** for each QA the course gives a checklist *per category* (7 pages each): availability `[1:58-64]`, performance `[1:74-80]`, usability `[1:89-95]`, security `[1:116-122]`, modifiability `[1:132-138]`, interoperability `[1:147-153]`, testability `[1:164-170]`.
✍ *How to use in an answer:* walk the 7 categories for the scenario — it shows completeness. Example (availability): allocation → add responsibility to log/notify faults; coordination → guaranteed delivery? degraded comms?; data model → which data can be disabled/rolled back; mapping → can elements be remapped for recovery; resources → enough left to log+notify; binding → late binding must also be covered by the fault strategy; technology → does the tech itself fail?

---

# C. THE SEVEN QUALITY ATTRIBUTES (+ OTHERS)

> Pattern for each: **definition → general scenario → tactic tree → trade-offs → one-line example**.

## C1. Availability `[1:44-65]`
- **What:** the system is there and ready when needed; builds on reliability by adding **recovery/repair**; fundamentally **minimise outage time by mitigating faults**. `[1:44]`
- **Fault vs failure:** a **failure** = service no longer consistent with spec, *observable by actors*; a **fault** can cause a failure. Tactics keep faults from becoming failures or bound their effect. `[1:47]`
- **Scenario:** source (internal/external: hardware, software, infra, environment) · stimulus **fault**: *omission, crash, incorrect timing, incorrect response* · artifact (processors, channels, storage, processes) · environment (normal, startup, shutdown, repair, degraded, overloaded) · response (**prevent** failure; **detect**: log + notify; **recover**: disable source, be temporarily unavailable, fix/mask, degrade) · measure (availability %, time to detect, time to repair, degraded-mode interval, % faults handled). `[1:45]`
- 🆕 *Quick numbers:* 99.9% ≈ 8.8 h downtime/yr; 99.99% ≈ 53 min/yr; 99.999% ≈ 5 min/yr. Availability ≈ MTBF/(MTBF+MTTR) → you can raise it by failing less *or* repairing faster.

| Category | Tactics | One-line meaning `[1:49-57]` |
|---|---|---|
| **Detect faults** | Ping/echo · Monitor · Heartbeat | reachability + round-trip; health watcher; periodic "I'm alive" |
| | Timestamp · Sanity checking · Condition monitoring | wrong event order; reasonableness of output; validate design assumptions |
| | Voting · Exception detection · Self-test | replicas compare results; timeout/parameter fence; component tests itself |
| **Recover — prepare & repair** | **Active redundancy (hot spare)** | all nodes process identical input; spare in sync → fastest failover, highest cost |
| | **Passive redundancy (warm spare)** | only active processes; periodic state updates to spare → slower, cheaper |
| | **Spare (cold)** | powered off; reset on failover → cheapest, slowest |
| | Exception handling · Rollback ("rollback line") · Software upgrade (in-service) · Retry (transient) · Ignore faulty behaviour · **Degradation** (drop less-critical functions) · **Reconfiguration** (reassign to remaining resources) | |
| **Recover — reintroduction** | Shadow · State resynchronisation · Escalating restart · Non-stop forwarding | run failed unit in shadow before reinstating; send state active→standby; restart at increasing granularity; supervisory/data split (router) |
| **Prevent faults** | Removal from service · Transactions (ACID) · Predictive model · Exception prevention · Increase competence set | planned reboot; atomic updates; monitor health predictively; wrappers/ADTs; handle more cases normally |

- **Trade-offs:** redundancy ↑ availability ↓ cost; ping/echo ↑ detection ↓ performance + DoS risk; all availability tactics involve the **coordination model**. `[1:65]`
- ✍ *Skeleton:* fault model (omission/crash/timing/response) → detect → recover → prevent → measure (99.99%, detect <1 s) → cost.

## C2. Performance `[1:66-81]`
- **What:** about **time** — ability to respond to events (interrupts, messages, requests, clock ticks) within timing requirements. Measured by **latency and throughput** (also jitter/miss-rate 🆕). `[1:66, 1:81]`
- **Scenario:** source (internal/external) · stimulus (events arriving — periodic/sporadic/stochastic 🆕) · artifact (system) · environment (normal/overload) · response (process events; change service level) · measure (latency, deadline, throughput, jitter). `[1:67]`
- **Idea:** events arrive; resources are consumed; **latency = processing time + blocked time (contention, resource unavailable, dependency)** 🆕. So tactics either **reduce demand** or **manage resources**.

| Category | Tactics `[1:70-73]` |
|---|---|
| **Control resource demand** | Manage sampling rate (↓ fidelity) · Limit event response (max rate) · Prioritise events · Reduce overhead (remove intermediaries ↔ modifiability) · Bound execution times · Increase resource efficiency (better algorithms) |
| **Manage resources** | Increase resources (faster CPU/more memory/network) · Increase concurrency · Maintain multiple copies of computations (replicas reduce contention) · Maintain multiple copies of data (caching) · Bound queue sizes · Schedule resources |

- **Trade-offs:** removing intermediaries speeds things but **hurts modifiability**; more resources cost money; concurrency adds complexity/non-determinism; sampling-rate reduction loses fidelity.
- ✍ *Example:* "Aadhaar registration screen slow (large State/District/Town drop-downs)" → *cache static lists server-side (multiple copies of data) + AJAX incremental load (reduce demand)*. `[2:50-51]`

## C3. Usability `[1:82-95]`
- **What:** how easy it is to accomplish a task + user support. Covers: learning features, using efficiently, **minimising impact of errors**, adapting to user needs, confidence & satisfaction. Cheap, effective way to improve perceived quality. `[1:82-83]`
- **Scenario measures:** task time, #errors, #tasks done, satisfaction, ratio successful operations, data/time lost on error. `[1:84]`
- **Tactics `[1:87-88]`:** *Support user initiative* — **Cancel** (listen, terminate, free resources, inform collaborators), **Pause/Resume**, **Undo** (keep state to restore), **Aggregate** (apply to group). *Support system initiative* — **Maintain task model** (infer what user attempts), **user model** (knowledge, expected response), **system model** (expected behaviour to give feedback).
- **Cost:** undo needs stored state (memory/consistency); user models add complexity & privacy concerns.

## C4. Security `[1:106-123]`
- **What:** protect data and information from unauthorised access while still giving access to authorised people/systems. An **attack** = action intended to harm (unauthorised access/modification, or denial of service). `[1:106]`
- **CIA:** **Confidentiality**, **Integrity**, **Availability**; supported by **authentication, authorisation, non-repudiation**. `[1:107]`
- **Scenario:** source (human/system, inside/outside, known/unknown) · stimulus (unauthorised attempt to display/change/delete data, access services, reduce availability) · artifact (services, data, components) · environment (online/offline, behind firewall…) · response (authenticate, grant/deny, record, encrypt, notify) · measure (time to detect/recover, damage extent). `[1:108]`
- *Concrete:* disgruntled employee tries to modify pay-rate table; audit trail kept; data restored within a day. `[1:109]`
- **Mental model:** physical security → **detect, resist, react, recover**. `[1:110]`

| Category | Tactics `[1:111-115]` |
|---|---|
| **Detect** | Detect intrusion (signatures) · Detect service denial (traffic vs DoS profiles) · Verify message integrity (checksum/hash) · Detect message delay |
| **Resist** | Identify actors · Authenticate · Authorise · Limit access · Limit exposure (minimise **attack surface**) · Encrypt data · Separate entities (VM, air-gap) · Change default settings |
| **React** | Revoke access · Lock computer (after repeated failures) · Inform actors |
| **Recover** | Availability recovery tactics + **Audit** (trace and identify the attacker) |

- **Key point:** *no tactic is foolproof* → layered defence: resist + detect + react + recover. `[1:123]`
- **Trade-off:** security vs performance/usability (encryption overhead, extra logins).

## C5. Modifiability `[1:124-139]`
- **What:** cost and risk of making changes. Three questions: **what can change? how likely? when and by whom?** `[1:124]`
- **Scenario:** source (end user, developer, admin) · stimulus (add/delete/modify function, change QA, capacity, technology) · artifact (code, data, interfaces, config) · environment (design, compile, build, initiation, runtime) · response (make, test, deploy modification) · measure (cost: #artifacts, effort, calendar time, money, side-effects on other functions/QAs). `[1:125]`
- **Idea:** cost of change ≈ (size of module) × (coupling) ÷ cohesion + late-binding flexibility.

| Goal | Tactics `[1:127-131]` |
|---|---|
| **Reduce size of module** | Split module |
| **Increase cohesion** | Increase semantic coherence (responsibilities that don't serve the same purpose go to different modules) |
| **Reduce coupling** | Encapsulate (explicit interface) · **Use an intermediary** · Restrict dependencies · Refactor (duplicates) · Abstract common services |
| **Defer binding** | Later binding = cheaper change *but* the mechanism itself costs more to build |

- **Trade-off:** intermediaries/late binding ↓ performance and ↑ complexity; over-splitting increases communication.
- ✍ *Example:* "Hospital increases fee for a service; configuration team changes it in 1 working day, no code change" → **defer binding (configuration/rules engine)**. `[2:14]`

## C6. Interoperability `[1:140-154]`
- **What:** degree to which two or more systems can **usefully exchange meaningful information** (not yes/no — shades). `[1:140]`
- **Goal of tactics:** systems must **know about each other** (→ *Locate*) and **exchange information in a semantically meaningful way** — correct sequence + information translated to an acceptable form (→ *Manage interfaces*). `[1:143]`
- **Tactics:** **Locate** — *discover service* (directory lookup, possibly multiple indirection); **Manage interfaces** — *orchestrate* (control mechanism sequences service calls for complex tasks) · *tailor interface* (add/remove capabilities: translation, buffering, smoothing). `[1:145-146]`
- **Cost/risks:** security exposure, load from external callers (flood can deny service to legit users), versioning. Checklist stresses data-model syntax/semantics and binding policy for unknown systems. `[1:147-153]`
- 🆕 *Syntactic* = can parse the data; *semantic* = agree on its meaning.

## C7. Testability `[1:155-171]`
- **What:** ease with which software can be made to **demonstrate its faults** through testing — probability that, if a fault exists, it **fails on the next test**. Needs **controllability** (set inputs/state) and **observability** (see outputs/state). `[1:155-156]`
- **Tactics `[1:160-163]`:** *Control & observe state* — Specialised interfaces · Record/playback · Localise state storage · Abstract data sources (swap test/prod data) · Sandbox · Executable assertions. *Limit complexity* — Limit structural complexity (cycles, external dependencies) · **Limit non-determinism** (unconstrained parallelism).
- ✍ *Connection to architecture* `[2:84-87]`: utility tree → prioritise test cases; module dependencies → integration test plan; design for test (switchable data sources, rollback of test changes, simulators for payment gateway/sensors/Aadhaar).

## C8. Other quality attributes `[1:96-105]`
- **Variability** (planned variants — special modifiability); **Portability** (run on another platform); **Development distributability** (support distributed teams) `[1:96]`
- **Scalability:** *horizontal* = add servers to a cluster (scale out); *vertical* = add resources to one machine (scale up). **Deployability** (how executable reaches/starts on a host). **Mobility** (size/display/bandwidth/battery). `[1:97]`
- **Monitorability** (ops can observe at runtime); **Safety** (avoid states that cause harm; same concerns as availability: prevent/detect/recover) `[1:98]`
- **Conceptual integrity** (same thing done the same way); **Marketability** (architecture itself sells: "cloud-based", "SOA"); **Quality in use** — effectiveness, efficiency, freedom from risk `[1:99-100]`
- **Standard lists:** ISO/IEC 25010 product-quality `[1:101-103]`.
- **Unknown "X-ability" (e.g. green computing):** ① model the attribute ② assemble tactics ③ build design checklists. `[1:104]`

---

# D. ARCHITECTURALLY SIGNIFICANT REQUIREMENTS (ASR) & DESIGN STRATEGY

## D1. ASR `[2:1-4]`
- **What:** a requirement with a **profound, measurable effect on the architecture** — the architecture would be dramatically different without it. Subset of requirements (software *and* hardware).
- **Why:** ASRs determine the overall structure; designing for everything is impossible, so you **find the few that drive the design** and spend effort there.
- **Indicators `[2:2]`:** high business value or technical risk · important to an influential stakeholder · **unique** (no existing component handles it) · QoS/SLA differs from what the evolving architecture accommodates · previously caused budget overrun/dissatisfaction.
- **Drivers `[2:3]`:** business goals & constraints (uptime, horizontal scaling) · stakeholder concerns (security over performance) · quality attributes · regulatory/compliance (GDPR, HIPAA) · technology constraints · cost/budget · time to market.
- **Typical ASRs `[2:4]`:** audit trail, alerts/notifications, variety of data-input mechanisms, third-party interaction, workflow, user-behaviour analytics, responsive web/mobile UI, business-maintained rules. *Probe each in detail with scenarios before deciding design.*

## D2. Five ways to find ASRs `[2:5]`
| Source | How `[page]` |
|---|---|
| **Requirements documents / user stories** | MoSCoW or stories; check QAs appear in equal proportion; absent QAs make the architect's job hard `[2:6]` |
| **Leads from design decisions** | Use the 7 categories as probes: e.g. allocation → planned evolution of responsibilities, user roles, modes; coordination → protocols, middleware, security; data model → major entities, access patterns `[2:7]` |
| **Stakeholder interviews — Quality Attribute Workshop (QAW)** | 7 steps: presentation & intro → business mission → architectural plan → identify drivers → scenario brainstorming → consolidation → prioritisation; output = list of architectural drivers `[2:8-9]` |
| **Business goals** | Goals lead to QAs, or affect architecture directly, or have no influence. Write as **"X to Y by when"** (current → future state). Categories: growth/continuity, financial, personal, responsibility to employees `[2:10-11]` |
| **Utility tree** | See D3 `[2:12-14]` |

## D3. Utility tree `[2:12-14]`
- **Structure:** Root *utility* → **quality attributes** (performance, availability, modifiability…) → **attribute refinements** (e.g. transaction response time, throughput) → **ASRs written as scenarios**, each tagged **(Importance, Difficulty) = (H/M/L, H/M/L)**.
- **Reading the tags:** (H,H) scenarios = highest priority — important to business *and* architecturally hard. E.g. *"A user updates a patient's account under peak load; completes in <0.75 s"* **(H,M)**; *"150 normalised transactions/s at peak"* **(M,M)**; *"fee change by configuration in 1 day, no code change"* **(H,L)**. `[2:13-14]`
- **Used again** in ATAM step 5 `[2:70]` and to prioritise test cases `[2:85]`.
- ✍ *Skeleton:* identify QAs from business goals → refine each → write measurable scenarios → rate (H,M,L)×(H,M,L) → design for the top-right first.

## D4. Design strategy — Attribute-Driven Design (ADD) ⚠ NOT-CARRIED in detail (`[1:173]` has only the heading)
🆕 **ADD** = iterative, ASR-driven decomposition: ① pick the element to design ② identify its ASRs ③ choose design concepts (patterns/tactics) that satisfy them ④ instantiate elements, allocate responsibilities, define interfaces ⑤ sketch views and record decisions ⑥ analyse (are ASRs met?) ⑦ iterate on the next element/ASR. *Generate-and-test:* evaluation by the designer is the "test" half `[2:57]`.

---

# E. ARCHITECTURE & AGILITY `[1:172-187]`

- **Why agile:** respond to stakeholders, deliver working functionality sooner, show progress early, avoid documenting what will change. **These needs do not conflict with architecture.** The question is not "agile *or* architecture" but **how much architecture up-front vs deferred, and how much to document, and when.** `[1:174-175]`
- **Manifesto:** individuals/interactions, working software, customer collaboration, responding to change — valued *over* processes/tools, documentation, contracts, plans. 12 principles (early/continuous delivery, welcome change, frequent delivery, daily business-developer collaboration…). `[1:176-177]`
- **How much architecture? — the "sweet spot" `[1:178-180, 2:15]`:** two things add time: **up-front architecture/risk work** and **rework** (defects, modification requests). They trade off. Boehm & Turner: **10 KSLOC** → sweet spot far left (little up-front); **100 KSLOC** → ~**20%** of schedule; **1000 KSLOC** → ~**40%**. *A million lines cannot be steered by agile principles alone.*
- **Documentation:** *write for the reader; if the reader doesn't need it, don't write it* — but a maintainer/newcomer may be the reader. `[1:181]`
- **Evaluation in agile:** ATAM focuses on the *most important* QA scenarios, so it can be tailored and lightweight. `[1:182]`
- **Agile architecting example (WebArrow):** work **top-down** (structures for QAs) and **bottom-up** (implementation/environment constraints) simultaneously; use **spikes/experiments** to answer trade-off questions (distributed DB vs flat files → latency? mod_perl → scalability? how many participants per meeting server?). `[1:183-184]`
- **Incremental Commitment Model (Boehm):** stakeholder commitment, satisficing, incremental growth, iterative development, interleaved definition/development, **risk-driven milestones**. `[1:185]`
- **Authors' advice `[1:186-187]`:** large, complex, stable requirements or distributed development → **lots of up-front architecture pays off**; large with unstable requirements → quickly design a *candidate* architecture, elaborate later; small/uncertain → agree only the **major patterns**. Agile architects take a middle ground and **refactor when technical debt gets too high**.

---

# F. DOCUMENTATION & VIEWS

## F1. Structures vs views, and what to document `[2:16-20]`
- View = representation of a structure for stakeholders. **Document only what communicates the design and mitigates risk** `[2:31]`.
- Table linking structure → relations → useful for: `[2:17-19]` (decomposition, uses, layered, class, client-server, process, concurrency, shared data, deployment, implementation, work assignment).

## F2. Kruchten 4+1 model `[2:21-31]`
- **Problem solved:** one diagram can't serve all stakeholders (end-user, system engineer, developer, project manager). **Solution: several views, each with its own notation, each addressing one set of concerns.** `[2:21]`
| View | Viewer | Considers | SEI category `[2:30]` |
|---|---|---|---|
| **Logical** `[2:23]` | End-user | functional requirements (services provided) — object model | Module |
| **Process** `[2:24]` | Integrators | non-functional: scalability, concurrency, performance — process decomposition | C&C |
| **Development** `[2:25]` | Programmers, managers | static organisation of modules/layers, reuse, tool constraints | Allocation (to dev environment) |
| **Physical** `[2:26]` | System engineers | mapping software to hardware; reliability, availability, performance (test vs deployment architecture) | Allocation (deployment) |
| **+1 Scenarios** `[2:27]` | All users/evaluators | puts the 4 views together; checks consistency and validity | — |
- **Correspondence:** start at Logical → Development/Process → Physical; the larger the project the larger the gap between logical and development (group by team, class categories, LOC). `[2:28-29]`
- **Not every project needs every view** — choose by dominant concern `[2:31]`: complex UI → Logical; high traffic/performance → Process; large distributed team → Development; cloud-native/complex security → Physical.
- Companion documents: Software Architecture Document (follows 4+1) + design guidelines (key decisions that must be respected). `[2:29]`

---

# G. LAYERED ARCHITECTURE — TECHNIQUES BY LAYER `[2:32-55]`

- **Why layers:** separation of concerns → maintainability, scalability; clearly defined functional layers, **loose coupling, reusable lower layers, exchangeable parts**. `[2:32-33]`
- **Layers:** Presentation (UI/UX) · Business/Logic (rules, validation, "heart") · **Service** (APIs/orchestration to external consumers) · Data access.

| Layer | Techniques & why `[page]` |
|---|---|
| **Presentation** `[2:34-36]` | Cache frequently used data — **client-side** (browser: preferences, address) vs **server-side** (shared: product codes, promotions); usually *both*. **AJAX** (async update of page parts, no full reload). **Responsive design** (different form factors). |
| **Business** `[2:37-42]` | **Application façade** (unified high-level interface hiding subsystems). **Session management** (client-based cookies vs server-based persistent). **Workflow/rules engine** (modifiability — engine invokes next activity, manual or automated). **Aspect-oriented design** for cross-cutting concerns (logging, auditing, caching, security). **Logging vs instrumentation** (log actionable errors only — logging is expensive; instrument request rate, error rate, duration, queue length). |
| **Data** `[2:43-45]` | Connection pool · multiple copies of data · transactions (atomicity) · **ORM** (Hibernate) · stored procedures · **parameterised SQL to prevent SQL injection** (`OR 'x'='x'` returns all rows if you concatenate input). |
| **Service** `[2:46-47]` | Provides services to external systems (e.g. Get Balance, Request cheque-book). Must handle **duplicate requests, out-of-sequence messages, communication failure (retry or queue-and-send-later)**. |
- ✍ **Exercises with model answers:** DCS layer components `[2:48-49]` · Aadhaar drop-downs → cache + AJAX `[2:50-51]` · hotel system for Makemytrip → *service layer* with *Get room availability* & *Reserve room* `[2:52-53]` · logistics (container + transporter + ship) → *façade in business layer* `[2:54-55]`. (More in `02_Scenario_Based.md`.)

---

# H. EVALUATION, CONFORMANCE, TESTING, RECONSTRUCTION

## H1. Why and how to evaluate `[2:56-60]`
- "Fail early" — find risks and **trade-offs** before building (e.g. security ↑ often performance ↓).
- **Three forms:** **by the designer** (the "test" in generate-and-test; depth depends on decision importance, number of alternatives) `[2:57]`; **peer review** (scenarios → architect presents → walk-through per scenario; can be lightweight, repeated) `[2:58]`; **by outsiders** (objective, specialised, managers listen more; usually complete architectures) `[2:59]`.
- **Contextual factors:** artifacts available? who sees results? who evaluates? which stakeholders participate? `[2:60]`

## H2. ATAM — Architecture Tradeoff Analysis Method `[2:61-77]`
- **What/why:** evaluators needn't know the architecture or business goals, the system needn't be built, many stakeholders involved; it surfaces **risks, sensitivity points and trade-offs** for the **most important QA scenarios** — it does *not* analyse the whole architecture. `[2:61]`
- **Participants:** evaluation team (3-5 external, unbiased), project decision-makers (manager, customer), architect (always), stakeholders. `[2:62]`
- **Phases `[2:65]`:** **0** partnership & preparation · **1** steps 1-6 (evaluation team + decision-makers, 1-2 days) · **2** steps 7-9 (with wider stakeholders) · **3** follow-up/report.
- **Nine steps:** ① present ATAM `[2:66]` ② present **business drivers** `[2:67]` ③ present **architecture** `[2:68]` ④ identify **architectural approaches** (patterns/tactics) `[2:69]` ⑤ generate **utility tree** `[2:70]` ⑥ **analyse approaches** against top scenarios — record decisions, risks, non-risks, sensitivity points, trade-offs `[2:71]` ⑦ **brainstorm & prioritise scenarios** (maintainer → modifiability, user → function, QA person → testing) `[2:72]` ⑧ analyse again with new top scenarios (5-10) `[2:73]` ⑨ **present results**; group risks into **risk themes** `[2:74]`.
- **Outputs `[2:63-64]`:** concise architecture presentation (1 h) · business goals · **prioritised QA scenarios** · **risks and non-risks** (risk = decision that may cause undesirable consequences) · **risk themes** · mapping of decisions to quality requirements · **sensitivity points** · **trade-off points**.
- 🆕 *Definitions:* **sensitivity point** = property of a component critical to achieving a QA response; **trade-off point** = sensitivity point affecting more than one QA. *Example risk:* heartbeat frequency affects how fast a failed component is detected `[2:71]`.
- **Cost:** 20-30 person-days for the evaluation team plus architect/stakeholders → only for large, costly projects. **Lightweight ATAM:** half-day to 1 day, internal team, participants already know the system; typical 4-6 h agenda. `[2:75-76]`
- **Rule:** if the architecture is important enough to design explicitly, it should be evaluated; extent varies. `[2:77]`

## H3. Conformance — keeping code aligned with architecture `[2:78-83]`
- **Drift examples:** skipping layers; bypassing the data-access layer; notifying modules one by one instead of publish-subscribe; not using the common logging mechanism. `[2:78]`
- **Techniques `[2:79-83]`:** ① **architecturally evident coding style** (mark layer, publisher/subscriber, MQ insert/retrieve in the code) ② **frameworks** (Spring, Hibernate, AUTOSAR) ③ **code templates** (e.g. fault-tolerant template: get event → process → send state to backup) ④ **update architecture documentation** with code changes; mark obsolete parts "no longer applicable"; sync at release. *Extra:* educate team, code reviews, folders per layer/service/UI.

## H4. Architecture & testing `[2:84-87]`
Prioritise tests using the **utility tree** (high business value × high architectural impact); build the **integration test plan** from module interactions and dependencies; design for testability (switch data sources, roll back test changes, replaceable components/simulators).

## H5. Architecture reconstruction `[2:88-92]`
- **Why:** no documentation exists; **migrate** old → new technology (mainframe → web); identify reusable components (logging, security). `[2:88]`
- **Phases `[2:89-92]`:** **raw view extraction** (source code, execution traces, build scripts → classes, files, caller–callee, global data) → **database construction** (common format) → **view fusion** (static + dynamic/trace + expert grouping into layers) → **architecture analysis** (check conformance: layer calls adjacent layers only; DB access only through entity bean).

---

# I. PATTERNS CATALOGUE `[3:1-114]`

## I0. What a pattern is `[3:1-4]`
- **Pattern = context + problem + solution.** Describes a recurring design problem in a context and a **well-proven generic scheme** (components, responsibilities, relationships, collaboration). Gives common vocabulary, documents design experience, manages complexity. `[3:1-3]`
- **Categories (POSA):** *From Mud to Structure* (Layers, Pipes & Filters, Blackboard) · *Distributed systems* (Broker) · *Interactive systems* (MVC, PAC) · *Adaptable systems* (Microkernel, Reflection). Plus SEI C&C/allocation patterns (SOA, client-server, P2P, publish-subscribe, shared-data, map-reduce, multi-tier). `[3:4]`
- **Patterns vs tactics:** a pattern is a molecule, a tactic an atom; patterns are *under-specified for a real system* and need tactics added; e.g. Broker has bottleneck + single point of failure → add *increase resources* and *multiple copies*. **Every tactic introduces new concerns** (ping/echo → DoS risk, overhead, modifiability) which need further tactics — it ends when side-effects are small enough to ignore. `[3:106-114]`

## I1. Layers `[3:5-12]`
- **Context/Problem:** parts of a system must be developed and evolved independently → segment so modules interact little (portability, modifiability, reuse).
- **Solution:** layers = groupings of modules offering a **cohesive set of services**; **allowed-to-use is unidirectional**; each exposed through a public interface. `[3:7-8]`
- **Implementation steps `[3:9]`:** define abstraction criterion → number of levels → name layers/assign tasks → specify services → refine → interface per layer → structure layers → specify adjacent communication → decouple/error handling.
- **Variants:** **Relaxed layered** (a layer may use *all* lower layers; partially opaque layers); **Layering through inheritance**. `[3:10-11]`
- **Benefits:** reuse of layers; standardisation; dependencies kept local. **Liabilities:** cascading changes; lower efficiency; unnecessary work; hard to choose right granularity. `[3:12]`

## I2. Pipes & Filters `[3:13-20]`
- **Context:** systems transform **streams of data items**; transformations recur → independent reusable parts.
- **Solution:** filters (consume/deliver data **incrementally**) connected by pipes; output of one step is input to the next; non-adjacent steps share no information. `[3:15-16]`
- **Steps:** divide into stages → define data format on each pipe → implement pipes → design filters → error handling → set up the pipeline. `[3:17]` **Variant:** tee-and-join (graph, even with feedback loops). `[3:18]`
- **Benefits:** no intermediate files needed, flexible by exchange/recombination, reuse, rapid prototyping, parallelism. **Liabilities:** sharing state is expensive/inflexible; parallel efficiency gain often an illusion; data-transformation overhead; **error handling is hard**. `[3:19-20]`

## I3. Blackboard `[3:21-28]`
- **Context/Problem:** immature domain with **no deterministic solution** (vision, speech recognition, surveillance); problem decomposes into subproblems needing different expertise.
- **Solution:** independent **knowledge sources** work cooperatively on a shared **blackboard** (central data store with a *vocabulary*) under a **control component**; knowledge sources don't call each other and have no fixed sequence. `[3:23-25]`
- **Steps:** define problem → solution space → steps → specialised knowledge sources → blackboard vocabulary → control. `[3:26]` **Variants:** production system (condition–action rules, working memory); **repository** (no internal control — e.g. a database). `[3:27-28]`
- **Use for:** approximate/partial solutions assembled incrementally. **Cost:** control complexity, hard to test, no guaranteed solution (liability slide not in carried notes — state this from understanding).

## I4. Broker `[3:29-43]`
- **Context/Problem:** distributed systems built from decoupled components that interact by **remote service invocation**; components must find each other and communicate without hard-wired links. `[3:29-30]`
- **Solution:** a **broker** coordinates — servers register; clients send requests via the broker, which **locates the server, forwards requests, and returns results/exceptions**. Makes **distribution transparent**. `[3:31-32]`
- **Six components `[3:33-39]`:** **clients**, **servers** (may also act as clients), **broker** (locate receiver by system identifier; API for register/forward), **client-side proxies** (remote object looks local; hide IPC and marshalling), **server-side proxies** (receive, unpack, marshal parameters, call service), **bridges** (optional; hide details when two brokers interoperate across heterogeneous networks).
- **Implementation:** define object model → interoperability style → APIs → proxies → broker design → IDL/compilers `[3:40]`. **Variants:** direct-communication, message-passing, trader, adapter, callback broker. `[3:41]`
- **Benefits:** location transparency; changeability/extensibility; portability; interoperability between brokers; reusability. **Liabilities:** restricted efficiency; lower fault tolerance (broker = single point of failure); harder testing/debugging. `[3:42-43]` → *fix with tactics: increase resources, multiple copies* `[3:107]`.

## I5. Model-View-Controller (MVC) `[3:44-55]`
- **Problem:** UIs change most often (new menus, different look-and-feel, porting, same data shown in many forms, immediate reflection of data changes, run-time change). `[3:45-46]`
- **Solution:** three areas — **Model** (core data & functionality, independent of output), **View** (displays info obtained from the model; many views per model), **Controller** (each view has one; translates input events into service requests). **Change-propagation:** model notifies all views, views refetch and redraw. `[3:47-51]`
- **Steps:** separate HCI from core → change propagation → views → controllers → view-controller relation → set-up → dynamic views → pluggable controllers… `[3:52]` **Variant:** **Document-View** (merges view + controller, loses controller exchangeability). `[3:53]`
- **Benefits:** multiple synchronised views; pluggable views/controllers; exchangeable look-and-feel; framework potential. **Liabilities:** increased complexity; excessive updates; tight view–controller coupling and coupling to model; inefficient data access in view; change when porting; poor fit with modern UI toolkits. `[3:54-55]`
- 🆕 Tactics inside MVC: increase semantic coherence, encapsulation, use an intermediary, runtime binding `[3:106]`.

## I6. Presentation–Abstraction–Control (PAC) `[3:56-65]`
- **Idea:** interactive system = **hierarchy of cooperating agents**, each with **Presentation, Abstraction (data/functionality), Control (communication with other agents & between P and A)**. `[3:56-59]`
- **Levels:** **top-level** agent = functional core + global UI (menu bar); **intermediate** = combine/coordinate lower agents (e.g. floor-plan + external view in CAD); **bottom-level** = self-contained semantic concepts (chart, spreadsheet) incl. zoom/move. `[3:60-62]`
- **Benefits:** separation of concerns, support for change/extension, **multi-tasking** (agents run concurrently/distributed). **Liabilities:** increased complexity, **complex control component**, applicability limited. `[3:64-65]`

## I7. Service-Oriented Architecture (SOA) `[3:66-69]`
- **Context/Problem:** providers offer described services; consumers must use them **without knowing implementation** → interoperability of distributed components on different platforms.
- **Elements:** service providers (published interfaces), service consumers, intermediaries (ESB, registry, orchestration server). **Connectors:** **SOAP** (sync, HTTP), **REST** (HTTP request/reply), **asynchronous messaging** (point-to-point/pub-sub). `[3:67-68]`
- **Weaknesses:** complex to build; you don't control service evolution; performance overhead. `[3:69]`

## I8. Microkernel `[3:70-80]`
- **Context:** systems that must **adapt to changing requirements**; separate a **minimal functional core** from extended and customer-specific parts; the kernel is the "socket" for plug-ins. `[3:70-71]`
- **Five components `[3:72-76]`:** **microkernel** (atomic services = *mechanisms*; resources; communication) · **internal servers/subsystems** (extend kernel, accessed only via kernel) · **external servers/"personalities"** (implement policies for a domain view) · **adapters/emulators** (hide kernel from clients) · **clients**.
- **Key idea:** **separate mechanism from policy.** **Variant:** distributed microkernel. `[3:73, 3:78]`
- **Benefits:** portability, flexibility/extensibility, separation of policy and mechanism, scalability, reliability, transparency. **Liabilities:** performance, complexity of design/implementation. `[3:79-80]`

## I9. Reflection `[3:81-90]`
- **Problem:** systems evolve; can't foresee all requirements → architecture open to modification.
- **Solution:** make software **self-aware** — **meta level** (metaobjects = type structures, algorithms, call mechanisms) + **base level** (application logic, uses metaobjects); changes via the **metaobject protocol (MOP)**, which also validates changes. `[3:83-87]`
- **Benefits:** no explicit source modification; easy to change; many kinds of change. **Liabilities:** meta-level changes can cause damage; more components; lower efficiency; not all changes/languages supported. `[3:89-90]`

## I10. SEI component-and-connector / allocation patterns `[3:91-105]`
| Pattern | Context → solution | Weaknesses / notes |
|---|---|---|
| **Client-Server** `[3:91-92]` | shared resources/services for many clients; clients initiate request/reply; factor out common services (modifiability, reuse, scalability, availability via central control) | server = bottleneck/single point of failure |
| **Peer-to-Peer** `[3:93-95]` | equal entities provide and consume services over a common protocol; constraints on attachments, hops, who knows whom, supernode topologies | security, consistency, availability, backup/recovery harder |
| **Publish-Subscribe** `[3:96-98]` | producers/consumers unaware of each other; components announce/listen via event distributor | ↑ latency, ↓ scalability/predictability; no ordering/delivery guarantee |
| **Shared-Data** `[3:99-100]` | components exchange **persistent data** via shared data store (database) | store is bottleneck/single point of failure; accessors tightly coupled to data schema |
| **Map-Reduce** `[3:101-103]` | analyse **petabyte-scale** data: **map** (stateless, parallel extract-transform) → shuffle/sort → **reduce** (summarise/load); data in files; infrastructure instantiates/monitors | batch only, requires restructuring problem as map/reduce |
| **Multi-Tier** `[3:104-105]` | split deployment into **tiers** (logical groupings of components allocated to platforms; "is part of", "communicates with", "allocated to") | latency between tiers, cost of more hardware |

# J. CLOUD ARCHITECTURE `[3:115-139, 4:18-29]`
## J1. Cloud basics (NIST) `[3:115-119]`
- **Essential characteristics:** on-demand self-service · ubiquitous network access · resource pooling · rapid elasticity · measured service.
- **Service models `[3:116]`:** **SaaS** (consumer = end user; apps running on cloud), **PaaS** (consumer = developer/admin; deploy on provider's languages/tools; e.g. LAMP stack, provider manages lower layers `[3:125]`), **IaaS** (consumer = developer/admin controlling OS, storage, deployed apps).
- **Deployment models `[3:117]`:** private · public · community · hybrid.
- **Why it pays:** virtualisation raises server utilisation; random access and time-of-day variation smooth the load `[3:118]`; **multi-tenancy** (one app serves many consumers: cheaper help desk, upgrade once, single version) `[3:119]`.
## J2. Inside IaaS & storage `[3:120-129]`
- **IaaS:** servers in clusters, each with a hypervisor; components: **Cluster Manager, Persistent Object Manager, Virtual Resource Manager (gateway), File System Manager (like HDFS)**; services: automatic IP re-allocation on VM failure, **auto-scaling**. `[3:122-124]`
- **HDFS write (sunny day) `[3:120]`:** client buffers a 64K block → asks NameNode → gets 3 DataNodes → sends to first, which pipelines to second and third → each reports completion → client confirms. **Failures `[3:121]`:** client fails → app retries (write not complete until committed); **NameNode** fails → backup takes over, log file, DataNodes hold true block list, client retries; **DataNode** fails → client/previous node asks NameNode for a different node; 3× replication.
- **Why NoSQL (key-value, document) `[3:126-129]`:** big sequential web data, relational overhead, CAP theorem, model mismatch. **Omitted:** transactions (no locking), schemas, consistency (→ eventual), normalisation/joins.
## J3. Architecting for the cloud `[3:130-133]`
- **Security:** multi-tenant → inadvertent info sharing, **VM escape**, side-channel attacks, DoS. **Performance:** auto-scaling response may be too slow → make apps self-aware, proactive. **Availability:** **failure is common** with 1000s of servers → developers must assume instances fail and build detection/correction.
- **AWS design principles `[3:133]`:** scalability · disposable resources instead of fixed servers · automation · loose coupling · services not servers · right database · remove single points of failure · optimise for cost · caching · security.
- **Multi-tenant case (Azure, Tailspin) `[3:134, 4:21-26]`:** see `04_Case_Studies.md`.
## J4. CAP theorem `[3:135-139]`
Only **two of three**: **C**onsistency (all nodes see same data), **A**vailability (survivors keep operating), **P**artition tolerance. **AP** (return older version — CouchDB, Cassandra, ScyllaDB) · **CP** (lock/unavailable until consistent — MongoDB, Redis) · **CA** only theoretical (no partition tolerance). In practice **partitions happen → choose C or A *per service*** — Amazon: shopping cart = **AP** (revenue; errors hidden and sorted out later), order submission = **C** (credit-card, shipping, reporting all read the data). `[4:18]`
## J5. Cloud issues & cloud-native `[4:19, 4:24-27]`
- 99.95% availability still means **0.05% downtime → design for it** (e.g. Netflix stores data across geographic zones). **Multi-tenancy risks:** shared table with org-ID column (poor design → leakage), VM escape.
- **Multi-tenancy degree `[4:25-26]`:** highest (IaaS, PaaS, SaaS all multi-tenant) · middle (small SaaS clusters) · lowest (SaaS single-tenant). Higher degrees suit **utilitarian, cross-industry workloads** (email, expense reporting, travel approval, sales-force) that can share a schema; strategic/volatile/secure workloads need less sharing.
- **Cloud-native:** PaaS, multicloud, microservices, agile, containers, CI/CD, DevOps. `[4:27]`

# K. MOBILE ARCHITECTURE `[4:1-17, 4:30]`
- **Design considerations `[4:1]`:** simple UI (large buttons, few options); **responsive design**; **compact code** (CPU, memory, storage); **few layers** for performance; **connectivity — store locally, sync later**.
- **App types `[4:2-5]`:** **Native** (iOS Swift/Obj-C; Android Java/Kotlin; best performance & UX, all OS features; *one platform only*, special skills, separate versions) · **Mobile web/PWA** (HTML5/CSS3/JS in browser; single code base, no install/store; *lower UX, slower, limited hardware*) · **Hybrid** (web app inside native shell; cross-platform via Xamarin/React Native; hardware access; *middle on performance/UX*). Also *cross-platform*, *mobile web app with server component* `[4:30]`.
- **Responsive technique `[4:6]`:** CSS/HTML hide/shrink/move content; flexible grids with media queries (% widths); flexible images.
- **Store locally, sync later `[4:7]`:** doctors enter patient data on a mobile (SQLite) and it syncs to the server (REST) later.
- **Android components `[4:8-11]`:** **Activity** (UI) · **Service** (background: music, download) · **Content provider** (storage: SQLite/files) · **Broadcast receiver** (reacts to OS/app events: SMS arrival, high-pulse alert). Android stack: **HAL** (standard hardware interfaces), **Java API framework** (reusable system services). `[4:16-17]`

# L. BIG DATA & NoSQL `[4:31-59]`
- **Big data** sources: surveillance, sensors, click-streams, archived records, trading data, social posts. Ecosystem: HDFS, MapReduce, Hive (DW), Spark, NoSQL, real-time analytics, visualisation. `[4:31-32]`
- **Hadoop `[4:33-34]`:** **HDFS** (files split into ~128 MB blocks, replicated, on commodity hardware; NameNode holds metadata in memory; DataNodes send heartbeats; pipelines) + **MapReduce**.
- **Map-Reduce example `[4:35-37]`:** average time on each page across thousands of users → *Map* emits ⟨page,(avg time,#users)⟩ per node → *Reduce* computes weighted average. Executes in parallel → low latency, high availability.
- **When to use Hadoop:** analytics, search, retention, **log processing**, text/image/audio, recommendations. **Not:** low-latency/near-real-time, many small files… `[4:38]`
- **Why Spark `[4:39-41]`:** MapReduce is **disk-based**; real-time fraud detection needs faster → Spark = **in-memory cluster computing**, good for iterative/real-time.
- **Real-time analytics `[4:42]`:** see/analyse data as it arrives — real-time ads, breach detection, sensor prediction.
- **MapReduce vs others `[4:43-45]`:** *MapReduce (batch)* high latency, excellent fault tolerance, massive volume, cheap for once-a-day historical logs · *Stream processing (Flink/Spark streaming)* low latency, complex state/checkpoints, memory-intensive · *MPP/SQL engines* fast on structured data, cost scales with storage+compute.
- **NoSQL `[4:46-55]`:** scalable (sharding), fast (in-memory), available (replication), handle semi/unstructured data; but **lack ACID transactions and joins**. Types: **Document** (MongoDB, CouchDB — JSON), **Key-value** (Riak, Redis, DynamoDB — hash access, phone directory/stock), **Column-oriented** (analytics, e.g. average electricity use in a region — read only needed columns), **Graph** (nodes+edges with properties — fraud detection, social networks), **In-memory** (response-time critical: telecom, mobile ads). RDBMS under CAP: one node = consistent but not partition tolerant. `[4:50]`
- **Stream processing `[4:56-59]`:** cab-hailing (pickup events, location deviation, emergency events → ops/customer SMS); use cases: trading, patient care, production lines, fraud, smart grid. **Storm** = real-time computation on unbounded streams; **Kafka** = distributed publish-subscribe queue feeding Storm.

# M. SECURITY TECHNOLOGIES `[4:60-73]`
| Technology | Purpose & how |
|---|---|
| **TLS/SSL** `[4:60-61]` | privacy + data integrity client↔server: agree version → select algorithms → authenticate via digital certificates → generate **shared secret key for symmetric encryption** (faster than asymmetric) |
| **OpenID** `[4:62-63]` | **authentication** — sign in to many sites with one account; site redirects to provider → provider authenticates → redirect back with credentials (not password) |
| **OAuth** `[4:64-65]` | **authorisation** — lets a client access *your data* in another site; provider authenticates you and returns a **token** |
| **LDAP** `[4:66]` | directory for read-mostly lookups of one piece of data fast (offload user validation for millions of users) |
| **IAM** `[4:67]` | authentication, authorisation, roles/groups, compliance |
| **Firewall / DMZ** `[4:68-71]` | allow/block traffic by rules; **DMZ = buffer between internet and internal network** (attacker must pass two firewalls); techniques: packet filtering, circuit-level gateways, application-layer filtering, NAT hiding |
| **Logging (Log4j)** `[4:72]` | capture information for debugging/maintenance (logging is expensive; log essentials) |
| **AJAX** `[4:73]` | async updates for interactive web apps |

# N. MACHINE-LEARNING SYSTEMS `[4:74-80]`
- ML = machines detect patterns and perform human-like activities (treatment selection, predictive maintenance).
- **Types `[4:75-78]`:** **Supervised** (labelled data → regression, decision tree) · **Unsupervised** (unlabelled → clusters, e.g. K-means crime hot-spots) · **Reinforcement** (learn by interacting; chess, robotics).
- **Steps to build a model `[4:79]`:** identify problem/benefit → identify **features** (domain knowledge) → decide model → train-test-validate → experiment/improve.
- **Architecture example `[4:80]`:** insurance claim image classification — image stored → event triggers function → calls ML service → result + image metadata stored → notification.

# O. MANAGEMENT & GOVERNANCE `[4:81-94]`
- **Planning `[4:81-83]`:** initial **top-down** plan (convince management; inherently wrong) → architecture team forms (each member lead architect for a subsystem) → **bottom-up** schedule (more accurate) → **reconcile** both → basis of the software development plan. Example top-down: ~150 KSLOC ≈ 150 components, 4 h paper design each, 8-week release cycle; 40% design.
- **PM–architect relationship** is the most important on the team: respect, coordinate, stay in own spheres. `[4:84]`
- **PMBOK `[4:85-87]`:** five process groups (initiating, planning, executing, monitoring & controlling, closing) × ten knowledge areas (integration, scope, schedule, cost, quality, resource, communications, risk, procurement, stakeholder) forming a matrix.
- **Global development:** driven by cost, skills, local knowledge; coordination between teams is critical. `[4:88]`
- **Trade-offs:** architect trades QAs; PM trades **features–schedule–quality**; resist **scope creep** (Change Control Board slows changes). **Incremental development:** each release in planning / development / test-and-repair, simultaneously. `[4:89-90]`
- **Metrics `[4:91-92]`:** global (size, schedule deviation, productivity, defects) tracked historically and per project.
- **Governance — four responsibilities `[4:93]`:** (1) controls over creation/monitoring of architectural components and activities; (2) compliance with internal/external standards and regulation; (3) processes to manage these; (4) accountability to stakeholders.

# P. ECONOMICS — CBAM (Cost-Benefit Analysis Method) `[4:95-116]`
- **Idea:** choose architectural strategies by **value for cost**: scenarios → utility → benefit across scenarios (weighted) ÷ cost.
- **Core formulas `[4:98-99]`:** benefit of one scenario `b(i,j) = U_expected − U_current`; total benefit `B(i) = Σ_j b(i,j) × W_j`; **`VFC(i) = B(i) / C(i)`** → rank strategies. Each strategy has a **cost** and impacts *multiple* QAs. `[4:95-97]`
- **Utility-response curve:** plot utility vs response level for a scenario `[4:96]`; scenarios are **weighted** because stakeholders care unequally.
- **Nine steps `[4:102-110]`:** ① collate scenarios (informal H/M/L; keep top ⅓) ② refine: worst/current/desired/best response levels ③ **prioritise by votes** (100 votes each; keep top 50%; top = weight 1.0) ④ assign **utility** to the four levels ⑤ map strategies → expected response levels ⑥ **interpolate** utility of expected level ⑦ compute **total benefit** ⑧ **cost + VFC**, choose until budget/schedule exhausted ⑨ confirm with intuition; iterate if misaligned with business goals.
- **Worked tables (Steps 6-8):** `[4:111-116]`.

---

# Q. LIKELY CONCEPT QUESTIONS (inferred — no PYQ archive) → where to answer from

| Likely question | Answer from |
|---|---|
| Define architecture; differentiate structure vs view; module vs C&C vs allocation | A1-A2 |
| What is a QA scenario? Write one for X | B1 + the QA's general scenario in C |
| Tactics for availability / performance / security / modifiability / testability / usability / interoperability | C1-C7 |
| Role of design checklist / seven categories of decisions | B3 |
| What is an ASR? How do you identify ASRs? Utility tree | D1-D3 |
| Agile vs architecture; how much up-front architecture | E |
| 4+1 view model; which views for which stakeholder | F2 |
| Techniques in each layer of layered architecture | G |
| ATAM steps / outputs / lightweight evaluation; sensitivity & trade-off points | H2 |
| How to avoid architectural drift; architecture reconstruction | H3, H5 |
| Explain pattern X (Layers, P&F, Blackboard, Broker, MVC, PAC, Microkernel, Reflection, SOA…) with benefits/liabilities | I1-I10 |
| Patterns vs tactics | A3, I0 |
| NIST cloud definitions; IaaS/PaaS/SaaS; multi-tenancy; CAP | J |
| Mobile app types; Android components | K |
| Hadoop/MapReduce/Spark; NoSQL types | L |
| TLS / OAuth / OpenID / DMZ | M |
| CBAM steps & VFC | P |
| Project planning/PM-architect/governance | O |
