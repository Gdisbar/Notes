# Software Architecture — Complete Lecture Summary
## SEZG651 / SSZG653 (CS01–CS08) | Based on Bass, Clements & Kazman, *Software Architecture in Practice*, 3rd Ed.

---

# MODULE 1 — Introduction to Software Architecture (CS01)

## 1.1 Why Software Architecture is Important (13 reasons, Len Bass)

1. **Inhibits or enables a system's driving quality attributes** — *the single most important message of the course.*
2. Allows reasoning about and **managing change** as the system evolves.
3. Enables **early prediction** of a system's qualities.
4. A documented architecture **enhances communication among stakeholders**.
5. Carrier of the **earliest, most fundamental, hardest-to-change design decisions**.
6. **Defines constraints** on subsequent implementation.
7. **Dictates the structure of the organization** (or vice versa) — becomes the work-breakdown structure.
8. Basis for **evolutionary prototyping** (skeletal system: infrastructure built before functionality).
9. Key artifact for **reasoning about cost and schedule** (top-down + bottom-up estimate consensus).
10. **Transferable, reusable model** at the heart of a software product line.
11. Focuses development on **assembly of components** rather than their creation (COTS, open source).
12. **Restricts design vocabulary** — channels creativity, reduces complexity.
13. Foundation for **training** new team members (module views = structure; C&C views = how it works).

## 1.2 How Architecture Enables Quality Attributes

| Quality attribute | Architectural lever |
|---|---|
| Performance | Manage time-based behavior of elements, shared-resource use, frequency/volume of inter-element communication |
| Modifiability | Assign responsibilities so most changes affect few elements |
| Security | Manage/protect inter-element communication; control access; add specialized elements (authorization) |
| Scalability | Localize resource use; avoid hardcoding resource limits |
| Incremental subset delivery | Manage inter-component usage (the *uses* structure) |
| Reusability | Restrict inter-element coupling so elements extract cleanly |

## 1.3 Reasoning About and Managing Change

- ~**80% of a system's total cost occurs after initial deployment.**
- Every architecture partitions possible changes into three categories:
  - **Local change** — modify a single element (most desirable).
  - **Nonlocal change** — multiple element modifications, architecture intact.
  - **Architectural change** — affects fundamental element interactions; changes all over the system.
- **A good architecture = one where the most common changes are local.**
- Finding problems early is cheaper, easier, less disruptive.

## 1.4 Enhancing Communication Among Stakeholders

- Architecture = common abstraction all stakeholders can use for mutual understanding, negotiation, consensus.
- **User** — fast, reliable, available. **Customer** — on schedule, on budget. **Manager** — teams work independently in disciplined ways. **Architect** — strategies to achieve all goals.

## 1.5 Earliest Design Decisions (examples of decisions embodied in architecture)

One processor or distributed? Layered — how many layers, what does each do? Synchronous or asynchronous communication? Control, data, or both? Dependence on OS/hardware features? Encryption of flowing information? Which communication protocol? — *imagine changing any of these late.*

## 1.6 Architecture as a Constraint on Implementation

- An implementation **exhibits** an architecture if it conforms to its prescribed elements, interactions, and responsibilities.
- Each prescription is a **constraint** on the implementer (who may be unaware of the trade-offs — e.g., a per-element **performance budget** whose sum meets the overall requirement).
- **Defining constraints on implementation** is a core architectural function.

## 1.7 Architecture and the Organization

- Architecture prescribes software units → units become the **work-breakdown structure** → dictates planning, scheduling, budget, inter-team communication, configuration control, integration/test plans, maintenance team structure. Formalized contractually, changing responsibilities can be expensive or litigious.

## 1.8 The Four Contexts of Software Architecture

| Context | Content |
|---|---|
| **Technical** | Quality attributes the architecture achieves; current technical environment (web-based, OO, service-oriented, cloud-based…); standard industry practices |
| **Project life-cycle** | Architecture activities exist in *every* process (waterfall, iterative, agile, model-driven): (1) business case, (2) understand ASRs, (3) create/select architecture, (4) document & communicate, (5) analyze/evaluate, (6) implement & test based on it, (7) ensure conformance |
| **Business** | Systems serve business goals of vested organizations; every QA should originate from a higher business purpose; some business goals never become requirements; some have no architectural effect at all |
| **Professional** | Architect needs diplomacy, negotiation, communication, leadership; up-to-date technical knowledge; business understanding; manages diverse workload |

**Architecture Influence Cycle:** requirements + business/social/technical influences → architecture → the architecture in turn affects future requirements, project/organization structure, business goals, and the architect's own future choices.

**Stakeholders:** anyone with a stake in system success; identify and engage them early to understand constraints, manage expectations, negotiate priorities, make trade-offs.

---

# MODULE 1 (contd.) — What is Software Architecture?

## 1.9 Definitions

> *"The software architecture of a system is the set of structures needed to reason about the system, which comprise software elements, relations among them, and properties of both."* (Bass)

- Preferred over "early/major design decisions" definitions: many architectural decisions are made early but not all; many early decisions are not architectural; "major" is unjudgeable. **Structures** are easy to identify and a powerful design tool.

- **Structure** = set of elements held together by a relation. No single structure *is* the architecture — a system has **many structures**; architecture = the set of them.
- **View** = representation of a coherent set of architectural elements, written by and read by stakeholders (a view is a representation of a structure). *Architects design structures; they document views.*

## 1.10 Three Categories of Architectural Structures

| Category | Elements | Answers questions like |
|---|---|---|
| **Module** | Implementation units (classes, layers, functionality divisions) with assigned responsibilities | What is each module's responsibility? What may it use? What does it actually use/depend on? Inheritance relations? |
| **Component-and-Connector (C&C)** | Runtime components (services, peers, clients, servers, filters) + connectors (communication vehicles: call-return, synchronization, pipes) | Major executing components and interactions? Shared data stores? What is replicated? Data flow? Parallelism? Runtime reconfiguration? |
| **Allocation** | Mapping of software to external environments (hardware, file systems, teams) | Which processor runs each element? Files/directories per element? Team assignment? |

A structure is **architectural** if it supports reasoning about a system property important to some stakeholder.

## 1.11 Key Properties of Architecture

- **Architecture is an abstraction** — omits private, purely internal element details; selects and suppresses information to tame complexity.
- **Every system has a software architecture** — even if undocumented, unknown, or the people are gone. But documentation is critical.
- **Architecture includes behavior** — behavior is architectural insofar as it supports reasoning about the system. Box-and-line drawings **without documented behavior are not architectures** (readers imagine behavior that could be wrong).

## 1.12 Useful Structures (detail)

**Module structures:**
- **Decomposition** — "is-a-submodule-of"; recursive decomposition; basis of modifiability (likely changes localized); basis of project organization, documentation, integration/test plans.
- **Uses** — A uses B if A's correctness requires a correct (not stubbed) B; enables extension and **functional subsets → incremental development**.
- **Layered** — carefully controlled uses; strictly layered: layer n uses only layer n−1; layers = virtual machines → portability.
- **Class/generalization** — "inherits-from"; reuse, incremental functionality.
- **Data model** — static information structure (entities, attributes, relationships; e.g., Account, Customer, Loan).

**C&C structures:** attachment relation between components and connectors.
- **Service** — services interoperate via coordination mechanisms (e.g., SOAP); components developed independently.
- **Concurrency** — logical threads; identifies parallelism opportunities and resource contention.
- **Process / communicating processes** — processes/threads + communication, synchronization, exclusion; drives execution performance and availability.
- **Shared data / repository** — components + connectors creating/storing/accessing persistent data; performance, data integrity.
- **Client-server** — clients, servers, protocols/messages; separation of concerns, physical distribution, load balancing.

**Allocation structures:**
- **Deployment** — software (usually processes) → processors + communication paths; "allocated-to", "migrates-to"; performance, availability, security, data integrity; vital in distributed systems.
- **Implementation** — modules → file structure of dev/integration/configuration environments; build management.
- **Work assignment** — modules → teams; architectural *and* management implications; determines inter-team communication paths.

*Mappings between structures are many-to-many: one module → one, part of, or several components.*

## 1.13 Architectural Patterns

Compositions of elements found useful across domains, documented and disseminated; packaged strategies with element types + interaction forms.
- **Module type:** Layered pattern (unidirectional uses → layers).
- **C&C type:** Shared-data/repository (database + SQL protocols); Client-server.
- **Allocation type:** Multi-tier (deployment); Competence-center and Platform patterns (work assignment — platform = one site builds product-line core assets, others build applications).

## 1.14 What Makes a "Good" Architecture?

- **No inherently good/bad architecture** — only more or less fit for purpose; evaluated only against stated goals. But there are rules of thumb:

1. Product of **a single architect or small group with identified technical leader** → conceptual integrity; strong architect–team connection (applies to Agile/open source too).
2. Based on a **prioritized list of well-specified QA requirements**.
3. **Documented using views** addressing the most important stakeholders' concerns.
4. **Evaluated early** (and repeated) for ability to deliver important QAs.
5. Lends itself to **incremental implementation** — build a "skeletal" system where communication paths are exercised with minimal functionality.
6. **Well-defined modules** on information hiding + separation of concerns; encapsulate things likely to change behind interfaces.
7. Achieve QAs via **well-known patterns and tactics** unless requirements are unprecedented.
8. **Never depend on a particular version** of a commercial product/tool (or make switching trivial).
9. **Separate producers from consumers of data** (confines change to one side).
10. **Don't expect 1:1 module↔component** correspondence.
11. Processes written so processor assignment can change, even at runtime.
12. **Small number of interaction mechanisms** — "same things done the same way" → understandability, reliability, modifiability.
13. **Small set of clearly resolved resource-contention areas.**

---

# MODULE 2 — Quality Attributes (CS02: QA framework + Availability, Performance, Usability, Security, Modifiability; CS03: Interoperability, Testability + others)

## 2.1 Requirements Categories

| Category | Nature |
|---|---|
| **Functional requirements** | What the system must do / how it reacts to runtime stimuli |
| **Quality attribute requirements** | Annotate/qualify functional ones — how fast, how resilient, how easy to learn |
| **Constraints** | Design decisions with **zero degrees of freedom** — already made for you |

- **Functionality does not determine architecture** — endless architectures can satisfy the same functionality; functionality and QAs are orthogonal (the "green button" example: same function, different performance/availability/usability annotations).
- Three problems with QA discussions: definitions not testable; endless argument over which attribute a concern belongs to (is a DoS failure availability, performance, security, usability?); each attribute community has its own vocabulary.
- **Solution:** quality attribute **scenarios** (solves 1 & 2) + per-attribute vocabularies of concerns (solves 3).

## 2.2 The Quality Attribute Scenario — 6 Parts

| Part | Meaning |
|---|---|
| **Source of stimulus** | Entity (human, system, actuator) generating the stimulus |
| **Stimulus** | Condition arriving at the system requiring response |
| **Artifact** | What is stimulated (whole system, collection, or part) |
| **Environment** | Conditions: normal, overload, startup/shutdown, degraded… |
| **Response** | Activity undertaken |
| **Response measure** | Measurable criterion so the requirement is testable |

- **General scenarios** = system-independent; **concrete scenarios** = specific to the system at hand.

## 2.3 Tactics

- **Tactic** = primitive design technique affecting a QA response; patterns = bundles/packages of tactics.
- Why tactics matter: patterns need adaptation; tactics allow first-principles design when no pattern fits; cataloguing makes design systematic.
- Choice of tactic depends on **trade-offs with other QAs and implementation cost**.

## 2.4 The Seven Design Decision Categories (apply to EVERY quality attribute)

1. **Allocation of responsibilities** — identify important responsibilities (functions, infrastructure, QA satisfaction); allocate to modules/components/connectors.
2. **Coordination model** — which elements coordinate; properties (timeliness, currency, completeness, correctness, consistency); mechanisms (stateful/stateless, synchronous/asynchronous, guaranteed delivery, throughput, latency).
3. **Data model** — major data abstractions, operations, properties; creation/initialization/access/persistence/manipulation/translation/destruction; metadata; organization (relational DB, objects, both).
4. **Management of resources** — resources + limits; who manages; sharing + arbitration; saturation impact.
5. **Mapping among architectural elements** — modules↔runtime elements; runtime elements→processors; data→stores; elements→delivery units.
6. **Binding time decisions** — every category has one; later binding = more flexibility but costlier mechanisms.
7. **Choice of technology** — availability of tools; internal familiarity/external support; side effects (imposed coordination model); compatibility with existing stack.

## 2.5 Availability

- **Availability** = property that software is *there and ready to carry out its task when needed*; encompasses reliability **plus recovery (repair)**; fundamentally about **minimizing service outage by mitigating faults**.
- **Fault** (has potential to cause failure) vs **Failure** (system no longer delivers service per spec, observable by actors).

**General scenario:** source = internal/external (people, hardware, software, infrastructure, environment); stimulus = fault (omission, crash, incorrect timing, incorrect response); artifact = processors, channels, storage, processes; environment = normal/startup/shutdown/repair/degraded/overloaded; response = prevent, detect (log, notify), recover (disable source, be unavailable, fix/mask, degraded mode); response measure = availability %, time to detect/repair, degraded-mode interval, proportion of faults handled.
- *Sample:* "The heartbeat monitor determines the server is nonresponsive during normal operations. The system informs the operator and continues to operate with no downtime."

**Tactics (goal: endure faults so services remain compliant):**

| Goal | Tactics |
|---|---|
| **Detect faults** | Ping/echo; Monitor; Heartbeat; Timestamp; Sanity checking; Condition monitoring; Voting (replication, functional, analytic redundancy); Exception detection (timeout, parameter fence/typing); Self-test |
| **Recover — preparation & repair** | Active redundancy (**hot spare**: all nodes process identical inputs in parallel; protection group = active + spare nodes); Passive redundancy (**warm spare**: active nodes send periodic state updates); Spare (**cold spare**: off until failover, power-on-reset); Exception handling; Rollback (to "rollback line"); Software upgrade (in-service); Retry; Ignore faulty behavior; Degradation (keep critical functions, drop others); Reconfiguration |
| **Recover — reintroduction** | Shadow mode; State resynchronization; Escalating restart (vary granularity); Non-stop forwarding (supervisory/data split) |
| **Prevent faults** | Removal from service; Transactions (ACID); Predictive model; Exception prevention (smart pointers, wrappers); Increase competence set |

All availability tactics involve the **coordination model**.

## 2.6 Performance

- **Performance** = time and the ability to meet timing requirements: characterize events (interrupts, messages, user requests, clock events — periodic, sporadic, stochastic) and time-based responses.
- **General scenario:** stimulus = event arrival; environment = normal/emergency/peak/overload; response = process events, change service level; response measure = **latency, deadline, throughput, jitter, miss rate**.
- *Sample:* "Users initiate transactions under normal operations. The system processes the transactions with an average latency of two seconds."

**Tactics (goal: respond within time constraint):**

| Goal | Tactics |
|---|---|
| **Control resource demand** | Manage sampling rate (↓ fidelity); Limit event response (max rate); Prioritize events; Reduce overhead (remove intermediaries — *note: intermediaries aid modifiability but cost performance*); Bound execution times; Increase resource efficiency (better algorithms) |
| **Manage resources** | Increase resources (faster/more CPUs, memory, networks); Increase concurrency; Maintain multiple copies of computations (replicas ↓ contention); Maintain multiple copies of data (different access speeds); Bound queue sizes; Schedule resources |

## 2.7 Usability

- **Usability** = how easy it is for the user to accomplish a desired task + the user support provided; one of the cheapest ways to improve perceived quality.
- Five areas: learning system features; using efficiently; minimizing impact of errors; adapting to user needs; increasing confidence and satisfaction.
- Response measures: task time, number of errors, tasks accomplished, satisfaction, knowledge gained, success ratio, time/data lost on error.
- *Sample:* "The user downloads a new application and is using it productively after two minutes of experimentation."

**Tactics** (user initiative vs system initiative):
- **Support user initiative:** Cancel; Pause/Resume; Undo; Aggregate.
- **Support system initiative:** Maintain **task model** (context → assistance), **user model** (user knowledge/behavior), **system model** (expected behavior → feedback).
- Checklist notes: data abstractions should support undo/cancel (transaction granularity not so coarse that undo takes excessively long).

## 2.8 Security

- **Security** = ability to protect data/information from unauthorized access while providing it to the authorized. **Attack** = action intended to harm (access, modify, deny service).
- **CIA:** **Confidentiality** (no unauthorized access), **Integrity** (no unauthorized manipulation), **Availability** (legitimate use not denied). Supporting: **Authentication** (parties are who they claim), **Nonrepudiation** (neither sender nor recipient can deny), **Authorization** (privileges to perform tasks).
- Response measures: extent compromised, time to detect attack, attacks resisted, recovery time, data vulnerability.
- *Sample:* "A disgruntled employee from a remote location attempts to modify the pay rate table during normal operations. The system maintains an audit trail and the correct data is restored within a day."
- Analogy to physical security → four tactic categories: **detect, resist, react, recover.**

| Goal | Tactics |
|---|---|
| **Detect attacks** | Detect intrusion (traffic vs known signatures); Detect service denial; Verify message integrity (checksums/hashes); Detect message delay |
| **Resist attacks** | Identify actors; Authenticate actors; Authorize actors; Limit access; Limit exposure (minimize attack surface); Encrypt data; Separate entities (physical/VM/air gap); Change default settings |
| **React to attacks** | Revoke access; Lock computer (repeated failures); Inform actors |
| **Recover from attacks** | **Audit** (record of user/system actions and effects → trace attacker) + availability recovery tactics |

No security tactic is foolproof — systems *will* be compromised; hence detect/limit/react/recover all exist.

## 2.9 Modifiability

- **Modifiability** = about change and the **cost and risk** of making changes. Three planning questions: **What can change? What is the likelihood? When is the change made and who makes it?**
- Response measures: number/size/complexity of affected artifacts, effort, calendar time, money, impact on other functions/QAs, new defects introduced.
- *Sample:* "The developer wishes to change the user interface by modifying the code at design time. The modifications are made with no side effects within three hours."

| Goal | Tactics |
|---|---|
| **Reduce size of a module** | Split module |
| **Increase cohesion** | Increase semantic coherence (responsibilities serving different purposes → different modules) |
| **Reduce coupling** | Encapsulate (explicit API); Use an intermediary (break A→B dependency); Restrict dependencies; Refactor (duplicate modules); Abstract common services |
| **Defer binding** | Later binding = cheaper change exercise, but mechanisms cost more upfront |

## 2.10 Interoperability

- **Interoperability** = degree to which two or more systems **usefully exchange meaningful information** — shades of meaning, not yes/no.
- General scenario: stimulus = request to exchange information; environment = systems known prior to runtime or discovered at runtime; response = (appropriately) reject + notify, accept + exchange, log; measures = % exchanges correctly processed/rejected.
- *Sample:* "Our vehicle information system sends our current location to the traffic monitoring system… correctly included with a probability of 99.9%."
- **Goal of tactics:** systems must (1) **know about each other → Locate**, (2) **exchange semantically meaningful info → Manage interfaces** (correct sequence; transform output of one actor into acceptable form for the other).

| Tactic group | Tactics |
|---|---|
| **Locate** | **Discover service** — search a known directory service (possibly multi-level indirection) |
| **Manage interfaces** | **Orchestrate** (control mechanism to coordinate/sequence service invocations for complex tasks); **Tailor interface** (add/remove capabilities: translation, buffering, data smoothing) |

- Checklist extras: log for **non-repudiation** in untrusted environments; resource management — can interoperation floods exhaust resources/deny service?; late-binding policies must support discovery and reject/log unacceptable bindings; prefer technologies designed for interoperability (e.g., web services).

## 2.11 Testability

- **Testability** = ease with which software can be made to demonstrate its faults through (execution-based) testing; formally: probability that software with at least one fault fails on its next test execution.
- Requires **controlling inputs/internal state** and **observing outputs/internal state** of each component.
- Response measures: effort to find a fault/class of faults; % state-space coverage achieved; probability next test reveals a fault; time to perform tests; preparation time; reduction in risk exposure (size(loss) × prob(loss)).
- *Sample:* "The unit tester completes a code unit during development and performs a test sequence whose results are captured and that gives 85% path coverage within 3 hours of testing."

| Goal | Tactics |
|---|---|
| **Control & observe system state** | Specialized interfaces (test harness/execution); Record/playback (capture interface traffic for reuse); Localize state storage (single place → start in arbitrary state); Abstract data sources (substitute test data); Sandbox (isolated experimentation, no consequences to undo); Executable assertions (hand-coded fault-state indicators) |
| **Limit complexity** | Limit structural complexity (avoid/resolve cyclic dependencies; encapsulate external-environment dependencies; reduce inter-component dependencies); Limit non-determinism (find and weed out unconstrained parallelism) |

- Anything reducing testing's high cost is a significant benefit; complex systems are hard to test (huge state space, many interconnections) — so **keep it simple**.

## 2.12 Other Quality Attributes

- **Variability** — special form of modifiability: producing preplanned variants. **Portability** — special form of modifiability: moving across platforms. **Development distributability** — supporting distributed development.
- **Scalability:** **horizontal (scale out)** = add resources to logical units (another server to a cluster); **vertical (scale up)** = add to a physical unit (more memory).
- **Deployability** — how an executable arrives at and is invoked on a host. **Mobility** — movement and platform affordances (display, input, bandwidth, battery).
- **Monitorability** — operations staff monitoring the executing system. **Safety** — avoiding states causing damage/injury/loss of life and limiting damage when they occur; architectural concerns nearly identical to **availability** (prevent, detect, recover).
- **Conceptual integrity** — consistency of design; demands "the same thing is done the same way"; drives understandability. **Marketability** — architecture itself marketed (service-oriented, cloud-based).
- **Quality in Use (ISO/IEC):** effectiveness (system correctness), efficiency (effort/time to develop), freedom from risk (economic/human-life/health/environment impact).
- **Software vs system QAs:** physical products add weight, size, power consumption, weather resistance, battery life — software architecture substantially affects these.
- **ISO/IEC 25010** product-quality standard; standard QA lists = useful checklists but never complete, generate controversy, force attention to irrelevant attributes.
- **Dealing with "X-ability":** model the attribute → assemble tactics → construct design checklists.

---

# MODULE 3 — Architecturally Significant Requirements & Architecture Design (CS04, CS05)

## 3.1 ASRs — Definition and Indicators

- **ASR** = requirement with a **profound (measurable) effect on the architecture**; the subset of requirements affecting architecture in measurably identifiable ways (software *and* hardware). Architecture might be dramatically different without it.
- Categorize requirements: **Critical / Important / Useful** (even in critical apps, only ~40% of features are used).
- **Indicators of architectural significance:** significant business value and/or technical risk; importance to an influential stakeholder; **uniqueness** (no existing component addresses it); QoS/SLA aspects differing from what the architecture currently accommodates; caused budget overruns/client dissatisfaction in similar past projects.
- Common ASRs: scalability, security, availability (with concrete SLAs).
- Straightforward functional requirements ("create an order", "track inventory") vs challenging ones ("detect fraud in real time and block the user"; "easy to customize for a different organization"; "operational under heavy load/hardware failure"; "millions of simultaneous requests"; "HIPAA-compliant EHR immune to breaches") — the latter are ASRs.

## 3.2 Probing Requirements (examples from slides)

- **Audit trail** — File or DB? Depends on *use*: analysis of activities vs searching for a specific activity.
- **Third-party interaction** — What communication mechanisms does the 3rd party support?
- **Notifications** — SMS / e-mail / message queue? One-way or acknowledged?
- These need scenario-level probing to decide the design approach.

## 3.3 Ways to Identify ASRs

1. **From requirement documents** — look for leads in the seven design-decision categories:
   - Allocation of responsibilities (planned evolution of responsibilities, user roles, system modes, major processing steps, commercial packages)
   - Coordination model (timeliness/currency/completeness/correctness/consistency; external elements, protocols, devices, middleware, network configs + security)
   - Data model (processing steps, information flows, domain entities, access rights, persistence, evolution)
   - Management of resources (time, concurrency, memory, scheduling, multi-user, devices, energy, buffers/queues; scalability requirements)
   - Mapping among architectural elements (teaming plans, processors, families, evolution, network configs)
   - Binding time decisions (functionality flexibility, regional/language distinctions, portability, calibrations, configurations)
   - Choice of technology (named technologies, planned/unplanned changes)
2. **Interviewing stakeholders — QAW.**
3. **Understanding business goals — PALM.**
4. **Capturing ASRs in a utility tree.**

## 3.4 Quality Attribute Workshop (QAW) — 6 Steps + Prioritization

*Expectation: a list of architectural drivers.*

1. **Presentation & introduction** — background, role, relationship to system.
2. **Business mission presentation** (~1 hr) — business context, broad functional requirements, constraints, known QA requirements.
3. **Architectural plan presentation** — architect presents current architectural thinking.
4. **Identification of architectural drivers** — facilitators share the distilled driver list (requirements, business drivers, constraints, QAs); stakeholders clarify/add/delete/correct to consensus.
5. **Scenario brainstorming** — each stakeholder expresses concerns as scenarios with **explicit stimulus and response**; ≥1 scenario per driver.
6. **Scenario consolidation** — merge similar scenarios with proposers' agreement.
7. **Scenario prioritization** — each stakeholder gets votes = **30% of the total (consolidated) scenario count**.

Flow: **Business goals + Quality attributes (+ non-architectural solutions) → Architecture.**

## 3.5 ASRs from Business Goals

- Business goals often → QA requirements; sometimes affect architecture directly without a QA; sometimes no influence at all.
- Format: **"X" to "Y" by when?** (X = current state, Y = future state).
- Standard goal categories: growth/continuity (market share +15%), financial (30% PAT), personal (learn new tech), responsibility to employees (workload −8%), society (green computing), state (regulatory compliance), shareholders (dividend), market position (IPR — 50 design registrations), business processes (productivity), product quality/reputation (automated testing), environmental change (dollar volatility).
- **BHIM example:** government goal "speed up cashless transactions" → app introduced → translates to QAs: **usability** (easy steps) and **security** (3-factor authentication).

## 3.6 PALM — Pedigreed Attribute eLicitation Method (7 steps)

*Purpose: capture business goals and give each architectural driver a "pedigree" — a strong, traceable business justification.*

1. PALM overview presentation.
2. **Business drivers presentation** — customer org goals + development org goals.
3. **Architecture drivers presentation** — architect briefs the ASRs.
4. **Business goals elicitation** — standard categories guide discussion; goals captured as "X to Y by when" scenarios; consolidate and prioritize.
5. **Identification of potential QAs from business goals** — for each important goal, describe a QA that would achieve it.
6. **Assignment of pedigree to existing QA drivers** — for each architectural driver, identify *which business goal it supports* (Why 40 ms and not 80 ms?).
7. Exercise conclusion — review, next steps, feedback.

## 3.7 Utility Tree

- **Hierarchical representation capturing and prioritizing key QAs**, refining them into scenarios.
- **Structure:** Root = **Utility** (how well the system satisfies stakeholder needs) → Branches = **quality attributes** → (attribute refinements) → **Leaf nodes = quality scenarios**.
- Each scenario rated **(Business value, Architectural impact)** with H/M/L.
- *Example leaves (from slides):* performance/transaction response time — "user updates a patient's account at peak load, completes < 0.75 s (H,M)"; usability/proficiency — "new hire proficient in core functions < 1 week (M,L)"; configurability — "hospital increases a service fee; config team changes it in 1 working day, no source-code change (H,L)"; security/integrity — "system resists unauthorized intrusion and reports it within 90 s (H,M)"; availability/no downtime — "DB vendor software hot-swapped with no downtime (H,L)".
- The utility tree is a **repository approach** — more like a repository; the architect is **accountable for ASRs even when no BSR exists**; stakeholder interviews must be planned well in advance; choose the method that fits available time.

## 3.8 Designing an Architecture — Design Strategy & ADD

**Attribute-Driven Design (ADD):** architecture designed by decomposing elements and applying **tactics per quality attribute driver**.
Steps (iterative):
1. **Choose the module to decompose** (start: the whole system / highest-level element).
2. **Refine the module** — apply architectural patterns and tactics that satisfy the QA scenarios driving this module.
3. **Define interfaces** for the children (module interactions).
4. **Verify and refine use cases and QA scenarios** — check the decomposition satisfies them.
5. **Repeat** until all requirements are covered.
*Design proceeds from drivers → scenarios → tactics → structures, never from functionality alone.*

## 3.9 Architecture in Agile Projects

- Agile manifesto values: individuals & interactions / working software / customer collaboration / responding to change (over processes & tools, comprehensive documentation, contract negotiation, following a plan). 12 principles include early continuous delivery, welcoming change, frequent delivery, business-developer daily work, motivated individuals, face-to-face conversation, working software as primary progress measure, sustainable pace, technical excellence, simplicity, self-organizing teams, regular reflection.
- Agile succeeded first on small-medium, short-timeframe projects; seldom used for large/distributed ones.

**How much architecture? (Boehm & Turner):** two schedule-consuming activities trade off — up-front architecture/risk work vs rework from defects/modifications. Plotted for 10 / 100 / 1,000 KSLOC projects:
- **10 KSLOC:** sweet spot at far left — much up-front work is wasted.
- **100 KSLOC:** sweet spot ≈ **20% of schedule**.
- **1,000 KSLOC:** sweet spot ≈ **40% of schedule** — "difficult to imagine how Agile principles alone can cope with this complexity without an architecture."
- *A million-line project needs an architecture to guide and organize the effort.*

**Agility and documentation:** "Write for the reader! If the reader doesn't need it, don't write it. But the reader may be a maintainer or newcomer not yet on the project."

**Agility and architecture evaluation:** Absolutely compatible — ATAM analyzes only the most important stakeholder scenarios, so it tailors naturally into a **lightweight evaluation**.

**WebArrow example:** team worked **top-down** (structures for demanding QA trade-offs) and **bottom-up** (implementation/environment constraints) simultaneously; resolved trade-offs via **spikes** (experiments): distributed DB vs flat files (latency impact)? mod_perl vs Perl (scalability, QA effort)? participants per meeting server? DB-server-to-meeting-server ratio? — **experimentation was the key factor**, not re-inventing methods.

**Guidelines for the Agile architect (Boehm's Incremental Commitment Model + authors' advice):**
- Large + complex + stable/well-understood requirements + distributed development → **large up-front architecture pays off**.
- Large + unstable requirements → **quickly design a candidate architecture**, leaving out details; elaborate as requirements emerge, via spikes.
- Small + uncertain → agree only on **major patterns**; don't over-invest up front.
- Middle ground: propose an initial architecture, run with it, **refactor when technical debt grows too great**.

---

# MODULE 4 — Documenting Software Architecture: Structures and Views (CS06)

## 4.1 The Documentation Problem & Solution

- **Problem:** one architecture document cannot address all stakeholders' concerns (end-user, system engineers, developers, project managers); complex diagrams don't fit a single document.
- **Solution:** **different notations for several views**, each addressing a specific set of concerns.
- *"Documenting an architecture is a matter of documenting the relevant views and then adding documentation that applies to more than one view."*

## 4.2 Kruchten's 4+1 Model (Rational Software Corp.)

| View | Elements | Viewer | Considers | Notation / Style |
|---|---|---|---|---|
| **Logical** | Object model — key abstractions (classes) | End-user | **Functional requirements** — services the system provides | Booch notation; tool: Rational Rose |
| **Process** | Concurrency & synchronization; process decomposition (major architecture-relevant tasks vs minor/helper tasks like buffering) | Integrators | **Non-functional**: scalability, concurrency, performance | Garlan & Shaw architecture styles |
| **Development** | Static organization of software in its development environment (layers, subsystems, libraries) | Programmers, software managers | Software module organization, reuse, tool constraints | Booch notation; **layered style** |
| **Physical** | Mapping of software onto hardware; distributed aspect; may have two architectures (test/development, deployment) | System engineers | **Non-functional**: reliability, availability, performance w.r.t. underlying hardware | — |
| **+1 Scenarios** | A few end-to-end scenarios (e.g., PABX local call — selection phase) traced through all views | All users & evaluators | System **consistency and validity** | Similar to logical view |

- **Views are interconnected** — start from logical, move to development/process, then physical. Two strategies: **inside-out** (from logical structure) or **outside-in** (from physical structure). Logical→process: classes active in each task; logical→development: grouping into subsystems by team organization, class categories/packages, lines of code — the larger the project, the greater the distance between these views.
- **Not all architectures need all views.** Scenario-driven, iterative development; the **Software Architecture Document** follows 4+1 closely; **Software Design Guidelines** capture the most important design decisions whose respect maintains architectural integrity.
- Successfully used in **air traffic control, telecom**; weakness: no tools to integrate views → inconsistency problem, most tangible during maintenance.

## 4.3 Synthesis — SEI Structures ↔ 4+1 Mapping

| SEI category | 4+1 view | Primary focus |
|---|---|---|
| Module | Logical | Functional decomposition, classes, layers |
| Component & Connector | Process | Runtime behavior, throughput, scalability |
| Allocation | Development | Software → file system/CM environment |
| Allocation | Physical | Software → hardware/cloud nodes |

*Use SEI definitions for **what** to document, 4+1 for **who** you're documenting for.*

## 4.4 Modernizing the Views (slide annotations)

- **Logical:** Booch → **UML** (class diagrams for static "is-a" hierarchies; package diagrams for module grouping). In service-oriented systems: defines service API contracts and domain boundaries.
- **Process:** components = containers (Docker) or serverless functions; use **sequence diagrams** (message exchange over time) and **activity diagrams** (parallel workflows); watch latency, deadlocks, eventual consistency.
- **Development:** account for DevOps/CI-CD; repository structure (monorepo vs multi-repo); third-party dependency management (NuGet/NPM/Maven); stakeholders = developers and build engineers.
- **Physical:** nodes are VMs/clusters (Kubernetes); document **elasticity**, network topologies (firewalls, gateways, load balancers) via deployment diagrams; infrastructure as code.
- **Architectural drift:** code evolves faster than documentation. Countermeasures: architecture-as-code (Mermaid/PlantUML in the repo), automated fitness functions (dependency checks against the module view). The architecture should be a **living document**, not a static PDF.

## 4.5 Choosing Views — Stakeholder Mapping Guide

- Complex UI? → prioritize **Logical**. High-performance/high-traffic? → **Process**. Large distributed team? → **Development**. Cloud-native/complex security? → **Physical**. *Document only what is necessary to communicate the design and mitigate risk.*

---

# MODULE 5 — Layered Architecture & ATAM (CS07)

## 5.1 Typical Layered Architecture

```
┌─────────────────────────────┐
│      Presentation Layer      │  UI / UX — React, Angular, mobile
├─────────────────────────────┤
│   Business / Logic Layer     │  Heart of the app: validation, rules, processing
├─────────────────────────────┤
│      Service Layer           │  APIs & orchestration — bridge to external consumers
├─────────────────────────────┤
│    Data Access Layer (DAL)   │  Persistence: ORM, DB connections
└─────────────────────────────┘
```
*(Judiciary system example: presentation — case creation/scheduling/judgement screens; business — workflow engine, case proceedings/judgement logic; service — generic SMS client + Vodafone/Airtel adapters; data access — Case/Proceedings/Schedule/Users DAOs, DB connections.)*

- **Why layers:** separation of concerns → **maintainability, scalability**.
- **Characteristics & benefits:** clearly defined functional layers; **loose coupling**; **reusable lower-layer components**; **exchangeable parts**.

## 5.2 Presentation Layer Techniques

- **Caching** frequently used data (e.g., product catalog):
  - **Client-side caching** — in the browser; user preferences, payment method, delivery address.
  - **Server-side caching** — shared by back-end modules; product codes/descriptions, promotions.
  - **Combination** used in practice. **Memcached** — high-performance distributed in-memory object caching; "short-term memory" taking memory from where it's surplus to where it's scarce; alleviates database load.
- **Asynchronous communication (AJAX)** — update parts of a page without full reload. Uses: product list per category; validating loan amount; chat panel; reload Captcha. Examples: Google Maps (drag entire map), Google Suggest. AJAX = Asynchronous JavaScript and XML (XML/HTML/CSS/JavaScript).
- **Responsive web design** — render well on varying devices/screen sizes; flexible grids; HTML5/CSS3 for display arrangement.

*Exercise solution pattern (Aadhaar registration screen — State/District/Town dropdowns slow): **cache states/districts/towns in the backend; use AJAX for dynamic loading of districts and towns.***

## 5.3 Business Layer Techniques

- **Application façade** — unified higher-level interface hiding internal modules. *Intent: make a subsystem easier to use.* Flight-booking example: `Booking_Engine` façade wraps Schedule (get flight list), Seat Inventory (availability), Pricing (price) for the Booking UI.
  - *Exercise (logistics): façade `placeShippingRequest()` in the business layer orchestrates container-location, transporter-selection, ship-capacity modules.*
- **Session management** — web farm: requests from one client may hit different server instances.
  - **Stateless session bean** — subsequent requests may go to another instance; **client maintains session state** (client-based state management, cookies).
  - **Stateful session bean** — all requests go to the same instance; bean maintains session state (server-based, persistent).
- **Workflow engine** — once an activity completes, engine invokes the next (manual or automated) step. Example: insurance policy processing. Engines: BizTalk Server, Oracle BPEL, IBM WebSphere Process Server. → supports **modifiability** (process change without code).
- **Rules engine** (e.g., DROOLS) — business rules externalized similarly.
- **Aspect-oriented design (AOP)** — encapsulate **cross-cutting concerns** into aspects: **logging & instrumentation, auditing (who is doing what), caching, security**. Aspects cut across layers; modules call the cross-cutting subsystem at different points.
- **Logging & instrumentation** — log *actionable* info ("Database is unavailable"); logging is expensive → log only essentials. Instrumentation reveals usage/behavior: request rate, error rate, duration, queue length. (Log4j → database, file, console, syslog.)
- **EJB appendix:** stateless vs stateful session beans (above); **SSL handshake** (certificate exchange, symmetric session key); **IPSec** (IETF standard: authentication, integrity, confidentiality at IP-packet level).

## 5.4 Data Layer Techniques

- **DB connection pool** — when too many users.
- **Multiple copies of data** for faster access.
- **Transactions** — atomicity.
- **Object-Relational Mapping (ORM)** — map program objects to DB tables (Hibernate).
- **Stored procedures** — improve performance.
- **Parameterized SQL queries** — reduce SQL injection: never concatenate user input into SQL strings. `'Toothpaste' OR 'x'='x'` returns all product data; use `WHERE product-name = ?` with the user value as parameter.

## 5.5 Service Layer

- Provides services to external systems. *Bank example: `getBalance(A/c#)`, `requestChequebook`.* *Hotel example: `getRoomAvailability(dates)`, `reserveRoom(...)`.*
- **Situations the service layer must handle:**
  - **Same request arriving twice** → duplicate handling.
  - **Messages arriving out of sequence** → reordering/sequence handling.
  - **Communication failure** → **retry mechanism**, or **queue the work and send once communication is restored**.

## 5.6 Architecture Evaluation & ATAM

**Why evaluate:** "fail early" philosophy; ATAM identifies **risks and trade-offs**; *you cannot have everything* (e.g., more security often decreases performance).

**Three forms of evaluation:** (1) by the **designer/architect**, (2) **peer review** (steps: preparation, presentation, analysis, findings), (3) **analysis by outsiders** (e.g., ATAM by an external evaluation team).

**Contextual factors for evaluation:** purpose, time in life cycle, who performs it, required thoroughness, etc.

**The Architecture Tradeoff Analysis Method (ATAM):**

*Evaluation team roles:*

| Role | Responsibility |
|---|---|
| Team leader | Sets up evaluation; coordinates with client; establishes contract; forms team; ensures final report delivered |
| Evaluation leader | Runs evaluation; facilitates scenario elicitation, selection/prioritization, analysis; facilitates on-site analysis |
| Scenario scribe | Writes scenarios on flipchart; captures exact agreed wording; halts discussion until wording is right |
| Proceedings scribe | Captures proceedings electronically: raw scenarios, motivating issues, resolutions; generates printed adopted-scenario list |
| Questioner | Raises architectural issues, usually in their QA area of expertise |

*Outputs:* **Risks** (potentially problematic architectural decisions), **Nonrisks** (good decisions), **Sensitivity points** (properties critical to achieving a QA — small change → big effect), **Trade-off points** (sensitivity points affecting >1 QA, typically in opposition), plus **risk themes** (risks grouping into systemic weaknesses). **Intangible outputs:** articulated business goals/priorities, shared vocabulary, basis for negotiation, understanding of each other's views.

*Phases:*

| Phase | Activity | Participants | Duration |
|---|---|---|---|
| 0 | Partnership & preparation (logistics, planning, stakeholder recruitment, team formation) | Eval team leadership + key decision-makers | Informal, weeks |
| 1 | Evaluation: Steps 1–6 | Eval team + project decision-makers | 1–2 days, then 2–3 week hiatus |
| 2 | Evaluation: Steps 7–9 | Eval team, decision-makers, stakeholders | 2 days |
| 3 | Follow-up: report, process improvement | Eval team + client | ~1 week |

*The 9 steps:*
1. **Present the ATAM** (method overview).
2. **Present business drivers** — system's business context, goals, constraints, QA requirements.
3. **Present the architecture** — views, approaches, documented behavior.
4. **Identify architectural approaches** (patterns/styles in use).
5. **Generate utility tree** — elicit, refine, prioritize QA scenarios (priorities by business value + architectural impact).
6. **Analyze architectural approaches** — map highly ranked utility-tree scenarios onto the architecture; identify risks, sensitivities, trade-offs.
7. **Brainstorm and prioritize scenarios** — all stakeholders generate scenarios; prioritize by vote.
8. **Analyze architectural approaches** — analyze against the new scenario set.
9. **Present results** — risks, nonrisks, sensitivities, trade-offs, risk themes.

*Lightweight ATAM (4–6 hours, for Agile/internal teams):* participants already know the method (Step 1 → 0 h); business drivers 0.25 h; architecture 0.5 h (1–2 scenarios traced through module + C&C views); approaches 0.25 h (may fold into Step 3); utility tree 0.5–1.5 h; **Step 6 consumes the bulk (2–3 h)**; Steps 7–8 **omitted** (internal stakeholders contribute scenarios at Step 5); results 0.5 h (review risks, nonrisks, sensitivities, trade-offs, risk themes).

---

# MODULE 6 — Conformance, Testing, Reconstruction (CS08)

## 6.1 Code Drifting from Architecture

*Examples of drift:*
- Not sticking to layer discipline — an object calling another object **beyond the adjacent layer**.
- **Accessing the database directly without going through the data access layer.**
- **Notifying modules one by one instead of using publish–subscribe.**
- **Not using the common logging mechanism.**

**Four techniques to keep code and architecture consistent:**

1. **Follow an "architecturally evident coding style"** — indicate in code which architectural aspect is being implemented: which **layer** the code belongs to; whether a component is a **publisher or subscriber**; whether a component **inserts into or retrieves from a message queue**.
2. **Use frameworks:** **Spring** (Spring MVC forces Model/View/Controller — Model encapsulates application data, View renders screens, Controller processes requests); **Hibernate** (ORM); **AUTOSAR** (Automotive Open System Architecture); JMS publish–subscribe; SalesForce workflow framework; DROOLS rules engine; Log4J logging.
3. **Use code templates** — a structure developers must code into. Fault-tolerance template example: Get event → Case(event type): Normal (primary processes X, Y; sends state to backup), Update state data (backup), Switch over (backup notifies clients of primary change) → End case.
4. **Update architecture documentation** — accompany code changes with doc changes; at minimum mark obsolete parts **"no longer applicable"** (increases trust in the remainder); **synchronize the doc with code at release time**.

*Exercise answers (additional techniques):* educate new team members about the architecture; **code reviews**; create **folders per architectural aspect** (layer, service, UI, external interfaces) and store code in the respective folders with discipline.

## 6.2 Architecture and Testing

**Three ways architecture helps testing:**
1. **Prioritize test cases based on ASRs** — using the **utility tree**: high-priority scenarios (high business value + high architectural impact) → high-priority test cases.
2. **Help create the integration test plan** — architecture shows **which modules interact and which depend on which** → identifies integration test cases and the modules needed for integration testing.
3. **Architecture must be designed to support testability requirements:**
   - **Ability to switch data sources** (test data ↔ production data).
   - **Roll back changes made by test cases** (restore system to original state → run more tests later).
   - **Ability to replace components** — simulators of payment gateways, sensors, external systems (e.g., Aadhaar).

## 6.3 Architecture Reconstruction

**Purposes:**
- Understand the architecture of a system for which **no documentation exists**.
- **Migrate a system from old technology to new** (e.g., mainframe → web).
- **Identify reusable components** (logging, security, etc.).

**Phases (detailed):**

| Phase | Activities |
|---|---|
| 1. **Raw view extraction** | Extract from **source code, execution traces, build scripts**. Yields: classes, files used, **caller–callee relationships**, global data accessed by objects |
| 2. **DB construction** | Put extracted data into a common format |
| 3. **View fusion** | Combine views stored in the DB: source analysis gives the **static** view; execution traces reveal **runtime/dynamically bound** interactions; an **expert then groups elements** (e.g., into layers) → consolidated view |
| 4. **Architecture analysis** | Validate correctness of fused elements; **check conformance** — e.g., "a layer may call adjacent layers only," "all DB access must be via entity beans" |
| 5. **Iterate** | Refine extraction/grouping; take help from the technical team for broad understanding before aggregation |

*Case study — "Vanish" system (tool: ARMIN — ARchitecture Reconstruction and MINing): the "white-noise" view showed all elements and relations; after aggregation, the architecture was visible — and analysis showed it was **not strictly layered**.*

*Tools:* **Dali, ARMIN, Lattix, SonarJ, Structure 101, SonarQube** (SonarQube explores all execution paths; SAR for service-oriented extraction).

**Tool-detected violation examples (from slides):**
- Rule "no portion of the application should depend upon JUnit" → Sonar flags the violation.
- Rule "all database access managed by entity beans" → violation discovered by DiscoTect.
- Sonar allows defining **layers and vertical slices**, populating them with components, and detecting cross-layer violations.

---

# APPENDIX — One-Page Exam Pointers

1. **The most important message of the course:** whether a system exhibits its desired quality attributes is substantially determined by its architecture.
2. **Quality attribute scenario = the currency of the course** — always name all six parts, never omit the response measure.
3. **Utility tree** = root (utility) → QA branches → refinements → scenarios rated **(business value, architectural impact)**.
4. **QAW:** 6 steps + prioritization with votes = **30% of scenario count**.
5. **PALM:** business goals in "**X** to **Y** by when" format; **pedigree** = which goal justifies each driver.
6. **ADD:** choose module → refine per driver tactics → define interfaces → verify scenarios → repeat.
7. **Agile sweet spots:** 10 KSLOC ≈ 0%, 100 KSLOC ≈ 20%, 1,000 KSLOC ≈ 40% of schedule; candidate architecture + spikes + refactor when debt grows.
8. **4+1:** Logical (end-user/functionality), Process (integrators/concurrency), Development (programmers/modules), Physical (system engineers/deployment), Scenarios (+1, consistency). SEI mapping: Module→Logical, C&C→Process, Allocation→Development/Physical.
9. **Layered techniques:** presentation = caching (client/server/memcached), AJAX, responsive design; business = façade, session mgmt (stateless/stateful), workflow/rules engine, AOP; data = connection pool, transactions, ORM, stored procedures, **parameterized SQL**; service = duplicates, out-of-sequence, retry/queue.
10. **ATAM outputs** = risks, nonrisks, sensitivity points, trade-off points, risk themes; Step 6 = bulk of analysis; lightweight = 4–6 h, Steps 7–8 omitted.
11. **Conformance:** evident coding style, frameworks, templates, doc updates (+ reviews, education, folder discipline).
12. **Testing:** utility tree prioritizes test cases; module structure drives integration plan; design for testability = switchable data sources, rollback, replaceable components.
13. **Reconstruction:** raw view extraction → DB construction → view fusion → architecture analysis → iterate; tools: Dali, ARMIN, Lattix, Sonar; purposes: understand undocumented systems, technology migration, reusable-component identification.
14. **Trade-offs are unavoidable:** security↔performance, availability↔cost, modifiability↔performance (intermediaries), scalability↔consistency — always state what a decision costs, and that fitness depends on business goals.
15. **Golden rules of thumb:** single technical leader; prioritized QAs first; views for stakeholders; evaluate early; incremental skeletal implementation; information hiding; no dependence on specific product versions; small number of interaction mechanisms.
