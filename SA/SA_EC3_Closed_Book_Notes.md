# Software Architectures (SE ZG651 / SS ZG653) — EC3 Closed-Book Revision Notes

> Built from: course handout, lecture decks (CS01–CS16), Mid-Sem answer keys (Set 1 & 2) and your question bank.
> Exam style (your analysis): **Conceptual · Scenario/design · Compare & justify · Text diagrams · Critical analysis (ATAM/trade-offs) · Case studies.**
> Marks tip: scenario/design answers win marks for **(a) naming the right ASRs, (b) naming specific tactics/patterns, (c) stating trade-offs, (d) one justified recommendation.**

**Legend:** ⭐ = very high-yield · 🧠 = memorise · ⚖️ = trade-off point · 🧩 = reusable answer template · 📝 = exam angle

---

## TABLE OF CONTENTS

| Part | Topic | Handout module |
|---|---|---|
| **A** | How to answer (templates, structure, time plan) | – |
| **1** | Foundations: definition, importance, structures, views, contexts | M1 |
| **2** | Quality attributes, scenarios, **tactics**, trade-offs ⭐ | M2 |
| **3** | ASRs, QAW, PALM, Utility Tree, ADD, Agile ⭐ | M3 |
| **4** | Documenting architecture: views, 4+1, doc package, Hatley-Pirbhai ⭐ | M4 |
| **5** | Layered techniques, **ATAM**, conformance, testing, reconstruction, real-time ⭐ | M5 |
| **6** | Architectural patterns (all 14) + comparison tables ⭐ | M6–7 |
| **7** | Integration strategies; cloud; multi-tenancy; cache; containers; serverless; **CAP & failure mgmt** ⭐ | M8 |
| **8** | Mobile architecture | M9 |
| **9** | New technologies: Big data, NoSQL, Hadoop, MapReduce, Spark/Kafka/Storm, ML, Blockchain, IoT, Security (TLS/OAuth/LDAP/DMZ) | M10 |
| **10** | Economic analysis — **CBAM** with worked numbers ⭐ | M11 |
| **11** | Emerging trends: WebAssembly, Service mesh, Edge; Forward engineering; Governance | M12 |
| **12** | **Case studies**: Netflix, Salesforce, CIGNA, TripAdvisor, Danske Bank, Amazon, NASA ECS, ATAM cases | R14 |
| **13** | **Question bank with model answers** (your Q1–Q12 expanded + 40 more by category) ⭐ | – |
| **14** | One-page cheat sheets (tactics, patterns, trade-offs, formulas) | – |

---

# PART A — HOW TO ANSWER (closed-book strategy)

## A.1 Time plan (2½ hrs, typically ~100 marks-equivalent / 40% weight)
- 5 min: read all questions; mark the scenario/design ones.
- Spend time **proportional to marks** (≈ 1.2–1.5 min per mark).
- Leave 10 min at the end for diagrams/tables you skipped.

## A.2 🧩 Universal scenario-answer template (use for ANY "design an architecture for X")
1. **Restate context** in 1–2 lines (domain, users, scale).
2. **Functional requirements** (3–5 bullets).
3. **Quality requirements → ASRs** as *measurable scenarios* (Source–Stimulus–Artifact–Environment–Response–Response-measure).
4. **Architectural drivers priority** (e.g., Security & Availability > Performance > Modifiability).
5. **Proposed architecture** (text diagram, 5–8 boxes).
6. **Patterns chosen + tactics per ASR** (table: ASR → Pattern/Tactic → Why).
7. **Trade-offs & risks** (⚖️ at least 3, each "X improves, Y degrades, mitigated by Z").
8. **Evaluation** (ATAM / prototype / load test / utility tree scenario walk-through).
9. **Final recommendation / justification** (1–2 lines).

## A.3 🧩 Compare-and-justify template
Table with rows = criteria (communication, coupling, scalability, performance, availability, modifiability, complexity, typical use) → then **"Recommendation for the given scenario: X because Y; combine with Z for W."** Never end with "it depends" — always pick.

## A.4 🧩 Critical analysis (ATAM) template
Stakeholders → business drivers → ASRs (utility tree) → approaches (patterns/tactics) → analysis per scenario → **Risks / Non-risks / Sensitivity points / Trade-off points** → risk themes → recommendation.

## A.5 🧩 Case-study template (Netflix, Salesforce, CIGNA, …)
**Context & problem → Drivers/ASRs → Architecture decisions (patterns + tactics) → Outcomes/benefits → Trade-offs/limits → Lesson.**

## A.6 Words examiners reward
*Architecturally significant · measurable scenario · tactic · pattern · trade-off · sensitivity point · loose coupling/high cohesion · separation of concerns · single point of failure · eventual consistency · stateless · horizontal scaling · defence in depth · fail-fast · graceful degradation.*

## A.7 Text-diagram conventions used in this note
```
[Box] component     ──► call/data flow     ◄──► bidirectional     ┄┄► async/event
( DB )  data store  ║ network boundary / tier   ░ external system
```

---

# PART 1 — FOUNDATIONS OF SOFTWARE ARCHITECTURE (M1)

## 1.1 Definition 🧠
> **Bass, Clements, Kazman:** *The software architecture of a system is the set of **structures** needed to reason about the system, which comprise **software elements, relations among them, and properties of both**.*

Key consequences (each is a possible 2-mark point):
- Architecture = **abstraction** (hides detail that has no effect outside one element).
- **Every system has an architecture** (even undocumented/poor).
- It is a **set of structures**, not one diagram.
- **Behaviour is part of architecture** ("properties of both" → how elements interact at runtime).
- Includes **externally visible properties** of elements (interfaces, resource usage, performance) — not internals.
- Architecture ≠ design in general: architecture = *significant* design decisions that shape quality attributes; detailed design = internals of elements.
- Alternative definition (IEEE 1471): fundamental organisation of a system embodied in its components, their relationships to each other and the environment, and principles guiding design and evolution.

## 1.2 Why is architecture important? (13 reasons — 🧠 memorise ≥ 8)
1. **Inhibits or enables** the driving quality attributes (most important message of the course).
2. Lets you **reason about and manage change** — ~80% of cost occurs *after* deployment. Changes are **local** (one element), **non-local** (several elements, approach intact) or **architectural** (changes how elements interact).
3. Enables **early prediction** of qualities (cheaper to fix design flaws early).
4. **Communication** among stakeholders.
5. Carries the **earliest, hardest-to-change** design decisions.
6. Defines **constraints** on implementation.
7. Influences **organisational structure** (Conway's Law — org structure mirrors system structure and vice-versa).
8. Basis for **evolutionary prototyping / incremental development**.
9. Lets architect & PM reason about **cost and schedule**.
10. **Transferable, reusable model** → product lines.
11. Focus on **assembly of components**, not just creation (incl. independently developed ones).
12. **Restricts design vocabulary** → consistency, fewer surprises.
13. Basis for **training** new team members.

## 1.3 Structures and views ⭐

**Structure** = the set of elements and relations *as they exist in software/hardware*. **View** = a *representation* of a structure, written/read by stakeholders.

| Category | Elements | Relations | Answers questions like | Example structures |
|---|---|---|---|---|
| **Module (static)** | Code/implementation units (modules, classes, packages) | *is-a-submodule-of, uses, inherits-from, allowed-to-use* | Primary responsibility of each module? What may it use? | **Decomposition**, **Uses**, **Layered**, **Class/generalisation**, Data model |
| **Component-&-Connector (C&C, runtime)** | Components (processes, services, data stores) + connectors (protocols, pipes, queues) | *attachment* | What executes, how do they interact, what runs in parallel, what is replicated? | **Service**, **Client-server**, **Process**, **Concurrency**, **Shared-data** |
| **Allocation** | Software elements ↔ environment (HW, file system, teams) | *allocated-to, stored-in, assigned-to* | Which processor runs it? Which team builds it? Where is it stored? | **Deployment**, **Implementation**, **Work assignment** |

**Useful structure table (🧠):**

| Structure | Relation | Useful for |
|---|---|---|
| Decomposition | is-submodule-of | Resource allocation, project planning, information hiding, config control |
| Uses | requires correct presence of | Engineering subsets & extensions, incremental development |
| Layered | allowed-to-use (unidirectional) | Portability, incremental development, virtual machines |
| Class | inherits-from | Reuse, incremental addition |
| Client-server / Service | communicates-with | Distribution, separation of concerns, load balancing |
| Process / Concurrency | runs concurrently with | Scheduling, performance, resource contention |
| Shared-data | produces/consumes data | Performance, data integrity |
| Deployment | allocated-to, migrates-to | Performance, availability, security analysis |
| Implementation | stored-in | Build/config control, integration, test |
| Work assignment | assigned-to | PM, expertise use, commonality |

**Relating structures:** a module (code) may become several runtime components; one component may span modules; **don't expect one-to-one mapping**. Structures give *insight into different quality attributes*: modifiability ← module structures; performance/availability ← C&C; security/availability/performance ← deployment.

## 1.4 Good architecture — rules of thumb 🧠
**Process rules:** single architect/small team with a leader (conceptual integrity); architecture driven by **prioritised quality attribute requirements**; documented with **views** useful to stakeholders; **evaluated** for ability to deliver the qualities; **incrementally implementable** (skeletal system first); resource-contention prevented.
**Product rules:** information hiding & separation of concerns; interfaces hide changeable aspects; use **well-known patterns/tactics**; no dependence on specific product versions; processes assignable to any processor; **small number of ways** for components to interact ("do the same thing the same way").

## 1.5 Architecture & contexts (4 contexts)
| Context | What it means | Influence on architecture |
|---|---|---|
| **Technical** | Quality attributes, current technology, existing systems | Drives pattern/tactic choices |
| **Project life-cycle** | Agile/waterfall/iterative; where architecture fits | Determines *how much* up-front design, documentation timing |
| **Business** | Time-to-market, cost, product-line, customers, regulation | Business goals → ASRs |
| **Professional** | Architect's skills, experience, org culture | Competence determines quality |

**Stakeholders:** customer, end users, developers, testers, maintainers, project manager, sysadmin/ops, marketing, regulators — each has *concerns* (performance, cost, usability, compliance…).
**Architecture competence:** *Individual* (duties + skills + knowledge — technical, business, soft skills) and *Organisational* (processes, career paths, governance) — SEI framework: **Duties, Skills, Knowledge**.

## 1.6 Architecture patterns vs tactics vs styles (quick)
- **Pattern** = context + problem + solution (bundle of design decisions, proven, reusable).
- **Tactic** = single design decision to control one quality attribute response (atom; pattern = molecule).
- See Part 6.7 for how tactics augment patterns.

📝 **Exam angle (2–5 marks):** "How does software architecture impact scalability & flexibility? Give example." → Use definition + modularity/SoC/information-hiding + horizontal scaling + microservice e-commerce example (checkout scales alone; new payment provider only touches Payment service) + **trade-off**: ops overhead, distributed data consistency.

---

# PART 2 — QUALITY ATTRIBUTES, SCENARIOS, TACTICS, TRADE-OFFS (M2) ⭐

## 2.1 Requirements categories
- **Functional** – what the system does.
- **Quality attribute (QA)** – qualifies functional requirements (how fast, how secure, how easy to modify).
- **Constraints** – design decisions with zero degrees of freedom ("must use Oracle", "deploy on AWS").
- Functionality **does not determine architecture** (many structures can implement the same function); **QAs do**.

## 2.2 Quality Attribute Scenario — 6 parts 🧠⭐
```
 SOURCE ──► STIMULUS ──► [ ARTIFACT in ENVIRONMENT ] ──► RESPONSE ──► RESPONSE MEASURE
 (who)      (event)      (what part, what state)         (what system  (how measured:
                                                           does)         time, %, cost)
```
| Part | Meaning | Example (performance) |
|---|---|---|
| Source | entity generating stimulus | 1,000 users |
| Stimulus | condition needing response | submit checkout requests |
| Artifact | part of system stimulated | order service |
| Environment | system state | peak load |
| Response | activity after stimulus | process transactions |
| Response measure | testable metric | avg latency < 2 s, 95th pct < 4 s |

**General scenario** = system-independent (can apply to any system). **Concrete scenario** = system-specific (written for *your* system) — used in utility trees.

## 2.3 Tactics — the idea 🧠
A **tactic** is a design decision that influences the control of a quality attribute **response**. Organised as: *Stimulus → [tactics] → Response*. Reasons to study tactics: patterns are bundles you must adapt; tactics let you construct/augment designs; and tactics give a vocabulary for analysis.

---
## 2.4 AVAILABILITY ⭐
**Definition:** probability the system is operational when needed; concerned with **faults → failures** and their duration/consequence. Availability = MTBF / (MTBF + MTTR).
**Scenario example:** *A server crashes (stimulus) during normal operation (env) → system detects, logs, switches to backup (response) → no downtime, recovery < 30 s (measure).*

| Nines | Downtime/year |
|---|---|
| 99% | 3.65 days |
| 99.9% | 8.76 hr |
| 99.99% | 52.6 min |
| 99.999% | 5.26 min |

### Tactics
| Group | Tactics |
|---|---|
| **Detect faults** | **Ping/Echo**, **Heartbeat**, **Monitor**, Timestamp, Sanity checking, Condition monitoring, **Voting** (TMR), Exception detection, Self-test |
| **Recover — preparation & repair** | **Active redundancy (hot spare)**, **Passive redundancy (warm spare)**, **Spare (cold)**, Exception handling, **Rollback**, Software upgrade (in-service), **Retry**, Ignore faulty behaviour, **Degradation**, **Reconfiguration** |
| **Recover — reintroduction** | Shadow, State resynchronisation, Escalating restart, Non-stop forwarding |
| **Prevent faults** | Removal from service, **Transactions**, Predictive model, Exception prevention, Increase competence set |

**Hot vs warm vs cold:** hot = all nodes process identical input in parallel (fastest failover, costliest); warm = only active processes, periodically updates standby (checkpoint); cold = standby powered off/unloaded (slowest, cheapest).
**⚖️ Trade-offs:** redundancy ↑ cost & complexity; consistency vs availability (CAP); heartbeat frequency vs network overhead/false positives.

---
## 2.5 PERFORMANCE ⭐
**Definition:** timing — ability to meet timing requirements (latency, throughput, jitter, deadlines) when events arrive. Resource consumption: **CPU, memory, I/O, network**. Response time = **processing time + blocked time** (contention, queueing, dependency).
**Scenario:** *Users initiate 2,000 transactions/sec at peak → system processes each → avg latency ≤ 2 s.*

### Tactics
| Group | Tactics |
|---|---|
| **Control resource demand** | Manage sampling rate, **Limit event response**, **Prioritise events**, **Reduce overhead** (remove intermediaries), **Bound execution times**, Increase resource efficiency (better algorithms) |
| **Manage resources** | **Increase resources** (scale up/out), **Increase concurrency**, **Maintain multiple copies of computation** (load-balanced replicas), **Maintain multiple copies of data (caching)**, **Bound queue sizes**, **Schedule resources** |

**Typical bank/e-commerce performance fix list:** cache hot data (client/server/distributed cache), CDN, DB connection pool, indexes/read replicas, async processing via message queue, load balancer + horizontal scaling, AJAX partial loads, batch heavy reports off-peak, rate limiting, compression, denormalised read models (CQRS).
**⚖️** caching ↔ stale data (consistency); replication ↔ cost; async ↔ complexity/eventual consistency; removing layers ↔ modifiability.

---
## 2.6 SCALABILITY (course adds this beyond Bass)
**Definition:** ability to handle growth in users/data/transactions by adding resources **without redesign**, ideally with cost growing ≤ linearly.
- **Vertical (scale-up):** bigger machine; simple, limit + SPOF.
- **Horizontal (scale-out):** more instances behind load balancer; needs **stateless** services, shared/externalised session, partitioned data.
- **Tactics:** stateless design; **load balancing**; **sharding/partitioning** (by tenant/geo/product); **caching**; **async messaging/queues** (load levelling); **read replicas / CQRS**; **auto-scaling**; CDN; microservices for independent scaling; avoid shared-state bottlenecks; NoSQL for scale-out.
- **Elasticity** = scale out *and in* automatically on demand (cloud).
**⚖️** scalability ↔ consistency (CAP), ↔ complexity/ops cost, ↔ latency (network hops).

---
## 2.7 MODIFIABILITY ⭐
**Definition:** cost/ease of making changes — *What can change? Likelihood? When is it changed (design/compile/build/deploy/run)? Who changes it?* Cost = f(number of elements touched, coupling).
**Scenario:** *Developer adds a new payment provider at design time → change made, tested, deployed in ≤ 3 person-days with no other module modified.*

| Goal | Tactics |
|---|---|
| **Increase cohesion** | Increase semantic coherence (split modules by purpose) |
| **Reduce coupling** | **Encapsulate**, **Use an intermediary** (facade, broker, adapter), **Restrict dependencies** (visibility, layers), **Refactor**, **Abstract common services** |
| **Defer binding time** | Runtime registration, **configuration files**, **plug-ins**, publish-subscribe, **dynamic lookup**, **parameterisation**, **resource files**, polymorphism, start-up time binding |

**Coupling & cohesion:** want **low coupling, high cohesion**. **Cost of change** = cost of modifying + cost of testing + cost of deploying.
**⚖️** intermediaries add latency (vs performance); late binding adds complexity/testing burden.

---
## 2.8 SECURITY ⭐
**Definition:** ability to protect data/services from unauthorised access while providing access to legitimate users. **CIA triad** (Confidentiality, Integrity, Availability) + **Authentication, Authorisation, Non-repudiation**.
**Scenario:** *A disgruntled employee attempts to modify pay data (stimulus) → system authenticates, denies, logs and alerts admin within 1 min (response).*
**Physical-security analogy:** checkpoints (resist), badges (detect), guards (react), off-site backup (recover).

| Group | Tactics |
|---|---|
| **Detect attacks** | Detect intrusion, detect service denial, verify message integrity, detect message delay |
| **Resist attacks** | **Identify actors, Authenticate, Authorise, Limit access, Limit exposure (reduce attack surface), Encrypt data, Separate entities (DMZ/VMs), Change default settings, Validate input** |
| **React to attacks** | Revoke access, **Lock computer** (after failed attempts), Inform actors |
| **Recover from attacks** | Availability recovery tactics + **Audit** (non-repudiation), maintain audit trail |

**Implementation techniques:** TLS/HTTPS, parameterised SQL (stop SQL injection), RBAC, OAuth/OpenID, MFA, WAF/firewall/DMZ, secrets vault, encryption at rest, least privilege, defence in depth, tenant isolation (cloud), logging/monitoring (SIEM).
**⚖️** security ↔ performance (encryption/auth overhead), ↔ usability (MFA), ↔ interoperability (restrictive), ↔ availability (lockouts).

---
## 2.9 USABILITY
**Definition:** how easy for user to accomplish a task + user support (learnability, efficiency, error handling, satisfaction, adaptability).
| Group | Tactics |
|---|---|
| **Support user initiative** | **Cancel, Undo, Pause/Resume, Aggregate** |
| **Support system initiative** | **Maintain task model, user model, system model** (feedback, assistance) |
Cheapest way to improve perceived quality. Design for **separation of UI from logic (MVC)** so UI can change quickly.

## 2.10 TESTABILITY
**Definition:** ease of demonstrating faults through testing (probability that a fault shows up in next test). Needs **controllability** and **observability**.
| Group | Tactics |
|---|---|
| **Control & observe state** | Specialised interfaces/test hooks, **Record/playback**, **Localise state storage**, **Abstract data sources** (swap test/prod data), **Sandbox**, **Executable assertions** |
| **Limit complexity** | Limit structural complexity (reduce cyclic dependencies, isolate environment), **limit non-determinism** |
Architecture-and-testing: utility-tree high-priority scenarios → high-priority test cases; architecture shows module dependencies → integration test plan; design for test (simulators for payment gateway, rollback of test data, switchable data sources).

## 2.11 INTEROPERABILITY
**Definition:** degree to which two or more systems can usefully **exchange and interpret** information (syntactic + semantic).
| Group | Tactics |
|---|---|
| **Locate** | **Discover service** (registry/directory, DNS) |
| **Manage interfaces** | **Orchestrate** (central coordination of services), **Tailor interface** (translate, buffer, smooth data) |
Enablers: open standards (REST/JSON, SOAP/XML), API gateway, ESB, adapters/canonical data model, versioned APIs. **⚖️** ↔ security (exposed surface), ↔ performance (translation).

## 2.12 INTEGRATION (course-specific QA)
Ease of combining the system with other systems/partners/legacy. Tactics = interoperability tactics + choose right **integration style** (Part 7.1) + loose coupling (messaging), anti-corruption layer for legacy, idempotent & versioned APIs.

## 2.13 Other QAs (one-line definitions)
- **Portability** – move to another platform (layers, abstraction, containers).
- **Deployability** – speed/ease of releasing (CI/CD, containers, blue-green, canary).
- **Energy efficiency, Safety, Maintainability, Reusability, Supportability/Observability** (logging, metrics, tracing), **Elasticity, Resilience, Auditability, Compliance (GDPR/HIPAA)**.
- ISO/IEC 25010 product quality list: functional suitability, performance efficiency, compatibility, usability, reliability, security, maintainability, portability. **Pros:** checklist. **Cons:** never complete, fuzzy taxonomy, forces attention to irrelevant QAs.

---
## 2.14 DESIGN TRADE-OFFS ⭐ (⚖️ master table)
| Pair | Why they conflict | Typical resolution |
|---|---|---|
| **Security ↔ Performance** | Encryption, authN/authZ checks, deep inspection add latency | TLS offload at gateway, token caching, hardware crypto, encrypt only sensitive fields |
| **Security ↔ Usability** | MFA, short sessions annoy users | Risk-based/adaptive auth, SSO (OpenID/OAuth), biometrics |
| **Availability ↔ Consistency (CAP)** | Under partition you cannot have both | Choose per use case (cart = AP; payment = CP); eventual consistency |
| **Availability ↔ Cost** | Redundancy, multi-region | Tier criticality: hot spare for core, cold for non-critical |
| **Performance ↔ Modifiability** | Layers/intermediaries/late binding add overhead | Keep intermediaries but cache; optimise hot paths only |
| **Scalability ↔ Consistency/Complexity** | Distribution → partitions, eventual consistency | Partition by tenant; sagas; idempotency |
| **Modifiability ↔ Time-to-market/cost** | Up-front design is costly | Right-size architecture (sweet spot ~20% of schedule for 100 KSLOC, ~40% for 1,000 KSLOC) |
| **Interoperability ↔ Security** | Open interfaces enlarge attack surface | API gateway, OAuth scopes, mTLS |
| **Testability ↔ Performance/Security** | Hooks, logging, assertions cost time & expose data | Disable in prod/feature flags |
| **Usability ↔ Performance** | Rich UI → heavy payloads | Lazy loading, CDN, AJAX, responsive images |
| **Microservices: Modifiability/Scalability ↔ Operational complexity/latency** | Many deployables, network hops, data consistency | Service mesh, API gateway, observability, CI/CD |

📝 **Exam angle:** *"Banking app slow at peak; improve performance without hurting security & availability."* → See Q2 in Part 13.

---

# PART 3 — CAPTURING ASRs, QAW, PALM, UTILITY TREE, ADD, AGILE (M3) ⭐

## 3.1 What is an ASR? 🧠
**Architecturally Significant Requirement** = a requirement that has a **profound, measurable effect on the architecture**; the architecture would be *dramatically different without it*. It is a subset of functional + quality + constraint requirements.

**Indicators of significance (any one):**
- High **business value** and/or **technical risk**.
- Important to a **key influential stakeholder**.
- **Unique** — no existing component addresses it.
- **QoS/SLA** differs from what the current architecture offers.
- Has previously caused **budget overruns or client dissatisfaction**.
- Hard to change later (affects hardware, communication, data model).

**Drivers that shape ASRs:** business goals & constraints, stakeholder concerns, quality attributes, regulation/compliance (GDPR, HIPAA), technology constraints, cost/budget, time-to-market.

**Examples:** audit trail, alerts/notifications, variety of data inputs, 3rd-party interaction, workflow, user-behaviour analytics, "detect fraud in real time and block the user", "customise easily for another organisation", "operate under heavy load/hardware failure".
**Why probe deeper?** "Audit trail" → file vs DB depends on how it's used (analysis vs lookup). "Notification" → SMS/email/MQ? one-way or acknowledged? "3rd-party" → which protocols?

## 3.2 Five ways to find ASRs
| Source | How |
|---|---|
| **Requirement documents** | MoSCoW (Must/Should/Could/Won't) or user stories; check quality attributes are included in equal proportion |
| **Design decisions** (leads) | Allocation of responsibilities, coordination model, data model, resource management, mapping among elements, binding time, technology choices → look for requirements behind each |
| **Stakeholder interviews → QAW** | Brainstorm scenarios with stakeholders |
| **Business goals → PALM** | Elicit goals, derive QAs |
| **Utility tree** | Top-down decomposition of QAs into scenarios, prioritised |

## 3.3 Quality Attribute Workshop (QAW) ⭐🧠
Facilitated, stakeholder-centric method (SEI), **before** architecture exists/finalised. **Output: list of architectural drivers + prioritised scenarios.**

| Step | Activity |
|---|---|
| 1 | **QAW presentation & introductions** – facilitators explain method, agenda |
| 2 | **Business/mission presentation** – sponsor describes business goals, context, requirements |
| 3 | **Architectural plan presentation** – architect presents current plan, constraints, existing systems |
| 4 | **Identification of architectural drivers** – facilitators list key drivers (functional, QA, constraints) from steps 2–3; stakeholders add/delete/correct → consensus |
| 5 | **Scenario brainstorming** – stakeholders create scenarios (use cases = normal use; growth scenarios; **exploratory** = stress/ unforeseen) |
| 6 | **Scenario consolidation** – merge similar, remove duplicates |
| 7 | **Scenario prioritisation** – each stakeholder gets votes (≈ 30% × number of scenarios) |
| 8 | **Scenario refinement** – top scenarios fleshed out into 6-part form (source, stimulus, artifact, environment, response, response measure) |

**Outputs:** list of drivers, raw & prioritised scenarios, refined top scenarios, stakeholder buy-in.

## 3.4 Business goals → PALM (Pedigreed Attribute eLicitation Method)
- **Business goal format:** "from **X** (current) to **Y** (future) by **when**" e.g., *increase market share 15% in 12 months*.
- Categories: growth & continuity, financial objectives, personal objectives, responsibility to employees, to society, to state, to shareholders, image, product-quality, etc.
- Business goals may (1) lead to QAs, (2) directly shape architecture without a QA, or (3) have no architectural influence.
- **Example (BHIM):** goal = speed up cashless transactions ⇒ QAs: **usability** (few steps), **security** (multi-factor).
- **"Pedigree"** = the goal's background (origin, rationale, who owns it) so QA requirements are traceable to business.
- **PALM steps:** overview → business drivers presentation → architecture drivers presentation → **business goals elicitation** → **goals→ASR** (identify QAs & scenarios) → **goal-ASR table/ pedigree** → expressing goals formally (*goal source, subject, object, environment, goal, measure, pedigree: origin, assumptions*) → consolidate & prioritise.

## 3.5 Utility Tree ⭐🧠
Top-down tool that translates vague QAs ("fast & secure") into **prioritised, concrete, measurable scenarios (leaves)**. Used in QAW follow-up and **ATAM step 5**.
```
Utility (root = overall "goodness")
 ├── Performance
 │    ├── Transaction response time
 │    │     └── (H,M) Update patient record < 0.75 s at peak load
 │    └── Throughput
 │          └── (H,H) Process 10,000 claims/hr at peak
 ├── Security
 │    ├── Confidentiality
 │    │     └── (H,M) Therapist sees only orthopaedic part of record
 │    └── Integrity
 │          └── (H,M) Intrusion detected & reported to authorities < 90 s
 ├── Availability
 │    └── No downtime
 │          └── (H,L) DB vendor patch hot-swapped with zero downtime
 └── Modifiability
      └── Configurability
            └── (H,L) Hospital changes service fee in 1 day, no code change
```
**Priority rating = (Business Value/Importance, Architectural Difficulty/Impact)** each H/M/L. **(H,H) scenarios = the true ASRs** → drive design and are walked through first in ATAM. (H,L) = important but easy; (L,H) = expensive for little value (candidates to push back).
**Building steps:** (1) root "Utility"; (2) list top QAs from business goals; (3) refine each QA into attribute refinements; (4) write concrete scenarios as leaves; (5) rate (value, difficulty); (6) review with stakeholders.
**Other uses:** prioritise test cases, plan iterations, justify trade-offs, repository of requirements.

## 3.6 Architectural drivers & design strategy
**Architectural drivers** = the combination of functional requirements, QA requirements and constraints that **shape the architecture** (a short, prioritised subset of ASRs). Typical drivers list: top 3–5 QAs + key use cases + constraints + business goals.

## 3.7 Attribute-Driven Design (ADD 3.0) ⭐🧠
Iterative, QA-driven, recursive decomposition method (SEI).
1. **Review inputs** – purpose, primary functional requirements, QA scenarios (ranked), constraints, concerns.
2. **Establish iteration goal** by selecting the drivers to address in this iteration.
3. **Choose element(s)** of the system to refine (start: whole system; then modules).
4. **Choose design concepts** (patterns, tactics, reference architectures, frameworks, technologies) satisfying the drivers.
5. **Instantiate elements, allocate responsibilities, define interfaces** (relationships).
6. **Sketch views & record design decisions** (rationale).
7. **Analyse current design, review iteration goal, achievement of purpose** (is each driver now satisfied? ⇒ iterate).
Repeat until all drivers satisfied; track progress with a kanban-style board (not started / partially / fully addressed).
**Design strategy:** *decomposition* (top-down), *composition* (bottom-up using existing parts), *generate-and-test*; typical design: pick pattern per ASR → add tactics → check conflict.

## 3.8 Agile & architecture ⭐
| Agile values | Traditional |
|---|---|
| Individuals & interactions | Processes & tools |
| Working software | Comprehensive documentation |
| Customer collaboration | Contract negotiation |
| Responding to change | Following a plan |

- Not "agile **or** architecture" → ask *"how much architecture up front vs deferred, and how much to document, and when?"*
- **BDUF** (Big Design Up-Front) vs **emergent design**: for small project (10 KSLOC) sweet spot is almost zero up-front; ~**20%** of schedule for 100 KSLOC; ~**40%** for 1,000 KSLOC. Adding up-front work reduces later rework (risk resolution), but too much wastes effort.
- **Practices:** architectural runway / spike, ASR-driven first sprints (walking skeleton), architecture as backlog items, evolutionary architecture (incremental design + refactoring + **fitness functions** + automated guardrails), lightweight docs (views + decisions log / ADRs), **WebArrow example**: team worked **top-down** (QA trade-offs) and **bottom-up** (implementation constraints) simultaneously, emphasising **experimentation**.
- **Evolutionary architecture:** designing so the architecture adapts incrementally to changing needs — supports "welcome changing requirements"; enabled by modularity, layering, refactoring, CI/CD, automated tests.
- **Sustainable architectures in agile:** modular/microservices, cloud-native, stable API boundaries, continuous refactoring, automated architecture conformance checks.

📝 **Exam angle:** "Explain QAW & build a utility tree for a healthcare platform" → Q3 in Part 13.

---

# PART 4 — DOCUMENTING SOFTWARE ARCHITECTURE (M4) ⭐

## 4.1 Why document? (uses & audiences)
- **Education** (onboard), **Communication** among stakeholders, **Construction** (blueprint for teams), **Analysis** (evaluate before building), **Evolution/maintenance**, **Reuse**, project management (estimate/plan).
- Mid-sem key answer points: **Reuse · Construction · Evolution · Analysis.**
- **Documenting = document relevant *views* + documentation that applies to more than one view** (rationale, mapping between views, context, glossary).

## 4.2 Challenges & best practices (Set-1 Q2 style)
| Challenges | Best practices |
|---|---|
| System complexity (many moving parts) | Use **views**, abstraction levels |
| Rapid iteration (agile) → docs go stale | Docs-as-code (Git, Mermaid/PlantUML), update at release, mark "no longer applicable" |
| Detail vs brevity | Document *only what is needed to communicate and mitigate risk*; "just enough" |
| Diverse audiences | Stakeholder-specific views & notation |
| Tool/notation inconsistency | **Standard templates** (e.g., ISO 42010, arc42, SEI Views & Beyond), UML |
| Architectural drift | Conformance checks, fitness functions, reviews |
Best practices: templates, **visual aids**, versioned docs, rationale/decisions (ADR), stakeholder reviews, keep the doc a **living artefact**.

## 4.3 Choosing views — stakeholder → view mapping 🧠
| Stakeholder | Main concern | Views |
|---|---|---|
| End user / customer | Functionality | **Logical**, Scenarios |
| Business stakeholders / PM | Scope, cost, schedule | Scenarios, Logical (overview), Work-assignment |
| Developers | Code structure, modules, layers | **Development**, Logical |
| Testers / QA | Testability, test cases | Scenarios, Development, Process |
| Integrators / performance engineers | Concurrency, throughput | **Process** |
| Infrastructure / DevOps / sysadmin | Deployment, scaling, topology | **Physical (Deployment)** |
| Security team | Trust boundaries, data flows | Physical, Process, Security view |
| Architect / evaluators | Consistency/validity | Scenarios (+1) |

**Quality-attribute views:** *Security view* (trust zones, authN/authZ flow, encryption), *Communication view* (protocols, message flow), *Reliability view* (redundancy, failover, recovery) — cut across structures to show how a QA is achieved.
**Combining views:** when two views map nearly 1-to-1 (e.g., logical + process for small system), merge to avoid inconsistency/overhead; document **correspondence** between views.

## 4.4 Kruchten's 4+1 View Model ⭐🧠
*Problem solved:* a single complex diagram can't address all stakeholders → use **multiple concurrent views, each with its own notation, tied together by scenarios**.

```
                 ┌──────────────────────┐
                 │   SCENARIOS (+1)      │  use cases tie all views
                 └─────────┬────────────┘
   ┌───────────────┬───────┴─────────┬──────────────────┐
   ▼               ▼                 ▼                  ▼
┌──────────┐ ┌─────────────┐ ┌──────────────┐ ┌────────────────┐
│ LOGICAL  │ │  PROCESS    │ │ DEVELOPMENT  │ │   PHYSICAL     │
│ (object  │ │(concurrency,│ │ (modules,    │ │ (HW nodes,     │
│  model)  │ │ sync, perf) │ │  layers,pkgs)│ │  network)      │
│End-user  │ │Integrators  │ │Programmers,  │ │System engineers│
│Function. │ │Non-function.│ │SW managers   │ │Topology,       │
│          │ │(scalability)│ │Reuse, build  │ │reliability     │
└──────────┘ └─────────────┘ └──────────────┘ └────────────────┘
```
| View | Describes | Viewer | Concerns | Notation / style | SEI category |
|---|---|---|---|---|---|
| **Logical** | Key abstractions (classes/objects), object model | End-user, analysts | Functional requirements | Booch → **UML class/package diagrams**; ER | Module |
| **Process** | Concurrency, synchronisation, distribution of processes/tasks | Integrators, performance engineers | Performance, scalability, availability, throughput | Garlan-Shaw styles; **sequence/activity diagrams** | C&C |
| **Development** (implementation) | Static organisation of software in dev environment: subsystems, layers, libraries | Programmers, SW managers | Organisation, reuse, tool constraints, build | **Layered style**, package/component diagrams | Allocation (implementation) |
| **Physical** (deployment) | Mapping of software onto hardware | System engineers | Availability, reliability, performance, scalability w.r.t. HW | **Deployment diagrams** | Allocation (deployment) |
| **Scenarios (+1)** | Few important use cases that illustrate/validate all four | All, evaluators | Consistency, validity | Use-case, sequence diagrams over elements of the four views | – |

**Correspondence:** start Logical → Development/Process → Physical. Logical→Process via *inside-out* (from logical structure) or *outside-in* (from physical constraints). Larger project ⇒ larger gap between logical and development.
**Iterative, scenario-driven:** pick key scenarios → build architecture to support them → test via prototype → refine. **Not every system needs all views.** Deliverables: *Software Architecture Document* (follows 4+1) + *Software Design Guidelines* (decisions that preserve integrity).
**Strengths:** stakeholder-centric, separates concerns, validated in industry (air-traffic control, telecom PABX). **Weakness:** no tool integration/consistency between views ⇒ inconsistency, especially in maintenance.
**Modernised:** Logical ⇒ UML + bounded contexts/API contracts; Process ⇒ containers/serverless, sequence diagrams, eventual consistency; Development ⇒ CI/CD, mono-/multi-repo, dependencies; Physical ⇒ VMs/Kubernetes/IaC, firewalls, gateways, load balancers.
**Which view first?** complex UI → Logical; high traffic → Process; large distributed team → Development; cloud/complex security → Physical.

### 4.4.1 Example: 4+1 for a distributed e-commerce platform (write this in the exam!)
```
LOGICAL:   Customer ──< Order >── OrderLine ──> Product ; Payment ; Inventory
PROCESS:   Browser → API-GW → [OrderSvc] ──msg──► [PaymentSvc] ──► [NotificationSvc]
           (async queue decouples; Order & Payment run as separate processes/containers)
DEVELOPMENT: repo/ ├ ui/  ├ order-service/ (controller|service|repo layers) ├ payment-service/ ├ common-lib/ ├ ci-cd/
PHYSICAL:  CDN → LB → [K8s cluster: N pods/service, 3 AZs] → (DB primary + replicas) → Cache cluster; DMZ+WAF in front
SCENARIO:  "Place order": Customer → UI → GW → OrderSvc (reserve stock) → PaymentSvc → event → NotificationSvc
```

## 4.5 SEI Structures ↔ 4+1 mapping
| SEI category | 4+1 | Focus |
|---|---|---|
| Module | Logical | Functional decomposition, classes, layers |
| C&C | Process | Runtime behaviour, throughput, scalability |
| Allocation (implementation) | Development | File system / CM environment |
| Allocation (deployment) | Physical | Software → hardware |
| Quality scenarios | Scenarios (+1) | Validation |

## 4.6 Documentation package (SEI Views & Beyond) 🧠
**Per-view template:** (1) **Primary presentation** (diagram), (2) **Element catalogue** (elements & properties, relations, interfaces, behaviour), (3) **Context diagram**, (4) **Variability guide**, (5) **Architecture background / rationale**, (6) **Other information**.
**Beyond views (cross-view):** documentation roadmap, **how to read**, view template, **system overview**, **mapping between views**, **rationale & design decisions**, **glossary, acronym list**, **management info** (e.g., stakeholders, risks).
**Documenting behaviour:** sequence diagrams, state machines, activity diagrams, communication diagrams.
**Documenting interfaces:** name, resources/operations, data types, error handling, quality attributes of the interface.
**Documentation & QAs:** performance (process view + timing), modifiability (module/uses view), security (deployment + security view), availability (deployment + failover behaviour).
**Fast-changing architectures:** document only stable decisions; automate extraction from code; lightweight ADRs.

## 4.7 Hatley-Pirbhai Architectural Template (HPAT) (handout item)
Template (from Pressman) for **real-time/embedded** architectures, built from **Architecture Context Diagram (ACD)**, **Architecture Flow Diagram (AFD)** and **Architecture Interconnect Diagram (AID)**. Every system is partitioned into 5 processing regions:
```
┌───────────────────────────────────────────────────────────┐
│  User interface processing  (I/O to human operator)        │
├───────────────┬───────────────────────────┬────────────────┤
│ INPUT         │ PROCESS & CONTROL FUNCTIONS│ OUTPUT          │
│ processing    │ (core functions + control) │ processing      │
├───────────────┴───────────────────────────┴────────────────┤
│  Maintenance & self-test processing (diagnostics, monitor)  │
└───────────────────────────────────────────────────────────┘
```
Steps: build **ACD** (system and external entities) → refine into **AFD** (data & control flows between the 5 regions) → map to hardware/software modules via **AID** (module interconnections). Benefit: systematic allocation of functions incl. control and self-test, common in real-time/embedded systems. *(Not in the decks — a 2–3 mark recall item.)*

📝 **Exam angle:** Q5 (Part 13): "Document a distributed app with 4+1; identify stakeholders and views".

---

# PART 5 — LAYERED ARCHITECTURE TECHNIQUES, ATAM, CONFORMANCE, TESTING, RECONSTRUCTION, REAL-TIME (M5) ⭐

## 5.1 Typical layered architecture & per-layer techniques
```
 Clients ──► [ PRESENTATION ]  UI/UX, validation, responsive design, AJAX, client caching
                  │
             [  SERVICE     ]  APIs for external systems (REST/SOAP), idempotency, retry
                  │
             [  BUSINESS    ]  facade, rules/workflow engines, session mgmt, AOP
                  │
             [  DATA ACCESS ]  ORM, connection pool, transactions, parameterised SQL
                  │
              ( Database )          ⟂ Cross-cutting: logging, security, caching, exception mgmt
```
**Benefits:** separation of concerns, loose coupling, reusable lower layers, exchangeable parts, maintainability, testability. **Costs:** performance overhead (each hop), "sinkhole" of pass-through layers, tight vertical coupling if layering rules violated.

| Layer | Techniques (🧠) | Why |
|---|---|---|
| **Presentation** | **Client-side vs server-side caching** (prefs vs product catalogue; use both); **AJAX** (async partial page update — product list by category, input validation, chat panel, captcha reload, Google Maps/Suggest); **Responsive design** (flexible grids, media queries, % widths) for form factors | Perceived performance, multi-device |
| **Business** | **Application façade** (unified interface hiding sub-systems, e.g., flight booking façade → schedule, seat inventory, pricing); **Session management** (client cookies vs server session; **stateful vs stateless** beans — stateless needs client-held state, scales better); **Workflow engine** (BizTalk, BPEL; modifiability of process), **Rules engine** (Drools); **Aspect-oriented design** for cross-cutting concerns (logging, auditing, caching, security); **Logging vs instrumentation** (log errors only, measure request rate/error rate/duration/queue length — logging is expensive) | Modifiability, performance, cohesion |
| **Data** | **Connection pooling**; multiple copies of data (replicas/caches); **Transactions** (atomicity); **ORM (Hibernate)**; **stored procedures**; **parameterised SQL (prevent SQL injection** — never concatenate user input: `' OR 'x'='x'`) | Performance, security |
| **Service** | Expose operations (e.g., GetBalance, RequestChequebook; hotel `GetRoomAvailability`, `ReserveRoom`); handle **duplicate requests (idempotency)**, **out-of-order messages**, **communication failure (retry / queue and send later)** | Integration with external apps (MakeMyTrip etc.) |
| **Cross-cutting** | Logging (Log4j), **security (SSL/TLS, IPSec)**, caching (Memcached), exception handling | – |

**Memcached:** high-performance distributed in-memory object cache that reduces DB load.
**Quick exercises (from deck):**
- *Aadhaar registration screen slow (state/district/town drop-downs)* → **cache** reference data server-side + **AJAX** to load districts/towns dynamically.
- *Logistics system hiding container-finding, transporter, ship modules* → **Façade in business layer.**
- *Hotel reservation exposing availability to MakeMyTrip* → **Service layer** with `getRoomAvailability`, `reserveRoom`.

---
## 5.2 ARCHITECTURE EVALUATION ⭐

### 5.2.1 Why evaluate & three forms
"Fail early": cheaper to find design problems before coding; identifies **risks & trade-offs**.
| Form | By whom | When | Effort |
|---|---|---|---|
| **By the designer** | Architect within design process (generate-and-test) | Each key decision | Depends on importance, # alternatives |
| **Peer review** | Peers within the project | Any time a coherent part exists | Few hours – half day |
| **Analysis by outsiders** | Independent team | Once architecture designed (e.g., **ATAM**) | Days; objective, credible to management |
**Contextual factors:** available artefacts, who sees results, who evaluates, which stakeholders participate, what the evaluation focuses on, and what the **evaluators' techniques** are.
**Peer-review steps:** (1) reviewers pick QA scenarios, (2) architect presents, (3) for each scenario architect walks through & explains, (4) capture problems, (5) review summary.

### 5.2.2 ATAM — Architecture Tradeoff Analysis Method ⭐🧠
**Purpose:** evaluate architecture against **QA requirements**, discover **risks**, **sensitivity points** and **trade-off points**, and reveal **business-goal ↔ architecture** links. Doesn't need system built; evaluators need not know the system.

**Participants:** (a) **Evaluation team** (3–5 external, unbiased: team leader, evaluation leader, scenario scribe, proceedings scribe, timekeeper, process observer, questioners); (b) **Project decision makers** (PM, customer, **architect always**); (c) **Other stakeholders** (developers, testers, maintainers, users, integrators, ops, business).

**Phases:**
| Phase | Activity | Who | Duration |
|---|---|---|---|
| **0** | Partnership & preparation (logistics, stakeholder recruitment) | Eval team + decision makers | Weeks |
| **1** | Evaluation steps 1–6 | Eval team + decision makers | 1–2 days |
| **2** | Evaluation steps 7–9 (with wider stakeholders) | All | 2 days |
| **3** | Follow-up (final report, process improvement) | Eval team + client | 1 week |

**The 9 steps 🧠:**
1. **Present ATAM** – method, steps, outputs.
2. **Present business drivers** – key functions, constraints (technical, managerial, economic, political), business goals, **major stakeholders**, **architectural drivers** (the top QAs).
3. **Present architecture** – by the architect: context, technical constraints, views (module, C&C, deployment), **how it addresses drivers**, ≤ 1 hr.
4. **Identify architectural approaches** – catalogue the patterns/tactics used (becomes basis of analysis).
5. **Generate the quality attribute utility tree** – prioritise scenarios (H/M/L importance, H/M/L difficulty).
6. **Analyse architectural approaches** – for the top-ranked scenarios, architect explains how the architecture supports each; team probes and records **risks, non-risks, sensitivity points, trade-off points**.
7. **Brainstorm & prioritise scenarios** – the larger stakeholder group proposes **use-case, growth and exploratory** scenarios; vote.
8. **Analyse architectural approaches** – repeat step 6 for the newly prioritised scenarios (top 5–10), test results against the utility tree (any surprises?).
9. **Present results** – summarise: approaches, utility tree, scenarios, **risks, non-risks, sensitivity points, trade-off points, risk themes** and the business drivers each threatens.

**Outputs 🧠:** concise architecture presentation; articulated business goals; **prioritised QA scenarios**; **risks & non-risks**; **risk themes**; mapping of decisions to QAs; **sensitivity & trade-off points**; plus intangibles (stakeholder community, communication, shared understanding).

**Definitions (with examples) ⭐🧠**
| Term | Definition | Example |
|---|---|---|
| **Risk** | An architectural decision that may lead to **undesirable consequences** in light of a QA requirement | "Single DB with no replica → availability goal at risk" ; "heartbeat interval too long → slow failure detection" |
| **Non-risk** | A decision that is **good in context**, with documented assumptions | "Replicated stateless services behind LB satisfy 99.95% availability" |
| **Sensitivity point** | A property of one/more components (or relationships) that is **critical to achieving a particular QA** — small change strongly affects that QA | "Cache hit ratio determines response time"; "number of concurrent DB connections determines throughput"; "encryption key length affects security" |
| **Trade-off point** | A property that is a sensitivity point for **two or more QAs** — improving one **degrades** another | "Encryption strength ↑ security but ↓ performance"; "replication factor ↑ availability but ↓ consistency/performance & ↑ cost" |
| **Risk theme** | Group of related risks revealing a **systemic weakness** (e.g., "insufficient attention to failover") threatening a business driver | – |
**Relation:** every trade-off point is a sensitivity point (the reverse is not true). Sensitivity/trade-off points are *candidate risks*.

**Lightweight Architecture Evaluation:** based on ATAM for smaller, less risky projects: ½–1 day, **internal** team, 4–6 hour agenda (skips Step 1; abbreviated steps 2–3), cheaper but less objective. Full ATAM ≈ **20–30 person-days** for the evaluation team (plus architect & stakeholders).

**ATAM benefits:** early risk detection, prioritised QAs, documented rationale, improved communication, stakeholder buy-in. **Limitations:** heavy cost, needs documentation & committed stakeholders, qualitative (relies on experts), snapshot in time.

**ATAM example — monolith → microservices (Q4 in Part 13):** utility tree leaves like *"Checkout scales to 5× load in 5 min (H,H)"*, *"zone failure → no customer-visible downtime (H,H)"*, *"JWT-protected inter-service calls (H,M)"*, *"new feature deployed independently in <1 day (M,M)"*. Risks: distributed transactions, shared DB, gateway SPOF, inconsistent security between services. Trade-off: independent scaling/modifiability ↔ latency/complexity/ops cost.

**Rockwell Collins CAAS & Battlefield Control System (course references R8, R9):** ATAM applied to a **avionics product line** (modifiability, portability, real-time performance across cockpit products) and an **army battlefield-control system** (availability/performance in distributed C2 nodes; modifiability of communication). Typical outputs: risk themes on redundancy/failover strategy, performance-vs-modifiability trade-offs and need for clearer interface documentation. *(Use as "examples of ATAM in practice" — cite generic outputs, not exact numbers.)*

---
## 5.3 Ensuring conformance (preventing architectural drift) ⭐
**Drift examples:** skipping layers (UI → DB directly), bypassing data-access layer, notifying modules one-by-one instead of publish-subscribe, not using common logging.
**Techniques 🧠:**
1. **Architecturally-evident coding style** (naming/annotations show layer/role: e.g., `@Publisher`, `@Subscriber`, package-per-layer; MQ producer vs consumer).
2. **Use frameworks** that enforce structure (Spring MVC – Model/View/Controller; Hibernate; AUTOSAR; JMS for pub-sub; Salesforce workflow; Drools; Log4j).
3. **Code templates** (e.g., fault-tolerant template: *Normal / Update-state / Switch-over* sections).
4. **Keep documentation updated** (sync at release; mark outdated parts "no longer applicable").
5. **Education, code reviews, folder-per-layer discipline.**
6. **Automated checks / fitness functions** (SonarQube, Lattix, Structure101, ArchUnit): "layer n may only call n-1", "all DB access through entity beans", "no module depends on JUnit".

## 5.4 Architecture & testing 🧠
- **Prioritise tests via utility tree** (H,H scenarios → top test cases).
- **Integration test plan** from module dependencies (who interacts with whom).
- **Design for testability**: switch test/prod data sources, roll back test changes, replace components with simulators (payment gateway, sensors, Aadhaar), observability hooks.
- Architecture drives **non-functional tests** (load, failover, security, penetration) derived from QA scenarios.

## 5.5 Architecture reconstruction (SAR) ⭐
**Purpose:** (1) understand a system with **no documentation**, (2) **migrate** (mainframe → web), (3) find **reusable components** (logging, security), (4) check conformance.
**Phases 🧠:**
| Phase | What |
|---|---|
| **Raw view extraction** | From source code, execution traces, build scripts: classes, files, caller–callee, global data |
| **Database construction** | Put extracted data in common repository/format |
| **View fusion** | Combine static view (source) + dynamic view (execution traces — late binding) + expert grouping → consolidated view |
| **Architecture analysis** | Check conformance (e.g., layer calls only adjacent, DB only via entity beans); find violations/missing components |
Tools: **Dali, ARMIN, Lattix, SonarJ/SonarQube, Structure101**. Case study "Vanish" (ARMIN): white-noise view of all elements → aggregate components with help of tech team → analysis showed the system **was not strictly layered**.

---
## 5.6 Real-time architectures (handout item — short)
- **Real-time system:** correctness depends on **time** as well as logic. **Hard** (missed deadline = failure; avionics, pacemaker) vs **soft** (degraded value; video streaming) vs **firm**.
- **Key concerns:** determinism, predictability, bounded latency, priorities, concurrency, fault tolerance, resource limits.
- **Techniques:** RTOS; **priority-based pre-emptive scheduling**, rate-monotonic / earliest-deadline-first; interrupt-driven I/O; bounded queues; watchdog timers; heartbeat; redundancy (hot standby); avoid dynamic allocation and unbounded blocking; separation of time-critical and non-critical tasks.
- **Patterns:** pipes-and-filters for sensor streams, publish-subscribe for events, shared-data/blackboard for state, cyclic executive, **Hatley-Pirbhai** template (4.7).
- **Real-time analytics** (big-data sense) is different: low-latency stream processing (Part 9).

---

# PART 6 — ARCHITECTURAL PATTERNS (M6–M7) ⭐

## 6.0 What is a pattern? 🧠
**Pattern = Context + Problem + Solution.** Solution = element types + interaction mechanisms (connectors) + topological layout + semantic constraints. Properties: proven design experience; abstractions above single classes; common vocabulary; documentation aid; supports construction with defined properties; builds complex/heterogeneous architectures; manages complexity. A pattern is a **scheme**, not a ready-made module.
**Categories (POSA):** *From mud to structure* (Layers, Pipes & Filters, Blackboard) · *Distributed systems* (Broker) · *Interactive systems* (MVC, PAC) · *Adaptable systems* (Microkernel, Reflection). **SEI (Bass) classification:** *Module* (Layered) · *C&C* (Broker, MVC, Pipe-Filter, Client-Server, P2P, SOA, Pub-Sub, Shared-Data) · *Allocation* (Map-Reduce, Multi-tier).

---
## 6.1 LAYERED ⭐ (module pattern)
- **Context:** parts need to be developed/evolved independently. **Problem:** modules with little interaction supporting portability, modifiability, reuse.
- **Solution:** layers = cohesive groups of modules with a public interface; **unidirectional allowed-to-use** (layer n uses n-1; strict or relaxed). Each software piece in exactly one layer; ≥ 2 layers (usually 3+).
- Examples: OSI 7-layer, TCP/IP, Android stack, 3-layer web app.
```
 [ Presentation ] ──► [ Business ] ──► [ Data access ] ──► ( DB )
```
- **Pros:** separation of concerns, replaceable layers, reuse, testability, portability. **Cons:** performance overhead, can be rigid, "layer bridging" erodes benefits, hard to choose granularity, upward notifications need callbacks/events.
- **Best for:** enterprise/business apps, platforms, virtual machines.

## 6.2 MODEL–VIEW–CONTROLLER (MVC) ⭐
- **Context:** UI is the most frequently changed part; multiple views of same data. **Problem:** keep UI separate from application logic yet responsive; support multiple synchronised views.
- **Elements:** **Model** (data + logic), **View** (rendering/input), **Controller** (mediates, translates user actions to model operations). **Relation:** *notifies* (observer). **Constraint:** model must not interact directly with controller; ≥ 1 of each.
```
 User ──► [Controller] ──updates──► [Model] ──notifies──► [View] ──► User
                 ▲_____selects view________________________│
```
- **Pros:** multiple views; UI changes isolated; testable model; parallel dev (designers/developers). **Cons:** overhead for simple UIs; many small classes; abstractions may not fit some toolkits; controller can bloat. **Frameworks:** Spring MVC, ASP.NET MVC, Angular (MV*), Django (MTV). Variant: **PAC (Presentation-Abstraction-Control)** — hierarchy of cooperating agents.
- Tactics used: increase semantic coherence, encapsulation, intermediary, runtime binding.

## 6.3 PIPE-AND-FILTER ⭐
- **Context:** streams of data items transformed in sequence. **Problem:** reusable, loosely coupled, parallel components with simple generic interaction.
- **Elements:** **Filter** (reads input port, transforms, writes output port); **Pipe** (connector; one source, one target; preserves order; buffers). Filters are independent and **stateless**, don't know neighbours.
```
 Source ─pipe─► [Filter A: validate] ─pipe─► [Filter B: enrich] ─pipe─► [Filter C: price] ─► Sink
```
- **Pros:** reusability, easy recomposition, parallelism/throughput, simple reasoning. **Cons:** poor for interactive/event-driven work, latency (data copies, serialisation), shared state hard, error handling hard, lowest-common-denominator data format.
- **Known uses:** Unix shell, compilers, ETL, image/video processing, order-validation pipelines, CMS Pipelines, Apache Camel/Kafka Streams.

## 6.4 PUBLISH–SUBSCRIBE (Event-based) ⭐
- **Context:** independent producers/consumers whose number/identity is not fixed. **Problem:** transmit messages so that producers and consumers are **unaware of each other**.
- **Elements:** components with publish/subscribe ports; **event bus/broker/topic** connector (announce & listen roles). **Constraint:** all components attach to an event distributor.
```
 [Order Svc] ┄publish OrderPlaced┄► ( Topic / Event Bus ) ┄┄► [Inventory]
                                                          ┄┄► [Billing]
                                                          ┄┄► [Notification]
```
- **Pros:** **loose coupling** (space & time decoupling), easy to add subscribers (modifiability), scalability, async. **Cons:** increased latency, unpredictable delivery time, ordering/delivery not guaranteed (unless configured), harder debugging/tracing, eventual consistency, broker = SPOF unless clustered.
- **Tech:** Kafka, RabbitMQ, JMS, AWS SNS/SQS, Google Pub/Sub. **Variants:** point-to-point queue vs topic; event sourcing.

## 6.5 BROKER (distributed systems)
- **Problem:** clients shouldn't know **location/nature** of servers; dynamic binding.
- **Elements:** client, server, **broker** (locates server, forwards request, returns result), client-side proxy, server-side proxy (marshal/unmarshal), bridge (broker-to-broker).
- **Pros:** location transparency, interoperability (CORBA, RMI, ORBs), flexibility. **Cons:** indirection adds latency, broker may be a **bottleneck and single point of failure**, up-front complexity, testing harder. *(Fixes: replicate broker – tactics.)*
- Differs from pub-sub: broker = request/response mediation; pub-sub = event dissemination.

## 6.6 CLIENT–SERVER, MULTI-TIER, P2P, SHARED-DATA, MAP-REDUCE, SOA, MICROSERVICES
### Client–Server ⭐
- Clients **initiate** request/reply to servers that provide services; servers can be clients to others. Promotes **modifiability & reuse** (common services in one place), **scalability/availability** via centralised control and replicated servers.
- **Cons:** server = bottleneck/SPOF; deciding where to place functionality (thin vs thick client) is costly to change. Thin client = logic on server; thick/rich client = logic on client.
### Multi-tier (allocation pattern)
- Group components into **tiers** (execution structures on different hardware) — each component in exactly one tier; e.g., Web tier – App tier – Data tier (+ DMZ). Relations: *is part of, communicates with, allocated-to*. **Pros:** independent scaling & security zones. **Cons:** up-front cost, network latency. **Layers ≠ Tiers:** layers = logical code organisation; tiers = physical deployment.
### Peer-to-Peer
- Equal peers both consume & provide services; special peers may index/route (supernodes). **Pros:** no central bottleneck, high availability & scalability, resilient. **Cons:** security, consistency, discovery, backup/recovery complex; small P2P may not meet QoS. Uses: BitTorrent, blockchain, file sharing, Skype (originally).
### Shared-Data (Repository / Blackboard)
- Accessors interact **only through shared data store(s)** — persistent data exchange. Pros: simple integration, data consistency, central backup. Cons: store is a **bottleneck & SPOF**; tight coupling via data schema. Uses: DB-centric enterprise apps, data warehouse, blackboard AI.
### Map-Reduce (allocation/analytic pattern) ⭐
- For **petabyte-scale** analysis: **Map** (stateless, many instances in parallel; extract & transform → `<key,value>`) + **Reduce** (aggregate per key) + **infrastructure** (splits data, schedules, monitors, handles failures, shuffle/sort). Constraints: data is files; maps don't communicate. Cons: **batch, disk-based, high latency**, poor for iterative/interactive jobs, only if data is distributable. (More in Part 9.)
### SOA (Service-Oriented Architecture) ⭐
- **Context:** services offered/consumed across platforms/languages/organisations. **Problem:** interoperability of distributed components.
- **Elements:** service providers, consumers, **ESB** (routing, transformation), **service registry** (UDDI), orchestration server; **connectors:** SOAP (WSDL), REST, async messaging.
- **Pros:** interoperability, reuse, legacy wrapping, business alignment, loose coupling via contracts. **Cons:** complex, performance overhead (XML/ESB), you don't control evolution of external services, ESB can become **central bottleneck/"smart pipe"**, governance heavy.
### Microservices ⭐
- Small, **independently deployable**, **single-business-capability** services, each with **own data store**, communicating over lightweight protocols (REST/gRPC/messaging); "smart endpoints, dumb pipes"; **decentralised governance**; built for **DevOps/CI-CD**; design for failure.
- **Enablers:** API gateway, service discovery, containers/Kubernetes, circuit breaker, distributed tracing, centralised config, saga for transactions.
- **Pros:** independent scaling & deployment, technology heterogeneity, fault isolation, team autonomy (Conway), faster releases. **Cons:** distributed-system complexity, network latency, **data consistency (no ACID across services)**, ops overhead, testing & monitoring harder, versioning.
- **SOA vs Microservices:** SOA = enterprise-wide reuse, ESB-centric, heavier protocols (SOAP), shared data/ governance central; MS = application-scoped, decentralised, lightweight, independently deployable, DB per service.

---
## 6.6b Adaptable-systems patterns (POSA)
**Microkernel:** minimal core (mechanisms: IPC, resource mgmt) + **internal servers** + **external servers (personalities)** + **adapters** + clients; separates *mechanism from policy*. Uses: OS (Mach, QNX), IDEs/plug-in systems (Eclipse). Pros: portability, extensibility, reliability; Cons: performance (message passing), design complexity.
**Reflection:** **meta-level** (metaobjects describing structure/behaviour) + **base-level** (app logic) + **metaobject protocol (MOP)** to change behaviour at runtime without modifying code. Pros: easy change, no source modification; Cons: meta-level mistakes dangerous, lower efficiency, language support needed.
**Blackboard:** knowledge sources cooperating via shared blackboard under a control component (speech recognition, AI).

---
## 6.7 TACTICS & PATTERNS — how they relate ⭐
> **Pattern = molecule, tactic = atom.** Patterns solve a problem but are neutral/weak on other QAs → augment with tactics.
- **MVC** uses: increase semantic coherence, encapsulation, use an intermediary, runtime binding.
- **Broker** weaknesses: bottleneck (→ *increase resources*, *maintain multiple copies*), SPOF (→ *redundancy*).
- **Chain of consequences (deck example):** add **Ping/Echo** (availability) → ping-flood security issue + performance overhead → **Increase resources** (cost, utilisation) → **Scheduling policy** (modifiability) → **Use an intermediary** → **Restrict communication paths** (performance) … ends when side-effects become small enough to ignore. *Every tactic has pluses and minuses; nothing is free.*

---
## 6.8 ⭐⭐ COMPARISON TABLES (write these in compare-and-justify questions)

### Table A — Core patterns
| Pattern | Communication mechanism | Coupling | Scalability | Performance/latency | Availability | Modifiability | Complexity | Best-fit |
|---|---|---|---|---|---|---|---|---|
| **Layered** | Synchronous calls downward (allowed-to-use) | Medium (vertical) | Scale layer/tier | Overhead per layer | Depends on tiers | High within layers | Low | Typical business apps, UI+logic+data |
| **MVC** | Observer notifications; user→controller | Low UI↔logic | Via web tier scaling | Good | – | UI changes easy | Low–medium | Interactive UIs/web front-ends |
| **Pub-Sub** | Async events via broker | **Very low** | **High** | Latency variable, non-deterministic | Good with clustered broker | **Very high** (add subscribers) | Medium–high | Event notifications, order status, IoT |
| **Pipe & Filter** | Streams through pipes | Low | High (parallel filters) | Batch/stream throughput good; per-item latency | Per-pipeline | High (recompose) | Low–medium | Validation/ETL pipelines, data transformation |
| **SOA** | SOAP/REST/ESB messaging | Medium (contracts) | Medium (ESB limits) | Overhead (XML, ESB) | ESB SPOF risk | High (service reuse) | High | Enterprise integration, legacy wrapping |
| **Microservices** | REST/gRPC + async events | Low (own DB) | **Very high**, independent | Network hops | High (fault isolation) | **Very high**, independent deploy | **Very high** (ops) | Large, fast-changing, scalable platforms |
| Broker | Mediated request/response | Low | Broker limits | Indirection latency | Broker SPOF | High | Medium | Distributed object/location transparency |
| Client-Server | Request/reply | Medium | Server-centric | Good | Server SPOF | Medium | Low | Web, DB apps |
| P2P | Direct peer calls | Low | High, no centre | Variable | Very high | Medium | High (security) | File-sharing, blockchain |
| Shared-Data | Read/write shared store | High (schema) | Store bottleneck | Good local | Store SPOF | Low–medium | Low | DB-centric/analytics |
| Map-Reduce | Key-value shuffle | Low | **Very high** (batch) | High latency | Fault-tolerant | Medium | High | Petabyte batch analytics |

### Table B — Selection by dominant ASR
| If the driver is… | Choose | Add tactics |
|---|---|---|
| Modifiability of UI | **MVC** | intermediary, encapsulation |
| Separation of concerns/portability | **Layered** | façade, restrict dependencies |
| Decoupled event notification, extensibility | **Pub-Sub** | clustered broker, DLQ, idempotent consumers |
| Throughput of stream processing | **Pipe & Filter** | parallel filters, bounded queues |
| Integration of heterogeneous enterprise apps | **SOA/ESB** | orchestrate, tailor interface, service discovery |
| Independent scaling/deployment, large teams | **Microservices** | API-GW, circuit breaker, service mesh, CI/CD |
| Location transparency | **Broker** | replicate broker |
| Petabyte batch analysis | **Map-Reduce** | replication of blocks, speculative execution |
| Plug-in extensibility | **Microkernel** | – |

### Table C — SOA vs Microservices vs Monolith
| Aspect | Monolith | SOA | Microservices |
|---|---|---|---|
| Deployment unit | Whole app | Services (often large) + ESB | Tiny independent services |
| Data | One shared DB | Often shared | **DB per service** |
| Communication | In-process | ESB, SOAP | REST/gRPC/events |
| Scaling | Whole app | Per service (limited) | Per service, fine-grained |
| Governance | Central | Central, enterprise-wide | Decentralised teams |
| Complexity | Low initially | High | High (ops) |
| Release | Slow, big-bang | Medium | Frequent, independent |
| Best for | Small/simple, early product | Enterprise integration | Cloud-native, scale |

### Table D — Layers vs Tiers
| | Layers | Tiers |
|---|---|---|
| Nature | Logical (module) | Physical/runtime (allocation) |
| Purpose | Separation of concerns | Distribution, scaling, security zones |
| Example | Presentation/Business/Data | Browser – Web server – App server – DB server |

📝 **Exam angle (Q6, Part 13):** *Real-time order-processing app: compare Layered, MVC, Pub-Sub, P&F, SOA, Microservices.* → Recommended: **Microservices (or modular SOA) with Pub-Sub backbone, layered inside each service, MVC at UI, P&F for order validation pipeline.**

---

# PART 7 — INTEGRATION, CLOUD, CACHE, CONTAINERS, SERVERLESS, CAP & FAILURE MANAGEMENT (M8) ⭐

## 7.1 INTEGRATION STRATEGIES ⭐ (Hohpe & Woolf, *Enterprise Integration Patterns*)
**Why integration is hard:** networks unreliable & slow; different platforms/formats; change; legacy. Choose style by **coupling, latency, volume, reliability, skills.**

| Style | How it works | Coupling | Timing | Pros | Cons | Use when |
|---|---|---|---|---|---|---|
| **File Transfer** | One app writes a file (CSV/XML), another reads (batch, SFTP/ETL) | Low technical, high on format | Batch/delayed | Simple, no special infra, works with legacy, large volumes | Stale data, no real-time, error handling & format drift, file locking | Nightly batch (payroll, reconciliation), legacy mainframe |
| **Shared Database** | Apps read/write common DB | **High** (schema) | Real-time-ish | Simple, consistent | Tight coupling, contention, SPOF, schema change hurts all | Same-team apps |
| **RPC (Remote Procedure Call)** — RMI, SOAP, gRPC, REST | Call remote function like local | High temporal coupling (caller waits) | Synchronous | Familiar, immediate response, strong typing (gRPC) | Brittle to failure/latency, fallacies of distributed computing, tight interface coupling | Request/reply queries where caller needs answer |
| **Messaging** (queues/topics — MQ, Kafka, RabbitMQ, SQS) | Async, store-and-forward via channel | **Low** (time & space) | Asynchronous | Reliable, decoupled, load levelling, retry, pub-sub, buffering during outages | Eventual consistency, ordering/duplication issues, debugging, broker ops | Order processing, notifications, microservice events, integration with unreliable links |
| **WebSockets** | Persistent full-duplex TCP channel after HTTP upgrade | Medium | Real-time bi-directional push | Low latency, server push, less overhead than polling | Stateful connections hard to scale/LB, firewall/proxy issues, reconnection logic, security (origin check) | Chat, live dashboards, trading tickers, collaborative editing, gaming, mobile live tracking |
| **API Gateway** | Single entry point to backend services: routing, authN/Z, rate limiting, aggregation, protocol translation, caching, monitoring | Decouples clients from internal topology | Sync front, async behind | Security centralisation, simplified clients (esp. mobile), versioning, throttling | **SPOF/bottleneck**, added latency, can become "god" component | Microservices exposed to web/mobile/partners |

**Recommended strategy for "legacy ERP + external services + mobile + internal apps" (Q7):** layered hybrid —
```
 Mobile/Web/Partners ──► [ API Gateway ] ──► REST/gRPC facades (service layer)
                                │                     │
                                │              [ Anti-corruption / Adapter layer ]──► Legacy ERP (SOAP/DB/MQ)
 Internal apps ◄──► [ Message broker (Kafka/MQ): events, async integration ] ◄──► Legacy (via adapter/CDC)
 Batch partners ──► File transfer (SFTP + ETL) ; Live UI feeds ──► WebSocket gateway
```
Rule of thumb: **API gateway + REST** for external/mobile (sync queries), **messaging** for internal decoupled workflows, **file transfer** only for bulk/batch legacy, **WebSockets** only for real-time push, **avoid shared DB**, and put an **adapter/anti-corruption layer** in front of legacy.

---
## 7.2 CLOUD ARCHITECTURE ⭐
### 7.2.1 NIST essentials
**Characteristics:** on-demand self-service, broad network access, **resource pooling** (multi-tenant), **rapid elasticity**, measured service (pay-per-use).
**Service models:** **IaaS** (VMs, storage, network — consumer = sysadmin; AWS EC2), **PaaS** (runtime/stack — developer; Azure App Service, Google App Engine, Heroku), **SaaS** (end user; Gmail, Salesforce). Responsibility shifts to provider as you go IaaS→SaaS.
**Deployment models:** **Public, Private, Community, Hybrid** (+ multi-cloud).
**Benefits (🧠):** elasticity & scalability, pay-as-you-go (CapEx→OpEx), faster time-to-market, high availability/multi-region, managed services, global reach, built-in security tooling. **Economic justification:** *economies of scale* (power 15–20% of cost, hardware discounts up to ~30%, shared security/DR investment), *higher utilisation* via virtualisation & workload variation, *multi-tenancy* (one version to upgrade/support).
### 7.2.2 Cloud mechanisms & QAs
- **Hypervisor** (manages VMs; page-table mapping), **Virtual machine** (isolated address space, own IP, any OS), **cloud file system** (HDFS: NameNode metadata + DataNodes, 3 replicas; failure cases: client/NameNode/DataNode failures handled by retry, backup NameNode, re-replication), **virtual network**.
- **IaaS components:** cluster manager, persistent object manager, virtual resource manager (gateway), file system manager. Services: auto-reallocation of IP on VM failure, **auto-scaling**.
- **QAs differ in cloud:** **Security** (multi-tenant: inadvertent information sharing, VM escape, side-channel, DoS); **Performance** (auto-scaling lag, make apps self-aware of resource needs); **Availability** (**assume failure** — 1000s of servers; vendors promise e.g. 99.95%, design for the 0.05% e.g., multi-zone replication).
### 7.2.3 AWS design principles 🧠
Scalability · **disposable resources instead of fixed servers** · automation · **loose coupling** · **services not servers** · right database · **remove single points of failure** · optimise for cost · **caching** · security everywhere. Cloud difference: IT assets become programmable resources, global & unlimited capacity, higher-level managed services, security built in.
### 7.2.4 Cloud-native
PaaS, microservices, containers, CI/CD, DevOps, agile, multi-cloud, immutable infrastructure, IaC, 12-factor (stateless processes, config in env, logs as streams, disposability).
### 7.2.5 Multi-tenancy ⭐
**Multi-tenant:** one application instance serves many tenants (customers/organisations) with **logical isolation**. (**Single-tenant:** instance per tenant.)
| Aspect | Single-tenant | Multi-tenant |
|---|---|---|
| Cost | High per tenant | **Low** (shared) |
| Isolation/security | Strong | Needs careful design |
| Customisation | Easy | Limited → configuration/metadata |
| Upgrade | Per tenant | **One upgrade for all** |
| Noisy neighbour | None | Risk → quotas/throttling |
**Data-architecture options (Azure guide):** **(1) Separate database per tenant** (best isolation/customisation, highest cost); **(2) Shared database, separate schema per tenant** (balance); **(3) Shared DB, shared schema with TenantID/OU_ID column** (cheapest, highest density, highest risk of leakage — enforce row-level security). Mix across tiers (e.g., *web tier multi-tenant, app tier single-tenant, data tier multi-tenant* when web processing light, app heavy, data non-confidential; reverse when data confidential).
**Degree of multi-tenancy:** highest (fully multi-tenant SaaS) → middle (small SaaS clusters) → lowest (SaaS single tenant on multi-tenant IaaS/PaaS). Higher degree fits **utilitarian cross-industry workloads** (email, expense reporting, CRM).
**Key design concerns:** tenant identification (subdomain/header/JWT claim), **data isolation & encryption per tenant**, per-tenant customisation (metadata/config, not code branches), **partitioning & sharding by tenant**, noisy-neighbour control (quotas, rate limits, **governor limits** as in Salesforce), per-tenant monitoring/billing, tenant onboarding automation, **availability/scalability/elasticity** (auto-scale per tier), backup/restore per tenant, compliance residency.
**Deployment choices for SaaS (deck exercise – Tailspin survey app / GST filing):** deploy on **PaaS** (fast, managed, auto-scale) vs **IaaS VMs** (more control) vs **containers/Kubernetes** vs **serverless**; choose PaaS/containers for start-ups.
**Amazon's approach to CAP at scale:** to scale you must partition → choose consistency vs availability **per service**: *shopping cart add-to-cart → favour availability* (revenue-producing, errors hidden); *checkout/payment → favour consistency.*

---
## 7.3 DISTRIBUTED CACHE ⭐
- **Definition:** an in-memory data store **spread across multiple nodes** that holds frequently used data so requests avoid slow back-end (DB/API). Examples: **Redis, Memcached, Hazelcast, Ehcache, AWS ElastiCache, Azure Cache, Apache Ignite, CDN (edge cache)**.
- **Levels:** browser/client → **CDN** → API gateway → application local (in-process) → **distributed cache** → DB buffer.
- **Patterns:** **Cache-aside (lazy loading)** (app checks cache; on miss loads DB & populates) — most common; **Read-through**; **Write-through** (write cache+DB synchronously → consistent, slower); **Write-behind/back** (async DB write → fast, risk of loss); **Refresh-ahead**.
- **Eviction/expiry:** LRU, LFU, FIFO, **TTL**; invalidation on update.
- **Distribution:** **consistent hashing** / sharding across nodes, replication for HA, client-side partitioning.
```
 Client ─► [App svc] ──hit──► [ Distributed Cache cluster: node1 | node2 | node3 ]
              │  miss                       ▲ populate / invalidate
              └────────────────► ( Database )
```
| Benefits | Limitations / risks |
|---|---|
| ↓ latency, ↑ throughput, offloads DB, scalability, handles read-heavy spikes, session store for stateless apps | **Stale data / consistency** (invalidation is hard), extra infra & cost, **cache stampede/thundering herd** (use locks, jitter), cold start, memory limits, cache = SPOF if not replicated, security of cached sensitive data, complexity |
**Use for:** product catalogue, session state, reference data (states/districts), leaderboards, rate-limit counters. **Avoid for:** highly volatile, strongly consistent data (balances) unless short TTL/write-through.
**Multi-tenant note:** namespace keys per tenant; per-tenant quotas to avoid cache eviction by noisy tenant.

## 7.4 CONTAINERS ⭐
- **Definition:** lightweight, portable packaging of an application **with its dependencies**, sharing the **host OS kernel**, isolated via namespaces/cgroups (**Docker**; images → containers; registry; **orchestration** by **Kubernetes** — scheduling, auto-scaling, self-healing, rolling updates, service discovery, load balancing).
```
 VM model:        Container model:
 [App][App]       [App][App][App]
 [OS ][OS ]       [Container engine (Docker)]
 [Hypervisor]     [Host OS]
 [Hardware]       [Hardware]
```
| VM | Container |
|---|---|
| Full guest OS, GBs, minutes to boot, strong isolation | Shares kernel, MBs, **seconds** to start, weaker isolation |
**Benefits:** portability ("works on my machine" solved), **consistent environments** dev→prod, fast start & density (cost), microservice fit, CI/CD, immutable deployments, easy scaling. **Limitations:** weaker isolation than VMs (kernel shared — security/multi-tenant risk), orchestration complexity (K8s learning curve), **stateful workloads/persistent storage** harder, networking complexity, image sprawl/vulnerabilities, monitoring overhead.
**In multi-tenant cloud:** namespaces/pods per tenant or tier; resource quotas; network policies.

## 7.5 SERVERLESS ⭐
- **Definition:** cloud execution model where provider manages servers entirely; you deploy **functions (FaaS — AWS Lambda, Azure Functions, Google Cloud Functions)** triggered by events (HTTP via API Gateway, queue message, file upload, timer, DB change) plus **BaaS** (managed DB, auth, storage — DynamoDB, S3, Cognito). **Pay per execution** (ms × memory), **auto-scales to zero and to thousands**.
```
 HTTP ─► [API Gateway] ─► [Function: validate] ─► ( DynamoDB )
 Image uploaded to S3 ─► [Function: thumbnail/ML] ─► [Queue] ─► [Function: notify]
```
| Benefits | Limitations |
|---|---|
| No server management, **automatic elasticity**, cost-efficient for spiky/low traffic, fast development, built-in HA, event-driven | **Cold start latency**, execution time/memory limits, **stateless** (state externalised), **vendor lock-in**, hard local testing/debugging, distributed tracing, cost can exceed containers at sustained high load, limited long-running/ websocket workloads, security misconfig of many small functions |
**Use for:** event processing, glue code, image/video processing, cron-like jobs, bursty APIs, IoT back-ends, ML inference triggers (Azure example: claim image → Azure Function → ML service → notification).
**Containers vs Serverless vs VMs:** control ↔ ops burden: VM (max control, max ops) > containers > serverless (min ops, min control).

---
## 7.6 CAP THEOREM & FAILURE MANAGEMENT ⭐⭐

### 7.6.1 CAP 🧠
A distributed data store can guarantee only **two of three** simultaneously — and since **network partitions are unavoidable**, the real choice during a partition is **C vs A**:
- **Consistency (C):** all nodes see the same data at the same time (linearisable).
- **Availability (A):** every request to a non-failed node gets a (non-error) response.
- **Partition tolerance (P):** system continues despite dropped/delayed messages between nodes.
```
              C
             /\
            /  \         CP: consistent, may refuse requests (MongoDB*, Redis, HBase, Zookeeper)
           / CP \        AP: always answers, may return stale data (Cassandra, CouchDB, DynamoDB, ScyllaDB)
          /------\       CA: only when no partitions (single-node RDBMS) — not realistic for distributed
         A ----- P
```
| Type | During partition | Examples / use cases |
|---|---|---|
| **CP** | Rejects/blocks some operations to keep data consistent | Banking balance/transfer, inventory decrement, booking seats, config store |
| **AP** | Serves possibly stale data, reconciles later (**eventual consistency**, BASE) | Shopping cart, social feeds, product catalogue, "likes", session cache |
| **CA** | Not partition tolerant — theoretical for distributed systems | Single-node RDBMS |
**ACID vs BASE:** ACID = Atomicity, Consistency, Isolation, Durability (RDBMS); BASE = Basically Available, Soft state, Eventually consistent (NoSQL). Extended view **PACELC**: even when no Partition, trade **Latency vs Consistency**.
**Design implications:** choose per feature (cart AP, payment CP); use **idempotent operations**, **versioning/vector clocks**, **conflict resolution** (last-write-wins, CRDTs), **quorum reads/writes (R+W>N)**, **sagas** instead of distributed 2PC, **compensating transactions**, user-visible "pending" states.

### 7.6.2 Failure management in distributed systems — "design for failure" 🧠
Principle: failures are **normal** (network drops, node crashes, slow dependencies, partial failures). Aim: **detect → contain → recover → degrade gracefully**; (Azure "self-healing" principles).
| Mechanism | What it does | Notes |
|---|---|---|
| **Health endpoint monitoring / heartbeat / ping-echo** | Detect failed instances | Liveness vs readiness probes (K8s) |
| **Timeouts** | Don't wait forever | Always set on remote calls |
| **Retry with exponential backoff + jitter** | Handle transient faults | Only for **idempotent** ops; cap retries |
| **Circuit breaker** (closed → open → half-open) | Stop calling failing dependency; fail fast; probe for recovery | Prevents cascading failures (Netflix Hystrix, Resilience4j) |
| **Bulkhead** | Isolate resources (thread/connection pools) per dependency/tenant | One failing component can't sink everything |
| **Fallback / graceful degradation** | Return cached/default/partial result | e.g., Netflix shows generic recommendations |
| **Queue-based load levelling** | Buffer spikes via queue | Protects backend; async |
| **Replication & redundancy (active-active / active-passive), multi-AZ/multi-region** | Survive node/zone/region loss | Netflix stores data in multiple geographic zones |
| **Load balancing + auto-scaling + auto-healing** | Replace failed nodes automatically | Cloud/K8s |
| **Failover (automatic) & leader election** | Promote standby | Split-brain avoidance via quorum |
| **Saga / compensating transaction** | Maintain business consistency across services | Choreography vs orchestration |
| **Idempotency keys & de-duplication** | Safe retries, at-least-once delivery | Payment APIs |
| **Dead-letter queue** | Park poison messages | Operability |
| **Rate limiting / throttling / load shedding** | Protect from overload | API gateway |
| **Chaos engineering** (Chaos Monkey) | Inject failures to test resilience | Netflix Simian Army |
| **Observability** (logs, metrics, distributed tracing, alerts) + **runbooks, backups, DR (RPO/RTO)** | Detect & recover quickly | – |
**RPO** = max data loss; **RTO** = max downtime.
```
 Client ─► [LB] ─► [Service A] ─(timeout+retry+CB)─► [Service B]──► ( DB primary ⇄ replica in other AZ )
                         │ fallback                         ▲
                         └──► cache / default response       └─ queue for async work (DLQ)
```
**Strategy answer for Q8 (partitions + temporary failures):** classify data → AP for cart/catalogue, CP for payments/inventory; async messaging with outbox + idempotent consumers; saga for cross-service transactions; circuit breaker + bulkhead + retries; multi-AZ replicas with quorum; graceful degradation; monitoring and chaos tests.

---

# PART 8 — ARCHITECTING FOR MOBILE (M9)

## 8.1 Design considerations 🧠
Simple UI (large buttons, minimal features); **responsive design**; **compact code** (low CPU/memory/storage/battery); **few layers** for performance; **connectivity handling — store locally, sync later**; security (device loss, secure storage, TLS); varying form factors/OS; app-store distribution; notifications; **battery & data usage**.

## 8.2 Types of mobile apps ⭐ (native vs cross-platform/hybrid vs web)
| Aspect | **Native** | **Cross-platform / Hybrid** | **Mobile Web / PWA** |
|---|---|---|---|
| Tech | Swift/Obj-C (iOS), Java/Kotlin (Android); one codebase per OS | React Native, Flutter, Xamarin, Ionic/Cordova (web UI in native shell) | HTML5, CSS3, JavaScript in browser; PWA = installable, service workers |
| Performance & UX | **Best**, platform look & feel | Good (near-native); some overhead | Lowest; browser-limited |
| Device access (camera, GPS, sensors, background) | **Full** | Most via plugins | Limited |
| Offline | Excellent (local DB) | Good | Possible via PWA service worker/IndexedDB; limited |
| Dev cost/time | **High** (2 teams/codebases) | Medium (shared code) | **Lowest**, one codebase |
| Updates | App-store release + user update | Store (or OTA for some) | **Instant**, no install |
| Distribution | App stores | App stores | URL/search |
| Examples | Maps, WhatsApp, games | Uber, Instagram, Twitter (hybrid) | Flipkart Lite, BookMyShow, MakeMyTrip PWA |
| Best for | Performance/graphics/hardware-intensive | Business apps on both OS with limited budget | Content/e-commerce, wide reach, low friction |
**Mobile-optimised site vs responsive design:** *Responsive* = single site adapting via **flexible grids (% widths), media queries, flexible images** to screen size/orientation; *mobile-optimised* = separate reduced-feature site.

## 8.3 Android architecture & components ⭐
**Platform stack:** Applications → Java API framework → Native C/C++ libraries + **Android Runtime (ART/Dalvik, DEX bytecode)** → **Hardware Abstraction Layer (HAL)** → **Linux kernel & drivers**. Each app runs in its own process/ART instance.
**4 app components 🧠:**
| Component | Role | Example |
|---|---|---|
| **Activity** | UI screen | Book-a-cab screen, appointment list |
| **Service** | Background work (no UI) | Music playback, download, send cab location to server |
| **Content Provider** | Shared persistent data (SQLite, files) | Calendar DB, package list |
| **Broadcast Receiver** | Reacts to system/app events | SMS arrival, high-pulse alert, cab-location event |
**Intents** = messages that start/connect components. **Courier app exercise:** `Activity` (package list) → `ContentProvider`/SQLite (local list) → user marks delivered → `Service` syncs via REST API to central server.
**Uber example:** UI (book cab) = Activity; location updates from backend = Broadcast receiver; send driver location = Service; user config = local DB.
**Other platforms:** iOS (Cocoa Touch, Core services, Core OS), Windows Phone, Tizen (Linux).

## 8.4 Pattern: STORE LOCALLY, SYNC LATER ⭐ (offline-first)
**Problem:** intermittent/poor connectivity (field workers, doctors in clinics, couriers).
**Solution:** app reads/writes to a **local store (SQLite/Realm/IndexedDB)**; changes queued in an **outbox/sync log**; a **background sync service** pushes/pulls deltas when network returns (or on schedule/trigger).
```
 [UI Activity] ◄─► [Local DB (SQLite)] ──► [Sync queue/outbox]
                          ▲                      │ connectivity restored (Broadcast receiver)
                          │                      ▼
                  merge/resolve ◄────────── [Sync Service] ◄──REST/HTTPS──► [API Gateway]─►[Backend DB]
```
**Design decisions:** sync triggers (connectivity change, interval, user action); **incremental/delta sync** with timestamps/version numbers; **idempotent** server APIs (safe retries); **conflict resolution** (last-write-wins, server-wins, field-level merge, manual); **optimistic UI**; local **encryption** of data at rest; data-size limits & purge; retry with backoff; sync status indicator; consistent **eventual consistency** model (AP in CAP terms).
**Benefits:** works offline, faster UI, lower data usage/battery, resilience. **Limitations:** conflict handling complexity, stale data, security of stored data, storage limits, testing difficulty.

## 8.5 Mobile app design for field workers (Q10 skeleton)
Choose **cross-platform (React Native/Flutter) or native Android** + local DB + sync service + REST API gateway + push notifications + MDM/security (device encryption, biometric, token auth); backend: API gateway, microservices, message queue, DB; **ASRs:** offline operation ≥ 8 hrs, sync conflicts resolved automatically, data loss = 0, battery ≤ 5%/hr, secure on lost device.

## 8.6 UI design patterns
Action bars/quick actions, **social login (OpenID/OAuth)**, large touch targets, push notifications, **discoverable/contextual controls** (WhatsApp forward button appears on selection), progressive disclosure, pull-to-refresh, skeleton screens, bottom navigation.

---

# PART 9 — NEW TECHNOLOGIES & THEIR ARCHITECTURE (M10)

## 9.1 Big Data ⭐
**Big data** = data sets so large/complex/fast that traditional tools can't process them. **Vs:** **Volume, Velocity, Variety**, (+ Veracity, Value). Sources: web/social media, e-commerce, sensors/IoT, logs, video, satellites. Structured, semi-, unstructured.
**Use cases:** fraud detection (banking), recommendation/targeted marketing (retail), predictive maintenance, healthcare analytics, call-centre/customer service analytics, weather, smart grid.
**Google lineage:** Google File System → **BigTable** (wide-column store: page contents, anchor text, timestamps; versioning, compression, distribution, fault tolerant, dynamic server addition, load balancing) → inspired **HDFS/Hadoop** (Yahoo) and many NoSQL DBs.

### Big-data ecosystem (layered view)
```
 Sources ─► Ingestion (Kafka/Flume/Sqoop) ─► Storage (HDFS / NoSQL / Object store)
        ─► Processing (MapReduce batch | Spark in-memory | Storm/Flink stream)
        ─► Query/Warehouse (Hive, Presto) ─► Analytics/ML (Mahout, MLlib) ─► Visualisation (Tableau, SAS, dashboards)
```
## 9.2 Hadoop ⭐
Open-source (Apache) framework for **distributed storage + batch processing** on clusters of **commodity hardware**.
- **HDFS:** large files split into **blocks (≈128 MB)**, replicated (default **3**) across **DataNodes**; **NameNode** holds metadata (file→block→node map, in memory); backup/secondary NameNode; **self-healing** (re-replicate on node failure); write pipeline: client → DN1 → DN2 → DN3. Scales to petabytes.
- **MapReduce** (processing); **YARN** (resource manager); ecosystem: **Hive** (SQL), **Pig**, **HBase** (column NoSQL), **Spark**, Oozie, Zookeeper.
- **Use when:** very large batch analytics, log processing, search indexing, data retention/archive, recommendation, text/image analysis. **Not when:** small data, low-latency/interactive/real-time, many small files, iterative ML (disk I/O heavy), transactional workloads.

## 9.3 MapReduce ⭐ (pattern in Part 6.6)
**Example (deck):** average time users spend on each web page over 30 days: each node's **Map** emits `<page, (avg_time, #visits)>` → shuffle/sort by key → **Reduce** computes **weighted average** per page. **Log processing:** Map parses log lines → `<hashtag,1>`; Reduce sums counts.
**Properties:** parallel execution, low latency per node, fault tolerance via re-execution, move computation to data (data locality).
**Trade-offs (MapReduce vs Spark/stream vs MPP SQL):** MR = high throughput, **high latency**, cheap for once-a-day batch, reliable; Spark = in-memory, 10–100× faster for iterative; Stream (Flink/Storm) = millisecond latency for events; MPP/SQL (Snowflake/BigQuery) = interactive SQL analytics.

## 9.4 Real-time analytics ⭐
Analyse data **as it arrives** to enable immediate decisions (fraud blocking, real-time ads, security breach detection, machine-failure prediction, Uber/Ola ops, **Dream11** contest-join rate/payment-gateway status/anomaly alerts).
- **Apache Spark** (in-memory cluster computing; **Spark Streaming** micro-batch), **Apache Storm** (true stream; topology of **spouts & bolts**; Twitter, Groupon, Yahoo), **Flink**; **Kafka** = distributed pub-sub log buffering events.
- **Reference flow (deck):** `Real-time system → Kafka → Storm/Spark → NoSQL/real-time store → BI/dashboards/alerts`.
```
 [Txn systems / apps / IoT] ─► ( Kafka topics ) ─► [Stream processor: enrich, aggregate, rules/ML]
        ─► (Real-time store: Redis/Cassandra/Druid) ─► [Alerts/Dashboards/Action: SMS, block txn]
                         └─► (Data lake / HDFS) ─► [Batch (MapReduce/Spark) for historical models]   ← **Lambda architecture**
```
**Lambda architecture** = batch layer + speed layer + serving layer; **Kappa** = streaming only. **Fraud example:** continuous monitoring of mobile/laptop/ATM events → rule/ML detects suspicious pattern → SMS to customer / block.
**Traditional vs real-time:** ODS → ETL → warehouse (hours/days) vs in-stream processing (ms–s).
**In-memory DB** (SAP HANA, Redis, VoltDB): data in RAM — response-time-critical (telecom, ad networks); with NVRAM, durability improves.

## 9.5 NoSQL ⭐
**Why:** RDBMS good at ACID transactions but scale-up, rigid schema, joins costly at web scale; many apps (IoT, marketing, catalogue, sessions) need **speed, flexibility, horizontal scale** more than transactions. **Features:** **scalable (sharding)**, **fast (in-memory)**, **available (replication)**, semi/unstructured data, schema flexibility; **lack** ACID transactions, joins, locking (→ eventual consistency; application must detect conflicts).
| Type | Model | Examples | Best for |
|---|---|---|---|
| **Key-Value** | key → opaque value, hash access | Redis, Riak, Memcached, DynamoDB | Sessions, caching, user profiles, shopping cart |
| **Document** | JSON/BSON documents, nested, indexed fields | **MongoDB**, CouchDB | Catalogues, CMS, e-commerce, user content (schema flexible) |
| **Column-family (wide-column)** | rows with dynamic columns; column-oriented reads | **HBase, Cassandra, BigTable** | Time series, analytics/aggregation, IoT, large write loads |
| **Graph** | nodes + edges with properties | **Neo4j**, JanusGraph | Social networks, fraud rings, recommendations, knowledge graphs |
**Column store example:** avg electricity usage Jan-2018 in East region → read one "usage for Jan" record instead of all billing records.
**RDBMS vs NoSQL:** schema (fixed vs flexible), scaling (vertical vs horizontal), consistency (ACID vs eventual/tunable), query (SQL joins vs key/document queries), use (transactions vs scale/flexibility). **Polyglot persistence:** use RDBMS for orders/payments + NoSQL for catalogue/session/analytics.
**CAP positions:** Cassandra/CouchDB = AP; MongoDB/Redis/HBase = CP (tunable).

## 9.6 Machine Learning / AI (architecture view)
**Types:** **Supervised** (labelled: regression, classification, decision trees), **Unsupervised** (clustering — customer segmentation, crime-area clusters), **Reinforcement** (learn by interaction), **Deep learning** (neural nets — image, speech, NLP).
**Steps to build ML system:** (1) define business problem, (2) **identify features**, (3) collect/prepare data, (4) select algorithm, (5) train, (6) evaluate/validate (accuracy, precision/recall, overfitting), (7) deploy, (8) monitor/retrain.
**Architecture of ML system (deck example – insurance claim image classification on Azure):** claim image uploaded → **Blob storage** → triggers **Azure Function** → calls **ML model/service** → result + metadata stored → **notification**. Typical pipeline: data sources → ingestion → feature store → training (batch/GPU) → model registry → serving (REST/ batch/ stream) → monitoring (drift) — **MLOps**.
**Tools:** Scikit-learn, TensorFlow, PyTorch, Apache Mahout/Spark MLlib. **Architectural concerns:** data quality, latency of inference, scalability of training, explainability, privacy, model versioning, feedback loops.

## 9.7 Blockchain
**Definition:** **distributed, append-only ledger** of **blocks** linked by cryptographic **hashes**, replicated on all nodes, with agreement via a **consensus mechanism** (Proof-of-Work, Proof-of-Stake, PBFT/Raft for permissioned). **Smart contracts** = code on chain. **Types:** public (Bitcoin/Ethereum), **permissioned/private** (Hyperledger Fabric).
**QAs:** **integrity, immutability, transparency/auditability, decentralised trust, non-repudiation**; weaknesses: **low throughput/high latency, storage growth, privacy, energy (PoW), governance/adoption**.
**Architecture:** clients/wallets → nodes (peer-to-peer) → consensus → ledger + world state → smart-contract execution; off-chain storage for large data; oracles.
**Case:** **IBM–Maersk TradeLens** (shipping/ supply-chain documents shared among ~94 early adopters: ports, carriers, customs) — goal: trust & single source of truth, fewer paper documents/delays. *Lesson:* technology worked but network adoption/competitor trust is hard (platform was later wound down).
**Use when:** multiple untrusting parties need shared, tamper-evident record without central authority. **Not when:** one trusted owner, high-throughput, private data.

## 9.8 IoT
Interconnected devices (sensors/actuators) exchanging data with little human intervention: machine monitoring, smart agriculture (soil moisture), asset/goods tracking, smart home/grid.
```
 Devices/sensors ─► Gateway (protocol translation, edge filtering; MQTT/CoAP/BLE) ─► Cloud ingestion (IoT Hub)
   ─► Stream processing (rules/ML) ─► Storage (time-series/NoSQL/data lake) ─► Analytics & Apps ─► Commands back to devices
```
**Architectural concerns:** constrained devices (battery/CPU), intermittent connectivity (store & forward), scale (millions of devices), **security** (device identity/certs, OTA updates), latency (→ **edge computing**), data volume (filter at edge), interoperability. Azure IoT reference: device → IoT Hub → Stream Analytics/Functions → storage → apps.

## 9.9 Security technologies ⭐
| Tech | What / how | Use |
|---|---|---|
| **TLS/SSL** | Handshake: agree version & ciphers → server sends certificate (public key) → client validates vs trusted CA → shared **symmetric session key** established (RSA/ECDHE) → bulk encryption (AES) + MAC (SHA-256). Gives **privacy + integrity + authentication** | HTTPS, API calls |
| **IPSec** | Packet-level authentication/integrity/confidentiality | VPNs |
| **OpenID (Connect)** | **Authentication** (who are you?) — user logs in at identity provider (Google/Facebook), site gets identity assertion (not password) → **SSO** | "Login with Google" |
| **OAuth 2.0** | **Authorisation** (what may this app access?) — user consents; provider issues **access token** (scoped); app uses token on API; no password sharing | App accessing Google Drive/photos/contacts |
| **LDAP** | Protocol to query a **directory service** (tree-structured: Active Directory, OpenLDAP) for users/groups/credentials; offloads user validation | Enterprise SSO/authN |
| **IAM** | Framework: authentication, authorisation, user lifecycle, RBAC/ABAC, audit, compliance (Azure AD, Okta, IBM) | Enterprise |
| **Firewall** | Filters traffic: packet filtering (IP/port), circuit-level gateways, **application-layer filtering**, IDS/IPS, application fingerprinting, bandwidth allocation | Perimeter |
| **DMZ** | Buffer subnet between internet & internal network hosting public-facing servers (web, API gateway) behind two firewalls → attacker must breach both | Defence in depth |
| **WAF, API gateway, mTLS, secrets vault, SIEM** | Extra layers | – |
**OpenID vs OAuth in one line:** OpenID = *authentication*; OAuth = *authorisation (delegated access)*.

---

# PART 10 — ECONOMIC ANALYSIS OF ARCHITECTURES: CBAM (M11) ⭐

## 10.1 Decision-making context 🧠
- Every architectural decision yields **quality attributes (utility/benefit)** *and* **costs** (and schedule implications, uncertainty/risk). E.g., redundant hardware vs checkpointing-to-disk — both give availability, different cost & level.
- **Economic analysis does not make the decision**; it structures elicitation of **Value For Cost (VFC)** = f(cost, benefit, uncertainty).
- Basis: scenarios → vary response values → **utility** of each response → **utility-response curve** → strategies (cost, effect on several QAs) → compare.

## 10.2 Core formulas 🧠
- **Utility-response curve:** plot QA response level (x) vs utility 0–100 (y). Four anchor points per scenario: **worst-case (utility 0)**, **current**, **desired**, **best-case (utility 100)** — beyond best-case no extra value (e.g., 0.1 s feels instant), below worst-case system is useless.
- **Interpolation:** `U = Ua + (x − xa)/(xb − xa) × (Ub − Ua)`
- **Benefit of a strategy i on scenario j:** `b(i,j) = U_expected − U_current`
- **Total benefit:** `B(i) = Σj b(i,j) × W(j)` (W = scenario weight, highest-priority = 1.0)
- **Value For Cost:** `VFC(i) = B(i) / C(i)` → rank strategies; pick highest until budget/schedule exhausted.
- Weights: stakeholders get **100 votes** to distribute; total votes → relative weights (top = 1.0). Alternative: rank-based weight (highest 1, next (N−1)/N …).
- Cost: use relative estimates ("A costs x, B 2x, C 0.5x"), analogy, function points, interaction-complexity cost models — absolute accuracy is not needed to *rank*.

## 10.3 CBAM — the 9 steps 🧠⭐
| # | Step | Detail |
|---|---|---|
| 1 | **Collate scenarios** | Stakeholders add new scenarios; prioritise (H/M/L) vs business goals; keep **top 1/3** |
| 2 | **Refine scenarios** | Define stimulus-response; elicit **worst, current, desired, best** response |
| 3 | **Prioritise scenarios** | **100 votes each**; keep **top 50%**; weights relative to top (=1.0) |
| 4 | **Assign utility** | Utility (0–100) for the 4 response levels for each scenario |
| 5 | **Develop strategies & expected responses** | List architectural strategies; for each scenario predict response level (and side-effects on other QAs) |
| 6 | **Determine utility of expected responses** | **Interpolate** on curves |
| 7 | **Calculate total benefit** | `B = Σ b×W` |
| 8 | **Choose strategies by VFC subject to cost & schedule** | Rank by `B/C`, pick till budget exhausted |
| 9 | **Confirm with intuition** | Check against business goals; iterate if surprising |

**Stakeholders:** those who can authoritatively state the *utility* of responses (often same as scenario sources). **CBAM benefits:** structured discussion, forces clear scenarios and response goals, quantifies trade-offs, social benefits (shared understanding). **Limitations:** subjective utility numbers, strategy interactions (benefits not additive), cost estimates uncertain, effort.
**Case: NASA ECS (EOSDIS Core System)** — satellite data processing (hundreds of GB/day, 250 standard products, 8 data centres); PM had **limited annual budget** to enhance system; scenarios: reduce hung/lost data-distribution requests, order submission failures, hung/lost orders, etc. 10 architectural strategies; used **consensus voting (100 votes)**, utility by consensus, **VFC ranking** (strategy 1 highest, 3 second; 9 lowest, 8 second lowest) — largely validated stakeholder intuition but made it explicit and defensible.

## 10.4 ⭐ WORKED EXAMPLE (use this format in the exam)
**Scenarios & weights (from 100-vote exercise):**
| Scenario | Weight W | Worst (U=0) | Current | Desired | Best (U=100) |
|---|---|---|---|---|---|
| S1 Performance: response time | **1.0** | 12 s (0) | 1.5 s (**50**) | 0.5 s (**80**) | 0.1 s (100) |
| S2 Availability: uptime | **0.6** | 90% (0) | 99% (**40**) | 99.9% (**85**) | 99.99% (100) |
| S3 Modifiability: change lead time | **0.4** | 10 d (0) | 5 d (**30**) | 1 d (**80**) | 0.5 d (100) |

**Strategies, expected response → interpolated utility → benefit b = U_exp − U_cur**
| Strategy (cost) | S1 | S2 | S3 | Total B = Σ b·W |
|---|---|---|---|---|
| **A. Add distributed cache** (cost 40) | 0.7 s → U = 50 + (1.5−0.7)/(1.5−0.5)×30 = **74**, b = **+24** | no change, b = 0 | 6 d → U = 30 − (1/5)×30 = 24, b = **−6** | 24(1.0) + 0 + (−6)(0.4) = **21.6** |
| **B. Active-passive failover/multi-AZ** (cost 70) | 1.4 s → U = 50 + 0.1×30 = 53, b = **+3** | 99.95% → U = 85 + (0.05/0.09)×15 = 93.3, b = **+53.3** | no change, b = 0 | 3(1.0) + 53.3(0.6) = **35.0** |
| **C. Refactor into microservices** (cost 120) | 1.8 s → U = 50 − (0.3/10.5)×50 = 48.6, b = **−1.4** | 99.5% → U = 40 + (0.5/0.9)×45 = 65, b = **+25** | 1 d → U = 80, b = **+50** | −1.4 + 25(0.6) + 50(0.4) = **33.6** |

**VFC = B/C:** A = 21.6/40 = **0.54**; B = 35.0/70 = **0.50**; C = 33.6/120 = **0.28**.
**Decision:** rank A > B > C. With a budget of 110 → choose **A + B** (cost 110). C deferred (best long-term modifiability but poor VFC now). *Caveat:* benefits of A and B aren't perfectly additive; confirm with intuition (step 9) — and note A's negative effect on modifiability (−6) as a trade-off.

## 10.5 Q12-style: comparing two architectures by cost-benefit (TCO / NPV / ROI)
**Option X:** higher initial cost, low operations. **Option Y:** low initial cost, high maintenance. Example (₹ crore): X initial 1.0, ops 0.2/yr; Y initial 0.5, maintenance 0.5/yr; benefit 0.8/yr each; 5 years; discount 10%.
| Metric | X | Y |
|---|---|---|
| 5-yr undiscounted TCO | 1.0 + 5×0.2 = **2.0** | 0.5 + 5×0.5 = **3.0** |
| PV of running cost (annuity factor 3.79) | 0.2×3.79 = 0.76 | 0.5×3.79 = 1.90 |
| PV total cost | **1.76** | **2.40** |
| PV of benefits (0.8×3.79) | 3.03 | 3.03 |
| **NPV** = PV benefit − PV cost | **1.27** | **0.64** |
| **ROI** = NPV / PV cost | **72%** | **27%** |
| **Break-even** (X cheaper from): 1.0+0.2t = 0.5+0.5t → t ≈ **1.7 years** | | |
**Steps to answer:** (1) define horizon & discount rate; (2) estimate **initial** (design, dev, licences, infra, training, migration), **recurring** (ops, support, cloud, maintenance, change cost), **benefits** (revenue, savings, avoided downtime, time-to-market) and **intangibles** (modifiability, risk, flexibility, vendor lock-in); (3) compute TCO/NPV/ROI/payback; (4) **sensitivity analysis** (horizon, load growth, change frequency, discount rate); (5) risk & uncertainty (options value: Y may be right for a short-lived product/MVP; X for long-lived core); (6) **recommend** with rationale & decision criteria. **Rule:** product life ≥ ~2 years and stable requirements → X; short life/uncertain market → Y (or staged approach: build Y, plan rewrite).
**Other cost lenses:** technical debt, cost of change (modifiability), cost of downtime, cloud economics (CapEx→OpEx, economies of scale).

---

# PART 11 — EMERGING TRENDS, FORWARD ENGINEERING & GOVERNANCE (M12)

## 11.1 WebAssembly (Wasm)
**Binary instruction format** (compile C/C++/Rust/Go/…) that runs at **near-native speed** in a **sandboxed** stack-based VM — browsers first, now **server/edge** (WASI). **Benefits:** performance, portability, security (sandbox, capability-based), language-agnostic, small & fast start (good for edge/serverless/plug-ins). **Limitations:** immature tooling/debugging, limited direct DOM/system access (via JS/WASI), GC/threads maturing, ecosystem. **Use:** heavy browser compute (Figma, video/image editing, games, CAD), plug-in systems, edge functions, alternatives to containers for lightweight isolation.

## 11.2 Service Mesh
**Dedicated infrastructure layer for service-to-service communication** in microservices: a **sidecar proxy** (Envoy) beside each service (**data plane**) + **control plane** (Istio, Linkerd, Consul) configuring them. **Provides:** service discovery, **load balancing, retries/timeouts, circuit breaking**, **mTLS** (zero-trust, encryption/identity), traffic shaping (canary, blue-green, A/B), **observability** (metrics, tracing). **Why:** moves cross-cutting concerns **out of application code** (fits "intermediary", "encapsulate" tactics) → polyglot services, consistent policy. **Costs:** extra hop latency & CPU/memory per sidecar, operational complexity, steep learning, debugging; overkill for few services. **Mesh vs API gateway:** gateway = **north-south** (external→internal); mesh = **east-west** (service↔service).

## 11.3 Edge computing
Process data **near where it is produced** (devices, gateways, local servers, CDN POPs, 5G MEC) instead of a distant central cloud. **Drivers:** **low latency** (autonomous vehicles, AR, industrial control), **bandwidth savings** (filter video/IoT before upload), **offline/intermittent connectivity**, data **privacy/sovereignty**. **Architecture:** Device → Edge node (filter, inference, cache, rules) → Cloud (training, aggregation, long-term storage); sync with store-and-forward. **Challenges:** managing/updating thousands of edge nodes, security of physical devices, limited resources, consistency, observability. **Examples:** CDN edge functions (Cloudflare Workers, Lambda@Edge), retail store analytics, smart factories, telemedicine.

## 11.4 Forward engineering — Object-Oriented & Client-Server architectures (handout item)
**Forward engineering** = moving from requirements/analysis models → architecture → design → code (vs reverse engineering/reconstruction).
- **OO architecture:** derive from **analysis model** (use cases, class/CRC, sequence): identify classes/objects → group into **packages/subsystems/layers** (boundary–control–entity) → define interfaces (information hiding, low coupling/high cohesion, inheritance vs composition) → apply design patterns (GoF) and frameworks. Quality tactics: encapsulation, polymorphism, dependency inversion. Output: class, package, component, deployment views.
- **Client–server architecture:** partition functions among **client (presentation, input validation)**, **application server (business logic)**, **database server (data)**: **2-tier** (fat client ↔ DB: simple, doesn't scale/ hard to maintain), **3-tier/n-tier** (thin client – app server – DB: scalable, maintainable), **thin vs thick client**, stateless servers, middleware (RPC, MQ, ORB), load balancing & replication, security zones (DMZ). Decide **where to put functionality** (costly to change later); evolve to web/SOA/microservices.

## 11.5 Management & governance (M10-ish deck "Management and Governance")
- **PM ↔ architect** is the most important working relationship — respect, coordinate, stay in own sphere. PM trades off **features/schedule/quality** (resist scope creep, use Change Control Board); architect trades off **quality attributes**.
- **Planning:** **top-down schedule** (size-based; convinces management; inherently inaccurate) + **bottom-up schedule** (from first-level decomposition by architecture team; more accurate) → **reconcile** → software development plan; teams formed from it.
- **Team coordination induced by module dependencies:** teams of dependent modules must coordinate on shared interfaces (**Conway's law**); local coordination informal; remote/global dev needs structure (documentation, meetings, electronic media). **Global software development** drivers: cost, skills, local market knowledge.
- **PMBOK** 5 process groups (initiating, planning, executing, monitoring & controlling, closing); 10 knowledge areas (integration, scope, schedule, cost, quality, resource, communications, risk, procurement, stakeholder). Architect's role in each: elicit run-time requirements, WBS, cost gathering, design for quality, risk identification, tech selection.
- **Incremental development:** releases overlap in planning/development/test states; **track progress** by meetings, metrics (size, schedule deviation, productivity, defects, open issues, unmitigated risks, cost-to-complete), risk management.
- **Governance board — 4 responsibilities (deck) 🧠:** (1) a system of **controls** over creation & monitoring of all architectural components/activities (effective introduction, implementation, evolution); (2) a system ensuring **compliance** with internal/external standards & regulatory obligations; (3) **processes** supporting effective management of the above within agreed parameters; (4) practices ensuring **accountability** to a clearly identified stakeholder community, inside and outside the organisation.
- **Summary:** plan (top-down + bottom-up, reconcile) → organise teams → implement/track → govern; PM & architect coordinate; global development needs explicit coordination; management trades off **schedule, function, cost**; larger systems need formal governance.

---

# PART 12 — CASE STUDIES (R14) ⭐

> ⚠️ **Source note:** your lecture decks only *mention* Netflix (multi-zone data replication), Salesforce (multi-tenancy, workflow framework, single app for many customers) and NASA ECS (CBAM). **CIGNA, TripAdvisor and Danske Bank are in the handout but have no slides** — the notes below are built from widely-known public descriptions of these architectures. In the exam, anchor on the **drivers → decisions → tactics → trade-offs** structure; if your case-study reading material contains specific figures, prefer those. Don't quote numbers you can't verify.

## 12.0 Template 🧩
**Context/problem → Architectural drivers (ASRs) → Key decisions (patterns + tactics) → Results → Trade-offs/limitations → Lessons.**

---
## 12.1 NETFLIX — Scaling, caching, reliability ⭐⭐
**Context:** streaming to 200M+ members worldwide; a 2008 database corruption outage in its own data centre triggered the move from a **monolith in private data centres to microservices on AWS**.
**Drivers (ASRs):** **availability** (always-on, survive instance/AZ/region failure), **scalability & elasticity** (evening peaks, global launches), **performance** (fast start, low buffering), **modifiability/deployability** (hundreds of teams ship daily), **cost efficiency**.

```
 Device (TV/phone/browser) ──► [ Zuul API Gateway ] ──► [ Microservices on AWS: Eureka discovery, Ribbon LB, Hystrix circuit breaker ]
        │                                                      │               │
        │ (video bytes)                                  ( EVCache )     ( Cassandra multi-region )
        ▼                                                      │
 [ Open Connect CDN appliances inside ISPs ] ◄── control plane (AWS: login, recommendations, play-request, DRM/licence)
```
**Control plane vs data plane:** AWS handles everything *before* playback (sign-in, browse, recommendations, "Play" request, licensing, choosing best CDN server); **Open Connect** (Netflix's own CDN) streams the video from servers close to the viewer.

| Driver | Decision / pattern / tactic |
|---|---|
| Modifiability, independent scaling | **Microservices**, DevOps/CI-CD (Spinnaker), API gateway (**Zuul**) |
| Availability | **Redundancy across AZs & regions (active-active)**; **data replicated in multiple geographic zones** (Cassandra); stateless services; **regional failover/evacuation** |
| Fault isolation | **Circuit breaker (Hystrix)**, bulkheads, timeouts, fallbacks (generic recommendations / cached list when personalisation fails) — **graceful degradation** |
| Resilience verification | **Chaos Monkey / Simian Army / chaos engineering** — deliberately kill instances/AZs in production |
| Performance | **Caching** (**EVCache**, memcached-based, regionally replicated), **CDN (Open Connect)**, adaptive bitrate streaming, pre-positioning popular content off-peak |
| Scalability | **Auto-scaling groups**, horizontal scale-out, NoSQL (Cassandra) for scale, event streaming (Kafka) for data pipelines |
| Discoverability | **Eureka** service registry (service discovery tactic) |
**Outcomes:** survives AZ/region loss with little user impact; very high deployment frequency; global scale. **Trade-offs/costs:** operational & distributed-systems complexity; eventual consistency; cost of multi-region duplication; need for strong observability/tooling; cultural investment (you-build-it-you-run-it).
**Lessons:** design for failure; cache aggressively; push content to the edge; test resilience continuously; per-feature CAP choices (availability favoured).

## 12.2 SALESFORCE.com — Multi-tenant SaaS/PaaS ⭐⭐
**Context:** CRM delivered as a service to ~150k+ organisations (tenants) on **one shared application**.
**Drivers:** **multi-tenancy economics** (one version to build/upgrade/support — lower help-desk & upgrade cost), **tenant data isolation & security**, **customisability without code forks**, **performance fairness** (no noisy neighbour), **availability**, **seamless releases for all**.
```
 Tenants (Org1..OrgN) ─► [ Load balancer ] ─► [ Pod / Instance: App servers (single codebase, shared) ]
                                                      │ runtime engine reads tenant METADATA
                                                      ▼
                      ( Shared DB: large tables with OrgID column | Universal Data Dictionary (metadata) | search indexes )
```
| Driver | Decision |
|---|---|
| One code base, per-tenant customisation | **Metadata-driven architecture** — custom objects, fields, workflows, UI layouts, rules stored as *metadata*; the runtime engine interprets them (Universal Data Dictionary). **Defer binding** tactic (configuration, runtime binding) |
| Cost/density | **Shared schema, shared tables with OrgID (OU_ID)** — row-level isolation enforced by the platform (deck's "same table contains data of different organisations") |
| Noisy neighbour / performance | **Governor limits** (per-transaction CPU, queries, DML limits on Apex code); resource quotas — "limit event response / bound execution times" tactics |
| Scalability & availability | **Instances/pods** (partitioning tenants across self-contained clusters), replication, disaster-recovery sites |
| Extensibility | **Force.com PaaS**, Apex, **workflow engine** (deck: "workflow framework of Salesforce") and rules, APIs (REST/SOAP/Bulk), AppExchange ecosystem |
| Upgrades | **Single version for all** — release 3×/year with backward-compatible APIs; sandbox testing |
**Trade-offs:** governor limits constrain developers; shared-schema isolation risk (mitigated by platform enforcement); limited low-level control; vendor lock-in; metadata interpretation overhead. **Lessons:** multi-tenancy + metadata gives economies of scale and fast change; isolation and fairness must be engineered in.

## 12.3 CIGNA — SOA implementation (healthcare insurance) ⭐
**Context (generic SOA-adoption story):** large insurer with many **siloed legacy systems** (mainframe, packaged apps) duplicating functions (eligibility, claims, member data) across business lines; need to offer **self-service web portals, partner integration and faster product changes**.
**Drivers:** **interoperability/integration**, **reuse** (avoid duplicate logic), **modifiability** (new products/regulations), time-to-market, preserve legacy investment.
**Decisions:** **SOA with ESB** — wrap legacy functions as **coarse-grained business services** (e.g., *Check eligibility, Get member profile, Submit claim*) with published contracts (**SOAP/WSDL, XML**, later REST), **service registry/repository**, **canonical data model**, **orchestration (BPEL/BPM)** to compose services into processes, **adapters** to mainframes, **governance** (service ownership, versioning, SLAs), **security** (WS-Security, SSO).
**Tactics:** locate (registry), orchestrate, tailor interface, encapsulate, use an intermediary (ESB) → interoperability & modifiability.
**Outcomes/benefits:** reuse of services across channels, faster integration of new channels/partners, incremental legacy modernisation. **Trade-offs/limits:** ESB performance/SPOF, governance overhead, XML/SOAP verbosity, service granularity mistakes, organisational change (shared services ownership).
**Lesson:** SOA succeeds with **governance + service definition tied to business capabilities**, not just technology.

## 12.4 TripAdvisor — SOA / service decomposition (travel reviews & metasearch) ⭐
**Context (generic):** high-traffic consumer site, many content types (hotels, reviews, photos, booking/metasearch partners), many global teams; evolved from a **large monolithic web application** toward **services** to scale and release independently.
**Drivers:** **scalability** (traffic spikes in travel seasons), **performance** (fast search/pages — caching essential), **modifiability** (rapid experiments/A-B tests), **availability**, **integration with many partner/price feeds**.
**Decisions:** decompose by business capability into **services** (search, reviews, media, pricing/booking partners, user profile), **API layer**, **heavy caching** (CDN + in-memory caches), **async messaging** for feeds & aggregation, **horizontal scaling** + load balancing, **read-optimised data stores** (denormalised/NoSQL/search indices), per-service teams.
**Outcomes:** independent scaling of hot services (search), faster releases, resilience. **Trade-offs:** distributed complexity, data consistency between services, cross-service latency, tooling/monitoring.
**Lesson:** decompose along business capabilities; cache read-heavy content; use async integration for external feeds.

## 12.5 Danske Bank — Microservices (banking) ⭐
**Context (generic):** large bank with **monolithic core/channel applications**, strict **regulatory** (security, audit, resilience) demands, need for **digital innovation speed** and 24×7 availability.
**Drivers:** **modifiability/time-to-market**, **scalability** (digital channels), **availability & resilience**, **security & compliance** (PSD2/open banking, audit), cost.
**Decisions:** gradual **strangler-fig migration** from monolith to **microservices** per business capability (accounts, payments, customer, notifications); **API gateway** (open banking/PSD2 APIs), **containers + orchestration (Kubernetes)**, **event-driven architecture** (Kafka) for decoupling and audit trail, **DB per service** with **sagas** for cross-service transactions, **CI/CD & DevOps**, **centralised security** (OAuth2/OIDC, mTLS, secrets mgmt), observability (logs/metrics/tracing), **bulkhead/circuit breaker** for resilience; legacy core retained behind **anti-corruption layer**.
**Outcomes:** faster releases, independent scaling; **Trade-offs:** distributed transactions & consistency (CP needed for money), regulatory audit across many services, operational complexity, skill uplift, security surface area ↑.
**Lesson:** in regulated domains migrate incrementally, keep strong consistency for money movements, invest early in security/observability/automation.

## 12.6 Amazon — CAP at scale (from deck)
Because scaling requires partitioning, choose **consistency vs availability per service**: **add-to-cart = always accept** (AP; revenue producing, errors hidden, reconcile later — Dynamo); **checkout/payment/inventory = consistency**. Also **multi-AZ/zone** replication for the "0.05%".

## 12.7 NASA ECS — CBAM (see Part 10.3)
Limited maintenance budget → 10 strategies ranked by **VFC**; validated intuition; produced structured discussion and justification.

## 12.8 ATAM in practice — Rockwell Collins CAAS (avionics) & Battlefield Control System (R8, R9)
- **CAAS (Common Avionics Architecture System):** product-line architecture of integrated cockpit displays/avionics for many aircraft → ASRs: **modifiability/reuse across products, portability, real-time performance, safety/dependability**; ATAM surfaced **risks & trade-offs** between flexibility (software layering/abstraction) and real-time timing; recommended clarifying interfaces & performance budgets.
- **Battlefield Control System (US Army):** distributed command-and-control nodes → ASRs: **availability/survivability (node loss), real-time performance (message latency), modifiability**; ATAM identified **risks** such as single-point-of-failure communications and replication-consistency approaches, plus **sensitivity points** (message rate, number of nodes) and **trade-off points** (replication ↔ performance).
- Take-away for exam: **ATAM = utility tree → scenario walk-through → risks/sensitivity/trade-offs → risk themes.**

## 12.9 Other course examples to name-drop
**BHIM** (goal → usability + security QAs) · **Aadhaar** (cache reference data + AJAX) · **Dream11** (real-time analytics on Kafka) · **Uber/Ola** (stream processing, Android components) · **Google** (BigTable/GFS → HDFS/NoSQL) · **IBM–Maersk TradeLens** (blockchain) · **Tesco/Ryanair/Marriott** (NoSQL) · **Twitter/Groupon/Yahoo** (Storm) · **Google Maps/Suggest** (AJAX) · **WebArrow** (agile architecting) · **Vanish** (reconstruction).

---

# PART 13 — QUESTION BANK WITH MODEL ANSWERS ⭐⭐

> **How to use:** cover the answer, attempt from memory with the template (A.2–A.5), then compare. Marks in brackets are indicative. Sections: **13.1** your Q1–Q12 (expanded) · **13.2** conceptual · **13.3** scenario/design · **13.4** compare & justify · **13.5** diagram/workflow · **13.6** critical analysis · **13.7** case study.

---
## 13.1 YOUR QUESTION BANK — FULL MODEL ANSWERS

### Q1. Multi-tenant e-commerce platform (high traffic, data isolation, high availability, independent scaling) [≈15 marks]
**1. Functional requirements:** tenant onboarding/management; catalogue & search; cart/checkout; order management; payment integration; inventory; user accounts/auth (per tenant); notifications; reporting/analytics; tenant admin & branding/customisation.
**Quality requirements:** performance, scalability, availability, security & **tenant isolation**, modifiability/customisability, observability, cost efficiency, compliance (PCI-DSS, GDPR).
**2. ASRs (measurable):**
| QA | Scenario | (Value, Difficulty) |
|---|---|---|
| Scalability | During a sale, traffic rises 10× in 5 min; checkout scales out automatically, p95 latency ≤ 2 s | (H,H) |
| Availability | An AZ fails at peak; service continues, ≤ 30 s failover, 99.95% yearly | (H,H) |
| Security/Isolation | Tenant A user attempts to read Tenant B order → denied, logged; zero cross-tenant leakage | (H,H) |
| Modifiability | Add a new payment provider in ≤ 5 person-days without redeploying other services | (H,M) |
| Performance | Catalogue page loads ≤ 1 s at 5,000 req/s | (H,M) |
| Customisability | Tenant changes theme/pricing rules via config, no code | (M,M) |
**3. Proposed architecture:**
```
 Shoppers/Tenant admins ─► [CDN + WAF] ─► [API Gateway: authN (OAuth/OIDC), tenant resolve, rate-limit]
        ─► [Kubernetes cluster, 3 AZs]  ┌ Catalogue svc ┐ ┌ Cart svc ┐ ┌ Order svc ┐ ┌ Payment svc ┐ ┌ Inventory ┐ ┌ Tenant-mgmt ┐
                                          └──────────── each: layered (API / business / data access), stateless ───────────┘
        ─► [Message broker (Kafka): OrderPlaced, PaymentDone, StockReserved]   [Distributed cache (Redis), tenant-namespaced]
        ─► Data: tenant-aware stores — Postgres (shared DB, schema/row-level security by TenantID; premium tenants → dedicated DB),
                 NoSQL for catalogue/cart, search index; object storage; analytics lake
```
**4. Patterns:** **Microservices** (independent scaling/deployment), **multi-tenant data pattern** (shared schema + TenantID + row-level security; DB-per-tenant for large/regulated tenants), **API Gateway**, **Layered** inside each service, **Pub-Sub/event-driven** (order workflow, saga), **CQRS** read models for catalogue, **Multi-tier**, **Client-server**, **Circuit breaker/bulkhead**.
**Tactics:** scalability—stateless services, auto-scaling, load balancer, caching, sharding by tenant; availability—multi-AZ replicas, health checks, failover, retry, graceful degradation; security—JWT tenant claim, authorise per tenant, encryption at rest/transit, secrets vault, audit; modifiability—encapsulate, intermediary, config/metadata-driven tenant customisation; performance—CDN, cache, async, DB indexes/read replicas; per-tenant **quotas/throttling** (noisy neighbour).
**5. Trade-offs ⚖️:** (a) shared schema = lower cost/higher density **vs** weaker isolation → RLS + encryption + premium DB-per-tenant; (b) microservices = scale/modifiability **vs** ops complexity, latency, eventual consistency → API-GW, tracing, saga; (c) caching = speed **vs** staleness; (d) multi-AZ/region = availability **vs** cost; (e) high customisation **vs** single-code-base simplicity → configuration not code forks.
**Justification:** microservices + shared-schema/row-level-secured multi-tenancy on Kubernetes meets the (H,H) ASRs at lowest cost per tenant; evaluate via ATAM + load/chaos tests.

### Q2. Banking application slow at peak; improve performance without compromising security & availability [≈12 marks]
**1. Relevant QAs:** **Performance** (primary: latency/throughput), **Scalability**, **Security** (confidentiality/integrity — must not degrade), **Availability**, plus modifiability/cost/observability.
**2. Conflicts:** caching sensitive data ↔ confidentiality; skipping validation/encryption ↔ security; replication/redundancy ↔ consistency (balances) & cost; async processing ↔ immediate consistency; adding intermediaries (gateway/WAF) ↔ latency; more nodes ↔ larger attack surface; aggressive timeouts/retries ↔ availability (retry storms).
**3. Tactics (first diagnose — profile, find bottleneck):**
| Layer | Tactic |
|---|---|
| Demand | Rate limiting/throttling, prioritise critical txns (payments > statements), bound execution time, move reports/statements to **off-peak/batch** |
| Resources | **Horizontal scaling + load balancer + auto-scaling**, **read replicas**, **connection pooling**, **distributed cache (Redis) for non-sensitive/reference & short-TTL data**, CDN for static, **async queue** for non-critical post-processing (notifications, statements), CQRS read models, DB indexing/query tuning, stored procedures |
| Security-preserving | **TLS offloading** at LB/gateway with hardware crypto, **token/session caching** (JWT validation locally), encrypt only sensitive fields, parameterised queries, WAF tuned, keep authN/authZ in the path |
| Availability-preserving | Multi-AZ replicas, health checks, circuit breaker, graceful degradation (read-only mode), **no cache for balances** or **write-through with short TTL**, active-passive DB failover |
**4. Evaluation:** define measurable scenarios (e.g., "10k concurrent users, txn ≤ 2 s p95, error < 0.1%") → baseline via **load/stress tests** (JMeter/Gatling), APM profiling → apply change incrementally (canary) → regression for security (pen-test, OWASP), failover test, chaos test → compare to baseline; use **utility tree/CBAM** to weigh benefit vs cost; **ATAM-style** walk-through to record sensitivity (cache hit ratio, pool size) and trade-off points (cache TTL: freshness vs speed).
**Recommendation:** profile first; scale out stateless tiers, add read replicas + cache for non-sensitive reads, async the non-critical, keep security controls (offload TLS, not remove).

### Q3. QAW & utility tree — online healthcare platform [≈15 marks]
**A. Conducting the QAW (8 steps):** 1 introductions & method; 2 **business/mission presentation** (sponsors: hospital group goals — e.g., "expand tele-consults from 5k to 50k/month in 12 months, HIPAA/DPDP-compliant"); 3 **architect presents** plan & constraints (existing EHR, cloud, vendors); 4 **identify architectural drivers** (key functions: tele-consult video, e-prescription, EHR access, appointment; QAs: privacy, availability, performance, interoperability; constraints: regulation, legacy EHR) and agree; 5 **scenario brainstorming** with stakeholders (patients, doctors, nurses, admin, compliance officer, IT ops, insurers, regulators); 6 **consolidate** duplicates; 7 **prioritise by voting**; 8 **refine** top scenarios into 6-part form. **Stakeholders/roles:** sponsor, architect, facilitators, domain experts; participants 10–20; 1–2 days. **Outputs:** drivers list, prioritised refined scenarios → feed utility tree and design.
**B. Utility tree (≥ 3 QAs):**
```
Utility
├─ Security & Privacy
│   ├─ Confidentiality ─ (H,M) A physiotherapist can view only orthopaedic notes; all other/ financial data denied and access logged
│   └─ Compliance/Audit ─ (H,M) Every record access is audit-logged & retrievable within 24 h for regulator request
├─ Availability
│   ├─ Fault tolerance ─ (H,H) A node/AZ fails during consults; active sessions reconnect within 30 s; 99.95% annual
│   └─ Maintenance ─ (M,L) DB patch applied with zero downtime (rolling)
├─ Performance
│   ├─ Response time ─ (H,M) Doctor opens patient record < 2 s at peak 2,000 concurrent users
│   └─ Video quality ─ (H,H) 95% consults maintain ≤ 300 ms latency at 10,000 concurrent sessions
├─ Interoperability
│   └─ Data exchange ─ (H,M) Integrate a new lab/ hospital via HL7/FHIR API in ≤ 2 weeks without core changes
├─ Modifiability
│   └─ New feature ─ (M,M) Add a new insurance provider rule set in ≤ 3 person-days
└─ Usability ─ (M,L) Patient books an appointment in ≤ 4 clicks on a mobile phone
```
**Priorities:** (H,H) first = availability of consults, video scalability → drive architecture (multi-AZ, media servers/CDN, autoscaling); then security (H,M).
**Resulting design hints:** microservices + API gateway; RBAC/ABAC + audit service; FHIR-based integration layer; event bus; multi-AZ.

### Q4. ATAM — migrating monolith to microservices with conflicting stakeholder expectations [≈15 marks]
**1. How ATAM is used:** Phase 0 prepare; Phase 1 (steps 1–6) with decision makers: present ATAM → business drivers (faster releases, scale, cost) → architecture (target microservices: gateway, services, per-service DB, broker, K8s) → identify approaches (microservices, API-GW, pub-sub, saga, circuit breaker, replication) → **utility tree** → analyse top scenarios. Phase 2 (steps 7–9): wider stakeholders brainstorm/prioritise → analyse again → present results (risks, non-risks, sensitivity/trade-off points, risk themes).
**2. Stakeholders → scenarios:**
| Stakeholder | Concern | Scenario |
|---|---|---|
| Business/PM | Time-to-market, cost | Release a feature independently in < 1 day (M,M) |
| Customers/Ops | Scalability | Checkout handles 5× load within 5 min (H,H) |
| SRE/Ops | Availability | Failure of Payment svc doesn't stop browsing; recovery < 1 min (H,H) |
| Security/Compliance | Security | Inter-service calls authenticated (mTLS/JWT); no PII exposed in logs (H,M) |
| Developers | Maintainability | New developer builds/tests one service locally in < 1 hr (M,M) |
| Testers | Testability | Contract tests detect breaking API change before release (M,M) |
**3. Sensitivity & trade-off points:** *Sensitivity:* number of synchronous hops per request (latency & availability); circuit-breaker timeout/thresholds; message broker partition count (throughput); cache hit ratio; DB connection pool size. *Trade-off:* **service granularity** (fine = scalability/modifiability ↑, latency/ops complexity ↑); **sync vs async** (immediacy vs resilience/consistency); **DB per service** (autonomy vs transactional consistency/reporting); **security hardening** (mTLS, gateway checks) vs performance; replication factor (availability vs consistency/cost).
**4. Risks (R) / non-risks (NR):** R1 *distributed transactions across Order–Payment–Inventory without saga/ compensation → data inconsistency*; R2 *API gateway/broker as SPOF unless clustered*; R3 *shared DB retained (distributed monolith)*; R4 *inconsistent security enforcement per service*; R5 *lack of observability/tracing → slow incident resolution*; R6 *team skills/DevOps maturity*; R7 *versioning/backward compatibility*. NR: *stateless services behind LB with auto-scale meet scaling scenario*. **Risk themes:** "insufficient attention to data consistency & failure containment", "operational readiness". **Mitigation:** saga + outbox, replicate gateway/broker, DB per service, central auth (OAuth2/mTLS), tracing/metrics, strangler-fig incremental migration, platform team. **Recommendation:** proceed incrementally (strangler), fix R1–R3 before scaling out.

### Q5. Documentation using Kruchten's 4+1 for a multi-team distributed application [≈10 marks]
**Approach:** one *Software Architecture Document* organised by the 5 views, driven by scenarios; each view has a primary diagram + element catalogue + rationale; cross-view mapping table; Design guidelines for integrity.
| View | Content (this application) | Diagrams | Stakeholders |
|---|---|---|---|
| **Logical** | Domain model & services/ APIs: Customer, Order, Payment, Inventory; bounded contexts | UML class/package | Business stakeholders, analysts, end users, developers |
| **Process** | Runtime: service processes/containers, async messaging, concurrency, throughput, sync points, failure handling | Sequence, activity, communication diagrams | Integrators, performance engineers, testers |
| **Development** | Repo structure, modules/layers, dependencies, build & CI/CD, team ownership | Package/component diagram, layer diagram | Developers, build engineers, project managers |
| **Physical** | Nodes, clusters, AZs, network, LB, DB, firewalls/DMZ, scaling | Deployment diagram | Infrastructure/DevOps, security, sysadmins |
| **Scenarios (+1)** | "Place order", "Payment fails", "Node failure", "Scale-out at sale" traced across views | Use-case + sequence | All; evaluators/testers |
**Correspondence:** Logical → Development (service → repo/ team) ; Logical → Process (service → process/container) ; Process → Physical (container → node).
Mention: not all views needed; modern notations (UML, C4, Mermaid as code); keep docs in Git; address stale/ inconsistent views with automated checks.

### Q6. Pattern selection — real-time order-processing app [≈15 marks]
| Pattern | Suitability for real-time order processing | Communication | Advantages | Limitations |
|---|---|---|---|---|
| **Layered** | Good *inside* a service (API/business/data) — not the whole system | Sync calls downward | Separation of concerns, simple, testable | Overhead, monolithic scaling, limited real-time/event handling |
| **MVC** | UI/admin front-end only | Observer + user actions | Multiple views, UI evolves independently | Not a system-level pattern; no help for back-end scale |
| **Publish-Subscribe** | **Excellent** for order events (OrderPlaced → inventory, payment, shipping, notification) | Async events via broker/topic | Decoupling, extensibility, scalability, real-time fan-out | Ordering/duplicates, eventual consistency, tracing, broker SPOF |
| **Pipe & Filter** | Good for **order validation/enrichment/fraud-check pipelines** and batch/stream processing | Streams through pipes | Reuse, parallelism, recomposition | Poor for interactive request/response; latency; shared state hard |
| **SOA** | Possible for integrating enterprise/legacy & partner systems | SOAP/REST/ESB | Interoperability, reuse | ESB bottleneck/complexity, heavy protocols, governance |
| **Microservices** | **Excellent** for independent scaling of order, payment, inventory | REST/gRPC + async events | Independent scale/deploy, fault isolation | Ops complexity, distributed data/transactions, latency |
**Recommendation (hybrid):** **Microservices backbone** with **pub-sub (Kafka) event bus** for order lifecycle events; **layered** structure inside each service; **pipe-and-filter** for the order validation/fraud pipeline; **MVC/SPA** at the UI; use **API gateway**; SOA/ESB only where legacy integration requires. Justify with ASRs: real-time latency (async events, cache), independent scaling (microservices), extensibility (pub-sub), reliability (saga, idempotent consumers, DLQ).

### Q7. Integration strategies — legacy app ↔ external services, mobile clients, internal apps [≈12 marks]
Use the table in **7.1**. **Recommendation:** layered hybrid — **API Gateway + REST facade** for mobile & external partners (security, throttling, aggregation, versioning); **messaging (broker)** for internal decoupled asynchronous integration & event propagation (change-data-capture from legacy); **adapter/anti-corruption layer** wrapping legacy (SOAP/DB/MQ/file); **file transfer** only for bulk/batch legacy and partner feeds; **WebSockets** only where live push is needed (status tracking, dashboards); **RPC (gRPC)** for low-latency internal sync calls; **avoid shared DB integration**. Justify: legacy limitations → wrap instead of rewrite; mobile needs coarse APIs & offline tolerance; internal apps need loose coupling & reliability; security centralised at the gateway. Mention risks: gateway SPOF (cluster), message duplication (idempotent consumers), schema/versioning governance.

### Q8. Distributed application — partitions & temporary failures: CAP + failure-management strategy [≈12 marks]
**CAP implications:** partitions are inevitable → choose **per use-case**: **CP** for payments, balances, inventory decrements (reject/queue operations during partition); **AP** for catalogue, cart, recommendations, sessions (serve stale; reconcile later). Accept **eventual consistency** with conflict resolution (LWW/CRDT/merge), quorum configurations, version vectors; show "pending" states to users.
**Strategy:** (1) **Classify data/ops** by consistency need; (2) **Detect** — health checks/heartbeats, timeouts; (3) **Contain** — circuit breaker, bulkheads, rate limiting; (4) **Recover** — retry with exponential backoff + jitter for idempotent ops, idempotency keys, **outbox pattern + message queue with at-least-once & dedupe**, **sagas** with compensations, DLQ; (5) **Redundancy** — replicate DB across AZs (quorum), stateless services behind LB, auto-healing; (6) **Degrade gracefully** — cached/default responses, read-only mode; (7) **Reconcile** — background anti-entropy, reconciliation jobs; (8) **Test & observe** — chaos engineering, distributed tracing, alerts, RTO/RPO drills. Include the failure-management diagram from 7.6.

### Q9. Multi-tenant cloud app — distributed caching, containers, serverless [≈12 marks]
**Design:** CDN → API-GW (tenant resolution, authN, throttling) → containerised microservices on Kubernetes (per-tier autoscaling) → **distributed cache (Redis cluster, tenant-prefixed keys)** → tenant-aware data tier (shared schema + TenantID/RLS; DB-per-tenant for premium) → **serverless functions** for event-driven/bursty tasks (report generation, file/image processing, webhooks, tenant-onboarding, scheduled jobs) → queue/event bus.
| Technology | Role | Benefits | Limitations |
|---|---|---|---|
| **Distributed cache** | Reduce DB load/latency; session store for stateless apps; per-tenant rate counters | Speed, scalability, resilience to spikes | Staleness/invalidation, extra cost/infra, stampede, security of cached tenant data (namespace keys), SPOF if unreplicated |
| **Containers** | Package & run microservices consistently; orchestrate scaling/self-healing | Portability, density, fast start, CI/CD, tenant/tier isolation via namespaces/quotas | Weaker isolation than VMs (kernel shared), orchestration complexity, stateful storage, security hardening |
| **Serverless** | Spiky/event-driven workloads, glue code | No ops, auto-scale to zero, pay-per-use | Cold starts, time/memory limits, statelessness, lock-in, debugging, cost at steady high load |
**Also address:** isolation, noisy neighbour (quotas), per-tenant encryption/backup, availability (multi-AZ), observability per tenant, cost allocation.

### Q10. Mobile app for field workers with intermittent connectivity [≈12 marks]
**Type choice:** **Native** best performance/device access/offline but 2 codebases; **cross-platform (Flutter/React Native)** one codebase, good offline + device access, lower cost — **recommended** for typical business field app; **web/PWA** lowest cost but limited offline/device features/performance — only if reach/zero-install > features. (Table 8.2.)
**Architecture:** UI (Activity/screens) ↔ **local SQLite/Realm (source of truth for UI)** ↔ **sync queue/outbox** ↔ **sync service** (background; triggered by connectivity broadcast/interval) ↔ **REST API via gateway** ↔ backend services/DB; push notifications; secure storage & token auth; MDM.
**Store-locally-and-sync-later:** write locally first (optimistic UI); queue changes with timestamps/version/ID; on connectivity, push batch + pull deltas; **idempotent APIs**, **conflict resolution** (LWW/ server-wins/ merge/ manual); retries w/ backoff; sync status UI; encrypt local DB; purge old data. ASRs: works offline ≥ 8 h, no data loss, sync < 1 min after reconnect, battery ≤ 5%/h. **Trade-offs:** stale data & conflicts vs availability; local security risk vs convenience; sync frequency vs battery/data.

### Q11. Real-time analytics over large data volumes — role of NoSQL, Hadoop, MapReduce, real-time tech [≈12 marks]
```
 Sources (apps, logs, IoT, txns) ─► Kafka (ingest/buffer)
      ├─► Stream layer (Spark Streaming/Storm/Flink): cleanse, enrich, window aggregates, rules/ML scoring ─► NoSQL/in-memory store (Cassandra/Redis/Druid) ─► dashboards, alerts, APIs   [SPEED layer]
      └─► HDFS/Data lake ─► MapReduce/Spark batch (historical aggregates, model training) ─► Hive/warehouse/NoSQL serving [BATCH layer]
```
- **NoSQL** (key-value, column, document): low-latency, horizontally scalable serving store for real-time results & high write rates, flexible schema.
- **Hadoop (HDFS)**: cheap, fault-tolerant storage for raw history (petabytes); replication ×3.
- **MapReduce**: batch processing of full history (cost-efficient, reliable but high latency) — nightly aggregates, model retraining; not for real-time.
- **Real-time technologies (Kafka + Spark Streaming/Storm/Flink)**: event ingestion & in-memory stream processing (ms–s latency).
**Pattern:** **Lambda architecture** (batch + speed + serving) or **Kappa** (stream-only). **Trade-offs:** latency vs accuracy/completeness; complexity of two code paths; cost of memory; eventual consistency; data quality/ ordering/late events; skills. Use case example: real-time fraud detection (block suspicious txn, SMS customer).

### Q12. Economic analysis — high initial/low operating vs low initial/high maintenance [≈10 marks]
Use **10.5** (TCO, NPV, ROI, break-even ≈ 1.7 yrs, sensitivity) and, for quality-driven decisions, **CBAM** (10.3–10.4): scenarios → utility curves → benefit → **VFC = B/C**. **Recommendation:** if horizon > break-even and requirements stable → Option X (higher upfront/lower run-cost); if horizon short/uncertain → Y; check risk, flexibility, lock-in; do sensitivity analysis; present as a decision to stakeholders (economic analysis informs but doesn't decide).

---
## 13.2 CONCEPTUAL / DESCRIPTIVE QUESTIONS (2–6 marks each)

**C1. Define software architecture and explain why it matters.** [5]
Set of structures needed to reason about the system: elements, relations, properties of both (Bass et al.). Matters because it enables/inhibits QAs, manages change (80% of cost post-deployment), allows early analysis, communication, carries hardest-to-change decisions, constrains implementation, mirrors organisation (Conway), supports cost/schedule estimation, reuse/product lines. (Give 6–8 of 1.2.)

**C2. Differentiate structure and view; name three categories of structures with examples.** [5]
Structure = set of elements & relations as they exist; view = representation of a structure for stakeholders. Module (decomposition, uses, layers, class), C&C (client-server, service, process, shared-data), Allocation (deployment, implementation, work-assignment). Say what each is "useful for".

**C3. Explain a quality attribute scenario with an example.** [4]
6 parts: source, stimulus, artifact, environment, response, response measure. *"Under peak load (env), 5,000 users (source) submit orders (stimulus) to the order service (artifact); orders are processed (response) with p95 ≤ 2 s (measure)."* General vs concrete scenarios.

**C4. Explain tactics and how they relate to patterns.** [4]
Tactic = single design decision influencing a QA response (atom); pattern = bundle of decisions (molecule). Patterns are weak/neutral on other QAs → augment with tactics (e.g., broker + redundancy). Each tactic has side effects → chain (ping/echo → ping-flood → ...) ends when side-effects negligible.

**C5. What is an ASR? How are ASRs identified?** [5]
Definition + indicators (business value/risk, influential stakeholder, unique, QoS differs, past overruns) + five sources (requirements, design-decision leads, QAW interviews, PALM business goals, utility tree).

**C6. Define evolutionary architecture and its significance in agile projects.** [4–6]
Architecture that adapts incrementally to changing requirements/technology via incremental design, refactoring, modular/layered structure, automated tests/CI-CD & fitness functions; supports "welcome changing requirements"; balances up-front (sweet-spot ~20–40% for large projects) with emergent design; manages uncertainty & risk.

**C7. Define sensitivity point, trade-off point, risk, non-risk (ATAM).** [4] — see 5.2.2 table with examples.

**C8. State CAP theorem and give examples of CP and AP systems.** [4] — see 7.6.1.

**C9. What is multi-tenancy? Compare data-isolation options.** [5] — see 7.2.5.

**C10. Explain Microkernel and Reflection patterns briefly.** [4] — see 6.6b.

**C11. Differentiate OpenID, OAuth and LDAP.** [3] — OpenID: authentication/SSO via identity provider; OAuth: delegated authorisation with scoped tokens; LDAP: protocol to query directory service (users/groups) for authentication.

**C12. Differentiate layers and tiers; thin and thick clients.** [3] — Table D in 6.8.

**C13. What are the benefits of cloud-based approach for architects? What QAs change in cloud?** [5] — 7.2.1–7.2.2.

**C14. Explain Map-Reduce with an example.** [4] — 9.3 (average time per web page).

**C15. Explain "architecture reconstruction" and its phases.** [4] — 5.5.

**C16. Why is documentation important & what are the challenges/best practices?** [6–10] — 4.1–4.2 (mid-sem Q2 style).

---
## 13.3 SCENARIO-BASED DESIGN QUESTIONS (8–15 marks) — skeleton answers

### S1. Ride-hailing / food-delivery platform (Uber/Swiggy)
**ASRs:** real-time location updates (≤ 2 s), matching latency (≤ 3 s), **availability** (99.95%), scalability (peak hours/ events), payment correctness, mobile offline tolerance, security/privacy of location.
**Architecture:** mobile apps (Android: Activity UI, Service sends location, Broadcast receiver gets driver position) → API gateway → microservices (user, driver, matching/dispatch, pricing, trip, payment, notification) → **Kafka** event bus (location/trip events) → **stream processing** (ETA, surge pricing, anomaly) → **geo-index store (Redis/Geo/ Cassandra)** for driver locations; **WebSockets/push** for live tracking; SQL for payments (CP), NoSQL for trips/locations (AP); cache; multi-AZ; CDN for static; maps via external API.
**Tactics/patterns:** pub-sub, microservices, CQRS, sharding by city/geohash, circuit breaker, idempotent payments (saga), graceful degradation (fallback ETA).
**Trade-offs:** location freshness vs battery/bandwidth; AP for locations vs CP for payments; matching accuracy vs latency.

### S2. Online banking migration to cloud (regulated)
**ASRs:** security & compliance (PCI/RBI), availability, data residency, auditability, performance, RPO≈0 for transactions.
**Design:** **hybrid cloud** (core ledger on-prem/private; digital channels, analytics in public cloud), DMZ + WAF, API gateway (OAuth2/OIDC, mTLS), layered/microservices, **CP database** for ledger with synchronous replication, event streaming for audit, HSM/KMS, IAM, SIEM, DR in second region (RTO/RPO defined), **strangler-fig** migration; **IaaS/PaaS** mix, containers; encryption at rest/in transit; tokenisation.
**Trade-offs:** security/compliance vs agility/ cost; latency between on-prem core and cloud; vendor lock-in; shared responsibility model.

### S3. Video-streaming platform (Netflix-like) — see 12.1.

### S4. Multi-hospital management system (multi-tenant SaaS)
**ASRs:** tenant isolation/ privacy (HIPAA/DPDP), availability for critical care, interoperability (HL7/FHIR), configurability per hospital, audit, performance for record retrieval.
**Design:** multi-tenant SaaS: shared app tier, **schema-per-tenant or DB-per-hospital-group** (health data sensitivity), metadata-driven config (Salesforce-style), API gateway + FHIR integration layer, RBAC/ABAC + consent, audit service, event bus (ADT events), cache for reference data, multi-AZ DR, mobile apps with store-locally-sync-later for ward rounds. **Trade-offs:** isolation vs cost; customisation vs single code base.

### S5. IoT smart-agriculture platform
Sensors (soil moisture, nutrients) → gateway (MQTT, edge filtering, store-and-forward for poor connectivity) → IoT hub/Kafka → stream processing (rules: irrigate if moisture < x) → time-series/NoSQL store + data lake → ML (yield/irrigation models) → dashboards/mobile apps → commands to actuators. ASRs: battery life, intermittent connectivity, scalability (100k sensors), security (device identity/OTA), latency for control loops (→ edge). **Trade-offs:** edge vs cloud processing; data volume vs cost; security vs constrained devices.

### S6. Ticket booking with flash-sale spikes (high concurrency)
**ASRs:** 50× spike, no overselling (consistency), fairness, p95 ≤ 2 s, availability.
**Design:** CDN + virtual **waiting room/queue**; API-GW rate limit; **queue-based load levelling** (SQS/Kafka); **inventory service CP** with atomic decrement/optimistic locking/ reservation with TTL; cache for read-heavy event pages; **autoscaling**; separate read/write paths (CQRS); idempotent payment; graceful degradation (disable non-essential features). **Trade-offs:** consistency vs throughput; user experience (queue wait) vs system protection.

### S7. Real-time fraud detection for a bank — see Q11 + stream flow in 9.4; ASR: decision ≤ 200 ms on each txn; ML scoring in stream, rules engine, feedback loop, SMS/blocks, explainability; trade-off false positives vs false negatives, latency vs model complexity.

### S8. GST returns-filing SaaS (deck exercise)
Users upload Excel/enter data; software computes and files with government system. ASRs: correctness/audit, security of tax data, peak load near filing deadlines (**elasticity**), integration with Govt API (retry, queue, idempotency, availability of external system), multi-tenancy. **Design:** SaaS on **PaaS/containers** with autoscaling; upload → object storage → **async batch/serverless processing** → validation pipeline (**pipe & filter**) → filing service with queue + retry + status tracking; multi-tenant shared schema with TenantID (or DB-per-large-client); encryption, audit log, RBAC; **deployment options:** IaaS VMs (control), PaaS (speed), containers (portability), serverless (spiky processing). **Trade-offs:** cost vs isolation; sync vs async filing; dependence on external gov API.

---
## 13.4 COMPARE-AND-JUSTIFY QUESTIONS

### X1. Monolith vs SOA vs Microservices — Table C (6.8). Justify: choose microservices when independent scaling/deployment, large teams and ops maturity; monolith for small team/early product; SOA for enterprise integration.
### X2. Native vs Hybrid vs Web — Table 8.2. Justify per scenario (games → native; business field app → cross-platform; content/e-commerce reach → PWA).
### X3. SQL vs NoSQL (and types)
| | RDBMS | NoSQL |
|---|---|---|
| Model/Schema | Tables, fixed schema, joins | Key-value/doc/column/graph, flexible |
| Consistency | **ACID** | BASE/eventual (tunable) |
| Scaling | Vertical (some read replicas) | **Horizontal sharding** |
| Queries | Rich SQL, joins | Simple key/doc queries |
| Best for | Transactions, reporting, integrity (banking, orders) | Scale, flexibility, speed (catalogue, sessions, IoT, social) |
Justify: **polyglot persistence.**
### X4. VM vs Container vs Serverless — control vs ops burden; startup time; isolation; cost model (always-on vs pay-per-exec); statefulness; lock-in. Choose: legacy/ strong isolation → VM; microservices → containers; spiky event-driven → serverless.
### X5. Hadoop/MapReduce vs Spark vs Stream processing (Storm/Flink) — batch disk-based high-latency cheap vs in-memory fast iterative vs true streaming ms latency. Real-time fraud → stream; nightly logs → MR/Spark.
### X6. Integration styles — table 7.1.
### X7. Sync (REST/RPC) vs Async (messaging) — coupling, latency, reliability, consistency, debugging; choose async for decoupling/spikes, sync for immediate answers.
### X8. Client-side vs server-side caching vs distributed cache vs CDN — location, shared/not, invalidation, typical content.
### X9. Layered vs Microkernel vs MVC vs Pipe&Filter — Table A; justify by dominant ASR (Table B).
### X10. Active vs passive vs cold redundancy — hot spare (parallel processing, fastest failover, highest cost), warm (checkpointed, minutes), cold (manual, cheapest).
### X11. OpenID vs OAuth vs SAML/LDAP — authN vs authZ vs directory.
### X12. CP vs AP — table in 7.6.1; choose by business impact of stale data vs unavailability.

---
## 13.5 DIAGRAM / WORKFLOW QUESTIONS (very basic text diagrams)

**D1. Draw a utility tree for an e-commerce site (≥ 3 QAs, with (H,M,L) ratings).** [6]
```
Utility
├─ Performance ── Latency ──── (H,M) Search results < 1 s at 5,000 users
│               └ Throughput ─ (H,H) 1,000 orders/s at flash sale
├─ Availability ─ Fault tolerance ─ (H,H) AZ failure → no visible downtime, failover < 30 s
├─ Security ──── Data protection ── (H,M) Card data tokenised; no PAN in logs; PCI-DSS
├─ Modifiability ─ New feature ───── (M,M) Add payment provider ≤ 5 person-days
└─ Usability ──── Checkout ───────── (M,L) Checkout in ≤ 3 steps on mobile
```
**D2. Draw a layered architecture for a bank/Departure-Control system and identify techniques per layer.** [6] — use 5.1 diagram: Presentation (UI, AJAX, caching, responsive), Service (APIs for airlines/partners), Business (check-in, seat assignment, baggage — façade, rules/workflow), Data (ORM, pooling, parameterised SQL) + cross-cutting (security, logging).
**D3. Draw the 4+1 view diagram and map stakeholders.** [6] — 4.4 diagram + table.
**D4. Draw a 3-tier cloud deployment (physical view).** [5]
```
 Internet ─► [CDN] ─► [WAF/Firewall] ─► (DMZ) [Load Balancer] ═══════════ AZ-1 ══════ AZ-2 ══════
                                              ▼                       [Web tier x N]  [Web tier x N]
                                      [App tier: containers x N]  ─►  [Cache cluster]
                                              ▼ (inner firewall)       (DB primary) ⇄ (DB replica)
```
**D5. Draw publish-subscribe flow for order events.** [4] — 6.4 diagram (+ DLQ and idempotent consumers).
**D6. Draw ATAM workflow.** [5]
```
 Phase 0 Prepare ► [1 Present ATAM] ► [2 Business drivers] ► [3 Architecture] ► [4 Approaches] ► [5 Utility tree] ► [6 Analyse approaches]
 Phase 2 ► [7 Brainstorm scenarios] ► [8 Analyse approaches] ► [9 Present results: risks, non-risks, sensitivity, trade-offs, risk themes]
```
**D7. Draw the CBAM workflow** — 9 steps list as a flow: `Collate → Refine → Prioritise → Assign utility → Strategies & expected response → Utility of expected → Total benefit → VFC ranking → Confirm`.
**D8. Draw store-locally-sync-later; draw the mobile courier app architecture** — 8.4 diagram + Activity/SQLite ContentProvider/Service/REST.
**D9. Draw circuit-breaker state machine** `CLOSED —failures>threshold→ OPEN —timeout→ HALF-OPEN —success→ CLOSED / —failure→ OPEN`.
**D10. Draw Lambda architecture** — 9.4 / Q11 diagram.
**D11. Draw MapReduce data flow** `Input splits → Map (parallel) → shuffle/sort by key → Reduce → Output`.
**D12. Draw big-data/Hadoop HDFS write path** `Client → NameNode (block locations) → DN1 → DN2 → DN3 (pipeline), acks back`.

---
## 13.6 CRITICAL-ANALYSIS QUESTIONS (ATAM style, risks, trade-offs)

**K1. A proposed design: single relational DB, synchronous REST calls between 8 services, one API gateway instance, caching of account balances for 10 min. Analyse risks and trade-offs.** [12]
- **Risks:** SPOF at DB & gateway (availability); **synchronous chains** → cascading failures & latency (availability/performance); shared DB = **distributed monolith**, coupling (modifiability); **stale balances for 10 min** (consistency/ correctness/security-fraud); no circuit breakers/ timeouts; no tenant/ service-level security; observability gap.
- **Sensitivity points:** number of sync hops, cache TTL, DB connection pool/ capacity.
- **Trade-off points:** cache TTL (performance ↔ consistency); sync vs async (simplicity/ immediacy ↔ resilience); gateway centralisation (security/simplicity ↔ SPOF/ bottleneck).
- **Non-risk:** stateless services can scale horizontally.
- **Recommendations:** replicate DB/ gateway across AZs; DB per service; async events for non-critical steps; circuit breaker + timeouts + bulkheads; no caching of balances or write-through/ very short TTL; tracing; threat model.

**K2. Analyse trade-offs of introducing a distributed cache.** [6] — +latency/throughput/DB offload/scalability; −staleness, invalidation complexity, additional failure mode (cache down → DB overload → need fallback/ warm-up), cost, security of cached data, stampede. Sensitivity: hit ratio, TTL. Mitigate: cache-aside, TTL+jitter, replication, stampede locks, namespaces.

**K3. ATAM scenario walk-through: "Payment gateway is unavailable during checkout."** [8]
Stimulus: external payment provider timeout; Environment: peak load; Artifact: Payment service; Response: order stays `PENDING`, user informed, retries via queue, fallback to secondary provider; Measure: no lost orders; ≥ 99% completed within 10 min after recovery.
Approaches: timeout + retry w/ backoff, **circuit breaker**, **queue**, idempotency key, saga/compensation, secondary provider (redundancy).
Risks: duplicate charges if retries not idempotent; stuck pending orders without reconciliation; retry storms. Sensitivity: timeout/ retry counts. Trade-off: user-visible latency vs completion rate; strict consistency vs availability. Non-risk: circuit breaker prevents thread exhaustion.

**K4. "Make everything serverless" is proposed for a high-traffic, steady-load streaming service. Critique.** [6] — cold starts, execution limits, cost at constant high load (containers/VMs cheaper), vendor lock-in, stateful/ long-running/ websocket needs, observability. Recommend hybrid: containers for steady core, serverless for spiky event tasks.

**K5. Critique: "Use microservices because Netflix does."** [6] — Netflix's context (hundreds of teams, DevOps maturity, scale); costs (ops, data consistency, latency); small team/ unclear domain boundaries → modular monolith first; adopt incrementally (strangler); decide by ASRs & team capability (Conway), not fashion.

**K6. Evaluate a layered architecture where UI code calls the DB directly "for performance".** [5] — architectural drift; violates layering, security (SQL injection/ no central authorisation), modifiability; use caching/ optimised data-access, façade, enforce via fitness functions; allow documented exceptions only with ADR.

**K7. Identify risks of a P2P/blockchain-based solution for a single-company inventory system.** [5] — unnecessary decentralisation, low throughput/ high latency, storage growth, privacy, governance; use conventional DB; blockchain only for multi-party untrusted record-keeping.

**K8. Which architectural trade-offs are inherent in a multi-tenant SaaS?** [6] — isolation↔cost, customisation↔single codebase, noisy-neighbour fairness↔utilisation, upgrade-all-at-once↔tenant stability, data residency↔global scale.

---
## 13.7 CASE-STUDY QUESTIONS

**CS1. Explain how Netflix achieves scalability, caching and reliability.** [10] — 12.1: microservices on AWS; Zuul/Eureka/Hystrix; EVCache; Open Connect CDN; multi-AZ/region active-active data replication (Cassandra); Chaos Monkey; auto-scaling; trade-offs.

**CS2. Describe Salesforce's multi-tenant architecture and the QAs/trade-offs.** [10] — 12.2: metadata-driven, shared schema OrgID, governor limits, pods, workflow engine, single version for all; trade-offs.

**CS3. Discuss SOA at CIGNA: drivers, patterns, tactics, outcomes, limitations.** [10] — 12.3 (ESB, registry, orchestration, wrap legacy; governance).

**CS4. Contrast SOA at CIGNA/TripAdvisor with microservices at Danske Bank.** [10]
| | CIGNA/TripAdvisor (SOA-style) | Danske (microservices) |
|---|---|---|
| Goal | Integrate/reuse legacy & services, new channels | Speed, scalability, resilience in digital banking |
| Granularity | Coarse business services | Fine-grained capabilities |
| Middleware | ESB, registry, orchestration | API gateway, event streaming, K8s, mesh |
| Data | Often shared/ canonical | DB per service + saga |
| Governance | Central | Decentralised teams + platform guardrails |
| Risks | ESB bottleneck, governance overhead | Distributed data consistency, ops complexity, regulation |
Conclusion: architecture follows drivers; both rely on loose coupling & contracts.

**CS5. What did the NASA ECS CBAM case teach?** [6] — structured, quantified decisions under limited budget; VFC ranking; stakeholder consensus; validated intuition; process benefits.

**CS6. Apply the Netflix lessons to a regional OTT/e-commerce start-up with limited budget.** [8] — start modular monolith or few services on managed cloud (PaaS/containers), CDN for media, managed cache & DB replicas multi-AZ, circuit breakers/timeouts, autoscaling, observability; add chaos testing later; avoid over-engineering — decide by ASRs & cost (CBAM).

**CS7. Amazon's approach to CAP: explain with shopping cart vs checkout.** [4] — 12.6.

**CS8. Explain how ATAM was applied in a case study (CAAS/BCS) and what outputs were produced.** [6] — 12.8 + outputs list from 5.2.2.

---

# PART 14 — ONE-PAGE CHEAT SHEETS (revise these last, 30 min before exam)

## 14.1 Tactics at a glance 🧠
| QA | Categories → tactics |
|---|---|
| **Availability** | *Detect:* ping/echo, heartbeat, monitor, timestamp, sanity check, voting, exception detection, self-test · *Recover:* active/passive/cold redundancy, rollback, retry, degradation, reconfiguration, shadow, state resync, escalating restart · *Prevent:* removal from service, transactions, predictive model |
| **Performance** | *Control demand:* sampling rate, limit event response, prioritise events, reduce overhead, bound execution, resource efficiency · *Manage resources:* increase resources, concurrency, replicas (compute/data = cache), bound queues, schedule |
| **Security** | *Detect:* intrusion/DoS detection, integrity check · *Resist:* identify, authenticate, authorise, limit access/exposure, encrypt, separate, change defaults · *React:* revoke, lock, inform · *Recover:* restore + audit |
| **Modifiability** | *Cohesion:* semantic coherence · *Coupling:* encapsulate, intermediary, restrict dependencies, refactor, abstract common services · *Binding time:* config, plug-ins, runtime registration, polymorphism |
| **Usability** | *User initiative:* cancel, undo, pause/resume, aggregate · *System initiative:* task/user/system model |
| **Testability** | *Control & observe:* specialised interfaces, record/playback, localise state, abstract data sources, sandbox, assertions · *Limit complexity* |
| **Interoperability** | *Locate:* discover service · *Manage interfaces:* orchestrate, tailor interface |
| **Scalability** (course) | stateless, load balancing, sharding, caching, async queues, replicas, auto-scale, CDN |

## 14.2 Pattern one-liners
Layered (unidirectional use) · MVC (UI/logic split, observer) · Pipe-Filter (stream transforms) · Pub-Sub (events, decoupled) · Broker (location transparency) · Client-Server (request/reply) · P2P (equals) · Shared-Data (repository) · SOA (ESB, contracts) · Microservices (small, independent, DB-per-service) · Map-Reduce (parallel batch) · Multi-tier (physical tiers) · Microkernel (core + plug-ins) · Reflection (meta-level changes).

## 14.3 Weakness → fix pairs
| Weakness | Tactic fix |
|---|---|
| Broker/gateway/DB = SPOF | replicate (active/passive), cluster |
| Server bottleneck | scale out, cache, LB |
| Pub-sub unreliable ordering | partition keys, sequence numbers, idempotent consumers |
| Layer overhead | cache, bypass with documented exceptions |
| SOA ESB bottleneck | lightweight gateways, decentralise |
| Microservices data consistency | saga, outbox, eventual consistency |
| Cache staleness | TTL, write-through, invalidation events |
| Cold start (serverless) | provisioned concurrency, keep-warm |
| Sync call cascades | timeouts, circuit breaker, bulkhead, async |

## 14.4 Key definitions (🧠 verbatim-ish)
- **Software architecture:** set of structures needed to reason about the system, comprising elements, relations among them, and properties of both.
- **ASR:** requirement with profound, measurable effect on the architecture.
- **Utility tree:** top-down QA → refinement → scenario (H/M/L value, difficulty).
- **Risk:** architectural decision that may lead to undesirable consequences wrt a QA requirement.
- **Sensitivity point:** property critical to achieving a QA. **Trade-off point:** sensitivity point for ≥ 2 QAs.
- **Pattern:** context-problem-solution; **tactic:** design decision to control a QA response.
- **CAP:** can't have C, A, P simultaneously; under partition choose C or A.
- **Multi-tenancy:** one app instance serving many customers with logical isolation.
- **VFC:** benefit/cost. **Utility-response curve:** response level → utility 0–100.
- **4+1:** logical, process, development, physical + scenarios.

## 14.5 Formulas
- Availability = MTBF/(MTBF+MTTR). Utility interpolation: `U = Ua + (x−xa)/(xb−xa)·(Ub−Ua)`. Benefit `b = Uexp − Ucur`; `B = Σ b·W`; `VFC = B/C`. NPV = Σ (benefit−cost)/(1+r)^t. ROI = NPV/PV(cost). Break-even: solve `I1 + R1·t = I2 + R2·t`. Quorum: **R + W > N**.

## 14.6 Numbers to remember
~80% of cost post-deployment · architecture up-front sweet spot ≈ 20% (100 KSLOC), ≈ 40% (1,000 KSLOC) · ATAM ≈ 20–30 person-days, 3–5 evaluators, ≈ 4 phases (0–3), 9 steps · Lightweight ATAM 4–6 hrs · CBAM 9 steps, 100 votes, top 1/3 then top 50% · HDFS block ≈ 128 MB (64 MB in older), replication 3 · 99.95% ≈ 4.4 h/yr downtime · Large data-centre ≈ 100k+ servers, power 15–20% of cost, hardware discount up to 30% · Peer-review: several hours to half a day.

## 14.7 Last-minute exam tactics
1. Every design answer ends with **trade-offs + one clear recommendation**.
2. Always write **measurable** scenarios (numbers!).
3. Name **specific** patterns/tactics (not "use a good design").
4. Tabulate comparisons; draw small text diagrams — they are fast marks.
5. For case studies: **Drivers → Decisions → Tactics → Outcome → Trade-off.**
6. If unsure of a fact in CIGNA/TripAdvisor/Danske, stay at the **pattern & tactic level**.
7. Use the same vocabulary as the course: *ASR, utility tree, QAW, PALM, ATAM, sensitivity, trade-off, CBAM, VFC, 4+1, store-locally-sync-later, CAP.*

---
*End of notes — good luck! Revise Parts 2, 3, 5, 6, 7.6 and 13 first (very high priority per your analysis).*
