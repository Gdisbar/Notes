# Software Architecture — Step-by-Step Execution & 50-Question Exam Bank

---

# PART 0 — STEP EXECUTIONS (How This Question Bank Was Derived)

Below is the full trace of the analysis I performed before writing a single question, exactly as instructed.

---

## STEP 1 — Analysis of the Lecture Slides (Primary Technical Reference)

**Step 1.1 — Course inventory.** The material is BITS Pilani SEZG651/SSZG653 *Software Architectures* (Harvinder S Jabbal), based on Bass, Clements & Kazman, *Software Architecture in Practice* (3rd ed.). Modules identified:

| Module | Topic | Key concepts explicitly taught |
|---|---|---|
| CS01 | Introduction | Definition (set of structures to reason about the system), 3 structure categories (Module / C&C / Allocation), architectural patterns (layered, repository, client-server, multi-tier, competence center, platform), why architecture matters (13 reasons), 4 contexts (technical, project life-cycle, business, professional), stakeholders, Architecture Influence Cycle, rules of thumb for "good" architecture, Architecture Competence |
| CS02/CS03 | Quality Attributes | **6-part QA scenario** (source, stimulus, environment, artifact, response, response measure), general vs concrete scenarios, **tactics** for Availability (detect/recover/prevent faults), Performance (control demand/manage resources), Usability (user/system initiative), Security (detect/resist/react/recover + CIA), Modifiability (reduce size/increase cohesion/reduce coupling/defer binding), Testability (control-observe/limit complexity), Interoperability (locate/manage interfaces), 7 categories of design decisions, design checklists |
| CS04/CS05 | ASRs & Design | ASR definition and **indicators**, drivers (business goals, stakeholder concerns, QAs, regulatory, technology, cost, time-to-market), **QAW 7 steps** (30% voting), PALM, business goal format ("X to Y by when"), **Utility Tree** (Utility → QA → refinement → ASR leaf with (Importance, Difficulty)), Nightingale hospital examples, Agile: manifesto, 12 principles, Boehm–Turner **sweet spot** (10 KSLOC ≈ far left; 100 KSLOC ≈ 20%; 1000 KSLOC ≈ 40% of schedule), ICM, WebArrow case study (**spikes**), ADD (design strategy, steps), documenting in Agile |
| CS06 | Documentation | View vs structure, **4+1 views** (Logical–end-user/functionality, Process–integrators/performance/scalability, Development–programmers/module organization, Physical–system engineers/hardware, Scenarios–everyone/consistency), summary table (components/connectors/containers/stakeholders/concerns/tools), notation, when to prioritize which view, Software Architecture Document outline, architectural drift, architecture-as-code, fitness functions |
| CS07 | Layered Architecture | Client/Middle/EIS tiers, **Presentation** techniques (client vs server-side caching, AJAX, responsive design), **Business** techniques (façade, session management, workflow engine, rules engine, AOP for cross-cutting), **Data** techniques (connection pool, transactions, ORM, stored procedures, parameterized SQL vs SQL injection), **Service layer** (duplicate requests, out-of-sequence messages, communication failure via retry/queuing), Judiciary & DCS exercises, SSL/IPSec, session management |
| CS07/CS08 | Evaluation, Conformance, Testing, Reconstruction | 3 forms of evaluation (designer, peer review 4 steps, outsiders), contextual factors, **ATAM** (participants: 3–5 external evaluators, decision-makers incl. architect, 12–15 stakeholders; 9 steps; outputs: risks, non-risks, sensitivity points, trade-off points, risk themes mapped to business drivers; Lightweight Evaluation: 4–6 hrs, internal, steps 7–8 skipped), **drift examples**, consistency techniques (architecturally evident coding, frameworks — Spring/Hibernate/JMS/Drools/Log4J, code templates — fault-tolerant template, doc updates), **Architecture & testing** (utility tree → test-case priority; integration test plan from module interactions; testability design: data-source switching, rollback, simulators), **Reconstruction** (raw view extraction → DB construction → view fusion → architecture analysis → iterate; tools: Dali, ARMIN, Lattix, SonarJ, Structure 101; Vanish case; DiscoTech dynamic violation detection) |

**Step 1.2 — Faculty emphasis detected.** Heavy, repeated emphasis on: (a) tactics and QA scenarios, (b) QAW/utility tree/ASR pipeline, (c) ATAM vocabulary (risk/non-risk/sensitivity/trade-off), (d) layered-architecture layer-specific techniques with in-class exercises (Aadhaar, DCS, hotel reservation, logistics façade), (e) conformance (the slides literally ask "What other architecture violations have you come across?"), (f) reconstruction pipeline terminology. These become the highest-frequency question topics.

**Step 1.3 — Terminology register.** I locked the bank to the faculty's exact vocabulary — e.g., "architecturally evident coding style," "raw view extraction," "view fusion," "sensitivity point," "sweet spot," "spikes," "utility tree" with (H,M) notation — rather than external textbook language.

---

## STEP 2 — Analysis of the Sample Question Paper

**Step 2.1 — Structure observed.** 30 marks, 4 questions, multi-part (a/b/c), 2–4 marks per part.

**Step 2.2 — Question patterns inferred:**

| Pattern | Evidence | How I replicated it |
|---|---|---|
| "How does X influence Y, with example" | Q1a (structures → maintainability/scalability), Q1b (ASRs → design), Q3a (documentation → ASRs) | Questions Q5, Q11, Q46 require mechanism + example |
| Trade-off with balancing recommendation | Q2a (effectiveness/efficiency/satisfaction), Q4a (availability vs inventory cost), Q4b (complexity vs cost) | Q6, Q15–18, Q41 — every trade-off answer ends with a business-grounded recommendation, never "one architecture wins" |
| Framework application with scenario | Q2c (4+1 → ASRs), Q3b (utility tree → refine QAs and drive design) | Q4, Q13, Q19, Q39, Q40 |
| Interplay/optimization conflicts | Q2b (Security/Communication/Reliability views — optimizing one compromises another) | Q36, Q44, Q47, Q50 |

**Step 2.3 — Expected answer depth.** Marks of 3–4 with the phrase "provide an example of a scenario" imply: concept → mechanism → concrete example → justification, roughly 150–250 words per part. Answers in the bank are calibrated to that depth.

---

## STEP 3 — Syllabus → Coverage Mapping

| Syllabus area | Weight | Question slots |
|---|---|---|
| Quality attributes & tactics | Very High | Q1, Q2, Q6, Q8, Q15–18, Q22, Q23, Q33, Q34 |
| Trade-offs | Very High | Q6, Q15, Q16, Q17, Q18, Q41, Q47, Q50 |
| ASRs / drivers / QAW / utility tree | Very High | Q4, Q5, Q11, Q12, Q13, Q39, Q45 |
| ADD & design decisions | High | Q14, Q42, Q43 |
| Documentation / 4+1 | High | Q7, Q9, Q19, Q46 |
| Layered architecture | Very High | Q3, Q20, Q21, Q22, Q23, Q37 |
| ATAM | Very High | Q24, Q25, Q26, Q27, Q40 |
| Conformance / testing / reconstruction | High | Q28, Q29, Q30, Q31, Q32, Q49 |
| Real-time | High | Q33, Q34, Q44 |
| Introduction / contexts / competence | Lower (not ignored) | Q10, Q35, Q48 |

---

## STEP 4 — Question Blueprint

- **Difficulty split:** Q1–10 Medium (10) · Q11–35 Medium-High (25) · Q36–50 High (15). ✔
- **Integrated questions (≥10–15):** Q36, Q37, Q38, Q40, Q42, Q44, Q45, Q46, Q48, Q49, Q50 → **11**. ✔
- **Scenario diversity:** Banking (Q2, 6, 16, 24, 31, 41, 49), E-commerce (Q1, 20, 23, 42), Healthcare (Q39, 44, 45), SaaS (Q10, 11, 17), Telecom (Q19, 40), Logistics/Airline (Q21, 30), Manufacturing/Industrial (Q33, 34), Enterprise IT (Q3, 9, 28, 29, 37, 46), Media (Q50), Government (Q12, 35, 47), Insurance (Q5, 38). ✔

---

## STEP 5 — Final Quality Verification (pre-delivery checklist)

| Check | Status |
|---|---|
| Exactly 50 scenario-based questions | ✔ |
| No definitional-only questions | ✔ |
| Answers contain reasoning, trade-offs, recommendations — not just conclusions | ✔ |
| All questions inside syllabus; no microservices/CQRS/DDD/event-sourcing | ✔ |
| Faculty terminology preserved (utility tree (H,M), view fusion, sensitivity point, sweet spot, spikes, architecturally evident coding) | ✔ |
| Sample-paper style (influence-analysis, trade-off-with-recommendation, framework-application) replicated | ✔ |

---
---

# Software Architecture — 50 Scenario-Based Questions & Answers

## Section 1 — Medium Difficulty

### Q1. [E-commerce — Performance under peak load]
During a flash sale, your e-commerce checkout screens take 6–8 seconds to load because the product catalog is fetched from the database on every page render. The business demands page loads under 1 second even at 10× normal traffic. As the architect, identify the driving quality attributes, express the requirement as a quality attribute scenario, and recommend which layered-architecture technique you would apply and why.

**Answer:**
The primary quality attribute is **performance (latency under peak load)**; secondary attributes are **availability** (the site must not collapse during the sale) and **scalability**.

**QA scenario (6 parts):** Source — end users during a flash sale; Stimulus — page/request arrives; Environment — peak load (10× normal); Artifact — presentation and data layers; Response — catalog data served and page rendered; Response measure — page load < 1s at 10× load, 95% of requests.

**Architectural reasoning:** The catalog is read-heavy, shared, and changes rarely — ideal for **server-side caching** (store frequently used data needed by back-end modules on the server side: product codes, descriptions, promotions). Complement with **client-side caching** for user-specific preferences (payment method, delivery address) and **AJAX** so parts of the page (catalog list by category) update without reloading the whole page — the exact Aadhaar-style solution taught in class. This reduces DB load and latency simultaneously.

**Trade-offs:** Cached data can be stale — mitigate with TTLs and cache invalidation on product updates; caching adds infrastructure complexity; too-aggressive client-side caching risks serving wrong user-specific data.

**Recommendation:** Cache the catalog server-side, cache user preferences client-side, load dependent drop-downs/regions via AJAX. This directly targets the measured response (<1s) without rewriting the application.

**Key concepts tested:** QA scenario construction, performance tactics, client vs server-side caching, AJAX, layered technique placement.

---

### Q2. [Banking — Vague requirement to concrete scenario]
A banking sponsor says: *"The system must be reliable and always available."* As the architect, explain why this statement is unusable for design, distinguish a general scenario from a concrete scenario, and convert the requirement into a testable concrete availability scenario using the six-part structure.

**Answer:**
**Why unusable:** The slides stress that quality attribute definitions like "reliable" or "modifiable" are not testable — a requirement must be measurable so the architecture can be designed and evaluated against it. "Always available" has no percentage, no environment, no measurable response.

**General vs concrete:** A *general* scenario is system-independent — a table of possible values for each part (e.g., source = internal/external people, hardware, software; stimulus = fault of type omission/crash/incorrect timing). A *concrete* scenario instantiates those values for this specific system, making it designable and testable.

**Concrete availability scenario:**

| Portion | Value |
|---|---|
| Source | Hardware component (payment-processing node) |
| Stimulus | Fault: crash |
| Artifact | Payment-processing subsystem |
| Environment | Normal operation during month-end peak |
| Response | Detect the fault (heartbeat), log and notify operators, fail over to passive redundant spare with no data loss |
| Response measure | Detect ≤ 2s; failover ≤ 30s; availability ≥ 99.99%; data loss ≤ 1 transaction per 100,000 |

**Recommendation:** Present this scenario to stakeholders, get sign-off, and design availability tactics (heartbeat detection, passive redundancy, state resynchronization) against the measured responses.

**Key concepts tested:** General vs concrete scenarios, 6-part scenario structure, measurability of ASRs.

---

### Q3. [Enterprise IT — Layer violation in UI]
Developers have started placing database queries directly inside presentation-layer UI components because it is faster to implement. Over time, a single database schema change now forces modifications across dozens of UI screens. Identify the architectural violation, explain its impact on quality attributes, and describe how the layered architecture should be restored and kept from regressing.

**Answer:**
**Violation:** This is a classic **layer discipline violation** — the presentation layer is calling elements in the data layer directly, bypassing the business layer (the slides list "an object in one layer calling an object located in a layer beyond the adjacent layer" and "accessing database directly without going through data access layer" as drift).

**Impact:** *Modifiability* collapses — one schema change propagates through every screen (change amplification). *Separation of concerns* is destroyed; business rules become embedded in UI code and can be duplicated. *Testability* suffers because business logic cannot be exercised without the UI. *Reusability* of lower-layer components disappears.

**Restoration:** Move all queries into **Data Access Components (DAOs)** in the data layer; expose them through business-layer components; UI calls only the business layer. Enforce with: (1) **architecturally evident coding style** — mark which layer each file belongs to; (2) folder-per-layer discipline; (3) automated dependency checks (e.g., a Sonar rule "layer calls adjacent layer only"); (4) mandatory code review checkpoints.

**Trade-off:** Strict layering adds indirection and a little latency (reduce-overhead is a performance tactic for exactly this), and slows short-term velocity — but the measured change cost (dozens of screens per schema change) far outweighs that.

**Recommendation:** Refactor incrementally screen-by-screen behind the new DAO interface; gate the CI pipeline on dependency checks to prevent regression.

**Key concepts tested:** Layer bypassing, modifiability, conformance techniques, drift.

---

### Q4. [Airline — Utility tree construction]
You are the architect for an airline **Departure Control System (DCS)**. Business goals: zero lost check-ins during network outages, fast baggage tagging at peak, passenger-data confidentiality, and fast onboarding of new partner airlines. Construct a utility tree (at least 3 levels) and explain what the priority notation means and which leaves become ASRs.

**Answer:**
```
Utility
├── Availability
│   └── Fault recovery
│       └── Kiosk loses network at peak → queues locally,
│           syncs within 5 min, zero lost bookings (H, M)
├── Performance
│   └── Latency
│       └── Baggage tag issued ≤ 2s under 3,000 concurrent check-ins (H, M)
├── Security
│   └── Confidentiality
│       └── Passenger PII displayed only to authorized ground agents (H, M)
└── Modifiability
    └── Adding new product/tenant
        └── Onboard a new airline ≤ 5 person-days, no core change (M, L)
```

**Notation:** Each leaf carries **(Importance to success, Difficulty to achieve)** using H/M/L. (H,M) = high business value and moderately hard — these are exactly the "high on business value and high on architectural impact" scenarios the slides say become high-priority test cases and ATAM analysis targets.

**Which are ASRs:** The three (H,M) leaves are ASRs — they are critical, measurable, and will shape structures (local queueing + sync = availability structure; caching/indexing = performance structure; role-based access = security structure). (M,L) is documented but not design-driving.

**Recommendation:** Feed the (H,M) scenarios into ADD (choose tactics) and, later, into ATAM step 5/6 for risk analysis.

**Key concepts tested:** Utility tree levels, (Importance, Difficulty) notation, ASR selection, utility tree → test/design priority.

---

### Q5. [Insurance — ASR identification]
An insurance claims portal's requirements document contains: (a) adjusters (500 concurrent during catastrophe season) must get responses in ≤ 3s; (b) new state-specific claim rules must be configurable without source-code change, within 1 working day; (c) 99.9% uptime with disaster-recovery failover completed within 15 minutes; (d) "Generate a PDF copy of the claim"; (e) all PII must be encrypted at rest and in transit per regulation. Identify which requirements are ASRs, justify each using the indicators, and categorize them Critical/Important/Useful.

**Answer:**
**ASRs (Critical):** (a) — measurable QoS/SLA aspect (500 concurrent, 3s) that fundamentally shapes the runtime structure (resource management, concurrency, caching); **(b)** — modifiability ASR with high business value (frequent, planned rule changes); it will drive a rules-engine/defer-binding decision; **(c)** — availability ASR driving redundancy, failover, DR-site structures; **(e)** — a regulatory compliance requirement that is effectively a constraint (a design decision with zero degrees of freedom) and a security ASR shaping data/access structures.

**Not an ASR (Useful):** (d) PDF generation is a straightforward functional requirement — many architectures satisfy it; no measurable effect on the architecture.

**Indicators used (from the slides):** significant business value and/or technical risk; importance to an influential stakeholder; uniqueness (nothing in current architecture addresses it); QoS/SLA aspects differing from current architecture; history of budget overruns in similar contexts. The slides also note that even in critical applications only ~40% of features are used — hence categorizing Critical/Important/Useful prevents over-design.

**Recommendation:** Carry (a)–(c) and (e) into the utility tree as high-priority scenarios; treat (d) as an ordinary backlog item.

**Key concepts tested:** ASR indicators, constraints vs functional vs QA requirements, Critical/Important/Useful triage.

---

### Q6. [Mobile banking — Security vs Usability trade-off]
Your mobile banking app suffers fraud attempts, so the security team mandates **multi-factor authentication on every transaction**. Within two weeks, transaction completion drops 30% and user complaints spike. As the architect, analyse this trade-off and recommend an architectural response that respects both quality attributes and the bank's business goal of growing digital adoption.

**Answer:**
**Competing attributes:** Security (resist unauthorized access and modification — the availability leg of CIA is also touched, since friction reduces legitimate use) vs Usability (learning time, efficiency, minimizing the impact of errors, satisfaction). The evaluation slides state the principle directly: *increasing security often decreases performance/usability — you cannot have everything.*

**Trade-off analysis:** Blanket MFA raises authentication cost on *every* transaction, including trivially low-risk ones. The 30% drop-off is lost revenue and contradicts the business goal — the same pattern as the BHIM case in the slides, where "speed up cashless transactions" translated simultaneously into **usability (easy steps)** and **security (3-factor)** — they were balanced, not maximized individually.

**Architectural reasoning:** Move from "always strongest" to **risk-proportional protection**: maintain a user/device/task model (system-initiative usability tactics: *maintain task model, maintain user model*) so the system decides when to escalate. Low-risk: single factor; new device, high amount, unusual pattern: step-up authentication. Detection tactics (intrusion detection, message-integrity checks) provide the security floor so the usability layer isn't doing all the work.

**Recommendation:** Implement step-up, risk-based authentication with audit logging for every transaction; measure drop-off and fraud rate together; revisit thresholds via the utility tree scenario (e.g., "fraud rate < 0.01% with completion time ≤ 15s, (H,M)").

**Key concepts tested:** Security vs usability trade-off, system-initiative tactics, business-goal linkage.

---

### Q7. [4+1 Views — Stakeholder matching]
Three people ask your architecture team questions about a payment platform: (1) a new graduate developer asks "where do I start understanding this system?"; (2) the CTO asks "will it hold up under 3× holiday load?"; (3) the sysadmin asks "which servers run which components, and what are the network paths?". Using Kruchten's 4+1 model, state which view answers each question, who the typical viewer of each view is, and what concern each view considers.

**Answer:**

| Question | View | Viewer | Concerns |
|---|---|---|---|
| New developer onboarding | **Logical View** (+ Scenarios) | End-user/developers — functionality | Functional requirements; classes and key abstractions (Booch/UML class & package diagrams). Module views are "excellent for showing someone the structure of a project — who does what." |
| CTO — 3× load | **Process View** | Integrators | Non-functional: scalability, concurrency, performance; process/thread decomposition and synchronization. The slides' guide: *high-performance/high-traffic → prioritize the Process View.* |
| Sysadmin — servers | **Physical View** | System engineers | Mapping software onto hardware; topology, communications; used to reason about availability, performance, security. |
| (Supporting) | **Development View** | Programmers & software managers | Static organization of code into modules/subsystems/layers. |
| Cross-check | **Scenarios (+1)** | All users & evaluators | System consistency and validity — "putting all 4 views together." |

**Recommendation:** Answer each stakeholder from their view, but keep all views consistent via shared scenarios — that is the +1 model's integrative role.

**Key concepts tested:** 4+1 stakeholder-view mapping, view concerns, scenario integration.

---

### Q8. [Payments — Availability detection tactics]
Merchants only discover outages in your payment gateway when payments fail, because nothing alerts operations. Ops learns nothing until revenue is lost. As the architect, select availability **detection** tactics, express the requirement as a concrete scenario, and explain which design-checklist area was neglected.

**Answer:**
**Detection tactics:** (1) **Heartbeat** — periodic message exchange between a system monitor and the monitored process; absence of expected heartbeats signals failure; (2) **Ping/Echo** — asynchronous request/response to determine reachability and round-trip delay; (3) **Monitor** — a component that tracks state of health of other parts and can detect failure or congestion; (4) **Exception detection** — timeouts on outstanding calls.

**Concrete scenario:** Source — system monitor; Stimulus — heartbeat not received for 3 consecutive intervals; Artifact — payment gateway process; Environment — normal operation; Response — detect the fault, log it, notify operators, fail over to the spare; Response measure — detect ≤ 2s, failover ≤ 30s, availability ≥ 99.95%.

**Checklist gap:** The "Allocation of Responsibilities" item of the availability design checklist was missed — it requires additional responsibilities to *detect an omission, crash, incorrect timing, or incorrect response*, and to log the fault and notify appropriate entities. No component owned detection, so faults surfaced only externally.

**Recommendation:** Add a monitor component receiving heartbeats from every gateway instance (frequency chosen with the known trade-off: higher frequency = faster detection but more processing/bandwidth), wire alerts to ops, and add self-test on the gateway.

**Key concepts tested:** Detection tactics, concrete scenario, design checklist application.

---

### Q9. [Enterprise — One-diagram documentation failure]
An enterprise system's entire architecture is captured in one large diagram containing UI, services, databases, servers, and firewall annotations. Developers interpret it inconsistently; the security team says nothing in it addresses their concerns. Diagnose the problem and describe how the documentation should be restructured.

**Answer:**
**Diagnosis:** This is exactly the problem stated in the slides: *"Architecture documents do not address the concerns of all stakeholders… architecture documents contained complex diagrams, sometimes hard to represent."* One diagram is not a **view** — a view is a representation of a coherent set of architectural elements **written for and read by specific stakeholders**. Mixing module, runtime, deployment, and security information in one drawing means every reader projects their own interpretation, and no reader's concerns are fully served.

**Restructure:**
- **Module (decomposition) view** → developers: responsibilities and allowed dependencies per module.
- **Component-and-connector (client-server) view** → integrators/ops: major executing components, connectors, data flow.
- **Allocation/deployment view** → sysadmins: software-to-hardware mapping, communication pathways.
- **Quality-attribute views** → security view (who may access what), communication view (protocols, SSL/IPSec points), reliability view (redundancy, failover), as taught under "quality attribute views" and "combining views."

**Documentation package:** documentation roadmap (view catalogue: primary presentation, element catalogue, context, rationale, glossary, cross-view references) — *"documenting an architecture is documenting the relevant views, then adding documentation that applies to more than one view."*

**Recommendation:** Split by concern, add a short roadmap explaining which view answers which stakeholder's question, and update documents at release time (mark obsolete parts "no longer applicable").

**Key concepts tested:** Views vs diagrams, stakeholder concerns, documentation package, combining views.

---

### Q10. [Agile — How much up-front architecture?]
Startup A is building a 10-KSLOC MVP with volatile requirements. Enterprise B is extending a 500-KSLOC platform with high assurance demands. Both teams argue about whether to do "agile only" or "big architecture first." Using the Boehm–Turner sweet-spot analysis, advise both teams on how much up-front architecture and risk-resolution effort to invest.

**Answer:**
**Framework:** Boehm and Turner plotted % of schedule for up-front architecture/risk work against rework cost. The curves show a **sweet spot per project**: 10 KSLOC → far left (almost no up-front work pays); 100 KSLOC → ~20% of schedule; 1,000 KSLOC → ~40%. **Drivers:** rapid change shifts the sweet spot left; high assurance shifts it right. The slides' conclusion: a million-line system is enormously complex — Agile principles alone can't organize that without an architecture to guide the effort.

**Startup A (10 KSLOC, volatile):** Sweet spot is far left. Do *not* invest in up-front architecture work — it is a waste. Per the authors' advice: "at least try to get agreement on the major patterns to be employed; don't spend too much time on architecture design, documentation, or analysis up front."

**Enterprise B (~500 KSLOC, high assurance):** Sweet spot between 20–40% of schedule. Per the authors: "if building a large, complex system with relatively stable and well-understood requirements and/or distributed development, doing a large amount of architecture work up front will pay off." If requirements are unstable, "start by quickly designing a candidate architecture even if it leaves out many details… be prepared to change and elaborate it as you perform spikes and experiments."

**Recommendation:** The right question is not "Agile or architecture?" but "how much architecture now vs later?" — A: patterns only, refactor as debt grows; B: candidate architecture (~20–25%) + risk-driven spikes, elaborated iteratively.

**Key concepts tested:** Boehm–Turner sweet spot, sweet-spot drivers, guidelines for the Agile architect.

---

## Section 2 — Medium-High Difficulty

### Q11. [SaaS HR platform — Extracting ASRs from vague wishes]
Stakeholders describe a multi-tenant SaaS HR platform only in vague terms: "it must be fast," "super secure," "easy to change per client," "we have 8 months and a fixed budget." As the architect, list the categories of architectural drivers you must extract, convert each vague wish into an architectural driver, and produce one concrete QA scenario for the most important driver.

**Answer:**
**Driver categories (per the slides):** business goals & constraints (high uptime, horizontal scaling); stakeholder concerns (which requirement beats which — e.g., security over performance); quality attribute requirements; regulatory & compliance; technology constraints/opportunities (framework, deployment environment, database); cost & budget constraints; time to market.

**Extraction:**
- "Fast" → Performance driver: seasonal payroll spikes.
- "Super secure" → Security driver: cross-tenant data isolation + payroll data confidentiality; also a compliance driver (privacy regulation).
- "Easy to change per client" → Modifiability driver: tenant-specific rules/formulas without code forks.
- "8 months, fixed budget" → **Constraint** (time-to-market and cost), which caps how many tactics can be afforded — it arbitrates trade-offs.
- Implicit business goal: land enterprise clients → availability during payroll runs becomes critical.

**Concrete scenario for the top driver (multi-tenant modifiability):** Source — tenant administrator; Stimulus — requests a new overtime formula; Artifact — payroll calculation subsystem; Environment — design/configuration time; Response — formula changed via configuration, validated, deployed to that tenant only; Response measure — ≤ 1 working day, no code change, zero impact on other tenants.

**Recommendation:** Take this driver list into a QAW-style brainstorm, consolidate, and vote to prioritize before choosing any architecture.

**Key concepts tested:** Driver categories, vague-requirement → scenario conversion, constraints as arbiters.

---

### Q12. [Government tax portal — Running a QAW]
You are facilitating a Quality Attribute Workshop for a national e-filing portal (30M taxpayers; filing deadline surges; new filing rules every budget). Describe the seven QAW steps, stating concretely what happens at each step for this system and what the workshop's expected output is.

**Answer:**
Expected output: **a list of architectural drivers** — distilled requirements, business drivers, constraints, and quality attributes — that will seed the utility tree and ADD.

1. **Presentation & Introduction** — facilitators, architect, IT ministry, tax-domain staff, ops introduce themselves and their relation to the system.
2. **Business Mission Presentation** — a management representative spends ~1 hour on business context: deadlines, 30M-user scale, budget/legal constraints, known QA needs (performance at deadline peak, security of taxpayer data).
3. **Architectural Plan Presentation** — the architect presents current architectural thinking (e.g., planned tiering, caching approach) so stakeholders know the baseline.
4. **Identification of Architectural Drivers** — facilitators share drivers assembled from steps 2–3 (e.g., "handle 10× surge in last 48 hours," "support new deduction rules annually," "PII protection") and stakeholders add/correct/delete until consensus.
5. **Scenario Brainstorming** — each stakeholder expresses concerns as scenarios with explicit stimulus and response; facilitators ensure at least one representative scenario exists per driver (e.g., assessor: "file return with 50 entries in < 30s at deadline peak"; maintainer: "deploy a new deduction rule in < 2 days").
6. **Scenario Consolidation** — similar scenarios merged with proposers' consent (several performance variants merge into one peak-load scenario).
7. **Scenario Prioritization** — voting: each stakeholder gets votes equal to 30% of the consolidated scenario count; top scenarios drive design and later ATAM analysis.

**Key concepts tested:** QAW steps, driver identification, 30%-voting prioritization.

---

### Q13. [Utility tree — Interpreting priorities]
A utility tree for your platform contains three leaves: Scenario X (H, L), Scenario Y (M, M), Scenario Z (H, H). A teammate wants to spend equal design effort on all three because "they're all in the tree." As the architect, explain what each priority pair means, which scenario you analyze/design first and why, and whether (H, L) should be ignored.

**Answer:**
**Notation:** each leaf is (Importance to business success, Difficulty to achieve). 
- **X (H, L):** critical to success but easy — e.g., "database vendor release hot-swapped with no downtime (H,L)" from the Nightingale examples. Low design risk: standard tactics suffice; do not over-invest, but *do* verify the tactic is actually in place.
- **Y (M, M):** moderately important, moderately hard — schedule normally; revisit if priorities shift.
- **Z (H, H):** critical *and* difficult — the true architectural driver. It consumes design attention first, gets tactics chosen deliberately in ADD, becomes a top ATAM analysis target, and (per the testing slides) translates into high-priority test cases because it is high on business value and high on architectural impact.

**Should (H,L) be ignored?** No — ignoring importance is how (H,L) items silently break. But its effort allocation differs: cheap insurance (e.g., connection pooling, parameterized SQL) rather than deep structural work. Equal effort on all three misallocates scarce design and testing resources away from Z.

**Recommendation:** Design and analyze in order Z → X (cheap verification) → Y; record priorities so testing and evaluation follow the same order.

**Key concepts tested:** (Importance, Difficulty) semantics, prioritization-driven design and testing.

---

### Q14. [E-commerce — Applying ADD]
Using Attribute-Driven Design, sketch how you would design the order-processing subsystem of an e-commerce platform whose top utility-tree scenarios are: "order placement succeeds during 10× sale peak, p95 ≤ 800ms (H,M)" and "replace the payment provider within 10 person-days (H,M)". Show the essential ADD steps and one iteration.

**Answer:**
**ADD essence:** the input is the architectural drivers (ASRs as QA scenarios); design proceeds by repeatedly selecting a module, establishing its drivers, choosing tactics to achieve them, and instantiating modules — a **generate-and-test** strategy.

**Iteration 1 — target module: Order Processing.**
1. **Establish drivers:** peak-load performance (p95 ≤ 800ms @ 10×) and payment-provider modifiability (≤ 10 person-days).
2. **Choose tactics:**
   - Performance → *manage resources*: maintain multiple copies of computations (replicated order services); *bound queue sizes* in front of the payment call; *reduce overhead* by removing intermediaries from the hot path; caching of catalog lookups in the order flow.
   - Modifiability → *encapsulate* the payment interaction behind an interface; *use an intermediary* (payment-adapter façade) so provider-specific details don't leak; *defer binding* — the adapter is selected/configured rather than hard-wired.
3. **Instantiate modules:** Order Entry, Order Orchestrator, Payment Adapter (interface + provider implementations), Inventory Check, Catalog Cache client; define each module's responsibilities and allowed dependencies (business layer may call data/service layer, never UI).
4. **Test against the scenarios:** the adapter isolates provider change (modifiability scenario satisfied); replicas + queue bounds address peak load; record the design decision and its rationale.

**Iterate:** next round targets the inventory module or a residual driver (e.g., availability of the orchestrator), with leftover scenarios reprioritized.

**Key concepts tested:** ADD steps, tactic selection per driver, generate-and-test, iteration.

---

### Q15. [Modifiability vs Performance — Intermediaries]
Your team proposes inserting a service-abstraction intermediary between UI and business services "so everything is replaceable." After deployment, transaction latency rises 40% and users complain. Analyse this using the relevant tactics from both sides and recommend how to proceed.

**Answer:**
**Two tactics in tension:**
- *Modifiability:* "use an intermediary" breaks dependency A→B by inserting an abstraction; encapsulation likewise adds interfaces. These raise modifiability by localizing change.
- *Performance:* "reduce overhead" — the slides state explicitly that *the use of intermediaries (important for modifiability) increases the resources consumed in processing an event stream; removing them improves latency.*

**Analysis:** The 40% regression is a measured performance response breaching the scenario (e.g., p95 ≤ 500ms). The question is where the change pressure actually is. If the replaced components change rarely (or are internal), the intermediary's insurance is not worth a permanent tax on every request. If they change often (external providers, plugins), keep the intermediary but make it cheap.

**Trade-off:** removing the intermediary wins latency but re-exposes coupling — future provider changes become nonlocal. Keeping it protects modifiability but permanently taxes performance.

**Recommendation:** Keep intermediaries **off the hot path**: (1) profile — remove the abstraction only on measured latency-critical operations, replacing it with a thin direct call plus a conformance test that catches interface drift; (2) keep abstractions on slowly-changing or high-variability seams; (3) re-state the QA scenarios so both are measurable ("p95 ≤ 500ms" and "provider replaceable ≤ 10 person-days") and pick tactics that jointly satisfy them rather than maximizing one.

**Key concepts tested:** Reduce-overhead vs use-intermediary conflict, measured trade-off, hot-path reasoning.

---

### Q16. [Banking month-end — Availability vs Cost]
A retail bank suffers outages during month-end processing. The sponsor demands 99.99% availability. The proposed design (active-active data centers, N+2 replicas) doubles infrastructure cost and the CFO objects. As the architect, analyse the architectural drivers, express the availability scenario, identify the tactics and their cost structure, and recommend a balanced solution.

**Answer:**
**Drivers:** business goal — retain customers/no regulatory exposure; QA — availability 99.99% (≈ 52 min downtime/year); constraint — infrastructure budget. The conflict is availability vs cost, the classic pairing.

**Concrete scenario:** Source — server/storage fault; Stimulus — crash during month-end batch + online peak; Artifact — core banking transaction path; Environment — peak load; Response — detect (heartbeat), fail over, continue serving; Response measure — availability ≥ 99.99%, failover ≤ 60s, no committed transaction lost.

**Tactics & cost structure:** *Active redundancy (hot spare)* — instant failover but pays double compute permanently; *passive redundancy (warm spare)* + periodic state updates — cheaper, small resynchronization delay on failover; *cold spare* — cheapest, longest recovery; *removal from service / transactions* — prevention, nearly free.

**Recommendation (tiered, not uniform):** Protect the *revenue-critical online transaction path* with active redundancy (business value of downtime justifies cost); run *month-end batch and reporting* on warm spares with state resynchronization — a 15–60 min degraded window there is acceptable and was never in the 99.99% scenario; use transactions + rollback everywhere to bound damage. This satisfies the measured availability where it matters at a fraction of double-DC cost. Present the split back to the CFO as a cost-per-outage-minute argument — architecture decisions are business decisions.

**Key concepts tested:** Availability tactics cost spectrum, scenario-driven tiering, business-value trade-off.

---

### Q17. [Streaming SaaS — Horizontal vs vertical scaling]
Your media-streaming SaaS faces a predictable 10× seasonal spike. One engineer argues for vertical scaling ("bigger machines, simpler"); another for horizontal scaling ("more machines"). Using the definitions from the course, compare the two, identify which quality attributes and design changes each implies (especially session/state management), and recommend an approach.

**Answer:**
**Definitions (from the slides):** *Vertical scalability (scale up)* = add resources to a physical unit (more memory/CPU on one server). *Horizontal scalability (scale out)* = add more resources to logical units (another server to the cluster).

**Vertical:** simple (no code change), but a hard ceiling, a single point of failure, and expensive non-linear pricing; availability barely improves.

**Horizontal:** near-linear capacity and better availability (more replicas), but demands architectural change: instances must be **stateless** at the web tier. The slides' session-management discussion applies directly — with multiple instances, a user's requests may land on different instances, so either use **client-based state management (cookies)** or **server-based persistent session management** (externalized session store), not in-memory per-instance state. Also implies load balancing, shared data stores, and bounded queues.

**Trade-off:** scale-out's complexity (statelessness, session externalization, consistency) is real; scale-up's simplicity hides a scalability and availability ceiling.

**Recommendation:** Scale out the presentation/web tier with externalized session state; cache hot catalog data server-side (Memcached-style pooled cache, as taught); keep any stateful batch jobs separate. Express it as a scenario: "During the 10× spike, add 20 instances in 30 minutes with no code change; error rate < 0.1% (H,M)." This satisfies the spike *and* availability, at the cost of a one-time statelessness refactor.

**Key concepts tested:** Scale up vs scale out, session/state management tactics, scalability scenario.

---

### Q18. [IoT/Automotive platform — Testability vs architectural complexity]
An automotive telematics platform must integrate with payment gateways, sensors, and government identity systems. QA insists on end-to-end regression with 85% coverage per sprint, but every test run hits external systems, is slow, and is non-deterministic. As the architect, propose testability tactics, and analyse the trade-off between added architectural complexity and testing speed.

**Answer:**
**Testability tactics applicable:**
- *Control and observe:* **specialized interfaces** (test harness access to component inputs/outputs); **record/playback** (capture traffic crossing interfaces and replay as test input); **abstract data sources** (interfaces that let test data substitute production data); **sandbox** (isolate from the real world so experiments need no undo); **executable assertions**.
- *Limit complexity:* avoid/resolve cyclic dependencies; limit non-determinism (unconstrained parallelism).

Applied here: build **simulators/stubs for the payment gateway, sensors, and identity system** — the slides' own example: "ability to replace components (simulators of payment gateway, sensors, external systems such as Aadhar)." Record/playback lets one recorded identity-system exchange test hundreds of error branches deterministically.

**Trade-off:** the simulators, interfaces, and sandbox are real architecture — extra components, maintenance, and risk that simulators drift from real systems. But the alternative costs more: the slides frame testability as "probability, assuming the software has at least one fault, that it will fail on its next test execution" — untestable systems find faults late, when they are most expensive. Mitigate simulator drift by recording real traces and replaying them into the simulator for calibration, and by periodic live environment runs.

**Recommendation:** Invest in stub-able interfaces (record/playback + abstract data sources) at exactly the three external boundaries; keep the core structure simple (no test-only coupling inside business logic).

**Key concepts tested:** Testability tactics, control & observe, testability-vs-complexity trade-off.

---

### Q19. [Telecom PABX — Which views to document]
You are documenting a high-traffic telephony call-processing system with strict uptime requirements. Developers, integrators (performance), and system engineers (hardware) all need architecture information. Using 4+1, decide which views to document first for this system, justify each choice, and explain why not all systems need all views.

**Answer:**
**The slides' prioritization guide applies directly:** *high-performance/high-traffic → prioritize the Process View.*

1. **Process View (first)** — integrators; concerns: performance, scalability, availability, software fault-tolerance, integrity; components are tasks/periodic processes, connectors are message/RPC/event-broadcast. For call processing with concurrency and deadlines, this view is where latency and failover reasoning happens (the PABX process-blueprint example in the slides is exactly this).
2. **Physical View (second)** — system engineers; concerns: topology, communications, availability, performance on hardware; needed because uptime and load are achieved partly by hardware allocation (primary/backup processors).
3. **Logical View (third)** — end-user/developers; functionality: conversation, terminal, connection services abstractions (as in the slides' PABX logical view).
4. **Development View** — as the codebase/team grows (layered style, module organization).
5. **Scenarios (+1)** — trace a local-call scenario through all views to keep them consistent and valid.

**Why not all views everywhere:** the slides state "not all architectures need all views" and "document only what is necessary to communicate the design and mitigate risk." Here, the *concerns that carry risk* are concurrency, latency, and failover — hence Process + Physical first; a heavyweight Development view can wait.

**Key concepts tested:** View prioritization by concern, 4+1 concerns per view, scenario integration.

---

### Q20. [E-commerce — Service-layer responsibilities]
After a network blip, your order service receives the same "place order" message twice, and a status-update message arrives before the order-creation confirmation. Today both cases corrupt order data. As the architect, decide which architectural layer owns these problems, design its behavior, and justify why the business layer is the wrong place.

**Answer:**
**Owner: the Service layer.** The slides define exactly these situations as service-layer responsibilities: *handle the same request coming twice; handle messages coming out of sequence; handle communication failure (using a retry mechanism or by queuing work and sending once communication is restored).*

**Design:**
- **Duplicates:** idempotency — carry a client-generated idempotency key with each order request; the service layer checks a processed-key table and returns the original result instead of re-executing.
- **Out-of-sequence:** attach sequence/version numbers to messages; the service layer buffers or discards-and-requests-resend for out-of-order arrivals per the protocol contract.
- **Communication failure:** queue outbound work (message-queue based communication) and retry with backoff; on restore, drain the queue — explicitly the slides' "queuing work and sending once communication is restored."

**Why not the business layer:** business components should implement order rules (pricing, inventory, approval) — pure domain logic. Connectivity semantics (exactly-once, ordering, redelivery) are protocol-level concerns that would otherwise be duplicated in every business component; centralizing them in the service layer keeps business components simple and testable, and it sits at the system boundary where external systems interact.

**Trade-off:** idempotency keys and sequencing add storage and coordination overhead at the boundary — acceptable, since it removes an entire class of distributed-failure bugs from business code.

**Key concepts tested:** Service layer responsibilities, idempotency/ordering/queuing, layer responsibility placement.

---

### Q21. [Logistics — Façade in the business layer]
In a container-shipping system, placing a shipping request requires calling three modules: "find nearest empty container," "find best transporter," and "find ship with spare capacity." Clients currently call all three, and you (the architect) want these modules hidden from the client layer. Which technique and which layer? Explain the pattern's intent, benefits, and one risk.

**Answer:**
**Technique:** **Application Façade** placed in the **Business layer** — exactly the slides' answer: "In the Business layer, provide a unified API — façade — to place a shipping request. The façade will in turn make use of modules to search for appropriate container, appropriate transporter, appropriate ship."

**Pattern intent:** "Provide a unified interface to a set of interfaces in a subsystem… a higher-level interface that makes the subsystem easier to use… typically a single wrapper class containing the members required by the client."

**Benefits:**
- **Coupling reduction:** clients depend on one stable operation (e.g., `placeShippingRequest(goods, destination)`) instead of three internal modules — internal re-orchestration (new routing module) does not ripple to clients.
- **Simpler clients:** the client no longer knows sequencing/orchestration rules.
- **Modifiability:** adding a fourth step (customs check) changes only the façade.

**Risk:** the façade can become a god-object absorbing business logic that belongs in underlying components, hurting cohesion; it can also add a performance layer on hot paths. Keep the façade thin — orchestration only.

**Recommendation:** Implement `ShippingFacade.placeRequest()` in the business layer, delegating to container/transporter/ship modules; document it in the business-layer view and enforce that UI only sees the façade.

**Key concepts tested:** Façade pattern intent, business-layer placement, coupling reduction.

---

### Q22. [Aadhaar-style screen — Presentation-layer performance]
A citizen-registration screen loads slowly because its State/District/Town drop-downs fetch ~4,000 towns from the database on every page load. Stakeholders also report that the layout breaks on tablets and phones used by registration operators. As the architect, state the techniques you would use, where each belongs, and the trade-offs.

**Answer:**
**This is the course's own worked exercise; the solution techniques:**
1. **Cache states, districts, and towns in the backend** — server-side caching of rarely-changing reference data (server-side caching = "frequently used data needed by different back-end modules stored on the server side — e.g., product codes, descriptions, promotions"). One cache instance serves all users.
2. **Use AJAX for dynamic loading of districts and towns** — fetch district list only after the user selects a state, town list after district: the page payload shrinks from 4,000 entries to a handful, and no full-page reloads. AJAX = asynchronous JavaScript & XML, updating parts of the page without reloading it.
3. **Responsive design for the form-factor problem** — flexible grids so layout adapts to screen size (HTML5/CSS3), because mobile/tablet operators use different devices.

**Trade-offs:** caching risks stale reference data after administrative boundary changes — mitigate with low-cost invalidation on updates (rare events); AJAX adds more, smaller requests (fine here, since lists are small and typed); responsive design increases CSS complexity and testing matrices across devices.

**Recommendation:** Server-side cache with update-triggered invalidation + AJAX cascade for the three drop-downs + responsive layout. This combination targets exactly the two scenarios: slow load (performance) and broken layout (usability).

**Key concepts tested:** Presentation-layer techniques, client vs server-side caching, AJAX, responsive design.

---

### Q23. [Retail DB — Data-layer hardening]
Your retail database serves 1,000 concurrent users; latency spikes under load, and a penetration test demonstrates a SQL injection vulnerability in the product-search field. As the architect, select data-layer techniques for both problems, justify each, and note one trade-off per technique.

**Answer:**
**Performance techniques:**
- **DB connection pooling** — "use DB connection pool if there are too many users": pooling reuses connections instead of paying connect/authentic cost per request; trade-off: pool sizing misconfigurations cause queuing.
- **Transactions** — bundle state updates to be atomic/consistent/isolated/durable; ensures order+inventory updates don't half-commit; trade-off: lock contention under load.
- **Object-Relational Mapping (e.g., Hibernate)** — maps objects to tables, reducing hand-written mapping code (modifiability); trade-off: ORM-generated SQL can be less efficient than hand-tuned queries on hot paths.
- **Stored procedures** — precompiled, server-side execution improves performance for heavy operations; trade-off: vendor coupling and versioning/difficulty of testing.
- **Multiple copies of data for faster access** — read replicas/cache of hot catalog data; trade-off: consistency management.

**Security technique — the injection fix:** never concatenate user-entered strings into SQL (the slides' example: appending `' OR 'x'='x` returns *all* product data). Use **parameterized SQL queries** (`WHERE product-name = ?`), so input is treated as a parameter value, never executable SQL. Trade-off: negligible; occasionally less flexible for dynamic queries — handle those with whitelisting.

**Recommendation:** Connection pool + read replicas for load; transactions on write paths; parameterized queries everywhere; ORM for standard CRUD, stored procedures for the few measured hot spots.

**Key concepts tested:** Data-layer techniques, SQL injection defense, per-technique trade-offs.

---

### Q24. [ATAM vocabulary — Heartbeat monitoring]
During an architecture evaluation of a payment cluster, the following design elements surface: (a) heartbeat frequency between gateway nodes, (b) number of simultaneous database clients, (c) watchdog component, (d) absence of a backup data channel, (e) failover routing table. As an ATAM evaluator, classify each as a risk, non-risk, sensitivity point, or trade-off point, with justification.

**Answer:**
Recall the definitions: **risk** = an architectural decision that may lead to undesirable consequences given QA requirements; **non-risk** = a decision deemed safe upon analysis; **sensitivity point** = a decision that markedly affects *one* quality attribute response; **trade-off point** = a decision that affects *multiple* attributes in opposing directions.

| Element | Classification | Justification |
|---|---|---|
| (a) Heartbeat frequency | **Trade-off point** | Higher frequency → faster fault detection (availability ↑) but consumes processing time and communication bandwidth (performance ↓). Affects two attributes oppositely — the slides' canonical example. |
| (b) Number of simultaneous DB clients | **Sensitivity point** | Directly determines transactions/sec the database can process — a marked effect on one attribute (performance). Also the slides' example. |
| (c) Watchdog | **Non-risk** | Simple mechanism, proved reliable — deemed safe upon analysis (again from the slides' example analysis). |
| (d) No backup data channel | **Risk** | If the primary data channel fails, availability requirement (e.g., 99.999%) is jeopardized — "availability requirement might be at risk due to lack of backup data channel." |
| (e) Failover routing | **Non-risk** (assuming tested) | Routing logic that redirects traffic post-failure supports availability without penalizing normal operation; document as safe, or escalate to risk if failover time is untested. |

**Recommendation:** Record all five in the ATAM outputs; risks feed the mitigation plan; the trade-off on (a) deserves an explicit scenario (detect ≤ 2s) to fix a defensible heartbeat value.

**Key concepts tested:** Risk / non-risk / sensitivity / trade-off definitions and discrimination.

---

### Q25. [ATAM — Utility tree vs brainstormed scenarios]
In your ATAM, step 5's utility tree is dominated by performance and security scenarios, but step 7's stakeholder brainstorming produces a flood of modifiability scenarios ("new payer integration," "per-state rule changes") that rank in the top five. What does this divergence indicate, what risk does it create, and what should the evaluation team and architect do next?

**Answer:**
**What divergence indicates:** the slides are explicit — the purpose of brainstorming is "to take the pulse of the larger stakeholder community"; comparing the prioritized list with the utility tree shows alignment or misalignment between what the architect had in mind and what stakeholders actually want. Here there is **significant misalignment**: the architect optimized for performance/security; the community's real concerns are modifiability.

**Risk created:** the slides state it directly — additional driving scenarios discovered "may itself be a risk, if the discrepancy is large. This would indicate some disagreement in the system's important goals between the stakeholders and the architect." Concretely: the architecture likely lacks the structures (rules engine, service interfaces, configuration points) that the new modifiability scenarios require — an architectural risk that will surface as painful, change-amplifying releases.

**What to do:**
1. The evaluation team treats the top new scenarios as driving scenarios and runs **step 8 (analyze architectural approaches)** on them: ask the architect how the current architecture carries out "integrate a new payer in ≤ 5 person-days." The likely answer exposes missing interfaces/coupling — recorded as risks.
2. Update the utility tree to reflect the community's priorities (the tree is a living artifact).
3. Group the resulting risks into a **risk theme** (e.g., "insufficient design for evolving payer/regulatory landscape") and, in step 9, map it to the affected business drivers (e.g., "expand into new states/markets").

**Recommendation:** Report the divergence prominently — it is one of ATAM's most valuable findings.

**Key concepts tested:** Step 7 comparison purpose, divergence-as-risk, step 8 continuation, risk themes.

---

### Q26. [ATAM — Risk themes and business drivers]
Your ATAM of a consumer banking platform uncovers these risks: no standby data center; failover tested only on the test environment; audit logs not tamper-protected; documentation of the recovery procedure three years stale. Show how to group these into risk themes and why mapping themes to business drivers matters to management.

**Answer:**
**Grouping into risk themes** — themes collect risks sharing "some common underlying concern or systemic deficiency":

- **Theme 1: Insufficient availability/DR design** — risks: no standby data center; failover untested in production-like conditions. (This mirrors the slides' example: risks about inability to function under failures lead to a theme about insufficient attention to backup capability/high availability.)
- **Theme 2: Auditability and documentation given insufficient consideration** — risks: tamper-protection gap; stale recovery documentation. (Slides' example: inadequate/out-of-date documentation → theme that documentation gets insufficient consideration.)

**Mapping themes to business drivers (step 9):**
- Theme 1 threatens drivers: "be the most reliable digital bank" and regulatory continuity obligations — an outage now means regulatory scrutiny and customer churn, not just lost minutes.
- Theme 2 threatens: "pass the regulator's audit without restrictions on new products" — an unverifiable audit trail can freeze product launches.

**Why the mapping matters:** management does not act on architectural vocabulary; it acts on business drivers. The slides: mapping "elevates the risks… to the attention of management, who cares about the business drivers." A theme phrased as "insufficient availability design threatens the flagship reliability commitment" gets budget; a list of tactical gaps does not.

**Recommendation:** Present each theme with its threatened drivers, severity, and the mitigation cost, so leadership can make an explicit accept/fix decision.

**Key concepts tested:** Risk themes, theme→business-driver mapping, communicating architecture to management.

---

### Q27. [Choosing ATAM vs Lightweight Evaluation]
Two upcoming evaluations: (1) a national payments core platform (bank-grade, high regulatory exposure, 2M LOC) before a go-live gate; (2) an internal reporting tool (30 KSLOC, internal users, low risk). For each, recommend full ATAM or Lightweight Architecture Evaluation, justify with the effort/participation characteristics, and state which ATAM steps a lightweight evaluation skips and why that is acceptable.

**Answer:**
**(1) National payments core → full ATAM.** Justification: ATAM is "a substantial undertaking — some 20 to 30 person-days of effort from an evaluation team, plus even more for the architect and stakeholders. Investing this amount makes sense on a large and costly project, where the risks of making a major architectural mistake are unacceptable." It uses an external evaluation team (3–5 competent, unbiased people), project decision-makers (including the architect, who *must* willingly participate), and 12–15 stakeholders, across phases 0–3 (preparation; steps 1–6 with a 2–3-week hiatus; steps 7–9; follow-up). Regulatory exposure and go-live risk make the deep, external, two-round scenario analysis (utility tree in step 5–6, brainstorm in step 7, re-analysis in step 8) worth the cost.

**(2) Internal reporting tool → Lightweight Architecture Evaluation.** Justification: the slides define it as ATAM-based, "a single day or half-day," "carried out entirely by members internal to the organization," typically 4–6 hours, with the standard agenda compressing steps 2–6 and 9.

**Skipped steps:** **Step 7 (brainstorm scenarios)** — because the assembled internal stakeholders are expected to contribute their scenarios already in step 5 (utility-tree generation); **Step 8 (analyze new scenarios)** — because all analysis is done in step 6. This is acceptable because the stakeholder population is small, internal, and shared understanding already exists — the divergence risk that steps 7–8 exist to catch is low. The trade-off, acknowledged in the slides: "lower scrutiny and objectivity may not probe the architecture as deeply."

**Key concepts tested:** ATAM cost/participation profile, lightweight evaluation scope, step-skipping rationale.

---

### Q28. [Conformance — Drift forensics]
Four code practices are discovered in a system that was designed with a layered, publish-subscribe architecture with a shared logging framework: (a) a UI screen executing SQL directly; (b) a new feature notifying six modules one-by-one instead of publishing an event; (c) three components writing their own log files; (d) pricing rules implemented in stored procedures bypassing the business layer. For each, name the violated architectural rule, the impact, and how conformance checking would detect it.

**Answer:**
All four are the slides' drift archetypes plus one extension.

| Practice | Violated rule | Impact | Detection |
|---|---|---|---|
| (a) UI executes SQL | Layer discipline / DB access only through data-access layer (slides' drift example) | Change amplification; schema coupled to UI; untestable logic | Dependency analysis (e.g., Sonar layer-slice rule "UI → DAO prohibited"); code review |
| (b) Point-to-point notifications instead of pub-sub | Prescribed connector type (publish-subscribe) bypassed | Tight coupling; adding a 7th subscriber requires editing the notifier — modifiability loss; failure coupling | Review the module-interaction view vs code; rule "notifications only via bus" |
| (c) Custom log files | "Not making use of a common logging mechanism" (slides' drift example) | No centralized diagnosis; inconsistent formats; ops blindness | Static search for direct file writes vs Log4J logger usage; Sonar rule |
| (d) Rules in stored procedures | Business rules must live in the business layer (separation of concerns; also workflow/rules-engine intent) | Rules invisible to version control/tests in the app; vendor coupling; duplicated logic | Architecture-analysis rule "all business logic in business layer"; review of procedure inventory |

**Recommendation:** For each: fix by relocating to the prescribed structure, then prevent via architecturally evident coding (annotate layer/publisher-role in code), folder discipline, and automated dependency gates — the slides' conformance toolkit.

**Key concepts tested:** Drift categories, prescribed structure vs implementation, conformance detection.

---

### Q29. [Conformance — Systemic prevention at scale]
An organization of 50 developers with high turnover finds that architectural drift recurs every two sprints despite code reviews. As the architect, apply the course's four consistency techniques (plus supporting practices) to make conformance systematic, explaining how each technique mechanically prevents drift.

**Answer:**
**The four techniques (from the slides):**

1. **Embed design concepts in the code — architecturally evident coding style.** Indicate in the code the architecture being implemented: which layer a class belongs to; whether a class is a Publisher or Subscriber; whether a component inserts or retrieves from the message queue. *Mechanism:* a new developer reading `// Layer: Business — Publisher` cannot innocently bypass the bus; reviewers instantly see mismatches.
2. **Use frameworks.** Spring (layering/dependency injection), Hibernate (ORM in the data layer), JMS (publish-subscribe), Log4J (common logging), Drools (rules engine), even Autosar in automotive. *Mechanism:* the framework *is* the architecture at runtime — writing around JMS to call a module directly is visible and awkward; the "path of least resistance" follows the architecture.
3. **Use code templates.** E.g., the slides' fault-tolerant template with sections `Get event → case Normal / Update State data / Switch over`, where "code needs to be accommodated in this template." *Mechanism:* template structure forces prescribed control flow — you cannot add out-of-protocol behavior without visibly diverging from the template.
4. **Update architecture documentation.** Changes to code accompanied by doc changes; at minimum mark obsolete parts "no longer applicable" (this preserves trust in the rest); synchronize arch doc with code at release time. *Mechanism:* the documented architecture remains the true reference that conformance is checked against.

**Supporting practices (slides):** educate new team members about the architecture (directly attacks the turnover problem); code reviews with an architectural checklist; folder-per-architectural-aspect discipline (layer/service/UI/external interfaces) so physical structure mirrors logical structure.

**Recommendation:** Combine all: evident coding + frameworks + templates make conformance the default; reviews + automated dependency checks catch the residue; documentation sync keeps the reference honest.

**Key concepts tested:** Consistency techniques, mechanism-level prevention, onboarding as conformance control.

---

### Q30. [Architecture & testing — Airline booking]
A test lead asks your help on an airline booking system that integrates a payment gateway and a loyalty system. Using the course's architecture-and-testing relationship, explain (a) how architecture prioritizes test cases, (b) how architecture helps build the integration test plan, and (c) what the architecture must provide for testability — with concrete examples for this system.

**Answer:**
**(a) Prioritizing test cases.** Architecture helps prioritize test cases based on ASRs — and the work product used is the **utility tree**: scenarios "high on business value and high on architectural impact" translate into high-priority test cases. Here: "book a seat at peak with card payment, p95 ≤ 2s (H,M)" and "switch payment provider without downtime (H,M)" → payment-path tests and provider-switch tests rank above, say, FAQ-page tests.

**(b) Integration test plan.** Architecture tells us **which modules interact with each other and which modules depend on which**; this identifies integration test cases and the modules needed for integration testing. For booking: Booking Engine ↔ Inventory ↔ Pricing ↔ Payment Adapter ↔ Loyalty client — the dependency structure yields the integration sequence (booking↔inventory first, then payment, then loyalty posting), and tells which stubs are needed at each stage.

**(c) Testability requirements the architecture must support** (slides): **ability to switch data sources** (test data vs production data) — booking services must accept a test DB config; **roll back changes made by test cases** (restore original state so more tests can run later) — transactions on booking writes; **ability to replace components** with simulators of the payment gateway, loyalty system, and external identity (the slides' explicit example), enabling deterministic failure-path testing (declined card, loyalty timeout).

**Recommendation:** Deliver the utility-tree-driven test priority list, the integration matrix from the dependency view, and the simulator/rollback requirements to QA.

**Key concepts tested:** Utility tree → test priority, dependency view → integration plan, testability design.

---

### Q31. [Core banking — Reconstruction pipeline]
A core-banking system's documentation is five years stale; developers keep finding dependencies absent from the documents (e.g., a transaction controller connecting straight to the database). As the architect, walk through the architecture-reconstruction pipeline end-to-end for this system, stating inputs, intermediate artifacts, and the violation-detection step, including the tool category you would use.

**Answer:**
**Pipeline (per the slides):**
1. **Raw view extraction** — extract information from source code, execution traces, and build scripts: classes, the files/data they use, caller–callee relationships, global data accessed by different objects. (For the banking system: entity/session bean classes, DAOs, controller classes, JDBC call sites.)
2. **DB construction** — put extracted data into a common format (repository), so views can be combined and queried.
3. **View fusion** — combine the views: *source-code analysis gives a static view* (structure); *if some objects are dynamically bound at runtime, the execution trace provides this information*; then an expert may group elements into layers. Use the technical team for a broad understanding of the system before starting aggregation (Vanish lesson).
4. **Architecture analysis** — validate correctness of the reconstructed elements against intended rules: "a layer calls objects in adjacent layers only" and "all DB access should be via entity beans only." Tools (Sonar-style layer/slice definitions; ARMIN/Lattix for visualization; **DiscoTech for dynamic analysis**) populate layers and highlight violations.
5. **Iterate** — refine groupings/rules until the reconstructed architecture stabilizes.

**Violation detection for this case:** the hypothesized layer/slice model (Controller/Service/Entity/Data) will flag the "illegal connection": a TxControllerBean writing directly to the database instead of going through entity beans — precisely the dynamic-analysis violation shown in the slides.

**Outcome:** reconstructed as-is architecture + violation list → either refactor the code or legitimately update the documented architecture (both, with rationale).

**Key concepts tested:** Raw view extraction → DB construction → view fusion → arch analysis → iterate; static vs dynamic evidence; tools.

---

### Q32. [Reconstruction inputs — Matching question to evidence]
During reconstruction of a 10-year-old logistics platform, you must answer four questions. For each, state the best information source (static source analysis, execution traces, build scripts, human expertise) and justify: (a) which classes should be grouped into which modules; (b) which service implementations are actually selected at runtime; (c) which external libraries each deployable unit depends on; (d) which interactions dominate the hot path.

**Answer:**
- **(a) Class → module grouping:** **static source analysis** (supplemented by human expertise). Source analysis extracts classes, files, and caller–callee relations — the raw material for grouping; an expert then groups elements into layers/modules (view fusion's expert step). Traces are secondary here because structure, not frequency, is the question.
- **(b) Which service implementation is selected at runtime:** **execution traces**. The slides: "if some objects are dynamically bound at run time then execution trace will provide this information" — static analysis cannot resolve runtime binding (polymorphism/dependency injection); only observed executions reveal which implementation actually runs.
- **(c) External library dependencies per deployable unit:** **build scripts** (with static analysis). Build scripts encode the compilation/packaging dependencies — exactly what "implementation" and build structure reasoning requires; source-level imports corroborate.
- **(d) Hot-path interactions:** **execution traces** (with instrumentation). Frequency, duration, and sequence of calls are dynamic properties; the trace graph representation lets you spot the dominant subtrees (the slides' graph-of-trace discussion).

**Recommendation:** Fuse all four into the common repository (DB construction) before grouping — each question is answerable from the fused view, and contradictions between static and dynamic views are themselves findings (e.g., a declared dependency never exercised).

**Key concepts tested:** Input-to-question mapping, static vs dynamic vs build-script evidence, view fusion rationale.

---

### Q33. [Industrial monitoring — Real-time performance tactics]
A plant-monitoring system receives sensor events every 100 ms per sensor (2,000 sensors), and each event must be processed within a 250 ms deadline; occasionally, alarm events must preempt routine analytics. Latency currently spikes to 900 ms during shift changes. As the architect, formulate the performance scenario and select tactics from both tactic families to meet the deadline.

**Answer:**
**Performance scenario:** Source — internal (sensors); Stimulus — arrival of periodic events (100 ms) plus sporadic alarm events; Artifact — processing pipeline; Environment — peak load (shift change, all sensors active); Response — process events; Response measure — latency ≤ 250 ms, **jitter** bounded, **miss rate** ≈ 0 for alarms.

**Tactics — Control resource demand:**
- **Manage sampling rate** — reduce sampling frequency for non-critical sensors if fidelity loss is acceptable (e.g., analytics sensors 200 ms).
- **Prioritize events** — alarms ranked above routine analytics; the scheduler serves high-priority events first (directly fixes the preempt requirement).
- **Bound execution times** — cap per-event processing; overflow goes to a secondary path.
- **Limit event response / increase resource efficiency** — process up to a set maximum rate for predictable behavior; optimize the parsing/aggregation algorithms on the hot path.

**Tactics — Manage resources:**
- **Introduce concurrency** — parallel processing of independent sensor streams.
- **Maintain multiple copies of computations** — replicated analytics workers to remove contention at shift-change peaks.
- **Bound queue sizes** — prevent unbounded buffering from turning latency into an outage; define overflow policy (drop routine, never alarms).
- **Schedule resources** — priority/fixed scheduling of the CPU and DB writer.

**Trade-off:** sampling reduction loses some analytic fidelity; bounding queues can drop routine data — both acceptable if the alarm path's deadline is sacred. Verify with a performance model and load test at 2× peak.

**Key concepts tested:** Performance scenario (latency/deadline/jitter/miss rate), both performance-tactic families.

---

### Q34. [Chemical plant — Availability under a hard deadline]
A safety-related controller must recover from a primary-unit failure within **2 seconds**. Current design: passive (warm) standby with periodic state updates, plus heartbeats at 1 s. Analysis shows state resynchronization after failover takes up to 6 s — during which control output is stale. As the architect, evaluate the availability-tactic combination and redesign within the deadline, showing the detection and recovery choices and their trade-offs.

**Answer:**
**Problem analysis:** two tactic interactions violate the deadline. (1) *Detection:* heartbeat at 1 s detects absence only after ≥ 1–2 s — consuming most of the budget. (2) *Recovery:* passive redundancy (warm spare) requires **state resynchronization** on failover — the 6 s stale-output window.

**Redesign:**
- **Detection:** raise heartbeat rate (e.g., 200 ms) and add a local **watchdog** on the spare that supervises the primary — the slides' combined heartbeat+watchdog pattern, chosen because "watchdog is simple and has proved reliable" and detection time becomes bounded and known (≤ 400 ms).
- **Recovery:** switch the protection group from **passive to active redundancy (hot spare)**: both units process identical inputs in parallel and the spare maintains **synchronous state** — failover needs no resynchronization. Worst-case rollover becomes compute-bounded (design to ≤ 1.5 s), leaving margin inside the 2 s deadline.
- **Reintroduction:** the recovered old primary re-enters in **shadow mode** for a predefined period before resuming active duty.

**Trade-off:** active redundancy doubles steady-state compute (cost) and requires ensuring identical input distribution — versus the warm spare's lower cost but unacceptable stale-output window. With a safety deadline and 99.999%-style availability (the slides' CPU-failure scenario with guaranteed ≤ 2 s detection), the compute cost is justified.

**Recommendation:** heartbeat (200 ms) + watchdog for detection; active hot-spare protection group for recovery; shadow-mode reintroduction; validate with fault-injection tests at the 2 s boundary.

**Key concepts tested:** Detection/recovery/reintroduction tactic interplay, active vs passive redundancy under deadlines.

---

### Q35. [Government digital payments — Business goals → architecture]
A government wants to "speed up cashless transactions" nationwide. Using the course's business-goal framework, (a) express the goal in the "X to Y by when" format, (b) trace how it produces quality attributes and architectural decisions, (c) note which business goals would produce *no* architectural influence, and (d) name one non-architectural solution that could also serve this goal.

**Answer:**
**(a) Format:** "From 20% to 60% of retail transactions cashless within 18 months" — current state X, future state Y, by when.

**(b) Trace (Figure 3.2 logic):** Business goals → (lead to) Quality attributes → Architecture.
- Widespread adoption by ordinary citizens → **Usability** ("easy steps" — the BHIM example in the slides): scenario: "A first-time smartphone user completes a payment in ≤ 3 steps within 2 minutes (H,M)"; tactics: minimal flows, clear feedback (maintain task/system model).
- Citizens' trust in money movement → **Security** ("3-factor authentication" in the BHIM example): scenario: "A fraudulent transfer attempt is blocked and reported within 5 s (H,M)"; tactics: authenticate actors, encrypt data, audit.
- National scale → performance/availability scenarios for peak festival loads; these produce structures: tiered architecture, caching, redundant payment rails.

**(c) No architectural influence:** some business goals never touch the architecture — e.g., "reduce operating cost by 5%" might be achieved by turning down office thermostats or printing less paper (the slides' example). Not every goal is an ASR source.

**(d) Non-architectural solution:** the same goal could be advanced by *incentive schemes* (cashback for digital payments) or merchant onboarding campaigns — organizational/marketing actions, not software structures. The architect must recognize which goals belong to the system and which to the program.

**Key concepts tested:** Business-goal format, goal→QA→architecture chain, goals with no architectural effect, non-architectural solutions.

---

## Section 3 — High Difficulty

### Q36. [Integrated — Financial transaction platform]
You are the chief architect for a financial transaction platform. Requirements: 99.99% availability; p95 latency ≤ 500 ms; full PCI-grade security (encryption, audit); and the ability to implement quarterly regulatory rule changes within 5 working days. Deliver: (1) the architectural drivers as quality attribute scenarios; (2) the principal conflicts; (3) tactic selections per attribute; (4) the top of a utility tree; (5) how you would evaluate this architecture and what risk themes you would expect.

**Answer:**
**(1) Drivers as scenarios:**
- Availability: "Primary payment node crashes at peak; detect ≤ 2s; failover ≤ 30s; availability ≥ 99.99% (H,M)."
- Performance: "A transfer completes in ≤ 500 ms (p95) at 5,000 TPS (H,M)."
- Security: "A compromised credential cannot read or modify another customer's data; all access audited and tamper-evident (H,M)."
- Modifiability: "A new regulatory rule is configured and deployed in ≤ 5 working days with no core-code change (H,M)."

**(2) Principal conflicts:**
- *Security vs performance:* encrypting all traffic and data at rest adds CPU per transaction — the slides' headline example ("increasing security often decreases performance").
- *Modifiability vs performance:* rules engines/workflow engines add interpretation overhead on the hot path.
- *Availability vs cost:* active redundancy doubles infrastructure.

**(3) Tactics:**
- Availability: heartbeat + monitor detection; active redundancy on the transaction path; transactions (ACID) for integrity; shadow-mode reintroduction.
- Performance: connection pooling; multiple copies of data (read replicas); bound queue sizes; keep intermediaries off the hot path; parameterized/stored queries.
- Security: authenticate/authorize actors; encrypt data; audit; separate entities; parameterized SQL.
- Modifiability: rules engine for regulatory logic (defer binding); encapsulate the payment/limits subsystems behind interfaces; restrict dependencies.

**(4) Utility tree (top):**
```
Utility
├── Availability – fault recovery: failover ≤30s, 99.99% (H,M)
├── Performance – latency: ≤500ms p95 @5,000 TPS (H,M)
├── Security – confidentiality/integrity + audit (H,M)
└── Modifiability – regulatory change ≤5 days, no core change (H,M)
```

**(5) Evaluation:** run a **full ATAM** (mission-critical, costly, high risk): external evaluation team, 12–15 stakeholders, steps 1–9. Expected risk themes: "encryption cost not budgeted into latency targets" (security–performance conflict untested at load), "rules-engine change process untested within 5 days," "failover at peak load unvalidated." Each theme maps to drivers (regulatory compliance, market trust), forcing management decisions on cost vs assurance.

**Key concepts tested:** Integrated driver analysis, cross-attribute conflict, tactics, utility tree, ATAM application.

---

### Q37. [Integrated — Layer violations + conformance regime]
In a 4-layer enterprise system (Presentation → Business → Service → Data), an audit reveals: the UI calls DAOs directly; two service components import presentation widgets; business logic sits inside JSPs; and data-access code is duplicated in three places. Deliver: (1) violation inventory with impact per violation; (2) a corrected dependency model; (3) a conformance regime that would have caught these; (4) how you prevent recurrence in CI.

**Answer:**
**(1) Violations & impact:**
- UI → DAO direct calls: skips business *and* service layers; schema changes ripple into UI; business rules bypassed; untestable logic. (The canonical "not sticking to the discipline of layers.")
- Service → presentation imports: inverts the layer direction; creates a dependency cycle; services become unusable headless (blocks reuse for other clients and tests).
- Business logic in JSPs: presentation contains domain rules; duplicated across screens; change amplification; no single place to verify rules.
- Triplicated data-access code: violates reuse/exchangeable-parts principle; three copies drift independently.

**(2) Corrected dependency model:**
```
Presentation  →  Business  →  Service  →  Data   (unidirectional, adjacent only)
   UI/JSPs        Façade,       Service      DAOs
                  Workflow,     interfaces   (single set)
                  Entities
```
Logic moves from JSPs into business components; services expose interfaces only; one DAO set; UI never touches DAOs.

**(3) Conformance regime that would have caught it:**
- **Architecturally evident coding style** — every file annotated with its layer (would have made "JSP with business rules" obvious).
- **Folder-per-layer discipline** — UI files in `/ui`, DAOs in `/data`; imports across folders visible in diffs.
- **Automated dependency analysis** (Sonar-style layer/slice rules: "layer may call adjacent only," "UI → DAO forbidden") — flags (a) and (b) mechanically.
- **Architecture & code review checklist** — "where does this logic belong?" catches (c); "does this already exist?" catches (d).

**(4) CI prevention:** make dependency rules a build-breaking gate (fitness-function style), run Sonar on every merge, and keep the module view updated at release so the rules checked match the intended architecture. Trade-off: gates add friction — justified because each violation above has already produced measurable change-amplification cost.

**Key concepts tested:** Layer violation taxonomy, dependency direction, conformance tooling, CI gating.

---

### Q38. [Integrated — Legacy reconstruction and re-documentation]
You inherit a 15-year-old insurance claims monolith: original architects gone, no documentation, key rules partially in code and partially tribal knowledge. Regulators demand an up-to-date architecture description within 3 months, and the business wants to extract reusable components (a logging facility, a security component). Plan the work: reconstruction method, evidence sources, role of experts, violation finding, reusable-component identification, and the documentation you will deliver.

**Answer:**
**Why reconstruction (purpose from slides):** understand architecture where no documentation exists; identify reusable components (logging, security) — both goals stated in the slides.

**Method (iterative pipeline):**
1. **Raw view extraction:** source code (classes, caller–callee, data access), execution traces (runtime bindings, hot paths for the claims flow), build scripts (deployable units, library dependencies).
2. **DB construction:** all extracted facts into a common repository format.
3. **View fusion:** static (source) + dynamic (traces) views fused; **use the technical team's broad understanding before aggregation** (the Vanish lesson) — surviving senior developers identify domain groupings (policy, claims, payment, correspondence) that pure statistics would miss. Note tribal knowledge: capture it *now* via interviews and encode it as grouping rules — before the last experts leave.
4. **Architecture analysis:** define intended rules ("all DB access via the persistence layer," "UI never calls payment internals") and let tooling (ARMIN/Sonar/Lattix-class) flag violations; iterate groupings.
5. **Violation findings → two classes:** genuine violations (refactor backlog) vs *de facto* architecture (document as-is).
6. **Reusable components:** from the fused view, the logging and security clusters will appear as heavily-called, low-inbound-coupling groups — extract with explicit interfaces; their reusability *depends on* the coupling analysis from fusion.

**Deliverables within 3 months:** as-is module/decomposition view, deployment view, claims-flow scenario trace, violation register with dispositions, extracted components with interfaces, and a forward documentation process: **update at release, mark obsolete sections "no longer applicable," architecturally evident coding** so drift doesn't recur. Trade-off: traces only cover executed paths — supplement with targeted instrumentation of untested flows.

**Key concepts tested:** Full reconstruction pipeline, expert involvement, reuse identification, forward-looking documentation.

---

### Q39. [Healthcare — Full utility tree construction]
You are architect for Nightingale-style hospital information system. Stakeholder goals: protect patient data, no downtime for clinical operations, fast response at peak, quick bug turnaround, and fee/protocol changes without releases. Construct a full utility tree (≥ 5 attributes) with concrete scenarios, (importance, difficulty) pairs, and then: (a) identify the (H,H)/(H,M) ASRs; (b) explain the design priority order; (c) flag one scenario that looks easy but is deceptively hard.

**Answer:**
```
Utility
├── Security
│   ├── Confidentiality — a physical therapist may view only the
│   │   orthopaedic portion of a record, never financial data (H, M)
│   └── Integrity — intrusion detected and reported to authorities
│       within 90 s (H, M)
├── Availability
│   └── No downtime — new database release hot-swapped
│       with zero downtime (H, L)
├── Performance
│   ├── Response time — account update at peak completes < 0.75 s (H, M)
│   └── Response time — update at double peak < 4 s (L, M)
├── Maintainability
│   └── Routine changes — search-performance bug fixed and
│       distributed ≤ 3 person-days (H, M)
├── Configurability
│   └── User-defined changes — hospital fee change applied by config
│       team in 1 working day, no code (H, L)
└── Extensibility
    └── New product — blood-bank donor tracking built ≤ 2 person-months (M, M)
```

**(a) ASRs:** confidentiality (H,M), integrity (H,M), 0.75 s response (H,M), bug fix ≤ 3 person-days (H,M) — critical and non-trivial; they will determine structures (authorization model, encryption/monitoring, resource management, modular fix-and-deploy pipeline).

**(b) Priority order:** design and evaluate in (H,H)→(H,M)→(H,L)→(M,…) order. (H,L) items get cheap, verified tactics — not deep design effort. (L,M) documents a tolerance ("graceful degradation" at 2× peak) but consumes little design attention.

**(c) Deceptively hard:** "hot-swap DB with zero downtime (H,L)" looks like a routine upgrade, but zero-downtime schema migration + state resynchronization is genuinely difficult — the (L) difficulty rating is wrong; it should be renegotiated to (H,M) in the ATAM, or consciously degraded to a maintenance-window scenario with stakeholder sign-off. Utility trees are refined exactly this way.

**Key concepts tested:** Multi-attribute utility tree, priority-driven ordering, priority renegotiation.

---

### Q40. [Integrated — ATAM end-to-end on telecom billing]
You lead an ATAM on a telecom billing system. Business drivers: accuracy and auditability of billing; 99.95% availability during month-end; ability to add new plan types monthly; strict subscriber-data privacy. The architecture uses a layered design with a rating engine, a rules-driven plan catalog, and an active/passive database pair. Execute the nine ATAM steps, stating the concrete artifacts your team produces at each step.

**Answer:**
**Participants (setup):** external evaluation team of 3–5 (team leader, evaluation leader, scenario scribe, proceedings scribe, questioners); decision-makers incl. project manager, customer, and the architect (participation mandatory); 12–15 stakeholders (developers, testers, ops, privacy officer, product).

1. **Present the ATAM** — evaluation leader explains process, steps, outputs; sets expectations.
2. **Present business drivers** — PM presents: functions (rating, invoicing), constraints (regulatory audit), business goals (accuracy → trust; monthly plan launch → competitiveness), stakeholders, and the architectural drivers (the four listed).
3. **Present the architecture** — architect describes the layered design, the rating engine, rules-driven catalog, DB failover, using the views that best address availability and modifiability; technical constraints (middleware) included.
4. **Identify architectural approaches** — team catalogs patterns/tactics: layered style; active/passive redundancy (availability tactic); rules engine (defer binding, modifiability); captured publicly as the analysis basis.
5. **Generate utility tree** — leaves e.g.: Availability–recovery: "active node fails at month-end peak; failover ≤ 60s, zero lost CDRs (H,M)"; Modifiability–new plan: "plan type added by config ≤ 2 days (H,M)"; Security–confidentiality: "no employee can read subscriber call detail without logged authorization (H,M)"; Performance: "rating of 1M CDRs/hour at month-end (H,M)."
6. **Analyze architectural approaches** — for each top scenario the architect walks through decisions; team records: **sensitivity point** — rules-engine evaluation speed determines rating throughput; **trade-off point** — rules engine improves modifiability but adds rating latency; **risk** — passive DB resync time unmeasured at month-end volume; **non-risk** — checksum-verified CDR ingestion proved reliable.
7. **Brainstorm & prioritize scenarios** — stakeholders vote; ops adds "hot plan migration during month-end" — a new driving scenario; comparison with the utility tree shows partial alignment (new modifiability concern = divergence signal).
8. **Analyze approaches (new scenarios)** — hot migration traced: rating engine holds plan cache; stale-cache risk identified; decision to add cache invalidation recorded.
9. **Present results** — outputs: approaches documented; prioritized scenarios; utility tree; risks (unmeasured resync, cache staleness), non-risks; sensitivity (rules-engine speed) and trade-off (rules engine: modifiability vs latency) points; **risk themes**: "availability under month-end load insufficiently validated" → threatens the 99.95% driver; "plan agility depends on untested config path" → threatens monthly-launch driver. Phase 3: report delivered; mitigation planning follows.

**Key concepts tested:** Complete ATAM execution with per-step artifacts and correct output vocabulary.

---

### Q41. [Integrated — Regulatory modifiability deep dive]
A bank must absorb quarterly rule changes (KYC thresholds, interest rules, reporting formats). Current practice: hard-coded rules; each change costs 4–6 weeks and occasionally breaks unrelated features. As the architect, propose the modifiability tactics and structures, analyse their side effects on performance, testability, and availability, and specify the design-checklist checks you would run.

**Answer:**
**Tactics/structures:**
- **Defer binding** — rules engine (e.g., Drools) so thresholds and interest formulas are data/configuration, changeable without compile; workflow engine for multi-step regulatory processes (per the slides' business-layer techniques: "use workflow engine, rules engine etc. for modifiability").
- **Increase cohesion / encapsulate** — move all regulatory logic into a single regulatory module with a stable interface; nothing else may interpret rules.
- **Reduce coupling — use intermediary/restrict dependencies** — business components ask the rules service; they never embed rule fragments; a façade keeps the rules model stable when the engine is replaced.

**Side effects:**
- *Performance:* rule interpretation adds per-transaction cost; on the payment hot path this can breach the latency scenario. Mitigate: precompiled rule sets, caching of rule results for repeated patterns; keep hard-coded implementations only for provably static rules (trade-off acknowledged: that reintroduces some change cost for those rules).
- *Testability:* rules change quarterly without full regression — the architecture must support **rule-level test suites** (record/playback of representative transactions; sandbox to run new rule versions against copies of production data — the slides' sandbox tactic).
- *Availability:* a broken rule can break a core flow. Mitigate: rule versioning with instant rollback; shadow evaluation (new rules run alongside, differences alerted) before activation.

**Design checklist checks:** Allocation of responsibilities — for each likely change category (quarterly), are responsibilities needing change placed together in one module? Binding time — for each change, the latest time it must be made, and a defer mechanism delivering capability at that time, with the cost of introducing it counted; do not create so many binding choices that their dependencies become unmanageable. Choice of technology — does the rules engine support testing/versioning/rollback, and is it modifiable itself?

**Recommendation:** Rules engine + regulatory module + versioned, shadow-evaluated rule deployment; measure both the 5-day change scenario and the latency scenario after each change.

**Key concepts tested:** Modifiability tactics with side-effect analysis, checklist application, cost of defer binding.

---

### Q42. [Integrated — ADD full application: insurance quoting]
Design the "online quote" subsystem of an insurance platform using ADD. Known drivers from the QAW: (D1) quote returned in ≤ 3 s at 2,000 concurrent users (H,M); (D2) adding a new product line ≤ 10 person-days without core change (H,M); (D3) rating service survives a node crash (H,M). Show the ADD iteration loop with tactic selection, module instantiation, and how you decide when to stop.

**Answer:**
**ADD loop (iterate): choose module → establish drivers → choose tactics → instantiate modules → verify scenarios → reprioritize remaining drivers.**

**Iteration 1 — Quote Orchestrator:**
- *Drivers:* D1, D3.
- *Tactics:* D1 → manage resources: multiple copies of computations (replicated rating workers); bound queue sizes in front of rating; caching of static product tables (control demand: reduce overhead). D3 → heartbeat-based detection + passive spare for the orchestrator; transactions for quote persistence (prevent faults: atomicity).
- *Instantiation:* QuoteAPI (service layer façade), Quote Orchestrator, Rating worker ×N, Quote store, Health monitor. Dependencies documented per module (orchestrator may call rating & store; nothing calls orchestrator directly except the API façade).

**Iteration 2 — Rating & Product catalog:**
- *Drivers:* D2 (product onboarding), residual D1.
- *Tactics:* D2 → encapsulate each product's rating rules behind a common interface; defer binding — product definitions and rules as configuration consumed by a generic rating engine (the regulatory/modifiability pattern); abstract common services so near-identical product rules share one implementation. Residual D1 → maintain multiple copies of data (read-only product tables cached in each rating worker).
- *Instantiation:* ProductRule interface, generic Rating Engine, per-product rule configs, catalog cache.

**Verification:** trace each scenario through the design: 2,000-user peak → queue bounds + N workers sized by performance model; node crash → heartbeat detects ≤ 2 s, passive spare resumes; new product → config + rule definition, no core edit.

**Stopping:** when no remaining driver has (H,·) priority and each top scenario is demonstrably satisfied by the instantiated structure (generate-and-test converged), hand the module-level design to teams; revisit ADD when new ASRs emerge (agile context).

**Key concepts tested:** ADD iteration mechanics, tactic-to-driver mapping, module instantiation, convergence criterion.

---

### Q43. [Integrated — Agile architecting at scale (WebArrow-style)]
A 60-person team is building a 200-KSLOC web-conferencing product with unstable requirements and demanding latency/scalability goals, using two-week sprints. Management asks: "how much architecture, when, and how do we resolve the latency-vs-scalability trade-offs?" Design the architecting approach: sweet-spot positioning, working modes, use of spikes, and documentation policy.

**Answer:**
**Sweet-spot positioning:** 200 KSLOC with unstable requirements and latency assurance goals → between the 100-KSLOC (≈20%) and 1,000-KSLOC (≈40%) sweet spots; rapid change pulls left, high assurance pulls right. **Recommend ≈ 20–25% early architectural investment**, focused on the latency/scalability-critical structures — not full BDUF. Per the slides' authors' advice for large projects with unstable requirements: "start by quickly designing a candidate architecture even if it leaves out many details; be prepared to change and elaborate as you perform your spikes and experiments, and as functional and QA requirements emerge and solidify."

**Working modes (WebArrow lesson):** operate simultaneously **top-down** (design/analyze structures against demanding QA requirements and their trade-offs) and **bottom-up** (analyze implementation-specific and environment-specific constraints and fashion solutions). Architecture evaluation fits the agile process as lightweight reviews tied to stakeholder concerns.

**Spikes (experiments to make trade-offs):** run time-boxed experiments answering the questions the architecture cannot settle on paper, e.g.: "does moving from local flat files to a distributed DB hurt feedback latency?" "what scalability gain from threading model A vs B?" "how many meeting participants can one server host?" "what is the right ratio between DB servers and meeting servers?" Spike results become measured trade-off points feeding architecture revision — the WebArrow team's emphasis on experimentation was the key factor.

**Sprint cadence:** sprint 0–2: candidate architecture + top 5 utility-tree scenarios; each sprint: architecture tasks in the backlog; refactoring scheduled as technical debt grows (the middle-ground: initial architecture, run with it, refactor when debt too great).

**Documentation policy:** "write for the reader; if the reader doesn't need it, don't write it — but the reader may be a maintainer or a newcomer not yet on the project." Document the module view, the concurrency/scalability decisions and their measured rationale, and the integration test plan — no exhaustive views.

**Recommendation:** Candidate architecture ≈ 20–25%, spike-driven trade-off resolution, lightweight peer evaluations, minimal but living documentation.

**Key concepts tested:** Sweet-spot calibration, top-down+bottom-up, spikes, agile documentation.

---

### Q44. [Integrated — Medical infusion pump]
An infusion pump must deliver medication doses within 50 ms of a commanded rate change, must never enter a state that harms the patient, must remain operable ≥ 99.999% of operation time, and nurses must be able to see its status at a glance. Hardware budget is tight (size, power). Produce: drivers as scenarios, the tactic set, the safety analysis (using availability tactics), and an explicit argument for why the "highest performance" architecture is not automatically correct here.

**Answer:**
**Drivers as scenarios:**
- Performance: "Rate-change command accepted → actuator command issued ≤ 50 ms (H,M)" — miss rate must be ~0.
- Safety (≈ availability tactics per the slides: safety's architectural concerns are almost identical — preventing, detecting, recovering): "Sensor disagrees with commanded state → pump enters safe mode and alarms ≤ 200 ms (H,M)."
- Availability: "Any single software fault does not stop or misdose therapy during a 12-hour infusion (H,M)."
- Monitorability: "Nurse sees current rate, remaining volume, and alarms within 1 s of change (M,M)."
- Constraints: size, power, certification — zero degrees of freedom.

**Tactics:**
- Performance/deadline: bound execution times; prioritize control-loop events over UI; schedule the control loop; limit sampling where safe.
- Detection: heartbeat between control module and supervision module; self-test at startup and periodically; condition monitoring (validating design assumptions — dose-rate plausibility); sanity checking of sensor values; timestamping to detect out-of-order messages.
- Recovery/prevention: active redundancy or watchdog-supervised escalation-of-restart for the controller (vary restart granularity to minimize affected service); transactions for dose records; **increase competence set** — the control component handles more abnormal cases as normal operation; removal from service on unsafe condition.
- Monitorability (the slides' attribute): status event stream to the nurse UI, separately scheduled so UI can never starve the control loop.

**Why not "highest performance":** a maximal-performance design (fastest CPU, aggressive concurrency, richest features) would violate the power/size constraints, expand the certification surface, and add concurrency-induced non-determinism — directly hostile to safety and testability (limit non-determinism is a testability tactic). There is "no inherently good architecture; architectures are more or less fit for some purpose." Here fitness = meeting the 50 ms deadline deterministically with minimal, analyzable structure, not raw speed.

**Recommendation:** Simple, deterministic control loop + layered supervision (heartbeat, watchdog, self-test) + prioritized status channel; validate with fault injection at every deadline boundary before certification.

**Key concepts tested:** Safety-via-availability tactics, deadline scheduling tactics, monitorability, fit-for-purpose reasoning.

---

### Q45. [Integrated — Health data exchange: interoperability ASR]
A national health exchange must let 300 hospitals exchange patient records; new hospital systems appear continuously and must be integrated without redeploying the exchange. Data is regulated (patient privacy). Produce: the interoperability ASR as a scenario; the tactics; the design-checklist items that most constrain the design; and the principal trade-offs (openness vs security vs performance).

**Answer:**
**Scenario (from the interoperability general scenario):** Source — a hospital system; Stimulus — request to exchange patient-record information; Artifact — the exchange and the hospital systems; Environment — participating systems **discovered at run time** (new hospitals onboard continuously); Response — the request is appropriately accepted and information exchanged successfully, or appropriately rejected with the right entities notified, and the request logged; Response measure — ≥ 99% of exchanges correctly processed/rejected; rejected requests logged with non-repudiation.

This is an **ASR**: uniqueness (no current structure handles unknown systems), SLA aspect (99%), high business value, technical risk.

**Tactics:** *Locate → discover service:* known directory service through which hospital systems register and are found, with levels of indirection allowed. *Manage interfaces → orchestrate* (control mechanism coordinates and sequences multi-step exchanges: identity match, consent check, record fetch, audit) and **tailor interface** (add translation/buffering/data-smoothing capabilities so each hospital's format maps to the canonical record).

**Most-constraining checklist items:**
- *Allocation of responsibilities:* must include responsibilities to accept, exchange, reject, notify, and **log** — in an untrusted environment, logging for non-repudiation is essential.
- *Data model:* determine syntax/semantics of exchanged abstractions; if the internal model is confidential, apply transformations between canonical and per-hospital representations.
- *Resource management:* a flood of external requests must never exhaust critical resources (denial of service to legitimate users) — bound queues, admission control.
- *Binding time:* policy for binding to **known and unknown** external systems; mechanisms to reject unacceptable bindings and log them; late-binding mechanisms support discovery of new services/protocols.

**Trade-offs:** *Openness vs security* — late discovery invites untrusted parties; mitigate with registry vetting, authentication, consent enforcement (privacy regulation), and encryption. *Openness vs performance* — discovery plus translation adds per-exchange latency; mitigate with cached directory entries and schema translation at registration rather than per message. *Interoperability vs integrity* — translation can corrupt semantics; mitigate with validation and canonical-model versioning.

**Recommendation:** Registry-based discovery + orchestration + tailored translation interfaces, with non-repudiation logging and admission control as non-negotiable responsibilities.

**Key concepts tested:** Interoperability scenario, locate/manage-interfaces tactics, checklist-driven design, openness trade-offs.

---

### Q46. [Integrated — Documentation redesign with QA views]
An enterprise platform's architecture is documented as one dense diagram despite four active concerns: security reviews fail to find access rules; the network team cannot verify communication paths; reliability engineers cannot find failover logic; and deployment staff ask which servers run what. Diagnose the documentation failure, prescribe a concrete view set (including the quality-attribute views taught), define each view's audience and content, and explain how the views stay consistent and current.

**Answer:**
**Diagnosis:** One diagram cannot address all stakeholders' concerns — the documented problem: "architecture documents do not address the concerns of all stakeholders… complex diagrams… hard to represent." The remedy: "using different notations for several **views**, each addressing one specific set of concerns," plus documentation that applies to more than one view.

**Prescribed view set:**

| View | Audience | Content |
|---|---|---|
| **Module / decomposition** | Developers, maintainers | Responsibilities per module, allowed dependencies, layering |
| **Security view** (QA view) | Security team, auditors | Who may access which data/services; authentication/authorization points; encryption zones; audit points |
| **Communication view** (QA view) | Network team, integrators | Protocols and connectors between components; SSL/IPSec boundaries; message flows |
| **Reliability view** (QA view) | Reliability/ops engineers | Redundancy (active/passive pairs), heartbeat/monitor placement, failover logic, degraded modes |
| **Deployment (allocation) view** | Deployment staff, sysadmins | Which software units run on which processors; network topology; communication pathways |
| **Scenarios** | All | Traces (e.g., "failed node during peak") that tie the views together |

**Consistency:** use the +1 mechanism — shared scenarios that walk through multiple views keep them mutually consistent and valid. **Currency:** changes to code accompanied by doc changes; at release time synchronize the arch doc with code; mark no-longer-applicable sections explicitly (this preserves trust). Optionally enforce with architecture-as-code + dependency fitness checks so the module view cannot silently drift.

**Recommendation:** Replace the mega-diagram with the six views, a short documentation roadmap (which view answers which stakeholder), and a release-time sync process.

**Key concepts tested:** QA views (security/communication/reliability), view selection per stakeholder, consistency & maintenance of documentation.

---

### Q47. [Integrated — Openness platform ASR & utility tree]
A government benefits platform must allow accredited third-party apps to query citizen benefit status, with new accreditations issued monthly. The ministry demands: correct responses ≥ 99% of exchanges; no citizen data leakage; no third-party flood can deny service to citizens. Identify the ASRs, build the top of the utility tree, select tactics per attribute, and state the conflicts the CTO must fund.

**Answer:**
**ASRs (indicators: SLA aspect, uniqueness, high business value, risk):**
- Interoperability: "An accredited app requests benefit status; exchange succeeds or is correctly rejected with logging ≥ 99% of the time; new apps onboard monthly without core change (H,M)."
- Security (confidentiality): "An unaccredited app obtains no citizen data; every access is attributable (H,M)."
- Availability: "A third-party request flood does not degrade citizen service; citizen requests keep p95 ≤ 2 s under attack conditions (H,M)."

**Utility tree (top):**
```
Utility
├── Interoperability – locate/discover: accredited app discovers
│   & calls benefits API, 99% correct handling (H,M)
├── Security – confidentiality: unaccredited access blocked
│   + logged (non-repudiation) (H,M)
├── Availability – resource management: flood absorbed,
│   citizen traffic unaffected (H,M)
└── Modifiability – accreditation onboarding ≤ 2 days (M,M)
```

**Tactics:** Interoperability → discover service via accreditation registry (locate); tailor interface (per-app translation/buffering). Security → identify/authenticate/authorize actors (API keys + OAuth-style actor verification), limit access & limit exposure (few access points), encrypt data, audit all access (recovery tactic: audit). Availability → bound queue sizes and admission control per app (resource management), monitor per-app request rates (detect service denial). Modifiability → defer binding (accreditation metadata as configuration).

**Conflicts requiring funding:** *Openness vs security* — every new integration point widens the attack surface; vetting, authentication, and audit all cost. *Fairness vs availability* — strict per-app quotas protect citizens but cap third-party traffic (business goal of ecosystem growth). *Granular logging vs performance* — full audit of every exchange costs throughput; sample or batch logs where regulation permits. These are explicit budget decisions, not free wins — present each as a trade-off point with the affected drivers.

**Key concepts tested:** Interoperability/security/availability integration, utility tree, trade-off funding decisions.

---

### Q48. [Architecture competence — Architect role failure]
A firm promoted its best coder to chief architect. Twelve months later: stakeholder meetings end in conflict, cost estimates are consistently wrong by 40%, developers ignore the architecture, and the board questions the architecture's value. Using the course's treatment of architecture competence and the professional context, diagnose the missing competencies and design a remediation plan at both individual and organizational levels.

**Answer:**
**Diagnosis — what's missing:** Individual competence "combines technical knowledge, business knowledge, and soft skills like leadership and negotiation." A star coder typically brings only the first.

- *Business knowledge:* the architect must explain to stakeholders "the chosen priorities of different properties, and why particular stakeholders are not having all their expectations fulfilled" — the conflict-laden meetings show this is absent; so is the habit of tracing quality attributes to business value ("why do you want fast response time? — it differentiates us and captures market share").
- *Soft skills:* "diplomatic, negotiation, and communication skills; ability to communicate ideas clearly; manage a diverse workload and switch contexts; be a leader in the eyes of developers and management." Developers ignoring the architecture signals failed leadership/communication, not failed design.
- *Duties beyond producing architecture:* "supporting management and dealing with customers" — and a specific duty: **helping the project manager create cost and schedule estimates early**, via top-down estimates apportioned over the structure, reconciled with bottom-up team estimates. Wrong-by-40% estimates show this duty was never performed.
- *Knowledge:* up-to-date patterns, platforms, standards — plus business considerations.

**Organizational competence gap:** the firm lacks processes that facilitate good architecture — no architect career path, no governance (evaluations, reviews), no architecture community.

**Remediation plan:**
- *Individual:* pair with a senior architect for two quarters; own one stakeholder negotiation and one top-down estimate cycle with mentoring; attend a QAW and an ATAM as an observer then as evaluation leader; explicit training in presentation and negotiation.
- *Organizational:* adopt standard architecture activities in the life cycle (business case → ASRs → create/select architecture → document → **evaluate** → implement/test → **ensure conformance**); institute lightweight periodic evaluations (engenders shared understanding — an intangible but "longest-lasting" benefit); define an architect career framework (duties/skills/knowledge).

**Key concepts tested:** Individual competence triad, professional-context duties (estimates!), organizational competence, governance.

---

### Q49. [Integrated — Conformance + architectural testing for a regulatory audit]
A banking regulator requires the bank to *prove* that, in the shipped code: (R1) layers call adjacent layers only; (R2) no component accesses the database except the DAO layer; (R3) every sensitive operation writes an audit record. Deliver: the conformance mechanism for each rule, the architectural tests that verify behavior (not just structure), and how architecture artifacts (views, utility tree, dependency model) feed the audit evidence.

**Answer:**
**Conformance mechanisms (structure):**
- **R1:** define the layer model (e.g., Sonar-style layers and vertical slices) and the rule "layer calls adjacent layer only"; automated dependency analysis on every build; architecturally evident coding annotations (`// Layer: Business`) make violations visible in review and diffable.
- **R2:** rule "only `*.dao.*` packages may import the DB driver"; static dependency analysis + folder-per-layer discipline; code-review gate.
- **R3:** architecturally evident coding plus a **code template** whose sections include the audit call (like the fault-tolerance template forcing prescribed sections) — a sensitive operation cannot be written without visibly omitting the audit hook; static scan for operations marked `@Audited` against the operations catalogue.

**Architectural tests (behavior):** architecture is tested, not just inspected:
- Failover test driven by the availability scenario (utility-tree top item): kill the primary; assert detection ≤ 2 s, failover ≤ 30 s, zero committed-transaction loss — the scenario's response measures are the assertions.
- Audit-trail test: execute a sensitive operation and an attempted unauthorized one; assert an immutable audit record exists for both, and tamper attempts are detected.
- Integration tests derived from the dependency view: architecture tells which modules interact and depend, so the test plan exercises each prescribed interaction (Business→Service→DAO; no UI→DAO path exercised *because none should exist* — a negative test: attempt construction of such a call and assert the build gate rejects it).
- Prioritization from the utility tree: (H,M) scenarios → highest-priority test cases in the regression suite.

**Audit evidence chain:** the module view (intended rules R1–R3) + dependency analysis reports (structural conformance per build) + scenario-driven test results (behavioral conformance) + release-time doc-sync record (the checked architecture matches the shipped one). This directly answers the "does the implementation conform to the architecture?" activity in the life cycle.

**Key concepts tested:** Rule-based conformance, behavioral architecture testing, evidence chain from views/ASRs/tests.

---

### Q50. [Capstone — Streaming platform: scale, drift, evaluation, documentation]
A media-streaming SaaS: 10× seasonal traffic spikes, 99.95% availability, video start-up latency ≤ 400 ms globally, per-tenant branding/config changes without releases. Independently: the architecture document is 3 years old; Sonar has flagged layer violations; and the CTO asks whether to run a full evaluation. Produce an integrated plan: (1) drivers & top utility-tree scenarios; (2) tactic selections and the scalability-vs-complexity trade-off; (3) evaluation choice with justification; (4) reconstruction response to the drift findings; (5) the documentation set going forward.

**Answer:**
**(1) Drivers & scenarios:**
- Scalability/Performance: "During the seasonal spike, the platform serves 10× normal concurrent streams; start-up latency ≤ 400 ms p90 (H,M)."
- Availability: "Any single node failure causes no playback interruption for affected users beyond 5 s (H,M)."
- Modifiability: "A new tenant's branding, catalog rules, and limits configured ≤ 1 day, no release (H,M)."

**(2) Tactics & the central trade-off:** *Manage resources:* scale-out with replicated stream servers (multiple copies of computations), multiple copies of data (edge-cached catalog), bound queue sizes on ingest, **manage sampling rate** via adaptive quality. *Availability:* heartbeat/monitor detection, warm spares with state resynchronization for control-plane components. *Modifiability:* per-tenant configuration via rules/config (defer binding), façade over the tenant model. **Trade-off (scalability vs complexity):** scale-out demands statelessness — externalized sessions (client-based cookies or server-based persistent session store) and distributed consistency; this complexity is the price of 10× elasticity and the 5 s failure response; scale-up would be simpler but caps both. Because both scenarios are (H,M), the complexity is justified — but it must be *managed* (statelessness enforced by conformance rules, not convention).

**(3) Evaluation choice:** **full ATAM.** Justification: large, costly, market-critical system where an architectural mistake is unacceptable; multi-stakeholder QA conflicts (elasticity vs complexity vs tenant agility) need the two-round scenario analysis and risk-theme mapping to business drivers (subscriber growth, uptime commitments). Lightweight evaluation is reserved for inner-sprint architecture checks between full evaluations.

**(4) Reconstruction response to drift:** the Sonar flags are symptoms of drift (layer-bypassing). Run the reconstruction pipeline: raw view extraction (source + execution traces of a peak-hour playback + build scripts) → DB construction → view fusion (static + dynamic; experts group elements into layers/slices; take the technical team's broad understanding before aggregation) → architecture analysis against rules ("stream worker calls adjacent layer only"; "tenant config read only via config façade") → violation register with dispositions (fix or legitimize) → iterate. Update the documented architecture to the reconstructed truth.

**(5) Documentation forward:** prioritize the **Process view** (high-traffic system — concurrency, latency, scalability per the course's view-selection guide), deployment view (edge/replica topology — availability, security, performance reasoning), module view (teams' responsibilities), and QA views for reliability; +1 scenarios ("10× spike playback") tie views together. Maintain currency: update at release, architecturally evident coding, dependency gates in CI so the 3-year drift cannot recur.

**Key concepts tested:** Capstone integration — drivers, tactics with named trade-off, evaluation selection, reconstruction, view prioritization.

---
---

# FINAL REVISION SECTION

## A. Top 20 Concepts to Revise

1. **Quality Attribute Scenario (6 parts)** — *Why:* every QA requirement in this course is expressed this way; the sample paper demands scenario construction. *Remember:* Source, Stimulus, Environment, Artifact, Response, **Response measure** — must be measurable.
2. **General vs concrete scenarios** — *Why:* distinction is directly examinable. *Remember:* general = system-independent value table; concrete = instantiated, testable.
3. **Utility tree** — *Why:* appears in ASR, QAW, ATAM, and testing contexts. *Remember:* Utility → QA → refinement → scenario leaf with **(Importance, Difficulty)**; (H,M)/(H,H) drive design and test priority.
4. **ASR definition & indicators** — *Why:* sample paper Q1b. *Remember:* profound, measurable effect; indicators = business value/risk, key stakeholder, uniqueness, QoS/SLA, past overruns; triage Critical/Important/Useful.
5. **Architectural drivers** — *Why:* QAW output and ATAM step 4. *Remember:* business goals & constraints, stakeholder concerns, QAs, regulatory, technology, cost, time-to-market.
6. **QAW 7 steps** — *Why:* named framework; sequencing examinable. *Remember:* Intro → Business mission → Arch plan → Drivers → Brainstorm → Consolidate → Prioritize (**30% voting**).
7. **Availability tactics** — *Why:* most heavily illustrated attribute. *Remember:* Detect (ping/echo, monitor, heartbeat, timestamp, sanity, voting, self-test) / Recover (active/passive/spare, rollback, retry, degradation, resync, escalation, shadow) / Prevent (removal from service, transactions, predictive model, competence set).
8. **Performance tactics** — *Remember:* Control demand (sampling rate, limit events, prioritize, reduce overhead, bound execution, efficiency) / Manage resources (increase, concurrency, copies of computations & data, bound queues, schedule). Metrics: latency, deadline, throughput, jitter, miss rate.
9. **Modifiability tactics** — *Remember:* Split module; increase semantic coherence; encapsulate, intermediary, restrict dependencies, refactor, abstract common services; **defer binding** (and its cost).
10. **Security & CIA** — *Remember:* Confidentiality/Integrity/Availability + authentication, non-repudiation, authorization; tactics: detect/resist/react/recover (+audit); "increasing security often decreases performance."
11. **Testability & interoperability tactics** — *Remember:* control/observe (specialized interfaces, record/playback, sandbox, simulators) & limit complexity; locate (discover service) & manage interfaces (orchestrate, tailor).
12. **4+1 Views** — *Why:* sample paper Q2c. *Remember:* Logical–end-user/functionality; Process–integrators/performance/scalability; Development–programmers/modules; Physical–system engineers/hardware; Scenarios–everyone/consistency.
13. **Views vs structures; Module/C&C/Allocation** — *Remember:* structure = elements + relations; view = representation for stakeholders; three categories with their sub-structures.
14. **Layered architecture & layer techniques** — *Remember:* Presentation (client/server caching, AJAX, responsive), Business (façade, session mgmt, workflow/rules engine, AOP), Data (pool, transactions, ORM, stored procedures, **parameterized SQL**), Service (duplicates, out-of-sequence, retry/queue).
15. **ADD** — *Remember:* drivers in → repeatedly: choose module → establish drivers → choose tactics → instantiate modules → verify; generate-and-test; iterate.
16. **ATAM** — *Remember:* participants (3–5 external team, decision-makers incl. willing architect, 12–15 stakeholders); steps 1–9; outputs: risks, non-risks, **sensitivity points** (one attribute), **trade-off points** (multiple, opposing), risk themes → business drivers; Lightweight = 4–6 h, internal, skips steps 7–8.
17. **Conformance & drift** — *Remember:* drift examples (skip layers, direct DB, no pub-sub, no common logging); prevention: architecturally evident coding, frameworks (Spring/Hibernate/JMS/Log4J/Drools), code templates, doc updates; educate, review, folder discipline.
18. **Architecture & testing** — *Remember:* utility tree → test priority; module interactions → integration test plan; testability design (data-source switch, rollback, component simulators).
19. **Reconstruction** — *Remember:* Raw view extraction → DB construction → **view fusion** (static + dynamic + expert) → arch analysis (rules) → iterate; tools Dali/ARMIN/Lattix/SonarJ/Structure101; purposes: undocumented systems, migration, reuse.
20. **Agile & architecture** — *Remember:* question is "how much, when," not "whether"; sweet spots 10/100/1000 KSLOC ≈ ~0/20/40%; drivers: rapid change→left, high assurance→right; **spikes** resolve trade-offs; refactor when debt grows; ICM.

## B. Quality Attribute Cheat Sheet

| Quality Attribute | What it means | Typical architectural concern | Example scenario |
|---|---|---|---|
| **Availability** | System ready to carry out its task when needed; minimizes outage by mitigating faults | Fault detection, redundancy, failover time, repair | Node crashes at peak; failover ≤ 30 s; 99.99% uptime |
| **Performance** | Meeting timing requirements when events arrive | Latency, throughput, jitter, miss rate; resource demand & scheduling | Checkout page < 1 s at 10× load; deadline 250 ms |
| **Security** | Protecting data/services from unauthorized access while serving the authorized | CIA; authentication/authorization/audit/encryption; attack surface | Fraudulent transfer blocked & audited ≤ 5 s |
| **Scalability** | Supporting growth (horizontal = scale out; vertical = scale up) | Statelessness, replication, load balancing | 10× seasonal spike absorbed by adding instances |
| **Modifiability** | Cost/risk of change: what can change, likelihood, when & who | Encapsulation, coupling, cohesion, defer binding, localization of change | Regulatory rule change ≤ 5 days, no core code |
| **Testability** | Probability a fault fails on the next test execution | Control & observe state; limit complexity; simulators | 85% path coverage in 3 h with stubbed payment gateway |
| **Interoperability** | Degree systems usefully exchange meaningful information | Discovery (locate), orchestration/tailoring, canonical data model | Unknown hospital system exchanges records, ≥ 99% correct |
| **Usability** | Ease of accomplishing tasks + support provided | User initiative (cancel/undo/pause/aggregate), system initiative (task/user/system models) | New user productive in 2 minutes; risky-transaction step-up |

## C. Important Frameworks / Methods (quick guide)

- **QAW** — *Purpose:* extract ASRs from stakeholders. *Steps:* intro → business mission → arch plan → identify drivers → brainstorm scenarios → consolidate → prioritize (30% votes). *When:* before architecture exists; requirements not yet distilled.
- **Architectural drivers** — *Purpose:* the distilled forces shaping design. *Content:* business goals/constraints, stakeholder concerns, QAs, regulatory, technology, cost, time-to-market. *When:* constant input to ADD, ATAM step 2/4.
- **Utility tree** — *Purpose:* capture, refine, prioritize QAs as scenarios. *Structure:* Utility → attribute → refinement → scenario (Importance, Difficulty). *When:* pre-design (ADD input) and ATAM step 5; also drives test-case priority.
- **ADD** — *Purpose:* derive architecture from drivers. *Steps:* pick module → drivers → tactics → instantiate modules → verify scenarios → repeat. *When:* designing greenfield or major subsystems; compatible with Agile.
- **4+1 View** — *Purpose:* document architecture for different stakeholders. *Views:* Logical, Process, Development, Physical, + Scenarios. *When:* any documented system; prioritize by dominant concern (high traffic → Process).
- **Layered Architecture** — *Purpose:* separation of concerns, loose coupling, reusable/exchangeable parts. *Layers:* Presentation / Business / Data (+ Service to external systems). *When:* enterprise transactional systems; enforce adjacent-only calls.
- **ATAM** — *Purpose:* expose risks/trade-offs against business drivers. *Steps:* 1 ATAM intro, 2 business drivers, 3 architecture, 4 approaches, 5 utility tree, 6 analyze, 7 brainstorm+prioritize, 8 analyze new, 9 results/risk themes. *When:* large, costly, risky systems; Lightweight variant (4–6 h) otherwise.
- **Architecture Conformance** — *Purpose:* keep code aligned to architecture. *Techniques:* architecturally evident coding, frameworks, code templates, doc updates; + education, reviews, folder discipline, automated dependency rules. *When:* continuously, from day one.
- **Architecture Testing** — *Purpose:* verify architectural properties and constraints. *Inputs:* utility tree (priority), dependency model (integration plan), scenarios (assertions). *When:* per increment and release.
- **Architecture Reconstruction** — *Purpose:* recover as-is architecture. *Phases:* raw view extraction → DB construction → view fusion → architecture analysis → iterate. *When:* no/stale docs, migration, reuse hunting, violation discovery.

## D. Common Exam Traps

1. **Confusing business goals with architectural drivers** — a goal like "cut costs" may yield *non-architectural* solutions; only goals producing measurable QA requirements drive architecture.
2. **Treating quality attributes as functional requirements** — "fast/secure/reliable" without a scenario is untestable and unusable for design.
3. **Naming a quality attribute without a measurable scenario** — always produce the six parts, especially the *response measure*.
4. **Ignoring trade-offs / declaring one architecture universally best** — there is no inherently good architecture; argue fit-for-purpose with business justification.
5. **Assuming the highest-performance option wins** — see Q44: constraints (power, certification, determinism) dominate.
6. **Confusing views with architectural patterns/styles** — a view is a stakeholder-facing representation of a structure; layered is a pattern/style.
7. **Mixing everything into one diagram** — documentation = relevant views + cross-view documentation, each view serving specific stakeholders.
8. **Treating documentation as one-time** — update at release; mark obsolete content; drift destroys trust in the whole document.
9. **Ignoring conformance after design** — code drifts (skip layers, direct DB, bypassed pub-sub); design the prevention mechanisms, not just the architecture.
10. **Assuming documented architecture = implemented architecture** — that assumption is exactly what reconstruction exists to test.
11. **Confusing reconstruction with redesign** — reconstruction recovers the *as-is* architecture; design decisions come later.
12. **Memorizing ATAM steps without application** — know *outputs* per step: risks, non-risks, sensitivity points (one attribute), trade-off points (multiple, opposing), risk themes mapped to business drivers.
13. **Mixing up sensitivity vs trade-off points** — heartbeat frequency = trade-off (availability ↑, performance ↓); DB client count = sensitivity (performance only).
14. **Misreading utility-tree notation** — (Importance, Difficulty); (H,L) still matters but gets cheap tactics; (H,H) gets design priority and top test priority.
15. **Forgetting the architect participates in ATAM and the evaluation team is external** — both are explicit, examinable participant facts.
16. **Skipping the "iterate" in ADD and reconstruction** — both are loops, not one-pass procedures.
17. **Placing concerns in the wrong layer** — connectivity semantics (duplicates, ordering, retry) belong in the service layer; business rules never in UI or stored procedures; façade goes in the business layer.
18. **Overstating caching** — always name the staleness/invalidation trade-off and mitigation.

---

**How to use this bank:** attempt each question under exam timing (a 4-mark part ≈ 8–10 minutes), then compare your answer structure against the model: *decision → scenario/driver identification → reasoning → trade-off → recommendation.* The examiners' pattern (from the sample paper) rewards exactly that chain, with a concrete example, over pure theory.