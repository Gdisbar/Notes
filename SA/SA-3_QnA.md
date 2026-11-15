# Software Architecture — 50 Scenario-Based Questions & Answers

---

## Section 1 — Medium Difficulty (Questions 1–10)

---

### Q1. [Availability — Banking Month-End Outage]

A core banking application becomes unavailable during month-end batch processing. The business demands 99.99% availability, but the operations team says the current monolithic batch design cannot be patched without a full shutdown. Identify the architectural problem and recommend an approach.

**Answer:**
The system couples a critical availability requirement with an architecture that has no fault-recovery structure — batch locks the whole system, so any fault becomes a total service failure (a fault has become a failure, which availability tactics must prevent). Since 99.99% allows only ~52 minutes of downtime/year, the architecture must support **fault detection + recovery without service interruption**.

**Architectural reasoning:** Apply availability tactics — *detect* faults (heartbeat, monitor), *recover* using redundancy (warm spare so a standby stays state-synchronized during batch), *prevent* faults (transactions, removal from service for maintenance). Architecturally, decouple the batch from online processing so the online path degrades gracefully rather than halting.

**Trade-off:** Redundancy raises infrastructure cost and complicates the coordination model (state resynchronization). The business goal (regulatory + customer SLA) justifies it; a 99.9% target would permit cheaper warm-spare-only designs.

**Key concepts tested:** availability tactics (detect/recover/prevent), fault vs failure, availability percentage as response measure.

---

### Q2. [Usability — Vague Requirement to Measurable Scenario]

A product owner for a tax-filing portal says: *"The system must be user-friendly so first-time users don't get frustrated."* Convert this into a proper quality attribute scenario and explain why the conversion matters.

**Answer:**
*"A first-time user (source of stimulus) attempts to complete an income-tax return (stimulus: learn/use the system efficiently) during normal operation. The system guides the user through the workflow, and 90% of first-time users complete the return without assistance within 30 minutes, with fewer than 3 errors requiring correction."*

**Why it matters:** "User-friendly" is untestable — the course identifies untestable attribute definitions as a key problem. Quality attributes must be expressed as **scenarios with six parts**: source of stimulus, stimulus, artifact, environment, response, response measure. Only then can the architect select tactics (e.g., maintain task model for contextual help) and testers verify them.

**Key concepts tested:** usability general scenario, scenario parts, measurability.

---

### Q3. [Modifiability — Insurance Rules Change]

An insurance company takes 6 weeks to change a premium-calculation rule because the rule is hard-coded in Java classes used by 14 modules. Regulatory changes arrive monthly. Which architectural technique addresses this, and in which layer?

**Answer:**
Use a **rules engine** in the **business layer**. The rules engine externalizes decision logic (premium rules as data/configuration rather than code), so a rule change becomes a rule-table update instead of modifying 14 coupled modules.

**Architectural reasoning:** This applies modifiability tactics — *reduce coupling* (use an intermediary; encapsulate), *defer binding* (rules bound at configuration time, not compile time). The scenario's response measure is the cost/effort of the change (e.g., "rule changed and deployed within 2 person-days with no code changes").

**Trade-off:** A rules engine adds infrastructure cost, a new technology competency, and indirection overhead at runtime. For monthly regulatory change, the trade-off is clearly favorable; for a static ruleset, it would be over-engineering.

**Key concepts tested:** business-layer techniques, modifiability tactics (intermediary, defer binding), response measure = cost of change.

---

### Q4. [Security — SQL Injection in E-Commerce Search]

Penetration testers show that entering `' OR 'x'='x` in the product-search box of an e-commerce site returns the entire product table. Explain the violation architecturally and the fix.

**Answer:**
This is a classic **SQL injection** attack — user input is concatenated directly into a SQL string, violating the integrity and confidentiality of data. The fix, taught in the data layer, is to use **parameterized SQL queries**: `WHERE product-name = ?`, where the user-entered value is supplied as a parameter, never as executable SQL.

**Architectural reasoning:** From the security general scenario — source: attacker (human, external); stimulus: unauthorized attempt to display/change data; response: attack resisted; response measure: number of attacks resisted / data compromised = none. Broader tactics: limit exposure, authenticate/authorize actors, and **audit** (log access attempts) for recovery and tracing.

**Trade-off:** Parameterized queries add minor coding discipline; ORM frameworks (e.g., Hibernate) make this systematic but introduce a technology dependency.

**Key concepts tested:** security tactics (resist, detect, recover/audit), data-layer technique, CIA characteristics.

---

### Q5. [ASR Identification — Logistics Platform]

A requirements document for a logistics platform states: (a) "System shall record shipment status," (b) "System shall detect fraud in real time and block the user," (c) "System shall support 50,000 concurrent tracking requests," (d) "The UI shall use the corporate blue color." Which are architecturally significant, and why?

**Answer:**
**(b) and (c) are ASRs; (a) and (d) are not.** ASRs are requirements with a *profound, measurable effect on the architecture* — their absence would produce a dramatically different architecture.

- (b) Fraud detection in real time drives the coordination model, data model, and possibly resource management — it cannot be satisfied by ordinary functional decomposition.
- (c) 50,000 concurrent requests drives scalability/performance tactics (caching, load distribution, resource management).
- (a) is a straightforward functional requirement satisfiable by standard responsibilities; (d) is a presentation detail with no architectural consequence.

**Indicators of significance** (from the slides): technical risk, significant business value/SLA, uniqueness — none met by (a)/(d).

**Key concepts tested:** ASR definition, significance indicators, functional vs quality-attribute requirements.

---

### Q6. [Layered Architecture — Facade for Shipping Request]

In a container-shipping system, placing one shipping request internally requires calling: locate nearest empty container, select best transporter, find a ship with spare capacity. The client/UI layer is becoming tightly coupled to all three modules. Which technique, in which layer, solves this?

**Answer:**
Use an **Application Façade in the business layer** providing one unified operation, e.g., `placeShippingRequest()`. The façade internally orchestrates the container, transporter, and ship modules, hiding them from clients.

**Architectural reasoning:** This is the *Facade* pattern — a single higher-level interface making a subsystem easier to use. It reduces coupling (clients depend on one interface, not three modules), encapsulates the orchestration order, and localizes change: if a fourth module (customs clearance) must be invoked, only the façade changes.

**Trade-off:** Adds an intermediary (minor latency/development cost) — a classic modifiability-vs-performance consideration, acceptable here.

**Key concepts tested:** business-layer techniques, façade, coupling reduction, separation of concerns.

---

### Q7. [Performance — Slow Dropdown Screen]

A citizen-registration screen has cascading "State → District → Town" dropdowns. Loading all three lists on page load takes 12 seconds and the page is used thousands of times daily. Apply the presentation/data-layer techniques from the course and justify.

**Answer:**
Two techniques from the lecture: **(1) server-side caching** of states/districts/towns (the reference data is static and shared across users — cache in memcached-style store to relieve the database); **(2) AJAX asynchronous loading** — load only states initially, fetch districts and towns on demand as the user selects, updating part of the page without a full reload.

**Architectural reasoning:** These apply performance tactics — *manage resources* (cache multiple copies of data on faster storage) and *control resource demand* (limit event response; load data only when actually requested). Response measure: screen becomes usable in under ~2 seconds.

**Trade-off:** Caching introduces staleness risk (acceptable for slowly changing reference data) and AJAX adds client-side complexity.

**Key concepts tested:** client-side vs server-side caching, AJAX, memcached, performance tactics.

---

### Q8. [Structures — Classifying a System's Structures]

An architect documents: (i) a class diagram of modules; (ii) a diagram of runtime processes and their message queues; (iii) a mapping of processes to servers; (iv) a mapping of modules to development teams. Classify each using the three structure categories from the course.

**Answer:**

| Document | Structure category |
|---|---|
| (i) Class diagram | **Module structure** (code units, functional responsibility, inheritance) |
| (ii) Processes + message queues | **Component-and-connector structure** (runtime elements and their communication vehicles) |
| (iii) Processes → servers | **Allocation structure** (deployment: software → hardware) |
| (iv) Modules → teams | **Allocation structure** (work assignment: software → organizational environment) |

Each structure answers different stakeholder questions: modules — "what is each unit responsible for?"; C&C — "how do executing elements interact?"; allocation — "where does software run / who builds it?". No single structure *is* the architecture; architecture is the **set of structures** needed to reason about the system.

**Key concepts tested:** module / C&C / allocation structures, structure vs view.

---

### Q9. [Conformance — Layer Bypass]

Code review of an enterprise app finds JDBC calls embedded directly in JSP presentation pages "because it was faster to code." Now every database schema change breaks multiple UI components. Identify the architectural violation and the correct placement.

**Answer:**
This is **architectural drift** — code deviating from the prescribed layered architecture. Specific violations: (1) **layer bypass** — the presentation layer accesses the data layer directly, skipping the business layer; (2) responsibility misplacement — SQL/data access belongs in the **data access layer**, business rules in the **business layer**, rendering in the **presentation layer**.

**Why it hurts:** it destroys separation of concerns and loose coupling — exactly the benefits layering provides — so changes propagate non-locally (a data change forces UI changes), and the data layer is no longer exchangeable.

**Corrective actions:** move queries into a DAL (via DAOs/ORM); restore the discipline that a layer uses only adjacent lower layers; enforce via framework (MVC), code reviews, and folder discipline.

**Key concepts tested:** examples of drift, layer responsibilities, consequences of bypassing.

---

### Q10. [Good Architecture — Rules of Thumb]

A startup's system has no single accountable architect, modules with mixed unrelated responsibilities, and deep dependencies on one specific commercial database version. Using the course's structural rules of thumb, identify three problems and fixes.

**Answer:**
Three violations of the "good architecture" rules of thumb:

1. **No identified technical leader** → architecture lacks conceptual integrity. Fix: product of a single architect/small group with an identified leader, with strong connection to the development team.
2. **Modules violate information hiding/separation of concerns** (unrelated responsibilities in one module) → changes ripple widely. Fix: assign responsibilities so likely changes fall within one module.
3. **Dependence on a specific product version** → fix: the architecture should never depend on a particular version of a commercial product; if unavoidable, structure so switching versions is straightforward and inexpensive.

Underlying principle: *there is no inherently good or bad architecture — only fitness for purpose* — but these heuristics correlate strongly with maintainable systems.

**Key concepts tested:** good-architecture rules of thumb, conceptual integrity, information hiding.

       +-------------------------------------------------+

       |           1. EDGE / EDGE OVERLAY LAYER          |
       |  (DNS, CDN, DDoS Protection - e.g., Cloudflare) |
       +------------------------+------------------------+
                                |
                                v
       +-------------------------------------------------+

       |          2. INTEGRATION & ROUTING LAYER         |
       |  (API Gateways, Load Balancers, Reverse Proxies)|
       +------------------------+------------------------+
                                |
                                v
       +-------------------------------------------------+

       |              3. PRESENTATION LAYER              |
       |  (JSP, React/Angular, Mobile Apps, Controllers) |
       +------------------------+------------------------+
                                |
          ======================+======================
         || CROSS-CUTTING CONCERNS LAYER             ||
         || (Security/OAuth2, Logging, Observability)||
          ======================+======================
                                |
                                v
       +-------------------------------------------------+

       |                 4. BUSINESS LAYER               |
       |  (Core Services, Domain Logic, Workflows)       |
       +------------------------+------------------------+
                                |
                                v
       +-------------------------------------------------+

       |            5. ASYNCHRONOUS MESSAGING            |
       |  (Message Brokers/Queues - Kafka, RabbitMQ)     |
       +------------------------+------------------------+
                                |
                                v
       +-------------------------------------------------+

       |         6. CACHING & DATA ACCELERATION          |
       |  (Distributed In-Memory Cache - Redis)          |
       +------------------------+------------------------+
                                |
                                v
       +-------------------------------------------------+

       |               7. DATA ACCESS LAYER              |
       |  (DAOs, Repositories, ORM/Hibernate)            |
       +------------------------+------------------------+
                                |
                                v
       +-------------------------------------------------+

       |                8. DATABASE LAYER                |
       +-------------------------------------------------+

```python

from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Generator
import abc

# =====================================================================
# 1. DATABASE LAYER (Simulation)
# =====================================================================
# Simulating a live production database schema
DATABASE_MOCK = {
    1: {"id": 1, "username": "alice", "email": "alice@example.com", "is_active": True},
    2: {"id": 2, "username": "bob", "email": "bob@example.com", "is_active": False}
}

# =====================================================================
# PRESENTATION / DATA TRANSFER OBJECTS (DTOs)
# =====================================================================
# Pydantic schemas define exactly what the presentation layer exposes.
# This prevents database implementation leaks to the client UI.
class UserResponse(BaseModel):
    id: int
    username: str
    email: str


# =====================================================================
# 2. DATA ACCESS LAYER (Repository Pattern)
# =====================================================================
class UserRepository:
    """Handles direct interactions with the storage engine."""
    def __init__(self):
        # In production, you would inject a SQLAlchemy or Tortoise session here
        self.db = DATABASE_MOCK

    def get_by_id(self, user_id: int) -> dict | None:
        # SQL-equivalent fetch. If schema fields change, ONLY this method updates.
        return self.db.get(user_id)


# Dependency provider for FastAPI injection
def get_user_repository() -> UserRepository:
    return UserRepository()


# =====================================================================
# 3. BUSINESS LAYER (Service Layer)
# =====================================================================
class UserService:
    """Enforces enterprise business rules and application logic."""
    def __init__(self, user_repo: UserRepository = Depends(get_user_repository)):
        self.user_repo = user_repo

    def get_active_user(self, user_id: int) -> dict:
        # Business Rule 1: Fetch the user data
        user = self.user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="User account does not exist"
            )
        
        # Business Rule 2: Evaluate if account is suspended/inactive
        if not user["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="This user account is inactive"
            )
            
        return user


# =====================================================================
# 4. PRESENTATION LAYER (FastAPI Routing / Controllers)
# =====================================================================
app = FastAPI(title="Enterprise Layered Architecture Example")

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user_profile(user_id: int, user_service: UserService = Depends()):
    """
    Presentation endpoint. It has ZERO knowledge of SQL, data storage, 
    or internal validation rules. It delegates entirely to the Service layer.
    """
    # Simply calls the brain (Business Layer) and returns the validated DTO mapping
    return user_service.get_active_user(user_id)

```
---

## Section 2 — Medium-High Difficulty (Questions 11–35)

---

### Q11. [Trade-off — Security vs Performance in API Design]

A fintech API team can add per-request encryption, mutual TLS, and full payload audit logging to satisfy the security team, but load tests show this adds 300ms latency per call, breaching the 200ms performance budget. Analyze the trade-off and recommend.

**Answer:**

**Competing attributes:** Security (confidentiality, integrity, nonrepudiation) vs Performance (latency).

**Analysis:** Security tactics — *resist attacks* (encrypt data, authenticate actors), *react/recover* (audit). These consume CPU cycles per request (encryption cost, logging I/O, handshake overhead), directly degrading the response measure (latency). The course's principle applies: *increasing security often decreases performance — you cannot have everything.*

**Resolution strategies (not "drop security"):**
- Re-examine the scenario: is 200ms a hard deadline for *all* calls, or only high-frequency ones? Prioritize events (performance tactic).
- Move costlier checks to session establishment (authenticate once per session, not per request).
- Use connection pooling/keep-alive to amortize TLS handshakes; audit asynchronously so logging I/O is off the critical path.

**Recommendation:** retain all three security controls, but restructure *where* they execute so latency is managed — the trade-off is shifted, not eliminated.

**Key concepts tested:** quality-attribute conflict, trade-off analysis, security and performance tactics.

---

### Q12. [Utility Tree — Hospital System]

For a hospital management system, stakeholders care about: fast patient-account updates, quick proficiency for new staff, configurable service fees, and no downtime during DB upgrades. Construct a utility tree fragment with at least four leaf scenarios and priorities.

**Answer:**

```
UTILITY
├── Performance
│   ├── Transaction response time
│   │   └── "User updates patient account during peak load; completes < 0.75 s" (H,M)
│   └── Throughput
│       └── "At peak load, 150 normalized transactions/second" (M,M)
├── Usability
│   └── Proficiency
│       └── "New hire with 2+ yrs experience becomes proficient in core functions < 1 week" (M,L)
├── Modifiability
│   └── Configurability
│       └── "Hospital increases fee for a service; config team changes it in 1 working day,
│            no source-code change" (H,L)
└── Availability
    └── No downtime
        └── "DB vendor releases new software; hot-swapped with no downtime" (H,L)
```

**Reasoning:** The tree refines each quality attribute into concrete, testable scenarios, each prioritized on **(business value, architectural impact)** — H/M/L. High-priority, high-impact scenarios drive design decisions and later test-case prioritization; the notation turns vague "we need availability" into actionable, measurable requirements.

**Key concepts tested:** utility tree structure (root → attributes → refinements → scenarios), dual-priority notation.

---

### Q13. [QAW — Applying the Workshop]

A company building a new toll-collection system has conflicting stakeholder views on what matters most. Describe how you would run a Quality Attribute Workshop to produce a prioritized set of architectural drivers.

**Answer:**
Run the six QAW steps:

1. **Presentation & introduction** — participants state roles and relationship to the system.
2. **Business mission presentation** — a business representative presents business context, functional requirements, constraints (~1 hour).
3. **Architectural plan presentation** — the architect presents current architectural thinking.
4. **Identification of architectural drivers** — facilitators share the distilled driver list (requirements, business drivers, constraints, QAs); stakeholders clarify, add, delete until consensus.
5. **Scenario brainstorming** — each stakeholder writes scenarios with explicit stimulus and response; ensure at least one scenario per driver.
6. **Scenario consolidation** — merge similar scenarios (with proposers' agreement).

Then **prioritize**: each stakeholder gets votes equal to **30% of the consolidated scenario count**, producing the ranked list of drivers.

**Expectation:** a consensus list of architectural drivers feeding the utility tree and design. The discipline (explicit stimulus/response, voting) prevents dominant-voice capture.

**Key concepts tested:** QAW steps, 30% voting rule, drivers vs scenarios.

---

### Q14. [PALM — From Business Goals to Quality Attributes]

The sponsor of a payments app states the business goal: *"Grow cashless transaction share from 8% to 20% in 18 months."* Using PALM, show how pedigree is established between this goal and architectural drivers.

**Answer:**

**PALM steps:** (1) PALM overview; (2) business drivers presentation; (3) architecture drivers presentation; (4) business-goal elicitation using standard categories ("X to Y by when" format); (5) identification of potential quality attributes from each goal; (6) **assignment of pedigree** — trace each architectural driver back to the goal it serves; (7) conclusion.

**Application:**
- Goal (elicit & confirm in X→Y format): cashless share 8%→20% in 18 months.
- Derived quality attributes (as in the course's BHIM example): **Usability** — transaction in minimal steps (a new user completes a payment in under 3 steps / 30 seconds); **Security** — strong authentication (the BHIM example cites 3-factor security) to build trust.
- **Pedigree assignment:** ask *"Why must onboarding take under 30 seconds? Why not 2 minutes?"* — the answer (mass adoption requires near-zero friction against cash) establishes that the performance/usability driver genuinely serves the business goal, not arbitrary preference.

**Key concepts tested:** PALM steps, pedigree, business-goal format, goals→QA derivation.

---

### Q15. [ADD — Attribute-Driven Design Application]

For the toll system of Q13, assume drivers are: (1) process 10,000 vehicle crossings/hour at peak, (2) continue operation if the central server fails, (3) add a new payment method within 2 weeks. Outline how ADD produces the architecture.

**Answer:**

ADD is iterative decomposition guided by quality-attribute requirements:

1. **Choose the module to decompose** — start from the system as one element (or the highest-level element implied by drivers).
2. **Refine the module using tactics for the drivers:**
   - Driver 1 (throughput) → performance tactics: multiple copies of computations (parallel lane processors), resource management, queue management.
   - Driver 2 (availability) → redundancy: local lane units store-and-forward transactions when the center is unreachable; retry/reconciliation when restored.
   - Driver 3 (modifiability) → payment handling behind an abstract interface / intermediary in the service layer; defer binding of payment-method selection.
3. **Define interfaces** — e.g., `processCrossing(vehicle, paymentInfo)`, with contracts for queuing and failover behavior.
4. **Verify** the resulting structures against the use cases and QA scenarios; check each driver is addressed by some structure/behavior.
5. **Repeat** for each newly created module until all requirements are satisfied.

**Key point:** ADD proceeds **from drivers → scenarios → tactics → structures**, not from functionality alone (functionality does not determine architecture).

**Key concepts tested:** ADD steps, tactics-driven decomposition, drivers to structures.

---

### Q16. [4+1 Views — Stakeholder Mapping]

For a new airline Departure Control System, map stakeholders to Kruchten's views and justify each mapping.

**Answer:**

| View | Primary stakeholder | What they need | QA concern |
|---|---|---|---|
| **Logical** | End-user / analysts | Key abstractions: flight, passenger, seat, boarding pass — what services the system provides | Functional requirements |
| **Process** | Integrators | Concurrency & distribution: check-in processes, load-balancing across terminals, synchronization | Scalability, concurrency, performance (non-functional) |
| **Development** | Programmers, software managers | Static organization: layers, subsystems, module libraries, reuse | Buildability, modularity, tool constraints |
| **Physical** | System engineers | Mapping to hardware: check-in kiosks, gate servers, redundant central servers | Reliability, availability, performance on underlying hardware |
| **Scenarios (+1)** | All users & evaluators | A small set of end-to-end scenarios (e.g., "check in a passenger with baggage") traced through all four views | Validity & consistency of the whole |

**Reasoning:** each stakeholder has distinct concerns; no single diagram serves all. The +1 scenario view *drives* the others and validates consistency (inside-out from logical, or outside-in from physical).

**Key concepts tested:** 4+1 view purposes, viewer→concern mapping.

---

### Q17. [Documentation — The Single-Diagram Problem]

A large enterprise system documents its entire architecture in one complex box-and-line diagram. New developers interpret it differently; the security team can't verify controls; operations can't plan deployment. What's wrong, and how should documentation be fixed?

**Answer:**

**What's wrong:** (1) A single view cannot address all stakeholders' concerns — architecture documents must document the *relevant views*; (2) box-and-line drawings without documented element behavior are not architectures — readers imagine functionality that may be wrong; (3) security/deployment concerns need allocation views, which are absent.

**Fix — build a documentation package:**
- **Module views** (decomposition/uses/layered): for developers — responsibilities, dependencies.
- **C&C views** (client-server/process): runtime behavior, communication — for understanding and integration planning.
- **Allocation views** (deployment/implementation): for system engineers and operations — processors, network, file structure.
- **Behavior documentation** for the parts of element behavior that affect other elements or system acceptability.
- Cross-view documentation tying views together; follow an architecture documentation template so each view is complete and consistent.

**Key concepts tested:** views & stakeholders, structure vs view, box-and-line insufficiency, documentation package.

---

### Q18. [Service Layer — Idempotency & Failures]

A bank exposes `getBalance` and `requestChequebook` services to external aggregators. It observes: duplicate chequebook requests after client retries, out-of-sequence balance enquiries, and lost requests during network outages. Which layer handles these, and how?

**Answer:**
These are **service-layer responsibilities** (the layer providing services to external systems). The situations, from the lecture:

1. **Duplicate requests** → make operations **idempotent**: detect a repeated request (e.g., by request ID) and return the original result instead of re-executing.
2. **Out-of-sequence messages** → sequence numbering and reordering/buffering logic in the service layer.
3. **Communication failure** → **retry mechanism**, or **queue the work and send once communication is restored** (store-and-forward).

**Architectural reasoning:** these concerns belong at the boundary with external systems because the architect cannot control external clients' behavior. Placing them in the service layer keeps business logic clean (single responsibility) and gives one controlled point for external-interface policy.

**Trade-off:** added complexity at the boundary — but correctness of external interaction demands it.

**Key concepts tested:** service layer role, reliability of external interactions.

---

### Q19. [Session Management — Stateless vs Stateful Beans]

An e-commerce site runs on a web farm (multiple server instances). The team debates stateful session beans ("keeps cart state in the bean") vs stateless beans with client-side state. Analyze architecturally.

**Answer:**

| Aspect | Stateful session bean | Stateless + client-side state |
|---|---|---|
| Request routing | All requests from a client → same bean instance | Any instance can serve; client sends state (cookies) |
| Scalability | Limited — instance pinned per client; memory grows with users | High — any server handles any request |
| Failure behavior | Instance loss = session loss | State rebuilt from client data |
| Coordination | None needed | Client maintains state; server stays simple |

**Architectural reasoning:** with a web farm, stateless beans + client-based state management (cookies) support **horizontal scalability** and load balancing — requests can be distributed freely. Stateful beans contradict the load-balancer model: subsequent requests *must* reach the same instance (as shown in the course's session-management slide).

**Recommendation:** stateless session beans with client-based state; if server-side session state is mandatory, externalize it to shared persistent storage rather than in-bean memory.

**Key concepts tested:** session management techniques, scalability implications of state placement.

---

### Q20. [ATAM — Risk, Sensitivity, Trade-off Identification]

During an ATAM of a telecom billing system, the evaluator finds: (i) a single shared database connection pool used by both online queries and batch rating; (ii) peak load causes pool exhaustion, delaying online queries; (iii) the architecture uses strong per-message encryption, adding 40% CPU load. Classify these findings as risk, sensitivity point, or trade-off point.

**Answer:**

| Finding | Classification | Why |
|---|---|---|
| (i) Shared pool for competing workloads | **Risk** — a potentially problematic architectural decision; contention between batch and online traffic threatens the response-time scenario |
| (ii) Pool exhaustion under peak load | **Sensitivity point** — a property critical to achieving the performance QA; small changes here (pool size, workload separation) significantly affect the response measure |
| (iii) Encryption ↔ CPU load | **Trade-off point** — a decision affecting multiple QAs in opposition: security (confidentiality, via encrypt-data tactic) vs performance (latency/throughput) |

**Architectural reasoning:** ATAM's purpose is risk identification and trade-off analysis — "fail early." Risks get documented with rationale; sensitivities tell the architect where design levers live; trade-offs must be *made explicit to stakeholders* rather than hidden.

**Recommendation:** separate pools / partition resources (fix risk i); document iii as an accepted, stakeholder-visible trade-off with monitoring.

**Key concepts tested:** ATAM outputs — risks, nonrisks, sensitivity points, tradeoff points.

---

### Q21. [ATAM — Team, Phases, Steps]

Your organization is preparing its first ATAM for a 40-person, multi-team payment platform. Describe the evaluation team roles, the phases with durations, and where risks are first formally produced.

**Answer:**

**Team roles:** **Team leader** (logistics, contract, team formation); **Evaluation leader** (facilitates scenario elicitation, prioritization, analysis); **Scenario scribe** (captures exact scenario wording); **Proceedings scribe** (records scenarios, motivating issues, resolutions; prints adopted scenarios); **Questioner** (raises quality-attribute-specific architectural issues).

**Phases:** Phase 0 — partnership & preparation (weeks, informal); Phase 1 — steps 1–6 with evaluation team + project decision-makers (1–2 days, then 2–3 week hiatus); Phase 2 — steps 7–9 with stakeholders (2 days); Phase 3 — follow-up report (~1 week).

**Where risks emerge:** during **Step 6 — Analyze Architectural Approaches**, where the highest-ranked utility-tree scenarios are mapped onto the architecture and the team probes for risks, sensitivities, and trade-offs (refined in Steps 7–8 with the broader scenario set).

**Key concepts tested:** ATAM roles, 4 phases, 9 steps, outputs.

---

### Q22. [Lightweight ATAM — Agile Team Evaluation]

A 12-person Agile team wants ATAM benefits without a 3-day workshop. Can ATAM be tailored? Describe the lightweight agenda and what is sacrificed.

**Answer:**
Yes — the course describes a **lightweight ATAM (4–6 hours)**: participants already know the process (Step 1 → 0 h); business drivers and architecture get brief refreshers (0.25 h / 0.5 h, tracing 1–2 scenarios through module and C&C views); architectural approaches identified within Step 3 (0.25 h); utility tree built from existing scenarios (0.5–1.5 h); **Step 6 consumes the bulk (2–3 h)** mapping high-ranked scenarios to the architecture; Steps 7–8 are **omitted** (internal stakeholders contribute scenarios directly at Step 5); results presented (0.5 h), reviewing risks, nonrisks, sensitivities, trade-offs.

**Sacrificed:** broad stakeholder scenario elicitation and the second round of analysis — the evaluation reflects the *team's own* view of concerns rather than a wide stakeholder base, so risks rooted in external stakeholder assumptions may be missed.

**Key concepts tested:** lightweight ATAM agenda, tailoring, cost/benefit of stakeholder breadth.

---

### Q23. [Architecture & Testing — Test Prioritization]

QA for a healthcare portal has time to run only 40% of its planned test cases before release. How should the architecture work products guide which tests to run first?

**Answer:**
Use the **utility tree**. It prioritizes scenarios by **business value and architectural impact**; the high–high scenarios (e.g., "update patient account at peak load < 0.75 s (H,M)"; "intrusion detected and reported within 90 s (H,M)") translate directly into **high-priority test cases**. Tests covering scenarios that are architecturally sensitive (where the architecture's riskiest structures live) get precedence because a failure there signals a systemic architectural weakness, not a localized defect.

**Additional architecture–testing links (from the course):**
- Architecture informs the **integration test plan**: module views reveal which modules interact and depend on which, defining integration sequences and needed stubs/drivers.
- Architecture must be **designed for testability**: switchable data sources, rollback of test changes, replaceable components (simulators for payment gateways, external systems).

**Key concepts tested:** utility tree → test prioritization, integration planning from module structure, testability requirements.

---

### Q24. [Integration Testing — From Module Views]

Two teams developing a warehouse system cannot agree on the order of integration testing; some modules have never been tested together though production behavior depends on their interaction. Explain how architecture documentation resolves this.

**Answer:**
The **module structures** — decomposition and especially the **uses structure** ("correctness of unit A requires a correct version of unit B") — define the dependency graph. From it:

1. **Identify interacting modules** and their dependencies → candidate integration test cases.
2. **Sequence tests** along the uses relation: integrate and test a module with the modules it uses before testing modules that use it — enabling **incremental integration** and early construction of working subsets (the uses structure exists precisely to engineer subsets and support incremental development).
3. Build the integration plan from this structure, matching it to the development WBS and team structure.

**Why architecture helps:** without the documented uses relation, teams guess at integration order; with it, test planning is derived, systematic, and traceable — architecture is the key artifact for reasoning about integration.

**Key concepts tested:** uses structure, incremental delivery, architecture → integration test plan.

---

### Q25. [Reconstruction — Purpose & Phases]

An insurance firm acquired a 15-year-old claims system with no documentation and wants to (a) understand it, (b) identify reusable components like logging and security, and (c) plan migration off the mainframe. Show how architecture reconstruction serves all three purposes and describe its phases.

**Answer:**

**Purposes (from the course):** understand architecture where no documentation exists; support **technology migration** (mainframe → web); **identify reusable components** (logging, security). Reconstruction addresses all three: it produces an as-built architecture to reason about.

**Phases:**
1. **Raw view extraction** — pull information from **source code, execution traces, build scripts**; yields classes, files used, caller–callee relations, global data accessed.
2. **DB construction** — put extracted data into a common format.
3. **View fusion** — combine views: static analysis gives the compile-time view; execution traces reveal runtime/dynamically-bound interactions; an expert then groups elements (e.g., into layers).
4. **Architecture analysis** — validate the reconstructed elements and check **conformance**: e.g., "a layer may call adjacent layers only," "all DB access via entity beans."
5. **Iterate** — refine extraction/grouping.

Tools: Dali, ARMIN, Lattix, SonarJ/Structure 101, SonarQube.

**Key concepts tested:** purposes, five phases, raw view extraction, view fusion, violation checking.

---

### Q26. [Conformance Techniques — Preventing Drift]

A 60-developer organization keeps finding that implemented code diverges from the documented architecture. Describe the four techniques from the course to keep code and architecture consistent, with examples.

**Answer:**

1. **Architecturally evident coding style** — make architecture visible in code: indicate which **layer** a class belongs to; mark components as **publisher/subscriber** in a pub-sub design; mark whether a component **inserts or retrieves** from a message queue.
2. **Use frameworks** — frameworks enforce the architectural structure: **Spring MVC** forces code into Model/View/Controller; **Hibernate** enforces ORM data access; **AUTOSAR** standardizes automotive architectures; workflow/rules engines and JMS pub-sub similarly constrain structure.
3. **Code templates** — provide structural skeletons developers must fill; the fault-tolerance template in the lecture (event loop with primary/backup state exchange) embeds the redundancy design directly.
4. **Update architecture documentation** — accompany code changes with doc changes; at minimum mark obsolete parts "no longer applicable" (preserving trust in the rest); synchronize docs at release time.

**Supporting practices (class exercise):** code reviews, educating new members, folder-per-architectural-aspect discipline.

**Key concepts tested:** the four conformance techniques, framework enforcement, template use.

---

### Q27. [Agile Architecture — How Much Up-Front?]

Compare architecture effort for: (A) a 10 KSLOC marketing microsite, (B) a 100 KSLOC retail platform, (C) a 1,000 KSLOC distributed banking system — all "Agile." Use the Boehm–Turner analysis and the authors' advice.

**Answer:**

The sweet-spot analysis: as up-front architecture/risk work increases, rework decreases — but total effort has a minimum per project size:
- **10 KSLOC (A):** sweet spot at the far left — minimal up-front work; heavy architecture is wasted. Just agree on major patterns, start coding.
- **100 KSLOC (B):** sweet spot ≈ **20% of schedule** — meaningful up-front design, utility tree, candidate architecture, lightweight evaluation.
- **1,000 KSLOC (C):** sweet spot ≈ **40%** — large, distributed, complex systems cannot be organized by Agile principles alone; a million-line system needs an architecture to guide and organize effort.

**Authors' advice:** stable requirements + distributed development → large up-front architecture pays off; unstable requirements → design a *candidate* architecture quickly and elaborate via spikes/experiments; small uncertain projects → only major patterns. Agile architects take the middle ground: propose an initial architecture, run with it, refactor when technical debt grows too great.

**Key concepts tested:** sweet-spot chart, sizing architecture effort, candidate architecture in Agile.

---

### Q28. [Availability Tactics — Choosing Redundancy]

A stock-trading gateway requires failover within 1 second; a reporting batch system can tolerate 15 minutes of recovery; a cost-sensitive IoT telemetry collector can tolerate hours. Recommend hot/warm/cold redundancy for each and justify.

**Answer:**

| System | Redundancy | Justification |
|---|---|---|
| Trading gateway | **Active redundancy (hot spare)** — all nodes process identical inputs in parallel, spares stay state-synchronized | 1-second failover leaves no time for state transfer; seamless continuation |
| Reporting batch | **Passive redundancy (warm spare)** — active node processes; spare receives periodic state updates | Minutes-level failover affordable; warm spare cheaper than full hot redundancy |
| IoT collector | **Spare (cold)** — redundant node powered on after failure (power-on-reset) | Hours of downtime tolerable; cold spares minimize cost/complexity |

**Architectural reasoning:** the choice maps *recovery-time response measure* to *state-synchronization cost*. Hot spares spend resources continuously for instant failover; cold spares spend nothing until failure but pay recovery latency. Protection groups + state resynchronization are the underlying mechanisms; the decision is a **cost vs availability trade-off driven by the business scenario**.

**Key concepts tested:** recover-from-fault tactics, protection groups, matching tactics to response measures.

---

### Q29. [Performance Tactics — Caching Trade-offs]

An e-commerce product catalog page reads the same 50,000 product records from the DB on every request. Propose performance tactics and analyze the caching trade-offs (client-side vs server-side).

**Answer:**

**Tactics:** *manage resources* — **maintain multiple copies of data**: cache the catalog on faster storage. Options from the lecture:
- **Server-side caching** (e.g., memcached — a distributed in-memory store shared by back-end modules): product codes, descriptions, promotions; relieves DB load for all users; staleness window must be defined for price/promo changes.
- **Client-side caching** (browser): user preferences, payment method, delivery address; personalized, no server benefit for shared data; must handle privacy and cache invalidation.
- **Combination** is typical practice.

**Supporting tactics:** asynchronous (AJAX) updates for parts of the page; responsive design is a presentation concern, not a data one.

**Trade-offs:** staleness vs freshness; memory cost; invalidation complexity on catalog updates. Modifiability note: cacheable reference data should be *separated* from frequently changing transactional data in the data model so cache policy changes don't ripple.

**Key concepts tested:** control-resource-demand vs manage-resources tactics, client vs server caching, memcached.

---

### Q30. [Interoperability — Emergency Services Integration]

A city's new emergency platform must exchange incident data with a national police system whose interface details are known, and with unknown third-party hospital systems discovered at runtime. Which interoperability tactics apply to each case?

**Answer:**

**Goal of interoperability tactics:** the systems must *locate* each other and *manage interfaces* so information is exchanged semantically.

- **National police system (known at design time):** **manage interfaces** — **tailor interface** (add translation/buffering/data-smoothing so incident data maps to police data abstractions), **orchestrate** if the exchange involves a sequence of service invocations in a complex task; define the coordination model (protocols, message currency/timeliness).
- **Unknown hospital systems (runtime discovery):** **locate** — **discover service** via a known directory service (possibly multi-level indirection); binding-time decisions must support late binding and **reject-and-log** unacceptable bindings.

**Design checklist considerations:** semantic consistency of the exchanged data abstractions; resource management so interoperation requests can't exhaust resources (flood → denial of service to legitimate users); logging for non-repudiation in untrusted exchanges.

**Key concepts tested:** locate vs manage-interface tactics, binding time, design checklist.

---

### Q31. [Testability — Architecture Support]

A QA team complains it cannot test the billing module: it always reads the production payment gateway, test transactions can't be rolled back, and faults can't be reproduced because internal state is scattered. Map these to testability tactics.

**Answer:**

Testability = ease of making software demonstrate its faults; requires **controllability and observability**, plus **limited complexity**:

1. **Cannot substitute the gateway** → use **abstract data sources**: abstract the payment-gateway interface so a simulator/stub can be substituted (also enables test-data vs production-data switching).
2. **No rollback of test changes** → provide **record/playback** and rollback support: capture inputs crossing the interface for reuse; design state updates as **transactions** so test effects can be undone and the system restored.
3. **State scattered, faults irreproducible** → **localize state storage**: keep module state in one place so the system can be started in an arbitrary state for a test; add **specialized interfaces** to set/observe state; capture **late bindings** at failure so state can be recreated.

**Design principle:** these are *architectural* responsibilities — the architect must allocate responsibilities for executing test suites, capturing fault-inducing activity, and observing state; otherwise testing cost dominates and risk exposure (loss × probability) stays high.

**Key concepts tested:** testability tactics (control/observe state; limit complexity), sandbox, executable assertions.

---

### Q32. [Business Context Trade-off — Availability vs Cost]

*(In the style of your sample paper's Q4.)* A retail distribution network wants product availability near 100% at every warehouse (no stockouts) but inventory carrying costs are crushing margins. Analyze this trade-off and connect it to architectural decision-making.

**Answer:**

**Competing objectives:** product availability (business goal) vs inventory cost (business constraint). Pushing availability to 100% requires inventory at every node for every SKU — cost explodes due to working capital, storage, and obsolescence. Minimizing cost means lean inventory — stockout risk rises, sales are lost.

**Balancing:** segment SKUs (high-velocity items buffered at regional nodes; slow movers centralized); use demand forecasting to set differentiated service levels; accept, e.g., 98.5% availability on C-class items while holding A-class items at 99.9%.

**Connection to architecture:** this *is* architecture reasoning. The system architecture must support: **monitorability** (real-time stock visibility — the ops staff monitor the system while it executes), **interoperability** (warehouse ↔ store ↔ supplier information exchange), **modifiability** (rules for reorder points configurable per SKU, e.g., rules engine), and **scalability** (handle demand spikes). The architect translates the business trade-off into these quality attribute scenarios and tactics — business goals drive quality attributes, which drive the architecture.

**Key concepts tested:** business goals → QAs → architecture; monitorability; business-context reasoning.

---

### Q33. [Trade-off — Security vs Usability]

A hospital EHR system must be HIPAA-compliant (strict access control, audit trails, session timeouts) yet emergency physicians complain that authentication steps delay critical care access. Analyze the conflict and propose an architectural resolution.

**Answer:**

**Competing attributes:** Security (confidentiality, integrity, accountability) vs Usability (efficiency, minimizing impact of errors, satisfaction) — one of the classic conflicts in the syllabus.

**Analysis:** Security tactics — authenticate/authorize actors, limit access, encrypt data, audit — all add steps or latency to the user workflow. Usability scenarios demand task time minimization and low error impact. Optimizing one view (security) demonstrably compromises the other — the sample paper tests exactly this kind of view intersection.

**Architectural resolutions:**
- **Role-based authorization** tailored to clinical roles: emergency physicians get rapid access to emergency-relevant record subsets (limit access *by data sensitivity*, not blanket barriers).
- **Contextual authentication**: strong authentication at session start; lighter re-verification within an active, monitored clinical session (avoids repeated friction without eliminating accountability).
- **Audit** all accesses (security recovery tactic) so loosening interactive barriers doesn't lose accountability.
- **Degraded-mode / break-glass access**: documented emergency access with mandatory post-hoc review.

**Recommendation:** don't uniformly relax security; restructure *where* controls execute — session-level strong auth + continuous audit + role-scoped access resolves most friction while preserving HIPAA accountability.

**Key concepts tested:** security view vs usability, tactic placement, audit, trade-off resolution.

---

### Q34. [Layered Architecture — Degradation and Recovery]

Over five years, an enterprise app accumulated: business rules in stored procedures, presentation code calling the workflow engine directly, and the data layer importing classes from the presentation layer (circular dependency). Explain the consequences and the recovery strategy.

**Answer:**

**Consequences:** (1) **Responsibility erosion** — rules in stored procedures split business logic across business and data layers, so a business change requires touching multiple layers; (2) **layer bypass** — presentation→workflow coupling destroys loose coupling and exchangeability; (3) **circular dependency** (data → presentation) makes incremental development, testing, and subsetting impossible — you cannot build or test any layer independently. Net effect: every change becomes nonlocal — the opposite of a good architecture (where the most common changes are local).

**Recovery strategy:**
1. **Extract business rules** from stored procedures into the business layer (rules engine/façade); stored procedures retained only for data-access performance.
2. **Re-route presentation calls** through the business layer façade; remove direct workflow-engine access from UI code.
3. **Break the cycle** — define what the data layer genuinely needs from "presentation" classes (likely shared domain types) and move those to a lower/common module; apply *restrict dependencies* (modifiability tactic).
4. **Enforce going forward:** architecturally evident coding style (layer annotations), MVC framework, code reviews, dependency-checking tools, and synchronized documentation updates at each release.

**Key concepts tested:** layer responsibility erosion, cyclic dependencies, modifiability tactics, conformance prevention.

---

### Q35. [Architecture Contexts & Competence — The Whole Picture]

A newly appointed architect asks: *"My job is just to produce the technical design documents, right?"* Using the four contexts of architecture and the professional-context duties, correct this misunderstanding with a scenario.

**Answer:**

**The four contexts:**
- **Technical** — architecture is the set of structures enabling reasoning about quality attributes.
- **Project life cycle** — architecture activities (business case → ASRs → create architecture → document → evaluate → implement/test → ensure conformance) exist in *every* development process, waterfall or Agile.
- **Business** — the system serves business goals; architects must know the vested organizations and their goals, since many profoundly influence the architecture.
- **Professional** — the architect's real duties beyond producing designs.

**Scenario correction:** In a payment-platform project, the architect must: help the PM create **cost and schedule estimates** (consensus of top-down and bottom-up); **communicate with stakeholders** — explain to the sponsor why security outranks the performance wish, to developers why constraints exist, to managers how teams will interact; **negotiate priorities and trade-offs** diplomatically; **lead** developers; keep **up-to-date knowledge** (patterns, database platforms, web-service standards); and understand **business considerations**. The architecture also shapes the organization: it becomes the work-breakdown structure — team formation, integration/test plans, and even contractual responsibilities follow it.

**Key concepts tested:** four contexts, professional duties, architecture ↔ organization structure.

---

## Section 3 — High Difficulty (Questions 36–50)

---

### Q36. [Integrated — Financial Transaction System]

A new real-time financial transaction platform must satisfy: 99.99% availability; sub-100ms authorization response; PCI-DSS security; and regulatory rule changes deployed within 5 days. Identify the architectural drivers, derive quality attribute scenarios, identify conflicts, and outline the evaluation approach.

**Answer:**

**Architectural drivers** (with stakeholder origins): 99.99% availability (business SLA), <100 ms authorization (business/customer), PCI-DSS (regulatory constraint), 5-day rule changes (business agility), cost & time-to-market (constraints).

**Quality attribute scenarios:**
- *Availability:* "A card-authorization server crashes during peak processing; the system detects the fault within 1 s and continues processing on a redundant node with no transaction loss" → measure: 99.99% availability.
- *Performance:* "Authorization requests arrive at 3,000 TPS at year-end peak; 99% complete within 100 ms" → measures: latency, throughput.
- *Security:* "An external attacker attempts unauthorized access to cardholder data; access is denied, attempt logged, authorities notified within 60 s."
- *Modifiability:* "A regulator mandates a new fraud-check rule; the change is made via configuration in ≤5 days, no code changes to 3+ modules."

**Conflicts & trade-offs:**
- Security (encryption, tokenization, audit) ↔ performance (100 ms) — mitigate: session-level crypto, async audit.
- Availability (redundancy) ↔ cost.
- Modifiability (rules indirection) ↔ performance (rules-engine lookup cost).

**Evaluation approach:** capture drivers via QAW/PALM; build utility tree with (value, impact) priorities; run **ATAM** — Steps 2–3 present drivers and architecture; Steps 5–6 map high-ranked scenarios onto architectural approaches (redundant authorization nodes, async queue, rules engine), surfacing **sensitivities** (queue depth, crypto placement) and **trade-off points** (security vs latency); document risks (single points of failure, state synchronization latency) and risk themes; verify against scenarios and repeat evaluation as the architecture evolves.

**Key concepts tested:** drivers → scenarios → conflicts → ATAM, integrated.

---

### Q37. [Integrated — Conformance Checking of Layer Violations]

Analysis of a large layered enterprise application (using SonarQube-style tooling) reports: JUnit classes referenced from production code; DAO classes calling presentation-layer formatters; direct JDBC calls from two JSP pages. Walk through how you confirm violations and restore conformance.

**Answer:**

**Step 1 — State the architectural rules** (the specification against which conformance is checked):
- R1: No production component may depend on JUnit (test-only).
- R2: Data-access layer may call only the layer(s) immediately below / its declared dependencies — never presentation.
- R3: All database access occurs via the data-access layer (entity beans/DAOs), never from presentation code.

**Step 2 — Detect:** static analysis tools (Sonar, Structure 101, Lattix — as in the course's violation examples: Sonar flagged the JUnit dependency; DiscoTect found DB access bypassing entity beans) extract the dependency graph from source code and report rule breaches, with the violating dependency chains.

**Step 3 — Confirm severity:**
- JUnit in production: test scaffolding leaked into shipped code (packaging/dependency hygiene violation).
- DAO → presentation: upward dependency — breaks strict layering; makes layers non-exchangeable.
- JSP → JDBC: **layer bypass** — data-access logic duplicated in UI; schema changes will ripple to UI.

**Step 4 — Restore conformance:** extract JUnit usage to test source sets; move shared formatting logic to a common/business module (break cycle with *restrict dependencies*); move JSP SQL into DAOs via the existing DAL; then **prevent recurrence**: folder discipline per layer, architecturally evident coding style, MVC framework enforcement, dependency rules in CI, code reviews, documentation updated at release.

**Key concepts tested:** architecture analysis/conformance, violation examples from the course, layered rules, tool-supported detection + process prevention.

---

### Q38. [Integrated — Full Reconstruction Pipeline]

A manufacturing firm has a 12-year-old MES (Manufacturing Execution System), no current documentation, rumored "spaghetti" dependencies, and a plan to add a new IoT telemetry module. Take the architect's seat: produce the current architecture and detect violations end-to-end.

**Answer:**

**Phase 1 — Raw view extraction:** run extraction over **source code, execution traces, and build scripts**. Gather: classes/files, caller–callee relationships, global data accessed by objects, and **runtime** invocation sequences (critical where objects are dynamically bound — static analysis alone misses these).

**Phase 2 — DB construction:** normalize all extracted facts into a common repository format (elements + relations), so static and dynamic facts are queryable together.

**Phase 3 — View fusion:** combine the **static view** (compile-time dependencies) with the **execution-trace view** (runtime call chains); then, with the technical team's help, **aggregate/group** low-level elements into abstract components and layers (as in the course's "Vanish" case study: the tool's white-noise view of all elements was aggregated into a comprehensible architecture — where the team discovered it was *not* strictly layered).

**Phase 4 — Architecture analysis:** validate element correctness and check conformance rules: "layers may call adjacent layers only," "all database access via entity beans/DAOs." Expected findings: UI classes calling DB directly, batch modules coupled to UI modules, shared global state — classic violations.

**Phase 5 — Iterate & act:** refine grouping; deliver: (a) reconstructed as-built architecture documented in views; (b) violation list with refactoring priorities; (c) the IoT module's integration points identified from the real dependency structure — not from obsolete documents.

**Key concepts tested:** extraction → DB → fusion → analysis → iterate; static vs dynamic views; as-built vs as-documented.

---

### Q39. [Real-Time — Automotive Braking System]

An automotive supplier is architecting an electronic braking system: brake-pedal commands must actuate within 10 ms; the system must remain operational after a single ECU fault; software must run on AUTOSAR-compliant platforms; and over-the-air updates are planned. Discuss the architectural implications across quality attributes.

**Answer:**

**Timing (performance):** 10 ms end-to-end is a hard deadline. Architectural implications: *control resource demand* — bound execution times, prioritize events (brake commands outrank diagnostics); *manage resources* — schedule resources deterministically, bound queue sizes so queuing delay can't consume the deadline; choice of technology must support **hard real-time scheduling** and known behavior under load. The coordination model: event-triggered with strict priority, synchronous within the safety path.

**Fault tolerance (availability/reliability):** single-ECU-fault operation → active redundancy (hot spare) with **voting** across replicated sensors/actuators; *detect faults* via heartbeat between ECUs; *recover* via state resynchronization and **degradation** (limp-home mode dropping non-safety functions). Safety tactics mirror availability: prevent, detect, recover from failures.

**Platform constraint (interoperability/portability):** AUTOSAR is a **framework enforcing architectural conformance** — components built to standardized interfaces; also supports **in-service software upgrade** (shadow mode before reintroduction).

**Updateability (modifiability) vs timing:** OTA updates must use *removal from service* / shadow tactics so updates never touch the active braking path; binding deferred to maintenance windows.

**Evaluation:** these drivers conflict (redundancy adds communication latency vs the 10 ms deadline) — a lightweight ATAM tracing the emergency-brake scenario through the C&C and deployment views would surface exactly this trade-off early.

**Key concepts tested:** real-time performance tactics, redundancy + voting, safety≈availability, framework conformance, update tactics.

---

### Q40. [Integrated — Real-Time Medical Infusion Pump]

An infusion pump must: deliver medication within ±2% dosage accuracy on a hard 50 ms control cycle; detect sensor faults and switch to a backup sensor without interrupting delivery; log all events for regulatory audit; run on a battery-constrained device. Architect the quality-attribute responses and their interactions.

**Answer:**

**Performance/real-time:** 50 ms control cycle with bounded jitter → *bound execution times*, *schedule resources* (deterministic RT scheduling), *limit event response* (non-control events deferred), careful *choice of technology* (RTOS with guaranteed scheduling).

**Availability/safety:** sensor fault → uninterrupted delivery requires **active redundancy** on the sensing path with **voting** (compare sensor readings; analytic redundancy if only one sensor type exists); *detect faults* (sanity checking against physical plausibility, condition monitoring); *recover* via state resynchronization to the backup sensor — hot spare, since delivery cannot pause; **degradation** to a safe rate if both sensors disagree (fail-safe, not fail-silent). Safety = avoiding states causing harm; recovery limits damage.

**Security/auditability:** every event logged (audit tactic) with integrity protection — but logging is expensive and the device is battery-constrained: log *actionable* data (faults, dosage changes, operator actions), not raw streams; buffer and transmit in batches to preserve the control cycle's energy/latency budget.

**Interactions/trade-offs:**
- Voting & double-sensing ↑ reliability but ↑ cost/energy and add latency to the 50 ms loop — allocate budget per element.
- Audit logging ↑ security accountability but contends for CPU/flash with the control cycle — schedule logging at lower priority.
- Degradation mode must itself be tested and documented (testability: replaceable sensor simulators, injectable fault states).

**Evaluation:** utility tree with (H,H) scenarios = control-cycle deadline and sensor-failover continuity; ATAM would probe the sensitivity point: control-loop latency vs added redundancy checks.

**Key concepts tested:** integrating real-time, availability/safety, security-audit, resource constraints, trade-off analysis.

---

### Q41. [Integrated — Utility Tree + ATAM on E-Commerce Platform]

You are the evaluation leader for an ATAM of an e-commerce platform rebuilt after repeated festival-season failures. Describe the complete flow from business drivers to results, including how the utility tree is built and used in scenario analysis.

**Answer:**

**Steps 1–2 (Present ATAM; business drivers):** the sponsor presents: festival-season revenue is 40% of annual sales; last year, cart-checkout failures during the Diwali peak cost an estimated ₹8 crore; drivers — survive 20× normal load, zero checkout failures, add payment methods quickly.

**Step 3 (Present architecture):** architect walks through module and C&C views — web tier, cart service, order service, inventory service, single shared relational database, payment-gateway integration.

**Step 4 (Identify architectural approaches):** session-based cart in app server; synchronous DB writes at checkout; single payment adapter.

**Step 5 (Generate utility tree):** top scenarios: *availability* — "checkout completes despite a web-server failure during peak, no lost carts (H,H)"; *performance* — "10,000 concurrent checkouts, p99 < 2 s (H,M)"; *modifiability* — "new payment method added in < 2 weeks, touching only the payment module (M,H)".

**Step 6 (Analyze architectural approaches):** map each high-ranked scenario to the architecture:
- Server failure → carts lost: **risk** (state in single app server); sensitivity = session-state placement.
- 10,000 concurrent checkouts → DB write contention: **sensitivity** (single DB, synchronous writes); **risk** of breaching p99.
- Payment adapter → new method touches only one module: **nonrisk** (good use of intermediary).

**Steps 7–9:** broader stakeholders brainstorm additional scenarios (warehouse integration, refund storms); re-analyze; present results — risks (session state, DB bottleneck), sensitivity points, trade-offs (session replication for availability vs its performance/memory cost), and **risk themes** ("single points of serialization in the write path").

**Key concepts tested:** full ATAM flow, utility tree generation and use, risks/nonrisks/sensitivities/tradeoffs, risk themes.

---

### Q42. [Integrated — Legacy Migration with Reconstruction]

An insurer must migrate a mainframe claims system to a web platform. The mainframe code is undocumented, the original architects are gone, and management fears "unknown couplings" will break the migration. Propose a plan combining reconstruction, documentation, and risk management.

**Answer:**

**Phase 1 — Reconstruct as-is architecture:** raw view extraction from COBOL/copybooks/JCL/build scripts (plus execution traces from CICS transaction logs if available) → DB construction → **view fusion** (static call graphs + runtime transaction flows) → expert aggregation into abstract components (claims intake, adjudication rules, payments, reporting). Deliver the *as-built* architecture — the only trustworthy baseline (documented architecture may not match the implementation).

**Phase 2 — Analysis for migration risk:** check the reconstructed structure for: tightly coupled modules (migration must move together), shared global data (transaction integrity risks), and reusable components (rules, logging, security — candidates for the new platform). This produces the **risk-ranked migration sequence**: strangulate the periphery (reporting, inquiry screens) before the adjudication core.

**Phase 3 — Document the target:** document the new architecture using views — module (decomposition/uses for the migrated components), C&C (web/CICS-bridge interactions), allocation (deployment) — plus behavior for cross-element interactions. Apply documentation rules: views for the stakeholders who need them; element behavior documented where it affects other elements.

**Phase 4 — Conformance during migration:** keep code and target architecture aligned: frameworks for structure, evident coding style, templates for fault-tolerance patterns, release-time doc synchronization; update the architecture document at every release, marking obsolete mainframe sections "no longer applicable."

**Key concepts tested:** reconstruction purposes (migration!), risk sequencing, target documentation, ongoing conformance.

---

### Q43. [Integrated — Flash Sale Scalability vs Consistency]

A ticketing platform faces flash sales: 100,000 users hit one event page simultaneously; overselling even one seat triggers refunds and regulatory scrutiny; the page must stay up under the load. Reason through the competing quality attributes and the architectural response.

**Answer:**

**Drivers & attributes:**
- Scalability/performance: 100,000 concurrent users; page must respond (<2 s) — *manage resources*: multiple copies of computations (scale out web tier), caching of the event page (server-side, memcached), bound queue sizes, load distribution.
- Integrity/correctness: never oversell — seat inventory updates must be **atomic**; this is both a *data-model* decision (inventory as a contended resource with correct arbitration) and a security/integrity concern.
- Availability: the page must not crash under load — *prevent faults* (bound queues, limit event response), *degrade gracefully* (read-only mode, waiting-room page) rather than fail.

**The core conflict:** massive parallelism (scale-out web servers) vs atomic inventory decrement (a serialization point). Uncontrolled concurrent updates → overselling (the airline overbooking problem). Full serialization → the single DB row becomes the bottleneck defeating scalability.

**Architectural response:** hold authoritative inventory in one transactional resource; have web servers **reserve** via short transactions through the service layer (which also handles duplicate requests and out-of-sequence messages); decouple the *read* path (cached page views — tolerably stale counts) from the *write* path (strictly consistent reservations). Queue incoming purchase requests (bounded queue) so the inventory service processes at its sustainable rate while users see a responsive waiting state.

**Trade-offs:** stale read counts vs integrity (accept approximate displayed availability, never approximate actual sales); queueing adds perceived latency vs hard overselling constraints. ATAM evaluation would flag inventory-service throughput as the sensitivity point.

**Key concepts tested:** scalability tactics, data integrity, graceful degradation, read/write path separation, conflict resolution.

---

### Q44. [Integrated — Security, Communication, and Reliability Views Intersect]

*(Modeled on your sample paper's Q2b.)* A national telemedicine platform connects rural clinics to urban specialists via video consultation, with e-prescriptions flowing to pharmacies. Show how the Security, Communication, and Reliability views intersect, and give a scenario where optimizing one compromises another.

**Answer:**

**The three views (as quality-attribute views of the architecture):**
- **Security view:** confidentiality of patient video and prescriptions (encrypt data), authentication of doctors/pharmacies, authorization by role, audit of access.
- **Communication view:** how components interact — video streams, message flows between clinic app, consultation server, pharmacy gateway; coordination model (timeliness, currency of messages), protocols, synchronous vs asynchronous prescription delivery.
- **Reliability view:** fault detection and recovery — session persistence across network drops, redundant consultation servers, heartbeat monitoring, degraded operation (audio-only if video path fails).

**Intersections:**
- Security meets Communication *at the channel*: encryption (IPSec/SSL-style session establishment) protects streams but adds handshake latency and CPU load — a communication-property cost.
- Reliability meets Security in *audit trails*: recording sessions for continuity/quality also creates confidential data needing protection.
- Reliability meets Communication: guaranteed delivery for prescriptions (queue-and-retry) vs real-time video (tolerates loss, not delay) — two coordination models in one system.

**Scenario where one compromises another:** enabling end-to-end encryption with mutual authentication and per-frame integrity checks (security view) raises video latency and jitter beyond acceptable thresholds on low-bandwidth rural links (communication/performance view). Resolution: authenticate once per session (not per message), use lighter integrity checks on the media path while keeping strong checks on prescriptions, and shift authentication cost to session setup — explicitly surfacing the trade-off to stakeholders rather than silently failing either view.

**Key concepts tested:** QA views, their intersection, trade-off analysis.

---

### Q45. [Integrated — Deep ASR Probing]

Requirement documents for a government benefits portal list: "audit trail," "notifications," "third-party interaction," "user behaviour analysis." Each is one line. As the accountable architect (no BSR detail exists), explain how you probe these into ASRs.

**Answer:**

The slides emphasize: such requirements must be **probed in detail** — understanding specific scenarios determines the design approach.

**Probe each:**
- **Audit trail:** *How will it be used?* If investigators search for specific activities → file-based log may suffice; if compliance analytics/cross-referencing → database storage with indexed queries. Also: what events, retention period, who accesses the audit data (itself sensitive → protection + access rights).
- **Notifications:** SMS, e-mail, or application-to-application via **message queue**? One-way or must the recipient **acknowledge**? Volume and timeliness (benefit-approval alerts within 5 minutes?). Wrong choice (e.g., synchronous e-mail send in the approval path) creates availability coupling.
- **Third-party interaction:** what communication mechanisms does the third party support (protocols, formats)? Is it known at design time or discovered at runtime (binding-time decision)? Security properties of their network? Failure handling — retry/queue?
- **User behaviour analysis:** real-time (performance-critical stream processing) vs batch analytics (off critical path)? Privacy constraints on behavioural data (data-model partitioning by sensitivity)?

**Method:** run a **QAW** to elicit scenario-level detail from stakeholders; derive quality attributes per scenario (e.g., notification timeliness → performance; third-party protocol diversity → interoperability); capture in a **utility tree** with (business value, architectural impact) priorities; where business justification is unclear, apply **PALM pedigree** questioning ("why 5 minutes? why not an hour?") to expose the real driver.

**Key concepts tested:** probing requirements, QAW, PALM pedigree, utility tree capture, binding-time and data-model decisions.

---

### Q46. [Integrated — Architecture as Constraints & Performance Budgets]

An architect allocates a 200 ms budget for an end-to-end funds-transfer transaction across five components (UI 30 ms, orchestration 40 ms, ledger 80 ms, fraud-check 30 ms, notification 20 ms). A developer, unaware of the budget, adds a 150 ms enrichment call inside the orchestration component. Explain this failure in architectural terms and the conformance fix.

**Answer:**

**Architectural framing:** the architecture is a **carrier of earliest, hardest-to-change design decisions** and *defines constraints on subsequent implementation*. The performance budget is one such constraint: each element must fulfill its responsibility *within its allocation* so the whole transaction meets its requirement. Implementers need not know the global budget — the architecture constrains them precisely so that local decisions cannot silently break global qualities. The developer's enrichment call is not a coding error; it is an **architecture-constraint violation**: an undisciplined local change that invalidated a global quality attribute.

**Detection:** the violation surfaces as a performance-scenario failure ("funds transfer completes within 200 ms at peak load") — detectable in integration/performance testing, but ideally earlier.

**Conformance fix:**
1. Make constraints **evident**: document per-element budgets in the architecture documentation (views + behavior), annotate budgets in code (architecturally evident coding style), review enforcement in code reviews.
2. If the enrichment is genuinely needed, the **trade-off must be renegotiated at the architectural level**: reallocate budget (ledger tightened by 30 ms?), move enrichment off the synchronous path (async via message queue), or accept a revised response measure with stakeholder sign-off — never silently.
3. Update documentation and budgets together at release.

**Key concepts tested:** architecture as constraint set, performance budgeting, conformance vs local optimization, disciplined change.

---

### Q47. [Integrated — Agile at Scale with ADD and Spikes]

A 300-developer, multi-site program is building a logistics suite using Agile. Requirements are unstable; management demands "no big design up front"; the lead architect fears architecture chaos. Propose a disciplined approach.

**Answer:**

**Start from evidence, not ideology.** The sweet-spot analysis: at this scale (far beyond 1,000 KSLOC-equivalent organizational complexity) Agile alone cannot organize the effort — significant architecture work pays off (~40%-of-schedule territory for the largest systems). But requirements instability argues against BDUF.

**The approach — candidate architecture + ADD + spikes:**
1. **Quickly design a candidate architecture** — the major structures and patterns only, deliberately leaving details open (authors' advice for large + unstable requirements).
2. **Prioritized utility tree early** (even partial) — the small set of high-value QA scenarios that constrain everything; revisit per release.
3. **Apply ADD incrementally:** each iteration, choose the module to decompose, refine it using tactics for the drivers of that increment, define interfaces, verify against scenarios — architecture *grows* with the system rather than being fixed up front.
4. **Spikes/experiments for trade-offs** (the WebArrow lesson): e.g., "will a distributed DB hurt latency vs local flat files?", "how many shipments can one routing server host?" — experiments answer architectural questions empirically within sprints, avoiding analysis paralysis.
5. **Lightweight ATAM per major increment** (4–6 h version) — keeps evaluation continuous, not a one-time gate; architects work **top-down** (structures for quality attributes) and **bottom-up** (implementation/environment constraints) simultaneously.
6. **Continuous conformance:** evident coding style, frameworks, folder discipline, code reviews; doc synchronized at each release; refactor when technical debt crosses the threshold.

**Key concepts tested:** how much architecture, candidate architecture, ADD in Agile, spikes, lightweight evaluation, middle-ground architecting.

---

### Q48. [Integrated — Modifiability vs Performance Deep Trade-off]

A market-data distribution system must (a) add new data feeds with minimal code change and (b) deliver quotes to traders within 5 ms. The architect's first design routes every feed through a generic translation/intermediary layer ("for modifiability"); profiling shows 40% of latency is spent in this layer. Analyze and redesign.

**Answer:**

**Diagnosis:** the intermediary achieves modifiability (new feeds plug in behind a stable interface — *use an intermediary*, *encapsulate*) but violates performance tactics: *reduce overhead* — the course explicitly notes that intermediaries, important for modifiability, consume resources in the event stream; removing them improves latency. A uniform mechanism applied where a uniform need doesn't exist.

**Reframe with scenarios:** modifiability scenario — "a new feed is added in ≤ 5 person-days, affecting only the feed-adapter module"; performance scenario — "a quote update is delivered to subscribers within 5 ms of arrival, 99.9% of the time." These conflict along the *coordination model* and *allocation of responsibilities* decision categories.

**Redesign — separate the paths by change-rate and criticality:**
- **Hot path** (price dissemination): minimal intermediaries; direct component-to-component attachment; feeds and subscribers that are stable members of the hot path get streamlined, dedicated connectors. This is *reduce overhead* + *increase resource efficiency* (tight code).
- **Adapter seam** (where change happens): keep the intermediary **only at the feed-ingestion boundary** — new feeds still plug in behind the stable adapter interface (encapsulation preserved where the change scenario lives).
- Verify with the utility tree: the modifiability scenario still passes (change is confined to the adapter), and the performance scenario now passes (hot path stripped).

**The deeper lesson:** tactics chosen for one quality attribute must be applied *where that attribute's scenario demands them* — architectural decisions are per-scenario responses, not global defaults. This tension (intermediary ↔ latency) is a textbook **trade-off point** for ATAM documentation.

**Key concepts tested:** modifiability vs performance tactics, intermediary overhead, change-rate-based decomposition, scenario-driven tactic placement.

---

### Q49. [Integrated — Real-Time Fraud Detection Platform]

A bank wants real-time card-fraud detection: score every transaction within 80 ms inline with authorization; models updated weekly without downtime; detection must never block a transaction for more than 120 ms (fail-open); complete audit of every scoring decision. Design the architectural response and evaluate it.

**Answer:**

**Drivers → scenarios:**
- Performance: "10,000 TPS at festival peak; each transaction scored within 80 ms, inline."
- Modifiability/availability: "fraud model updated weekly via hot deployment, zero downtime, no service interruption."
- Availability (degraded mode): "if the scoring engine is unavailable or exceeds 120 ms, authorization proceeds unflagged; fault is logged and alerting raised."
- Security/audit: "every scoring decision, model version, and outcome is recorded immutably."

**Architectural response:**
- *Inline scoring as a separate service with a strict timeout*: the authorization path calls the scoring engine synchronously with a 120 ms deadline — *bound execution times*. This is **degradation** (availability tactic): scoring is dropped before authorization is delayed. Response measure codifies the business fail-open decision.
- *Model updates*: **software upgrade in non-service-affecting manner** + **shadow mode** — the new model scores in parallel with the old (shadow) and is promoted after validation; rollback to the rollback line if behavior diverges. This is *defer binding* of model selection + state resynchronization between engine instances.
- *Weekly updates without downtime* also imply *removal from service* per-instance (rolling update) behind the redundant engines.
- *Hot-path performance*: active redundancy (multiple scoring engines) also serves throughput (multiple copies of computations); queue the scoring call with a small *bounded queue* so overload degrades predictably; *prioritize events* (fraud-score calls outrank analytics).
- *Audit*: asynchronous, append-only audit pipeline (off the synchronous path) — **audit** tactic decoupled from the latency-critical coordination model.

**Trade-offs surfaced:** fail-open protects availability/performance but accepts fraud exposure during engine failure (documented risk with monitoring); audit completeness vs hot-path latency (solved by async); model-update frequency vs validation rigor.

**Evaluation:** utility tree ranks (scoring-latency, fail-open-continuity, audit-completeness) as (H,H)/(H,H)/(H,M); ATAM Step 6 maps them to the architecture, identifying **sensitivity points** (timeout value, engine count) and the central **trade-off point** (fail-open threshold vs fraud risk).

**Key concepts tested:** real-time constraints, degradation/fail-open, shadow upgrades, redundancy for throughput, async audit, full driver→scenario→tactic→evaluation chain.

---

### Q50. [Integrated — Architecting for Testability in a Layered Payment System]

A payment system in layered architecture (presentation, business, service, data) must be regression-tested nightly against a mock gateway, with full rollback of test data, and fault-injection for the bank-switch interface. Design the testability into the architecture and explain how conformance keeps it testable.

**Answer:**

**Testability requirements → architectural decisions (per the course's three support requirements):**

1. **Ability to switch data sources (test vs production):**
   - *Abstract data sources*: data access behind interfaces (ORM/repository) so test databases substitute transparently; configuration-driven binding (test vs prod) — a *binding-time* decision made test-friendly.
   - Payment gateway behind an **abstract interface** in the service layer so a simulator replaces the real gateway — the service layer already isolates external interaction; testability extends it with a mock implementation.
2. **Rollback of test changes:**
   - Wrap test transactions in **transactions** (atomicity) so test data changes roll back; design business operations to be transaction-bounded rather than auto-commit scattered across layers.
   - **Record/playback** at the gateway interface: capture production-like traffic for repeatable tests.
3. **Ability to replace components (simulators/fault injection):**
   - The bank-switch connector defined by a narrow interface with **specialized test interfaces** for state injection; **localize state storage** in the service layer's switch-connector so the system can be started in arbitrary states; **executable assertions** at connector boundaries to detect faulty states during fault injection.
   - **Sandbox**: a full test environment isolated from production.

**Test design consequences:** the **uses structure** of the module view defines integration order; the **utility tree** ranks which scenarios the nightly regression must cover; architecture documentation tells QA which module interactions carry the highest ASR risk.

**Conformance to preserve testability:** simulators and switches exist because of architectural decisions — drift (bypassing the DAL, hard-coding gateway endpoints, leaking transactions out of the business layer) silently destroys them. Enforce via evident coding style (annotating test seams), framework configuration, code reviews of the seam interfaces, and release-time documentation sync.

**Key concepts tested:** testability tactics mapped to the three course requirements, layered placement, integration planning, conformance maintenance.

---

# Final Revision Section

## A. Top 20 Concepts to Revise

**1. Quality attribute scenario (6 parts)**
*Why:* It is the atomic unit of almost every answer — the sample paper repeatedly demands scenarios.
*Remember:* Source of stimulus, stimulus, artifact, environment, response, **response measure** (the part most often missed).

**2. Utility tree**
*Why:* Sample paper Q3b directly tests it; it drives design, test prioritization, and ATAM Step 5.
*Remember:* Root = utility → branches = quality attributes → refinements → scenarios with **(business value, architectural impact)** H/M/L priorities.

**3. ASRs and their indicators**
*Why:* Q1b of the sample paper; foundational to modules 3–5.
*Remember:* Profound measurable effect on architecture; indicators = business value, technical risk, stakeholder influence, uniqueness, SLA/QoS difference, past overruns.

**4. Architectural drivers**
*Why:* Everything flows from them: business goals, constraints, stakeholder concerns, quality attributes, regulations, technology, cost, time-to-market.

**5. QAW steps and the 30% voting rule**
*Why:* Direct exam favorite; precise steps distinguish strong answers.
*Remember:* Introduction → business mission → architectural plan → identify drivers → brainstorm scenarios (explicit stimulus + response) → consolidate → prioritize (votes = 30% of scenarios).

**6. PALM and pedigree**
*Why:* Business-goal elicitation method; pedigree = tracing a QA driver back to the business goal it serves ("why 40 ms and not 80?").

**7. Attribute-Driven Design (ADD)**
*Why:* Syllabus core for architecture design.
*Remember:* Choose module → refine using tactics for each driver → define interfaces → verify scenarios → repeat. Design is driven by **quality attributes, not functionality**.

**8. The seven design decision categories**
*Why:* Underlies every "how would you design for X" answer.
*Remember:* Allocation of responsibilities, coordination model, data model, management of resources, mapping among architectural elements, binding time, choice of technology.

**9. Availability tactics**
*Why:* Heavily detailed in slides.
*Remember:* Three groups — **detect faults** (ping/echo, heartbeat, monitor, voting, exception detection), **recover** (active/passive/cold redundancy, retry, rollback, degradation, reconfiguration), **prevent** (transactions, removal from service).

**10. Performance tactics**
*Why:* Real-time questions depend on them.
*Remember:* **Control resource demand** (prioritize events, bound execution times, limit event response) and **manage resources** (increase concurrency, multiple copies of data/computations, bound queues, schedule).

**11. Security tactics + CIA**
*Why:* Security appears in the sample paper and trade-off questions.
*Remember:* Confidentiality, Integrity, Availability (+ authentication, authorization, nonrepudiation); tactics = **detect, resist, react, recover (audit)**.

**12. Modifiability tactics**
*Why:* Change-cost scenarios are ubiquitous.
*Remember:* Reduce module size, increase cohesion, reduce coupling (encapsulate, intermediary, restrict dependencies), **defer binding** (later binding = cheaper change, but mechanisms cost more).

**13. Kruchten's 4+1 views**
*Why:* Sample paper Q2c.
*Remember:* Logical (end-user, functional), Process (integrators, concurrency/performance), Development (programmers, module organization), Physical (system engineers, deployment), +1 Scenarios (all, validity). Know the SEI mapping: Logical=module, Process=C&C, Development/Physical=allocation.

**14. Module / C&C / allocation structures**
*Why:* The definitional backbone of the course.
*Remember:* Modules = implementation units; C&C = runtime components + connectors; allocation = mapping to hardware/files/teams. Key module structures: decomposition, uses, layered, class.

**15. Layered architecture — layer responsibilities and techniques**
*Why:* Sample questions and Q9-style violations.
*Remember:* Presentation (caching, AJAX, responsive design), Business (façade, session management, workflow/rules engine, AOP for cross-cutting concerns), Data (connection pool, ORM, transactions, stored procedures, **parameterized SQL**), Service (idempotency, sequencing, retry/queue).

**16. ATAM — steps, phases, outputs, roles**
*Why:* Guaranteed exam territory.
*Remember:* 9 steps; 4 phases; outputs = **risks, nonrisks, sensitivity points, trade-off points** (+ risk themes); roles = team leader, evaluation leader, scenario scribe, proceedings scribe, questioner. Lightweight version = 4–6 hours, Steps 7–8 omitted.

**17. Architecture conformance techniques**
*Why:* Module 5 opener.
*Remember:* Evident coding style, frameworks (Spring MVC, Hibernate, AUTOSAR), code templates, **update documentation** (+ reviews, education, folder discipline).

**18. Architecture & testing**
*Why:* Distinctive part of the syllabus.
*Remember:* Utility tree → test-case prioritization; module views → integration test plan; design for testability = **switchable data sources, rollback, replaceable components**.

**19. Architecture reconstruction phases**
*Why:* Sample legacy-scenario pattern.
*Remember:* Raw view extraction (source, execution traces, build scripts) → DB construction → **view fusion** (static + dynamic + expert grouping) → architecture analysis (conformance/violations) → iterate. Tools: Dali, ARMIN, Lattix, Sonar.

**20. Good architecture rules of thumb & contexts**
*Why:* Frames "why architecture matters" questions.
*Remember:* Single architect/technical leader; prioritized QA requirements; views for stakeholders; early evaluation; incremental skeletal implementation. Four contexts: technical, project life-cycle, business, professional.

---

## B. Quality Attribute Cheat Sheet

| Quality Attribute | What it means | Typical architectural concern | Example scenario |
| --- | --- | --- | --- |
| **Availability** | System is there and ready when needed; fault → recover without service failure | Redundancy (hot/warm/cold), detection (heartbeat), graceful degradation | "Server fails at peak; system continues on standby with zero transaction loss; 99.99% uptime" |
| **Performance** | Meeting time-based constraints: latency, throughput, jitter, deadlines | Resource management, concurrency, queuing, scheduling | "10,000 concurrent checkouts; p99 < 2 s at festival peak" |
| **Security** | Protect data/services from unauthorized access while serving the authorized | Authentication, authorization, encryption, audit, limiting exposure | "External attacker attempts data access; denied, logged, authorities notified within 60 s" |
| **Scalability** | Handle growth by scaling out (more nodes) or up (bigger nodes) | Stateless design, load distribution, avoiding hardcoded resource limits | "Support 10× user growth by adding servers, no architectural change" |
| **Modifiability** | Cost/risk of change: what changes, likelihood, when/who makes it | Cohesion, coupling reduction, intermediaries, deferred binding, rules engines | "New payment method added in 2 weeks, touching one module only" |
| **Testability** | Ease of making software demonstrate its faults | Controllability & observability: abstract data sources, rollback, replaceable components | "Swap production gateway for simulator via configuration; test data rolls back after suite" |
| **Interoperability** | Degree to which systems usefully exchange meaningful information | Service discovery, interface tailoring/translation, orchestration | "Unknown hospital system discovered at runtime; incident data exchanged correctly 99.9% of the time" |
| **Usability** | Ease of accomplishing tasks + user support provided | Task/user/system models; cancel, undo, aggregate; feedback | "First-time user completes a tax return in 30 min with < 3 errors, unaided" |

---

## C. Important Frameworks / Methods

**Quality Attribute Workshop (QAW)**
*Purpose:* Elicit architectural drivers from stakeholders collaboratively.
*Key steps:* Introductions → business mission presentation → architectural plan presentation → identify drivers (consensus) → scenario brainstorming (explicit stimulus/response) → consolidation → prioritization (30% votes).
*When:* Early, before architecture is fixed; when stakeholder views conflict.

**Architectural Drivers**
*Purpose:* The distilled set of requirements (business goals, constraints, QAs, regulations, technology) that shape the architecture.
*Key steps:* Gather from requirement documents (look at design-decision categories), QAW, business goals (PALM).
*When:* Before and during design; they are the input to ADD and ATAM.

**Utility Tree**
*Purpose:* Capture and prioritize ASRs as concrete scenarios.
*Key steps:* Root=utility → branches=quality attributes → refinements → leaf scenarios, each rated (business value, architectural impact) H/M/L.
*When:* During design (ADD input), ATAM Step 5, and test prioritization.

**Attribute-Driven Design (ADD)**
*Purpose:* Systematic, quality-attribute-driven architecture design.
*Key steps:* Choose module to decompose → refine using tactics per driver → define interfaces → verify against scenarios → repeat until done.
*When:* Creating the architecture; also incrementally in Agile (candidate architecture first).

**4+1 View (Kruchten)**
*Purpose:* Document architecture so every stakeholder's concerns are addressed.
*Key steps:* Produce Logical, Process, Development, Physical views; validate with scenario view traced through all four; document only needed views.
*When:* Documenting; select views by project concerns (UI-heavy → logical; high-traffic → process; distributed teams → development; cloud/security → physical).

**Layered Architecture**
*Purpose:* Separation of concerns → maintainability, scalability, exchangeability.
*Key steps:* Presentation (caching/AJAX/responsive), Business (façade, session mgmt, workflow/rules engine, AOP), Data (ORM, connection pool, transactions, parameterized SQL), Service (external interfaces; idempotency, retry, queuing). Rule: layer uses adjacent lower layer only.
*When:* Default structure for enterprise applications; enforce via frameworks and reviews.

**ATAM**
*Purpose:* Evaluate an architecture's fitness for its quality-attribute requirements; find risks and trade-offs early.
*Key steps:* 9 steps across 4 phases — present ATAM, business drivers, architecture; identify approaches; generate utility tree; analyze approaches; brainstorm & prioritize scenarios; re-analyze; present results. Outputs: risks, nonrisks, sensitivity points, trade-off points, risk themes.
*When:* After an architecture candidate exists; repeat as it evolves; lightweight (4–6 h) for Agile teams.

**Architecture Conformance**
*Purpose:* Keep implementation aligned with the prescribed architecture.
*Key steps:* Architecturally evident coding style → use frameworks → code templates → update documentation; plus reviews, education, folder discipline; tool-based dependency checking (Sonar, Structure 101).
*When:* Continuously during implementation; synchronize docs at each release.

**Architecture Testing**
*Purpose:* Use architecture to make testing systematic, and design for testability.
*Key steps:* Prioritize test cases from utility tree; build integration plan from module/uses views; design for switchable data sources, rollback, replaceable components (simulators).
*When:* Test planning and architecture design (testability is an architectural requirement, not an afterthought).

**Architecture Reconstruction**
*Purpose:* Recover the as-built architecture of an existing (often undocumented/legacy) system.
*Key steps:* Raw view extraction (source, execution traces, build scripts) → DB construction → view fusion (static + dynamic + expert grouping) → architecture analysis (conformance/violation checks) → iterate. Tools: Dali, ARMIN, Lattix, Sonar.
*When:* Understanding legacy systems, planning migration (mainframe→web), identifying reusable components.

---

## D. Common Exam Traps

1. **Confusing business goals with architectural drivers.** A goal ("grow market share") is not a driver until refined into quality attributes/scenarios via PALM-style questioning.
2. **Treating quality attributes as functional requirements.** "The system shall be secure" is not implementable; only scenarios with response measures are.
3. **Naming a quality attribute without producing a measurable scenario.** Always write the six-part scenario — the response measure is where marks are earned.
4. **Ignoring trade-offs.** The sample paper explicitly rewards trade-off analysis (Q2b, Q4). Never present a solution as universally better; state what it costs.
5. **Assuming the highest-performance architecture is automatically the best.** Fitness for purpose depends on business goals, constraints, and priorities.
6. **Confusing views with styles/structures.** A *structure* is the set of elements; a *view* is its documented representation for stakeholders. 4+1 = views, not tactics.
7. **Treating documentation as a one-time activity.** Docs drift from code; mark obsolete parts and synchronize at release — an examined point from Module 5.
8. **Ignoring conformance during implementation.** Architecture without enforcement (evident coding style, frameworks, reviews) drifts — know the four techniques.
9. **Assuming documented architecture = current implementation.** Reconstruction exists precisely because they diverge; extraction must include **execution traces** for runtime bindings.
10. **Confusing reconstruction with design.** Reconstruction derives the as-built architecture bottom-up (extraction → fusion → analysis); design is top-down from drivers. Know both directions.
11. **Memorizing ATAM steps without knowing how to apply them.** Be ready to classify risks/sensitivities/trade-offs in a given scenario and explain what Step 6 actually does.
12. **Confusing fault, failure, and availability tactics.** A *fault* becomes a *failure* when service violates spec; tactics aim to prevent/detect/recover — and "recovery" has three sub-phases (preparation & repair, reintroduction) plus prevention.
13. **Mixing up redundancy types.** Hot = all nodes process in parallel; warm = periodic state updates to spare; cold = spare off until failover. Match to the recovery-time response measure.
14. **Forgetting that architecture constrains implementers.** Budget examples (Q46) and layer rules illustrate this; violations are architectural problems, not mere coding issues.
15. **Skipping stakeholder mapping in view questions.** Every 4+1 answer should name the viewer and the concern — that's the grading rubric implied by the sample paper's Q2c.

---

**Final quality check performed:** exactly 50 questions (10 medium, 25 medium-high, 15 high); all scenario-based; 15+ integrated multi-concept questions (Q15, 20–25, 32–50); coverage across all syllabus units with heaviest weight on quality attributes, ASRs/utility tree/QAW/PALM, ADD, views/4+1, layered architecture, ATAM, conformance, testing, reconstruction, and real-time systems; terminology and examples (facade, memcached, AJAX, rules engine, hot/warm/cold spare, BHIM, Vanish/ARMIN, Sonar/JUnit example, 30% voting, sweet-spot percentages) drawn directly from your faculty's slides; style mirrors the sample paper's "how does X influence Y — provide an example/analyze trade-offs" format. Good luck with the exam!