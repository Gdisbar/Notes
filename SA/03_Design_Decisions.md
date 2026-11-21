# PART 3 — DESIGN-DECISION QUESTIONS
*Choose between specific architectural paths → **explicit trade-offs · selection criteria · boundaries (when NOT to use)***
REF format `[file:page]` → see `00_READ_FIRST.md`. Concepts in `01_Concepts.md` (**C1…**, **I1…**).

---

# 0. HOW TO ANSWER ANY "WHICH ONE / WHY" QUESTION

**The 5-line decision answer**
1. **Drivers first:** "The dominant ASRs here are … (state 2-3, from the scenario)." `[2:1-3]`
2. **Options:** name 2-3 realistic candidates.
3. **Criteria table:** compare on the *drivers* (not on everything).
4. **Choose + because + at the cost of:** "We choose X **because** it gives … **at the cost of** …".
5. **Boundary:** "X would be wrong if … (scale, team, latency, data sensitivity changes)"; add *how you'd validate* (ATAM scenario / spike / CBAM VFC).

> A decision answer with **no cost and no boundary** is a half-answer. These two phrases are what the marks are for.

**Selection criteria menu** (pick those relevant): quality-attribute drivers · business goals & time-to-market · cost (build, run, skills) · team familiarity & tool support `[1:42]` · risk · reversibility (binding time) · compliance · scale · compatibility with existing stack · evolution/maintainability.

---

# 1. CHOOSING TACTICS WITHIN A QUALITY ATTRIBUTE

## D1.1 Redundancy level — hot vs warm vs cold spare `[1:52]`
| Option | Recovery time | Cost | Data-loss window | Choose when |
|---|---|---|---|---|
| **Active (hot)** — spare processes identical input | seconds | highest | ~none | payments, trading, life-critical; 99.99%+ |
| **Passive (warm)** — periodic state updates | longer (state catch-up) | medium | since last update | business apps with minutes of tolerance |
| **Cold spare** — powered off, reset on failover | longest | lowest | everything since last backup | batch/non-critical, DR |
- **Boundary:** hot spare duplicates all processing cost *and* needs deterministic replicas; warm/cold risk stale state. All involve the **coordination model** (state sync). `[1:65]`

## D1.2 Fault-detection choice `[1:49-51]`
- **Ping/echo** (simple reachability; ↑ network load, DoS-able `[3:108]`) vs **heartbeat** (cheap periodic signal; detection delay = interval) vs **voting** (detects *wrong results*, needs replicas) vs **sanity/condition checks** (need design knowledge). Pick by *fault type*: crash → heartbeat; incorrect response → voting/sanity.

## D1.3 Performance: reduce demand vs add resources `[1:70-73]`
| Path | Pros | Cons | Choose when |
|---|---|---|---|
| **Control demand** (rate-limit, prioritise, bound execution, sampling) | cheap, predictable | lost fidelity, rejected work | demand is bursty/can be shed |
| **Increase resources** (scale up/out) | simple, no code change | cost; may not fix contention/algorithm | capacity truly short |
| **Increase concurrency** | cuts blocked time | complexity, non-determinism (↓ testability) | independent requests |
| **Replicate computation/data (cache)** | cuts latency & contention | stale/consistency, memory | read-heavy |
| **Remove intermediaries** | lowest latency | **↓ modifiability** | hot path with stable interfaces |

## D1.4 Caching: client-side vs server-side `[2:35]`
Client-side = per-user data (preferences, address); fast, but not shared and may be stale. Server-side = shared data (product codes, promotions) used by many back-end modules. **In practice both.** Boundary: sensitive data should not live in the browser cache.

## D1.5 Security: which tactic category to emphasise `[1:110-115]`
**Resist** (authenticate, authorise, limit exposure, encrypt, separate) is cheapest *before* the attack; **detect/react** accept that "no tactic is foolproof"; **recover (audit)** gives accountability. For high-value assets use **defence in depth** (several categories). Trade-off: each layer costs performance/usability.

## D1.6 Modifiability: which lever `[1:127-131]`
| Lever | Reduces | Cost |
|---|---|---|
| Split module | size | more interfaces/communication |
| Increase cohesion | ripple of change | refactoring effort |
| Encapsulate / intermediary / restrict dependencies | coupling | indirection ↓ performance |
| **Defer binding** (config, rules engine, plug-ins) | cost of *each* change | the mechanism is expensive to build; harder to test |
- **Selection rule:** defer binding only for changes that are **likely and frequent**; otherwise encapsulation is cheaper (YAGNI).

## D1.7 Tactic side-effects chain `[3:107-113]`
Every tactic creates a new concern: *ping/echo → security (ping flood), performance overhead, modifiability* → *increase resources → cost* → *scheduling policy → modifiability* → *intermediary → ensure all traffic passes → restrict communication paths → performance overhead* … It terminates when side-effects become small enough to ignore. **Use this to show trade-off awareness in any tactic answer.**

## D1.8 Attribute conflicts to cite (classic trade-offs)
| Attribute ↑ | Often hurts | Why |
|---|---|---|
| Security | performance, usability | encryption/auth overhead, extra steps `[2:56]` |
| Modifiability (intermediaries, late binding) | performance | indirection `[1:71]` |
| Availability (redundancy) | cost, complexity | duplicate resources + state sync |
| Performance (concurrency) | testability | non-determinism `[1:163]` |
| Interoperability | security, performance | exposed interfaces, translation |

---

# 2. CHOOSING A PATTERN

## D2.1 Master selection table (problem → pattern)
| If the dominant problem is… | Choose | Why | Watch out |
|---|---|---|---|
| Independent evolution, portability, reuse of services; strict separation | **Layers** `[3:5-12]` | unidirectional use, local dependencies | efficiency, cascading changes, granularity |
| Stream of data transformed in steps; reusable steps | **Pipes & Filters** `[3:13-20]` | recombine/exchange filters; parallel | shared state, error handling, transform overhead |
| No deterministic algorithm; need cooperating experts (speech/vision) | **Blackboard** `[3:21-28]` | incremental, opportunistic solution | control complexity, testing, no guarantee |
| Distributed, heterogeneous components that must find/invoke each other transparently | **Broker** `[3:29-43]` | location transparency | broker bottleneck/SPOF, efficiency |
| Interactive UI with many views of the same data, easy UI change | **MVC** `[3:44-55]` | multiple synchronised views, pluggable controllers | complexity, too many updates, coupling |
| Interactive system as hierarchy of cooperating, possibly concurrent agents | **PAC** `[3:56-65]` | multi-tasking, extension | complexity, control component |
| Must adapt to changing platforms/requirements; customer-specific parts | **Microkernel** `[3:70-80]` | mechanism vs policy; plug-in servers | performance, design complexity |
| Run-time change of structure/behaviour without recompiling | **Reflection** `[3:81-90]` | meta level adaptation | damage by meta changes, overhead, language support |
| Many unaware producers/consumers; loose coupling | **Publish-Subscribe** `[3:96-98]` | anonymity, dynamic subscribers | latency, ordering, no delivery guarantee |
| Consumers need services without knowing implementation, across platforms | **SOA** `[3:66-69]` | interoperability | complexity, no control over services |
| Shared persistent data accessed by many | **Shared-Data (repository)** `[3:99-100]` | single source of truth | store bottleneck/SPOF, schema coupling |
| Petabyte-scale batch analysis | **Map-Reduce** `[3:101-103]` | parallel, fault-tolerant | batch latency, restructuring problem |
| Shared services for many distributed clients | **Client-Server** `[3:91-92]` | centralised control/reuse | server bottleneck |
| Equal nodes sharing resources; no central authority | **Peer-to-Peer** `[3:93-95]` | scalability/availability | security, consistency, backup |
| Allocate components to separate hardware groups | **Multi-Tier** `[3:104-105]` | independent scaling/security per tier | latency, cost |

## D2.2 Head-to-head comparisons (likely question forms)

### Layers (strict) vs Relaxed layers `[3:10]`
Strict: only next-lower layer → maximal isolation/portability, but extra pass-through cost. Relaxed: any lower layer → efficiency, but weaker isolation and portability. *Choose relaxed for performance-critical paths; strict where portability dominates.*

### Layers vs Microkernel
Layers = vertical decomposition by abstraction level; **Microkernel** = minimal core + pluggable servers/personalities and **separation of policy from mechanism**, ideal for *product-line / platform adaptation*. Layers can't easily swap whole "personalities". Microkernel costs performance (IPC via kernel) and design complexity `[3:79-80]`.

### Microkernel vs Reflection (both "adaptable")
Microkernel adapts **at design/deploy time by plugging servers**; Reflection adapts **at run time by changing metaobjects** through the MOP. Reflection is more powerful but riskier (damage), slower and language-dependent `[3:90]`.

### Pipes & Filters vs Blackboard
P&F: **known, sequential** processing; deterministic; filters don't share state. Blackboard: **no known algorithm**; knowledge sources *share* the blackboard and work opportunistically under a control component. Use P&F for ETL/compilers; Blackboard for recognition/AI.

### Broker vs Publish-Subscribe vs SOA vs Client-Server
| | Broker `[3:29]` | Publish-Subscribe `[3:96]` | SOA `[3:66]` | Client-Server `[3:91]` |
|---|---|---|---|---|
| Coupling | clients know *service interface*, not location | producers/consumers don't know each other | consumers know published contract | clients know the server |
| Interaction | request/reply via broker | event announce → many listeners | request/reply or async messaging, REST/SOAP | request/reply |
| Best for | distributed object invocation | event broadcast, dynamic subscribers | enterprise integration across platforms | shared central service |
| Weak point | broker SPOF/bottleneck | latency, ordering, delivery not guaranteed | complexity, no service control | server bottleneck |

### MVC vs PAC
MVC: **one** model with many views and controllers — good for single-application UIs; tight view-controller coupling. PAC: **hierarchy of self-contained agents** each with its own presentation/abstraction/control — good for **distributed/concurrent** interactive subsystems; more complex control. Choose MVC for conventional UIs; PAC when the UI decomposes into independent semantic concepts or needs multi-tasking `[3:64]`.

### Shared-Data vs Message passing (pub-sub / broker)
Shared data = consistency and single truth, but a bottleneck/SPOF and schema coupling. Messaging = decoupling and scalability, but eventual consistency and delivery issues.

### Client-Server vs Peer-to-Peer
C-S centralises control (easy security, consistency, backup) but concentrates load. P2P spreads load and removes the SPOF but makes security, consistency, availability and recovery harder `[3:95]`.

## D2.3 Pattern + tactic augmentation `[3:107]`
A pattern is rarely sufficient: *Broker (SPOF) + multiple copies*; *Layers (efficiency) + relaxed layering/caching*; *MVC + intermediary for view updates*. Always state **which tactic patches which weakness**.

---

# 3. CHOOSING WHAT TO DOCUMENT (VIEWS & STRUCTURES)

## D3.1 SEI structures vs Kruchten 4+1 `[2:30-31]`
| Project concern | Prioritise | Reason |
|---|---|---|
| Complex UI / functional understanding | **Logical** (module) | end-user & functionality `[2:23]` |
| High traffic / performance / concurrency | **Process** (C&C) | scalability, concurrency `[2:24]` |
| Large or distributed team | **Development** (allocation) | module organisation, layers, reuse `[2:25]` |
| Cloud-native / security / availability | **Physical** (allocation/deployment) | hardware mapping `[2:26]` |
| Verify consistency | **Scenarios (+1)** | `[2:27]` |
- **Rule:** *document only what is necessary to communicate the design and mitigate risk.* `[2:31]`
- **Which structure for which analysis `[2:17-19]`:** modifiability → decomposition; incremental subsets → uses; portability → layers; performance/availability/security → deployment; contention → concurrency; project planning → work assignment.

---

# 4. CHOOSING AN EVALUATION APPROACH

| Approach | When | Cost | Pros | Cons |
|---|---|---|---|---|
| **Designer self-evaluation** `[2:57]` | every key decision; generate-and-test | low | continuous, immediate | bias |
| **Peer review** (lightweight) `[2:58]` | repeatedly, small/medium risk | hours | cheap, internal | less objective |
| **Outsider analysis** `[2:59]` | complete architecture, high risk; managers must believe findings | high | objective, expertise | cost, ramp-up |
| **Full ATAM** `[2:61-74]` | large, costly project where a major mistake is unacceptable | **20-30 person-days** + stakeholders | risks, sensitivity & trade-off points, business alignment | heavy |
| **Lightweight ATAM** `[2:75-76]` | smaller/less risky, tight schedule | **½-1 day**, internal | agile-friendly | less coverage |
- **Selection factors:** importance of decision, number of alternatives, artifacts available, who sees results, who evaluates, which stakeholders participate. `[2:57, 2:60]`
- **Boundary:** evaluation needs an architecture artifact; it focuses only on the top QA scenarios, not the whole system. `[2:61]`

---

# 5. AGILE vs UP-FRONT ARCHITECTURE `[1:172-187]`
| System profile | Decision | Reason |
|---|---|---|
| Small (~10 KSLOC), uncertain requirements | agree only the **major patterns**; minimal up-front | sweet spot far left `[1:180, 1:186]` |
| Medium (~100 KSLOC) | ≈ **20%** of schedule on architecture/risk | sweet spot |
| Large (~1000 KSLOC), stable requirements or distributed dev | ≈ **40%** up-front; **BDUF pays off** | complexity needs guidance |
| Large, unstable requirements | **candidate architecture quickly**, elaborate with **spikes** | balance `[1:184, 1:186]` |
- **Cost of too little:** rework and technical debt; **cost of too much:** delayed value, wasted work. Agile architects accept a skeleton, then **refactor when debt gets too high**. `[1:187]`

---

# 6. MOBILE: NATIVE vs WEB vs HYBRID `[4:2-5, 4:30]`
| Criterion | Native | Mobile web / PWA | Hybrid |
|---|---|---|---|
| Performance & UX | **best** | lower, slower | medium |
| Device features (camera, GPS, sensors) | all | limited | via native shell |
| Code base / cost | one per platform; special skills | **single** code base | **mostly common** code |
| Distribution/updates | app-store, version updates | no install, deploy instantly | app-store |
| Examples | Maps, Facebook, games | Flipkart, BookMyShow, MakeMyTrip | Twitter, Uber, Instagram |
- **Choose Native** for games/graphics/real-time device use; **Web/PWA** for content/transactions with wide reach and fast release; **Hybrid** when you need device hardware but want one code base and fast delivery (tools: React Native, Xamarin).
- **Other decisions:** responsive design vs separate mobile site; **local storage + sync** vs always-online (connectivity); few layers vs clean layering (performance vs modifiability). `[4:1, 4:6-7]`

---

# 7. CLOUD & DATA DECISIONS

## D7.1 IaaS vs PaaS vs SaaS `[3:116, 3:122-125]`
| | IaaS | PaaS | SaaS |
|---|---|---|---|
| Consumer | developer/admin | developer | end user |
| You control | OS, storage, apps | code & config | nothing (just use) |
| Provider manages | hardware, hypervisor | OS, runtime, stack | everything |
| Choose when | need OS/network control, lift-and-shift | want to ship code fast, standard stack (LAMP) | commodity function (email, CRM) |
| Risk | most ops effort | lock-in, platform limits | least customisation, data in vendor hands |

## D7.2 Public / private / community / hybrid `[3:117]`
Public = cost & elasticity, less control; Private = control/compliance, higher cost; Community = shared concerns across organisations; **Hybrid** = sensitive data private + burst to public. Decision drivers: data sensitivity, regulation, workload variability, budget.

## D7.3 Multi-instance (single-tenant) vs single-instance (multi-tenant) `[4:22-26, 3:119]`
| | Multi-instance / single-tenant | Single-instance / multi-tenant |
|---|---|---|
| Isolation/security | **strong** | weaker (schema/org-ID separation, VM escape risk) |
| Cost per tenant | high | **low** (shared help desk, one upgrade, one version) |
| Customisation | easy per tenant | harder |
| Upgrade | per tenant | once for all |
| Choose when | strategic, volatile, confidential workloads | **utilitarian cross-industry** workloads (email, expense, travel, sales-force) `[4:25]` |
- **Per-tier mixing:** decide *per tier* by processing load and data confidentiality `[4:28-29]` (see Part 2 §9.2).
- **Degree:** highest / middle / lowest multi-tenancy `[4:26]`.

## D7.4 CAP — which two? `[3:135-139, 4:18]`
| Pick | Behaviour in partition | Examples | Use when |
|---|---|---|---|
| **AP** | stay up, return older data (eventual consistency) | CouchDB, Cassandra, ScyllaDB | cart, feeds, availability-critical |
| **CP** | block/lock until consistent | MongoDB, Redis | orders, payments, inventory counts |
| **CA** | only without partitions (single node) — theoretical | — | don't promise in distributed systems |
- **Decide per service**, not per system (Amazon cart vs order).

## D7.5 SQL vs NoSQL; which NoSQL `[3:126-129, 4:46-55]`
RDBMS: transactions (ACID), joins, schemas, consistency. NoSQL: scale (sharding), speed, availability, flexible data — **but no ACID/joins**. Choose NoSQL when volume/velocity/variety dominate and transactions aren't needed; keep RDBMS for transactional integrity. Within NoSQL: **key-value** (fast lookups), **document** (nested JSON), **column** (analytics over few columns), **graph** (relationships), **in-memory** (ultra-low latency; volatility/cost).

## D7.6 Batch vs streaming vs in-memory `[4:38-45]`
| | MapReduce/Hadoop | Spark | Stream (Storm/Flink + Kafka) | MPP/SQL engines |
|---|---|---|---|---|
| Latency | high | low-medium | **lowest** | low for structured |
| Fault tolerance | **excellent** | good | complex (checkpoints) | managed |
| Volume | petabyte cold data | needs memory | memory-intensive | cost scales |
| Choose when | cheap nightly historical logs | iterative/real-time analytics | sub-second events, fraud, trending | interactive SQL on structured data |

## D7.7 Security technologies `[4:60-71]`
| Need | Pick | Why not the others |
|---|---|---|
| Encrypt browser↔server | **TLS** | OpenID/OAuth don't encrypt channels |
| Sign in with an existing identity | **OpenID** (authentication) | OAuth is about data access |
| Let an app access my data elsewhere | **OAuth** (authorisation, token) | OpenID only proves identity |
| Fast directory lookups for millions of users | **LDAP** (read-mostly) | not for frequently-updated data |
| Enterprise roles/compliance | **IAM** | |
| Protect internal network | **Firewall + DMZ** | |

## D7.8 ML approach `[4:75-79]`
Labelled data and a prediction target → **supervised**; unlabelled, discover structure → **unsupervised**; sequential decisions with feedback → **reinforcement**. Features need domain knowledge; always train-test-validate and iterate.

---

# 8. ECONOMIC DECISIONS (CBAM) `[4:95-116]`
- **Decision rule:** pick strategies by **VFC = B/C**, not by benefit alone or cost alone; stop when budget/schedule is exhausted `[4:99, 4:109]`.
- **Boundary:** results depend on stakeholder-elicited utilities (subjective) → step 9 **intuition check** against business goals; iterate if misaligned `[4:110]`.
- **Answer line:** "Strategy X ranks first because VFC=…; Strategy Y has higher total benefit but is 3× the cost; we'd revisit if the budget grows."

---

# 9. CONFORMANCE: WHICH TECHNIQUE? `[2:78-83]`
| Technique | Strength | Weakness |
|---|---|---|
| Architecturally evident coding style | cheap, visible in code | relies on discipline |
| Frameworks (Spring, Hibernate, AUTOSAR) | enforce structure | lock-in |
| Code templates | repeatable QA support (e.g. fault-tolerant) | rigid |
| Update documentation | trust, onboarding | manual effort |
| Code reviews / education / folders per layer | cultural, cheap | inconsistent |
- **Pair with reconstruction** `[2:88-92]` to *check* conformance on legacy code.

---

# 10. PROJECT-LEVEL DECISIONS `[4:81-94]`
- **Top-down vs bottom-up schedule:** top-down for go/no-go and budgeting (inaccurate); bottom-up (by architecture team) for accuracy; **reconcile**. 
- **Trade-offs:** architect → QA vs QA; PM → feature vs schedule vs quality; resist scope creep with a Change Control Board.
- **Governance responsibilities** (controls, compliance, process, accountability) `[4:93]`.

---

# 11. LIKELY DESIGN-DECISION QUESTIONS (inferred) → ANSWER FROM
| Likely question | Section |
|---|---|
| Which pattern for X and why? Compare Layers/Microkernel/Broker/Pub-Sub/MVC/PAC | §2 |
| Hot/warm/cold spare; ping vs heartbeat | D1.1-D1.2 |
| Reduce latency: cache vs scale vs remove layers | D1.3-D1.4 |
| Which tactics for security/modifiability; trade-offs | D1.5-D1.8 |
| Which views to document for a given project | §3 |
| Which evaluation (peer review vs ATAM vs lightweight) | §4 |
| How much architecture up-front | §5 |
| Native vs web vs hybrid | §6 |
| IaaS/PaaS/SaaS; public/private/hybrid; single vs multi-tenant; CAP; SQL vs NoSQL; Hadoop vs Spark vs streaming | §7 |
| Choose strategy with CBAM | §8 |
