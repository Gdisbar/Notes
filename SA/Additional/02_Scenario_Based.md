# PART 2 — SCENARIO-BASED QUESTIONS
*Handle a concrete situation, failure or edge case → **Diagnose → Mitigate → Recover → Prevent***
REF format `[M:page]` (page of your merged PDF, four slides per page) → see `00_READ_FIRST.md`. Concepts are in `01_Concepts.md` (cross-referenced as **C1…**).

---

# 0. THE SCENARIO-ANSWER ALGORITHM (use on every scenario question)

| Step | Do this | Write this phrase |
|---|---|---|
| 1. **Classify** | Which QA(s) does the situation threaten? (many incidents hit 2-3) | "This primarily threatens *availability* and secondarily *performance*." |
| 2. **Restate as a QA scenario** | Source · stimulus · artifact · environment · response · **measure** `[M:24-25]` | "Stimulus: DB node crashes in normal operation; response: fail over; measure: < 30 s, no data loss." |
| 3. **Diagnose** | What fault model / bottleneck / attack type? Where in the structure (module, C&C, deployment)? | "Fault of type *crash* at the data tier." |
| 4. **Mitigate now (tactics)** | Use the tactic *categories* (detect/recover/prevent; control demand/manage resources; detect/resist/react/recover) | "Detect via heartbeat; recover via passive redundancy + state resync." |
| 5. **Recover & restore** | Reintroduction, data repair, audit, rollback | "Shadow the repaired node, then reintroduce." |
| 6. **Prevent recurrence** | Architecture change, monitoring, test, process | "Add predictive monitoring; chaos test; update runbook." |
| 7. **Trade-offs & edge cases** | Cost, other QAs affected, what if mitigation itself fails | "Hot spare doubles cost; heartbeat adds load; add timeout to avoid false positives." |
| 8. **Measure success** | the response measure from step 2 | "We verify with failover drill; target 99.99%." |

> **Edge-case checklist** (examiners love these): *transient vs permanent fault · partial failure · duplicate / out-of-order messages · overload during recovery · false alarms of the detector · failure of the mitigation component itself · data inconsistency after failover · security impact of the mitigation.*

---

# 1. AVAILABILITY SCENARIOS (concept C1 · `[M:29-35]`)

## S1.1 Server / node crash during normal operation
- **Diagnose:** fault = *crash* (or *omission*); detected by missing **heartbeat/ping-echo** from the **monitor**. `[M:30]`
- **Mitigate:** fail over to **redundant spare** — *hot (active)* if zero-downtime required; *warm (passive)* for cheaper, slower; *cold* for non-critical. **Reconfigure** remaining resources; operate in **degraded mode** (drop non-critical features). `[M:31-32]`
- **Recover:** repair node → run in **shadow mode** → **state resynchronisation** → reintroduce; use **escalating restart** (process → service → node). `[M:32]`
- **Prevent:** **removal from service** (planned restart before memory leaks bite), **predictive model**, **transactions**. `[M:32]`
- **Edge cases:** *split-brain* (both think they're primary → use quorum/monitor); false positive from a slow network (use timeouts + retry before failover); spare is out-of-date (warm/cold) → data loss window → state the RPO.
- **Measure:** time-to-detect, time-to-repair, availability %. `[M:29]`

## S1.2 Transient failure / flaky dependency
- Use **retry** (only for *transient* faults) with back-off; **ignore faulty behaviour** for spurious messages; **exception handling** that masks and retries; **timeout** (exception detection). `[M:31-32]`
- **Edge case:** retries can amplify load on an already-struggling service (↓ performance) → bound retries, use **bound queue sizes** `[M:38]`; duplicate effects → idempotent operations; make the *service layer* handle **duplicate requests, out-of-sequence messages and communication failure by retry or queue-and-send-later**. `[M:112]`

## S1.3 Bad deployment / software upgrade breaks service
- **Mitigate:** **rollback** to the previous known-good state ("rollback line"); `[M:31]` **shadow** the upgraded component before activation; **in-service software upgrade** without service impact. `[M:31-32]`
- **Prevent:** staged rollout, **sandbox** and regression testing `[M:66]`, update docs/code in sync `[M:127]`.

## S1.4 Cloud instance / zone failure
- **Facts:** failures are *normal* with 1000s of servers; provider keeps the cloud up but **applications must detect and correct instance failures**. `[M:216]` 99.95% still = 0.05% downtime. `[M:236]`
- **Mitigate/prevent:** auto-scaling & automatic IP reallocation `[M:214]`; **replicate across geographic zones (Netflix)** `[M:236]`; **remove single points of failure**, disposable resources `[M:217]`.
- **Edge case:** data replicated across zones → *CAP*: during a partition choose **availability (AP)** for revenue-producing paths (shopping cart) and **consistency (C)** for order submission. `[M:236]`

## S1.5 HDFS write failure cases (very exam-friendly) `[M:212]`
| Failing party | Detection → response |
|---|---|
| **Client** | application detects, retries; write isn't complete until committed by client |
| **NameNode** | **backup NameNode** takes over; **log file** prevents metadata loss; DataNodes hold the true block list; client detects and retries |
| **DataNode** | client or earlier DataNode in the pipeline detects → asks NameNode for a different DataNode; block already **replicated 3×**, so one failure loses nothing |
- *Tactics to name:* **active/passive redundancy, exception detection, retry, rollback, state resync**. HDFS also uses **heartbeats** from DataNodes to NameNode `[M:246]`.

## S1.6 "Broker is a single point of failure"
- Pattern weakness `[M:162]`; **fix with tactics**: *maintain multiple copies* (replicated brokers, passive redundancy) for availability; *increase resources* for performance. `[M:203]` Cost: more infrastructure, state sync between brokers.

---

# 2. PERFORMANCE SCENARIOS (C2 · `[M:36-40]`)

**Diagnosis grid:** *Is it demand (too many events) or resource (too little capacity / too much blocking)?* → choose tactic category.

## S2.1 A frequently-used screen loads slowly (Aadhaar State/District/Town) `[M:113]`
- **Diagnose:** large static reference data fetched and rendered every time.
- **Mitigate:** **cache** States/Districts/Towns in the back-end (**multiple copies of data**); **AJAX** to load districts/towns dynamically (**reduce demand per request**). Also client-side caching for user preferences. `[M:106]`
- **Edge case:** stale cache when reference data changes (invalidation policy); cache memory cost; mix client and server caching. 

## S2.2 Peak-load slowdown at the data tier
- **Mitigate:** connection pool; **multiple copies of data** (read replicas); **stored procedures**; ORM tuning; **increase resources** (vertical) or **increase concurrency / replicas** (horizontal). `[M:37-38, M:111]`
- **Prevent:** **bound queue sizes**, **prioritise events**, **limit event response** (rate limiting), **schedule resources**. `[M:37-38]`
- **Edge case:** auto-scaling too slow for sudden spikes → application must be **self-aware and proactive** about resource needs. `[M:216]` Replicas reduce read latency but raise consistency problems.

## S2.3 Event flood / overload (sensor stream, burst of requests)
- **Reduce demand:** **manage sampling rate** (accept lower fidelity), **limit event response**, **prioritise events**, **bound execution times**. `[M:37]`
- **Add capacity only if demand can't be cut:** increase resources/concurrency.
- **Edge case:** overload *during recovery*; priority starvation of low-priority events.

## S2.4 Latency added by "clean" design (too many intermediaries/layers)
- **Reduce overhead:** remove intermediaries on the hot path (trade **modifiability ↓**) `[M:37]`; **few layers** on mobile `[M:226]`; façade/caching to cut round trips.

## S2.5 MapReduce too slow for fraud detection `[M:248-249]`
- MapReduce is **disk-based**; real-time fraud needs sub-second → move to **Spark (in-memory)** or **stream processing (Storm + Kafka)**. Keep MapReduce for cheap nightly historical batches `[M:251]`.

---

# 3. SECURITY SCENARIOS (C4 · `[M:49-53]`)

## S3.1 Insider attempts to modify a protected table `[M:49]`
- **Resist:** identify + **authenticate + authorise** actors, **limit access** (least privilege), separate sensitive data with different access rights. `[M:50, M:52]`
- **Detect:** log/audit all access to sensitive data; verify message integrity.
- **React:** **revoke access**, lock account after repeated failures, **inform actors**. `[M:51]`
- **Recover:** **audit trail** to identify the attacker; restore data (from backup/rollback) within target time (e.g. one day). `[M:51]`
- **Edge case:** the attacker is an *authorised* user — authentication alone isn't enough; log files themselves must be protected.

## S3.2 Denial-of-service attempt
- **Detect:** compare traffic to historic DoS signatures; **monitor** (also detects congestion). `[M:30, M:50]`
- **Resist:** **limit exposure**, **limit access**, firewall filtering, **separate entities**; **bound queues**/**limit event response**. `[M:37, M:50]`
- **Recover:** availability recovery tactics (spares, reconfiguration). **Edge case:** ping/echo itself can be abused (**ping flood**) → rate-limit health checks. `[M:204]`

## S3.3 SQL injection against a web form `[M:111]`
- Cause: concatenating user input into SQL (`' OR 'x'='x'` returns all rows).
- **Fix (data layer):** **parameterised queries**, stored procedures, input validation, least-privilege DB account, ORM. **Detect:** log + intrusion detection. Cross-ref: *encrypt data, limit exposure*.

## S3.4 Data in transit between browser and site
- Use **TLS**: agree version/algorithms → authenticate via **certificates** → shared **symmetric key** (fast). Provides **privacy + integrity**. `[M:263]`

## S3.5 "Let users log in with Google" / "app wants to read user's contacts"
- **OpenID = authenticate** (redirect to provider, return credentials not password). **OAuth = authorise** an app to access your data via a **token**. `[M:264-266]` Use **LDAP** for fast read-mostly validation of millions of users; **IAM** for roles/groups/compliance. `[M:266]`

## S3.6 Multi-tenant leakage / VM escape `[M:216, M:238]`
- Risks: shared-table with org-ID column and poor design → inadvertent sharing; **VM escape**; side-channel; DoS from a neighbour.
- **Mitigate:** tenant isolation in data model (separate schemas/instances for sensitive data), **separate entities**, encryption, hypervisor hardening, access checks on every query. **Trade-off:** more isolation = less sharing = higher cost. `[M:238-239]`

## S3.7 Perimeter design for a web app
- **Firewall + DMZ:** public-facing servers in DMZ between two firewalls; attacker must pass two barriers; use packet filtering, application-layer filtering, NAT. `[M:267-268]`

---

# 4. MODIFIABILITY & CHANGE SCENARIOS (C5 · `[M:54-58]`)

## S4.1 "Marketing wants a new payment method / fee change next week"
- **Diagnose:** change type & likelihood — *what can change, how likely, when, by whom?* `[M:54]`
- **Mitigate:** if anticipated → **defer binding** (configuration, rules/workflow engine) so the change needs *no code change*; `[M:83, M:108]` if not → **encapsulate** the payment module behind an interface, **use an intermediary/adapter**.
- **Measure:** cost (effort, artifacts touched, side-effects). `[M:54]`
- **Edge cases:** the change touches several modules (non-local change) → *split module*, *increase cohesion*, *refactor duplicates*, *abstract common services*. `[M:55-56]`

## S4.2 "Replace the database / port to another platform / upgrade a vendor product"
- **Principle:** never depend on a specific version of a commercial tool; isolate behind a data-access layer/ORM; **layers** give portability. `[M:17, M:21, M:111]`
- **Edge case:** vendor lock-in through *choice of technology* decisions (checklist) `[M:58]`.

## S4.3 Code has drifted from the architecture (layer violations)
- **Diagnose:** reconstruct the as-built architecture (raw view extraction → DB → view fusion → analysis). `[M:131, M:134-135]`
- **Fix:** conformance checks (adjacent-layer calls only; DB access via data layer), **architecturally evident coding style**, frameworks, code templates, code reviews, folder-per-layer, update docs at release. `[M:125-126]`
- **Edge case:** documentation lies → mark obsolete parts "no longer applicable" to keep trust in the rest `[M:127]`.

## S4.4 Legacy system, no documentation, must migrate (mainframe → web)
- **Use architecture reconstruction** to find reusable components (logging, security) and recover layers; then re-architect target. `[M:131]`

---

# 5. INTEROPERABILITY SCENARIOS (C6 · `[M:60-63]`)

## S5.1 External partner (e.g., Makemytrip) must query/book our hotel rooms `[M:113-114]`
- **Answer:** build a **service layer** exposing *GetRoomAvailability(from,to) → rooms by type* and *ReserveRoom(count,type,from,to) → success/failure*.
- **Tactics:** **discover service** (registry), **tailor interface** (translation/buffering), **orchestrate** if multi-step. `[M:61]`
- **Edge cases:** duplicate booking requests, out-of-sequence messages, partner timeout → idempotency + retry/queue `[M:112]`; flood of requests exhausting resources → limit rate; data model mismatch → translate; unknown partner → binding policy to **reject and log** unacceptable bindings. `[M:62-63]`
- **SOA weaknesses to mention:** complexity, no control over service evolution, performance. `[M:185]`

---

# 6. TESTABILITY SCENARIOS (C7 · `[M:65-68]`, `[M:129-130]`)

## S6.1 "The system can't be tested without hitting the real payment gateway / sensors"
- **Mitigate:** **abstract data sources** and replaceable components (simulators), **sandbox**, **record/playback**, **specialised interfaces**, **localise state storage**; design for *rollback of test changes* to run many tests. `[M:66, M:130]`
- **Non-deterministic bugs:** **limit non-determinism** (unconstrained parallelism), **executable assertions**, capture activity leading to the fault. `[M:66-67]`
- **Which tests first?** Use the **utility tree** (high business value × high architectural impact). **Integration plan** from module dependencies. `[M:129-130]`

---

# 7. USABILITY SCENARIOS (C3 · `[M:41-44]`)
- **User makes an error** → **undo**, **cancel**, **pause/resume**; **system model** for feedback. `[M:42]`
- **User repeats the same operation on many items** → **aggregate**.
- **System should anticipate** → **task model / user model**. **Trade-off:** state for undo costs memory; models cost complexity/privacy.

---

# 8. MOBILE SCENARIOS (K · `[M:226-231]`)

## S8.1 Intermittent connectivity (doctors entering patient data) `[M:229]`
- **Store locally (SQLite content provider), sync later (service + REST)**; conflict resolution on sync; compact code; few layers.
- **Edge cases:** conflicting edits; partial sync; battery/bandwidth; security of data stored on device.

## S8.2 Courier delivery boy app — identify components `[M:231-232]`
| Function | Component |
|---|---|
| View packages to deliver | **Activity** (UI list) + **content provider** (SQLite) |
| Mark package delivered | Activity updates content provider |
| Send info to central server | **Service** (background sync) → **REST API call** to server |
- Draw: UI(Activity) ↔ SQLite content provider; Service ↔ content provider; Service → server. Mention **store-locally/sync-later** for connectivity.

## S8.3 Uber-style components `[M:231]`
UI = book-a-cab screen · **Broadcast receiver** = receives cab location from back-end and hands to UI · **Service** = sends cab location to back-end after journey starts · **Database/config file** = user data.

## S8.4 Emergency alert on device (high pulse) `[M:230]`
**Broadcast receiver** reacts to the event → service raises alarm / calls doctor → content provider supplies doctor details.

---

# 9. CLOUD, MULTI-TENANCY & DEPLOYMENT SCENARIOS

## S9.1 SaaS start-up — "what options exist for deploying in the cloud?" (GST filing / Tailspin surveys) `[M:237]`
- Options to discuss: **IaaS** (VMs, you manage OS/runtime — max control, more ops), **PaaS** (deploy code; provider manages stack), **SaaS-style multi-tenant app on a platform**; **single-tenant (multi-instance)** vs **multi-tenant (single instance)** `[M:237-238]`; degree of multi-tenancy per tier `[M:239]`.
- **Decision criteria:** isolation/security, cost per tenant, customisation, upgrade ease, tenant data confidentiality, scalability, compliance (GST data!). Full worked answer in `04_Case_Studies.md`.

## S9.2 Choosing multi-tenancy per tier `[M:239]`
| Option | Choose when |
|---|---|
| Web **multi**-tenant, App **single**, Data **multi** | web processing light, app processing heavy, data **not confidential** |
| Web **single**, App **multi**, Data **single** | web processing heavy, app processing light, data **confidential** |

## S9.3 Consistency vs availability for an e-commerce site `[M:236]`
Add-to-cart → **AP** (always accept; hide errors, reconcile later). Submit-order → **CP/consistency** (payment, shipping, reporting read the same data). *Partitions happen, so the choice is per service.*

## S9.4 Auto-scaling is too slow / provider outage `[M:216, M:236]`
Make apps resource-aware; pre-warm; replicate across zones; design for the 0.05%.

---

# 10. LAYERED-ARCHITECTURE EXERCISES — PRACTISE, THEN CHECK THE SLIDES
*The four exercises and their model answers are in your PDF (`[M:111-115]`). Attempt each on paper first (10 min each), then compare.*

| Exercise | What it is really testing | Reasoning to bring |
|---|---|---|
| **1. Departure Control System — components per layer** | placing responsibilities in presentation / business / data / **service** layers | the service layer faces *external* systems → duplicates, out-of-order messages, communication failure (retry or queue) |
| **2. Aadhaar drop-downs slow** | performance on a read-heavy static list | **cache** (copies of data) + **AJAX** incremental load (reduce demand); mention staleness and invalidation |
| **3. Hotel system for an external travel app** | designing a service layer | two operations (availability, reserve), idempotent reservation, authenticate the partner, handle duplicates/timeouts |
| **4. Logistics (container + transporter + ship)** | hiding complexity | **façade** in the business layer → fewer dependencies, one change point; risk: façade becomes a god object |

**Self-check:** did each answer name the *layer*, the *technique*, and a *cost*?

---

# 11. EVALUATION SCENARIOS (ATAM) `[M:119-123]`

## S11.1 "Evaluate this architecture / what would you find in an ATAM?"
1. Business drivers → **utility tree** → top scenarios (H,H first). `[M:121-122]`
2. For each scenario, ask the architect *how* it is satisfied; log **approach, decision, risk, non-risk, sensitivity point, trade-off point**. `[M:122]`
3. Example finding: *"Heartbeat frequency affects time to detect a failed component"* → **sensitivity point**; if too slow → **risk**.
4. Group risks → **risk themes** (e.g. "documentation insufficient", "no backup/HA strategy") with business impact. `[M:123]`
5. **Edge case:** limited time → **lightweight ATAM** (4-6 h, internal). `[M:124]`

## S11.2 "How would you prioritise testing for the new system?" → **utility tree** (value × impact). `[M:129]`

---

# 12. BIG-DATA / STREAM SCENARIOS (L · `[M:245-254]`)

## S12.1 Average time users spend on each page (30 days of click-stream) `[M:246-247]`
**Map** per node → ⟨page, (avg, #users)⟩; **Reduce** → ⟨page, weighted avg⟩. Weighted average (not simple average of averages) is the *edge case* that shows understanding. Run nightly (cheap, fault-tolerant). `[M:250-251]`

## S12.2 "Trending hashtags in seconds" / fraud detection `[M:248, M:251]`
MapReduce too slow → **stream processing (Storm + Kafka)** or **Spark**; accept higher memory cost and complex state/checkpointing. `[M:250]`

## S12.3 Cab-hailing event handling `[M:258]`
Driver app events (pickup, location, emergency) → event handlers → analysis (delay, route deviation) → messages to ops team/customer (SMS queue). Use **publish-subscribe**; **Kafka** buffer; **Storm** compute.

## S12.4 When NOT to use Hadoop `[M:248]`
Low-latency/near-real-time access, many small files, etc.

## S12.5 Choosing NoSQL for a given data shape `[M:254]`
Phone directory/stock → **key-value**; JSON records → **document**; analytics over a column (average electricity use) → **column-oriented**; fraud/social relationships → **graph**; telecom/ad response time → **in-memory**.

---

# 13. PROJECT-MANAGEMENT SCENARIOS (O · `[M:276-285]`)
- **Top-down schedule disagrees with bottom-up:** reconcile; bottom-up is typically more accurate; this becomes the development plan. `[M:276-277]`
- **Scope creep threatens schedule:** PM resists creeping functionality; **Change Control Board**; architect protects QAs. `[M:283]`
- **Global teams:** coordination critical; architecture (work-breakdown, module views) is the shared basis; **development distributability** QA. `[M:45, M:282]`
- **PM vs architect conflict:** most important working relationship — respect, coordinate, stay in spheres. `[M:277]`

---

# 14. CBAM NUMERIC SCENARIOS `[M:288-296]`
**Procedure for any numeric question:** (1) list scenarios with **weights** (top vote = 1.0, others proportional); (2) for each strategy and scenario get **U_expected** and **U_current**; (3) **b = U_expected − U_current**; (4) **B = Σ b × W**; (5) **VFC = B / C**; (6) rank by VFC; choose until budget/schedule ends; (7) sanity-check against business goals.
- **Mini example (made-up numbers to practise):** Scenario weights: S1 = 1.0, S2 = 0.5. Strategy A: S1 current 50 → expected 80 (b=30); S2 current 40 → expected 60 (b=20). B = 30×1.0 + 20×0.5 = **40**. Cost 20 → **VFC = 2.0**. Strategy B: B = 55, cost 50 → VFC = 1.1. **Choose A** despite smaller absolute benefit — *cheaper per unit of value*.
- **Edge cases:** a strategy may **reduce** utility for some scenarios (b negative → subtract); strategies interact; costs uncertain → intuition check (step 9). Use the **NASA ECS tables** as the model solution: `[M:296]`.

---

# 15. LIKELY SCENARIO QUESTIONS (inferred) → ANSWER FROM
| Likely question | Section |
|---|---|
| Server/DB crash; how to design for 99.99%? | S1.1-S1.3 |
| HDFS write/NameNode/DataNode failure | S1.5 |
| Slow page / peak-load degradation | S2.1-S2.3 |
| Insider/DoS/SQL-injection/multi-tenant leak | S3.1-S3.6 |
| New requirement/change request/vendor change | S4.1-S4.2 |
| Integrate with external partner | S5.1 |
| System not testable | S6.1 |
| Intermittent connectivity; components of courier/Uber app | S8.1-S8.3 |
| Deployment & multi-tenancy options for a SaaS | S9.1-S9.2 |
| CAP choices for cart vs order | S9.3 |
| Layered-architecture exercises (DCS, façade, service layer) | §10 |
| ATAM risks / sensitivity points | §11 |
| Click-stream average, real-time fraud, NoSQL choice | §12 |
| CBAM calculation | §14 |
