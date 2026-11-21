# PART 2 — SCENARIO-BASED QUESTIONS
*Handle a concrete situation, failure or edge case → **Diagnose → Mitigate → Recover → Prevent***
REF format `[file:page]` → see `00_READ_FIRST.md`. Concepts are in `01_Concepts.md` (cross-referenced as **C1…**).

---

# 0. THE SCENARIO-ANSWER ALGORITHM (use on every scenario question)

| Step | Do this | Write this phrase |
|---|---|---|
| 1. **Classify** | Which QA(s) does the situation threaten? (many incidents hit 2-3) | "This primarily threatens *availability* and secondarily *performance*." |
| 2. **Restate as a QA scenario** | Source · stimulus · artifact · environment · response · **measure** `[1:30-32]` | "Stimulus: DB node crashes in normal operation; response: fail over; measure: < 30 s, no data loss." |
| 3. **Diagnose** | What fault model / bottleneck / attack type? Where in the structure (module, C&C, deployment)? | "Fault of type *crash* at the data tier." |
| 4. **Mitigate now (tactics)** | Use the tactic *categories* (detect/recover/prevent; control demand/manage resources; detect/resist/react/recover) | "Detect via heartbeat; recover via passive redundancy + state resync." |
| 5. **Recover & restore** | Reintroduction, data repair, audit, rollback | "Shadow the repaired node, then reintroduce." |
| 6. **Prevent recurrence** | Architecture change, monitoring, test, process | "Add predictive monitoring; chaos test; update runbook." |
| 7. **Trade-offs & edge cases** | Cost, other QAs affected, what if mitigation itself fails | "Hot spare doubles cost; heartbeat adds load; add timeout to avoid false positives." |
| 8. **Measure success** | the response measure from step 2 | "We verify with failover drill; target 99.99%." |

> **Edge-case checklist** (examiners love these): *transient vs permanent fault · partial failure · duplicate / out-of-order messages · overload during recovery · false alarms of the detector · failure of the mitigation component itself · data inconsistency after failover · security impact of the mitigation.*

---

# 1. AVAILABILITY SCENARIOS (concept C1 · `[1:44-65]`)

## S1.1 Server / node crash during normal operation
- **Diagnose:** fault = *crash* (or *omission*); detected by missing **heartbeat/ping-echo** from the **monitor**. `[1:49]`
- **Mitigate:** fail over to **redundant spare** — *hot (active)* if zero-downtime required; *warm (passive)* for cheaper, slower; *cold* for non-critical. **Reconfigure** remaining resources; operate in **degraded mode** (drop non-critical features). `[1:52, 1:54]`
- **Recover:** repair node → run in **shadow mode** → **state resynchronisation** → reintroduce; use **escalating restart** (process → service → node). `[1:55]`
- **Prevent:** **removal from service** (planned restart before memory leaks bite), **predictive model**, **transactions**. `[1:56]`
- **Edge cases:** *split-brain* (both think they're primary → use quorum/monitor); false positive from a slow network (use timeouts + retry before failover); spare is out-of-date (warm/cold) → data loss window → state the RPO.
- **Measure:** time-to-detect, time-to-repair, availability %. `[1:45]`

## S1.2 Transient failure / flaky dependency
- Use **retry** (only for *transient* faults) with back-off; **ignore faulty behaviour** for spurious messages; **exception handling** that masks and retries; **timeout** (exception detection). `[1:53-54, 1:51]`
- **Edge case:** retries can amplify load on an already-struggling service (↓ performance) → bound retries, use **bound queue sizes** `[1:73]`; duplicate effects → idempotent operations; make the *service layer* handle **duplicate requests, out-of-sequence messages and communication failure by retry or queue-and-send-later**. `[2:47]`

## S1.3 Bad deployment / software upgrade breaks service
- **Mitigate:** **rollback** to the previous known-good state ("rollback line"); `[1:53]` **shadow** the upgraded component before activation; **in-service software upgrade** without service impact. `[1:53, 1:55]`
- **Prevent:** staged rollout, **sandbox** and regression testing `[1:162]`, update docs/code in sync `[2:82]`.

## S1.4 Cloud instance / zone failure
- **Facts:** failures are *normal* with 1000s of servers; provider keeps the cloud up but **applications must detect and correct instance failures**. `[3:132]` 99.95% still = 0.05% downtime. `[4:19]`
- **Mitigate/prevent:** auto-scaling & automatic IP reallocation `[3:124]`; **replicate across geographic zones (Netflix)** `[4:19]`; **remove single points of failure**, disposable resources `[3:133]`.
- **Edge case:** data replicated across zones → *CAP*: during a partition choose **availability (AP)** for revenue-producing paths (shopping cart) and **consistency (C)** for order submission. `[4:18]`

## S1.5 HDFS write failure cases (very exam-friendly) `[3:120-121]`
| Failing party | Detection → response |
|---|---|
| **Client** | application detects, retries; write isn't complete until committed by client |
| **NameNode** | **backup NameNode** takes over; **log file** prevents metadata loss; DataNodes hold the true block list; client detects and retries |
| **DataNode** | client or earlier DataNode in the pipeline detects → asks NameNode for a different DataNode; block already **replicated 3×**, so one failure loses nothing |
- *Tactics to name:* **active/passive redundancy, exception detection, retry, rollback, state resync**. HDFS also uses **heartbeats** from DataNodes to NameNode `[4:34]`.

## S1.6 "Broker is a single point of failure"
- Pattern weakness `[3:43]`; **fix with tactics**: *maintain multiple copies* (replicated brokers, passive redundancy) for availability; *increase resources* for performance. `[3:107]` Cost: more infrastructure, state sync between brokers.

---

# 2. PERFORMANCE SCENARIOS (C2 · `[1:66-81]`)

**Diagnosis grid:** *Is it demand (too many events) or resource (too little capacity / too much blocking)?* → choose tactic category.

## S2.1 A frequently-used screen loads slowly (Aadhaar State/District/Town) `[2:50-51]`
- **Diagnose:** large static reference data fetched and rendered every time.
- **Mitigate:** **cache** States/Districts/Towns in the back-end (**multiple copies of data**); **AJAX** to load districts/towns dynamically (**reduce demand per request**). Also client-side caching for user preferences. `[2:34-35]`
- **Edge case:** stale cache when reference data changes (invalidation policy); cache memory cost; mix client and server caching. 

## S2.2 Peak-load slowdown at the data tier
- **Mitigate:** connection pool; **multiple copies of data** (read replicas); **stored procedures**; ORM tuning; **increase resources** (vertical) or **increase concurrency / replicas** (horizontal). `[2:43, 1:72-73]`
- **Prevent:** **bound queue sizes**, **prioritise events**, **limit event response** (rate limiting), **schedule resources**. `[1:70, 1:73]`
- **Edge case:** auto-scaling too slow for sudden spikes → application must be **self-aware and proactive** about resource needs. `[3:131]` Replicas reduce read latency but raise consistency problems.

## S2.3 Event flood / overload (sensor stream, burst of requests)
- **Reduce demand:** **manage sampling rate** (accept lower fidelity), **limit event response**, **prioritise events**, **bound execution times**. `[1:70-71]`
- **Add capacity only if demand can't be cut:** increase resources/concurrency.
- **Edge case:** overload *during recovery*; priority starvation of low-priority events.

## S2.4 Latency added by "clean" design (too many intermediaries/layers)
- **Reduce overhead:** remove intermediaries on the hot path (trade **modifiability ↓**) `[1:71]`; **few layers** on mobile `[4:1]`; façade/caching to cut round trips.

## S2.5 MapReduce too slow for fraud detection `[4:39-41]`
- MapReduce is **disk-based**; real-time fraud needs sub-second → move to **Spark (in-memory)** or **stream processing (Storm + Kafka)**. Keep MapReduce for cheap nightly historical batches `[4:45]`.

---

# 3. SECURITY SCENARIOS (C4 · `[1:106-123]`)

## S3.1 Insider attempts to modify a protected table `[1:109]`
- **Resist:** identify + **authenticate + authorise** actors, **limit access** (least privilege), separate sensitive data with different access rights. `[1:112, 1:118]`
- **Detect:** log/audit all access to sensitive data; verify message integrity.
- **React:** **revoke access**, lock account after repeated failures, **inform actors**. `[1:114]`
- **Recover:** **audit trail** to identify the attacker; restore data (from backup/rollback) within target time (e.g. one day). `[1:115]`
- **Edge case:** the attacker is an *authorised* user — authentication alone isn't enough; log files themselves must be protected.

## S3.2 Denial-of-service attempt
- **Detect:** compare traffic to historic DoS signatures; **monitor** (also detects congestion). `[1:111, 1:49]`
- **Resist:** **limit exposure**, **limit access**, firewall filtering, **separate entities**; **bound queues**/**limit event response**. `[1:113, 1:70]`
- **Recover:** availability recovery tactics (spares, reconfiguration). **Edge case:** ping/echo itself can be abused (**ping flood**) → rate-limit health checks. `[3:108]`

## S3.3 SQL injection against a web form `[2:45]`
- Cause: concatenating user input into SQL (`' OR 'x'='x'` returns all rows).
- **Fix (data layer):** **parameterised queries**, stored procedures, input validation, least-privilege DB account, ORM. **Detect:** log + intrusion detection. Cross-ref: *encrypt data, limit exposure*.

## S3.4 Data in transit between browser and site
- Use **TLS**: agree version/algorithms → authenticate via **certificates** → shared **symmetric key** (fast). Provides **privacy + integrity**. `[4:60-61]`

## S3.5 "Let users log in with Google" / "app wants to read user's contacts"
- **OpenID = authenticate** (redirect to provider, return credentials not password). **OAuth = authorise** an app to access your data via a **token**. `[4:62-65]` Use **LDAP** for fast read-mostly validation of millions of users; **IAM** for roles/groups/compliance. `[4:66-67]`

## S3.6 Multi-tenant leakage / VM escape `[3:130, 4:24]`
- Risks: shared-table with org-ID column and poor design → inadvertent sharing; **VM escape**; side-channel; DoS from a neighbour.
- **Mitigate:** tenant isolation in data model (separate schemas/instances for sensitive data), **separate entities**, encryption, hypervisor hardening, access checks on every query. **Trade-off:** more isolation = less sharing = higher cost. `[4:25-26]`

## S3.7 Perimeter design for a web app
- **Firewall + DMZ:** public-facing servers in DMZ between two firewalls; attacker must pass two barriers; use packet filtering, application-layer filtering, NAT. `[4:68-71]`

---

# 4. MODIFIABILITY & CHANGE SCENARIOS (C5 · `[1:124-139]`)

## S4.1 "Marketing wants a new payment method / fee change next week"
- **Diagnose:** change type & likelihood — *what can change, how likely, when, by whom?* `[1:124]`
- **Mitigate:** if anticipated → **defer binding** (configuration, rules/workflow engine) so the change needs *no code change*; `[2:14, 2:37]` if not → **encapsulate** the payment module behind an interface, **use an intermediary/adapter**.
- **Measure:** cost (effort, artifacts touched, side-effects). `[1:125]`
- **Edge cases:** the change touches several modules (non-local change) → *split module*, *increase cohesion*, *refactor duplicates*, *abstract common services*. `[1:127-130]`

## S4.2 "Replace the database / port to another platform / upgrade a vendor product"
- **Principle:** never depend on a specific version of a commercial tool; isolate behind a data-access layer/ORM; **layers** give portability. `[1:22, 1:11, 2:44]`
- **Edge case:** vendor lock-in through *choice of technology* decisions (checklist) `[1:138]`.

## S4.3 Code has drifted from the architecture (layer violations)
- **Diagnose:** reconstruct the as-built architecture (raw view extraction → DB → view fusion → analysis). `[2:88-92]`
- **Fix:** conformance checks (adjacent-layer calls only; DB access via data layer), **architecturally evident coding style**, frameworks, code templates, code reviews, folder-per-layer, update docs at release. `[2:78-83]`
- **Edge case:** documentation lies → mark obsolete parts "no longer applicable" to keep trust in the rest `[2:82]`.

## S4.4 Legacy system, no documentation, must migrate (mainframe → web)
- **Use architecture reconstruction** to find reusable components (logging, security) and recover layers; then re-architect target. `[2:88]`

---

# 5. INTEROPERABILITY SCENARIOS (C6 · `[1:140-154]`)

## S5.1 External partner (e.g., Makemytrip) must query/book our hotel rooms `[2:52-53]`
- **Answer:** build a **service layer** exposing *GetRoomAvailability(from,to) → rooms by type* and *ReserveRoom(count,type,from,to) → success/failure*.
- **Tactics:** **discover service** (registry), **tailor interface** (translation/buffering), **orchestrate** if multi-step. `[1:145-146]`
- **Edge cases:** duplicate booking requests, out-of-sequence messages, partner timeout → idempotency + retry/queue `[2:47]`; flood of requests exhausting resources → limit rate; data model mismatch → translate; unknown partner → binding policy to **reject and log** unacceptable bindings. `[1:151-152]`
- **SOA weaknesses to mention:** complexity, no control over service evolution, performance. `[3:69]`

---

# 6. TESTABILITY SCENARIOS (C7 · `[1:155-171]`, `[2:84-87]`)

## S6.1 "The system can't be tested without hitting the real payment gateway / sensors"
- **Mitigate:** **abstract data sources** and replaceable components (simulators), **sandbox**, **record/playback**, **specialised interfaces**, **localise state storage**; design for *rollback of test changes* to run many tests. `[1:161-162, 2:87]`
- **Non-deterministic bugs:** **limit non-determinism** (unconstrained parallelism), **executable assertions**, capture activity leading to the fault. `[1:163, 1:164]`
- **Which tests first?** Use the **utility tree** (high business value × high architectural impact). **Integration plan** from module dependencies. `[2:85-86]`

---

# 7. USABILITY SCENARIOS (C3 · `[1:82-95]`)
- **User makes an error** → **undo**, **cancel**, **pause/resume**; **system model** for feedback. `[1:87-88]`
- **User repeats the same operation on many items** → **aggregate**.
- **System should anticipate** → **task model / user model**. **Trade-off:** state for undo costs memory; models cost complexity/privacy.

---

# 8. MOBILE SCENARIOS (K · `[4:1-17]`)

## S8.1 Intermittent connectivity (doctors entering patient data) `[4:7]`
- **Store locally (SQLite content provider), sync later (service + REST)**; conflict resolution on sync; compact code; few layers.
- **Edge cases:** conflicting edits; partial sync; battery/bandwidth; security of data stored on device.

## S8.2 Courier delivery boy app — identify components `[4:13-15]`
| Function | Component |
|---|---|
| View packages to deliver | **Activity** (UI list) + **content provider** (SQLite) |
| Mark package delivered | Activity updates content provider |
| Send info to central server | **Service** (background sync) → **REST API call** to server |
- Draw: UI(Activity) ↔ SQLite content provider; Service ↔ content provider; Service → server. Mention **store-locally/sync-later** for connectivity.

## S8.3 Uber-style components `[4:12]`
UI = book-a-cab screen · **Broadcast receiver** = receives cab location from back-end and hands to UI · **Service** = sends cab location to back-end after journey starts · **Database/config file** = user data.

## S8.4 Emergency alert on device (high pulse) `[4:11]`
**Broadcast receiver** reacts to the event → service raises alarm / calls doctor → content provider supplies doctor details.

---

# 9. CLOUD, MULTI-TENANCY & DEPLOYMENT SCENARIOS

## S9.1 SaaS start-up — "what options exist for deploying in the cloud?" (GST filing / Tailspin surveys) `[4:20-21]`
- Options to discuss: **IaaS** (VMs, you manage OS/runtime — max control, more ops), **PaaS** (deploy code; provider manages stack), **SaaS-style multi-tenant app on a platform**; **single-tenant (multi-instance)** vs **multi-tenant (single instance)** `[4:22-23]`; degree of multi-tenancy per tier `[4:26]`.
- **Decision criteria:** isolation/security, cost per tenant, customisation, upgrade ease, tenant data confidentiality, scalability, compliance (GST data!). Full worked answer in `04_Case_Studies.md`.

## S9.2 Choosing multi-tenancy per tier `[4:28-29]`
| Option | Choose when |
|---|---|
| Web **multi**-tenant, App **single**, Data **multi** | web processing light, app processing heavy, data **not confidential** |
| Web **single**, App **multi**, Data **single** | web processing heavy, app processing light, data **confidential** |

## S9.3 Consistency vs availability for an e-commerce site `[4:18]`
Add-to-cart → **AP** (always accept; hide errors, reconcile later). Submit-order → **CP/consistency** (payment, shipping, reporting read the same data). *Partitions happen, so the choice is per service.*

## S9.4 Auto-scaling is too slow / provider outage `[3:131-132, 4:19]`
Make apps resource-aware; pre-warm; replicate across zones; design for the 0.05%.

---

# 10. LAYERED-ARCHITECTURE EXERCISES (with model answers) `[2:48-55]`
1. **DCS (Departure Control System) layers** `[2:48-49]`: Presentation (check-in UI, boarding-pass kiosk) · Business (check-in logic, seat allocation, baggage/weight & balance rules) · Data (passenger/flight/seat DB access) · **Service layer** (interfaces to airline partners/airport systems; handles duplicate/out-of-sequence/comm-failure). 
2. **Aadhaar drop-downs** → cache + AJAX `[2:50-51]`.
3. **Hotel reservation + external apps** → service layer with GetRoomAvailability / ReserveRoom `[2:52-53]`.
4. **Logistics (container, transporter, ship)** → **façade in business layer** with one API *place shipping request*; façade calls modules for container/transporter/ship search `[2:54-55]`. Benefit: hides complexity, **reduces coupling**, one place to change.

---

# 11. EVALUATION SCENARIOS (ATAM) `[2:61-77]`

## S11.1 "Evaluate this architecture / what would you find in an ATAM?"
1. Business drivers → **utility tree** → top scenarios (H,H first). `[2:67, 2:70]`
2. For each scenario, ask the architect *how* it is satisfied; log **approach, decision, risk, non-risk, sensitivity point, trade-off point**. `[2:71]`
3. Example finding: *"Heartbeat frequency affects time to detect a failed component"* → **sensitivity point**; if too slow → **risk**.
4. Group risks → **risk themes** (e.g. "documentation insufficient", "no backup/HA strategy") with business impact. `[2:74]`
5. **Edge case:** limited time → **lightweight ATAM** (4-6 h, internal). `[2:75-76]`

## S11.2 "How would you prioritise testing for the new system?" → **utility tree** (value × impact). `[2:85]`

---

# 12. BIG-DATA / STREAM SCENARIOS (L · `[4:31-59]`)

## S12.1 Average time users spend on each page (30 days of click-stream) `[4:35-37]`
**Map** per node → ⟨page, (avg, #users)⟩; **Reduce** → ⟨page, weighted avg⟩. Weighted average (not simple average of averages) is the *edge case* that shows understanding. Run nightly (cheap, fault-tolerant). `[4:43-45]`

## S12.2 "Trending hashtags in seconds" / fraud detection `[4:45, 4:39]`
MapReduce too slow → **stream processing (Storm + Kafka)** or **Spark**; accept higher memory cost and complex state/checkpointing. `[4:44]`

## S12.3 Cab-hailing event handling `[4:56]`
Driver app events (pickup, location, emergency) → event handlers → analysis (delay, route deviation) → messages to ops team/customer (SMS queue). Use **publish-subscribe**; **Kafka** buffer; **Storm** compute.

## S12.4 When NOT to use Hadoop `[4:38]`
Low-latency/near-real-time access, many small files, etc.

## S12.5 Choosing NoSQL for a given data shape `[4:49-54]`
Phone directory/stock → **key-value**; JSON records → **document**; analytics over a column (average electricity use) → **column-oriented**; fraud/social relationships → **graph**; telecom/ad response time → **in-memory**.

---

# 13. PROJECT-MANAGEMENT SCENARIOS (O · `[4:81-94]`)
- **Top-down schedule disagrees with bottom-up:** reconcile; bottom-up is typically more accurate; this becomes the development plan. `[4:81-83]`
- **Scope creep threatens schedule:** PM resists creeping functionality; **Change Control Board**; architect protects QAs. `[4:89]`
- **Global teams:** coordination critical; architecture (work-breakdown, module views) is the shared basis; **development distributability** QA. `[4:88, 1:96]`
- **PM vs architect conflict:** most important working relationship — respect, coordinate, stay in spheres. `[4:84]`

---

# 14. CBAM NUMERIC SCENARIOS `[4:98-116]`
**Procedure for any numeric question:** (1) list scenarios with **weights** (top vote = 1.0, others proportional); (2) for each strategy and scenario get **U_expected** and **U_current**; (3) **b = U_expected − U_current**; (4) **B = Σ b × W**; (5) **VFC = B / C**; (6) rank by VFC; choose until budget/schedule ends; (7) sanity-check against business goals.
- **Mini example (made-up numbers to practise):** Scenario weights: S1 = 1.0, S2 = 0.5. Strategy A: S1 current 50 → expected 80 (b=30); S2 current 40 → expected 60 (b=20). B = 30×1.0 + 20×0.5 = **40**. Cost 20 → **VFC = 2.0**. Strategy B: B = 55, cost 50 → VFC = 1.1. **Choose A** despite smaller absolute benefit — *cheaper per unit of value*.
- **Edge cases:** a strategy may **reduce** utility for some scenarios (b negative → subtract); strategies interact; costs uncertain → intuition check (step 9). Use the **NASA ECS tables** as the model solution: `[4:111-116]`.

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
