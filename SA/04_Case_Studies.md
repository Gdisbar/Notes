# PART 4 — CASE STUDIES
*End-to-end analysis under real constraints → **a cohesive architecture tied to business & system goals***
REF format `[file:page]` → see `00_READ_FIRST.md`. Concepts: `01_Concepts.md`; scenarios: `02_...`; decisions: `03_...`.

---

# 1. THE CASE-STUDY BLUEPRINT (write these headings first)

| # | Heading | What to write (1-3 lines each) | Source |
|---|---|---|---|
| 1 | **Business goals & context** | Why does the system exist? Use "X to Y by when". Who are the stakeholders and what does each care about? | `[2:10-11]` |
| 2 | **Constraints** | Fixed decisions: regulation (GST, GDPR), platform, budget, time-to-market, existing systems | `[2:3]` |
| 3 | **ASRs / utility tree** | 4-6 scenarios with measures and (H/M/L, H/M/L) ratings | `[2:12-14]` |
| 4 | **Architectural drivers → candidate patterns & tactics** | for each top ASR: tactic categories + pattern; walk the **7 design-decision categories** | `[1:35-43]`, `[3:1-114]` |
| 5 | **Structure & views** | module (layers), C&C (runtime), allocation (deployment); pick 4+1 views by concern | `[2:16-31]` |
| 6 | **Technology & deployment** | cloud model (IaaS/PaaS/SaaS), tenancy, mobile type, data store (SQL/NoSQL), security tech, big-data stack | `[3:115-139]`, `[4:1-80]` |
| 7 | **Trade-offs & risks** | what each choice costs; top 3 risks; sensitivity points | `[2:63-64, 2:71]` |
| 8 | **Evaluation** | how you'll validate: ATAM (full/lightweight), prototype spikes, tests, conformance | `[2:56-77]` |
| 9 | **Economics** | which strategies first — benefit vs cost (CBAM VFC); phased delivery | `[4:95-116]` |
| 10 | **Delivery & governance** | agile vs up-front, team structure from modules, metrics, governance | `[1:172-187]`, `[4:81-94]` |

**Tie-back sentence template:** "Because the business goal is **…**, the dominant ASR is **… (measure)**; therefore we chose **pattern/tactic** … which costs **…**, and we mitigate by …"

**Marks usually go for:** (a) explicit ASRs with measures, (b) named patterns/tactics, (c) at least one diagram (layers/deployment), (d) trade-off + risk statements, (e) link back to business goals.

---

# 2. DOMAIN → TYPICAL ASRs → TYPICAL ARCHITECTURE (use for unseen cases)

| Domain | Dominant ASRs | Typical patterns & tactics | Technology notes |
|---|---|---|---|
| **E-commerce / retail** | availability (cart), performance at peak, security (payments), modifiability (offers) | Layers + service layer; cache; **AP for cart, C for orders** `[4:18]`; pub-sub for events; auto-scaling; transactions | cloud, NoSQL for catalogue/cart, RDBMS for orders, TLS, OAuth |
| **Banking / payments / GST filing** | security, integrity, availability, auditability, compliance | Layers + façade; **audit**, authenticate/authorise, encrypt, **active redundancy**, transactions, rollback | single-tenant or strongly isolated data tier; TLS; LDAP/IAM; DMZ |
| **Healthcare (patient records, alerts)** | availability, security/privacy, performance (<0.75 s), interoperability, offline mobile | Layers; **service layer** for external systems; **store-local-sync-later**; pub-sub/broadcast for alerts; audit | mobile + SQLite, REST, TLS, HIPAA |
| **Ride-hailing / logistics** | real-time performance, availability, scalability, geo-distribution | **stream processing**, pub-sub, broker/queue (Kafka), microservices, replicate/zone | mobile (hybrid), Storm/Spark, key-value store |
| **Social / analytics / logs** | throughput, volume, cost, fault tolerance | **Map-Reduce** batch + stream for real-time; HDFS 3× replication; NoSQL (graph for relationships) | Hadoop, Spark, Hive |
| **Smart devices / IoT / smart grid** | throughput (100Ks events/s), latency, reliability, security | stream processing, pub-sub, shared-data, edge filtering | Kafka+Storm, time-series/NoSQL |
| **E-governance (Aadhaar-like)** | scalability, availability, performance on static data, security/privacy | caching + AJAX, layers, replicas, auth/authorise, audit | CDN/cache, DMZ |
| **Enterprise integration (hotel/airline partners)** | interoperability, security, idempotency | **SOA/service layer**, orchestrate, tailor interface, queue-and-retry | REST/SOAP, ESB |
| **Platform / product line / OS-like** | modifiability, portability, extensibility | **Microkernel**, reflection, layers, plug-ins | — |
| **ML-enabled apps (insurance claims)** | accuracy, latency, scalability, explainability | event-driven pipeline (store → trigger → ML → notify); **pipes & filters** | cloud functions + ML service |

---

# 3. WORKED CASES FROM THE COURSE

> Each case: **Problem → ASRs → Architecture → Trade-offs/risks → Evaluation/business tie.** Use as a model answer or adapt.

## CASE 1 — Departure Control System (DCS): layered design `[2:48-49]`
- **Problem:** airline check-in/boarding system integrating with airport and airline-partner systems; must be correct, fast at the counter, and extensible.
- **ASRs:** availability (counters can't stop), performance (<few seconds per check-in), modifiability (new rules/airlines), interoperability (partners), security (passenger data).
- **Architecture:** **Layers** — *Presentation* (counter/kiosk UI) · *Business* (check-in, seat allocation, baggage & weight rules; **façade** hides subsystems; **workflow/rules engine** for changing rules) · *Service layer* (APIs to partner/airport systems; handle **duplicate requests, out-of-order messages, comms failure via retry/queue**) · *Data* (DB with **connection pool**, parameterised SQL, replicas). Cross-cutting aspects: **logging, audit, security** (AOP). `[2:32-47]`
- **Tactics:** warm/hot spare for availability; client+server caching; retry for transient faults.
- **Trade-offs:** more layers ↓ latency; service-layer retries can duplicate work → idempotent operations.
- **Evaluation:** utility tree with (H,H) scenario "check-in completes in <3 s at peak with one node failed"; lightweight ATAM; conformance checks (no layer skipping). `[2:78-83]`
- ⚠ The exact component diagram is on `[2:49]` — check it for names used in class.

## CASE 2 — Courier delivery mobile app `[4:13-15]`
- **Problem:** delivery boy views packages, marks delivered, and the central server is informed; connectivity is intermittent.
- **ASRs:** usability (large buttons, few steps), performance (compact code, few layers), availability offline (**store locally, sync later**), consistency on sync.
- **Architecture (Android):** **Activity** (package list / mark-delivered UI) ↔ **Content provider (SQLite)**; **Service** (background sync) → **REST call to server**; optional **broadcast receiver** for connectivity-change events. `[4:8-11]`
- **Decisions:** native Android (device GPS/camera, performance) vs hybrid (shared code with iOS) — choose per fleet device mix `[4:2-5]`.
- **Edge cases:** conflicting updates; sync retry; data-at-rest security.
- **Business tie:** faster, reliable proof of delivery → customer trust, fewer disputes.

## CASE 3 — Smart-Health: doctors entering patient data + emergency alerts `[4:7, 4:11]`
- **ASRs:** offline entry (connectivity), **real-time alert** on high pulse, privacy/security, interoperability with doctor's phone.
- **Architecture:** mobile app with **local SQLite → sync to server (REST)**; **broadcast receivers** (high-pulse event) → **services** (alarm, recorded message, call doctor) → **content provider** (doctor details). Back-end: pub-sub for events, audit log.
- **Tactics:** detect (condition monitoring on pulse), react (alert/inform actors), recover (retry sync), audit; TLS + encryption of stored data.
- **Trade-offs:** battery vs polling frequency; false alarms vs missed alerts (sensitivity point).

## CASE 4 — Tailspin: customer-survey & analysis SaaS on the cloud (Azure) `[4:21-26, 3:134]`
- **Problem:** clients create/launch surveys; participants answer; the system analyses and presents results; offered as **SaaS**.
- **Options to evaluate `[4:22-23]`:** (a) **multi-instance, single-tenant** per client; (b) **single-instance, multi-tenant** shared app/data; (c) hybrid per tier.
- **Decision criteria:** tenant data confidentiality, cost per tenant (survey workloads are utilitarian → shareable), customisation, upgrade effort, scale of participants (bursty), compliance.
- **Recommended architecture (typical answer):** **multi-tenant web tier** (light), **multi-tenant app tier with tenant context**, **data tier multi-tenant with tenant-ID partitioning, with an option to isolate premium/confidential tenants** (per-tier mixing `[4:28-29]`); worker roles/queues for analysis (decouple, auto-scale); **auto-scaling** with self-aware app `[3:131]`; **replicate across zones** `[4:19]`.
- **Security:** inadvertent sharing (org-ID column discipline), VM escape, DoS `[3:130, 4:24]` → tenant filters on every query, encryption, separate entities for sensitive tenants.
- **Trade-offs:** multi-tenancy cuts help-desk, upgrade, and infra costs `[3:119]` but weakens isolation and customisation. **Boundary:** if a client demands strict data residency/isolation → single-tenant instance.
- **Business tie:** lower cost per subscriber → competitive price; faster releases.

## CASE 5 — GST tax-return filing SaaS `[4:20]`
- **Problem:** SMEs input/upload Excel data; system processes and **files returns to the government GST system** on their behalf.
- **ASRs:** **security & integrity** (financial, compliance), **availability near filing deadlines** (peak), **interoperability** with government system, **auditability**, usability (Excel upload).
- **Deployment options:** IaaS vs PaaS vs multi-tenant SaaS; public vs private/hybrid cloud `[3:116-117]`. *Recommended:* PaaS/cloud-native with **autoscaling** for deadline spikes; **single-tenant or strongly isolated data tier** (confidential financial data `[4:29]`); web tier shared.
- **Architecture:** layers; **service layer** for the government API (idempotent submissions, retries/queue if GST portal is down `[2:47]`); **workflow engine** for filing steps; **audit** every submission; TLS; IAM/roles per business; backups.
- **Trade-offs:** isolation ↑ cost; queue-and-retry ↑ latency but ↑ reliability.
- **Evaluation:** ATAM scenario "deadline-day load ×10 with portal timeouts" ; CBAM to prioritise HA vs features.

## CASE 6 — Amazon-style e-commerce: availability vs consistency `[4:18-19]`
- **Problem:** scale needs partitioning → CAP forces C vs A **per service**.
- **Decisions:** **shopping cart = high availability (AP)** — always accept add-to-cart, hide errors, reconcile later (revenue-producing); **order submission = consistency (C)** — credit card, shipping, reporting access the same data simultaneously.
- **Plus:** replicate across geographic zones (Netflix) to design for the **0.05%** unavailability; auto-scaling; remove single points of failure; caching `[3:133]`.
- **Trade-offs:** eventual consistency for carts can cause stale/duplicate items; strict order path may reject during partitions → degrade gracefully.

## CASE 7 — Click-stream / log analytics platform `[4:35-45]`
- **Problem:** analyse every user click over 30 days across thousands of users (e.g., average time per page; daily active users; trending hashtags).
- **Architecture:** **HDFS** (blocks ~128 MB, 3× replication, NameNode metadata + backup) + **MapReduce** batch (Map → ⟨page,(avg,#users)⟩, Reduce → weighted average) → **Hive/BI**. For *real-time* needs add **Kafka + Storm/Spark Streaming**. NoSQL (key-value/document) for serving results. `[4:32-34, 4:56-59]`
- **Decisions:** MapReduce for cheap, fault-tolerant nightly batches; stream/in-memory when seconds matter `[4:44-45]`.
- **Risks:** small-files problem, NameNode memory, stream state complexity.
- **Business tie:** insights for product decisions at lowest unit cost.

## CASE 8 — Cab-hailing (Ola/Uber) real-time platform `[4:12, 4:56-59]`
- **ASRs:** low-latency location updates, availability, scalability, emergency handling.
- **Mobile:** UI (book cab), **broadcast receiver** (cab location from backend → UI), **service** (send location after journey starts), local DB/config. `[4:12]` Hybrid app (Uber is hybrid `[4:4]`).
- **Back-end:** driver/customer APIs → **event handlers** → **Kafka** → **Storm** analysis (delay, deviation, emergency) → ops messages + **SMS queue**; key-value store for live state; **geo-replicated** for availability.
- **Trade-offs:** stream processing = complexity and memory vs sub-second reaction; AP data store with eventual consistency for location.

## CASE 9 — Insurance claim image classification (ML) `[4:79-80, 2:40]`
- **Flow:** image stored → trigger function → call ML service → result + image metadata stored → notification; claims processed by a **workflow engine** (activities invoked step by step, manual or automated). 
- **ASRs:** scalability (burst of claims), accuracy, audit, security of PII.
- **ML steps:** problem → features → model (supervised classification) → train-test-validate → iterate. `[4:79]`
- **Pattern view:** **pipes & filters**/event-driven pipeline with a human-in-the-loop for low-confidence results.

## CASE 10 — NASA ECS: choosing architectural strategies with CBAM `[4:111-116]`
- **Problem:** 10 candidate strategies (e.g. order persistence, order chunking, bundling, reassignment, retry, notification, granule-level tracking) with limited budget.
- **Method (state the 9 steps `[4:102-110]`):** collate & refine scenarios → vote (weights) → utilities → expected response per strategy → **interpolate utility** → **benefit = Σ(Δutility × weight)** → **cost** → **VFC = benefit/cost** → rank → confirm with intuition.
- **Reading the results:** `[4:112]` utility tables, `[4:114]` total benefits, `[4:116]` cost, VFC and rank — **the highest-benefit strategy is not always the top-ranked; the cheap, good-benefit one wins on VFC.**
- **Tie-back:** spend budget top-down by VFC until exhausted; the rank order "roughly follows the order in which strategies were proposed" (architect intuition validated).

## CASE 11 — WebArrow web-conferencing: agile architecting `[1:183-185]`
- **Problem:** demanding QAs (scalability, latency) in an agile project.
- **Approach:** work **top-down** (structures to meet QA trade-offs) *and* **bottom-up** (implementation/environment constraints) simultaneously; use **spikes** to answer: distributed DB vs flat files (latency)? mod_perl vs Perl (scalability)? effort to convert? participants per meeting server? DB-to-meeting-server ratio?
- **Lesson:** evidence-based trade-offs; incremental commitment; refactor when debt rises. `[1:185-187]`

## CASE 12 — Logistics (shipping & containers): hide complexity `[2:54-55]`
- **Problem:** place a shipping request needing three modules (nearest empty container, best transporter, ship with spare capacity).
- **Architecture:** **façade** in the **business layer** exposing *PlaceShippingRequest*; façade orchestrates the three modules; **service layer** exposes it to partners; workflow engine for multi-step approval.
- **Benefits:** reduced coupling, single change point, simpler client; **cost:** façade can become a god-object; extra hop.

## CASE 13 — Hotel reservation + Makemytrip `[2:52-53]`
- **Architecture:** **service layer** with *GetRoomAvailability* and *ReserveRoom*; handle duplicate/out-of-sequence/comms-failure; secure (authenticate partner, rate-limit); idempotent reservation IDs; SOA/REST connector `[3:67-68]`.

## CASE 14 — Healthcare utility tree (patient account system) `[2:13-14]`
- **Scenarios:** update patient address under peak load <0.75 s **(H,M)**; double peak load <4 s **(L,M)**; 150 normalised transactions/s **(M,M)**; hospital fee change by configuration in 1 day, no code change **(H,L)**; maintainer fixes bug ≤3 person-days **(H,M)**; reporting metadata change in 4 person-hours **(M,L)**.
- **Architecture response:** performance → caching, replicas, concurrency, bounded queues; configurability → **defer binding via configuration/rules engine**; maintainability → modules with encapsulation and metadata-driven reports.
- **Use:** the tree also drives **test prioritisation** and **ATAM step 5**. `[2:85, 2:70]`

---

# 4. TEMPLATE FOR AN UNSEEN CASE (fill-in)

```
Business goal:  <system> must <X→Y by when> so that <business value>.
Stakeholders:   end users (...), operators (...), regulators (...), developers (...)
Constraints:    <regulation>, <platform>, <budget>, <time>
Top ASRs (H,H first):
  1. <QA> — <source> <stimulus> on <artifact> in <environment> → <response> within <measure>
  2. ...
Architecture:
  Style/pattern:  <Layers / Broker / Pub-Sub / Microkernel / MVC / Map-Reduce> because <driver>
  Tactics:        availability → ...; performance → ...; security → ...; modifiability → ...
  Structures:     module (layers/decomposition), C&C (runtime), allocation (deployment, tiers)
  Data:           <SQL/NoSQL type> with <CAP choice> per service
  Deployment:     <IaaS/PaaS/SaaS>, <public/hybrid>, tenancy <single/multi>, mobile <native/web/hybrid>
Trade-offs:     <choice> gives <benefit> at cost of <cost>
Risks:          1) ... 2) ... 3) ...   (sensitivity/trade-off points)
Evaluation:     lightweight ATAM on top scenarios; spike for <uncertain item>; conformance checks
Economics:      rank strategies by VFC; deliver in phases: <MVP>, <next>
Governance:     modules → teams; metrics (size, schedule deviation, defects); change control
```

---

# 5. DIAGRAM CHEAT-SHEET (sketch these quickly)
- **Layered:** Presentation → Business (façade, workflow) → Service layer (to externals) → Data; cross-cutting bar (security, logging, operations) on the side `[2:46]`.
- **Deployment (multi-tier/cloud):** Client → Load balancer → Web tier → App tier → DB (replicated across zones) + queue/stream + cache; DMZ between two firewalls `[4:69]`.
- **Mobile:** Activity ↔ Content provider (SQLite) ; Service → REST → Server ; Broadcast receiver → Service `[4:9-11]`.
- **MapReduce/HDFS:** Client → NameNode (metadata) ; DataNodes (3× blocks) ; Map tasks → shuffle → Reduce `[3:120, 4:35-37]`.
- **Stream:** Producers → Kafka → Storm/Spark → stores/alerts `[4:56-59]`.
- **ATAM flow:** business drivers → utility tree → scenarios → approaches → risks/non-risks/sensitivity/trade-offs → risk themes `[2:66-74]`.

---

# 6. LIKELY CASE-STUDY QUESTIONS (inferred) → ANSWER FROM
| Likely question | Section |
|---|---|
| Design a layered architecture for an airline/hotel/logistics/banking system | Cases 1, 12, 13 + §1 blueprint |
| Mobile app architecture for delivery/health/cab | Cases 2, 3, 8 |
| Deploy a SaaS on the cloud (multi-tenant? which cloud model?) | Cases 4, 5 |
| E-commerce availability vs consistency | Case 6 |
| Big-data/real-time analytics platform | Cases 7, 8 |
| ML-enabled application architecture | Case 9 |
| Choose strategies under budget with CBAM | Case 10 |
| Agile project with demanding QAs | Case 11 |
| Evaluate an architecture / write a utility tree | Case 14 + `01_Concepts.md` H2 |
