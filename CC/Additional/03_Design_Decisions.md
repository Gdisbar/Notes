# PART 3 — DESIGN-DECISION QUESTIONS
*Choose between options → **explicit trade-offs · selection criteria · boundaries (when NOT to use)***
REF format `[M:page]` (page of your compressed PDF `Cloud_Computing_Slides.pdf`, two slides per page) → see `00_READ_FIRST.md`. Concepts in `01_Concepts.md` (**A…N**).

---

# 0. HOW TO ANSWER ANY "WHICH ONE / WHY" QUESTION

1. **Drivers first** — state the 2-3 forces: *control, cost (CapEx/OpEx, steady vs spiky), time-to-market, isolation/compliance, portability, skills, availability target.*
2. **Options** — 2-3 realistic candidates.
3. **Criteria table** — compare only on the drivers.
4. **Choose + because + at the cost of.**
5. **Boundary** — "this flips if load becomes steady/very high, if compliance demands isolation, if portability is mandatory…"

**Selection criteria menu:** control · operational effort · cost model & break-even · scalability/elasticity · availability/RTO · security/isolation · compliance/residency · portability/lock-in · team skills · latency/locality · maturity of tooling.

---

# 1. DEPLOYMENT MODEL: PUBLIC vs PRIVATE vs HYBRID vs COMMUNITY (+ MULTI-CLOUD) `[M:24-28]`

| Criterion | Public | Private | Hybrid | Community |
|---|---|---|---|---|
| Cost model | pay-per-use, no capex | high capex/opex | mixed | shared among members |
| Control / customisation | low | **highest** | selective | agreed governance |
| Scalability | **near-unlimited** | finite (capacity visible) | burst to public | limited |
| Security/compliance | shared responsibility | **strongest control** | sensitive data stays private | common compliance needs |
| Complexity | low | medium-high | **integration complexity** | governance overhead |
| Choose when | variable load, speed, startups, dev/test | regulated data, predictable load, need customisation | sensitive data + elastic front-end; gradual migration; **cloud bursting** | several organisations with the same regulatory/security needs |
- **Hybrid vs multi-cloud `[M:303]`:** hybrid = private + public; multi-cloud = ≥2 *public* providers. **Multi-cloud** pros: negotiating leverage, best-of-breed, less single-vendor dependence; cons: operational complexity (different consoles/APIs), network & cost overhead, skills × N `[M:28]`. *Both can coexist.*
- **Boundary:** don't go multi-cloud "for resilience" without the skills/tools; a multi-AZ single provider is often enough.

---

# 2. SERVICE MODEL: IaaS vs CaaS vs PaaS vs FaaS vs BaaS vs SaaS `[M:159-182]`

| | IaaS | CaaS | PaaS | FaaS | BaaS | SaaS |
|---|---|---|---|---|---|---|
| Control | highest | high | medium | low-medium | low | lowest |
| Ops effort | high (patch, scale, backups) | medium (orchestrator, nodes) | medium | low | low | lowest |
| Portability | best | good (containers) | medium | low | low | lowest |
| Scaling | you build | orchestrator | platform rules | automatic per event, to zero | vendor | vendor |
| Billing | VM-hour | node/pod | plan/instance-hours | per invocation | usage | subscription |
| Risk | misconfig | complexity | deprecation, lock-in | cold start, 15-min limit, cost at scale | lock-in | borrowed availability, exit |
- **Decision path `[M:182]`:** standard function exists? → SaaS. Need backend without building? → BaaS. Need OS/kernel/hardware control (custom kernels, GPU/HPC, licensed OS images)? → IaaS. Event-driven, short, spiky? → FaaS. Otherwise → PaaS (or CaaS for portability).
- **PaaS fits / doesn't `[M:163]`:** ✔ standard stacks, small teams, variable traffic, dev/test; ✘ custom kernels, specialised networking/GPU, steady very high load (VM cost wins), hard portability without containers.
- **SaaS: buy vs build `[M:175-176]`:** SaaS cheaper for ~4 years then ownership wins (200 users: $240K/yr vs $350K + $150K/yr); standard processes → SaaS; embedded/offline/ultra-low-latency/sovereignty → no.
- **Mix per component** (FreshCart `[M:182]`): storefront → PaaS; image resize on upload → FaaS; login for prototype → BaaS; support desk → SaaS; database → managed service.
- **Boundary:** serverless is *not* for long-running jobs or heavy steady load; PaaS lock-in when using proprietary services.

---

# 3. VIRTUALIZATION APPROACH `[M:47-51, M:59]`

| | Emulation | Binary translation (full) | Paravirtualization | Hardware-assisted |
|---|---|---|---|---|
| Unmodified guest OS | ✔ | ✔ | ✘ (open-source only) | ✔ |
| Different guest ISA | ✔ | ✘ | ✘ | ✘ |
| Performance | lowest | medium | high | **high** |
| Complexity | medium | **high** (translator) | guest source change | simple |
| Needs special CPU | ✘ | ✘ | ✘ | **✔** (VT-x/AMD-V) |
| Use when | cross-architecture testing (ARM on x86) | legacy hardware without VT | Linux guests on Xen, I/O performance | **default today** (KVM, Hyper-V) |
- **Selection rule:** modern CPU → **hardware-assisted**; need other ISA → **emulation**; old CPU/closed OS → binary translation. **Boundary:** emulation too slow for production `[M:45]`.

## 3.1 Type 1 vs Type 2 hypervisor `[M:65-66]`
Type 1 (bare-metal): performance, isolation, scale → **data centres/clouds**. Type 2 (hosted): convenience for desktops/dev, extra OS layer ⇒ slower, relies on host OS security.

## 3.2 Overcommit policy `[M:6-7, M:56-58]`
| Workload | CPU ratio | Memory |
|---|---|---|
| Latency-sensitive (DB, real-time) | **1:1** | no overcommit |
| Mixed | **3:1 – 5:1** | modest, with ballooning/page-sharing |
| Dev/test, bursty idle | up to 8:1 (risky above) | higher, accept swapping risk |
- Trade-off: **density/cost ↑ ↔ latency/risk ↑**. Monitor **CPU Ready (>5%)**.

---

# 4. VM vs CONTAINER `[M:111-118]`

| Criterion | VM | Container |
|---|---|---|
| Isolation | **strong** (hypervisor, own kernel) | weaker (shared kernel; namespaces + cgroups) |
| Start-up | seconds–minutes | milliseconds–seconds |
| Size / density | GBs / lower | MBs / **higher** |
| OS flexibility | any guest OS | **must match host kernel** |
| Portability | image-heavy | **very portable (image)** |
| Best for | multi-tenant untrusted, different OSs, legacy apps | microservices, CI/CD, scale-out stateless apps |
- **OS container (LXC) vs application container (Docker)** `[M:116-117]`: OS container = lightweight machine, many processes (fleet of VMs-like envs); app container = one service per container (preferred for microservices).
- **Boundary:** for hostile multi-tenant workloads, run containers **inside VMs/microVMs**.

---

# 5. ORCHESTRATOR: Docker Swarm vs Kubernetes vs managed `[M:143-146, M:158]`

| Need | Swarm | Kubernetes |
|---|---|---|
| Learning/ops effort | **low** (Compose files, `swarm init`) | higher (many objects, add-ons) |
| Auto-scaling | **none built-in** | **HPA/VPA**, cluster autoscaler |
| Stateful workloads | weak (volume stuck on a node) | StatefulSets, PV/PVC, operators |
| Health checks | one (healthy or replaced) | **separate readiness & liveness** |
| Scheduling control | basic constraints | requests/limits, affinity/anti-affinity, taints/tolerations, spreading |
| Multi-team governance | limited | **namespaces, RBAC, quotas** |
| Extensibility | little | CRDs/operators, large ecosystem |
| Managed cloud option | none | **EKS/AKS/GKE** |
| Long-term | uncertain future | industry standard |
- **Choose Swarm** for small teams, simple apps, edge/small VMs. **Choose K8s** for growing teams, stateful data, autoscaling, multi-tenant clusters, portability across clouds. **Managed K8s** when you don't want to run the control plane.
- **Boundary:** K8s' power has an ops cost; for 2-3 services, PaaS or Swarm may be the *better* answer.

---

# 6. SCALING & PLACEMENT DECISIONS

## 6.1 Vertical vs horizontal scaling `[M:236, M:244]`
| | Vertical (scale up) | Horizontal (scale out) |
|---|---|---|
| Pros | simple, no app change | no hard ceiling, failure tolerant, fine-grained cost |
| Cons | ceiling, restart, single point of failure | needs **stateless/partitioned** design, load balancer, coordination (Amdahl) |
| Choose | quick relief; legacy/stateful monolith | elastic, internet-scale, HA |

## 6.2 Placement heuristic `[M:235]`
First-fit (fast, arrival order) vs **first-fit decreasing** (fewer hosts: 2 vs 3 in the deck's example) vs best-fit/worst-fit (leave more/less spare). **Trade-off:** tight packing saves hosts but increases blast radius of a host loss; spreading improves resilience but wastes capacity.

## 6.3 Provisioning/reservation model `[M:237-240, M:242]`
| Model | Use when | Cost |
|---|---|---|
| Immediate (public cloud) | capacity assumed unlimited | pay-per-use |
| Best-effort lease (queue/backfill) | finite private cloud, flexible jobs | may wait/pre-empted |
| **Advance reservation** | known time window (instrument 2–4 pm) | **under-utilisation** if naive; preparation overhead |

## 6.4 AWS purchase model `[M:237, M:246-247]`
| Workload | Choose |
|---|---|
| unknown/new | **On-Demand** |
| steady base load | **Savings Plan / Reserved** (price only — no capacity guarantee) |
| spiky, unpredictable | **autoscale** |
| interruptible batch | **Spot** + checkpointing (2-min notice) |
| unmissable date | **Capacity Reservation / Capacity Blocks** |

## 6.5 Request/limit setting `[M:245-246]`
Request = **p95 + 20–30%**; **CPU limit throttles**, **memory limit OOM-kills** → be conservative with memory; too-low request → contention, too-high → waste/Pending.

---

# 7. STORAGE & DATABASE CHOICE (AWS) `[M:98-106]`

| Need | Choose | Why |
|---|---|---|
| Static assets, media, backups, data lake, unlimited | **S3** (object) | cheap, durable, HTTP API; not a disk |
| Boot volume, database disk, low latency, one instance | **EBS** (block) | persistent, fast; **single-AZ** |
| Shared files across many servers | **EFS** (file/NFS) | multi-attach, elastic; costlier per GB |
| Cold archive, rare access (hours OK) | **Glacier** | lowest cost; 3–5 h retrieval |
| Relational with managed ops/HA | **RDS (Multi-AZ)** | sync standby, auto failover |
| Key-value at scale | **DynamoDB** | serverless-style NoSQL |
| Sub-ms caching | **ElastiCache** | in-memory |
| Analytics warehouse | **Redshift** | columnar |
- **Boundary:** don't run a database on instance store (lost with instance); don't use S3 as a POSIX file system.

---

# 8. MULTI-TENANCY DESIGN `[M:173, M:250-284]`

## 8.1 Silo vs Pool vs Bridge `[M:173]`
| | Silo | Pool | Bridge |
|---|---|---|---|
| Isolation | **strong** | weak (tenant-ID logic) | tiered |
| Cost per tenant | highest | **lowest** | medium |
| Upgrade | many things to patch | **one path** | mixed |
| Risk | cost, ops sprawl | **noisy neighbour**, leak bug | complexity |
| Choose | regulated/enterprise tenants, per-tenant tuning | many small similar tenants | premium vs standard tiers |

## 8.2 Data-layer options `[M:278, M:284-285]`
| Option | Isolation | Efficiency | Customisation | Ops |
|---|---|---|---|---|
| **Database/schema per tenant** | complete | lower | easy | complex DB management |
| **Dedicated tables** (one DB) | good | medium | easy | per-tenant objects |
| **Shared tables + tenant ID** | logical only | **best** | hard (pre-allocated columns waste/limit; name-value pairs; XML) | metadata table, views/joins |
- Customisation by **pre-allocated columns** (fixed count: waste vs restrict), **name-value pairs**, extension tables.

## 8.3 Multi-tenancy maturity level `[M:252]`
Custom (hard to manage) < Configurable (one codebase) < **Configurable + single instance** (efficient) < **Scalable cluster-based** (max sharing, e.g., Gmail). **Choose higher level** for many similar small customers; **lower** for few big, demanding customers.

## 8.4 Authentication `[M:282]`
Centralised user DB (tenant admins manage their users, provider authenticates) vs decentralised (each tenant authenticates its users — integrates with their directory).

## 8.5 Benefits vs risks of MT `[M:279-280]`
✔ cost savings, easy cross-tenant analytics, single upgrade · ✘ complexity, **all-tenants downtime** on one release, cross-tenant bugs, constrained release windows.

---

# 9. MIGRATION & RELEASE DECISIONS

## 9.1 Migration technique `[M:199-207]`
| | Cold | Live (classic) | Live shared-nothing | Storage migration |
|---|---|---|---|---|
| VM running | no | yes | yes | yes |
| Downtime | whole copy | tens–hundreds ms | short switchover | none |
| Shared storage | no | **required** | **not required** | n/a |
| CPU compatibility check | no | yes | yes | n/a |
| Complexity/time | simplest | medium | **most complex/slowest** | medium |
| Choose | bulk DR/DC exit, non-critical | maintenance, load balancing | no SAN, cross-cluster | change datastore |

## 9.2 6 R's `[M:8, M:210]`
| R | Effort | Gain | Pick when |
|---|---|---|---|
| Rehost | low | fast move | tight deadline, no code change |
| Replatform | low-medium | some managed benefits | e.g., DB → RDS |
| Repurchase | medium | SaaS | commodity apps (CRM) |
| Refactor | **high** | full cloud-native | strategic, scale/agility needed |
| Retire | — | cost saved | unused apps |
| Retain | — | — | compliance/latency/not ready |

## 9.3 Release strategy `[M:298]`
| | Rolling | Blue-green | Canary |
|---|---|---|---|
| Rollback speed | moderate | **instant (switch)** | fast (stop routing) |
| Extra resources | low | **2× environment** | small |
| Risk exposure | gradual | all at switch | **smallest** (small share) |
| Choose | routine low-risk | zero-downtime & easy rollback | risky change, need real-traffic validation |

## 9.4 IaC style `[M:208]`
Declarative (state what, tool converges; **idempotent**; plan/apply; drift detection) over imperative scripts (sequence of API calls). Choose declarative for repeatability/audit; imperative only for one-off glue.

---

# 10. APPLICATION ARCHITECTURE & INTEGRATION

## 10.1 Monolith vs microservices vs serverless `[M:291-292]`
| | Monolith | Microservices | Serverless |
|---|---|---|---|
| Deploy | one unit | independent services | per function |
| Scale | whole app | **only the busy service** | automatic per event |
| Complexity | low | **high** (network, orchestration, observability) | medium (limits, cold start) |
| Choose | small team, early product | large teams, differing scale needs | event-driven, spiky |
- **Evidence:** Netflix → microservices over years; **Prime Video monitoring tool → monolith, >90% cheaper** — fit the style to the workload.

## 10.2 Sync vs async integration `[M:296]`
Sync = simple, immediate answer, but caller blocked if the other is down; Async (queue) = absorbs spikes, decouples, result later/complexity. Use async for spikes and non-immediate work.

## 10.3 Four deployment ways `[M:299-300]`
VMs (max control/effort) → containers → managed platform → serverless (min control/effort). Choose by **control needed vs effort tolerated** and load shape.

## 10.4 Edge vs cloud for AI `[M:309-311]`
**Training** → cloud (heavy, rented accelerators); **inference** → cloud or **edge** when latency, bandwidth or offline operation matter. **Accelerator choice `[M:316]`:** CPU (general) · GPU (parallel AI) · TPU/ASIC (specific ML, least flexible, most efficient) · FPGA (reprogrammable).

---

# 11. SLA & AVAILABILITY DECISIONS `[M:253-265]`
- **Which HA tier?** Compute downtime budget `(1−A)×period`: 99.9% 43.2 min · 99.95% 21.6 · 99.99% 4.32 per 30 days. Each extra "9" needs more redundancy: server → zone → cloud `[M:255]`. **Trade-off:** availability ↑ ⇒ cost/complexity ↑.
- **Load balancer type `[M:260]`:** class-agnostic (simple) vs **class-aware** (inspect client/request type; better SLO targeting, more overhead).
- **Admission control `[M:260]`:** request-based (simple; may split sessions) vs **session-based** (finish admitted sessions; reject new) — choose session-based for e-commerce/checkout.
- **Infrastructure vs application SLA `[M:261]`:** infra SLA = hardware/power/network uptime; application SLA = response time/throughput — you need both; reservations if demand predictable, capacity planning if elastic `[M:261]`.

---

# 12. GREEN & ECONOMIC DECISIONS `[M:175, M:246, M:318]`
- Prefer providers/regions with low **PUE** and clean energy; PUE ignores server efficiency/water/electricity source. Use Arm/accelerators where efficient; consolidate and power off idle hosts `[M:217]`.
- **SaaS vs on-prem horizon:** break-even ~year 4 in the deck's example.

---

# 13. LIKELY DESIGN-DECISION QUESTIONS (inferred) → ANSWER FROM
| Likely question | Section |
|---|---|
| Public/private/hybrid/community; multi-cloud vs hybrid | §1 |
| IaaS vs PaaS vs SaaS vs FaaS; buy vs build | §2 |
| Virtualization techniques; Type 1 vs 2; overcommit policy | §3 |
| VM vs container | §4 |
| Swarm vs Kubernetes | §5 |
| Vertical vs horizontal; placement; reservation; purchase model; requests/limits | §6 |
| S3 vs EBS vs EFS; RDS vs DynamoDB | §7 |
| Multi-tenancy silo/pool/bridge; shared vs dedicated tables | §8 |
| Cold vs live migration; 6 R's; blue-green vs canary; IaC | §9 |
| Monolith vs microservices vs serverless; sync vs async | §10 |
| HA tier / SLA / admission control | §11 |
