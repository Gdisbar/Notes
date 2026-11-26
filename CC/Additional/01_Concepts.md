# PART 1 — CONCEPTS (lean version)
*What it is · Why it exists · How it works · Where it fits (and what it costs) · A minimal example*

**You carry every slide, so this file is not a copy of the slides.** It gives the *idea* behind each topic, how topics connect, the trade-offs and the vocabulary. Use the page references `[M:…]` to look up lists, tables and the **glossary (formulas and worked calculations are on pages 6–7)**. Page numbers are for your compressed PDF (see `00_READ_FIRST.md`).

**Contents:** A Cloud fundamentals · B Virtualization · C Hypervisors & IaaS · D AWS core · E Containers & Docker · F Orchestration · G PaaS / SaaS / serverless · H VM management, migration, IaC · I Capacity, VIM & scheduling · J SLA & high availability · K Multi-tenancy · L Application development & deployment · M Emerging cloud · N Likely questions

---

# A. CLOUD FUNDAMENTALS

## A1. What is cloud computing?
- **Idea:** computing delivered as a **utility/service** over the Internet — applications and services running on a distributed network of **virtualized resources**, reached through standard protocols. Cloud is the **convergence** of web services, SOA, distributed computing, grids and clusters. `[M:15-16]`
- **Two enabling ideas:** **abstraction** (implementation details hidden from users) and **virtualization** (resources pooled and shared; each user feels like the only owner). *Cloud exists because of virtualization and abstraction.* `[M:17]`
- **The business shift:** CapEx (buy hardware) → **OpEx** (pay as you go). `[M:5-6]`

## A2. NIST 3-4-5 `[M:7, M:18]`
**3** service models (IaaS, PaaS, SaaS) · **4** deployment models (public, private, hybrid, community) · **5** essential characteristics.

## A3. Five essential characteristics `[M:29-32]`
On-demand self-service · broad network access · resource pooling · rapid elasticity · measured service. **How they connect:** pooling (multi-tenancy) + elasticity (auto-scaling) + metering (billing) are what let a provider run shared hardware efficiently *and* let you pay only for what you use. Self-service + APIs is what makes **Infrastructure-as-Code** possible.

## A4. Service models — think "who controls what" `[M:18-23]`
**IaaS** (you control OS and above) → **PaaS** (you control the app) → **SaaS** (you control only settings). Moving right: less control and effort, less portability, more lock-in. The "who manages what" picture is `[M:23]`; guidance on which to pick `[M:23]`. **Data, identity and access remain your responsibility in every model.** `[M:7, M:160]`

## A5. Deployment models — think "control vs scale" `[M:24-28]`
Public = scale and speed, less control · Private = control and compliance, finite capacity and higher cost · Hybrid = right workload in the right place, but integration complexity · Community = shared governance. **Multi-cloud** = two or more *public* providers (leverage, best-of-breed vs complexity); hybrid ≠ multi-cloud and a company can be both. `[M:27-28, M:303]`

## A6. Advantages, challenges, failures `[M:33-34]`
Advantages: cost, scalability, agility, reach. A **cloud application** is designed *for* cloud capabilities (elasticity, distribution, automation); a "web app" merely runs behind a browser. Challenges follow the **shared-responsibility** split. Major outages happen → **design for failure** (multi-AZ, DR).

---

# B. VIRTUALIZATION

## B1. Idea and reason `[M:39-41]`
One physical host runs many isolated guests through a **hypervisor**. Reason: consolidation and utilisation, isolation, fast provisioning (VMs are files you can snapshot, clone, migrate) — the foundation of IaaS.

## B2. Why x86 was hard `[M:42-43]`
Guest OSs assume they own ring 0, but the hypervisor needs it too. Classic **trap-and-emulate** requires all sensitive instructions to be privileged; on x86 some are **sensitive but not privileged** — they misbehave *silently* instead of trapping. That gap is why the four approaches below exist.

## B3. Four approaches — the trade-off in one line each `[M:44-51, M:59]`
- **Emulation:** software imitates a different CPU — flexible (cross-ISA), slowest.
- **Binary translation (full virtualization):** hypervisor rewrites sensitive instructions on the fly — unmodified guests, translation overhead.
- **Paravirtualization:** guest kernel modified to use hypercalls — fast, but needs source access.
- **Hardware-assisted:** new CPU privilege mode traps sensitive instructions — fast *and* unmodified guests; today's default.
- **Do not confuse emulation with virtualization:** virtualization shares the *same* hardware among isolated guests. `[M:51]` Comparison table: `[M:59]`.

## B4. What gets virtualized `[M:52-55]`
Server, storage, network, memory (also desktop, application).

## B5. Overcommitment — the reasoning `[M:6-7, M:56-58]`
- **CPU** can be overcommitted because it is **time-sliced**; watch CPU Ready time. Ratio = vCPUs ÷ pCPUs.
- **Memory** cannot be time-sliced, so the hypervisor *reclaims* from least to most invasive (page sharing → ballooning → compression → swapping); swapping is a last resort because disk is vastly slower than RAM.
- **Rule:** the ratio is a heuristic, not a guarantee — what matters is *simultaneous* demand. Latency-sensitive workloads need ~1:1.

## B6. Caveats `[M:59-60]`
Software **licensing** per VM, VM sprawl, training, security of the shared host.

---

# C. HYPERVISORS & IaaS

## C1. Hypervisor `[M:65-67]`
**Type 1** (bare-metal; clouds) vs **Type 2** (hosted; desktops). **Popek–Goldberg goals:** equivalence, resource control, efficiency. Levels of virtualization trade performance against implementation effort.

## C2. IaaS and NIST actors `[M:69-70]`
IaaS = hypervisors + cluster/resource management + storage + network + APIs. Actors: **consumer, provider, broker, auditor, carrier.**

---

# D. AWS CORE `[M:71-107]`

- **Global infrastructure:** Region (independent geography) → Availability Zones (isolated, low-latency); spread across AZs to survive zone failure. Local Zones, Wavelength, Outposts extend to cities, telecoms, your site. **Shared responsibility:** AWS secures *of* the cloud; you secure *in* it. `[M:7, M:71-76]`
- **IAM:** authentication (who) vs authorisation (what); **roles** (temporary permissions, no long-term keys) and **policies**; least privilege. `[M:6, M:77-80]`
- **EC2:** AMI → instance; lifecycle (stopped keeps disks, no compute charge; terminated is final); tenancy (shared vs dedicated); placement groups (low latency within one AZ); **VPC** = isolated virtual network (subnets, route tables, security groups, gateways). `[M:85-96]`
- **Choose storage by access model:** **S3** (objects via API, unlimited, cross-AZ durable) · **EBS** (block disk for one instance, **single AZ**) · **EFS** (shared file system) · **Glacier** (archive, hours to retrieve). Comparison `[M:103]`. **Databases:** RDS (managed relational, Multi-AZ failover), DynamoDB (NoSQL), ElastiCache (in-memory), Redshift (analytics). `[M:97-106]`
- **Case:** Zomato — Graviton (Arm) + Spot cut analytics cost by ~30% and improved query performance. `[M:107]`

---

# E. CONTAINERS & DOCKER `[M:111-129]`

- **Idea:** OS-level virtualization — containers **share the host kernel**. **Namespaces isolate (what a process can see); cgroups limit (what it can use).** `[M:7, M:111-114]`
- **VM vs container:** strong isolation, any OS, heavier vs fast, dense, portable, but kernel-bound and weaker isolation. `[M:118]` **OS containers** behave like light machines; **application containers** package one service. `[M:116-117]`
- **Docker:** client → daemon → registry; **image** (read-only layered template from a Dockerfile) vs **container** (running instance with a thin writable layer). **Layers + copy-on-write** make images share storage and start quickly. `[M:119-126]`
- **Why Dockerfiles:** reproducible, reviewable, versionable (vs hand-committed containers); order instructions so the **build cache** survives code changes. `[M:125-128]`

---

# F. CONTAINER ORCHESTRATION `[M:131-155]`

- **Why:** one container is easy; many containers on many machines need placement, scaling, networking, self-healing, rolling updates. `[M:131-136]`
- **Imperative vs declarative:** declare the **desired state** and let the platform **reconcile** reality to it. `[M:7, M:132]`
- **Swarm:** simple, built in; stack → service → task → container; manager fault tolerance follows a **quorum** rule (⌊(N−1)/2⌋ failures). Pros: easy, secure by default; cons: **no autoscaling, weak stateful support**. `[M:6-7, M:139-144]`
- **Kubernetes:** control plane (API server, etcd, scheduler, controllers) + worker nodes (kubelet, kube-proxy, runtime); **observe → compare → act** loop; pods are the unit. `[M:145-151]`
- **Swarm vs K8s reasoning:** choose Swarm for small teams and simple apps; K8s when you need autoscaling, stateful workloads, fine scheduling control, multi-team governance or a managed service. `[M:146, M:158]`
- **FreshCart** is the worked example (web, API, Redis, Postgres under both orchestrators); the database is the hard part. `[M:151-155]`

---

# G. PaaS, SaaS, SERVERLESS `[M:158-182]`

- **Shared responsibility across models:** `[M:160, M:180]` — *your data and access always stay yours.*
- **PaaS:** you bring code; provider runs runtime/OS/scaling. Gains: speed, patching, scaling. Costs: limited control, lock-in, cost at scale. Fits standard stacks and small teams; not custom kernels or very steady heavy load. `[M:159-169]`
- **SaaS:** one codebase serving many tenants. **Tenancy models:** silo (isolated, costly), pool (shared, noisy-neighbour risk), bridge (tiered). Economics example: SaaS is cheaper for roughly four years, then ownership wins. `[M:172-176]`
- **Serverless:** **BaaS** (ready backend, fastest, lock-in) vs **FaaS** (event-driven functions, stateless, cold starts, run-time limits). `[M:178-179]`
- **Five-model comparison and decision path:** `[M:180-182]` — real systems **mix models per component** `[M:182]`.

---

# H. VM MANAGEMENT, MIGRATION, IaC `[M:189-210]`

- **Private cloud problem:** capacity is finite, so the public cloud's "immediate provisioning" model is not enough. `[M:189-190]`
- **Provisioning:** validate → place → image → network → boot → **contextualise** (cloud-init, IaC) — manual installs defeat the cloud. `[M:194-196]`
- **Migration — why and the logic:** maintenance, load balancing, consolidation, DR, exit. **Live pre-copy** copies memory in rounds while the VM runs, re-sending dirtied pages, then a short **stop-and-copy**. **It converges only if the dirty rate is below the bandwidth.** Classic live migration needs shared storage; **shared-nothing** mirrors disks too; **cold** is simplest but the VM is down. `[M:6-8, M:199-207]`
- **IaC:** declare infrastructure in code → plan → apply; **idempotent**; **drift** = reality changed outside code. Tags, quotas and budgets give governance. `[M:208-209]`
- **6 R's:** rehost, replatform, repurchase, refactor, retire, retain — chosen per application. `[M:210]`

---

# I. CAPACITY, VIM & SCHEDULING `[M:216-247]`

- **Capacity balance:** too little → SLA breaches; too much → waste, sprawl, licences. Sizing method: measure → forecast → headroom → convert to hosts → add spare (worked in the glossary). `[M:6-7, M:216-219]`
- **VIM:** control plane that decides where VMs run and manages their life cycle; OpenNebula is the example (pluggable hypervisor drivers, placement policies). `[M:221-230]`
- **Scheduling logic:** **filter** hosts that fail hard constraints, **score** the rest; no passing host → pod Pending / "no valid host". Placement is NP-hard, so heuristics (first-fit, first-fit-decreasing). **Requests** reserve, **limits** cap (CPU limit throttles, memory limit kills). `[M:6-7, M:231-236, M:245]`
- **Leases:** immediate, best-effort, advance reservation; naive reservation wastes capacity. AWS purchase models differ in *price* vs *capacity guarantee*. `[M:237-243]`
- **Scaling:** vertical (simple, ceiling) vs horizontal (needs stateless design; limited by **Amdahl's law**). `[M:236, M:244]`
- **Cost control:** rightsizing, scale-in, commitments for steady load, Spot for interruptible work. `[M:246-247]`

---

# J. SLA & HIGH AVAILABILITY `[M:250-269]`

- **SLA** is the whole agreement; **SLO** is a measurable target inside it; **SLI** is the measurement. Downtime budget = (1 − availability) × period. `[M:6-7, M:253-256]`
- **HA steps:** build for server failure → zone failure → cloud failure (fire-door policy) → automate and test. `[M:254-255]`
- **Traditional SLO management:** load balancing (class-agnostic vs class-aware) and **admission control** (request-based vs session-based). `[M:257-260]`
- **SLA life cycles:** the provider's contract life cycle and the cloud-application life cycle (feasibility → on-boarding → pre-production → production → termination). `[M:261-269]`

---

# K. MULTI-TENANCY `[M:250-252, M:277-286]`

- **Idea:** one instance serves many tenants with per-tenant customisation; four maturity levels from custom instances to scalable shared clusters. `[M:250-252]`
- **By layer:** IaaS (VM or network isolation), PaaS (container, dedicated, resource pool), SaaS (schema per tenant vs partitioned data). `[M:277-278]`
- **Benefits vs risks:** cost and easy analytics vs complexity, **one release can hit every tenant**, isolation. `[M:279-282]`
- **Data design:** shared tables (best efficiency, hard to customise) vs dedicated tables vs schema per tenant; customisation via pre-allocated columns or name-value pairs. `[M:7, M:284-285]`

---

# L. APPLICATION DEVELOPMENT & DEPLOYMENT `[M:291-303]`

- **Monolith vs microservices vs serverless** — pick by workload: Netflix moved over years; Prime Video moved a monitoring tool back to a monolith for >90% saving. `[M:291-292]`
- **12-factor, APIs/gateways, sync vs async:** portability, front door (authenticate, throttle, route), spike absorption. `[M:8, M:293-296]`
- **CI/CD and safe release:** rolling vs blue-green vs canary; DORA metrics. `[M:297-298]`
- **Four ways to deploy:** custom VMs → containers → managed platform → serverless — trading control for effort. `[M:299-300]`
- **Observability:** logs, metrics, traces, alerts; the four golden signals (latency, traffic, errors, saturation). `[M:301-302]`

---

# M. EMERGING CLOUD `[M:307-318]`
Edge (move compute near data), AI/ML (training in cloud vs inference anywhere), quantum/neuromorphic/photonic (early), accelerators (match chip to job), **green cloud** — PUE = total facility energy ÷ IT energy (ignores server efficiency, water, energy source). `[M:7, M:307-318]`

---

# N. LIKELY CONCEPT QUESTIONS (inferred — no PYQ archive)
| Likely question | Where |
|---|---|
| NIST definition, five characteristics, service and deployment models | A |
| x86 challenge; four virtualization approaches; overcommit | B |
| Hypervisor types and goals | C |
| AWS regions/AZ, IAM, EC2/VPC, S3 vs EBS vs EFS | D |
| Containers vs VMs; Docker architecture and layers | E |
| Swarm vs Kubernetes; components; pod life cycle | F |
| PaaS vs SaaS vs FaaS vs BaaS; tenancy models | G |
| Live migration (pre-copy); IaC; 6 R's | H |
| Capacity sizing; scheduling; Amdahl; leases | I |
| SLA/SLO/SLI; HA steps; SLA life cycle | J |
| Multi-tenancy levels, models, risks, data sharing | K |
| Monolith vs microservices; CI/CD; observability | L |
| Edge, AI on cloud, green cloud | M |
