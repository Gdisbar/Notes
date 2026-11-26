# PART 4 — CASE STUDIES
*End-to-end analysis under real constraints → **a cohesive cloud architecture/strategy tied to business & system goals***
REF format `[M:page]` (page of your compressed PDF `Cloud_Computing_Slides.pdf`, two slides per page) → see `00_READ_FIRST.md`. Concepts: `01_`; scenarios: `02_`; decisions: `03_`.

---

# 1. THE CLOUD CASE-STUDY BLUEPRINT (write these headings first)

| # | Heading | What to write | Source |
|---|---|---|---|
| 1 | **Business goals & workload profile** | goal (cost, speed, availability, compliance); users; load shape (steady / spiky / predictable); data sensitivity | `[M:217]` |
| 2 | **Constraints** | regulation/residency, budget, skills, legacy systems, SLA/RTO targets | `[M:176, M:253]` |
| 3 | **Service & deployment model** | IaaS/CaaS/PaaS/FaaS/BaaS/SaaS **per component**; public/private/hybrid/multi-cloud; justify with the decision path | `[M:24-28, M:182]` |
| 4 | **Compute & virtualization layer** | VMs vs containers vs functions; orchestrator (Swarm/K8s/managed); overcommit and placement policy | `[M:118, M:146, M:235-236]` |
| 5 | **Data & storage** | S3/EBS/EFS/RDS/NoSQL; state placement (stateful vs stateless); backup/DR | `[M:98-106, M:155]` |
| 6 | **Network, identity & security** | VPC/subnets/security groups, IAM (roles, least privilege, MFA/SSO), shared-responsibility, tenant isolation | `[M:80-95, M:282]` |
| 7 | **Scalability & capacity** | horizontal vs vertical; autoscaling; **sizing calculation** (hosts = ⌈p95×(1+g)×(1+h)÷cap⌉+1) | `[M:219, M:244]` |
| 8 | **Availability & SLA** | AZ/region design, HA steps, SLO numbers (downtime minutes), DR/RTO, monitoring (golden signals) | `[M:6-7, M:254-255, M:302]` |
| 9 | **Delivery & operations** | IaC, CI/CD, blue-green/canary, tagging, quotas, observability | `[M:208-209, M:297-298, M:301]` |
| 10 | **Cost & sustainability** | purchase model mix, rightsizing, Arm/Spot, PUE-aware region; break-even (SaaS vs on-prem) | `[M:175, M:246-247, M:318]` |
| 11 | **Migration path (if existing)** | 6 R's per app; live/cold migration method | `[M:199-207, M:210]` |
| 12 | **Risks & trade-offs** | lock-in, noisy neighbour, cold start, complexity — each with mitigation | `[M:163, M:175, M:280]` |

**Tie-back template:** "Because the business needs **…**, we selected **model/service** …, which costs **…**; we mitigate with …; not suitable if …."

---

# 2. WORKLOAD → TYPICAL CLOUD ARCHITECTURE (use for unseen cases)

| Workload | Dominant concerns | Typical architecture |
|---|---|---|
| **E-commerce / retail web** | spiky traffic, availability, payments security | CDN + load balancer; stateless web/API on **PaaS or K8s** with autoscaling; **Redis** for sessions/carts; **RDS Multi-AZ** for orders; queue for async tasks; blue-green/canary; golden signals |
| **Start-up MVP** | speed, low cost, small team | **PaaS/BaaS/serverless** first; managed DB; pay-per-use; move to containers when load steady |
| **Regulated enterprise (bank/health)** | compliance, isolation, audit | **Hybrid**: sensitive data private/on-prem, front-end public; **silo/bridge tenancy**; IAM with MFA/SSO; encryption; DR region; SCP guardrails |
| **SaaS product** | multi-tenancy, cost per tenant, upgrades | **Pool or bridge tenancy**, tenant-ID row-level security, control plane (onboarding, billing, metering); canary per tenant cohort |
| **Batch/analytics/ML training** | throughput, cost | **Spot** + checkpointing, accelerators rented only while training, data in object storage near compute; schedule off-peak |
| **Real-time/IoT/low-latency** | latency, weak links | **Edge** processing (Greengrass/Wavelength) + cloud region for storage/training; async messaging |
| **Event-driven/spiky small tasks** | pay per use | **FaaS** (Lambda) triggered by uploads/queues + object storage; stateless, idempotent |
| **Legacy/lift-and-shift** | deadline, risk | **Rehost** with live/cold migration → replatform DB → refactor strategic parts later |
| **Private-cloud data centre** | finite capacity, utilisation | VIM (OpenNebula/OpenStack) + **lease scheduler** (best-effort + AR), consolidation, **cloud-burst** |

---

# 3. WORKED CASES FROM THE COURSE

> Format: **Problem → Decisions → Numbers/Evidence → Trade-offs → Business tie.**

## CASE 1 — FreshCart: containerising a retail app (Swarm vs Kubernetes) `[M:151-155, M:182]`
- **Problem:** an online grocery with a React front end, Node/Express API, Redis carts and Postgres data must run reliably and scale; team must choose and operate an orchestrator.
- **Application view `[M:154]`:** **web** (nginx serving React, forwards `/api/*`; 2 copies), **api** (3 copies, **autoscale 2–8**; `/api/healthz` alive, `/api/ready` DB reachable), **redis** (carts — *disposable*, 1 copy), **postgres** (**must never be lost**; 1 + its own disk). Config via env vars; secrets for DB password.
- **Swarm deployment `[M:154]`:** 1 manager + 2 workers; **4 services → 7 tasks**; **overlay networks** `frontend` (web↔api) and `backend` (api↔redis/db); ingress/routing mesh opens port 80 on every node; **db pinned by label** to the node with the disk; secret + config + volume.
- **Kubernetes deployment `[M:155]`:** control-plane node (apiserver, etcd, scheduler, controller-manager; tainted) + workers (kubelet, kube-proxy, containerd, Flannel CNI); **Deployments** web(2), api(3), redis(1); **StatefulSet** postgres-0 + PVC; Services (web **NodePort 30080**, api/redis **ClusterIP**, postgres headless); ConfigMap/Secret; **HPA api 2–8**; CoreDNS resolves `api.freshcart.svc.cluster.local`.
- **Decision logic:** Swarm = quickest for a small team (no autoscaling, weak stateful); **K8s** gives HPA, readiness/liveness, StatefulSets, RBAC `[M:146]`. Database is the hard part in both: pinned/stateful, ideally replaced by **managed RDS Multi-AZ** in production.
- **Five-service-model mapping `[M:182]`:** storefront → PaaS (App Service/Cloud Run); **image resize on upload → FaaS**; customer login for a prototype → BaaS; support desk/CRM → SaaS; DB → managed service.
- **Trade-offs/risks:** postgres single point of failure; local-path disk ties pod to node; HPA needs requests/limits set honestly `[M:245]`.
- **Business tie:** faster releases, elastic capacity for sales days, lower ops with managed services.

## CASE 2 — Zomato: cut analytics cost with Graviton + Spot `[M:107]`
- **Problem:** India's food-delivery platform (70M+ daily users, 24 countries) runs analytics (Apache **Trino** and **Druid**) on AWS; compute cost and query speed matter.
- **Decision:** migrate clusters from **x86 EC2** to **Graviton2 (Arm) EC2** and use **Spot** for the savings.
- **Result (AWS-published):** **−30% compute cost, +25% query performance, ≈$300K/yr saved** on Trino alone.
- **Why it works:** analytics clusters are **interruptible, horizontally scalable batch/query** workloads → Spot fits `[M:247]`; Arm (Graviton) is price-performance efficient `[M:5-6]`.
- **Trade-offs:** software must support Arm; Spot interruptions need checkpoint/replace; keep stateful/critical parts on committed capacity.

## CASE 3 — Netflix: a multi-year cloud migration `[M:292]`
- **Trigger:** **database corruption in own data centre (Aug 2008)** → move to cloud; completed **Jan 2016** on AWS.
- **Architecture:** streaming built from many **microservices behind an API gateway (Zuul)**; **own CDN, Open Connect,** for video delivery; Spinnaker created for deployments (open-sourced 2015) supporting **blue-green and canary** `[M:298]`.
- **Lessons:** a cloud move is a **journey**; architecture changes along the way; build for failure (multi-AZ/region); use CDN for what the cloud is not best at.
- **Link to 6 R's:** started with rehost/replatform, then **refactor** to microservices `[M:210]`.

## CASE 4 — Prime Video: microservices → monolith `[M:292]`
- **Problem:** a stream-quality **monitoring tool** built as microservices + serverless functions.
- **Decision:** consolidate into a **single program on EC2/ECS**.
- **Result:** reported cost reduction **> 90%** for that service.
- **Lesson:** *smaller is not always better*; choose style per workload (per-invocation/network overhead dominates for chatty, steady, high-volume pipelines). Contrast **FaaS** best fit: short, spiky, event-driven `[M:300]`.

## CASE 5 — Private-cloud capacity plan for an online retailer (calculation case) `[M:6-7, M:219]`
- **Scenario:** retailer on a **private cloud**; hosts offer **48 vCPU** each; busy-hour p95 = **120 vCPU**; business expects **15% growth**; headroom **25%**; must survive **one host failure**.
- **Solution (5 steps):** measure 120 → forecast ×1.15 = **138** → headroom ×1.25 = 172.5 → **173** → hosts 173 ÷ 48 = 3.6 → **4** → add spare **N+1 = 5 hosts**.
- **Extras to mention:** overcommit ratio effects (1:1 for latency-critical), **right-size** requests (p95+20–30%), consolidation and power-off of idle hosts, **cloud-burst** to public for peaks (hybrid), **leases** (best-effort for batch, AR for known windows) `[M:237-243]`.
- **Risks:** forecast error; p95 ignores the worst 5%; capacity fragmentation.

## CASE 6 — SaaS vs on-premises for 200 users (economic case) `[M:175-176]`
- **Assumptions:** SaaS **$100/user/month** (=$240K/yr); on-prem **$350K up-front + $150K/yr** (maintenance, hosting, admin).
- **Cumulative:** Year 4 → SaaS **960K** vs on-prem **950K**; Year 5 → **1,200K vs 1,100K**. **SaaS is cheaper for ~4 years, then ownership wins.**
- **Decision framing:** horizon, growth in users (SaaS scales linearly), customisation needs, data-residency, exit costs, vendor SLA `[M:175-176]`; **hidden costs:** integration, migration, training.
- **Recommendation pattern:** short horizon/uncertain growth → SaaS; stable 5+ year, heavy customisation, strict sovereignty → on-prem/private.

## CASE 7 — Live migration for planned host maintenance (calculation case) `[M:6-7, M:203]`
- **Scenario:** evacuate an 8 GiB VM; link B = 1 GiB/s; dirty rate D = 0.2 GiB/s; stop-and-copy when remainder < 0.3 GiB.
- **Rounds:** 8 → 1.6 → 0.32 → **0.064 GiB** (<0.3 → stop). **Downtime = 0.064/1 = 64 ms**; **total ≈ 8+1.6+0.32+0.064 ≈ 10 s**.
- **Why safe:** source stays primary until commit; post-migration code reattaches drivers and advertises IP `[M:204]`. **Fails to converge if D ≥ B.**
- **Alternatives:** shared-nothing if no SAN `[M:204]`; cold for bulk non-critical.

## CASE 8 — Multi-tenant SaaS platform (design case) `[M:173, M:250-284]`
- **Goals:** many small customers, low cost per tenant, quick customisation, safe upgrades, tenant isolation.
- **Architecture:** **bridge tenancy** — shared app tier resolving tenant on each request; **row-level security by tenant ID** for standard tenants; **dedicated DB/schema for premium/regulated tenants**; **configuration metadata** (branding, workflow, custom fields via name-value/extension tables) `[M:280, M:285]`; centralised authentication with tenant-admin rights `[M:282]`; control plane for onboarding/billing/metering `[M:173]`; **per-tenant quotas** vs noisy neighbours.
- **Operations:** canary releases by tenant cohort (MT release risk `[M:280]`); observability per tenant; tags/cost allocation.
- **Maturity target:** level 3–4 (single configurable instance, cluster scaling) `[M:252]`.
- **Trade-offs:** complexity of per-tenant metadata; one release affects all; weaker isolation than silo.

## CASE 9 — Event-driven image pipeline (serverless) `[M:179, M:182, M:300]`
- **Flow:** user uploads image to **object storage** → event triggers **FaaS** → resizes → writes to storage/DB → optional queue/notification.
- **Why FaaS:** short, spiky, event-driven; pay per invocation; auto-scales to zero.
- **Edge cases:** stateless/idempotent handlers; **15-minute** limit; cold starts; concurrency caps; keep large state outside. **Boundary:** a long continuous video-processing job belongs on containers/VMs.

## CASE 10 — ML on the cloud with edge inference `[M:309-311, M:316]`
- **Lifecycle:** collect → **train** → evaluate → **deploy** → **monitor**. **Training** in cloud using rented **GPUs/TPUs** only while training (Spot + checkpoints where possible); data in object storage near compute. **Inference**: cloud endpoint for general users; **edge** (Greengrass/Wavelength) when latency/bandwidth/offline matter.
- **Trade-offs:** accelerator cost vs time; data egress; model drift needs monitoring; edge devices harder to update.

## CASE 11 — Hybrid cloud for a regulated company `[M:26, M:255, M:303]`
- **Design:** sensitive data and legacy systems in **private/on-prem**; customer-facing web/API and burst capacity in **public**; secure links, **consistent tools/IAM**; DR in a second region.
- **Why:** compliance + elasticity. **Risks:** integration complexity, link reliability, double operations. **Option:** add a second public provider only if concentration risk justifies the complexity (multi-cloud `[M:28]`).

## CASE 12 — Green data-centre strategy `[M:318]`
- **Facts:** data centres ≈ **1.5%** of world electricity in 2024; IEA projects **945 TWh** by 2030; **PUE** = total facility energy ÷ IT energy (Google 1.09 vs industry 1.54).
- **Actions:** choose low-PUE/clean-energy regions, consolidate and power off idle hosts, rightsizing, Arm/accelerators, schedule batch for cleaner times. **Caveat:** PUE excludes server efficiency, water, energy source.

---

# 4. TEMPLATE FOR AN UNSEEN CASE (fill-in)

```
Business goal:     <X to Y by when>, because <value>
Workload profile:  steady / spiky / predictable; data sensitivity: <low/high>; users: <n>
Constraints:       <regulation>, <budget>, <skills>, <legacy>, SLA <99.x%>, RTO <t>
Model:             public/private/hybrid/multi-cloud — because <driver>
Services:          <component → IaaS/CaaS/PaaS/FaaS/BaaS/SaaS> (decision path)
Compute:           VMs / containers (<Swarm|K8s|managed>) / functions
Data:              <S3|EBS|EFS|RDS|NoSQL>; state placement; backup/DR
Security:          IAM roles + MFA, VPC/security groups, encryption, tenant isolation <silo|pool|bridge>
Scale & capacity:  <horizontal|vertical>; autoscale on <metric>; sizing: hosts = ⌈…⌉ + 1
Availability:      multi-AZ (+ region DR); SLO <x%> = <y min/month>; golden signals
Delivery:          IaC, CI/CD, <blue-green|canary>, tags, quotas
Cost:              base→Savings Plan, spiky→autoscale, batch→Spot, rightsizing p95+20–30%
Migration (if any):6 R's per app; live vs cold
Risks:             lock-in, noisy neighbour, cold start, complexity → mitigations
```

---

# 5. DIAGRAM CHEAT-SHEET
- **IaaS stack:** hardware → hypervisor (Type 1) → VMs (guest OS + app) ↔ VIM/cluster manager ↔ APIs `[M:67, M:69]`.
- **VM vs container stack:** infra → host OS → (hypervisor + guest OS) vs (Docker engine + shared kernel) `[M:118]`.
- **Docker flow:** Dockerfile → build → image → push → registry → pull → container `[M:128]`.
- **Kubernetes cluster:** control plane (apiserver, etcd, scheduler, controller-manager) + workers (kubelet, kube-proxy, runtime, pods) + add-ons `[M:147]`.
- **VPC:** region → AZs → subnets (public/private) → IGW/NAT → security groups/NACLs `[M:95-96]`.
- **Live migration:** pre-copy rounds → stop-and-copy → commit → activate `[M:201-204]`.
- **SLA life cycle:** define → publish/discover → negotiate → operate (monitor/account/enforce) → decommission `[M:263-264]`.
- **CI/CD:** commit → build → test → release → deploy `[M:297]`.

---

# 6. LIKELY CASE-STUDY QUESTIONS (inferred) → ANSWER FROM
| Likely question | Section |
|---|---|
| Containerise/orchestrate a multi-tier app; Swarm vs K8s | Case 1 |
| Cut cloud cost / choose purchase models | Case 2, `03_` §6.4 |
| Migrate a legacy system to the cloud; lessons | Case 3, Case 7, `03_` §9 |
| Is microservices right? | Case 4, `03_` §10.1 |
| Size capacity for next quarter | Case 5 |
| SaaS vs on-prem decision with numbers | Case 6 |
| Design a multi-tenant SaaS | Case 8 |
| Serverless pipeline | Case 9 |
| ML/AI on the cloud and edge | Case 10 |
| Hybrid cloud for a regulated business | Case 11 |
| Green cloud / PUE | Case 12 |
