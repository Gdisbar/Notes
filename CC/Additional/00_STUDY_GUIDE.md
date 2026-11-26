# STUDY GUIDE — what to study now that you carry every slide

*The reframe:* you can look **anything** up. Marks go to what cannot be looked up — **choosing the right option for a situation, justifying it with trade-offs, doing the arithmetic correctly, and structuring the answer.** Study *judgement and calculation*, not content.

---

## 1. The three tiers

### Tier 1 — Understand deeply (cannot be copied from a slide)
| # | Skill | What you must be able to do | Practise with |
|---|---|---|---|
| 1 | **Pick the service model** | place each component of an application on IaaS / CaaS / PaaS / FaaS / BaaS / SaaS using the decision path, and say what you give up | `03` §2; `04` Case 1 |
| 2 | **Overcommit and capacity sizing** | compute ratios and hosts needed (forecast → headroom → hosts → spare) and explain the assumptions | glossary pages 6–7; `02` §2.1–2.4 |
| 3 | **Live-migration reasoning** | explain pre-copy, show why it converges only if the dirty rate is below the bandwidth, compute downtime, choose between migration types | `02` §3; `03` §9.1 |
| 4 | **VM vs container; Swarm vs Kubernetes** | justify the choice from isolation, density, statefulness, autoscaling, team size | `03` §4–5 |
| 5 | **Diagnose scheduling problems** | requests vs limits vs allocatable; why a pod is Pending / OOM-killed / throttled | `02` §1.7, §2.5 |
| 6 | **Design for availability** | server → zone → cloud failure; SLO downtime arithmetic; stateful vs stateless handling | `02` §1; `03` §11 |
| 7 | **Design multi-tenancy** | silo / pool / bridge, shared vs dedicated tables, per-tier decisions, release risk | `03` §8; `04` Case 8 |
| 8 | **Control cost** | match purchase models to workload shape; rightsizing; SaaS vs on-premises break-even | `03` §6.4; `04` Cases 2, 5, 6 |
| 9 | **Choose an architecture style** | monolith vs microservices vs serverless, with the Netflix and Prime Video evidence | `03` §10 |
| 10 | **Structure a case answer** | business goal → service/deployment model → compute → data → security → scale → availability → delivery → cost → risks | `04` §1 |

### Tier 2 — Know exactly where it is (look up, don't memorise)
Definitions (NIST, essential characteristics) · AWS service names and their features · IAM terms · Docker commands and Dockerfile instructions · Kubernetes object kinds and components · Swarm commands · SLA/SLO definitions · SLA life-cycle phases · multi-tenancy levels · DORA metrics · golden signals · 6 R's names · every formula and its worked example (glossary pages 6–7).

### Tier 3 — Skim only
Introduction and history slides · cloud facts and statistics · "Who is a cloud engineer" · quantum, neuromorphic and photonic computing (one reading is enough) · Azure primer detail · session recaps and Q&A slides.

---

## 2. Where your time should go

| Question type | Share of study time | What to do |
|---|---|---|
| **Concept** | **15%** | read `01` once; be able to explain every Tier-1 skill in your own words |
| **Scenario** | **30%** | learn the 8-step algorithm in `02` §0; spend a third of this time on the **numeric drills** |
| **Design decision** | **30%** | learn the comparison tables in `03` by *reasoning*; practise the five-line decision answer |
| **Case study** | **25%** | do two timed cases from `04` using the blueprint |

> If time is short, **do Tier-1 skills 1, 2, 4, 6 and 10 first**, then the numeric skills 3 and 5.

---

## 3. Drills (on paper, timed)

| # | Drill | Time | Success check |
|---|---|---|---|
| D1 | **Capacity sizing**: p95 = 120 vCPU, growth 15%, headroom 25%, host = 48 vCPU, survive one host failure | 10 min | 5 hosts, steps shown |
| D2 | **Overcommit**: 6 VMs × 2 vCPU on 4 pCPUs; what if three peak together? | 10 min | ratio 3:1 and the simultaneous-demand caveat |
| D3 | **Live migration**: 8 GiB VM, 1 GiB/s link, 0.2 GiB/s dirty rate, stop below 0.3 GiB | 10 min | rounds, ≈64 ms downtime, and why D < B is required |
| D4 | **Amdahl**: parallel fraction 0.9, speed-up 10 | 5 min | ≈5.26× and the 10× ceiling |
| D5 | **Kubernetes fit**: 2-vCPU node, pods requesting 500 m | 5 min | allocatable ≈1.9 → 3 pods; fourth Pending |
| D6 | **Availability**: downtime per month at 99.9 / 99.95 / 99.99% | 5 min | 43.2 / 21.6 / 4.32 min |
| D7 | **SaaS vs on-premises**: $100 × 200 users × 12 vs $350K + $150K/yr | 10 min | break-even around year 4 |
| D8 | Put each part of a retail app on a service model (storefront, resize-on-upload, login, support desk, database) | 15 min | each choice has a stated cost |
| D9 | Multi-tenancy: choose silo / pool / bridge and table design for two different SaaS products | 15 min | linked to confidentiality and load |
| D10 | Full case: migrate a legacy retail system to the cloud — blueprint headings plus one diagram | 40 min | all 12 blueprint headings |

---

## 4. Self-test questions (inferred; answer on paper, then check against the notes)
1. Compare IaaS, PaaS and SaaS for a start-up and for a bank.
2. Why does the x86 architecture complicate virtualization, and how do the approaches solve it?
3. A host runs 40 vCPUs on 16 cores and users complain — diagnose and fix.
4. Explain pre-copy live migration; when does it fail to converge?
5. Containers or VMs for a multi-tenant SaaS with untrusted code?
6. Swarm or Kubernetes for a three-service app with a database?
7. Design for 99.95% availability on AWS — what do you build at server, zone and cloud level?
8. A release broke production — blue-green or canary, and why?
9. Choose between single-tenant and multi-tenant for a financial product.
10. Your cloud bill doubled — what do you check and change?
11. When is microservices the wrong answer?
12. Size a private cloud for the next quarter with N+1.

---

## 5. Tab list — flag these pages in your PDF
| Topic | Pages |
|---|---|
| Fundamentals: NIST, characteristics, service & deployment models | 14-37 |
| Virtualization (x86 problem, four approaches, overcommit) | 38-62 |
| Hypervisors & IaaS architecture | 64-70 |
| AWS intro: regions/AZ, shared responsibility, IAM | 70-82 |
| AWS compute: EC2, AMI, VPC | 85-96 |
| AWS storage & databases: S3, EBS, EFS, Glacier, RDS | 97-107 |
| Containers & Docker | 110-129 |
| Container orchestration: Swarm | 130-144 |
| Kubernetes & FreshCart hands-on | 145-157 |
| PaaS (incl. Azure primer) | 159-171 |
| SaaS | 171-177 |
| BaaS, FaaS & five-model comparison | 177-185 |
| VM management & VM life cycle / provisioning | 188-198 |
| VM migration (live, cold, storage) | 198-207 |
| Infrastructure as Code & 6 R's | 207-212 |
| Capacity management | 214-221 |
| Virtual Infrastructure Manager (VIM) & OpenNebula | 222-230 |
| Workload scheduling, leases, K8s scheduling, cost | 231-243 |
| Scalability & elasticity | 244-249 |
| SLA management & high availability | 253-271 |
| Multi-tenancy | 274-289 |
| Cloud application development models, APIs, CI/CD | 291-298 |
| Deployment options, observability, hybrid/multi-cloud | 299-306 |
| Emerging cloud: edge, AI/ML, quantum, accelerators, green | 307-320 |
| **Glossary — abbreviations** | 5-6 |
| **Glossary — formulas & worked calculations** | **6-7** |
| **Glossary — key concepts** | 7-8 |

*(Glossary slide numbers: page = 8 + ⌈slide ÷ 2⌉.)*

---

## 6. Exam-day workflow
1. **Tag** each question (Concept / Scenario / Decision / Case) — 2 min.
2. **Plan** with the skeleton: need → mechanism → how it works → cost → comparison → decision rule or number.
3. **Look up only lists and formulas** (use the tab list).
4. **Write** short structured points; add *because / at the cost of / not suitable when*.
5. **Numbers:** show the steps, state assumptions, and sanity-check the answer.
