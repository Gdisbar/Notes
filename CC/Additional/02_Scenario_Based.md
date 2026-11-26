# PART 2 — SCENARIO-BASED QUESTIONS
*Handle a concrete situation, failure or edge case → **Diagnose → Mitigate → Recover → Prevent***
REF format `[M:page]` (page of your compressed PDF `Cloud_Computing_Slides.pdf`, two slides per page) → see `00_READ_FIRST.md`. Concepts in `01_Concepts.md` (**A…N**).

---

# 0. THE SCENARIO-ANSWER ALGORITHM

| Step | Do this | Write this |
|---|---|---|
| 1. **Classify** | Which concern? availability · performance · capacity/cost · security/isolation · migration · deployment | "This is primarily an *availability* problem at the *zone* level." |
| 2. **Locate the layer** | hardware/hypervisor · VM · container/pod · platform · application · network · tenant | "Fault is at the *node/pod* layer, not the application." |
| 3. **Diagnose** | the evidence you'd check (metrics, events, states) | "`kubectl describe pod` → *Pending: insufficient CPU*; or CPU-Ready > 5%." |
| 4. **Mitigate now** | fastest safe action that restores service | "Fail over / reschedule / scale out / roll back." |
| 5. **Recover** | restore redundancy and data | "Re-add the node, resync, rebalance." |
| 6. **Prevent** | architectural fix, automation, policy | "Multi-AZ, requests/limits, canary, IaC, alerts." |
| 7. **Edge cases & trade-offs** | what could go wrong with the fix; cost | "Spot reclaim during failover; stateful pod can't move disk." |
| 8. **Quantify** | one number from the deck's formulae `[M:6-7]` | "Downtime budget at 99.95% = 21.6 min/month." |

> **Cloud edge-case checklist:** *stateful vs stateless · noisy neighbour · partial failure · cascading failure · cold start · quota/limit hit · capacity unavailable ("insufficient capacity") · drift between code and reality · credential leakage · cost explosion · data residency.*

---

# 1. AVAILABILITY & FAILURE SCENARIOS (J · `[M:253-255]`)

## S1.1 Physical host fails (or needs maintenance)
- **Diagnose:** hypervisor/host health lost; VMs on it unreachable.
- **Mitigate:** if *planned/failing-but-alive* → **live migrate** VMs away first `[M:199]`; if *dead* → VIM/orchestrator **restarts VMs on other hosts** (needs spare capacity — the **+1 spare / N+1** in sizing `[M:219]`).
- **Recover:** repair/replace host, re-balance.
- **Edge cases:** no spare capacity (overcommit hid the problem); VMs on local-only disks can't restart elsewhere; live migration blocked by GPU/PCI pass-through `[M:205]`.

## S1.2 An Availability Zone fails
- **Mitigate:** traffic shifts to instances in other AZs behind a load balancer; **RDS Multi-AZ** promotes the synchronous standby `[M:106]`; EBS volumes are *AZ-bound* (replicated within the AZ only) so attach copies/snapshots in another AZ `[M:100]`.
- **Prevent:** design **"for zone failure"** — redundant servers across AZs `[M:73, M:255]`. **Edge case:** **EFS/S3** survive AZ loss but EBS/instance-store don't; **placement groups** are single-AZ so they don't protect `[M:93]`.

## S1.3 Whole region / provider outage
- **Mitigate:** DR site/region failover; DNS switch. **Prevent:** **"build for cloud failure"** — contain errors with a **fire-door policy** so failures don't cascade; multi-region or **multi-cloud** (cost: complexity, networking overhead `[M:28]`). **Check the SLA remedy** — it's a credit, not your lost revenue `[M:175]`. **Metric:** RTO.

## S1.4 Spot instances are reclaimed
- Spot = spare capacity, **2-minute notice**, a best-effort lease that can be pre-empted `[M:237]`. **Mitigate:** handle the interruption signal, **checkpoint**, drain, resume elsewhere. **Prevent:** use Spot **only for interruptible batch**; keep base load on committed/on-demand `[M:247]`. **Edge case:** don't put the database or session store on Spot.

## S1.5 Docker Swarm manager failure `[M:6-7, M:140]`
- **Rule:** tolerated failures = ⌊(N−1)/2⌋. **3 managers → 1 failure; 5 → 2.** Lose quorum ⇒ cluster can't schedule changes (running tasks continue). **Edge case:** **even number** (4) tolerates the same as 3 — pointless; **split brain** across zones → keep majority in one place or use odd count across 3 zones.

## S1.6 Kubernetes node dies `[M:151]`
- **Diagnose:** kubelet heartbeats stop → controller marks node unhealthy. **Mitigate:** controller **reschedules pods** on healthy nodes; Services stop routing to unready pods (readiness). **Edge cases:** **StatefulSet/Postgres pod** is tied to the node holding its **local-path disk** `[M:155]` → can't just move; use network storage (EBS/EFS CSI) or replication. Pods can go **Pending** if remaining nodes lack allocatable CPU `[M:245]`.

## S1.7 Pod keeps restarting / killed
| Symptom | Likely cause | Fix |
|---|---|---|
| **Pending** | requests > node allocatable; taints/affinity; no node passes the filter `[M:235, M:245]` | right-size requests, add nodes, relax constraints |
| **OOMKilled** | **memory limit** exceeded `[M:245]` | raise limit/request (be conservative with memory), fix leak |
| **CPU throttling** (slow, not killed) | **CPU limit** too low | raise/remove CPU limit |
| **CrashLoopBackOff** | app exits at start (bad config/secret/DB not ready) | check logs/readiness, init containers |
| Traffic hits a pod that isn't ready | no readiness probe | add readiness (`/api/ready`), liveness (`/api/healthz`) `[M:154]` |

## S1.8 Cold-start / function timeout (serverless) `[M:179, M:300]`
- First call after idle waits for an environment (**cold start**); runs longer than **15 min** are killed; concurrency caps throttle bursts. **Mitigate:** provisioned/warm instances, smaller packages, **split long jobs** (queue + step functions) or move to containers/PaaS. **Edge case:** state must live outside (DB/cache); duplicates → idempotent handlers.

## S1.9 Bad release breaks production `[M:298]`
- **Fastest recovery:** **blue-green** → switch traffic back to the old environment. **Limit blast radius:** **canary** (small share first) + watch the **golden signals** (latency, traffic, errors, saturation) `[M:302]`; **rolling** with rollback. **Measure:** DORA — **change failure rate** and **time to restore** `[M:298]`. **Edge case:** DB schema changes must be backward-compatible or blue-green rollback fails.

## S1.10 SLA breach `[M:253-256, M:264, M:268]`
- **Diagnose:** compare **SLI** (measured) to **SLO** (target); find deviation in monitoring. **Mitigate:** remedial action (add capacity, re-place, throttle via admission control `[M:260]`). **Account:** record adherence; **penalties/credits** per SLA. **If sustained:** production behaves unlike pre-production → **renegotiate/adjust resource policies** `[M:268]`. **Prevent:** profile at on-boarding, pre-production validation `[M:266, M:268]`.

---

# 2. PERFORMANCE, CAPACITY & COST SCENARIOS

## S2.1 VMs on a host are slow though CPU "looks fine" `[M:56-57]`
- **Diagnose:** compute **overcommit ratio** = vCPUs ÷ pCPUs; check **CPU Ready time (> ~5%)** = vCPUs ready but waiting. **Worked example:** 6 VMs × 2 vCPU = 12 vCPUs on 4 pCPUs = **3:1** (edge of workable); if 3 VMs peak together → 6 vCPUs demanded on 4 → **1.5:1 squeeze** → latency.
- **Mitigate:** migrate a VM away (live migration), reduce vCPUs, lower ratio (1:1 for latency-sensitive DB/real-time), set shares/reservations for critical VMs `[M:218]`.

## S2.2 Memory pressure / "VM is swapping" `[M:57-58]`
- **Order of reclamation:** **page sharing (TPS/KSM) → ballooning → compression → host swapping.** Swapping ≈ disk ~**100,000× slower** than RAM → last resort.
- **Mitigate:** reduce memory overcommit ratio (**configured ÷ physical**), add RAM, move VMs; reserve memory for critical VMs. **Edge case:** ballooning needs the guest driver; identical-page sharing gives little for diverse workloads.

## S2.3 Noisy neighbour in a pooled SaaS `[M:173, M:280]`
- **Diagnose:** one tenant's load degrades others (shared stack). **Mitigate:** per-tenant **rate limits/quotas**, queues, **move heavy tenant to a silo/bridge tier**, QoS isolation. **Trade-off:** silo costs more `[M:173]`.

## S2.4 Capacity sizing for next quarter (worked) `[M:6-7, M:219]`
**Given:** p95 = 120 vCPU, growth 15%, headroom 25%, host = 48 vCPU.
1. Forecast: 120 × 1.15 = **138**
2. Headroom: 138 × 1.25 = 172.5 → **173**
3. Hosts: 173 ÷ 48 = 3.6 → **4**
4. Spare for one host failure (N+1): 4 + 1 = **5 hosts**
- **Edge cases:** p95 hides the worst 5% of the time (so headroom); if committed with *overcommit*, recompute on pCPU; if hosts are heterogeneous, use bin-packing.

## S2.5 Pods Pending though the node is idle `[M:245]`
- 2-vCPU node → **allocatable ≈ 1.9 vCPU** (capacity − system reserved). Pods request **500m** → **3 fit** (1.9/0.5 = 3.8 → 3); **4th Pending** because *requests reserve capacity* regardless of live usage. **Fix:** right-size requests to **p95 + 20–30%** `[M:246]`, add node, or use VPA/HPA.

## S2.6 Cloud bill is out of control `[M:209, M:246-247]`
- **Diagnose:** untagged resources, idle/oversized instances, always-on dev, wrong purchase model.
- **Mitigate:** **tags** (Owner/Project/Env/CostCenter) for allocation; **budgets & alerts** (50/80/100%); **quotas/SCP guardrails**; **rightsizing**; schedule dev/test off-hours; **Graviton/Arm** where supported (Zomato: −30% cost `[M:107]`); **purchase model**: steady base → Savings Plan; spiky → autoscale; batch → Spot.
- **Edge case:** SaaS/PaaS convenience premium at high steady load — consider VMs `[M:163]`; SaaS vs on-prem break-even around year 4 `[M:175]`.

## S2.7 Sudden traffic spike `[M:31, M:244]`
- **Mitigate:** **auto-scaling (horizontal)**, **async queues absorb spikes** `[M:296]`, **API gateway throttling** `[M:295]`, cache. **Requirement:** stateless services; **Amdahl** — scale-out gain capped by the serial part (1/(1−p)) `[M:236]`. **Edge case:** autoscaler lags the spike → pre-scale for known events; **vertical scale** for immediate relief `[M:244]`.

## S2.8 "Adding 10× servers only gave 4× speed-up" `[M:6-7, M:236]`
Use **Amdahl:** S = 1/[(1−p)+p/s]. With p = 0.9 and s = 10 → **5.26×**; even infinite s → **10×** max. Fix by reducing the serial fraction (shared locks, DB bottleneck, coordination).

---

# 3. MIGRATION SCENARIOS (H · `[M:199-207]`)

## S3.1 Live migration doesn't converge / downtime too long
- **Mechanism:** iterative **pre-copy** — each round re-sends pages dirtied during the previous round. Round size shrinks by factor **D/B**.
- **Worked numbers `[M:6-7, M:203]`:** M = 8 GiB, B = 1 GiB/s, D = 0.2 GiB/s, threshold 0.3 GiB → rounds **8 → 1.6 → 0.32 → 0.064**; stop-and-copy of 0.064 GiB = **64 ms downtime**; total ≈ **10 s**.
- **If D ≥ B** the rounds never shrink → **pre-copy never converges**. **Mitigate:** throttle the VM's CPU/dirty rate, compress, increase bandwidth, **auto-converge**, or fall back to **post-copy** (pull faulted pages) `[M:8]`, or schedule at low-load time. **Edge cases:** failure mid-migration — source stays primary until commit, so a consistent copy always exists `[M:202, M:204]`.

## S3.2 "Migrate without shared storage" `[M:204-205]`
Use **shared-nothing (block) live migration**: mirror disks first while VM runs → pre-copy memory rounds → short switchover. **Cost:** more data to move, longer, more complex than classic (shared-storage) migration. **Preconditions:** compatible CPU features, reachable network, enough free capacity; **blockers:** PCI pass-through, SR-IOV, local USB/CD, very large/busy VMs, affinity rules.

## S3.3 Which migration for a data-centre exit with 500 VMs? `[M:199, M:206]`
Bulk **cold migration** (powered-off; no CPU-compat check; no shared storage; simpler, but VM is down for the entire copy) for non-critical VMs; **live/shared-nothing** for critical ones; use **6 R's** per application (rehost vs replatform vs retire) `[M:210]`. **Edge case:** licensing and IP changes; verify post-migration (drivers reattach, IP advertised `[M:204]`).

## S3.4 Moving a running VM's disk to new storage `[M:207]`
Live **storage migration**: copy while reads/writes continue on source → mirror new writes to both → when synchronised switch to destination → delete source. **Edge:** high write rate slows convergence.

## S3.5 New VM boots empty — who configures it? `[M:8, M:194, M:196]`
**AMI + user-data/cloud-init** (hostname, keys, install packages) or **IaC/golden image**; otherwise the cloud benefit is lost. **Edge:** secrets in user-data are visible — use a secrets manager/roles `[M:79]`.

## S3.6 "Console changes broke our environment" (IaC drift) `[M:208]`
**Diagnose:** **drift** = real resources changed outside the code. **Mitigate:** `plan` shows the diff; re-apply (idempotent) or import the change into code; lock down console edits (SCP/IAM). **Prevent:** all change through Git review; tag resources.

---

# 4. SECURITY & ISOLATION SCENARIOS

## S4.1 Credentials leaked / over-permissive access `[M:80]`
- **Mitigate:** revoke keys immediately; **use roles (temporary, no long-term keys)** instead of hard-coded credentials; enforce **MFA/SSO**; **least privilege** policies; **SCP guardrails** `[M:209]`. **Edge:** shared responsibility — **IAM, data and config are always the customer's** `[M:34, M:160]`.

## S4.2 Tenant A sees Tenant B's data `[M:282, M:284]`
- **Diagnose:** shared-table query missing tenant-ID filter; weak isolation. **Mitigate:** enforce **row-level security** per tenant ID, views, **separate schema/DB for sensitive tenants** (bridge/silo `[M:173]`), centralised vs decentralised authentication. **Edge:** an update to the single shared instance can also break everyone at once `[M:280]`.

## S4.3 Update to a multi-tenant app causes downtime for all `[M:280]`
**Risk of MT:** one instance → one release for all tenants; bugs appear in others' customised views; release window constrained by several tenants' usage. **Mitigate:** canary per tenant cohort, feature flags, tenant-aware rollouts, off-peak windows.

## S4.4 Container escape worry `[M:112-114, M:118]`
Containers **share the host kernel** → weaker isolation than VMs. **Mitigate:** namespaces + cgroups, non-root, minimal images, separate untrusted workloads on **VMs or microVMs (Firecracker `[M:179]`)**, node-level isolation. **Edge:** kernel vulnerability affects all containers on the node.

## S4.5 Stolen/deleted data in object storage `[M:98-103]`
Use bucket policies/IAM, encryption, **versioning**, cross-region replication; remember **EBS is single-AZ** and **instance store disappears** `[M:7]`. Glacier for archival (3–5 h retrieval).

---

# 5. DEPLOYMENT & DESIGN-FOR-FAILURE SCENARIOS

## S5.1 Stateful workload in a stateless platform `[M:144, M:155, M:179]`
- **Swarm:** volume tied to one node, can't move data → weak for stateful `[M:144]`. **K8s:** use **StatefulSet + PVC + replication**. **FaaS:** keep state externally. **Edge:** DB on one node is single point of failure — use managed DB (RDS Multi-AZ) `[M:106]`.

## S5.2 "Works on my machine" `[M:120-123]`
Package with **Dockerfile** (reproducible, byte-for-byte layers) and run the same image everywhere; don't `docker commit` by hand. **Edge:** kernel must match host; secrets not baked in image.

## S5.3 Service A calls Service B and B is down `[M:295-296]`
**Sync** call blocks/fails → use **async queue** (absorbs spikes; result later), retries with backoff, timeouts, circuit breaker (🆕), health checks; gateway routes only to healthy. **Edge:** duplicates → idempotent consumers.

## S5.4 Microservices became too expensive `[M:292]`
**Prime Video:** a stream-monitoring tool with microservices + serverless → merged into a **monolith on EC2/ECS** → **>90% cost reduction**. **Lesson:** choose the architectural style per workload; excessive distribution adds network, orchestration and per-invocation costs.

## S5.5 Private cloud runs out of capacity `[M:189, M:237-240, M:242]`
Public clouds hide limits through scale; **private cloud finite** → the **immediate provisioning** model fails. **Mitigate:** **lease-based scheduler** — **best-effort** leases queued/backfilled, **advance reservations** for known windows, suspend/resume/migrate VMs to fit; **cloud-burst** to public (hybrid) `[M:26]`. **Edge:** naive reservation underutilises resources; image preparation delays lease start `[M:240, M:242]`.

## S5.6 Choosing how to reserve capacity for a fixed launch date `[M:237, M:247]`
Product launch can't miss → **On-Demand Capacity Reservation** (guaranteed in one AZ; pay even if unused). ML training on a date → **Capacity Blocks**. Savings Plans/RIs reduce price but **do not reserve capacity**.

---

# 6. LIKELY SCENARIO QUESTIONS (inferred) → ANSWER FROM
| Likely question | Section |
|---|---|
| Host/AZ/region failure; HA design for X% availability | S1.1-S1.3, J2 |
| Quorum/manager failures in Swarm; K8s node/pod failures | S1.5-S1.7 |
| Release goes wrong; blue-green vs canary | S1.9 |
| SLA breached — what to do | S1.10 |
| Overcommit / memory pressure calculation | S2.1-S2.2 |
| Capacity sizing numbers; pods Pending | S2.4-S2.5 |
| Cost optimisation | S2.6 |
| Live migration numbers / non-convergence; storage migration | S3.1-S3.4 |
| Leaked credentials; tenant data leak; MT release risk | S4.1-S4.3 |
| Stateful workloads in orchestrators/serverless | S5.1 |
| Private cloud capacity exhaustion; lease models | S5.5-S5.6 |
