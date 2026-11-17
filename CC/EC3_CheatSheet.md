# EC3 Cheat Sheet — Last-Day Revision (CSI/SE/SS ZG527)

> Everything numerical and every high-frequency comparison on 4 pages. If you can reproduce this sheet from memory, you can attempt any predicted EC3 question.

---

## 1. Formula sheet (with one-line worked examples)

| # | Formula | Worked example |
|---|---|---|
| 1 | **SLA downtime** = (1 − availability) × period | 30-day month = 43,200 min: 99% → 432 min; **99.9% → 43.2 min**; 99.95% → 21.6 min; 99.99% → 4.32 min. Week (10,080 min) at 99.5% → 50.4 min |
| 2 | **Achieved availability** = 1 − (downtime ÷ total) | 105 min outage/43,200 → 99.757% |
| 3 | **Composite availability (serial)** = A1 × A2 × … | 99.9% × 99.9% = 99.8001% |
| 4 | **Redundant pair (parallel)** = 1 − (1−A)² | pair of 99.9% → 99.9999% |
| 5 | **CPU overcommit** = total vCPU ÷ pCPU | 6 VMs × 2 vCPU ÷ 4 pCPU = 3:1 (safe band 3:1–5:1; >8:1 risk; CPU Ready >~5% = oversubscribed) |
| 6 | **Memory overcommit** = configured ÷ physical | 4 × 10 GB ÷ 32 GB = 1.25:1; reclaim order: share → balloon → compress → swap (disk ≈100,000× slower) |
| 7 | **Pods per node** = min(⌊allocatableCPU ÷ reqCPU⌋, ⌊allocatableMem ÷ reqMem⌋, maxPods) | 4 vCPU, 500m → 8; add memory 8 GiB, 2 GiB → 4 (min wins); 1 CPU = 1000 m |
| 8 | **Node allocatable** = capacity − system reserved | 2 vCPU node ≈ 1.9 allocatable → 500m pods fit 3 (1.9/0.5 = 3.8 → 3); 4th pod Pending |
| 9 | **Capacity sizing (hosts)** = ⌈p95 × (1+growth) × (1+headroom) ÷ vCPU/host⌉ + 1 spare | 120 × 1.15 = 138; ×1.25 = 172.5 → 173; ÷48 = 3.6 → 4; **+1 = 5 hosts** |
| 10 | **Bin-packing first-fit vs FFD** | capacity 10; items 4,5,6,3,2: arrival order → 3 hosts; sorted 6,5,4,3,2 → **2 hosts** |
| 11 | **Swarm manager tolerance** = ⌊(N−1)/2⌋ (Raft majority) | 3 → 1; 5 → 2; **4 → still 1** (odd counts only) |
| 12 | **Live migration pre-copy** | M=8 GiB, B=1 GiB/s, D=0.2: rounds 8 → 1.6 → 0.32 → 0.064; downtime = 0.064 s = 64 ms; total ≈ 10 s; converges only if **D < B**; shrink factor per round = D/B |
| 13 | **Amdahl** Smax = 1 ÷ [(1−p) + p/s] | p=0.9, s=10 → 1/(0.1+0.09) ≈ 5.26×; cap = 1/(1−p) = 10× |
| 14 | **PUE** = facility energy ÷ IT energy | 1.6 → 1.2 saves (1.6−1.2)/1.6 = 25%; Google 1.09 (2024), global avg 1.54 |
| 15 | **Autoscaling count** = ⌈demand ÷ (capacity × targetUtil)⌉ | 10,000 rps ÷ (500 × 0.7 = 350) → 29 → 30 for AZ balance; AZ-failure proof = size one AZ to carry all |
| 16 | **Consolidation** = totalWork ÷ targetUtil, round up | 40 × 12% = 4.8 ÷ 0.70 → 7 hosts (−82.5%) |
| 17 | **Rightsizing** = p95 + 20–30% headroom (memory more conservative) | p95 900m → 1125m → use 1200m |
| 18 | **Cost compare** | on-demand 20×8760×$0.12 = $21,024; reserved at $0.078 → $13,665 (≈35% save); Savings Plan/RI = price only, **not capacity** |

---

## 2. The 10 comparison tables you must be able to redraw

**2.1 VM vs Container:** hardware-level + own guest OS / GBs / minutes / strong isolation **vs** OS-level + shared kernel / MBs / seconds / namespace-cgroup isolation / high density / portable.

**2.2 Full vs Para vs HW-assisted virtualization:** guest unmodified+binary translate / guest modified+hypercall / guest unmodified+Ring −1 hardware trap; VMware / Xen PV / KVM-Hyper-V-VT-x AMD-V; overhead / better / near-native.

**2.3 Type 1 vs Type 2 hypervisor:** bare-metal (ESXi, KVM, Xen, Hyper-V — production) vs hosted on OS (VirtualBox, Workstation — dev/desktop).

**2.4 Emulation vs virtualization:** cross-ISA possible, slow (Bochs/QEMU/Rosetta) vs same-ISA only, near-native.

**2.5 IaaS vs PaaS vs SaaS (+BaaS/FaaS):** deploy VM image / code-image / nothing—configure / client+rules / function; you patch OS: yes / no / no; scaling: you build / platform rules / per-event to zero; control highest → lowest IaaS→SaaS. Data+identity accountability never transfers.

**2.6 Block vs File vs Object (EBS vs EFS vs S3 vs Glacier):** raw disk, one-mount, DB/boot / shared POSIX, many mounts / whole-object API, buckets+keys, unlimited, HTTP PUT/GET / archive, 3–5 h retrieval.

**2.7 Public vs Private vs Hybrid vs Community:** rent apartment / own house / own+rent-burst / group facility; scale-speed vs control vs flexibility vs common compliance.

**2.8 Hybrid vs Multi-cloud:** private+public vs 2+ public providers (can combine).

**2.9 Horizontal vs Vertical scaling:** add replicas+LB (stateless, no ceiling, failure-tolerant) vs bigger single instance (ceiling, restart, SPOF); "scale up for relief, design to scale out". K8s: HPA vs VPA.

**2.10 Cold vs Live vs Storage migration:** powered-off, cross-platform OK, downtime=copy / running, ms-level downtime, shared storage, same platform / disks of running VM, zero interruption (copy→mirror→switch). Downtime vs migration time distinction. 6 stages: pre-migration, reservation, iterative pre-copy, stop-and-copy, commitment, activation.

Bonus one-liners: **Multi-tenant vs multi-instance** (one shared instance vs separate copies per tenant); **MT levels**: custom → configurable → configurable+efficient → scalable+efficient (Gmail); **Silo/Pool/Bridge**; **conserving vs non-conserving allocation**; **request vs limit** (reserve+schedule vs cap: CPU throttles, memory OOM-kills); **SLI→SLO→SLA** (measure→target→contract); **static vs dynamic LB**; **class-agnostic vs class-aware**; **request-based vs session-based admission control**.

---

## 3. Lists that get asked verbatim

- **NIST 3-4-5:** IaaS/PaaS/SaaS · public/private/hybrid/community · on-demand self-service, broad network access, resource pooling, rapid elasticity, measured service.
- **IaaS actors:** consumer, provider, broker, auditor, carrier.
- **Hypervisor goals:** fidelity, performance, safety/isolation, resource multiplexing.
- **Memory reclaim order:** TPS/KSM sharing → ballooning → compression → host swap.
- **Docker build benefits:** reproducibility, customisation, self-documentation, CI/CD automation, layer caching (deps before code; CoW for containers).
- **K8s control plane:** kube-apiserver (front door: authn→authz→admission), etcd (state, Raft, apiserver-only access), kube-scheduler (filter→score→bind), kube-controller-manager (control loops); workers: kubelet, kube-proxy, runtime; add-ons: CNI, CoreDNS, metrics-server, CSI.
- **K8s objects:** Pod (ephemeral, own IP), Deployment (rolls ReplicaSets), ReplicaSet (keeps count), Service (stable VIP via label selector; ClusterIP/NodePort/LB/headless), Namespace, StatefulSet, ConfigMap/Secret, HPA.
- **Swarm:** stack → service → task → container; replicated vs global services; constraints/preferences/reservations; ports 2377/7946/4789; overlay + ingress routing mesh + DNS/VIP.
- **AMI 4 sources:** AWS-published, Marketplace, generated-from-instance (golden), uploaded VMs (VHD/VMDK/OVA).
- **EC2 tenancy:** shared (default) / dedicated instance / dedicated host (licensing). Placement group = one-AZ low-latency grouping. Instance store = ephemeral (lost on stop/terminate).
- **VM provisioning 7 steps:** validate → place → image → network → boot → contextualise (cloud-init; user-data runs once at first boot; AMI saves disk only) → health.
- **IaC keywords:** declarative, idempotent, drift, state (Terraform state file protect; CFN managed), plan→apply; GitOps = Git as source of truth.
- **Governance:** tags (Owner/Project/Environment/CostCenter, at creation), quotas, budgets, SCP guardrails; loop define→enforce→monitor→report→adjust.
- **6 R's:** rehost, replatform, repurchase, refactor, retire, retain.
- **Capacity 5 steps:** measure (p95/p99) → characterise → forecast → headroom+N+1 → choose purchase model.
- **Reservation models:** on-demand (best effort) · spot (preemptable, 2-min notice) · capacity reservation (guaranteed AZ, billed anyway) · capacity blocks (future GPU) · Savings/RI (price, not capacity).
- **HA steps:** server failure → zone failure → cloud failure → automate & test.
- **SLA lifecycle:** contract definition → publication & discovery → negotiation → operationalization (monitor/account/enforce) → de-commissioning. MSP app phases: feasibility (technical/infrastructure/financial) → on-boarding (OVF packaging, profiling) → pre-production → production → termination.
- **SLA types:** no SLA / probabilistic (cheaper, percentiles) / deterministic (100%, costliest). Infrastructure SLA vs Application SLA (response-time SLOs).
- **12-factor highlights:** config in env, backing services attached, stateless processes, logs as streams, disposability, dev-prod parity.
- **CI/CD:** commit→build→test→release→deploy; CI vs Delivery (human gate) vs Deployment (auto); rolling / blue-green / canary; DORA: deployment frequency, lead time, change failure rate, time to restore.
- **Observability:** logs (what) / metrics (how much) / traces (where) / alerts; golden signals: latency, traffic, errors, saturation; OpenTelemetry, Prometheus, Jaeger.
- **Edge layers:** device → local edge (Greengrass) → network edge (Wavelength 5G) → cloud region. Edge AND cloud.
- **Chips:** CPU general / GPU parallel / TPU-ASIC one job / FPGA reprogrammable; training heavy-occasional vs inference constant-latency.
- **Green levers:** efficient DC (PUE), efficient chips, cleaner electricity, switch off idle + rightsize.

---

## 4. Top traps (each one costs marks)

1. **Multi-tenancy ≠ multi-cloud.** Tenants share a provider; multi-cloud = one customer, many providers.
2. **AMI = software template** (OS+patches+software), NOT hardware spec (that's the *instance type*). AMI saves disk only, no memory; user-data runs once at first boot, executed by cloud-init *inside* the VM.
3. **etcd stores state only.** ReplicaSet keeps replica counts; Service gives stable networking via label selectors; scheduler only *decides placement* (filter→score→bind), kubelet runs pods.
4. **Request ≠ limit.** Requests reserve & drive scheduling; limits cap at runtime (CPU throttles, memory OOM-kills). Pods go Pending on *requests*, not limits.
5. **Savings Plan/RI do not reserve capacity** — only On-Demand Capacity Reservation does (billed whether used or not).
6. **Multi-AZ RDS = availability (sync standby, auto failover); Read Replicas = read scale (async, lag, not failover targets).** Standby is not readable.
7. **Security group = stateful, instance-level, allow-only; NACL = stateless, subnet-level, allow+deny in order.**
8. **S3 = whole-object API, not a file system; buckets flat (no sub-buckets); EBS volume attaches to one instance at a time; EFS = concurrent shared mounts.**
9. **Availability: replicas in ONE AZ share the zone's fate** — spread AZs; serial availability multiplies.
10. **Autoscaling does not replace capacity planning** (lead time, min/max, steady-base purchase model are planned decisions).
11. **Vertical scaling has a ceiling and stays a SPOF; horizontal needs statelessness.**
12. **Live migration converges only if dirty rate < bandwidth; downtime = final stop-and-copy, not total transfer time.**
13. **Spot is preemptable (2-min notice)** — pair with checkpointing; interruptible work only.
14. **Type 2 hypervisor overhead is fine for dev, not production** — bare-metal for density/HA.
15. **First-fit vs first-fit-decreasing give different host counts** — packing tight saves money but raises blast radius (spread vs pack trade-off).
16. **Numericals:** always write formula → substitute → units → assumption; round *up* for hosts/instances; p95/p99, never averages.

---

## 5. Answer skeletons (write these first, then fill scenario facts)

- **"Recommend X"** = name it → define (1 line) → 2 reasons tied to scenario constraints → one trade-off/mitigation.
- **"Differentiate A vs B"** = 4-row table (definition, mechanism, performance/cost, when-to-use) → one-line decision rule.
- **"Calculate"** = formula → substitution → result+units → assumption/caveat sentence.
- **"Is X necessarily true?"** = "No — not necessarily" → condition where true → condition where false → conclusion.
- **Architecture question** = draw/describe layers with named components (VPC/subnet/SG/AZ/ASG/LB/DB) → data flow sentence → why each constraint is met → failure-mode sentence.
- **Security question** = shared responsibility split → threat list → control per threat (MFA/roles/least privilege/encryption/private subnet/monitoring) → "accountability for data & identity never transfers".

## 6. Memory hooks

- **3-4-5** (NIST). **"Namespaces limit what you see; cgroups limit what you use."** **"Request reserves, limit caps."** **"Scale up for relief, scale out for life."** **"Logs=what, metrics=how much, traces=where."** **"Spot = interruptible, Reserved = cheap-but-not-reserved, Capacity Reservation = reserved-but-billed."** **"Multi-AZ for availability, Read Replicas for reads."** **"etcd remembers, apiserver gates, scheduler places, controllers fix."** **"⌊(N−1)/2⌋ for quorum."** **"Downtime = (1−A) × period; 99.9% = 43.2 min/month."**
