# EC3 Question Bank with Model Answers — CSI/SE/SS ZG527 Cloud Computing

> **60 practice questions** — Q1–Q29 are the original analysed bank; **Q30–Q60 are newly constructed** in the same style (scenario + application + numerical/justification), weighted toward the high-priority modules.
>
> **How to use:** attempt each subpart in 2–4 minutes, then compare with the model answer. Every answer shows the *key phrases* that earn marks (bold) — reproduce the bold skeleton in the exam, then attach scenario specifics.
> Modules follow the blueprint numbering. The Notes cover the same nine syllabus areas but give container orchestration its own module — mapping: bank Module 2 (containers+orchestration) = Notes Modules 2–3; bank Modules 3–9 = Notes Modules 4–9.

---

# Module 1 — Introduction to Cloud Computing

## Q1. Cloud adoption and architecture decision [10]

**Scenario:** A manufacturing company moving its production monitoring system to the cloud: highly variable workloads during production hours, sensitive operational data, and a requirement to keep some applications on premises.

**(a) Recommend an appropriate cloud deployment strategy. [3]**
**Model answer:**
- Recommend a **hybrid cloud**: keep the sensitive operational data and the mandated applications in a **private cloud/on-premises** environment, and run the **variable production-monitoring workload on a public cloud**.
- **Justification from the scenario:** variability → public cloud **rapid elasticity** handles production-hour spikes without permanent hardware; sensitivity/mandate → private segment retains **control, security, compliance**.
- Connect the two with **secure networking and unified management**; use **cloud bursting** so the private baseline extends to the public cloud at peak.

**(b) Advantages and limitations of this architecture vs a completely private cloud. [3]**
**Model answer:**
- *Advantages:* elasticity for bursty load (pay only for peak usage); no over-provisioning of hardware for peaks; faster deployment of new monitoring components; economies of scale of the provider.
- *Limitations vs all-private:* **integration complexity** (secure links, consistent identity/monitoring across environments); data must cross the boundary (egress cost, latency, governance); split **operational skill** and tooling; dependency on provider connectivity.
- *Trade-off line:* hybrid trades some control and simplicity for elasticity and cost efficiency — a completely private cloud keeps full control but forces capacity sized to peak (higher CapEx, lower utilisation).

**(c) Would adopting multiple public cloud providers necessarily improve availability? Justify. [4]**
**Model answer:**
- **No — not necessarily.** Multi-cloud *can* reduce the blast radius of a single provider outage, but only if the architecture actively supports cross-provider failover.
- Requirements: workloads **portable** (containers/IaC), data **replicated or synchronised across providers**, **common identity and monitoring**, and tested failover procedures.
- Without these, multi-cloud *adds* failure modes: **integration complexity, inconsistent IAM/security, cross-cloud latency and egress cost, split operational skills** — a second provider that cannot take traffic during an incident improves nothing.
- Conclusion: availability improves with **redundant, tested architecture**, not merely with a second bill. Multi-cloud is a resilience *strategy*, not a resilience *guarantee*.

## Q2. Cloud economics and evolution [10]

**(a) Why is cloud an evolution of distributed computing rather than a new paradigm? [3]**
**Model answer:**
- Cloud is a **convergence of existing technologies**: distributed computing lineage (parallel computing → clusters → grids), web technologies, web services/SOA, and virtualization, progressively modified to contemporary needs.
- Every ingredient pre-exists cloud: virtualization (1960s mainframes; VMware 1999), utility-style billing ideas, grid computing over the Internet.
- What is "new" is the **integration and scale**: self-service APIs, metered pay-as-you-go economics, rapid elasticity at data-centre scale — a business-model and engineering consolidation, not a fundamentally new computing principle.

**(b) CapEx vs OpEx in cloud adoption. [3]**
**Model answer:**
- **CapEx (capital expenditure):** upfront purchase of servers, data-centre space, licences; assets are owned and depreciated; capacity must be bought **ahead of demand** (sized for peaks).
- **OpEx (operational expenditure):** pay-as-you-go consumption billed per hour/second/GB; **no upfront investment**; cost scales with actual usage; stops when resources stop.
- Cloud shifts IT spending CapEx→OpEx: agility (try/terminate) and elasticity, at the price of ongoing per-unit charges and less asset ownership. For a bursty workload OpEx is usually cheaper overall; for a large steady workload the per-unit premium can exceed amortised CapEx.

**(c) Predictable workloads: is pay-as-you-go necessarily more economical than owning? [4]**
**Model answer:**
- **No.** Pay-as-you-go carries a per-unit premium in exchange for elasticity. With **highly predictable, steady, high-utilisation** workloads, that flexibility is never needed, so the premium is pure overhead.
- Cost logic: owning (or **reserved/Savings-Plan** pricing) amortises to a lower effective hourly rate over years; the crossover depends on utilisation rate, horizon and hardware refresh cycles.
- Numbers-illustration: 10 instances × 8,760 h × $0.10/h on-demand = **$8,760/yr**; at a reserved effective $0.06/h = **$5,256/yr** (≈40% saving) — steady load justifies commitment.
- Conclusion: choose pay-as-you-go for **variable/spiky** demand; choose commitment (reserved instances, or owned infrastructure) for the **steady base**. Most real organisations mix: committed baseline + elastic peak.

## Q30 (new). NIST characteristics in a private deployment [10]

**Scenario:** A company runs a "private cloud" that is 20 virtualised servers behind a manual ticket-based request process; staff wait days for VMs; no usage billing; capacity is statically assigned.

**(a) Which of the five NIST essential characteristics are missing? [3]**
- **On-demand self-service** — provisioning needs tickets/human intervention (missing).
- **Rapid elasticity** — capacity statically assigned; no automatic scale in/out (missing).
- **Metered/measured service** — no usage-based billing or metering (missing).
- **Resource pooling** — partially present (virtualization pools server hardware) but without dynamic allocation; **broad network access** likely present via the network. State explicitly which are present vs missing.

**(b) What must be added to make this a true private cloud? [4]**
- **Self-service portal/API** + approval policies so users provision without administrator intervention.
- **Orchestration/VIM layer** (e.g., OpenStack/OpenNebula/vCenter automation) for dynamic placement, quotas and lifecycle automation.
- **Metering & chargeback** — collect usage (CPU/memory/storage hours), publish cost dashboards.
- **Automation of scaling and lifecycle** (templates, auto-decommission of idle VMs), plus governance: tagging, quotas.

**(c) Should they instead buy public cloud? Give one argument each way. [3]**
- *For:* elasticity and zero maintenance burden; faster delivery; pay only for use; provider compliance certifications.
- *Against:* sensitive/regulatory data may mandate on-prem control; steady predictable load may be cheaper owned; exit/egress and skills.
- Conclusion line: a **hybrid** (private baseline + public burst) or a properly automated private cloud both fix the deficiencies; the deployment model is not the defect — the missing automation is.

## Q31 (new). Community cloud and provider selection [10]

**Scenario:** Seven hospitals want a shared imaging-archive platform. Each must meet the same health-data regulation, wants cost sharing, but cannot use a fully public multi-tenant service; one larger hospital is willing to host first.

**(a) Which deployment model fits best and why? [3]**
- **Community cloud** — infrastructure **shared by organizations with common concerns** (here: one regulation, one security baseline, shared mission), access **restricted to participating members**.
- Distinguish: not public (open to anyone), not single-tenant private (seven organizations share), can be owned/operated jointly or by a third party — both are allowed by the definition.

**(b) Advantages and two risks of the community model. [3]**
- *Advantages:* cost sharing; compliance requirements shared and uniformly enforced; collaboration (shared analytics/imaging standards); better control than public.
- *Risks:* **shared governance** (who decides upgrades, budgets, policies); smaller scale → weaker economies of scale and capacity headroom than public; inter-organisation legal agreements needed.

**(c) Three criteria for choosing who operates it. [4]**
- **Compliance capability** — certifications, audit rights, data-residency guarantees matching the health regulation.
- **Operational maturity** — HA/DR record, monitoring, security operations, incident response.
- **Commercial terms** — cost-sharing model, exit/portability clauses, data export, SLA with penalties/credits.
- Plus: scalability headroom for new members; integration/identity federation with hospital systems.

---

# Module 2 — Virtualization and Containers

## Q3. Hypervisor and virtualization architecture [10]

**(a) Role of a hypervisor in hardware virtualization. [3]**
- A hypervisor (**VMM**) is software/firmware/hardware that **creates, runs and manages VMs**; the machine it runs on is the host, each VM a guest.
- It presents each guest a **complete virtual platform** (virtual CPU, memory, devices), **multiplexes physical resources** across guests, and **enforces isolation** (a guest's failure or access cannot affect others).
- It must do this with **fidelity** (guest behaves as on real hardware) and **efficiency** (most instructions execute natively; only sensitive/privileged operations are intercepted).

**(b) Differentiate full virtualization, para-virtualization, hardware-assisted virtualization. [4]**
- **Full virtualization (binary translation):** guest OS **unmodified**; hypervisor **dynamically rewrites sensitive instructions** at runtime (guest deprivileged to Ring 1; VMM at Ring 0). Any OS incl. Windows; overhead from translation + device emulation. Example: VMware ESX.
- **Para-virtualization:** guest **kernel modified** to issue **hypercalls** for privileged operations — no runtime trap-and-translate; better performance; needs source access (historically open-source guests) or para-virtualized drivers. Example: Xen PV.
- **Hardware-assisted:** CPU adds a privilege level **below Ring 0 ("Ring −1", Intel VT-x/AMD-V)**; privileged/sensitive instructions **trap automatically**; guest unmodified; near-native performance; basis of KVM, Hyper-V, modern Xen.
- Summarise with: guest modified? (no / yes / no) · mechanism (translate / hypercall / hardware trap) · performance (overhead / better / best).

**(c) How are resource isolation and sharing achieved with multiple guests on one server? [3]**
- **Isolation:** each VM has private virtual hardware (vCPU, RAM, virtual disks/NICs); memory and device state are per-VM; the hypervisor mediates **all** privileged operations; a guest crash does not propagate.
- **Sharing:** the hypervisor **schedules physical CPUs across vCPUs** (time-slicing — enabling CPU overcommit), maps guest physical memory onto host memory (with reclaim techniques), and routes I/O through virtual devices to shared physical hardware.
- Controls: shares/limits/reservations per VM; overcommit guarded by CPU Ready time (<~5%) and memory reclaim (sharing → ballooning → compression → swap as last resort).

## Q4. Docker and container internals [10]

**Scenario:** Deploy a web application identically across development, testing and production using Docker.

**(a) Role of Dockerfile, image, container and registry in the deployment process. [4]**
- **Dockerfile:** text recipe of build instructions (`FROM` base image; each instruction adds a layer). Ensures **reproducibility** (same file + context → same image), **self-documents** the build, enables **layer caching** (copy dependency files before code), and automates CI/CD builds.
- **Image:** the resulting **read-only, layered, immutable template** — application + all dependencies baked in ("blueprint of the container environment"). Versioned by tags.
- **Container:** a **running instance of the image** with a **writable layer** on top (copy-on-write). The identical image yields identical behaviour in dev, test and prod — killing "works on my machine".
- **Registry:** stores and distributes images (Docker Hub public, or private). CI pushes the built image; every environment **pulls the same tag** — the handoff artefact is the image, not the code tree.
- Flow: `docker build` (Dockerfile→image) → `push` to registry → `pull`/`run` as containers in each environment.

**(b) How namespaces and cgroups support isolation and resource management. [3]**
- **Namespaces** wrap global kernel resources so processes see **their own isolated instance** (own PIDs, network stack, mounts) — *"limit what you can see"*; e.g., `ps` inside a container shows only its processes.
- **cgroups** aggregate processes into groups with **enforced resource limits** (CPU, memory, disk I/O) — *"limit how much you can use"*; a container that exceeds its ceiling is throttled or OOM-killed while the rest of the host is protected.
- Together they let many containers **share one kernel** safely: namespace = view isolation; cgroup = resource protection. Both are required (isolation without limits still allows noisy-neighbour starvation).

**(c) Why containers over VMs for microservices — and when VMs remain preferable? [3]**
- Containers: **seconds to start** (no guest OS boot), **MB-sized** images → high density (many microservices per host), **immutable, portable** packaging → identical dev/test/prod, per-service scaling and independent deployment — the microservices operational model.
- VMs remain preferable when: a **different OS/kernel** is required (Windows guest on Linux host); **stronger isolation** is mandated (separate kernel — compliance, multi-tenant boundaries); software needs **kernel modules/hypervisor-level control**; legacy/licensed monoliths; hardware-level tenancy requirements.
- Balanced conclusion: containers optimise density and velocity; VMs optimise isolation and OS flexibility — many platforms combine both (VMs under, containers above).

## Q5. Kubernetes orchestration [10]

**Scenario:** Multiple container replicas of a payment service; some containers fail; application traffic increases significantly.

**(a) How does Kubernetes maintain the desired number of replicas? [3]**
- The **Deployment** object declares the desired state (`replicas: N`); its **ReplicaSet controller** runs a **reconciliation loop**: *observe actual replicas → compare with desired → act* (create replacement pods when containers/nodes fail, delete extras when scaled in) → repeat.
- Failed pods are replaced **automatically** (self-healing) — the controller does not care *why* a pod died; it only drives actual → desired.
- The loop is **declarative**: the YAML is the source of truth; no one issues "start pod" commands.

**(b) Role of Pods, Nodes, Deployments and Services in this architecture. [4]**
- **Pod:** smallest deployable unit — one or more containers sharing network/storage; **ephemeral** (IPs change; pods are replaced, not repaired).
- **Node:** a worker machine (VM/bare metal) running **kubelet, kube-proxy, container runtime**; hosts pods; capacity defined by allocatable CPU/memory.
- **Deployment:** manages ReplicaSets to provide **declarative updates, scaling, rolling updates and rollbacks** for the payment service.
- **Service:** **stable virtual IP/DNS name** that **selects pods by label** and load-balances across them — clients keep one endpoint while pods churn underneath (types: ClusterIP internal, NodePort external).
- Data flow: client → Service (stable address) → currently-ready pods (selected by labels) on nodes, created/managed by the Deployment.

**(c) How do CPU requests and limits influence scheduling and consumption? [3]**
- **Request** = the **reserved, guaranteed share** the **scheduler** uses for placement: a pod fits a node only if allocatable CPU/memory ≥ its requests; requests also set the cgroup scheduling weight.
- **Limit** = the **runtime hard cap**: exceeding the CPU limit → **throttling**; exceeding the memory limit → **OOM kill**.
- Consequence: requests drive *packing* (too high → pods Pending, capacity wasted; too low → node overcommitted and throttled under load); limits protect neighbours. **A request is not a cap** — between request and limit a pod may burst.

## Q6. Container resource calculation [10]

**Scenario:** Cluster of **3 nodes, each 4 vCPU allocatable for scheduling**; each application pod **requests 500m CPU**.

**(a) Theoretical maximum pods based on CPU requests. [2]**
- Per node: 4 vCPU = 4000m; 4000m ÷ 500m = **8 pods/node**.
- Cluster: 8 × 3 = **24 pods**.

**(b) If 20 pods are already running, how many additional pods can be scheduled? [2]**
- 24 − 20 = **4 additional pods** (assuming all existing pods request 500m and nodes have no other consumers).

**(c) Why may actual scheduling capacity and performance be lower than the calculated maximum? [3]**
- **System reservations:** kubelet/OS/eviction reserves reduce *allocatable* below capacity; add-ons (CNI, DNS, metrics) consume capacity.
- **Other consumers and limits:** maxPods-per-node limit; system pods; non-uniform requests (bigger pods don't fit leftover space — **fragmentation**); nodes unschedulable (taints/maintenance).
- **Performance ≠ capacity:** pods may burst above requests; packed nodes cause CPU throttling/memory pressure/noisy neighbours; the scheduler also spreads replicas across nodes for resilience instead of packing to the max.

**(d) Importance of memory requests, limits and cgroups. [3]**
- **Memory request** reserves RAM for scheduling (memory cannot be overcommitted by time-slicing — a page is allocated or not).
- **Memory limit** is enforced by **cgroups**: exceeding it triggers the kernel **OOM-killer** — protects other containers from a leaking one.
- cgroups translate requests/limits into kernel enforcement (throttle CPU, cap memory), so declared values are real guarantees/caps, not documentation; be **more conservative with memory** than CPU (OOM kills are fatal to the pod).

## Q32 (new). Server consolidation numerical [10]

**Scenario:** A firm runs **40 physical servers at 12% average CPU utilisation**. Power/cooling per server ≈ $1,200/yr. After virtualization, target host utilisation is **70%**.

**(a) Workload "size" and number of target hosts. [3]**
- Total work = 40 × 0.12 = **4.8 servers' worth** of load.
- Hosts needed = 4.8 ÷ 0.70 = 6.86 → **7 hosts** (round up; 7 × 70% = 4.9 ≥ 4.8).

**(b) Cost/energy reduction. [2]**
- Servers: 40 → 7 (**82.5% reduction**). Power+cooling: $1,200 × 33 fewer = **≈$39,600/yr saved** (before new host costs — mention the caveat).

**(c) Why is 100% utilisation not targeted? [2]**
- Headroom absorbs **peaks and spikes** (p95 ≫ average; averages mislead); room for **one host failure** (N+1) and maintenance; avoid CPU-ready/thrashing degradation.

**(d) Two consolidation risks. [3]**
- **Noisy neighbours / contention** — mixed workloads collide at peak; mitigate with shares/limits/reservations and monitoring (CPU Ready).
- **Blast radius** — many VMs on one host: a host failure hits many services; mitigate with HA clusters, anti-affinity, N+1. Also: licensing per VM and I/O contention on shared storage.

## Q33 (new). Hypervisor choice and virtualization levels [10]

**Scenario:** A team runs 3 VMs on **VirtualBox on a Windows desktop** for development; production runs **200 VMs on a dedicated Linux cluster**.

**(a) Identify the hypervisor types in each environment. [2]**
- Desktop: **Type 2 (hosted)** — VirtualBox runs as an application on the host OS (Windows).
- Production: **Type 1 (bare-metal)** — hypervisor runs directly on hardware (e.g., KVM/ESXi).

**(b) Why is the hosted type acceptable for development but not production? [3]**
- Hosted adds an **extra layer** (host OS) → overhead, weaker isolation, dependence on desktop OS patching/uptime — tolerable for a dev workstation.
- Production needs **density, performance, HA, live migration, central management, hard multi-tenant isolation** — bare-metal provides these; hosted does not.

**(c) At which level of the virtualization stack does a container run? Compare with VM level. [3]**
- Containers = **operating-system level** virtualization: isolated user-space environments **sharing one kernel** (namespaces/cgroups).
- VMs = **hardware-abstraction level**: each guest gets virtual hardware and its **own kernel**.
- Consequence: containers lighter/faster/denser but weaker isolation; VMs heavier but stronger isolation — hence containers sit *inside* VMs in hardened setups.

**(d) When would ISA-level (emulation) virtualization be the only option? [2]**
- When **guest and host instruction sets differ** (ARM image on x86, legacy mainframe/PowerPC apps) — true virtualization requires the same ISA; only emulation can execute the foreign binary (with heavy performance cost).

## Q34 (new). Memory overcommitment [10]

**Scenario:** Host has **32 GB RAM**; four VMs are each configured with **10 GB** (40 GB total). VMs are mostly idle; occasionally two VMs run memory-heavy batch jobs.

**(a) Overcommit ratio and its meaning. [2]**
- 40 ÷ 32 = **1.25:1** — 25% more memory configured than physically present; workable *if* guests don't all touch their full allocation simultaneously.

**(b) Which reclamation techniques apply, in order? [4]**
1. **Page sharing (TPS/KSM):** deduplicate identical pages (e.g., common OS pages across the four similar VMs) — cheapest; inter-VM sharing often disabled by default for security, intra-VM remains.
2. **Ballooning:** guest driver reclaims memory from the **guest's own idle processes** — the guest OS chooses what to evict (cooperative; preferred under pressure).
3. **Compression:** hypervisor compresses pages rather than swapping (slower than RAM, far faster than disk).
4. **Host swapping:** last resort — disk is ~100,000× slower than RAM; performance collapse if reached.

**(c) What happens when both batch jobs peak simultaneously? [2]**
- Demand approaches/exceeds 32 GB: sharing/ballooning exhaust → compression → swapping; guests slow dramatically (swap I/O); risk of host-level memory exhaustion; mitigations: reservations for critical VMs, DRS/migration to another host, or reducing configured memory.

**(d) Why is memory harder to overcommit than CPU? [2]**
- **CPU is time-sliced** — cycles are a renewable resource; waiting vCPUs get the next slice.
- **Memory is not time-sliceable** — a page is allocated or not at any instant; the hypervisor must physically **reclaim or deduplicate** memory, each technique costing performance.

## Q35 (new). Emulation scenario [10]

**Scenario:** A lab must run a legacy **PowerPC-architected** control application and an ARM-built Android test image, both on existing x86 servers.

**(a) Why can neither full virtualization nor containers run these directly? [3]**
- Full/hardware-assisted virtualization is **same-ISA only** — guests must share the host's instruction set (x86 here). Containers share the **host kernel and ISA** — they isolate but do not translate instructions. PowerPC/ARM binaries cannot execute natively on x86.

**(b) Which technique works, and what is its mechanism? [3]**
- **Emulation (ISA-level virtualization)**: software reproduces the behaviour of the foreign CPU — guest sees a virtual hardware profile decoupled from the physical host; guest instructions are interpreted or translated (dynamic binary translation, e.g., QEMU-TCG; Rosetta 2). Nothing runs natively — no host privilege needed.

**(c) Performance expectation and mitigation. [2]**
- Expect **significant overhead** (every guest instruction interpreted/translated; even cached DBT is slower than native). Mitigate: use DBT-based emulators (cache translated blocks), pre-compiled translations, or re-platform/rebuild the app for x86 if source is available.

**(d) Emulation vs virtualization in one comparison line each. [2]**
- Emulation: cross-ISA possible; slow; guest unaware of host; used for legacy/cross-platform.
- Virtualization: same-ISA only; near-native; hypervisor traps privileged ops; the standard cloud mechanism.

## Q36 (new). Kubernetes pod Pending troubleshooting [10]

**Scenario:** After adding memory limits, several new pods of a service stay **Pending**; `kubectl describe pod` shows `FailedScheduling: insufficient memory`. Nodes report 60% memory in use.

**(a) Why does a node with 40% "free" memory still refuse pods? [3]**
- Scheduling is based on **requests (reserved)**, not live usage. Other pods' *requests* already reserve most of allocatable memory; the live 60% metric is irrelevant to the filter stage. The new pods' requests exceed remaining *allocatable* → no node passes the filter → **Pending**.

**(b) Distinguish request vs limit for memory in this incident. [3]**
- **Request:** reserved at scheduling (guaranteed share; drives placement). **Limit:** runtime cap; exceeding it → **OOMKilled**. Raising limits alone would not fix Pending — placement uses requests. Raising *requests* beyond allocatable keeps pods Pending; lowering requests frees reservation but risks pressure.

**(c) Three remediation options. [2]**
- Right-size requests (p95 + 20–30% headroom, not 2× guesses); add worker nodes / cluster autoscaler; consolidate (remove zombies, lower over-requested pods); taint cleanup if nodes are wrongly tainted.

**(d) Why did adding limits trigger this? [2]**
- Team likely converted "limits = requests" or raised requests while defining limits — reservations grew; also cgroups now enforce the cap, exposing previously hidden overcommit. Lesson: **declare requests from measurement**, keep limits above requests with headroom.

## Q37 (new). Pod capacity — CPU vs memory bound [10]

**Scenario:** Node: 4 vCPU capacity, 3.7 vCPU allocatable; 8 GiB capacity, 7.5 GiB allocatable. Pod: requests 250m CPU, 1 GiB memory.

**(a) Pods per node. [3]**
- CPU fit: 3700m ÷ 250m = 14.8 → 14.
- Memory fit: 7.5 ÷ 1 = 7.5 → 7.
- Pods per node = **min(14, 7, maxPods≈110) = 7 pods — memory-bound**.

**(b) Which resource is the binding constraint, and what does that tell you? [2]**
- **Memory** binds. CPU has spare reservation (14 − 7 = 7 pods' worth) — right-sizing should target memory (or choose smaller-memory pods / bigger-memory nodes).

**(c) If the pod's memory request doubles, what changes? [2]**
- Memory fit: 7.5 ÷ 2 = 3.75 → 3 pods/node; binding constraint still memory; capacity nearly halves (7 → 3).

**(d) Where do limits fit in this calculation? [3]**
- They don't affect *scheduling* at all — only requests do. Limits act at **runtime** via cgroups (throttle/OOM-kill). Miscalculating capacity using limits overpacks nodes; using requests understates *actual* usage but correctly states *guaranteed* capacity. Capacity planning uses requests; performance engineering uses limits.

## Q38 (new). Swarm quorum and routing [10]

**Scenario:** A Swarm cluster has **5 manager nodes**. During a data-centre network incident, two managers become unreachable simultaneously. Separately, a browser calls port 80 on a worker node that runs no web-task copy.

**(a) Does the cluster keep operating? [2]**
- Failures tolerated = ⌊(5−1)/2⌋ = **2** → yes, 3 managers retain **Raft majority**; scheduling and reconciliation continue. A third manager loss would freeze quorum (reads OK, writes/scheduling halt).

**(b) Why is 4 managers no better than 3? [3]**
- Quorum = majority: 3 managers tolerate ⌊(3−1)/2⌋=1; 4 tolerate ⌊(4−1)/2⌋=1 — **still one failure**. The extra manager adds cost and split-vote risk without adding tolerance. **Always use odd counts** (3, 5, 7).

**(c) What does the browser get, and which mechanism answers? [3]**
- A successful response: the **ingress routing mesh** — a published port is open on **every** node; the receiving node forwards the connection over the overlay to an available task, **even on another node** (Virtual IP + internal load balancing).

**(d) Service vs task in one line each. [2]**
- **Service:** the declared definition — image, replica count, ports ("run 3 copies of api:v1"); the desired state.
- **Task:** one scheduled instantiation of the service (freshcart_api.1…3); each task runs exactly one container; the scheduler places and replaces tasks.

## Q39 (new). Declarative self-healing scenario [10]

**Scenario:** An engineer manually deletes three pods of a 5-replica Deployment to "free memory", then edits the running replica count via `kubectl scale` to 2. Overnight, traffic returns and the team's GitOps file still says `replicas: 5`.

**(a) What happens immediately after the manual deletions? [2]**
- The **ReplicaSet controller** detects actual (2) vs desired (5) and **recreates 3 pods** — manual deletion does not persist; the declarative loop restores the declared state.

**(b) What does the overnight GitOps reconciliation do? [2]**
- The pipeline compares the cluster with Git (`replicas: 5`), detects drift from the manual `scale 2`, and **re-applies the file** → scale back to 5. Git is the source of truth; manual changes are reverted (drift reconciliation).

**(c) Why is this behaviour desirable? Contrast with imperative management. [3]**
- Declarative: **idempotent, self-healing, auditable** — desired state in a reviewed file; every change is a commit; drift is detected and corrected; rollback = revert. Imperative: command sequences depend on order and prior state; no automatic correction; undocumented manual changes ("who changed this?").

**(d) One legitimate exception and its control. [3]**
- Emergency break-glass scaling during an incident; control it by **committing the emergency change to Git immediately** (or using a documented override window), then reviewing — keeping audit trail intact rather than fighting the reconciler.

---
# Module 3 — Infrastructure as a Service (IaaS)

## Q7. AWS infrastructure architecture [10]

**Scenario:** A three-tier web application (web, app, database) to be deployed on AWS.

**(a) How EC2, VPC, subnets and security groups implement the architecture. [4]**
- **VPC:** dedicated logically isolated network with a chosen IP range — your private section of AWS.
- **Subnets:** public subnets (route to an **Internet Gateway**) host the load balancer/web tier; **private subnets** (no IGW route) host app and database tiers; span **≥2 AZs** for zone failure tolerance.
- **EC2 instances:** launched from a corporate-standard **AMI** into the appropriate subnets; an **Auto Scaling group** maintains web/app capacity; **bootstrapping via cloud-init** user-data.
- **Security groups:** stateful instance firewalls chained by rule-reference — ALB SG allows 443 from the internet; web SG allows traffic *only from the ALB's SG*; app SG only from web SG; DB SG only from app SG (default deny).
- Result: layered isolation — only the web tier is internet-reachable; DB has no internet path.

**(b) Role of Regions, AZs and Edge Locations. [3]**
- **Region:** independent geographic cluster of AZs — choose for **data residency, latency to users, service availability**; regions are isolated from each other (blast-radius and compliance boundary).
- **AZs:** discrete data centres within a region, physically separated (power/flood/distance) but **low-latency linked** — deploy across AZs to survive one-location failure.
- **Edge locations:** CDN (CloudFront) cache sites close to users — serve static content with low latency; (course extension: Local Zones/Wavelength/Outposts extend compute closer or on-premises).

**(c) Storage selection for application files, transactional DB storage, shared file storage. [3]**
- **Application/static files → S3 (object):** durable, cheap, HTTP-accessible, auto-replicated in-region.
- **Transactional database volumes → EBS (block):** persistent, low-latency block device attached to EC2/RDS (Provisioned-IOPS SSD for sustained IOPS).
- **Shared file storage across many servers → EFS (file):** managed, elastic, concurrently mounted NFS-style.
- Justification pattern: *access pattern decides* — whole-object reads (S3), raw disk blocks for a DB (EBS), POSIX shared files (EFS); Glacier for archive tier.

## Q8. IAM and security [10]

**Scenario:** Developers, database administrators and application operators access AWS resources.

**(a) Authentication vs authorization. [2]**
- **Authentication** proves *who you are* (password/MFA/key/federation) — yields a confirmed identity only.
- **Authorization** decides *what that identity may do* — which actions on which resources, evaluated **on every request** against policies.

**(b) How users, roles and policies enforce least privilege. [4]**
- **Users** = human identities (with MFA); **groups** bundle job-function users.
- **Policies** = JSON rule books (Effect/Action/Resource/Condition) attached to users/groups/roles; **explicit Deny overrides Allow**; scoped, e.g., developers: their project's EC2/S3 actions only; DBAs: RDS/DB actions; operators: monitoring + start/stop.
- **Roles** = assumable identities **without permanent credentials** — short-lived auto-rotating credentials for services/machines (EC2 reading one bucket; Lambda writing one table) — removes hardcoded keys.
- Enforcement chain: request → authentication → policy evaluation (identity + resource policies, SCP guardrails) → allow/deny; least privilege = smallest sufficient permission set, reviewed regularly.

**(c) Why is admin-for-every-developer a poor practice? [2]**
- Violates least privilege; a compromised developer credential = whole-account compromise (huge blast radius); no separation of duties; accidental production damage possible; audits cannot attribute intent. (Also: admins can tamper with logs/monitoring.)

**(d) Reviewing and improving access over time. [2]**
- Periodic **access reviews**: IAM access-analyser / last-used data to remove unused permissions; replace wildcard actions; enforce MFA and rotate keys; adopt roles for machines; policy-as-code review in Git + SCP guardrails; incident-driven tightening.

## Q9. Cloud storage and data services [10]

**Scenario:** An e-commerce business stores product images, transactional records, and customer activity logs.

**(a) Compare block, file and object storage. [3]**
- **Block:** lowest level — fixed-size numbered blocks presented as a raw disk (SAN/iSCSI); mounted by one machine; best for DBs/boot volumes (EBS).
- **File:** OS-level hierarchy of files/folders over the network (NAS/NFS); **concurrent multi-machine sharing** (EFS).
- **Object:** flat containers (buckets) holding whole objects (data+metadata+key) accessed by **API over HTTP**; unlimited scale, cheap, server-independent (S3); no partial updates — whole-object PUT/GET.

**(b) Recommend AWS services for each requirement. [3]**
- Product images → **S3** (static assets, CDN-fronted, versioned).
- Transactional records → **RDS on EBS volumes** (relational, ACID, Multi-AZ); hot activity analytics later via Redshift/EMR.
- Customer activity logs → **S3** (append-only, cheap, lifecycle → **Glacier** for archive) — or a managed streaming/NoSQL pipeline if queried live (DynamoDB/Kinesis).

**(c) When is a relational DB preferable to NoSQL? [2]**
- Structured, stable schema; **ACID transactions** (orders, payments, inventory); complex joins and reporting; strong consistency requirements. NoSQL wins on flexible schema, massive horizontal scale, key-value lookups.

**(d) Purpose of distributed processing services (Hadoop HDFS, EMR). [2]**
- Store and process datasets **too large for one machine**: HDFS replicates blocks across cluster nodes; MapReduce/Spark process partitions **in parallel near the data**; EMR = managed Hadoop/Spark clusters for batch analytics (e.g., nightly activity-log aggregation from S3).

## Q40 (new). VPC design and security layers [10]

**Scenario:** A payments web app: internet-facing load-balanced web tier, internal API tier, PCI-scoped database. Auditors demand network segmentation evidence.

**(a) Design the VPC. [4]**
- VPC CIDR; **public subnets** (2 AZs) for ALB; **private app subnets** for API instances; **isolated data subnets** for RDS Multi-AZ; route tables: public → IGW; private → NAT gateway (patches only); data → no internet route.
- Everything duplicated across **2 AZs**; SGs chained (ALB→web→api→db); NACLs as coarse subnet guard (e.g., block non-app ports); VPC flow logs enabled for audit evidence.

**(b) Security group vs NACL — compare. [3]**
- **SG:** instance level, **stateful** (return traffic auto-allowed), allow-only rules, evaluates all rules, can reference other SGs — the primary control.
- **NACL:** subnet level, **stateless** (return traffic needs explicit rules), numbered allow **and deny** evaluated in order — coarse boundary/quick block (e.g., deny a bad CIDR).

**(c) How does this satisfy the auditor? [2]**
- Evidence: diagram + SG/NACL rules showing DB reachable only from API SG; no IGW route in data subnets; flow logs; PCI scope reduced to isolated subnets.

**(d) One residual risk and mitigation. [1–2]**
- Misconfigured SG (0.0.0.0/0 on SSH) — mitigate: no direct SSH (SSM/bastion), IaC review, SCP guardrails, config-drift alerts.

## Q41 (new). IAM roles vs long-lived keys [10]

**Scenario:** An application on EC2 reads from S3. The team baked an IAM user's **access key into the AMI**. Audit flagged it.

**(a) Why is a baked-in key dangerous? [3]**
- **Long-lived, exportable, shared** credential copied into every instance from that AMI; leakage = permanent account-level access until rotated; rotation requires rebuilding every instance; no attribution (who used the key?); violates least privilege (probably admin-ish scope).

**(b) Correct pattern and its mechanics. [3]**
- Attach an **IAM role** to the instance profile: the app retrieves **temporary, auto-rotating credentials** from the instance metadata service; policy grants only `s3:GetObject` on the specific bucket; no secrets on disk; role change = policy edit, no redeployment.

**(c) Authentication vs authorization in this flow. [2]**
- Authn: instance identity proven via role assumption (signed temporary credentials). Authz: every S3 API call evaluated against the role's policy (scoped Allow; Deny wins).

**(d) Cleanup checklist. [2]**
- Deactivate/delete the leaked key; create role; rebuild AMI without secret (use SSM Parameter Store/Secrets Manager for any remaining config); scan logs for key misuse; add SCP/guardrail denying static keys on instances; add to IaC so it never returns.

## Q42 (new). Storage selection with cost tiers [10]

**Scenario:** Imaging platform: 50 TB of active study images (random reads all day), 500 TB of last-year studies (occasional reads), 2 PB of >3-year-old studies (compliance archive, rarely read).

**(a) Match each tier to a storage class. [3]**
- Active 50 TB → **S3 Standard** (frequent access, ms latency) behind CDN.
- 500 TB occasional → **S3 Standard-IA / Intelligent-Tiering** (lower storage cost, retrieval fee per access).
- 2 PB archive → **Glacier / Deep Archive** (lowest cost; retrieval in hours — acceptable for compliance requests).

**(b) Why not keep everything on one tier? [2]**
- Cost scales with tier: paying Standard prices for 2 PB of read-once-a-year data wastes most of the budget; tiering trades retrieval latency for ~10–20× lower storage cost, matching access frequency.

**(c) Which access patterns would force a different choice? [2]**
- If archived studies needed **second-level** retrieval (active clinical use), Glacier's hours-level retrieval is unacceptable → keep in IA/Standard. If "shared file" semantics were needed (POSIX by many servers), S3 is wrong → EFS.

**(d) Durability/replication note. [3]**
- S3 replicates objects across multiple devices/facilities in the region (designed for very high durability); for compliance add **cross-region replication**, versioning, Object Lock (WORM) — and lifecycle policies that automate the tier transitions.

## Q43 (new). RDS Multi-AZ vs Read Replicas [10]

**Scenario:** An order service's RDS MySQL: nightly batch reporting slows the primary; last month an AZ incident caused 8 minutes of unavailability.

**(a) Diagnose each problem with the right feature. [3]**
- Reporting load → **Read Replicas** (asynchronous read-only copies; up to 5) — offload heavy reads.
- AZ outage → **Multi-AZ** (synchronous standby in a second AZ; automatic failover) — availability feature.

**(b) Why Multi-AZ does not fix the reporting problem — and vice versa. [3]**
- Multi-AZ standby is **not readable** and is kept synchronously for failover — it adds durability/availability, zero read throughput.
- Read replicas are **asynchronous** — replication lag means they are not a failover target (possible data loss) and add write-forwarding overhead, not automatic failover.

**(c) Design the combined architecture. [2]**
- Multi-AZ primary (failover safety) + 2 read replicas (reporting/analytics) in the same or another region; direct reporting jobs at replica endpoints; monitor replica lag.

**(d) Availability improvement estimate. [2]**
- Failover to the standby typically cuts AZ-failure downtime from minutes-to-hours to a brief automatic switch — helping meet e.g. 99.9% (43.2 min/month) targets; pair with multi-AZ app tier and health-checked failover end-to-end.

---

# Module 4 — PaaS and SaaS

## Q10. Service model selection [10]

**Scenario:** Startup app: customer portal, backend database, payment processing, image-processing function.

**(a) IaaS vs PaaS vs SaaS responsibility split. [3]**
- **IaaS:** customer manages OS/runtime/app/data; provider manages hardware + virtualization.
- **PaaS:** customer manages **app + data only**; provider runs runtime/OS/servers/scaling.
- **SaaS:** customer only uses/configures; vendor runs everything (customer owns data policy).
- Ladder: each step up transfers one more layer to the vendor; **data/identity accountability never transfers**.

**(b) Recommend service models per component. [3]**
- **Customer portal → PaaS** (standard stack, variable traffic, small team; platform handles scaling/patching).
- **Backend database → managed DBaaS (PaaS-style service, e.g., Azure SQL/RDS)** rather than self-run DB on IaaS.
- **Payment processing → SaaS** (buy a payment provider — standard business process; integrate via API; never build PCI scope yourself).
- **Image processing → FaaS/serverless** (event-driven per upload, short bursts, scale-to-zero cost).

**(c) How BaaS and FaaS reduce infrastructure management effort. [4]**
- **BaaS:** ready-made backend behind an SDK — auth, database, file storage, push — the startup *configures* rather than *operates*; scales with vendor; no backend servers, patches, or backups to run. Watch: lock-in, misconfigured security rules.
- **FaaS:** code runs only when an event occurs (upload → resize function); provider handles provisioning, scaling (incl. to zero), availability; billed per invocation. Watch: statelessness (state outside), 15-min-type runtime caps, cold starts.
- Combined effect: the team writes only product logic and client code; infra ownership shrinks to configuration — at the cost of vendor coupling and per-use pricing that must be modelled at scale.

## Q11. PaaS versus IaaS [10]

**Scenario:** An organization develops and deploys a custom enterprise application.

**(a) Can you build and deploy SaaS directly on IaaS without PaaS? [3]**
- **Yes.** SaaS is a *delivery model* (vendor runs an application for end customers); IaaS provides the infrastructure. A company can rent VMs (IaaS), install its app and offer it to customers — that application *is* SaaS from the customers' viewpoint.
- What PaaS adds is *operational convenience* (managed runtime, scaling, deployment pipeline) — it is not a prerequisite. Trade-off: on IaaS you own OS patching, scaling setup, backups — full control, full responsibility.

**(b) Compare IaaS and PaaS on flexibility, control, operations, scalability. [4]**
- **Flexibility:** IaaS — any OS/runtime/licence; PaaS — only supported runtimes/services (contract).
- **Control:** IaaS — root/kernel, network design; PaaS — app settings and plan size only.
- **Operational responsibility:** IaaS — patch OS/runtime, build scaling, manage backups; PaaS — provider patches, scales, heals; customer: app, data, cost.
- **Scalability:** IaaS — build (ASG/LB) and tune yourself; PaaS — platform autoscaling as a setting.
- One-line rule: *ask what you deploy and what you patch*: VM image → IaaS; code → PaaS.

**(c) When is choosing IaaS justified? [3]**
- Custom kernels/drivers, licensed OS images; specialised networking or GPU/HPC needs; workloads outside provider runtimes; steady very-high load where VM economics beat plan pricing; hard portability/exit requirements; strict compliance needing OS-level audit/control.

## Q44 (new). Serverless design for an image pipeline [10]

**Scenario:** Users upload product photos; each must be resized to 5 variants, virus-scanned, and catalogued. Uploads burst unpredictably; currently one always-on VM idles 90% of the day.

**(a) Why is FaaS the right core? [3]**
- **Event-driven, short, spiky** work matches FaaS exactly: each upload triggers a function; scaling is automatic per event incl. **to zero** between bursts; billing per invocation → the 90% idle cost disappears.

**(b) Design the flow. [3]**
- Upload → object storage event → **function 1** (resize variants → write back) → scan step (queued function/queue for slow scans) → cataloguer function writes metadata to a DB → CDN serves variants. Keep state in storage/DB (functions **stateless**); queue decouples slow steps; dead-letter queue for failures.

**(c) Three constraints to design around. [2]**
- **Runtime caps** (e.g., Lambda 15 min — split huge videos); **cold starts** (keep warm/miniimum memory or provisioned concurrency); **statelessness** (persist intermediate state outside); plus vendor trigger lock-in, observability needs (structured logs/traces).

**(d) When would this design be wrong? [2]**
- Steady very-high sustained volume (per-invocation pricing exceeds VMs); long-running jobs; tight runtime/GPU needs; strict VPC/latency coupling — then containers on an orchestrator or plain VMs fit better.

## Q45 (new). Five-model selection for a retail platform [10]

**Scenario (FreshCart-style):** storefront web+API; product-image resize on upload; mobile login/wishlist for a prototype; support-desk/CRM; legacy inventory system tied to one OS image.

**(a) Assign a model to each component with a one-line why. [5]**
- Storefront/API → **PaaS** (standard stack, variable traffic, small team).
- Image resize → **FaaS** (event-driven, short, spiky).
- Mobile login/wishlist → **BaaS** (standard auth/data backend without server code).
- Support desk/CRM → **SaaS** (buy standard business process; configure + integrate).
- Legacy inventory → **IaaS** (needs OS-level control no higher model offers).

**(b) Which single component carries the most lock-in risk and why? [2]**
- The **BaaS** prototype (and FaaS triggers): proprietary data models/APIs and vendor-specific event sources make exit costly; mitigate with containers/portable packaging where possible and exit-plan clauses.

**(c) Who patches the guest OS in each of PaaS and IaaS here? [2]**
- PaaS storefront: **vendor** patches OS/runtime. IaaS inventory: **you** patch the guest OS (and runtime) — the defining difference.

**(d) Name the accountability that never transfers in any model. [1]**
- **Data, identity and access** accountability remains with the customer in every model.

---
# Module 5 — Provisioning, Migration and Cloud Resource Management

## Q12. VM provisioning and image-based deployment [10]

**Scenario:** An organization needs to provision 50 identical application servers.

**(a) Steps involved in VM provisioning. [3]**
1. **Validate** — IAM permissions, quotas, parameters.
2. **Place** — scheduler picks a host with free capacity in the chosen AZ.
3. **Image** — root disk created from the AMI/template.
4. **Network** — interface, private IP from subnet, security groups attached.
5. **Boot** — hypervisor starts the VM (pending → running).
6. **Contextualise** — cloud-init applies hostname/keys/user-data software installs.
7. **Health** — status checks pass → running; app health check → registered behind the load balancer.

**(b) How image-based provisioning and templates improve consistency and speed. [3]**
- **Consistency:** one golden image (corporate patch level, agents, hardening) is the single source — every server identical; drift across 50 servers is designed out; compliance evidence is the image itself.
- **Speed:** boot from a pre-built image in minutes vs hours of manual installs; user-data handles per-instance customisation only; the same image feeds Auto Scaling for future capacity.
- **Governance:** versioned images (v2 → v3 rollouts, instant rollback to previous AMI); non-conforming instances become impossible by policy.

**(c) Role of tagging, quotas and governance at this scale. [4]**
- **Tags** (mandatory set: Owner/Project/Environment/CostCenter, applied at creation, one spelling): find owners, allocate costs per department, drive automation (stop dev at night), feed audits; enforced by policies refusing untagged creation.
- **Quotas:** cap service usage per account/region (vCPU limits) — contain mistakes (a runaway script asking for 500 instances) and protect the provider relationship; raisable deliberately.
- **Guardrails:** SCPs/budgets/alerts — what is never allowed (unapproved regions/types), spend thresholds (50/80/100%).
- **Governance loop:** define policy → enforce (IaC + guardrails) → monitor → report → adjust — with 50+ servers, manual tracking does not scale.

## Q13. VM migration [10]

**Scenario:** A financial organization migrates application infrastructure from an on-premises data centre to the cloud.

**(a) Differentiate cold, live and storage migration. [3]**
- **Cold migration:** moves a **powered-off** VM; no shared storage or CPU-compatibility checks; downtime = entire copy duration; can **cross hypervisors/clouds** (disk conversion).
- **Live migration:** moves a **running** VM between hosts; user-visible downtime **tens–hundreds of ms**; needs shared storage (or shared-nothing variant); normally same platform.
- **Storage migration:** moves a **running VM's disks** to a new datastore with **no interruption** (copy → mirror writes → switchover).

**(b) Recommend an approach for a business-critical application that cannot tolerate significant downtime. [3]**
- **Within the platform (e.g., during data-centre consolidation): live migration** — pre-copy keeps the VM running; downtime is milliseconds.
- **For the DC→cloud move itself:** live migration across hypervisors is generally not offered, so combine: **Repurchase/Rehost in stages** — replicate data continuously, stand up the cloud stack via IaC, cut over during a short planned window (cold-style cutover), or use hybrid pattern (Outposts/VPN) to shift gradually. Justify with: critical app ⇒ downtime budget decides the technique; plan cutover windows + rollback.

**(c) Technical challenges of live migration. [2]**
- Moving **large in-memory state** while it changes (dirty pages; converge only if dirty rate < bandwidth); **soft real-time** requirements of some VMs; CPU compatibility checks; network identity redirection; storage sync (shared-nothing).

**(d) How IaC helps reproduce migrated infrastructure consistently. [2]**
- The target environment is **declared in code** (Terraform/CloudFormation) — dev/test/prod are rebuilt identically in minutes; changes reviewed in Git; **idempotent re-apply**, drift detection; disaster recovery = re-run the code; new regions = same files, new provider variables.

## Q14. Infrastructure as Code [10]

**Scenario:** A company creates resources manually in the console; dev and production configurations have drifted; nobody knows what exists.

**(a) Concept and benefits of IaC. [3]**
- **IaC:** infrastructure described in **text files** that a tool turns into real resources; files live in Git.
- Benefits: **repeatability** (same file → same environment; idempotent applies), **speed** (dev/test/prod in minutes; teardown one command), **review/audit** (PRs, history, "who changed this?" answered), prevents **snowflakes** and "worked in test" mismatches, enables tested DR rebuilds.

**(b) How Terraform/CloudFormation maintain repeatable deployments. [3]**
- **Declarative** desired state; **plan/preview** shows exactly what will change before apply; **apply** converges reality to the file; **state** records what was created (CloudFormation manages it for you; Terraform's state file must be protected).
- Parameterised modules → same code, different environments (variables per env); applies are **idempotent** (second apply = no changes).

**(c) How version control, review and tagging support governance. [4]**
- **Version control:** every infrastructure change is a commit — history, blame, rollback = revert; branch protections on prod.
- **Review:** PR approvals + automated plan checks (policy-as-code: deny open security groups, unapproved regions) before merge.
- **Tagging:** mandatory key/value labels (Owner, Environment, CostCenter) applied by the code at creation — cost allocation, ownership lookup, automation (nightly stop of tagged dev), audit trails; enforced with SCP/IAM conditions.
- **Drift management:** scheduled plan runs detect out-of-band console edits → reconcile by re-apply or import. Loop: define → enforce → monitor → report → adjust.

## Q46 (new). IaC drift incident [10]

**Scenario:** A Terraform-managed security group allows only port 443. Someone edited it in the console to open port 22 for "quick debugging" and forgot. A week later an audit flags exposure.

**(a) What is this problem called and why did IaC not prevent it? [2]**
- **Drift** — real resources changed outside the code. IaC does not prevent out-of-band edits by itself; the file still says 443 while reality says 443+22 (state ≠ reality until reconciled).

**(b) How would the team detect it? [2]**
- Scheduled/CI **plan runs** showing an unexpected in-place update (rule 22 deletion diff); cloud config-drift detectors/audit logs (who/when via CloudTrail); alert on changes to critical resources outside pipelines.

**(c) Correct remediation. [2]**
- Remove the console rule (or `terraform apply` to re-converge — either way the code wins), delete the temporary need by providing a proper access path (bastion/SSM), then commit any legitimate change through the pipeline with review.

**(d) Three preventive controls. [2]**
- Read-only console/IAM for prod humans; **SCP guardrail** denying risky SG rules; mandatory PR + plan-approval workflow; drift alerts scheduled; tagging ownership so someone is accountable.

**(e) One-line lesson. [2]**
- The declaration is the source of truth; if reality can differ silently, governance has failed — make Git the *only* write path (GitOps).

## Q47 (new). Live migration numerical [10]

**Scenario:** VM memory **M = 16 GiB**; usable network bandwidth **B = 2 GiB/s**; memory dirty rate **D = 0.5 GiB/s**; stop condition: remaining dirty data < **0.1 GiB**.

**(a) Compute the pre-copy rounds. [3]**
- Round 0: copy 16 GiB in 16/2 = **8 s**; dirtied meanwhile = 0.5 × 8 = **4 GiB** (> 0.1 → continue).
- Round 1: copy 4 GiB in **2 s**; dirtied = 0.5 × 2 = **1 GiB** (> 0.1 → continue).
- Round 2: copy 1 GiB in **0.5 s**; dirtied = 0.25 GiB (> 0.1 → continue).
- Round 3: copy 0.25 GiB in **0.125 s**; dirtied = 0.0625 GiB (< 0.1 → **stop**).

**(b) Downtime. [2]**
- Stop-and-copy transfers the remaining 0.0625 GiB with the VM paused: 0.0625 ÷ 2 = **31.25 ms downtime**.

**(c) Total migration time. [2]**
- ≈ 8 + 2 + 0.5 + 0.125 + 0.03125 ≈ **10.66 s** (downtime is only the last term).

**(d) Convergence condition. [2]**
- Each round shrinks remaining data by factor **D/B = 0.25**. Pre-copy converges **only if D < B**; if dirty rate ≥ bandwidth, remaining data never decreases and migration never converges (must throttle the VM or use post-copy).

**(e) Why is downtime so much smaller than total time? [1]**
- Most copying happens **while the VM keeps running** (push phase); only the final tiny inconsistent remainder is copied with the VM paused.

## Q48 (new). Provisioning cost/time comparison [10]

**Scenario:** Manual OS installation per server takes 6 hours of engineer time ($60/h). The team must deploy **20 servers** monthly for a test cycle, then destroy them. A golden AMI plus cloud-init boots a server in **8 minutes** with 10 minutes of oversight.

**(a) Cost of manual provisioning per cycle. [2]**
- 20 × 6 h × $60 = **$7,200** labour per monthly cycle (plus error/rework risk).

**(b) Cost with image-based provisioning. [2]**
- 20 × (8 min boot + 10 min oversight ≈ 18 min) = 6 h total → 6 × $60 = **$360** per cycle (**95% reduction**).

**(c) Two qualitative benefits beyond cost. [2]**
- **Consistency** (identical servers — no per-server drift); **elasticity** (autoscaling reuses the same AMI for peaks); auditability via versioned images; instant teardown.

**(d) What must the AMI + user-data split respect? [2]**
- AMI = static software state (OS, agents, patches — **disk only**, no runtime memory); **user-data/cloud-init** = per-instance boot customisation (hostname, keys, app config) and runs **once at first boot** only.

**(e) When does manual provisioning still make sense? [2]**
- One-off bespoke experiments; learning/debugging a novel build; initial construction of the *first* golden image itself.

---

# Module 6 — Capacity Management and Scheduling

## Q15. Autoscaling and capacity planning [10]

**Scenario:** An e-commerce application sees a sudden traffic increase during a major sale.

**(a) How capacity management maintains performance. [2]**
- Keeps resources **sufficient at the right time/cost to meet demand and SLAs — no more than needed**: measure (p95) → characterise → forecast (sale events!) → add headroom → provision → monitor and adjust continuously. Under-capacity → SLA breach; over-capacity → wasted spend.

**(b) Horizontal vs vertical scaling with examples. [3]**
- **Horizontal (scale out):** add replicas behind a load balancer — e.g., web servers 5→20 for the sale; needs stateless design/session store; no hard ceiling; tolerates instance failure.
- **Vertical (scale up):** make one instance bigger — e.g., DB instance m5.xlarge→m5.4xlarge; simple, no app change, quick relief; but ceiling (largest type), restart often required, still a single point of failure.
- Rule: **scale up for quick relief; design to scale out.**

**(c) How autoscaling responds to changing workloads. [3]**
- Monitors metrics (CPU, request rate, queue depth) against targets; adds/removes instances automatically within **min/max** bounds; **cooldown** prevents flapping; supports **scheduled scaling** for known sale windows plus **metric-based** reactive scaling; combined with load balancer health checks and multi-AZ spread.

**(d) Can autoscaling eliminate capacity planning? [2]**
- **No.** Autoscaling *executes* a plan; someone still must: choose instance types, set min/max and thresholds, provide **lead-time headroom** (instances take minutes to boot — a spike can breach SLA before scale-out completes), plan the steady baseline purchase model, and budget. Bad autoscaling configs cause cost blowouts or breaches; it complements, not replaces, planning.

## Q16. Kubernetes resource scheduling [10]

**Scenario:** Node A: 8 vCPU, 16 GB RAM. Node B: 4 vCPU, 8 GB RAM. Each pod requests 1 vCPU and 2 GB RAM.

**(a) Maximum pods per node (requests-based). [3]**
- Node A: CPU 8÷1 = 8; memory 16÷2 = 8 → **8 pods**.
- Node B: CPU 4÷1 = 4; memory 8÷2 = 4 → **4 pods**.
- (Both constraints bind equally here; with allocatable < capacity, substitute allocatable first.)

**(b) Why can't the scheduler place pods once requests exceed allocatable resources? [2]**
- The **filter stage** eliminates any node whose **allocatable** CPU/memory minus existing **requests** cannot fit the new pod's requests. Requests are *reservations*, not optional hints — admitting beyond them would break the guarantee and invite overcommit; unplaced pods stay **Pending** even if the node looks idle.

**(c) Requests vs limits. [3]**
- **Request** = guaranteed share **reserved at scheduling**; drives placement and cgroup scheduling weight.
- **Limit** = **hard runtime cap**: CPU over limit → **throttling**; memory over limit → **OOM-kill**.
- Between request and limit a pod may burst; neighbours are protected by limits; capacity math uses requests only.

**(d) How poor allocation leads to contention. [2]**
- Requests **too low** → node overpacked; real usage exceeds capacity at peak → CPU throttling, memory pressure/OOM kills, noisy neighbours, latency/SLA breaches.
- Requests **too high** → stranded reservations, Pending pods, wasted money. Fix: right-size to p95 + 20–30% headroom and monitor.

## Q17. SLA-aware and cost-aware scheduling [10]

**Scenario:** A cloud application must maintain 99.9% availability while minimising operating cost.

**(a) Relationship between capacity provisioning and SLA commitments. [3]**
- The SLA (99.9%) defines the **failure budget**: 43.2 min/month. Capacity must be planned so measured demand (p95/p99) + headroom + **N+1 host/zone redundancy** fit inside that budget; infrastructure SLAs with the provider must themselves guarantee the underlying availability the app SLA assumes. Shortage → SLA breach penalties/credits; surplus → cost. Provisioning = the mechanism that converts the SLA into resource decisions.

**(b) Reservation-based provisioning and its advantages. [2]**
- Capacity **pre-reserved and guaranteed** for a time window (e.g., On-Demand Capacity Reservation; Capacity Blocks for ML; advance reservations in private clouds via lease schedulers like Haizea).
- Advantages: guarantees availability at a **specific time** (cannot-miss launches, planned experiments) inside finite capacity; removes admission uncertainty. Cost: billed **whether used or not**; reduces pool flexibility (vacation of resources hurts utilisation).

**(c) How rightsizing and cost-aware scheduling reduce waste. [3]**
- **Rightsizing:** measure p95 CPU/memory over ≥2 weeks → new request ≈ p95 + 20–30% headroom (conservative for memory/OOM) → roll out gradually → repeat; tools: CloudWatch, kubectl top, VPA, Compute Optimizer.
- **Cost-aware scheduling:** scale in on demand drop; **bin-pack/consolidate** (fewer, fuller nodes); schedule dev/test off-hours; Arm/Graviton where supported; correct purchase-model mix (on-demand unknown, spot interruptible, committed steady base, reservation only when a guarantee is needed).

**(d) What if cost is minimised without considering SLA? [2]**
- No headroom → spikes breach latency/availability SLOs; consolidation amplifies single-host failure; preempting/under-provisioning breaks guarantees → **SLA credits/penalties, churn, and incident cost usually exceed the savings**. The triad must balance: cost ↔ performance/headroom ↔ reliability.

## Q49 (new). Autoscaling instance count numerical [10]

**Scenario:** Peak demand **10,000 requests/sec**; each instance serves **500 rps** at 100%; the SLA target keeps utilisation ≤ **70%**; the fleet spans **2 AZs** and must survive one AZ failure with no performance loss.

**(a) Instances needed at the 70% target. [2]**
- Effective per instance = 500 × 0.70 = **350 rps**.
- 10,000 ÷ 350 = 28.57 → **29 instances** (round up).

**(b) AZ-failure requirement. [3]**
- Spread 29 → **30 instances = 15 per AZ**. If one AZ fails, the surviving 15 serve 10,000 rps → each at 10,000/15 ≈ 667 rps = **133% of capacity → SLA breach**.
- Therefore size for **N+1-zone**: total capacity needed on **one** AZ = 10,000/350 = 28.57 → 29 in *each* AZ → **58 total (29 + 29)**.

**(c) Scaling policy design. [2]**
- Min = baseline (e.g., steady-day p95), Max = 58 (or budgeted), target metric = request rate per instance or CPU ≤ 70%; scheduled pre-scale before known sales; cooldown to avoid flapping.

**(d) Cost caveat. [3]**
- 58 steady on-demand instances is expensive: commit the steady base via Reserved/Savings Plan (≈30–40% off), keep the peak delta on-demand, and let **min** be the normal-day size — resilience sizing and purchase-model optimisation are separate decisions.

## Q50 (new). Purchase-model mix cost numerical [10]

**Scenario:** Steady baseline **20 instances** 24×7; additional spiky load ≈ 10 instances average during 12 business hours/day. Rates (assumed): on-demand $0.12/h; 1-yr reserved effective $0.078/h; spot ≈ 40% of on-demand for interruptible batch.

**(a) Cost of running everything on-demand for a year. [2]**
- Steady: 20 × 8,760 × 0.12 = **$21,024**. Spiky: 10 × (12 × 365) × 0.12 = 10 × 4,380 × 0.12 = **$5,256**. Total ≈ **$26,280**.

**(b) Optimised mix. [3]**
- Steady base → Reserved: 20 × 8,760 × 0.078 = **$13,665.60**.
- Spiky web load → On-demand (must stay reliable): **$5,256**.
- If part of the spiky load is *interruptible batch* (say 5 instances-worth) → Spot: 5 × 4,380 × 0.048 = **$1,051.20** instead of $2,628 → spiky total ≈ 2,628 + 1,051.20 = **$3,679.20**.
- Optimised total ≈ 13,665.60 + 5,256 + 1,051.20 ≈ **$19,972.80** ≈ **24% saving** vs all-on-demand.

**(c) Which commitment does NOT reserve capacity? [2]**
- **Savings Plans / Reserved Instances = price commitments only.** Guaranteed *capacity* needs an **On-Demand Capacity Reservation** (billed whether used or not) — the classic confusion.

**(d) Risk line. [3]**
- Commitments are 1–3 yr locks: commit only the **stable floor** (measured over months); spot can be interrupted in 2 min → checkpointing/queues required; keep a small on-demand buffer for failover.

## Q51 (new). Rightsizing numerical [10]

**Scenario:** 50 pods each **request 2000m CPU**. Two weeks of p95 data: actual p95 usage **900m**. Nodes hold 4 vCPU of allocatable each.

**(a) New request. [2]**
- 900 × 1.25 (25% headroom) = **1125m** → round to **1200m** (1.2 vCPU).

**(b) Freed reservation. [3]**
- Per pod: 2000m − 1200m = **800m freed**; across 50 pods = **40,000m = 40 vCPU** ≈ **10 nodes** of 4 vCPU (up to 10 hosts' worth of capacity released — cost/energy saving).

**(c) Why not set the request to p95 exactly? [2]**
- Zero headroom → any burst above p95 (p99 spikes, new release regression) throttles/OOMs and breaches SLOs; headroom is the price of the SLA. Memory gets *more* conservative headroom due to OOM-kill severity.

**(d) Rollout discipline. [2]**
- Gradual canary rollout watching SLOs/throttling metrics; repeat periodically (workloads drift); use VPA recommendations/Compute Optimizer as input, measurement as truth.

**(e) What if p95 was 1900m instead? [1–2]**
- New request = 2375m → keep ≈ 2000m–2400m; near-zero savings — rightsizing only pays when measurement shows real gap.

## Q52 (new). Lease scheduling with advance reservation [10]

**Scenario:** A 4-node private cluster holds leases (start, duration, preemptable): L1 node1 (1 pm, 1 h, yes); L2 node2 (1:30 pm, 1 h, no); L3 node3 (1 pm, 3 h, no); L4 node4 (now, 2 h, no). New arrivals: **L5 (now, 1 h, not preemptable)** and **L6 (1:30 pm, 2 h, preemptable)**.

**(a) Place L5. [3]**
- Non-preemptable leases occupy nodes exclusively. Now: node1 free until 1 pm (L1 is preemptable but not yet running — a *pending* preemptable lease yields to firm requests); node2 free until 1:30; node3 busy till 4; node4 busy till ~now+2.
- **L5 (now, 1 h, firm)** → schedule on **node1** (or node2), completing before its next firm lease (1:30). If L1 must start at 1 pm, L5 (now–1 pm, 1 h) ends exactly then — fits.

**(b) Place L6. [3]**
- **L6 (1:30 pm, 2 h, preemptable)**: at 1:30 node2 becomes free (L2 done) → run L6 on **node2** till 3:30. Alternatively at 1 pm place on node1 after L5/L1 logic — but firm L2 owns node2 1:30–2:30; using node2 at 1:30 works only if L2 finished. State assumption explicitly — scheduling answers must declare their assumptions.

**(c) Why can advance reservations hurt utilisation? [2]**
- Reserved capacity **idles** waiting for its window (billed whether used); other work must be vacated/kept off it; schedulers mitigate via **preemption + checkpointing** (suspend lower-priority work, resume later) and backfilling best-effort leases into gaps.

**(d) Difference: best-effort vs advance lease. [2]**
- **Best-effort (BEL):** queued, starts when resources free (spot-like; preemptable). **Advance reservation (AR):** guaranteed start/duration (capacity reservation-like; non-preemptable usually).

---
# Module 7 — Availability, Multi-Tenancy, Security, SLA and Compliance

## Q18. Availability and fault tolerance [10]

**Scenario:** A cloud-hosted online banking application must remain operational even if an availability zone fails.

**(a) Availability vs fault tolerance. [2]**
- **Availability:** the fraction of time the service is usable (e.g., 99.95% — measured over a period).
- **Fault tolerance:** the ability to **continue correct operation despite component failure** — achieved by redundancy + automatic failover, not by wishing.

**(b) Multi-AZ architecture to improve availability. [3]**
- Deploy the full stack **in ≥2 AZs**: ALB with targets in each AZ; auto-scaling web/app tiers across AZs; **RDS Multi-AZ** (synchronous standby, automatic failover); session/state in a shared store (ElastiCache/DB), not instance memory; health checks remove failed-zone targets; static content via CDN.
- Justification: AZs are **independent failure domains** (separate power grids/flood plains, low-latency linked) — one AZ event leaves the other serving.

**(c) How redundancy, failover and load balancing contribute. [3]**
- **Redundancy** eliminates single points of failure (extra servers, standby DB, replicated data).
- **Failover** switches traffic to the healthy/standby component automatically (health-check driven; DB promotes standby) — converts failure from outage to brief degradation.
- **Load balancing** distributes load and continuously **removes unhealthy targets**; spreads replicas so no single server's failure is user-visible.

**(d) Why multiple replicas within ONE AZ are not enough. [2]**
- An AZ is a shared failure domain — its data centres share regional power/network/disaster risk; a zone event (grid failure, flood, network corruption) hits **all replicas simultaneously**. Zone-level spread is the minimum; full DR needs a **second region** for cloud-level failures.


## Q19. Multi-tenancy [10]

**Scenario:** A cloud provider hosts applications of several unrelated enterprises on shared physical infrastructure.

**(a) Explain multi-tenancy. [2]**
- One **instance of software serves multiple customers (tenants)** on shared infrastructure; tenants may customise configuration/UI/business rules but **not the code**; the app is *designed* to share resources safely. Contrast multi-instance (separate copies per tenant).

**(b) How virtualization and containerization enable multi-tenancy. [3]**
- **Virtualization:** hypervisor partitions physical hosts into **isolated VMs** per tenant — separate kernels, hard isolation, the illusion of owning full infrastructure; enables IaaS tenancy.
- **Containerization:** namespaces (view isolation) + cgroups (resource caps) let **denser** per-tenant/app isolation on shared kernels — enables PaaS/SaaS tenancy levels with less overhead.
- Both convert "shared physical machine" into "logically isolated tenants" — the prerequisite for pooling with isolation.

**(c) Security and performance isolation challenges. [3]**
- **Performance:** noisy neighbours — one tenant's peak steals CPU/memory/I-O (hence conserving vs non-conserving allocation, quotas, shares).
- **Security:** side-channel/data-leakage risk on shared kernels/hypervisors; one tenant must never read another's data; patching one shared instance risks all tenants (update windows).
- Multi-tenant *data* layer: shared tables need tenant-ID row isolation; misconfiguration leaks rows across tenants.

**(d) Preventing resource abuse and cross-tenant access. [2]**
- **Resource:** per-tenant quotas/limits (cgroups, rate limits, conserving allocation), autoscaling headroom, monitoring per tenant.
- **Access:** per-tenant authentication/authorization (centralised IdP with per-tenant admin, or federated SSO), encryption per tenant, row-level security by tenant ID, audit logging; hard isolation (VM) for tenants that require it.


## Q20. Cloud security and shared responsibility [10]

**Scenario:** A company stores confidential employee records on a public cloud and assumes the provider is responsible for ALL security.

**(a) Shared responsibility model. [3]**
- Provider secures **the cloud itself**: physical data centres, network, hardware, virtualization layer, managed-service internals (patching of the service).
- Customer secures **what they put in it**: IAM/users/MFA, application configuration, data classification, **encryption and keys**, guest patching (IaaS), access policy.
- The boundary **moves with the service model** (IaaS: customer owns OS-up; SaaS: customer owns data/access only) — but accountability for data/identity never transfers. Assuming the provider covers everything is precisely the misconfiguration threat.

**(b) Identify the threats in this scenario. [3]**
- **Compromised credentials** — no MFA/long-lived keys; an admin phished = full data access.
- **Misconfigured storage** — bucket/database left public; permissive security groups; encryption disabled.
- **Unauthorized access** — over-privileged IAM (admin-for-all), missing network segmentation, unpatched guest OS.

**(c) Mitigations. [2]**
- MFA + roles (no static keys) + least-privilege policies with periodic review; encryption at rest/in transit with customer-managed keys; private subnets + SG/NACL; automated config checks (IaC policy, SCP guardrails), audit logging + alerts; tested backups.

**(d) Why public cloud is not automatically secure or insecure. [2]**
- Security = **joint** implementation: provider-hardened infrastructure + a negligent tenant (public buckets, admin-everyone, no MFA) is insecure; a disciplined tenant can be *more* secure than most on-prem. The model changes **who** does each control, not **whether** the work exists.


## Q21. SLA and SLO management [10]

**Scenario:** A cloud provider commits to 99.9% monthly availability.

**(a) SLA vs SLO. [2]**
- **SLA:** the **contract** — service to be provided, support, times/locations, costs, responsibilities, and **consequences** (credits/penalties) for missing targets.
- **SLO:** the **objective** — specific measurable targets inside the SLA (availability, throughput, response time); measured by SLIs. Chain: SLI (measure) → SLO (target) → SLA (contract).

**(b) Maximum downtime in a 30-day month at 99.9%. [2]**
- Downtime = (1 − 0.999) × 43,200 min = **43.2 minutes/month** (≈ 0.1% of the period).

**(c) SLA life-cycle stages. [3]**
1. **Contract definition** — provider builds a catalogue from standard templates.
2. **Publication & discovery** — offerings advertised; customers search/shortlist.
3. **Negotiation** — terms agreed and signed (automated for packaged apps; manual with performance analysis for customised).
4. **Operationalization** — **monitoring** (measure, compute metrics, detect/notify deviations), **accounting** (archive adherence, report actual vs guaranteed), **enforcement** (credits/corrective action).
5. **De-commissioning** — terminate activities when the hosting relationship ends.

**(d) How automated policy-based management maintains SLAs. [3]**
- Policies map measured SLIs to **automatic actions**: latency SLO at risk → scale out / migrate VMs; overload → **admission control** (session-based rejection of new sessions only); deviation → notify + open enforcement workflow; accounting archives evidence for credits.
- Benefits: continuous conformance without human latency, consistent enforcement, auditable reports, and it is how a provider can credibly *promise* elasticity-backed SLAs (application-level metrics drive app-specific scaling).

## Q22. Sovereign cloud and compliance [10]

**Scenario:** A government organization wants to migrate citizen records to cloud infrastructure but requires control over data location, access and legal jurisdiction.

**(a) Explain data residency, regulatory compliance and sovereign cloud. [3]**
**Model answer:**
- **Data residency:** the legal/geographical requirement that data be **stored (and often processed) within a specific jurisdiction** (e.g., citizen records stay in-country).
- **Regulatory compliance:** adherence to the laws/standards governing that data — data-protection acts (e.g., India's DPDP Act 2023, Rules phased to ~2027), sector regimes, audit and retention duties.
- **Sovereign cloud:** a cloud operated so that **data location, access control and legal jurisdiction remain under the nation/customer's control** — in-country operations and legal entity, staff/access controls, protection from foreign legal access; often delivered with/by a national operator.

**(b) Why may a conventional public cloud deployment not satisfy all sovereign requirements? [3]**
**Model answer:**
- Data, replicas or **backups may silently leave the country** (cross-region replication, CDN, DR).
- **Foreign jurisdiction exposure:** the provider is subject to foreign legal-access laws; the government cannot fully control whose law reaches the data.
- **Operational access:** provider staff anywhere may hold privileged access; shared multi-tenant operations complicate **audit proof**; the SLA remedy (service credits) is not a compliance guarantee, and some services may not exist in-region.

**(c) Recommend suitable architectural and governance measures. [4]**
**Model answer:**
1. **In-country Region/Local Zone or Outposts** (data can stay on government premises) — residency by architecture; disable/limit cross-border replication and backups.
2. **Strict IAM:** in-country admin roles, MFA, least privilege, approval gates on any foreign support access; **customer-managed encryption keys (BYOK) held in-country**.
3. **Governance:** data classification, audit logging with in-country retention, verified certifications (SOC 2/ISO 27001), data-processing agreement + sub-processor list, exit plan with tested export.
4. **Guardrails:** SCPs/policy-as-code refusing resource creation outside approved regions; tag-based cost/ownership tracking; periodic compliance audits and DR drills.


## Q53 (new). Availability arithmetic: serial vs redundant [10]

**Scenario:** A banking portal chain: load balancer **99.95%** → web tier (**two identical app servers, each 99.9%**, behind the LB) → database **99.9%** → external payment gateway **99.9%**. Target: 99.9% overall.

**(a) Availability of the redundant web pair. [2]**
**Model answer:**
- Pair = 1 − (1 − 0.999)² = 1 − 0.000001 = **99.9999%** — a redundant pair beats every single component's availability.

**(b) Overall composite availability. [3]**
**Model answer:**
- A serial chain multiplies: 0.9995 (LB) × 0.999999 (web pair) × 0.999 (DB) × 0.999 (gateway) ≈ 0.997501 → **≈ 99.75%**.
- Lesson: one weak serial link (the 99.9% DB or gateway) drags the whole chain far below its best component.

**(c) Downtime in a 30-day month and verdict against the 99.9% target. [2]**
**Model answer:**
- Downtime = (1 − 0.997501) × 43,200 ≈ **108 minutes** vs the allowed 43.2 minutes → **SLA breached**.

**(d) Cheapest changes that restore the target. [3]**
**Model answer:**
- Make the **database a redundant pair** (99.9% → 99.9999%): composite ≈ 99.85% → ≈ 64.9 min — **still a breach**.
- Additionally make the **payment gateway redundant** (two independent gateways/providers): composite ≈ 99.9497% → ≈ **21.7 min ≤ 43.2 min → target met**.
- Insight: fix the **weakest serial links first**; converting a weak single component into a redundant pair multiplies availability far more than polishing an already-strong one.



## Q54 (new). Multi-tenancy levels and isolation [10]

**Scenario:** A SaaS vendor serves 3 enterprise customers: one regulated bank demanding dedicated infrastructure, two standard customers sharing everything.

**(a) Which MT levels/models fit? [3]**
- Bank → **silo** (dedicated stack) or IaaS-style **VM isolation** — strongest isolation/compliance, highest cost.
- Standard customers → **pool** (shared stack, tenant-ID separation) — lowest cost, one upgrade path; or **bridge**: shared app + dedicated DB for the bank.

**(b) What must the shared application do on every request? [2]**
- **Resolve the tenant** (from token/subdomain/header), load tenant metadata, and **enforce row-level filtering** (every query scoped by tenant ID) — isolation by code, not by infrastructure.

**(c) Name two data-layer designs for the pooled customers and a trade-off. [2]**
- **Shared table + tenant-ID rows:** space-efficient, simple ops; isolation depends on correct code (a missing WHERE leaks rows).
- **Schema (or DB) per tenant:** stronger isolation, easy per-tenant backup/restore; more objects to manage/migrate.

**(d) Release-management risk unique to multi-tenancy. [3]**
- One codebase/instance → **one update affects all tenants simultaneously**: a bad release or its downtime hits everyone (even tenants who didn't request it); update windows constrained by tenants' schedules; mitigations: staged rollouts (canary per tenant cohort), feature flags, backward-compatible migrations, tested rollback.


## Q55 (new). SLA availability numerical [10]

**Scenario:** Contract: 99.5% availability per 30-day month, service credits if breached. Actual outages this month: 45 min planned maintenance + 60 min unplanned.

**(a) Allowed downtime. [2]**
- (1 − 0.995) × 43,200 = **216 minutes** (3.6 h) per month.

**(b) Achieved availability and verdict. [3]**
- Total downtime = 45 + 60 = 105 min.
- Availability = 1 − 105/43,200 = 0.99757 → **99.757% ≥ 99.5% → SLA met** (no credits). Note: whether *planned* maintenance counts depends on the SLA's exclusion clauses — state the assumption; many SLAs exclude scheduled maintenance windows, which would make the month even stronger (99.896%).

**(c) Same outages under a 99.9% target? [2]**
- Allowed = 43.2 min < 105 min → **breach** → service credits per contract; two consecutive breaches typically trigger contract review.

**(d) Availability arithmetic to reduce breaches. [3]**
- Serial components multiply: 99.9% LB × 99.9% DB = 99.80% composite. Redundant parallel pair: 1 − (0.001)² = 99.9999%. So: remove serial single points (Multi-AZ standby), convert serial links into redundant pairs, and cut **mean time to repair** (automation, runbooks) — downtime budget is spent by frequency × duration.


## Q56 (new). SaaS due diligence and shared responsibility [10]

**Scenario:** HR data (personal data of 40,000 employees) will move to a SaaS HR platform headquartered abroad.

**(a) Which responsibilities stay with the customer? [3]**
- **Users, access and data policy**: who is provisioned, SSO/MFA enforcement, data classification, backup/export verification, compliance obligations (DPDP/GDPR) as data fiduciary; the vendor runs the app — but *accountability for data and access never transfers*.

**(b) Five questions to ask the vendor with evidence. [3]**
- Security/tenant isolation → SOC 2 / ISO 27001 reports, SSO+MFA support; Residency → region list, DPA, sub-processors; Availability → SLA text + credits + status history; Exit → export formats, deletion terms; Integration/vendor risk → API docs/rate limits, SCIM/SAML, ownership & financial health.

**(c) What does the SLA remedy usually cover — and not cover? [2]**
- Remedy = **service credits** for missed uptime; does **not** compensate business/regulatory loss; availability is "borrowed" — a vendor outage is your outage operationally even when the SLA pays you pennies.

**(d) Exit plan elements. [2]**
- Tested periodic export (full data dump format), retention-after-termination terms, migration effort estimate, escrow/portability clauses, and an alternative-vendor gap analysis — written **before** signing.

---

# Module 8 — Cloud Application Development and Deployment

## Q23. CI/CD and cloud deployment [10]

**Scenario:** A team deploys releases manually to production: inconsistent environments, frequent deployment failures.

**(a) CI and CD explained. [3]**
- **Continuous Integration:** merge **small changes often**; every merge is **built and tested automatically** — integration problems surface early.
- **Continuous Delivery:** every change that passes is **kept release-ready**; a human decides when to release.
- **Continuous Deployment:** every passing change **goes live automatically** — no manual step. (CD = delivery or deployment; say which.)

**(b) Stages of a CI/CD pipeline. [3]**
- **Commit** → **Build** (compile/package/container image) → **Test** (unit → integration → end-to-end, fast tests first) → **Release** (store the artefact, version it) → **Deploy** (go live) — plus observe (metrics/alerts) and rollback.
- Containerized: the image built in CI is the same artefact promoted through environments. Serverless: functions deployed per event-trigger config.

**(c) How containerized and serverless models support delivery. [2]**
- **Containers:** immutable, versioned images ⇒ identical dev/test/prod; orchestrator rolling updates/canary; rollback = redeploy previous tag.
- **Serverless:** deploy function code + triggers; platform scales per event (incl. to zero); per-version traffic shifting (canary) supported by managed platforms.

**(d) Two risks of deploying without automated testing/rollback. [2]**
- Bad releases hit **all users at once** with no blast-radius control; **no fast revert** → prolonged outages (high time-to-restore); plus inconsistent environments and human error in manual steps.


## Q24. APIs and observability [10]

**Scenario:** A microservices application runs across multiple cloud instances; users report intermittent slowness.

**(a) Role of APIs and API gateways in service integration. [3]**
- **APIs:** the agreed contract for program-to-program requests (request → validate → work → response) — services integrate without knowing each other's internals (S3 example).
- **API gateway:** the **single front door** — one address for clients; performs **authentication** (only allowed callers), **throttling/rate limits** (protects backends), **routing** to the right service; lets teams change backends without touching clients (Zuul/API Gateway).
- Sync (wait for answer) vs async (queues absorb spikes) integration choices.

**(b) Logs vs metrics vs distributed traces. [3]**
- **Logs:** discrete event records ("payment failed for order 42") — *what happened*.
- **Metrics:** numeric time series (rps, p95 latency, memory) — *how much*, cheap to alert on.
- **Traces:** one request's journey across services with timings per hop — *where* it slowed/failed; stitched by a shared request/trace ID.
- Plus **alerts** = notifications when a metric crosses a threshold.

**(c) How alerts and observability find bottlenecks/failures. [2]**
- Golden signals (**latency, traffic, errors, saturation**) alert on threshold breach → the trace of a slow request identifies the *exact service/hop* → its logs reveal the cause (query, deadlock, dependency) → targeted fix (scale that service, add cache, fix query).

**(d) Why host-level monitoring is insufficient. [2]**
- One user request spans many ephemeral services/hosts; host CPU charts cannot attribute delay to a *service*; instances scale in/out so hosts vanish; correlation needs request IDs and service-level aggregation — that is distributed tracing + service metrics, not machine monitoring.


## Q25. Hybrid and multi-cloud operations [10]

**Scenario:** A business uses AWS for compute, Azure for selected managed services, and an on-premises database.

**(a) Hybrid vs multi-cloud. [3]**
- **Hybrid:** *your own* infrastructure (private/on-prem) + a public cloud working as one (cloudbursting, keep sensitive data local; needs reliable links + consistent tooling).
- **Multi-cloud:** *two or more public providers* (best-of-breed per provider, vendor independence; needs skills/tools per provider).
- The design here is **both** (on-prem + two publics) — the categories can combine.

**(b) Operational and networking challenges. [3]**
- **Networking:** cross-cloud links (VPN/interconnect) latency + **egress fees**; consistent addressing/security across environments; data gravity (DB on-prem = chatty apps suffer).
- **Operations:** three consoles/APIs/IAM models; duplicated monitoring without a common view; skill/tooling sprawl; inconsistent security policy; change management across independent provider release cycles.

**(c) How portability, interoperability and common monitoring help. [4]**
- **Portability:** container packaging (K8s runtime everywhere) + **multi-cloud IaC** (Terraform) + avoid proprietary dependencies where exit matters → workloads move, environments are reproducible.
- **Interoperability:** federated identity/SSO across providers; standard APIs; abstraction layers for secrets/config; documented data flows with queues (async decoupling).
- **Common monitoring:** one observability stack (OpenTelemetry collection → unified dashboards/alerts) across AWS/Azure/on-prem → single operational picture, comparable SLOs, faster incident triage.
- Governance: tag/cost reporting consolidated; DR runbooks that assume provider-specific failure.

---


## Q57 (new). Pipeline and safe release design [10]

**Scenario:** A payment API: 30-minute manual deploy, weekly releases, rollbacks by re-deploying an older build from a laptop.

**(a) Design the pipeline. [4]**
- Commit → CI: build image + run unit/integration tests → publish image to registry (tag = git SHA) → deploy to **staging** automatically → end-to-end tests + load test → **canary 5% of production traffic** for 30 min watching error rate/latency → progressive 25/50/100% rollout → if golden-signal alarms trip: **automatic rollback** to previous tag (orchestrator or gateway traffic switch). Infrastructure changes ride the same pipeline via IaC plan/apply.

**(b) Which release strategy fits payments, and why? [2]**
- **Canary** (progressive exposure with automatic metric gates) — limits blast radius of a money-path regression; **blue-green** also viable for instant full-switch rollback, at double-environment cost.

**(c) How do DORA measures track improvement? [2]**
- Deployment frequency ↑, **lead time for changes** ↓ (days→hours), **change failure rate** ↓ (each release smaller/tested), **time to restore** ↓ (automated rollback in minutes) — the four metrics of delivery performance.

**(d) One trade-off of continuous deployment for this team. [2]**
- Fully automatic go-live on a payment API needs **very strong automated tests + feature flags** (no human gate); many teams choose **continuous delivery** (one-click human release) as the pragmatic compromise.


## Q58 (new). Observability troubleshooting scenario [10]

**Scenario:** Checkout latency p95 rose from 200 ms to 2 s after Monday's release. Host dashboards are green. Errors: 0.1% → 2%.

**(a) Which artefact finds the culprit first? [2]**
- **Distributed traces** — compare slow vs fast request timelines; the extra latency concentrates in one hop (e.g., the pricing service call changed from 5 ms to 1.8 s).

**(b) Which metrics confirm, and which logs explain? [3]**
- Metrics of the offending service: latency (golden signal #1) and **saturation** (thread pool/DB connections pegged) + traffic correlation with the release; error rate by service. Logs of that service show repeated slow queries / lock waits introduced Monday.

**(c) Why were host dashboards green? [2]**
- Hosts were healthy — the regression is **software** (a bad query) not resource exhaustion; host-level monitoring measures machines, not request journeys.

**(d) Alerting improvements. [3]**
- Alert on **SLO burn rate** (fast/slow burn) instead of host thresholds; per-service golden-signal dashboards; canary analysis in the pipeline would have caught it at 5% traffic; add trace-based anomaly detection on new releases.

# Module 9 — Future Cloud and Compute Paradigms

## Q26. Edge computing [10]

**Scenario:** An autonomous vehicle fleet generates massive sensor data and needs near-real-time decisions.

**(a) Edge computing and the cloud-edge continuum. [3]**
- **Edge computing:** process data **near where it is created** — faster response, less data transmitted, keeps working when links drop.
- **Continuum layers:** **device** (sensor/camera — tiny decisions) → **local edge** (on-site gateway, e.g., Greengrass — offline-capable local decisions) → **network edge** (telecom 5G, e.g., Wavelength — low mobile latency) → **cloud region** (storage, training, big analytics). Apps are **split** across layers — edge AND cloud, not edge OR cloud.

**(b) Why sending ALL sensor data to a central cloud fails. [3]**
- **Latency:** round-trip to a region (tens–hundreds of ms) cannot support split-second safety decisions.
- **Bandwidth/cost:** raw sensor/video streams are enormous — transmitting all of it is economically and technically infeasible.
- **Reliability:** connectivity drops; a safety function that dies with the link is unacceptable. Plus privacy/residency constraints on some data.

**(c) How edge and cloud cooperate in this architecture. [4]**
- **On the edge (device/local):** sensor fusion + obstacle detection run locally in real time; only **events/summaries** (near-misses, telemetry aggregates) are uploaded; decisions continue offline.
- **In the cloud:** fleet-scale **model training** on accumulated data; simulation; route/fleet planning; long-term analytics.
- **Feedback loop:** trained models are validated in the cloud and **pushed down** to edge devices (OTA model updates); policies/thresholds tuned centrally, executed locally. Network edge (5G/Wavelength) hosts latency-critical services that need more compute than the vehicle but less delay than a region.


## Q27. AI/ML and accelerator-based cloud computing [10]

**Scenario:** An organization trains and serves large AI models on cloud infrastructure.

**(a) Why GPU/accelerator instances suit AI/ML. [3]**
- Model training/inference is dominated by **matrix operations across huge tensors** — GPU's thousands of simple cores execute these **in parallel**, orders faster than a CPU's few general-purpose cores; TPUs/ASICs (Trainium/Inferentia) go further: purpose-built for ML maths with better perf/watt.

**(b) CPU vs GPU trade-offs. [3]**
- **CPU:** few powerful cores; best for serial, branch-heavy, general workloads (web servers, DBs); flexible, available everywhere.
- **GPU:** massive parallelism for regular dense maths; huge speedups for training/inference; but only helps **parallelisable** work, costs more, needs data-movement management (chip-to-chip links, batch sizing).
- ASIC/TPU: best efficiency for one job; least flexible. FPGA: reprogrammable middle ground.

**(c) Selecting compute for training vs inference. [2]**
- **Training:** occasional, heavy, batch — large/many accelerators, high memory, fast interconnect; can rent in blocks (Capacity Blocks) and release after.
- **Inference:** continuous, latency-sensitive, many small requests — smaller cost-optimised accelerators or CPUs (quantised models), autoscaling, possibly **edge deployment** for response time.

**(d) Resource optimisation for cost/energy. [2]**
- Rightsize instance types to measured utilisation; spot instances for pre-processing/checkpointed training; scheduled shutdown of idle training clusters; Arm/efficient chips for serving; scale-to-zero serverless inference for spiky traffic; measure energy (PUE per service) and batch requests.


## Q28. Emerging compute paradigms [10]

**Scenario:** A research organization evaluates quantum, neuromorphic and photonic computing for future workloads.

**(a) Motivation behind quantum computing. [2]**
- **Qubits hold a superposition** (mix of 0 and 1 until measured), enabling fundamentally new approaches to special problems — molecule simulation, certain optimisation/crypto research — that scale exponentially worse on classical bits.

**(b) Neuromorphic computing and AI relevance. [2]**
- Brain-inspired chips of many simple **artificial neurons communicating via spikes**; **event-driven** (parts active only on a spike → energy savings) with **memory next to compute** (state per neuron, less data movement). Relevance: ultra-low-power, always-on AI sensing/inference (contrast: CPU follows a clock; neuromorphic reacts to events).

**(c) Motivation behind photonic computing. [2]**
- Use **light instead of electrons**: photonic **interconnects** move data between chips/racks with lower energy and delay (already in fibre, moving onto the chip); photonic **compute** performs maths (matrix operations) optically — bandwidth and energy are the motivation.

**(d) Compare with conventional accelerator-based computing (potential + limitations). [4]**
- **Maturity:** GPU/TPU clouds are **production-ready** with mature software ecosystems; quantum/neuromorphic/photonic are **early-stage** — accessed as cloud *experiments* (IBM Quantum, Braket; Hala Point research system; optical interconnects are the most deployed, e.g., TPU v4 optical switches).
- **Potential applications:** quantum — chemistry/materials/optimisation research; neuromorphic — event-driven low-power AI; photonic — first interconnect energy savings, later optical matrix math.
- **Limitations:** quantum machines are small and error-prone, no proven real-world advantage yet; neuromorphic lacks programming models/ecosystem; photonic compute struggles with storing data in light.
- **Conclusion:** complements, not replaces — keep accelerators for production AI; treat the three as watch-list technologies with cloud-based access lowering the experiment cost.

## Q29. Green cloud computing [10]

**Scenario:** A cloud service provider wants to reduce its data-centre carbon footprint while continuing to support growing customer workloads.

**(a) Explain green cloud computing. [2]**
**Model answer:**
- Designing and operating cloud infrastructure to **minimise energy use and carbon emissions** while still meeting service levels. Context: data centres used ~1.5% of world electricity in 2024 (~415 TWh), projected to roughly double by 2030 (IEA). Measured via **PUE = total facility energy ÷ IT equipment energy** (1.0 is perfect; lower is better).

**(b) How virtualization, consolidation and workload scheduling reduce energy consumption. [3]**
**Model answer:**
- **Virtualization** decouples workloads from physical machines — the enabler for the other two levers.
- **Consolidation:** pack many VMs onto fewer, fuller hosts (bin-packing) and **power off / decommission idle servers** — energy follows host count, not VM count.
- **Workload scheduling:** place work to maximise utilisation; run batch/dev work off-peak or co-locate complementary patterns; scale-to-zero for spiky services; migrate VMs to consolidate and shut whole hosts down. Example: 100 servers at 30% utilisation → consolidate to ~43 at 70%.

**(c) Trade-off between performance, availability and energy efficiency. [3]**
**Model answer:**
- **Tight packing saves energy but raises blast radius** — one host failure now hits many tenants (spread vs pack).
- **Switching off idle capacity lengthens scale-out** (cold capacity) — sudden spikes may breach latency/availability SLAs while servers boot.
- **SLA headroom must stay powered** — the N+1 spare and peak headroom that protect the SLA are precisely the energy the provider wants to remove. Equilibrium: keep SLA-driven headroom only (measured p95 + N+1), consolidate the rest.

**(d) Recommend two practical measures. [2]**
**Model answer:**
- (1) **Rightsize + consolidate:** measure p95, size instances/requests to p95+20–30%, bin-pack and retire idle hosts (utilisation 30% → 70%).
- (2) **Efficiency upgrades:** improve cooling/power delivery (PUE 1.6 → 1.2 ≈ 25% facility-energy cut) and move steady services to efficient silicon (Arm/accelerators); complement with renewable-energy matching (noting it is annual accounting, not hourly clean power).



## Q59 (new). Split-brain: what runs where [10]

**Scenario:** A smart-factory video analytics system: 200 cameras, real-time defect detection, monthly model retraining, audit reports.

**(a) Assign each function to a continuum layer. [3]**
- Cameras/**device**: capture + motion pre-filtering (only frames with activity processed).
- **Local edge gateway:** defect-detection inference (real-time, offline-capable), local buffering.
- **Network edge (Wavelength-like):** aggregation of multiple lines if latency budget allows heavier models.
- **Cloud region:** retraining jobs, dataset storage, dashboards, audit reports.

**(b) Justify with the three constraints. [3]**
- Latency (line-speed decisions can't wait for a region round-trip), bandwidth (200 camera streams ≫ any uplink — filter at source), resilience (production continues when the WAN link drops; cloud syncs later).

**(c) The feedback loop. [2]**
- Edge sends labelled events/summaries → cloud retrains monthly → validated model deployed back to gateways (versioned rollout, rollback).

**(d) One failure mode and mitigation. [2]**
- Model drift/edge outage: mitigation — local fallback rules + alerting, staggered model rollout with canary per line, offline queueing of events.


## Q60 (new). Green cloud decision [10]

**Scenario:** A provider's DC: PUE 1.6, 30% average server utilisation, electricity from a mixed grid.

**(a) Compute and interpret PUE improvements. [2]**
- PUE = total facility energy ÷ IT energy. At PUE 1.6, every IT watt costs **0.6 W overhead**. Dropping PUE to 1.2 cuts overhead to 0.2 W → total energy falls from 1.6 to 1.2 units = **25% facility-energy reduction** before any workload change.

**(b) Three levers, with mechanisms. [3]**
- **Efficient data centres:** better cooling/power delivery (free/liquid cooling, hot-aisle containment) → lower PUE.
- **Efficient chips:** Graviton/Trainium-class silicon — more work per watt for the same SLA.
- **Smarter use:** raise 30% → 65–70% utilisation via consolidation/bin-packing, switch off idle hosts, scale-to-zero; cleaner electricity via renewable matching (with the caveat that annual matching ≠ hourly clean power).

**(c) Consolidation numerical. [2]**
- 100 servers at 30% = 30 units of work. At 70% target: 30/0.7 = 42.86 → **43 servers**; retire ~57 → large energy/cost cut; keep N+1 spare for SLA.

**(d) The trade-off. [3]**
- Tight packing saves energy but **raises blast radius** of one host failure (spread vs pack); switching off idle capacity lengthens scale-out (cold capacity vs SLA); over-provisioned "just in case" capacity is simultaneously a cost and energy problem. Balance: measure p95, keep only SLA-driven headroom, consolidate the rest, and report both PUE **and** its blind spots (server efficiency, water, energy source).

---

## Answer-writing checklist (before you move to the next question)

1. Did I **name the concept** in the first line?
2. Did I **define it** in one sentence (the rote part)?
3. Did I **use the scenario's facts/numbers** in the justification?
4. For numericals: formula → substitution → **units** → one caveat/assumption line?
5. Did I end comparisons with a **"choose X when..." rule** rather than a bare table?
6. 2-markers: 2–3 lines. 3-markers: 3–4 lines. 4-markers: 5–7 lines with a trade-off.

