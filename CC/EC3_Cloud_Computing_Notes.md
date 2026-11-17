# CSI / SE / SS ZG527 — Cloud Computing
# EC3 Final Exam Notes (Closed-Book Preparation Edition)

> Built from the 2026 BITS WILP slide deck (13 parts, 623 slides), the course formula glossary, and the EC2 pattern analysis. Since the exam will be attempted as **closed book**, every concept below is written to be self-contained: definition → mechanism → when to choose → trap. Nothing says "see textbook".

---

## Part 0 — How to Use These Notes (Read This First)

### 0.1 The closed-book reality

An open-book EC3 rewards lookup. A closed-book EC3 rewards **recall + application speed**. These notes are therefore organised so that each concept can be rebuilt from memory in three moves:

1. **What it is** — one crisp sentence (the 2-mark answer).
2. **How it works** — the mechanism in 3–5 bullet points (the 3–4 mark answer).
3. **When you would choose it over the alternative** — the decision rule (the justification marks).

Every big topic also carries a **comparison table** (comparison questions are the most repeated pattern) and a **trap note** (the mistakes that cost marks).

### 0.2 The question pattern you are preparing for

From the EC2 paper and historical EC3 papers, expect a 3-subpart structure per 10-mark question:

| Marks | What they ask | What scores |
|---|---|---|
| 3–4 | Identify / explain / recommend a concept in a scenario | Correct term + definition + one scenario-linked sentence |
| 3–4 | Apply it to an architecture or business problem | Named components + data flow + why this fits the constraint |
| 2–3 | Numerical, trade-off or justification | Formula → substitution → answer with units → one caveat sentence |

**Answer skeleton for every scenario question (IDJR):**
**I**dentify the concept → **D**efine it in one line → **J**ustify with scenario facts (quote the numbers/constraints from the question) → **R**isk/trade-off (one limitation or mitigating control).

**Mark-depth guide:**
- 2 marks = definition + example.
- 3 marks = definition + mechanism (how it works) + example.
- 4 marks = definition + mechanism + comparison/justification + limitation.
- Numerical = always show formula, substitution, result with units, and one assumption/caveat — even if the arithmetic is trivial. Method marks exist.

### 0.3 The three corrections from your EC2 feedback (fix these first)

| Concept | Wrong belief | Correct version |
|---|---|---|
| **Multi-tenancy** | "Using multiple cloud providers" | Multiple **customers (tenants)** sharing one provider's infrastructure with logical isolation. Using multiple providers is **multi-cloud**. |
| **AMI** | "Defines custom physical hardware specs" | An AMI is a **template of the initial software state** (OS, patches, pre-installed software) used to launch EC2 instances. The hardware spec is the **instance type**. |
| **Kubernetes components** | "etcd maintains replicas / provides networking" | **ReplicaSet/Deployment controllers** maintain desired replicas; **Services** provide stable networking via label selectors; **etcd** only stores cluster state (key-value). |

Two more distinctions to burn in:
- **CPU request** = what the *scheduler reserves* (guaranteed share, drives placement). **CPU limit** = the *runtime cap* (throttling). A request is not a cap.
- **Vertical scaling** = make one instance bigger (ceiling exists). **Horizontal scaling** = add more instances behind a load balancer (needs stateless design).

### 0.4 Formula sheet is at the end

All numerical areas seen in papers — SLA downtime, pod capacity, overcommit ratios, capacity sizing, bin-packing, live migration rounds, autoscaling instance counts, consolidation savings — are in the Cheat Sheet with one worked example each. Practise them by hand; closed-book numericals fail on forgotten formulas, not arithmetic.

### 0.5 Preparation priority (your analysis, confirmed against syllabus breadth)

- **Tier 1 (must be automatic):** Kubernetes objects + requests/limits; Docker + namespaces/cgroups; virtualization techniques (full/para/hw-assisted); AWS architecture (EC2, AMI, VPC, IAM, Region/AZ); block vs file vs object storage; VM migration + IaC; autoscaling + capacity planning; multi-tenancy + shared responsibility; SLA/SLO + availability calculations.
- **Tier 2 (strong):** IaaS/PaaS/SaaS/BaaS/FaaS selection; microservices & cloud-native; hybrid vs multi-cloud; CI/CD + API gateways; observability (logs/metrics/traces); sovereign cloud & compliance.
- **Tier 3 (conceptual):** cloud evolution & economics; edge computing; GPU/accelerators; quantum/neuromorphic/photonic; green cloud & PUE.

### 0.6 Module map (slide-deck parts → notes modules)

| Notes module | Slide deck parts |
|---|---|
| 1. Introduction to Cloud | Part 1 |
| 2. Virtualization & Containers | Parts 2, 5 |
| 3. Container Orchestration (Swarm, Kubernetes) | Part 6 |
| 4. IaaS & AWS | Parts 3, 4 |
| 5. PaaS, SaaS, BaaS, FaaS | Part 7 |
| 6. Provisioning, Migration, IaC & Governance | Part 8 |
| 7. Capacity Management, VIM & Scheduling | Parts 9, 11 (capacity half) |
| 8. Multi-Tenancy, Availability, Security & SLA | Parts 10, 11 (MT half) |
| 9. Application Development & Future Compute | Parts 12, 13 |

---

# Module 1 — Introduction to Cloud Computing

## 1.1 What cloud computing is (the definition to reproduce)

**Definition (course wording):** Cloud computing is applications and services that run on a **distributed network** using **virtualized resources**, accessed through **common Internet protocols** and networking standards. It converts technology, services and applications into a **self-service utility**.

**Two enabling pillars (exam favourite — "what makes cloud possible?"):**
- **Abstraction** — implementation details are hidden from users and developers; applications run on unspecified physical systems, with unknown data locations and outsourced administration.
- **Virtualization** — resources are pooled and shared among users, giving each the *illusion of sole ownership*; resources scale up/down in a very short time without human intervention, billed on a metered basis, with multi-tenancy support.

**Trap:** Cloud exists *because of* virtualization + abstraction — but virtualization alone is not cloud. A single VMware host is virtualized but not a cloud (no self-service, metering, elasticity at platform scale).

## 1.2 NIST 3-4-5 rule (rote — memorise exactly)

**3 service models:** IaaS, PaaS, SaaS.
**4 deployment models:** Public, Private, Hybrid, Community.
**5 essential characteristics:** On-demand self-service, Broad network access, Resource pooling, Rapid elasticity, Measured/metered service.

## 1.3 The five essential characteristics (each with its exam hook)

| Characteristic | Core idea | Signature evidence in a scenario |
|---|---|---|
| **On-demand self-service** | Provision via portal/API/IaC *without human interaction with the provider* | "Developers launch servers themselves in minutes via console/Terraform" |
| **Broad network access** | Available over standard networks (HTTPS/REST), device-independent | "Accessible from browsers, mobile apps, APIs anywhere" |
| **Resource pooling** | Provider's compute/storage/network dynamically allocated from a common pool to multiple customers (multi-tenant, isolation preserved) | "Shared physical infrastructure, logically isolated tenants" |
| **Rapid elasticity** | Scale out/in automatically with demand; appears unlimited to the consumer | "Traffic spikes handled without manual procurement" |
| **Measured service** | Usage automatically metered; pay-per-use; dashboards for cost transparency | "Pay only for what is consumed; billing tied to usage" |

**Why each matters in answers:** pooling → economies of scale + multi-tenancy; elasticity → CapEx-to-OpEx shift + handling bursty load; metering → cost transparency and pay-as-you-go economics.

## 1.4 Service models (IaaS / PaaS / SaaS) — responsibility split

**IaaS** — consumer provisions **processing, storage, networks**, and can deploy and run *any software including OS and applications*. Consumer does **not** control the underlying cloud infrastructure but controls **OS, storage, deployed applications**, and limited networking (e.g., host firewalls). Bottom layer of the stack. Examples: Amazon EC2, Eucalyptus, Rackspace.
**Use IaaS when:** need compute on demand; dev/test throwaway environments; sudden traffic spikes; CapEx→OpEx conversion; GPU/HPC rental; migration/lift-and-shift of legacy systems needing OS control.

**PaaS** — consumer deploys **applications created with provider-supported languages/tools**; does **not** manage network, servers, OS or storage, but controls the **deployed application and possibly its hosting configuration**. Provides runtimes, build/deploy pipelines, managed data services, autoscaling, logging. Examples: Azure App Service, Google App Engine, Elastic Beanstalk, Force.com.
**Use PaaS when:** standard-stack web apps/APIs, small teams without platform engineers, variable traffic, dev/test environments.

**SaaS** — consumer **uses the provider's application** running on cloud infrastructure, via a thin client (browser) or API. Consumer manages **nothing** except limited user-specific configuration. Examples: Salesforce (CRM), Microsoft 365, Google Workspace.
**Test for SaaS:** can a business *user* sign in and use it without deploying anything? If not, it is probably not SaaS. (AWS/Azure are IaaS/PaaS *platforms*, not SaaS — though they sell some SaaS products.)

**Who manages what (memorise this ladder):**

| Layer | On-prem | IaaS | PaaS | SaaS |
|---|---|---|---|---|
| Application | You | You | You | Vendor |
| Data | You | You | You | Shared |
| Runtime | You | You | Vendor | Vendor |
| Middleware | You | You | Vendor | Vendor |
| OS | You | You | Vendor | Vendor |
| Virtualization | You | Vendor | Vendor | Vendor |
| Servers/Storage/Network | You | Vendor | Vendor | Vendor |

**Golden rule:** accountability for **data, identity and access** never transfers to the provider in any model.

## 1.5 Deployment models + quick comparison

| Model | Analogy | Definition | Strength | Trade-off | Typical adopter |
|---|---|---|---|---|---|
| **Public** | Rent an apartment | Provider-owned infrastructure, services sold to the public on pay-per-use | Scale & speed, zero maintenance, economies of scale | Less control; multi-tenant; compliance concerns | Startups, variable workloads |
| **Private** | Own your house | Infrastructure exclusively dedicated to one organization (self-owned/operated or third-party; on-prem, colo or hosted) | Control, security, customisation, compliance | Higher cost; own capacity planning; under-utilisation risk | Banks, government, regulated data |
| **Hybrid** | Own house + rent when needed | Two or more distinct environments (private+public or on-prem+public) kept **separate but connected** as one unified environment | Flexibility; cloud bursting; keep sensitive data local | Integration complexity; needs unified management | Enterprises with legacy + bursty load |
| **Community** | Shared facility for a group | Shared by organizations with **common concerns** (compliance, security, mission); access restricted to members | Shared cost + common compliance | Shared governance; smaller scale than public | Hospitals, universities, government bodies |

**Hybrid key vocabulary:** **cloud bursting** = run baseline on private, dynamically extend to public during peak. Portability and data integration between the environments are what make hybrid work.

## 1.6 Multi-cloud (distinct from hybrid, distinct from multi-tenancy!)

**Definition:** using **two or more public cloud providers** (e.g., AWS + Azure). Hybrid combines *your own* infrastructure with a public cloud; the two strategies can be combined.

**Advantages:** negotiating leverage (no vendor lock on price/roadmap); best-of-breed services; resilience & redundancy (failover across providers, smaller blast radius); regulatory flexibility (keep data in-region per market).

**Disadvantages:** operational complexity (different consoles/APIs/services); networking & cost overhead (cross-cloud egress fees, latency); inconsistent security & IAM models; higher tooling and skills cost.

**Trap (your EC2 correction):** multi-cloud ≠ multi-tenancy. Multi-tenancy = many *customers* sharing one provider's infrastructure. Multi-cloud = one *customer* using many providers.

**Does multi-cloud automatically improve availability?** No — it *can* reduce blast radius, but only if the architecture actively supports failover across providers (consistent IAM, portable packaging, replicated data, common monitoring). Poorly integrated multi-cloud adds failure modes instead of removing them (integration complexity, inconsistent security, split-brain operations).

## 1.7 Evolution: cloud as convergence, not invention (classic 3-mark justification)

Cloud is the **convergence of existing technologies**, progressively modified for contemporary needs:
- **Distributed computing** lineage: parallel computing → **clusters** (loosely coupled machines acting as one) → **grids** (most distributed form; Internet-connected machines working on one problem) → utility computing.
- **Web technologies + Web services + SOA** (service-oriented architectures) provided the access and interface layer.
- **Virtualization** (1960s mainframes → VMware 1999 → commodity x86) made resource pooling practical.
- **Economics**: Amazon renting its spare retail compute capacity; information explosion from web/mobile making traditional in-house capacity insufficient.

**Why "evolution not innovation" (answer skeleton):** every ingredient existed before (virtualization, distributed systems, SOA, utility billing); cloud's novelty is the *integration at scale with self-service, metering and elasticity* — a business-model and engineering-convergence shift, not a fundamentally new paradigm. Counterpoint to concede: the *scale*, the API-driven self-service model, and the economic model are genuinely new *combinations*.

## 1.8 Economics: CapEx vs OpEx (numerical-adjacent)

- **CapEx (capital expenditure):** upfront purchase of servers, data-centre space, licences — owned asset, depreciated, must be sized for *peak* demand years ahead.
- **OpEx (operational expenditure):** pay-as-you-go consumption — no upfront cost, scales with actual demand, predictable per-unit pricing.

**When cloud (OpEx) is cheaper:** variable/unpredictable/bursty workloads; short-lived environments (dev/test); rapid growth where buying ahead is risky.
**When owning (CapEx) can be cheaper:** large, **steady, highly predictable** baseline loads that would utilise owned hardware at high rates for years; existing sunk infrastructure; very long horizons where per-unit cloud premiums exceed amortised ownership cost. (Same logic as the SaaS-vs-on-premises cumulative-cost crossover: subscription is cheaper early, ownership wins after the crossover year — the horizon decides.)

## 1.9 Cloud vs hosted web application (why "cloud-native" is a real distinction)

| Dimension | Traditional web app (hosted) | Cloud-native application |
|---|---|---|
| Hosting | One or a few servers | Distributed cloud infrastructure |
| Scaling | Manual | Automatic elastic scaling |
| Architecture | Monolithic | Microservices |
| Deployment | Manual | CI/CD automated |
| Fault tolerance | Limited | Self-healing, highly available |
| Technology | Basic virtualization | Containers, Kubernetes, serverless |
| Integration | Limited APIs | API-first |

## 1.10 Advantages, challenges and provider selection

**Advantages:** cost (no upfront CapEx, pay-per-use), elasticity, speed/agility, global reach, managed reliability, focus on core business.
**Challenges / customer-side responsibilities:** security configuration (IAM, MFA, encryption, key management), data governance and regulatory compliance, application patching, cost management (idle/oversized resources), vendor lock-in, connectivity dependence.
**Provider-side responsibilities:** physical data-centre and network security, hardware and virtualization security, HA/DR of the underlying platform, compliance certifications, infrastructure threat detection.
This split is the **shared responsibility model** (developed fully in Module 8).

**Choosing a provider (scenario answer):** consider service coverage, region/AZ locations vs user geography and data-residency law, SLA terms, pricing model fit, ecosystem/tooling, compliance certifications, exit/portability options.

## 1.11 Terminology quick-fire (2-mark fodder)

- **Hosted application:** runs on a server you rent but is an ordinary app (manual scaling).
- **Cloud-enabled:** legacy app partially reworked to use cloud services.
- **Cloud-native:** designed *for* cloud — elastic, distributed, API-first, self-healing.
- **Cloud-agnostic:** designed to run on any cloud without provider-specific dependencies (portability via containers/open standards).
- **Paradigm:** a standard perspective or set of ideas; cloud converged several paradigms rather than replacing them.

---

# Module 2 — Virtualization and Containers

## 2.1 Virtualization: what and why

**Definition:** virtualization creates a virtual (rather than actual) version of something — compute, storage, network, memory or devices — so multiple users share pooled resources while each feels like the sole owner.

**Why (the business case):** server consolidation (many under-utilised physical servers → few fully-used hosts), isolation (failure/tenants separated), encapsulation (entire VM state is a file → easy backup/restore), mobility (copy/migrate VMs like files), reduced space/cooling/cabling/CapEx, faster provisioning.

## 2.2 The x86 virtualization problem (the technical heart — expect 3-4 marks)

- x86 operating systems are written to run directly on bare metal — they **assume they fully own the hardware** and expect to run in **Ring 0** (kernel mode, full access).
- x86 has four privilege rings: **Ring 0** kernel (full access), Ring 1/2 (drivers; Ring 2 unused today), **Ring 3** user applications. Access to sensitive resources must go through syscalls handled in Ring 0.
- To virtualize, the hypervisor must sit *below* the OS — but there is no natural room: the guest OS still expects Ring 0. This is the **ring conflict / ring deprivileging** problem.
- The deeper blocker: **sensitive but not privileged instructions.** The classic technique is *trap-and-emulate* — privileged instructions trap to the hypervisor, which emulates the result. But ~17 x86 instructions are **sensitive without being privileged**: they behave differently depending on the ring yet **do not trap** when executed outside Ring 0 — they **fail silently**, and the hypervisor cannot catch them. Popek & Goldberg's formal virtualization requirements are therefore not met by x86.
- Two fixes emerged: **binary translation** (VMware, 1998/99 — rewrite sensitive instructions in software before execution) and **hardware-assisted virtualization** (Intel VT-x / AMD-V, 2005–2006 — add a privilege level *below* Ring 0 so the CPU itself traps correctly). KVM dominance and OVF image standards later brought partial convergence/openness.

## 2.3 Emulation vs virtualization (a favourite comparison)

| Aspect | Emulation | Virtualization |
|---|---|---|
| Mechanism | Software reproduces the *behaviour* of different hardware (CPU instruction set, memory, I/O); guest sees a virtual hardware profile decoupled from the physical host | Guest executes (mostly) natively on the same ISA; hypervisor traps/handles privileged operations |
| ISA | **Cross-ISA possible** (ARM guest on x86 host; Bochs, QEMU-TCG, Rosetta 2) | Same ISA required (x86 guest on x86 host) |
| Execution | Nothing runs natively; guest OS+apps are ordinary user-mode processes — no host privilege needed | Guest executes at near-native speed; only sensitive operations are intercepted |
| Techniques | Interpretation (fetch-decode-execute each instruction, simple but slow) and Dynamic Binary Translation (translate a block once, cache, reuse) | Binary translation (full virt.), hypercalls (para-virt.), hardware assist (VT-x/AMD-V) |
| Performance | Significant overhead even with DBT | Near-native (hardware-assisted) |
| When used | Cross-platform, legacy hardware support, architecture migration | Standard cloud same-ISA virtualization |

**Memory hook:** emulation = *translator* (can translate any language, slowly); virtualization = *interpreter doing the real job with supervision*.

## 2.4 The three virtualization approaches (THE comparison table — memorise)

| Aspect | Full virtualization (binary translation) | Para-virtualization | Hardware-assisted |
|---|---|---|---|
| Guest OS | **Unmodified**; believes it owns full hardware | **Modified kernel** (aware it is virtualized) | **Unmodified** |
| Mechanism | Hypervisor dynamically rewrites/translates sensitive instruction streams at runtime (ring deprivileging: guest kernel to Ring 1 on 32-bit, VMM at Ring 0) | Guest replaces privileged instructions with **hypercalls** — explicit API calls to the VMM | CPU adds **Ring −1** (below ring 0); privileged & sensitive instructions trap automatically to the hypervisor |
| Guest support | Any OS incl. closed-source Windows | Historically open-source only (Linux/BSD) — needs kernel source changes or para-virtualized drivers (e.g., for network/graphics devices) | Any OS; introduced by IBM System/370 concept, then Intel VT-x / AMD-V |
| Performance | Overhead from translation + software device emulation | Better: no runtime trap-and-translate; guest asks directly | Best/near-native; removes translation complexity |
| Examples | VMware ESX/Workstation (first commercial 1999) | Xen (para-virt. mode) | KVM, Hyper-V, Xen HVM, VirtualBox (with VT-x) |

**Answer skeleton for "differentiate the three":** state (1) guest OS modified or not, (2) the mechanism (translation / hypercall / hardware trap), (3) performance consequence, (4) one example each.

## 2.5 Hypervisors (VMM): definition, goals, types

**Definition:** a hypervisor (Virtual Machine Monitor) is software, firmware or hardware that **creates, runs and manages virtual machines**. The machine it runs on = **host**; each VM = **guest**. It presents guests a virtual operating platform and manages execution of guest OSes.

**Goals (Popek-Goldberg flavored):** fidelity/equivalence (guest runs identically), performance/efficiency (most instructions execute natively), safety/isolation (full control of physical resources; guests isolated from each other), resource multiplexing/sharing.

**Type 1 (bare-metal):** runs **directly on hardware**; no host OS beneath. Higher efficiency, stronger isolation, used in data centres/cloud. Examples: VMware ESXi, Xen, Hyper-V, KVM (Linux kernel itself acts as the hypervisor).
**Type 2 (hosted):** runs as an **application on a host OS** (e.g., VirtualBox, VMware Workstation). Easier setup, good for desktops/dev/test; extra layer → more overhead.

**Virtualization stack levels (where virtualization can be implemented):** Instruction-set level (ISA emulation, e.g., Bochs) → Hardware abstraction level (classic VMs) → OS level (containers sharing one kernel) → Library/API level (WINE-style) → Application level. Hardware & OS levels give highest performance; ISA gives best flexibility; hardware and application levels are costliest to implement; user isolation is hardest.

## 2.6 Virtualization applied to each resource (2-mark each)

- **Server virtualization:** one physical machine hosts multiple VMs running independent OSes; enables IaaS. Benefits: partitioning, isolation, encapsulation (VM = file), hardware independence, faster provisioning/restore.
- **Storage virtualization:** logical abstracted view of physical devices; many devices appear as one pool (SAN/iSCSI/Fibre Channel for block; NAS for file). Benefits: optimal utilisation, easy non-disruptive growth, improved availability & performance.
- **Network virtualization:** logical segmentation of physical networks — **VLANs** (split a LAN into broadcast domains on a port basis), VRF (multiple routing tables in one router, used in MPLS), overlay networks (used by Swarm/K8s).
- **Memory virtualization:** sharing physical RAM and dynamically allocating to VMs; like OS virtual memory — applications see contiguous address space decoupled from physical pages; MMU + TLB manage mappings (hypervisor adds nested/shadow page tables).
- **Device virtualization:** each VM sees a virtual peripheral (NIC, disk controller, video); the hypervisor routes I/O between virtual devices and shared physical hardware.

## 2.7 Overcommitment — CPU vs memory (numerical + conceptual, very examinable)

**CPU overcommitment — possible because CPU time is *time-sliced*:**
- Overcommit ratio = **total vCPUs ÷ total pCPUs**.
- Guidance: **1:1** latency-sensitive (databases, real-time); **3:1–5:1** workable for mixed loads; **>8:1** high risk.
- **CPU Ready time** = time a vCPU is ready to run but waits for a physical core; rising above ~5% signals oversubscription.
- Worked example (course): 4 pCPU host, 6 VMs × 2 vCPU = 12 vCPU → ratio 12÷4 = **3:1** (edge of safe band). Subtlety: if 3 VMs peg both vCPUs simultaneously, the busy subset demands 6 vCPU on 4 pCPU = **1.5:1 squeeze** — the aggregate ratio is a planning heuristic, not a guarantee.

**Memory overcommitment — harder, because memory cannot be time-sliced** (a page is allocated or not at any instant). The hypervisor must actively reclaim; techniques least → most invasive:
1. **Page sharing (TPS/KSM)** — deduplicate identical pages (inter-VM sharing disabled by default since vSphere 6.0 for security; intra-VM remains).
2. **Ballooning** — a guest-side driver reclaims memory *from the guest's own processes*; the **guest OS decides** what to page out (cooperative, preferred).
3. **Compression** — hypervisor compresses pages instead of swapping (slower than RAM, far faster than disk).
4. **Host-level swapping** — last resort; swap to disk, which is ~100,000× slower than RAM.

Ratio = **configured memory ÷ physical RAM**. Numerical: 4 VMs × 8 GB configured on a 24 GB host → 32÷24 = 1.33:1; reclaim techniques cover the 8 GB shortfall if guests don't actively use it all.

## 2.8 Virtualization caveats (small but asked)

- **Software licensing:** each VM needs its own licence; server sprawl inflates licence costs.
- **IT training:** staff must learn virtualization-specific debugging, security, capacity planning.
- **Hardware investment:** consolidation pays off only with powerful hosts; weak existing hardware may need upfront investment.

## 2.9 Containers: what and why

**Motivation (the story to tell):** code that "works on my machine" fails on the server — different libraries, OS versions, configs. Developers waste time debugging environment differences. Containers package the app **with all its dependencies** so it runs identically everywhere.

**Definition:** a container is an **OS-virtualization technology** that packages application code + dependencies and runs it in **isolated user-space environments sharing the host kernel**, under a **container engine** (Docker, CRI-O) that runs on top of the OS. Containers run consistently on any container-capable host; dependencies are baked into the image, simplifying handoff and release.

**Position:** kernel (process mgmt, CPU scheduling, file system, I/O) → container engine → containers. Apps normally reach hardware via **syscalls** (e.g., fopen/fwrite) handled by the kernel.

## 2.10 Namespaces and cgroups — the two kernel pillars (must-know mechanism)

| | **Namespaces** | **cgroups (Control Groups)** |
|---|---|---|
| Question answered | **What can you see?** | **How much can you use?** |
| Mechanism | Wrap a global system resource in an abstraction so processes inside see their **own isolated instance** (own PIDs, own network stack, mounts, etc.) — `ps aux` in a container shows 3 processes, on the host hundreds, same process different view | Aggregate/partition sets of processes (and children) into hierarchical groups with **resource limits** (CPU, memory, disk I/O) — "this group gets at most 2 cores and 4 GB combined"; kernel **throttles or OOM-kills** violators |
| Effect | Isolation of view (PID, network, mount, IPC, UTS, user namespaces) | Protection of resources — one runaway container hits its ceiling while everyone else keeps running |
| Origin | Linux kernel feature | Built by Google engineers (2006), mainline kernel 2008 |

**One-liner to memorise:** *namespaces limit what you can see; cgroups limit how much you can use.* A container needs **both**: isolation without limits still lets a noisy container starve the host; limits without isolation still exposes other processes.

## 2.11 LXC, LXD, OS vs application containers

- **LXC (Linux Containers):** a *userspace interface* to the kernel's containment features (not an OS) — tools + libraries letting one Linux kernel host multiple isolated environments using namespaces + cgroups. All containers on a host share one kernel: no guest OS, no hypervisor, no boot time.
- **LXD:** a **management layer on top of LXC** (REST API over liblxc) providing centralised daemon-based administration, networking/storage/security management at fleet scale.
- **OS containers** (LXC): behave like a *lightweight machine* — run many processes/services; you can "install" and configure like an OS. Good for fleets of similar/different distro environments.
- **Application containers** (Docker, Rocket): package and run **a single application/service** process. This is the microservices-friendly model.

## 2.12 VM vs Container (THE classic comparison — reproduce confidently)

| Dimension | Virtual Machine | Container |
|---|---|---|
| Virtualization level | Hardware-level (hypervisor) | OS-level (kernel features) |
| Guest OS | Each VM carries a full guest OS | **Shares the host kernel**; no guest OS |
| Isolation | Strong (hardware-level, separate kernel) | Process-level (namespaces/cgroups); weaker — kernel shared |
| Size | GBs | MBs |
| Boot/start | Minutes | Seconds (or less) |
| Density on one host | Fewer (heavy) | Many more (light) — high compute density |
| Portability | Image formats (OVF); heavier to move | Highly portable image; "build once, run anywhere" container-capable |
| Overhead | Hypervisor + full OS per VM | Negligible; processes near-native |
| Use when | Different OSes/kernels needed, strong isolation/compliance, legacy apps, monoliths | Microservices, CI/CD, dev-prod parity, high-density microservice fleets |

**When VMs remain preferable:** different OS/kernel requirement, hard compliance/isolation boundaries, software requiring kernel modules, legacy licensing, multi-tenant SaaS platforms selling whole machines.

## 2.13 Docker: architecture, components, objects

**What Docker adds:** a tool to **create, ship and run** containers easily — wraps software in a complete filesystem (code, runtime, system tools, system libraries); consistent deployment regardless of environment; **compute density** via many isolated containers on one host; isolation via namespaces, limits via cgroups.

**Architecture (client–server):**
- **Docker daemon (dockerd):** listens for Docker API requests; does the heavy lifting — builds, runs, manages objects (images, containers, volumes, networks); can talk to daemons on other hosts (e.g., in a swarm). Delegates container creation to containerd → runc (OCI runtime).
- **Docker client (docker CLI):** primary user interface; sends commands (docker run, build, pull) to the daemon via REST API over UNIX socket or network; can talk to multiple daemons. Docker Compose = multi-container client tool; Docker Desktop bundles daemon+client+Compose+Kubernetes.
- **Registry:** stores images. **Docker Hub** = default public registry; private registries possible. `pull`/`run` fetch images; `push` uploads. Repository = set of related images distinguished by **tags**. (Caution: use official/verified images.)

**Docker objects:**
- **Image:** read-only **layered template**. First Dockerfile instruction is `FROM` (base image); each instruction adds a layer; the **union filesystem** stacks layers into one view. Immutable and stateless.
- **Container:** a **running instance of an image** with a **writable layer** on top. Isolated, encapsulated; resource/network/port/storage config from the image or run flags. Very light, starts in milliseconds.
- **Dockerfile:** text recipe that builds images. Why it matters (5 reasons — exam list): (1) **reproducibility** (same file + context → same image; vs undocumented `docker commit`); (2) only legitimate way to customise a base image (COPY/RUN); (3) self-documenting (instruction order reveals dependency graph); (4) automates CI/CD builds; (5) **layer caching** — unchanged layers are reused, so rebuilds after small code changes are fast. This is why the dependency file (`requirements.txt`) is COPYed **before** the source code. Naming: exactly `Dockerfile` (no extension) by default; variants need `-f Dockerfile.prod`; location = build-context root.
- **Copy-on-Write:** containers **share image layers**; only when a container modifies a file is it **copied into its own writable layer**. Makes container creation instant and storage-efficient; many containers from one image cost almost nothing extra.

**Docker lifecycle (states):** created → running → paused → stopped → deleted (docker create / start / pause / unpause / stop / rm).

---

# Module 3 — Container Orchestration: Docker Swarm & Kubernetes

## 3.1 Why orchestration exists (the scenario framing)

Single-machine container management is easy (`docker run` = one container, one machine; `docker compose` = several containers, one machine). Problems appear **across machines**: machine failure, traffic growth, container crashes, new version rollouts.

**Container orchestration** automates **deployment, placement, scaling and management** of containers across many machines. It continuously **compares desired state vs observed state** and takes corrective action: replace failed replicas, scale with traffic, move containers off dead/short hosts, expose services, load-balance, monitor health, manage configuration.

Typical automated tasks: provisioning/deployment; scaling & load spreading; re-placement on failure; external exposure of services; service discovery; health monitoring; configuration of apps relative to their containers.

## 3.2 Imperative vs declarative model (recurring 3-marker; also used in IaC)

- **Imperative = "HOW":** a sequence of commands changes the system step by step (TV-remote analogy: press buttons; if someone else changes the channel, it stays changed). Commands depend on order and previous state; no automatic correction.
- **Declarative = "WHAT":** you write the **desired end state** in a file (YAML/JSON); the tool continuously converges reality to it (thermostat analogy: set 24 °C; the thermostat corrects automatically). Same file applied repeatedly is safe (**idempotent**). Enables **GitOps**: desired state lives in Git; every change is a reviewed, versioned commit; rollback = revert a commit.
- Orchestration tools (and Kubernetes) are declarative: the config defines images + registries, resources, secured network connections, versioning/canary rollout; the orchestrator schedules replicas onto the best hosts by CPU/memory/constraints.

## 3.3 Docker Swarm — the lightweight orchestrator

**What:** "swarm mode" turns several Docker hosts into **one cluster**; built into Docker Engine since 1.12 (2016) — no extra software; `docker swarm init` creates it. Same images, same CLI, Compose-file compatibility. Analogy: team lead distributes jobs and replaces anyone who stops working.

**Nodes:** manager and worker machines (physical or VM, e.g., EC2 instances).
- **Inside a manager:** API (receives service/stack commands) → Orchestrator (compares wanted vs running replicas) → Allocator (assigns IPs on overlay networks) → Scheduler (chooses the node per task) → Dispatcher (sends tasks to nodes, receives heartbeats) → **Raft store** (replicated record of desired state — consensus).
- **Inside a worker:** Docker Engine + Swarm agent (receives tasks, reports state).

**Object hierarchy (memorise):** **Stack** (whole application, one file) → **Service** (one part as a definition: "run 3 copies of api:v1") → **Task** (one scheduled copy; carries its own state) → **Container** (exactly one running process per task).

**Service types:** **Replicated** (exactly N copies anywhere — web servers/APIs) vs **Global** (exactly one per node, auto-added to new nodes — monitoring/log agents).
**Placement controls:** **constraints** (hard rules, e.g., `node.labels.db == true`; unmatched tasks wait Pending), **preferences** (soft spread across zones/racks), **resource reservations** ("0.1 CPU, 96 MB"), default strategy = fewest tasks of that service.

**Networking:** **overlay network** spans nodes (frontend: web↔api; backend: api↔redis/db); **ingress/routing mesh** — publish a port once, *any* node accepts the connection and forwards to an available task even on another node; built-in **DNS** for service-name discovery (`http://api:3000`); **Virtual IP** + internal load balancing (or DNS round-robin mode).
**Cluster ports:** 2377/tcp cluster management; 7946/tcp+udp node gossip; 4789/udp overlay data. TLS mutual authentication set up automatically.

**Manager fault tolerance (numerical — appears in papers):** managers agree by **Raft**; failures tolerated = **⌊(N−1)/2⌋**. 3 managers tolerate 1 failure; 5 tolerate 2. **An even count adds nothing** over the odd count below it (4 managers still tolerate only 1) — the quorum is majority.

**Pros/cons:** + zero extra install, simplest learning curve, Compose reuse, decent built-in security; − smaller ecosystem, fewer features than K8s (no native autoscaling of replicas, simpler scheduling), weaker community momentum.

## 3.4 Kubernetes — concepts and architecture

**What:** open-source platform that **runs containers across a cluster**; you *describe the desired result* in files; Kubernetes continuously works to make the cluster match. From Google's internal **Borg**, open-sourced 2014, v1.0 2015, donated to CNCF. K8s does **not run containers itself** — it decides and supervises; a **container runtime** on each machine does the running. Acts like a self-service **PaaS** with a hardware-abstraction layer; portable across AWS/Azure/GCP/on-prem (anti-lock-in standardisation).

**Reconciliation loop (why declarative matters):** you declare `replicas: 3` → K8s stores desired state → observes (2 running) → acts (start 1 more) → repeats forever. Thermostat analogy.

**Why K8s when Swarm exists:** richer ecosystem and feature set — stronger scheduling (filter → score → bind), richer object model (Deployments, StatefulSets, ConfigMaps/Secrets, HPAs, CRDs), larger community, self-healing controllers, rolling updates/rollbacks, autoscaling, RBAC.

**Control plane (the brain):**
- **kube-apiserver** — *the front door*: the only component everyone talks to (kubectl, dashboard, controllers). Performs **authentication → authorization → admission**; stores accepted objects in etcd; notifies watchers; runs HA behind a load balancer.
- **etcd** — *the memory*: small consistent **key-value store** holding the *entire cluster state* (every object, desired + current status); replicas agree by **Raft majority**; **only the apiserver reads/writes it**; back up etcd = back up the cluster. *"If it is not in etcd, Kubernetes does not know it exists."*
- **kube-scheduler** — *the planner*: watches for pods with no node; **filter** (eliminate nodes that can't fit CPU/memory requests, wrong labels/taints) → **score** (rank: spread copies, balance load) → **bind** (record the choice). Never starts containers — only decides *where*.
- **kube-controller-manager** — *the fixers*: dozens of control loops, each owning one job (Deployment/ReplicaSet controllers keep replica counts; Node controller detects dead nodes; Job controller runs tasks to completion; EndpointSlice controller updates each Service's pod addresses).
- **cloud-controller-manager** — links the cluster to the cloud provider's API (managed clusters).

**Worker node components:**
- **kubelet** — node agent; talks to the apiserver; ensures the pods assigned to *its* node are running and healthy.
- **kube-proxy** — network rules on every node; implements Services (e.g., opens the NodePort on every node, forwards to ready pods).
- **Container runtime** (containerd/CRI-O) — actually pulls images and runs containers.

**Add-ons:** CNI network plugin (pod network, e.g., Flannel), **CoreDNS** cluster DNS, metrics-server, CSI storage drivers, ingress controller.

**Every K8s object has four parts:** `apiVersion` + `kind` (what it is) · `metadata` (name, namespace, labels) · `spec` (**desired state — you write it**) · `status` (**observed state — K8s writes it, never edit**). `kubectl apply -f file.yaml` sends it.

## 3.5 Core K8s objects (the 4-mark answer bank)

- **Pod:** smallest deployable unit; one or more containers sharing storage/network and a lifecycle; each pod gets **its own IP** in the pod network. Ephemeral — replaced constantly, IPs change.
- **Labels & selectors:** key-value tags (`app: api`, `tier: backend`); queries over labels are how Services/Deployments/ReplicaSets find their pods. **Why not IPs?** Pods churn; labels are stable.
- **ReplicaSet:** maintains N identical pods; the **workhorse that keeps the desired replica count** (replace failed pods).
- **Deployment:** manages ReplicaSets to provide **declarative updates, rolling updates and rollbacks** of pods; the usual way to run stateless apps.
- **Service:** **stable networking** for a dynamic set of pods — a fixed virtual IP/DNS name; **selects pods by label** and load-balances across them. Types: **ClusterIP** (internal), **NodePort** (opens the same port on every node, e.g., 30080), **LoadBalancer** (cloud LB), **headless** (no VIP; returns pod IPs — used by StatefulSets).
- **Namespace:** named virtual space inside one cluster (`freshcart`, `kube-system`); names unique per namespace; access rules and resource quotas set per namespace.
- **StatefulSet:** for stateful pods needing stable identity + persistent storage (e.g., postgres-0 with a PVC).
- **ConfigMap / Secret:** configuration and sensitive data injected into pods.
- **HPA (Horizontal Pod Autoscaler):** changes replica count with load (e.g., api 2–8).

**Your EC2 correction, restated:** ReplicaSets/Deployments maintain replicas; Services provide stable networking via selectors; **etcd only stores state** — it does neither of the above.

## 3.6 Requests vs limits + scheduling (numerical territory)

- **Request** = the resources the **scheduler reserves** — the pod's *guaranteed share*; drives placement (filter stage) and maps to cgroups shares/weights.
- **Limit** = the **hard runtime cap**. CPU limit exceeded → **throttling** (slowdown). Memory limit exceeded → the kernel **OOM-kills** the container.
- **Node allocatable** = node capacity − system reserved (kubelet, OS, eviction threshold). Example: a 2-vCPU node has ~1.9 vCPU allocatable → pods requesting 500m each fit **3** (1.9 ÷ 0.5 = 3.8 → floor 3); the **4th pod stays Pending even though the node is idle** — requests reserve capacity whether or not it is used.
- Units: **1 CPU = 1000 millicores (m)**; memory in MiB/GiB.

**Pod-capacity method (memorise the 3 steps):**
1. Per node: floor(allocatable CPU ÷ request CPU) and floor(allocatable memory ÷ request memory).
2. Pods per node = **min(CPU-fit, memory-fit, maxPods limit)**.
3. Cluster total = pods per node × usable nodes (subtract tainted/broken nodes); remaining capacity = total − already running.

**Why real capacity < theoretical maximum (the caveat marks):** system daemons and K8s overheads eat allocatable capacity; non-evictable system pods; maxPods-per-node limits (kubelet, ~110 default); other workloads sharing the node; nodes not schedulable (taints, maintenance); pods needing more than requests at runtime (limits/bursts); fragmentation (one big pod can't fit the leftover of two small pods' worth); images/startup overheads; scheduler spreads pods for resilience rather than packing to the max.

## 3.7 K8s troubleshooting playbook (scenario marks)

| Symptom | Likely cause | First checks |
|---|---|---|
| Pod **Pending** | Not enough allocatable CPU/memory (requests too high), taints, unsatisfiable constraints, no PVC bound | `kubectl describe pod` → FailedScheduling message; node allocatable vs requests |
| Pod **CrashLoopBackOff** | App crashes at start; bad config/secret; memory limit → OOMKill | Logs (`kubectl logs --previous`), exit codes, secret/config correctness |
| **OOMKilled** | Memory limit too low for real usage | Usage vs limit; raise limit or fix leak; be conservative with memory requests |
| Service unreachable | Label/selector mismatch; wrong port; kube-proxy rules; endpoints empty | `kubectl get endpoints` — if empty, selector doesn't match pod labels |
| Pod on wrong node | Node selector/affinity/taints | Node labels, taints/tolerations |

**Scenario framing (payment-service replicas failing + traffic rising):** ReplicaSet controller replaces failed pods (self-healing); Deployment manages rollout of new versions; HPA scales replicas with load; Service keeps a stable endpoint across all of it.

## 3.8 Swarm vs Kubernetes (comparison table)

| Aspect | Docker Swarm | Kubernetes |
|---|---|---|
| Install | Built into Docker Engine | Separate, heavier setup (managed offerings mitigate) |
| Model | Stack → Service → Task → Container | Deployment/ReplicaSet → Pod, Services, namespaces |
| State store | Raft on managers | etcd (via apiserver) |
| Scheduling | Simple strategy + constraints/preferences | Filter-score-bind, rich constraints/affinity/taints |
| Autoscaling | Manual replica changes | HPA/VPA/Cluster Autoscaler |
| Updates | Rolling service updates | Rolling updates, canary, automatic rollback |
| Learning curve | Low | High |
| Ecosystem | Small | Huge (CNCF) |
| Fit | Small clusters, simple apps, Docker-native teams | Large, complex, production-scale platforms |

---

# Module 4 — Infrastructure as a Service (IaaS) and AWS

## 4.1 IaaS model and actors (NIST/NIST-adjacent, asked as definition + application)

**Recap:** IaaS delivers virtualized compute, storage, network on demand; consumer controls OS, storage, applications, limited networking; provider controls the physical infrastructure and virtualization layer.
**IaaS motivations (course list):** no upfront cost; multi-tier auto-provisioning within hours; scalability on demand; flexible/secure direct connectivity; SLA with (e.g.) 99.9% uptime.

**Five actors (2-3 marks — name + one line each):**
1. **Cloud Consumer** — the person/organization that uses services.
2. **Cloud Provider** — builds and operates the cloud, responsible for the service.
3. **Cloud Broker** — intermediates: aggregation, integration, customisation of services between provider and consumer.
4. **Cloud Auditor** — independent assessor of cloud services (security, privacy, performance).
5. **Cloud Carrier** — the connectivity/transport (network) between consumer and provider.

**IaaS architecture essence:** virtualization + management layer turns physical compute/storage/network into on-demand VMs — "the final outcome of the interactions of the technologies we learnt is what we colloquially refer to as IaaS."

## 4.2 AWS global infrastructure (definition-level precision matters)

| Construct | What it is | Key property | Example code |
|---|---|---|---|
| **Region** | A geographic area (Mumbai, Frankfurt) — a **naming boundary** grouping a cluster of AZs, *not a building* | Each Region **completely independent/isolated** from others (fault isolation; data residency); services run only in your selected/default Region (changeable via console, `AWS_DEFAULT_REGION`, `--region`) | `ap-south-1` |
| **Availability Zone (AZ)** | One or more **discrete data centres** within a region, with dedicated power substation, cooling, network uplinks; **physically separated** (different flood plains, different power grids, enough distance that one disaster can't take out two) | AZs are isolated from each other but **connected by low-latency links**; spreading instances across AZs protects against a location failure; each AZ has redundant power (UPS + generators, multiple grids) and multiple tier-1 transit providers | `us-east-1a` |
| **Local Zone** | Extension of a Region **close to users** (city), own internet + Direct Connect | Low-latency serving to local users; opt-in required | `us-west-2-lax-1` |
| **Wavelength Zone** | AWS infrastructure **embedded in telecom 5G networks** | Traffic never leaves the provider's network → ultra-low latency for mobile/edge apps (autonomous cars, smart robots) | — |
| **Outposts** | Fully **managed AWS hardware installed at the customer's premises** | Same APIs/tools as the Region; local low-latency processing and **data residency** (bank core processing, factory floor); part of the parent Region's VPC | — |
| **Data centre** | The actual building (racks, servers, concrete) | The only term with a street address | — |

**Failure-design lesson (course slide):** if all instances sit in one location and that location fails, none are available → distribute across AZs. Also: not all services exist in all Regions — check availability before launching.

## 4.3 IAM: authentication, authorization, roles, policies (scenario gold)

**What IAM is:** manages digital identities and controls access to resources — the layer deciding **who (or what)** can access **which resource** under **what conditions**. In IaaS every resource is network-reachable — there is no physical wall — IAM *is* the locked door (e.g., "who may spin up a $50K/month cluster?"). Examples: AWS IAM, Azure Entra ID, Google Cloud IAM.

- **Authentication = "Who are you?"** Verifies identity via credentials (password+hash, key pairs, MFA tokens, federated SSO). Output: a confirmed identity — **nothing more**.
- **Authorization = "What are you allowed to do?"** Decides which actions on which resources an authenticated identity may take; evaluated **on every request** (continuously), not once per session. Enforces **least privilege**. Example: user logged in fine but gets *Access Denied* on `ec2:TerminateInstance` because the policy lacks it.
- **Roles** = identity **without a person attached**; assumable temporarily by humans, applications or AWS services; **no long-term password or access key** — assumption yields **short-lived, auto-rotating credentials**. Solves: hardcoded-permission anti-pattern (permanent exportable keys), and machine identity (a Lambda function can't type a password). Examples: EC2 instance role reading a specific S3 bucket (no keys on the box); Lambda execution role writing one DynamoDB table; cross-account auditor role.
- **Policies** = the rule book (JSON in AWS) attached to users/groups/roles: **Effect (Allow/Deny) + Action + Resource + optional Condition**. Explicit **Deny always overrides Allow**. Convert organisational rules ("interns can't delete anything") into mechanically enforced checks at scale. Example: student role with `s3:GetObject` on `course-materials/*` only.

**Least privilege with users/roles/policies (scenario pattern):** developers get scoped policies for their services; DBAs get database-service policies; operators get monitoring/start-stop policies; **roles** (not keys) for machines; **MFA** for humans; review via access logs and policy audits (remove unused permissions, tighten wildcards).

**Why "admin access for every developer" is bad practice (2 marks):** violates least privilege; blast radius of a compromised credential becomes the whole account; no separation of duties; accidental actions (deleting prod data) become possible; harder to audit *who did what*.

**Authentication vs authorization in one line each (2 marks):** Authentication proves *identity* (who you are); authorization decides *permissions* (what you may do) — authn once per session, authz on every request.

## 4.4 EC2 — compute service

**What:** AWS's primary web service providing **resizable compute capacity** via **instances** (virtual servers); billed **per hour/second while running**; stop = no compute charge (storage still billed).

**Two launch decisions:**
1. **Instance type** = the **virtual hardware** (vCPUs, memory, storage, network) — "like the processor". Families (letter mnemonics): **t** tiny/turbo (burstable), **m** balanced general purpose, **c** compute-optimised, **r** RAM/memory-optimised, **x** extreme memory, **p** GPU/processing (AI-ML training), **i** IOPS/storage-optimised SSD, **d** dense HDD; **g** Graviton (ARM), **a** AMD. Sizes scale linearly within a family (vCPU:memory ratio constant). Network performance: low/moderate/high or explicit Gbps; **enhanced networking (SR-IOV)** for latency-sensitive work.
2. **AMI (Amazon Machine Image)** = the **initial software state**: OS + configuration + patch level + pre-installed software — "like the OS". **Your correction:** an AMI does *not* define custom physical hardware.

**AMI sources (4 types — asked directly):**
1. **Published by AWS** — standard Linux/Windows images (patch immediately after launch).
2. **AWS Marketplace** — partner software pre-installed; licence cloud-appropriate; hourly software surcharge possible.
3. **Generated from existing instances** — configure a golden instance to corporate standards, create AMI, use it for all future instances (consistency + governance).
4. **Uploaded VMs (VM Import/Export)** — from raw, VHD, VMDK, OVA formats (migrate on-prem/other-hypervisor servers).

**Accessing/connecting:** public DNS name (auto-generated), public IP (unique while running, not transferable), **Elastic IP** (static, reserved, re-attachable; billed hourly when *not* attached to a running instance). Login via **key pairs** — AWS stores the public key; you keep the private key (needed for first login).

**Lifecycle states:** pending → running → stopping → stopped → shutting-down → terminated. Notes: **stopped** keeps disks; no compute charge but storage charged; start may place on a different host and **changes an auto-assigned public IP**; **terminated is final** (root volume released unless configured to keep); **reboot** stays running; **hibernate** saves RAM to the root volume then stops. **Bootstrapping** = providing code to run at launch.

**Tenancy options (3):** **Shared** (default — one host may house instances of different customers; AWS isolates instances fully and does not overprovision); **Dedicated Instance** (hardware dedicated to one customer; more dedicated instances may get more dedicated hardware); **Dedicated Host** (a whole physical server for one customer — helps server-bound software **licensing** and compliance).

**Placement groups:** logical grouping of instances **within one AZ** for low-latency 10 Gbps inter-instance networking (choose enhanced-networking instance types). (AWS today also has spread/partition groups — the deck emphasises the low-latency cluster idea.)

**Instance store:** **temporary, ephemeral block storage physically attached to the host**; included in instance cost; data is lost on **drive failure, instance stop, or termination** (persists only across a reboot) → use for scratch/temp; keep durable data on EBS.

**Tags & monitoring:** tags = key/value pairs (project, environment, cost centre) for management; up to 50 per instance; **Amazon CloudWatch** provides monitoring/alerting.

## 4.5 VPC — your private network in AWS

**What:** a logically isolated virtual network dedicated to your AWS account — like designing your own data-centre network without hardware. You choose the **IP range (CIDR)**, create **subnets**, configure **route tables, gateways, security**.

**Components:** subnets (a range of IPs in the VPC, within one AZ); route tables; DHCP option sets; **security groups**; **network ACLs**. Optional: **Internet Gateway (IGW)**, Elastic IPs, ENIs, VPC endpoints, **peering**, **NAT instances/gateways**, Virtual Private Gateway + Customer Gateway + VPN.

**Public vs private subnets (architecture pattern for a 3-tier app):**
- **Public subnet:** route to the IGW; hosts internet-facing tiers (load balancer, web servers) — resources get public IPs.
- **Private subnet:** no route to the IGW; hosts internal tiers (application servers, databases); outbound internet (patches) via **NAT gateway** in a public subnet; inbound management via VPN/bastion.
- **Security groups** = stateful, instance-level firewalls (allow rules only; referenced by SG or CIDR). **Network ACLs** = stateless, subnet-level, numbered allow/deny rules evaluated in order. (Classic comparison: SG stateful at instance level vs NACL stateless at subnet level.)

## 4.6 AWS storage services (selection questions are guaranteed)

**Storage types first:** **Block** = low level, raw device, fixed-size numbered blocks (SAN/iSCSI/Fibre Channel). **File** = OS level, hierarchy of files/folders (NAS). **Object** = independent of servers, accessed via API over HTTP; each object = data + metadata + unique key in a flat container.

| Service | Type | Key properties | Use for |
|---|---|---|---|
| **S3 (Simple Storage Service)** | Object | Buckets (flat, no sub-buckets) hold unlimited objects identified by unique keys; managed via API on HTTP verbs (PUT/GET whole object — *not* incremental partial updates like a file system); pay only for used storage; automatic replication across devices/facilities in a region; optimized for reads | Backup/archive, content/media distribution, big-data analytics, static website hosting, cloud-native app data, disaster recovery |
| **EBS (Elastic Block Store)** | Block | **Persistent** block volumes attached to EC2; replicated **within its AZ**; a volume attaches to **one instance at a time** (many volumes can attach to one instance); persists independently of instance life | Boot volumes, SQL/NoSQL databases, big data, data warehouses, transaction processing |
| **EBS General-Purpose SSD** | Block | Cost-effective, balanced; 1 GB–16 TB; baseline ~3 IOPS/GB up to 10,000 IOPS | Boot volumes, small-medium DBs, dev/test |
| **EBS Provisioned-IOPS SSD** | Block | Highest, *predictable* performance; most expensive per GB; 4 GB–16 TB | I/O-intensive, mission-critical DBs needing sustained IOPS |
| **EFS (Elastic File System)** | File | **Shared elastic file system** mounted by many servers simultaneously; fully managed; scales up/down automatically; pay for what you use | Lift-and-shift, shared web-server content, big-data analytics, dev/test |
| **Glacier** | Object (archive) | Extremely low cost, durable archive; retrieval time of **3–5 hours** acceptable | Infrequently accessed archives, compliance retention |

**EFS vs EBS vs S3 decision rules:** need a *bootable disk / database volume* → **EBS**; need *many machines sharing files* → **EFS**; need *write-once/read-many objects, static content, backups* → **S3**; deep archive → **Glacier**.

## 4.7 AWS database and data services

- **RDS (managed relational):** engines — MySQL, PostgreSQL, MariaDB, Oracle, SQL Server, Db2, **Aurora** (AWS's MySQL/PostgreSQL-compatible engine). **Multi-AZ** = synchronous standby in a second AZ, **automatic failover** — an *availability/durability* feature, not performance. **Read Replicas** = asynchronous read-only copies (same or cross-region) to **scale read-heavy workloads** (up to 5 most engines, 15 Aurora). Automated backups on by default (1–35 days, point-in-time recovery).
- **When relational beats NoSQL:** structured schema, ACID transactions (payments, orders, inventory), complex joins/analytics, strong consistency requirements.
- **NoSQL (e.g., DynamoDB):** massive scale, flexible schema, key-value/document access, predictable single-digit-millisecond latency at any scale.
- **Distributed processing (Hadoop HDFS, EMR):** split huge datasets across a cluster (HDFS block storage + MapReduce-style processing); EMR = managed Hadoop/Spark — parallel batch analytics on S3/cluster data.
- **Case study (Zomato):** migrated analytics clusters (Trino, Druid) from x86 EC2 to **Graviton2 (ARM)** + **Spot instances** → ~30% compute-cost cut, ~25% faster queries, ~$300K/yr saved on Trino alone. Use in answers as evidence for *right-sizing/hardware-selection cost optimisation*.

## 4.8 Reference architecture: three-tier web app on AWS (scenario answer template)

1. **VPC** with public + private subnets **across ≥2 AZs**.
2. **Web tier:** EC2 instances (or ALB targets) in private subnets behind an **Application Load Balancer** in public subnets; **Auto Scaling group** spans both AZs.
3. **App tier:** EC2/containers in private subnets; **security groups** allow only the needed ports from the specific peer SG (SG chaining: ALB→web→app→DB).
4. **DB tier:** RDS Multi-AZ in isolated private subnets; no internet route.
5. **AMIs** baked to corporate standard (golden image) for consistency; **cloud-init user-data** for bootstrapping.
6. **IAM roles** per tier (web role ≠ DB-backup role); MFA for admins; CloudWatch monitoring + alarms.
7. **Storage:** EBS gp3 for boot/DB local volumes; S3 for static assets/backups; EFS if shared file storage is needed.
**Justification lines to include:** AZ spread protects against one-AZ failure (course slide); SGs are stateful instance firewalls; private subnets keep DBs unreachable from the internet; Multi-AZ RDS gives automatic failover (availability, not performance).

---

# Module 5 — PaaS, SaaS, BaaS and FaaS (Serverless)

## 5.1 PaaS — "you bring code, the provider runs the rest"

**NIST-in-plain-words:** the consumer deploys applications built with **provider-supported languages, libraries, services and tools**; the consumer does **not** manage network, servers, OS or storage; the consumer controls the **deployed application and possibly its hosting configuration**. Contract: only supported runtimes/versions.

**What PaaS gives beyond a bare VM (6 items):** language runtimes (patched by provider) · build & deploy (git push, CI/CD hooks, buildpacks/images) · scaling & balancing (autoscale rules, LB, health checks included) · managed data services (DB/cache/queue attached as services) · identity & secrets (platform identity, key/secret stores, env-based config) · logs/metrics/tracing built in.

**The PaaS delivery loop:** **Commit** (push to Git) → **Build** (buildpack/Dockerfile → image) → **Release** (config + build = release; slots/rollbacks) → **Run** (platform places instances, routes traffic) → **Scale & heal** (rules add/remove instances; failed ones replaced).
Platform traits: multi-tenant platform, elastic by rule, self-healing, opinionated runtimes, plan-based billing.

**Why teams choose PaaS:** faster time-to-market (no servers before first deploy); platform patching is provider's job; consistent environments (dev/test/prod same platform); shared pipelines; scaling as a setting, not a project; no hardware CapEx.

**What PaaS costs you (6):** limited control (no kernel tuning, fixed runtimes) · forced upgrades (runtimes hit end-of-support) · scale/cold-start limits (plan tiers cap instances) · cost at scale (convenience premium over raw VMs at steady high load) · harder debugging (no OS access; only platform logs/traces) · **lock-in** (proprietary services/config raise exit price). Also weigh data residency and platform-lifecycle risk (platforms retire — e.g., Heroku free plans ended 2022; Azure Cloud Services classic retired 2024; AWS App Runner closed to new customers Apr 2026).

**When PaaS fits / does not:**
- Good fit: web apps & APIs on standard stacks; small teams without a platform engineer; variable/unpredictable traffic; dev/test/staging; internal line-of-business apps.
- Poor fit: custom kernels/drivers/licensed OS images; specialised networking or GPU/HPC; steady very-high load (VM cost wins); hard portability requirements without containers; workloads outside supported runtimes.

**Designing for PaaS (good habits = 12-factor overlap):** config in environment (not baked into build); **stateless processes** (sessions in a backing store — otherwise a scaled-to-3-instances app misbehaves because sessions live in one process's memory); backing services as attached resources (swap DB by URL, not code); disposable instances; logs as event streams (stdout); dev–prod parity; package as containers to keep exit options.

**PaaS vs CaaS vs IaaS (what you deploy decides):** IaaS → deploy a **VM image** (you patch guest OS, runtime, scaling); CaaS → deploy a **container image** (managed orchestrator; you manage images + cluster config); PaaS → deploy **code or container image** (platform runs the rest). Boundaries blur (Fargate/Autopilot; Cloud Run behaves like serverless containers) — **ask what you deploy and what you patch, not what the vendor calls it.**

**Managed ≠ unmanaged by you (managed PostgreSQL example):** provider operates hardware/host OS, engine patching, backup mechanism, basic health; **you still own** schema/queries/indexes, who connects from where, backup retention + restore *tests*, encryption/key choices, sizing and the bill.

## 5.2 SaaS — vendor runs the app, you use it

**Definition:** consumer uses the provider's application on cloud infrastructure via thin client (browser) or API; manages nothing but limited user-level configuration.

**Installed software vs SaaS (comparison):** who runs it (you vs vendor) · cost shape (licence+hardware upfront + maintenance vs subscription per seat/usage, low entry) · upgrades (you schedule, large/infrequent vs vendor frequent, can't defer) · customisation (deep/code vs configuration+APIs only) · integration (custom behind firewall vs REST APIs/webhooks/connectors) · security duties (yours end-to-end vs shared: vendor secures service, you manage users/access/data policy) · time to value (months vs days-weeks).

**Inside a SaaS product:** one codebase, many tenants. Shared application tier resolves the tenant on every request; behaviour driven by **tenant metadata (configuration, not code)**; data tier uses **tenant-ID columns + row-level security**, or one database per tenant; control plane handles onboarding, billing, metering.

**Tenancy models (silo / pool / bridge — asked as 3-marker):**
- **Silo** — dedicated stack per tenant: strongest isolation, per-tenant tuning/compliance; highest cost, many things to patch.
- **Pool** — shared stack separated by tenant ID: lowest cost per tenant, one upgrade path; noisy-neighbour risk, isolation depends on code.
- **Bridge** — mix: e.g., shared app tier, tiered data (premium tenants get dedicated DB); flexible but more moving parts.

**Benefits:** customers — low entry cost, no servers, any-device access, frequent improvements, fast start; vendors — recurring revenue (MRR/ARR), one codebase, fast usage feedback, larger market.

**Limits/concerns (due-diligence list):** limited customisation; vendor-set upgrade timing; availability is *borrowed* (vendor outage = your outage; SLA sets remedy, usually **service credits, not business-loss compensation**); integration limits (API rate limits); exit & portability (export formats, retention after termination); cost growth (seat true-ups, add-ons).
**Vendor questions (evidence to request):** security → tenant isolation, SSO/MFA (ask SOC 2 / ISO 27001 reports); compliance/residency → where data stored/processed (region list, DPA, sub-processors); availability → SLA text, credits, status history; exit → export formats, deletion terms; integration → APIs, SAML/OIDC, SCIM; vendor risk → ownership, financial health.
(India note: DPDP Act 2023 + Rules 2025 phase in through ~May 2027 — check obligations as a data fiduciary.)

**Where SaaS fits / does not:** strong fit — standard business processes (CRM, HR, finance, collaboration); conditional — productivity/design tools (Google Docs, Figma); poor fit — embedded firmware, offline-first/ultra-low-latency, strict sovereignty, highly bespoke processes. **Not SaaS:** AWS/Azure/GCP platforms (IaaS/PaaS) even though they sell SaaS products; databases/OSes consumed as managed services.

## 5.3 Serverless: BaaS and FaaS

**"Serverless" meaning:** no servers to manage; scales with demand **including to zero**; pay for use, not idle. Servers still exist — you never see them. **Two faces:**
- **BaaS (Backend as a Service):** ready-made backend behind an SDK — auth, database, file storage, push notifications, analytics. Examples: Firebase, Supabase, AWS Amplify. Strengths: fastest route to a working app; no backend to operate; scales with vendor platform. Watch for: lock-in to vendor data model/APIs; easily misconfigured security rules; limited custom logic; cost at scale.
- **FaaS (Function as a Service):** your function runs **on an event** (HTTP via API gateway, file upload, queue message, timer), stateless, starts-runs-stops, billed per invocation + run time. Examples: AWS Lambda, Azure Functions, Google Cloud Run functions.

**FaaS constraints to design around (6):**
1. **Stateless & short-lived** — keep state outside; AWS Lambda caps one run at **15 minutes**.
2. **Cold starts** — first call after idle waits for environment start.
3. **Concurrency limits** — account/function caps.
4. **Billing by use** — cost tracks traffic (good for spiky, dangerous for steady high volume).
5. **Event-source lock-in** — triggers/permissions are vendor-specific.
6. **Harder debugging** — distributed, short-lived runs need tracing + structured logs.
Under the hood: Lambda uses **Firecracker microVMs** — VM-style isolation with container-like startup (the VM-vs-container trade-off pushed to the limit).

**FaaS vs PaaS (one line):** a PaaS app is an **always-on process you size**; a function exists **only while handling an event**.

## 5.4 The five models side by side (memorise this grid)

| Model | You deploy | You manage | Vendor manages | Scaling | Billing | Examples | Main risk |
|---|---|---|---|---|---|---|---|
| **IaaS** | VM image | OS, runtime, app, data, scaling setup | Hardware, virtualization | You configure | Per VM-hour/storage/network | EC2, Azure VMs | Misconfiguration, unpatched systems |
| **PaaS** | Code / container image | App, settings, plan size, data | OS, runtime, scaling | Platform rules you set | Plan / instance-hours | App Service, Beanstalk, App Engine, Cloud Run | Deprecation, plan limits |
| **FaaS** | Function code | Function + triggers | Everything runtime | Automatic per event, to zero | Per invocation + run time | Lambda, Azure Functions | Cold starts, limits, cost at scale |
| **BaaS** | Client app + rules | Rules, client code | Whole backend | Vendor-managed | Tiers/usage | Firebase, Supabase, Amplify | Lock-in, misconfigured access |
| **SaaS** | Nothing — configure | Users, settings, data policy | Entire app | Vendor-managed | Subscription per seat/usage | Salesforce, Microsoft 365 | Vendor dependence, exit, compliance |

Control: **IaaS highest → SaaS lowest**. Operations effort follows the reverse. Portability: IaaS best (standard VMs/OS), PaaS medium (containers help), FaaS/BaaS low (vendor triggers/APIs), SaaS lowest (data export only).

**Decision path (memorise the flow):** Need custom code at all? **No → SaaS.** Need OS/kernel/hardware control? **Yes → IaaS.** Event-driven, short, spiky? **Yes → FaaS.** Need standard backend (auth/data/files) without building? **Yes → BaaS.** Otherwise → **PaaS.** Real systems **mix models**.

**Worked application (retail app, five components):** storefront web+API → **PaaS** (standard stack, variable traffic, small team); image resize on upload → **FaaS** (event-driven, short, spiky); mobile login/wishlist prototype → **BaaS** (no server code wanted); support desk/CRM → **SaaS** (standard business process); legacy inventory tied to one OS image → **IaaS** (needs OS-level control).

---

# Module 6 — VM Provisioning, Migration, IaC and Governance

## 6.1 VM provisioning — the process (course's 7 steps; reproduce in order)

**Definition:** server provisioning = defining the server's configuration to meet requirements — hardware/software components (CPU, RAM, storage, network, OS, applications).

**Ways to provision VMs:** manual OS install · launch from a machine image (AMI) configured at boot with user-data/cloud-init · pre-configured VM template · clone an existing VM · import from another platform (VHD/VMDK/OVA) · **P2V** (physical-to-virtual conversion).

**Seven steps (AWS-flavoured but generic):**
1. **Validate** — identity & permissions (IAM), quota, parameters checked.
2. **Place** — scheduler picks a host with free capacity in the chosen AZ.
3. **Image** — root disk created from the AMI's snapshot (blocks loaded lazily).
4. **Network** — ENI attached, private IP from subnet, security groups applied.
5. **Boot** — hypervisor starts the VM (state: pending).
6. **Contextualise** — **cloud-init** reads metadata + user-data: sets hostname, SSH keys, installs software.
7. **Health** — status checks pass → running; app health check passes → registered with the load balancer.

**cloud-init / user-data (3 precise facts — trap questions):**
- cloud-init runs **inside** the VM at first boot (not "AWS runs your script").
- User-data runs **once, at first boot by default** — a reboot or stop/start does NOT re-run it unless reconfigured.
- An AMI saves **the disk only** — not the running memory state.

**Templates and image-based provisioning (the "50 identical servers" answer):** after building one correct VM, save it as a template (AWS AMI, VMware template, OpenStack Glance image). Provisioning from a template: cuts creation time; guarantees identical configuration (corporate standard baked in); reduces drift ("harder for individual projects to launch non-conforming instances"); enables scaling (Auto Scaling launches from the AMI); supports rollback (previous AMI version). Two kinds of "template": **image templates** (machine state) and **declarative infrastructure templates** (CloudFormation — describes the whole stack).

**Vagrant (context):** HashiCorp CLI tool to create reproducible dev VMs from a text file; faded as Docker/dev containers took over; still used in teaching labs and config-management testing.

## 6.2 VM lifecycle & instance states

**Generic lifecycle:** request submitted to IT → requirements matched against the resource pool → VM provisioned & started → serves per SLA → when no longer needed, **released and resources returned to the pool**.
**EC2 instance states:** pending → running (serving) → stopping → stopped (disks kept; no compute charge; start → pending, public IP may change) → shutting-down → terminated (final; compute + root volume released). Plus **reboot** (stays running) and **hibernate** (RAM saved to root volume, then stopped).

## 6.3 Governance: tags, quotas, guardrails (the "manage 50+ servers" answer)

- **Tags** — key/value labels: Owner, Project, Environment, CostCenter (+ Expiry). Why: find the owner when something breaks; **cost allocation**; automation (stop dev servers at night via tag); access rules. Rules: agree a small **mandatory** set, one spelling (Owner, not owner/Own), **apply at creation** (stack tags/launch templates), audit regularly. Enforcement: organisation policies / IAM conditions **refusing to create untagged resources**. Limits: tags are **not** a security boundary; child resources don't always inherit tags.
- **Quotas** — limits on how much of a service an account/Region may use (e.g., vCPU limit for On-Demand); protect customer and provider; raisable on request.
- **Budgets & alerts** — spend thresholds with notifications (50%/80%/100%).
- **Guardrails** — organisation-level policies (AWS **Service Control Policies**) defining what is *never* allowed regardless of IAM permissions (e.g., restrict Region to ap-south-1, instance types to t3a.nano–medium).
- **Governance loop:** define policy → enforce → monitor → report → adjust.

## 6.4 VM migration (very examinable — definitions + mechanism + numbers)

**Definition:** moving a VM from one host (or storage location) to another. Possible because **all key machine components — CPU, disks, network, memory — are virtualized**, so the VM's entire state (disks, config, and while running: memory + device state) can be moved.

**Why VMs migrate:** host maintenance or failing hardware (move VMs away first — live); load balancing & consolidation (spread hot hosts, pack idle ones and power them off); disaster recovery / DC exit (bulk, usually cold); moving to cloud/between clouds (image/server-migration tools); cost/perf tuning (change instance type or storage class).

**Types:**
- **Cold migration** — moving a **powered-off** VM. No shared storage needed; no CPU compatibility checks; VM is off for the whole copy. Reach: can **cross hypervisors and clouds** (convert the disk image). Configuration files (incl. NVRAM/BIOS settings), logs and disks move; VM re-registered on the destination; old copy deleted.
- **Live (hot) migration** — moving a **running** VM. User-visible **downtime typically tens to hundreds of milliseconds**; whole migration seconds to minutes. Normally assumes **shared storage** (else shared-nothing variant); CPU compatibility checks apply; stays **within one platform** normally. Enables **proactive maintenance** (fix before failure) and load balancing.
- **Storage migration** — moving a **running VM's disks** to a new datastore with **no service interruption**: reads/writes go to the source disk while contents copy; then writes are **mirrored to both**; after sync, VM switches to the destination disk; source deleted.
- **Live migration with storage / shared-nothing** — no shared storage required: disks mirrored first (while running), then memory pre-copied in rounds, then a short switchover. More flexible, more complex (both storage and memory move).

**Downtime vs migration time (definitions asked):** **downtime** = interval the service is unavailable (synchronisation pause); **migration time** = from start of pre-migration to the VM running at the destination (for multiple migrations: start of first to completion of last).

**Live migration mechanism (Xen 6 stages — name + one line each):**
0. **Pre-migration** — active VM on host A.
1. **Reservation** — resources ensured on host B; VM container of the right size.
2. **Iterative pre-copy** — first pass copies **all pages**; subsequent passes copy only **pages dirtied during the previous pass** (push phase, VM keeps running).
3. **Stop-and-copy** — suspend VM at A, redirect network traffic, transfer CPU state + remaining dirty pages; consistent suspended copy now exists at both A and B (A is primary until commit).
4. **Commitment** — B confirms receipt; A acknowledges; B becomes primary; A may discard its copy.
5. **Activation** — VM resumes on B; post-migration code reattaches device drivers and advertises the moved IP.
Memory transfer phases: **Push** (while running) → **Stop-and-copy** (paused) → **Post-copy/pull** (alternative: VM runs on B immediately and pulls faulted pages from A).

**Worked example (memorise — the deck's own numbers):** M = 8 GiB, bandwidth B = 1 GiB/s, dirty rate D = 0.2 GiB/s, stop when remainder < 0.3 GiB.
- Round 0: copy 8 GiB (8 s); dirtied meanwhile = D×8 = 1.6 GiB (> 0.3 → continue).
- Round 1: copy 1.6 GiB (1.6 s); dirtied 0.32 GiB (> 0.3 → continue).
- Round 2: copy 0.32 GiB (0.32 s); dirtied 0.064 GiB (< 0.3 → stop).
- **Stop-and-copy 0.064 GiB = 64 ms = the downtime.** Total ≈ 8 + 1.6 + 0.32 + 0.064 ≈ **10 s**.
- Each round shrinks by factor D/B = 0.2; **pre-copy converges only if the dirty rate < bandwidth** (else it never catches up). Challenges: VMs hold lots of state in memory; some VMs have soft real-time requirements.

**Live migration is a hypervisor/platform capability** — not every cloud exposes it to customers.

## 6.5 The 6 R's of migration (application-move options)

**Rehost** (lift-and-shift: same stack, re-deploy on cloud VMs — fastest, least optimised) · **Replatform** (minor changes: managed DB instead of self-run) · **Repurchase** (drop to SaaS) · **Refactor** (re-architect for cloud-native: microservices, serverless — highest gain, highest effort) · **Retire** (decommission) · **Retain** (keep on-prem for now — compliance/latency). For each application, pick per business value, effort, risk.

## 6.6 Infrastructure as Code (IaC)

**Why:** console-clicking is slow, unrepeatable, undocumented — two manually built environments are never identical (**"snowflakes"**). **IaC = describing infrastructure in text files that a tool turns into real resources.** Files live in Git: review, history, rollback, automated tests. Same files rebuild dev/test/prod in minutes; teardown is one command. Prevents "it worked in test" (different manual steps) and "who changed this?" (no audit trail).

**Declarative IaC (the norm):** state **what** you want (one web server, tagged); the tool works out how. Imperative = scripting the sequence of API calls. Cycle: **write → plan/preview (change set) → apply → state recorded**.
Key properties (definitions asked):
- **Idempotent:** applying the same file twice changes nothing (second apply is a no-op).
- **Drift:** real resources changed **outside the code** (e.g., a console edit). Detect (plan shows diffs) and reconcile (re-apply code or import the change).
- **State:** the tool's record of what it created. CloudFormation manages state for you; **Terraform/OpenTofu keep a state file you must protect** (it can contain secrets; losing it orphans resources).
**Tools:** Terraform/OpenTofu (multi-cloud), CloudFormation (AWS), Bicep→ARM templates (Azure), Pulumi. Analogy: declarative = thermostat (desired state, self-correcting) vs imperative = TV remote (command sequences).

**IaC for migration consistency (scenario line):** after migration, encode the target infrastructure as code → identical reproduction across environments, reviewed changes, drift detection, and disaster-recovery rebuilds from Git.

## 6.7 Manual provisioning failures → IaC scenario (exam favourite)

Console-built environments cause: inconsistent configs dev vs prod; no record of what exists; unrepeatable recovery; no review before change. Fix: IaC (Terraform/CloudFormation) + Git (version control, PR review) + **tagging policy** (ownership/cost) + **quotas/SCPs** (guardrails) + **drift detection** in CI + **idempotent applies** in pipelines.

---

# Module 7 — Capacity Management, VIM and Workload Scheduling

## 7.1 Capacity management — why and what

**Definition:** ensuring **enough resources, at the right time and cost, to meet demand and SLAs — and no more than needed.** Strategic and proactive; manages service quality and cost; aligns business and IT.

**Why hard in cloud:** demand is uncertain and bursty while provisioning takes time; large distributed environments have no complete current view of state; multi-tenancy means one tenant's peak disturbs another (noisy neighbour); **SLAs turn a shortage into penalties and a surplus into cost.**

**The two failure modes (classic table):**
| Too little (under-capacity) | Too much (over-capacity) |
|---|---|
| Application sizing not performed; controlled by limits | Unlimited access to resources; service levels unaffected |
| Service levels affected → loss of service, SLA breach penalties | Higher cost (resource usage, licences); wasted resources; VM sprawl; pressure on management; impacts other services |

Gartner (course quote): most organizations overprovision infrastructure by at least 100%; a large share of cloud spend goes on idle or oversized resources.

**The right balance:** application sizing performed; efficient IaaS use; acceptable service levels with **continuous review and adjustment**; controlled by shares/limits/reservations; continuous monitoring and tuning (config adjustments, VM consolidation, power off unused hosts).

**Old habits to avoid:** infrastructure/application hugging (silo mentality); lack of resource sharing; **"just in case" capacity planning → overprovisioning**; no chargeback; VM sprawl; unclear ownership.

## 7.2 The 5-step capacity planning method (numerical — memorise the worked example)

1. **Measure** — CPU, memory, disk, network at **p95/p99, not averages**.
2. **Characterise** the workload — steady, daily cycle, bursty, seasonal.
3. **Forecast** — trend + seasonality + known events (sale, release).
4. **Add headroom** — growth buffer **plus N+1 spare capacity** for a host failure.
5. **Decide how to buy** — on-demand, reserved, spot, capacity reservation mix.

**Course worked example (reproduce exactly):** retailer's private cluster, hosts of 48 vCPU each; busy-hour cluster CPU demand p95 = 120 vCPU; 15% growth next quarter.
- Measure: p95 = **120 vCPU**.
- Forecast: 120 × 1.15 = **138 vCPU**.
- Headroom: 138 × 1.25 = 172.5 → round up **173 vCPU**.
- Convert to hosts: 173 ÷ 48 = 3.6 → **4 hosts**.
- Add spare (N+1): **5 hosts**.
Formula: hosts = ⌈ p95 × (1+growth) × (1+headroom) ÷ vCPU-per-host ⌉ + 1 spare.

## 7.3 Scaling: vertical vs horizontal (guaranteed comparison)

| | **Vertical (scale up)** | **Horizontal (scale out)** |
|---|---|---|
| Action | Make one VM/pod **bigger** (more vCPU/RAM) | **Add more copies** behind a load balancer |
| Pros | Simple, no application change, quick relief | No hard ceiling; tolerates instance failure; fine-grained cost |
| Cons | A **ceiling** (largest instance size); often needs restart; **still one point of failure**; downtime during resize | Needs **stateless design** (or shared/partitioned state), load balancing, coordination overhead (Amdahl's law) |
| K8s form | VPA / in-place resize of requests | **HPA changes the replica count** |
| Rule | **Scale up for quick relief; design to scale out.** | |

**Autoscaling (how it responds to workload):** monitor a metric (CPU, request rate, queue depth) → compare with target → add/remove replicas automatically within **min/max bounds**; **cooldown periods** prevent flapping; can combine **schedule-based** (known events) with **metric-based** scaling. Elasticity ⇒ pay only for what is used.

**Can autoscaling eliminate capacity planning? No (justify):** autoscaling reacts to *measured* load — it needs correct min/max bounds, correct trigger thresholds, instance types sized for the workload, and lead-time headroom (VM boot time, scale-out lag during sudden spikes). Planning still decides the baseline (steady-load purchase model), the ceilings, the budget guard, and multi-AZ fault tolerance; misconfigured autoscaling causes cost blowouts or SLA breaches. Autoscaling *executes* the capacity plan; it does not *replace* it.

## 7.4 Load balancing modes (from SLO management)

- **Load balancing:** distribute incoming requests across machines hosting replicas so load is spread evenly; front-end node faces clients, back-end nodes serve.
- **Class-agnostic LB:** front end ignores who the client is and what service is requested — simple, uniform distribution.
- **Class-aware LB:** front end inspects client type / request category and routes accordingly (e.g., premium tenants to dedicated nodes).
- **Static vs dynamic (historical EC3 favourite):** static = fixed pre-computed distribution (round-robin, weighted); ignores current load — simple but unresponsive to hot spots. Dynamic = decisions use live state (least connections, response time, queue depth) — adapts, costs more monitoring. Combine: dynamic with autoscaling.
- **Admission control (under overload):** protect the server by rejecting some work. **Request-based:** reject new requests at capacity (breaks multi-request sessions). **Session-based:** complete running sessions, reject only *new* sessions (better user experience).

## 7.5 Virtual Infrastructure Manager (VIM)

**Definition:** the **control plane for a pool of hypervisor hosts** — it decides where VMs run and manages their lifecycle; allocation happens **based on current requirements rather than statically**.
**Tasks:** placement & scheduling; VM lifecycle (create/start/stop/**migrate**/delete); image management; network & IP management; storage; quotas & multi-tenancy; monitoring & events.
**Why needed:** VMs alone don't answer "where is/should my VM run?", "how do I clone a node?", "which IPs are free?" A VIM provides a **uniform view of the resource pool**, integrates image/network/virtualization management, and enables **dynamic deployment and re-placement** — turning rigid physical infrastructure into flexible virtual infrastructure.
**Roles:** backend of public clouds (internal management); private-cloud virtualization for internal users; cloud interoperation (hybrid, on-demand access to public clouds).
**Examples:** OpenStack Nova, VMware vCenter, OpenNebula, Proxmox VE; **Kubernetes plays the same role for containers**.
**Scheduling evolution:** early VM schedulers placed each VM once with a greedy rule; modern platforms **re-balance continuously** (VMware DRS, OpenStack Nova filter/weigher + Watcher, K8s descheduler). VIMs must support suspend/resume/migrate to enable this.
**OpenNebula (course case):** open-source cloud middleware managing heterogeneous data centres; private/public/hybrid; supports Xen/KVM/VMware/LXC/Firecracker; driver-based architecture decouples core from hypervisor (adding a platform = writing a driver); image management (≈ AMI catalogue/Glance); virtual network manager with MAC/IP tracking (≈ VPC/Neutron, IPAM); scheduler with requirement/rank matchmaker; **Haizea** adds advance-reservation leasing. VM lifecycle stages: resource selection → resource preparation (disk transfer + contextualization) → creation → migration → termination.
**Choosing a VIM:** OpenStack (large scale, high ops effort), OpenNebula (simpler; VMs + LXC + edge), CloudStack (single management server), VMware vCenter (commercial, licensing changed post-Broadcom 2023), Proxmox VE (KVM+LXC, small/medium), K8s+KubeVirt (containers first, VMs added). Compare on scale, ops effort, licence/cost, ecosystem/skills.

## 7.6 Workloads, leases, jobs (vocabulary that appears in numericals)

- **Workload:** a specific application/service/capability (and its resource demand) run on cloud resources — VMs, databases, containers, Hadoop nodes, marketing/e-commerce sites, analytics platforms.
- **Classify by:** resource profile (CPU-bound, memory-bound, I/O-bound, GPU-bound); arrival pattern (steady, daily cycle, bursty, batch); service objective (latency-sensitive web/API vs throughput-oriented batch); GPU workloads (training = long/large/batch vs inference = latency-sensitive).
- **Lease:** the contract between provider and user for using resources to run a workload (start time, duration, pre-emptable or not).
- **Application → Job → Task:** application = business process in code; job = work-breakdown part of an application; tasks = steps within a job (sequential or parallel).
- **Resource allocation vs task scheduling:** RA = identifying/allocating resources for a workload type; TS = shuffling/managing tasks to operate efficiently on available resources.

## 7.7 Placement and bin-packing (numerical asked in papers)

Finding the optimal placement is computationally hard → schedulers use **heuristics: first-fit, best-fit, worst-fit**. Items = VMs/pods with sizes; bins = hosts with capacities; goal = fewest hosts (cost/power) **or** best spread (resilience/performance).

**Course example (memorise):** hosts of capacity 10; requests 4, 5, 6, 3, 2 (total 20).
- **First-fit in arrival order:** H1 = 4+5, H2 = 6+3, H3 = 2 → **3 hosts**.
- **First-fit decreasing** (sort 6,5,4,3,2): H1 = 6+4, H2 = 5+3+2 → **2 hosts**.
Lesson: **order and policy change the result**; tight packing saves hosts but **increases the blast radius of one host failing** (spread vs pack trade-off).

**VM vs pod for the scheduler:** VM = fixed size, minutes to start, heavy to move, strong isolation; pod = millicores/MiB, seconds to start, cheap to reschedule, resizable, finer packing but weaker isolation (shared kernel). Schedulers act on **declared requests**, not live usage — wrong declarations waste capacity or cause contention.

## 7.8 Amdahl's law (the parallelism cap — formula asked)

**Smax = 1 ÷ [ (1 − p) + p/s ]** — Smax = max overall speedup; p = fraction of work improved (0–1); s = speed-up factor of that fraction.
Worked example: p = 0.9, s = 10 → 1 ÷ (0.1 + 0.09) = **≈5.26×**. Even with s → ∞, cap = 1 ÷ (1−p) = **10×**. Meaning: the serial fraction dominates; adding processors stops helping. Relevance: horizontal scaling of a service with serial dependencies (single coordinator, shared DB lock) hits this ceiling — a justification mark in "why not scale out infinitely".

## 7.9 Provisioning models and reservation-based capacity

- **Immediate provisioning** — used by commercial public clouds (capacity looks infinite); served if capacity exists.
- **Best-effort provisioning** — requests queued and prioritised (private clouds with finite capacity).
- **Advance reservation** — resources pre-reserved, **guaranteed available at a specific time** (e.g., equipment-driven experiment 2–4 pm; GPU Capacity Blocks). Problem: reservations **hurt utilisation** (capacity idles waiting; others must be vacated) and are billed **whether or not used** (AWS Capacity Reservation). Mitigations: preemption of lower-priority work (**Spot** = preemptable with 2-minute notice; K8s PriorityClass preemption), **checkpointing** (save state; resume later — OS-level CRIU or app-level), migration of checkpointed jobs, Haizea lease scheduling in OpenNebula.
- **AWS reservation models (name + property):** On-Demand (best effort) · **Spot** (discounted spare capacity, interruptible, 2-min warning) · **On-Demand Capacity Reservation** (capacity guaranteed in one AZ; pay whether used or not) · **Capacity Blocks for ML** (GPU capacity booked for a future date — true advance reservation) · **Savings Plans / Reserved Instances** — *price* commitment only; **do NOT reserve capacity** (classic trap).

## 7.10 SLA-aware and cost-aware scheduling (triad: capacity–SLA–cost)

- **SLI/SLO/SLA vocabulary:** **SLI** = what you measure (latency, availability); **SLO** = the target for the SLI; **SLA** = the contract with consequences (violations → credits).
- **Capacity–SLA relationship:** infrastructure SLAs (provider↔service owner) formalise capacity availability; to protect high-level SLAs (owner↔end users), the provider must offer elasticity on demand; scaling decisions are best driven by **application-level metrics** (app knows its bottlenecks).
- **Capacity planning + continuous optimization are the two complementary IaaS provider tasks:** plan so contracted SLAs can be met; continuously optimise utilisation of existing capacity.
- **Rightsizing (method):** measure p95 CPU & memory over ≥ 2 weeks including peak days → compare with current request/instance size → set new request ≈ **p95 + 20–30% headroom** (be more conservative with memory — OOM risk) → roll out gradually watching SLOs → repeat. Tools: CloudWatch, `kubectl top`, VPA recommendations, AWS Compute Optimizer.
- **Cost-aware levers:** rightsizing; scale-in when demand falls; **bin-packing/consolidation** (fewer, fuller nodes); schedule dev/test off-hours; Arm/Graviton instances where supported; correct purchase model mix.
- **Waste sources:** oversized requests, idle nodes, always-on dev environments, wrong purchase model.
- **The decision triad (memorise):** steady base → commit (Savings Plan/reserved); spiky unpredictable → autoscale; interruptible batch → **Spot with checkpointing**; must-not-miss launch time → capacity reservation. Whatever chosen: honest requests/limits, measure, review monthly.
- **What if cost is minimised ignoring SLA?** Requests get starved/throttled under peak; no headroom for spikes or host failure; SLO breaches → SLA credits/penalties, lost customers; consolidation packing amplifies one-host-failure impact. Conversely SLA-first with no cost view → chronic overprovisioning. The equilibrium is the deliverable.

## 7.11 Kubernetes scheduling numericals (recap of method)

- allocatable = capacity − system reserved; 1 CPU = 1000 m.
- Pods per node = min( floor(CPU_alloc ÷ cpu_request), floor(mem_alloc ÷ mem_request), maxPods ).
- Requests reserve; limits cap (CPU throttles; memory OOM-kills).
- Poor allocation → **resource contention**: requests set too low → scheduler overpacks nodes → real usage exceeds capacity → CPU throttling, memory pressure, OOM kills, noisy neighbours, SLA breaches; too high → wasted reservations, Pending pods, higher cost.

---

# Module 8 — Multi-Tenancy, Availability, Security, SLA and Compliance

## 8.1 Multi-tenancy — definition and models

**Definition (reproduce):** an architectural pattern in which a **single instance of a software application serves multiple customers (tenants)** on the provider's infrastructure; tenants may customise some parts (UI colour, business rules) but **cannot customise the application's code**. The application is *redesigned* to handle resource sharing between tenants. Example: Salesforce hosts one CRM instance; companies are tenants; their employees are users.

**Contrasts (trap-prone):**
- **Multi-tenancy ≠ multi-cloud** (tenants share one provider's stack; multi-cloud = one customer, many providers).
- **Multi-tenancy vs multi-instance:** multi-instance = *separate software instances* per tenant (isolation by copies); multi-tenant = one shared instance (isolation inside the app).
- **Multi-tenancy vs multi-user:** multi-user = many people using one app configured for one organisation; multi-tenancy = many *organisations* on one instance, each customised.
- In virtualization the user gets the *illusion of owning the whole infrastructure* (VM); the **hypervisor** achieves separation between users.

**Four levels of multi-tenancy (resource-sharing spectrum):**
1. **Custom instances** — each customer has their own custom version; hardest to manage; dedicated support per customer.
2. **Configurable instances** — same version, separate instances, per-tenant configuration (logos, workflows); one codebase to manage; simple upgrades.
3. **Configurable, multi-tenant efficient instances** — **one shared instance** for all customers; most efficient resource use; simplest management.
4. **Scalable, configurable, multi-tenant efficient resources** — level 3 on a **cluster**, scaling with demand; maximum sharing. Example: Gmail.

**Silo / pool / bridge (SaaS tenancy patterns):** silo = dedicated stack per tenant (strong isolation, highest cost); pool = shared stack with tenant ID separation (cheapest, noisy-neighbour risk); bridge = mix (shared app, tiered data for premium tenants).

**MT models per service layer:**
- **IaaS MT models:** VM isolation (each tenant gets isolated VMs — strong security, less efficient) · network isolation (tenant's own virtual network — better performance, complex network management) · resource-pool isolation (dedicated CPU/memory/storage sets — good allocation, weaker isolation) · hybrid combination.
- **PaaS MT models:** application-level isolation (containerization per tenant) · tenant-level isolation (dedicated platform instance) · resource-pool isolation · hybrid.
- **SaaS MT models:** **database-level isolation** (separate schema per tenant — complete isolation, complex DB management) · **application-level isolation** (data partitioned + config separated in app — flexible, less secure) · hybrid.

**Data sharing in a shared database (asked as design question):**
- **Dedicated tables per tenant** — own copy of tables; privileges restricted; easy customisation (own schema); space-inefficient.
- **Shared table + tenant-ID rows** — every row carries a tenant ID; isolate rows by ID (row-level security); space-efficient; needs views/joins and a metadata table; customisation hard.
- **Customisation support in shared tables:** **pre-allocated columns** (reserve N custom columns — wasted space if unused, restrictive if exceeded) · **name-value (pivot) table** (metadata table + pointer column; custom fields stored as rows with datatype metadata — space-efficient, joins to fetch) · **XML column** (last column stores XML — flexible, querying harder).

**Resource sharing at app-server level (4 architectures):** fully isolated application server per tenant → virtualized application server (VM per tenant) → shared virtual server → shared application server (session/thread separation).

**Benefits of MT:** cost savings (one instance's overhead amortised across tenants); **data aggregation** (single schema → cross-tenant queries, mining, trends); **release management** (one install on one server vs packages to every machine).
**Disadvantages:** development complexity (customisation + per-tenant metadata); **release risk** (one update can downtime *all* tenants; bugs affect everyone; update windows constrained by tenants' schedules).

**Isolation challenges & controls (scenario marks):** challenges — performance isolation (tenant A's burst steals resources from tenant B), security isolation (one tenant must never access another's data), noisy neighbours. Controls — virtualization/containers for isolation; **conserving vs non-conserving resource allocation** (conserving: a VM may NOT take spare unused capacity beyond quota; non-conserving: it may use spare capacity, else migrate); quotas/limits; per-tenant rate limits; authentication (centralized user database administered per-tenant, or decentralized federated SSO with trust relationship — centralised is more generally applicable given self-service clouds can't vet every tenant's auth infrastructure); encryption per tenant; audit logging.

## 8.2 Availability and fault tolerance (banking-style scenario)

- **Availability** = proportion of time the service is usable (measured, e.g., 99.95%).
- **Fault tolerance** = the system continues operating correctly despite component failure (redundant components + automatic failover).
- **High availability must conform to the SLA** — the promised quality defines the required architecture. What deters HA: unplanned outages (server/human error), planned maintenance windows, performance degradation under peak load (infrastructure shortage or critical-component failure). **Bottom line: effective capacity planning is required to meet promised QoS/HA.**

**Steps for HA (course's four — memorise):**
1. **Build for server failure** — redundant servers that can be brought online.
2. **Build for zone failure** — DR sites / multi-AZ deployment to switch over.
3. **Build for cloud failure** — robust, contained setup; errors must not cascade ("fire door" policy; consider multi-region/multi-provider).
4. **Automate and test** — test & test again; minimal manual interference (failover drills).

**Scenario skeleton (banking app surviving an AZ failure):** deploy identical stacks in ≥2 AZs; load balancer health-checks and removes failed-zone targets; database = RDS Multi-AZ (synchronous standby, automatic failover); stateless web/app tiers auto-scale; redundant data + backups cross-region; automated failover tested regularly. **Why replicas in a single AZ are not enough:** an AZ is a failure domain — its data centres share power/network/seismic/flood risk; one AZ event (grid failure, flood, networking) can take down *all* replicas in it simultaneously; zone-level redundancy is the minimum to survive that. (Also true: multiple replicas still fail together on a bad application update — hence versioned rollouts.)

**Redundancy, failover, load balancing roles:** redundancy = duplicated components (no single point of failure); failover = automatic switch to the standby on failure detection; load balancing = spreads load and isolates unhealthy targets (health checks) — together they convert component failure from outage into brief degradation.

## 8.3 Security: shared responsibility, threats, mitigations

**Shared responsibility model:** the provider secures **the cloud itself** (physical data centres, network, hardware, virtualization layer, managed-service internals); the customer secures **what they put on it** (identity/access, application configuration, data encryption & keys, patching, compliance). The split **moves with the service model** (IaaS: customer owns OS-up; SaaS: customer owns data/access only) — but **accountability for data, identity and access never transfers**.
Course phrasing: cloud security = protecting cloud applications, data, infrastructure, services via technologies + policies + operational controls; modern emphasis: Zero Trust, encryption, identity management, continuous monitoring, compliance (CIA triad).

**Typical threats (scenario asks to identify):**
- **Compromised credentials** — leaked keys/passwords; no MFA; long-lived access keys on machines.
- **Misconfigured storage** — public S3 buckets; permissive security groups; disabled logging.
- **Unauthorized access** — over-privileged IAM (admin for everyone); missing network segmentation; unpatched instances.
- (SaaS-specific) data leakage between tenants, vendor compromise, shadow SaaS.

**Mitigations (map 1:1):** MFA everywhere + short-lived **roles** instead of static keys; least-privilege IAM policies + periodic access reviews (remove unused permissions); encryption at rest/in transit + customer-managed keys; **private subnets + SG/NACL** segmentation; automated configuration checks (IaC review, policy-as-code, SCP guardrails); CloudWatch/audit logging + alerts; patch management; tested backups.

**"Cloud is not automatically secure or insecure" (2-mark justification):** security depends on the *joint* implementation — the provider hardens infrastructure, but a customer with no MFA, public buckets and admin-everyone IAM is insecure *on* secure infrastructure; conversely a well-configured tenant can be very secure. The shared responsibility model makes security a *split* obligation: moving to cloud changes **who** does what, not **whether** security work exists.

**Conserving vs non-conserving allocation (from co-hosting discussion):** in **conserving** mode a VM demanding more than its quota cannot take other VMs' unused spare resources (strict isolation); in **non-conserving** mode it may use spare capacity, or migrate to a host that can satisfy it (efficiency at slight isolation cost). Co-hosting arose because dedicated hosting wasted capacity; virtualization solved the isolation problems co-hosting created.

## 8.4 SLA, SLO, SLI (definitions + types + lifecycle)

**SLA (Service Level Agreement):** a legal agreement specifying non-functional requirements of the service — what service is provided, how supported, times, locations, costs, performance, responsibilities; includes metrics, payment, and **violation penalties** (in AWS the remedy is a **service credit**, not business-loss compensation). Historical driver: enterprises outsourced hosting → needed a legal guarantee of QoS (availability of CPU, storage, network at peak).
**SLO (Service Level Objective):** the objective of service quality to be achieved — specific measurable characteristics of the SLA (availability, throughput, frequency, response time, quality). Good SLO properties: attainable, repeatable, measurable, understandable, meaningful, controllable, affordable, mutually acceptable.
**SLI (Service Level Indicator):** the actual measured value of an SLO, used as the benchmark to compare against the promised target.
Chain: **SLI (measure) → SLO (target) → SLA (contract with consequences).**

**SLA types:**
- **Infrastructure SLA** — provider guarantees infrastructure availability: hardware (99%/month), power (99.99%), DC network (99.99%); machines leased and isolated per customer.
- **Application SLA** — provider responsible for application-level SLOs in the co-hosting model: web response time ≤ 3.5 s/user request; web-server latency ≤ 0.2 s; DB latency ≤ 0.5 s/query.
- **By strictness:** **No SLA** (best-effort; assumes spare capacity always available; non-QoS-sensitive workloads) · **Probabilistic (epistemic) SLAs** — availability *percentiles* over the period; lower percentile = cheaper (suits SMBs, many enterprise apps) · **Deterministic (ontic) SLAs** — 100% availability percentile; no capacity multiplexing admitted; most stringent and costly (critical services only).

**SLA life cycle (5 phases — memorise):**
1. **Contract definition** — provider defines standard offerings + SLA templates → catalogue.
2. **Publication & discovery** — offerings advertised; customers search and shortlist.
3. **Negotiation** — terms mutually agreed & signed; automated for packaged apps; manual (with scalability/performance analysis) for customised apps.
4. **Operationalization** — **SLA monitoring** (measure parameters, compute metrics, detect deviations, notify) + **SLA accounting** (capture/archive adherence for compliance; report actual vs guaranteed) + **SLA enforcement** (penalties/credits; corrective action).
5. **De-commissioning** — terminate all SLA activities when the hosting relationship ends; terms of termination defined in the SLA.

**SLA management for cloud applications (MSP view — 5 phases):** **Feasibility** (technical: scale-out ability, platform compatibility, hardware/software needs; infrastructure: resources available for projected demand; financial: profitable for both) → **On-boarding** (move app to platform; profile runtime with profilers; package via **OVF**; baseline on physical, then virtualized; note scale-out ability and performance bounds; candidate SLAs + costs; create management policies) → **Pre-production** (simulated production; customer validates findings; sign-off) → **Production** (live under SLA; re-onboard if breached frequently or if new terms requested) → **Termination** (data returned; only legally required retention; sign-off ends relationship).

**Traditional SLO management techniques:** load balancing (class-agnostic vs class-aware) · admission control (request-based vs session-based, above) · conserving/non-conserving allocation.

**Availability numericals (memorise):** downtime = (1 − availability) × period. 30-day month = 43,200 min → 99% = 432 min (7 h 12 m); **99.9% = 43.2 min**; 99.95% = 21.6 min; **99.99% = 4.32 min**. Achieved availability = (total − downtime)/total: 20 min outage in a 30-day month → 1 − 20/43200 = 99.9537% ≈ **99.954%** (breaches a 99.99% target). Composite: two serial components 99.9% each → 0.999 × 0.999 = 99.8001%; a redundant parallel pair → 1 − (0.001)² = 99.9999%.

## 8.5 Automated, policy-based SLA management

**Why:** SLAs are monitored continuously; manual reaction is too slow; deviations must trigger *automatic* corrective action.
**How (pattern):** define policies that map measured SLIs to actions — e.g., latency SLO at risk → scale out / migrate VMs / shed load; provider policies auto-allocate/de-allocate resources per component as load rises/falls (the MSP on-boarding step creates exactly these policies); admission control protects overload; monitoring feeds accounting (compliance reports) and enforcement (credits).
**Benefit lines for answers:** continuous conformance without human latency; consistent enforcement; auditable (accounting archives); enables elasticity promises (SLA risk sits with the provider — automation is how the provider honours it).

## 8.6 Sovereign cloud, data residency, compliance

**Definitions:**
- **Data residency** — legal/geographical requirement that data be stored (and often processed) *within a specific jurisdiction*.
- **Regulatory compliance** — adherence to laws/standards governing data: e.g., GDPR-style protection, India's **DPDP Act 2023 (Rules 2025, phased to ~May 2027)**, industry regimes (banking, health), audit requirements.
- **Sovereign cloud** — cloud operated so that **data location, access control and legal jurisdiction remain under the nation/customer's control**: in-country operations, local legal entity, staff/citizenship controls, protection from foreign legal access, often run by/with a national operator.

**Why conventional public cloud may not suffice (3 marks):** data may replicate/backup outside the country; foreign CLOUD Act-style jurisdiction may expose data; provider staff (anywhere) may hold access; shared multi-tenant operations complicate audit proof; some services not available in-region; SLA remedies are credits, not compliance protection.

**Architecture + governance measures (the 4-mark recommendation set):**
1. In-country **Region / Local Zone** or **Outposts** (data stays on-premises/at named sites) — residency by architecture.
2. Strict **IAM**: in-country admin roles, MFA, least privilege, no foreign support access without approval; customer-managed **encryption keys** (BYOK) held in-country.
3. **Private subnets/no cross-border replication**; disable/limit cross-region backups; DPA with provider; sub-processor list.
4. **Governance:** data classification; audit logging + retention; compliance certifications (SOC 2/ISO 27001) verified; exit plan with exportable data; tag-based enforcement (SCPs refusing resources outside approved regions).

---

# Module 9 — Application Development, Deployment and Future Compute Paradigms

## 9.1 Development models: monolith, microservices, serverless

| | **Monolith** | **Microservices** | **Serverless functions** |
|---|---|---|---|
| Idea | One program with everything inside | Many small services, each does one job | Small functions run when an event happens |
| Deploying | Whole app as one unit | Each service deployed independently | Each function deployed independently |
| Scaling | Copy the whole app | Scale only the busy service | Platform scales functions for you |
| Good fit | Small teams, simple apps | Large apps, many teams | Uneven/event-driven work |
| Watch out | Hard to change one part safely | More moving parts to manage | Time/memory limits per function |

**Key line:** no model is best everywhere — match to workload and team.
**Case studies (use as evidence):** **Netflix** — 2008 DB corruption started the cloud move; completed January 2016; rebuilt as microservices behind the **Zuul** API gateway; still runs its own CDN (**Open Connect**) for video → a cloud move is a long journey and the architecture changes along the way. **Prime Video** — one monitoring service moved *from* microservices+serverless *to* a monolith on EC2/ECS, >90% cost reduction for that service → smaller is not always better; choose per workload.

## 9.2 The 12-factor app (cloud-ready habits — list ≥5 for marks)

Methodology for building **portable, robust, scalable SaaS apps** (origin: Heroku):
- declarative setup automation; clean contract with the OS; cloud-platform friendly; minimal dev/prod divergence; scales without changing tooling/practices.
Selected factors (course highlights): **config in the environment** (secrets outside code — one build runs anywhere) · **backing services as attached resources** (swap DB by URL) · **stateless processes** (store user data in a DB, not app memory) · **logs as event streams** (write to stdout; platform collects) · **disposability** (fast start, clean stop) · **dev/prod parity**.

## 9.3 APIs and API gateways

**API:** an agreed way for one program to ask another to do something (waiter analogy): client sends request (GET /orders/42) → server validates and does work → response (often JSON) → client presents it. Example: Amazon **S3** (2006) — programs store/fetch via web API without knowing internals.
**API gateway = one front door for many services:** clients call a single address; the gateway (1) **authenticates** (only allowed users/apps), (2) **throttles/limits** (stops one client overloading services), (3) **routes** each request to the right backend service. Value: teams change services behind the gateway without changing front-end apps. Examples: Amazon API Gateway (REST/HTTP/WebSocket, authorizers, usage plans); Netflix **Zuul**; Coca-Cola Freestyle (Lambda + API Gateway + CloudFront).
**Integration styles:** **synchronous** (phone call — caller waits; simple to debug; caller stuck if the other service is down) vs **asynchronous** (letter — message queues, e.g., **SQS**, AWS's first infra service beta 2004; absorbs spikes; result arrives later). Choose sync for immediate answers, async to decouple and buffer bursts.

## 9.4 CI/CD (scenario: manual deployments causing inconsistent environments)

**Pipeline stages:** **Commit** (developer saves change to shared code) → **Build** (compile + package) → **Test** (automated checks) → **Release** (tested package stored, ready) → **Deploy** (goes live).
**CI vs CD vs CD:** **Continuous Integration** — merge small changes often; each built and tested automatically. **Continuous Delivery** — every passing change is *ready* to release; a person decides when. **Continuous Deployment** — every passing change goes live automatically, no manual step.
**Tests (fast/cheap first):** unit (one piece) → integration (pieces together) → end-to-end (full user journey).
**Safe release strategies:** **rolling** (replace servers a few at a time) · **blue-green** (two identical environments; switch traffic; instant rollback) · **canary** (small share of users first; expand on success).
**Tooling:** Spinnaker (open-source CD from Netflix, 2015; supports blue-green/canary). **DORA four measures:** deployment frequency, lead time for changes, change failure rate, time to restore service.
**Risks without automated testing + rollback (2 marks):** broken releases reach all users at once (no blast-radius control); bad changes can't be reverted quickly → long outages; inconsistent environments (the "worked in dev" problem); human error in manual steps; slower recovery (high time-to-restore) — precisely what DORA measures expose.

## 9.5 Deployment models (four ways to deploy)

| Model | You manage | Scaling | Control | Setup effort | Good fit | Example |
|---|---|---|---|---|---|---|
| **Custom (VMs)** | OS, runtime, app, scaling setup | You build it | Highest | Highest | Special software, full control | EC2 |
| **Containers** | App + image + cluster settings | Orchestrator scales containers | High | Medium | Many services, portable apps | Kubernetes, ECS |
| **Managed platform** | App code and data | Platform scales | Medium | Low | Standard web apps | Heroku, App Service |
| **Serverless** | Function code | Automatic, per event | Lowest | Lowest | Event-driven/uneven traffic | Lambda |

Key line: from left to right — less control, less work. **Pick the least effort that still gives the control the app needs.** Lambda specifics: introduced Nov 2014; one run ≤ 15 min; memory 128 MB–10,240 MB (limits change — check docs).

## 9.6 Observability (logs, metrics, traces, alerts)

- **Logs** — written records of events ("user signed in", "payment failed") — **what happened**.
- **Metrics** — numbers measured over time (requests/sec, memory used) — **how much**.
- **Traces** — the path of one request as it travels through many services — **where** (bottleneck/failure point).
- **Alerts** — messages to people when a number crosses a safe limit.
Memory hook: **logs say what happened; metrics say how much; traces say where.**
**Four golden signals (Google SRE):** **latency, traffic, errors, saturation.**
**Tools:** OpenTelemetry (vendor-neutral CNCF standard for collecting all three), Prometheus (metrics + alerts, CNCF graduated), Jaeger (traces, from Uber, CNCF graduated).
**Why host-level monitoring is insufficient for distributed apps (2 marks):** one user request crosses many services/hosts — a host CPU chart cannot show *which* service slowed the request or where it failed; services scale up/down so hosts are ephemeral; correlation needs request IDs across services (distributed tracing); metrics must aggregate at service level (golden signals), not machine level.

**Scenario pattern (microservices slow at peak):** use traces to find the slow hop; metrics (latency/saturation of that service) confirm; logs show the error; alerting catches recurrence; fix = scale that one service (microservices advantage) or fix the query; verify via DORA/SLO dashboards.

## 9.7 Hybrid and multi-cloud operations

- **Hybrid cloud:** your own (private) infrastructure + a public cloud working together — keep sensitive data/legacy on-site, use public for extra capacity (**cloudbursting**); needs reliable links and consistent tooling. Example: AWS Outposts.
- **Multi-cloud:** services from **two or more public providers** — best-of-breed per provider, vendor-independence; needs skills/tools per provider. Examples: Dropbox (moved most user data off S3 to its own storage, 2016); Walmart's **Triplet model** (Azure + Google Cloud + private cloud).
- **The two can be combined** — one company can be both.
**Operational & networking challenges:** different consoles/APIs/IAM models per provider; cross-cloud networking (latency, **egress fees**, VPN/interconnect management); inconsistent security/monitoring; duplicated skills & tooling cost; data gravity and consistency.
**Mitigations (portability & interoperability):** containerised packaging (K8s as the portable runtime), IaC (Terraform multi-cloud), abstraction of identity (federated SSO) and secrets, common observability stack (OpenTelemetry/Prometheus), avoid provider-proprietary dependencies where portability matters, standardise on API-first integration.

## 9.8 Edge computing and the cloud-edge continuum

**Definition:** process data **near where it is created**; send only what matters to the cloud; keep working when links drop.
**Why not send everything to the cloud (3 marks):** latency (round-trip too slow for real-time decisions); bandwidth/cost (raw sensor/video volumes are huge); reliability (links drop — must keep operating offline); privacy/residency (some data must stay local).
**The continuum (4 layers — memorise with examples):**

| Layer | Where it runs | Good for | Example |
|---|---|---|---|
| **Device** | Sensor, camera, phone | Collecting data, tiny decisions | Camera detecting motion |
| **Local edge** | On-site gateway/server | Quick local decisions, offline operation | AWS IoT Greengrass |
| **Network edge** | Inside telecom network | Low delay for mobile users | AWS Wavelength (5G) |
| **Cloud region** | Large data centres | Storage, training, big analysis | Training an AI model |

Key line: apps are **split** across layers — **edge AND cloud, not edge OR cloud**. Pattern: edge filters/aggregates/decides in real time → cloud trains models, stores history, runs big analytics → updated models pushed back down to the edge.
**Practice examples:** AWS Wavelength (5G edge); AWS IoT Greengrass (local code/ML on devices, weak links tolerated); **Netflix Open Connect** (ISP-embedded CDN servers).

## 9.9 AI/ML workloads on the cloud

**Training vs inference (the contrast asked):**
- **Training:** teach the model from lots of data — heavy, runs hours+ on many accelerators, done occasionally, needs large storage and fast chip-to-chip links.
- **Inference:** use the trained model to answer requests — many small requests needing fast replies, runs constantly once live, can run in cloud **or at the edge**.
Example: spam filter learns from sample emails, then labels each new email.
**ML lifecycle on cloud:** Collect → Train → Evaluate → Deploy → Monitor. Cloud fit: rent accelerators only while training; big datasets in cloud storage near compute; managed tools (SageMaker, Vertex AI, Azure ML) so teams focus on models.
**Managed AI platforms:** Amazon SageMaker AI (build/train/deploy + MLOps), Amazon Bedrock (generative AI), Google Vertex AI, Azure ML.
**CPU vs GPU trade-off (3 marks):** CPU = few powerful general-purpose cores, best for serial/branchy logic, web servers, databases; GPU = thousands of simple cores in parallel, ideal for matrix maths (AI training/inference, graphics), high throughput but only for parallelisable work; TPU/ASIC = built for one job (ML maths) — best perf/watt, least flexible; FPGA = reprogrammable hardware, medium flexibility. Rule: **use the most specialised chip that still fits the workload.**
**Training vs inference resource selection:** training → large GPU/accelerator instances, reserved in blocks (Capacity Blocks for ML), spot-tolerant preprocessing; inference → smaller, latency-optimised, cost-per-request matters, autoscaling, possibly edge deployment.
**Cost/energy optimisation:** rightsize GPU instances; spot for pre-processing; Graviton/Arm for serving where supported; scheduled shutdown of idle training clusters; managed platforms' autoscaling; measure (PUE/energy per inference).

## 9.10 Quantum, neuromorphic, photonic (conceptual comparison — Tier 3 but easy marks)

- **Quantum:** normal bit is 0 or 1; a **qubit** can be a mix of 0 and 1 **until measured** → new ways to attack special problems (e.g., molecule simulation). Access model: write a **quantum circuit**, send to a cloud service (IBM Quantum — 5 qubits on cloud May 2016, now 100+; **Amazon Braket** GA Aug 2020 — one service over simulators + multiple vendors' hardware; Google **Willow** 105-qubit chip, Dec 2024, error correction improves with scale). Status: small, error-prone machines; research/learning value; **no proven real-world advantage yet — watch, don't replace**.
- **Neuromorphic:** brain-inspired chips — many simple artificial neurons sending **spikes**; **event-driven** (parts work only when a spike arrives → energy savings); **memory next to compute** (each neuron keeps state; less data movement). Contrast: CPU follows a clock; neuromorphic reacts to events. Example: Intel **Hala Point** (1.15 billion neurons, with Sandia). Status: research labs; promising for energy-efficient AI.
- **Photonic (two distinct ideas):** **photonic interconnect** — light carries data between chips/boards/racks (already in data-centre fibre; moving closer to the chip: Google TPU v4 optical circuit switches — ISCA 2023; NVIDIA co-packaged optics switches, Mar 2025; Lightmatter Passage) — aims to cut energy/delay of moving data. **Photonic compute** — light does the maths (Lightmatter Envise matrix math) — mostly research; hard problems (storing data in light). Key line: **light for moving data is further along than light for computing.**

**Comparison with conventional accelerators (4-mark answer):** GPU/TPU clouds are *production-ready today* with mature ecosystems; quantum/neuromorphic/photonic are early-stage, accessed via cloud services as experiments; potential applications differ (quantum: chemistry/optimisation/crypto research; neuromorphic: event-driven, ultra-low-power AI; photonic: first interconnect energy savings, later compute); limitations (error rates & tiny qubit counts; programming models; optical memory). Position: complements, not replaces, conventional cloud compute.

## 9.11 Green cloud computing

**Why:** data centres used **~1.5% of world electricity in 2024 (~415 TWh)**; IEA projects **945 TWh by 2030** (base case); local concentration (Ireland ≈ one-fifth of electricity).
**PUE (Power Usage Effectiveness) = total facility energy ÷ IT equipment energy.** 1.0 = perfect; 2.0 = one extra watt of cooling/losses per watt of computing. Lower is better. Figures: Google fleet PUE **1.09 (2024)**; Uptime Institute global average **1.54 (2025 survey)**. **Caveat:** PUE ignores server efficiency, water use, and the *source* of electricity.
**Levers (4):**
1. **Efficient data centres** — better cooling/power delivery → lower PUE.
2. **Efficient chips** — purpose-built silicon (Graviton, Inferentia, Trainium) does more per watt.
3. **Cleaner electricity** — renewable matching (Amazon matched 100% in 2024/2025 — but *annual accounting*, not every-hour clean power).
4. **Smarter use** — switch off idle servers; rightsize workloads (consolidation, scheduling, scale-to-zero serverless).
**Trade-off (3 marks):** performance/availability vs energy — consolidation packs hosts (energy saved) but raises blast radius of one host failing; switching off idle capacity saves power but lengthens scale-out time (cold capacity); SLAs may force keeping headroom/spares powered; "just-in-case" capacity is both a cost and an energy problem. Balance: measure p95, keep N+1 for the SLA, consolidate the rest.
**Two practical measures (2 marks):** e.g., (a) migrate steady-state services to Arm/efficient instance types and rightsize (p95+20–30%); (b) schedule dev/test off-hours with auto-stop, and raise utilisation via consolidation to retire idle hosts.

---
