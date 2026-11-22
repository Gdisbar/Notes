# PART 1 — CONCEPTS
*What it is · Why we need it · How it works · Where it fits (and what it costs) · Minimal example*
REF format `[C:page]` → see `00_READ_FIRST.md`. Glossary = `[C:1-7]` (formulae `[C:3]`).

**Contents:** A Cloud fundamentals · B Virtualization · C Hypervisors & IaaS · D AWS core · E Containers & Docker · F Orchestration (Swarm, Kubernetes) · G PaaS / SaaS / Serverless · H VM management, migration, IaC · I Capacity, VIM & scheduling · J SLA & high availability · K Multi-tenancy · L Application development & deployment · M Emerging cloud · N Likely concept questions

---

# A. CLOUD FUNDAMENTALS

## A1. What is cloud computing? `[C:8-11]`
- **What:** applications and services that run on a **distributed network, using virtualized resources, accessed by common Internet protocols and standards** — computing delivered as a **utility/service** by professional providers over the Internet. `[C:10]`
- **Why it arose:** cloud is the **convergence of earlier technologies** — web technologies, web services, **SOA**, distributed computing, **grids**, **clusters** (and parallel computing) — a "mash-up" re-shaped for modern demand. `[C:8-9]`
- **How (two enabling ideas) `[C:11]`:** **Abstraction** (implementation details hidden; apps run on unspecified machines with unknown data locations; outsourced administration) + **Virtualization** (resources pooled and shared, each user feels like the sole owner; scale up/down quickly, metered, multi-tenant). *Cloud exists because of virtualization and abstraction.*
- **Cost shift:** CapEx (buy hardware up-front) → **OpEx** (pay as you go). `[C:1]`
- ✍ *Terms:* convergence, utility computing, abstraction, virtualization, CapEx→OpEx.

## A2. NIST 3-4-5 rule `[C:12, C:4]`
**3** service models (IaaS, PaaS, SaaS) · **4** deployment models (public, private, hybrid, community) · **5** essential characteristics.

## A3. Five essential characteristics `[C:29-34]`
| Characteristic | Meaning | Key points |
|---|---|---|
| **On-demand self-service** `[C:32]` | provision without provider intervention | seconds/minutes; console + **API-driven automation**; **IaC** (Terraform, CloudFormation, Bicep, Pulumi) |
| **Broad network access** `[C:31]` | reach services over the network from any device | standard protocols (HTTPS, REST, gRPC); device-independent |
| **Resource pooling** `[C:30]` | shared pool serves many customers | **multi-tenancy** with logical isolation; economies of scale; high utilisation |
| **Rapid elasticity** `[C:33]` | add/remove capacity fast | scale out/in; **auto-scaling** policies; looks "unlimited" |
| **Measured service** `[C:34]` | usage metered and billed | pay-as-you-use, dashboards, **stop paying when idle**, no up-front investment |

## A4. Service models `[C:13-20]`
| Model | You get | Consumer controls | Provider manages | Examples |
|---|---|---|---|---|
| **IaaS** `[C:14-15]` | processing, storage, networks, VMs | OS, storage, deployed apps, some networking (firewalls) | physical infrastructure | EC2, Azure VMs |
| **PaaS** `[C:16-17]` | runtime/tools to deploy your own app | the app and its config | network, servers, OS, storage | App Service, Elastic Beanstalk |
| **SaaS** `[C:18]` | the provider's application via thin client | only user settings | everything | web email, CRM |
- *Who manages what:* `[C:20]`; *which one to pick:* `[C:19]`.

## A5. Deployment models `[C:21-28]`
| Model | Definition | Strength | Trade-off |
|---|---|---|---|
| **Public** `[C:22]` | provider-owned, pay-per-use for anyone (EC2, Azure, App Engine) | scale & speed, no capex | less control, vendor lock-in, shared |
| **Private** `[C:23]` | dedicated to one organisation (on-prem, colo or hosted) | control, customisation, compliance | higher cost, you carry responsibility, **finite capacity** |
| **Hybrid** `[C:24]` | two or more environments (private+public/on-prem) kept separate but connected | run each workload where it fits; burst | integration complexity, reliable links, consistent tools |
| **Community** `[C:25]` | shared by organisations with common requirements | shared governance & cost | limited scale, governance overhead |
- **Pros/cons grid:** `[C:26]`; analogy table (rent an apartment / own a house …): `[C:27]`.
- **Multi-cloud** `[C:28]`: two or more *public* providers — negotiating leverage, best-of-breed; **but** operational complexity, networking/cost overhead. **Hybrid ≠ multi-cloud; a company can be both** `[C:299]`.

## A6. Advantages, challenges, failures `[C:35-38]`
- **Advantages `[C:35]`:** lower cost, scalability, agility/faster delivery, reliability, global reach, security services.
- **Cloud app vs web app `[C:36]`:** a web app is merely browser-accessed and may sit on one or few servers; a **cloud(-native) application is designed for cloud capabilities** — elastic scalability, distributed infrastructure, automation.
- **Challenges & shared responsibility `[C:37]`:** provider secures physical DC, network, hardware/virtualisation, HA/DR, compliance certificates; **customer** secures IAM, MFA, data encryption/keys, application configuration, data governance. *The split moves with the service model* `[C:4, C:163]`.
- **Failures `[C:38]`:** major public-cloud outages occur → design for failure (multi-AZ, DR) — see `02_Scenario_Based.md`.

---

# B. VIRTUALIZATION

## B1. What / why `[C:39-41]`
- **What:** a computer-architecture technology where **one physical machine (host) runs multiple virtual machines (guests)** by pooling and sharing resources through a **hypervisor**.
- **Why:** server consolidation, isolation, higher utilisation, faster provisioning (VMs are *files*: snapshot, clone, restore), the foundation of IaaS.

## B2. The x86 problem `[C:42-44]`
- x86 OSes assume they **own bare metal**; four **privilege rings** (0 = kernel, 3 = user; ring 2 unused today). A guest OS *expects* ring 0 but the hypervisor needs it too — "no room for both".
- **Classic trap-and-emulate** works only if all sensitive instructions are privileged. On x86 some instructions are **sensitive but not privileged**: outside ring 0 they **execute incorrectly and silently — no trap**. That is the real obstacle.

## B3. Four virtualization approaches `[C:45-53, C:65]`
| | Emulation `[C:46]` | Binary translation / full `[C:48-49]` | Paravirtualization `[C:50]` | Hardware-assisted `[C:51-52]` |
|---|---|---|---|---|
| Idea | software imitates whole CPU/devices | hypervisor **rewrites** sensitive instructions on the fly | guest kernel **modified** to call hypervisor via **hypercalls** | CPU adds a new privilege mode ("Ring −1"); hypervisor runs there; sensitive instructions trap automatically |
| Guest modified? | no | no | **yes** (source-level; so open-source guests) | no |
| Guest ISA vs host | **can differ** (ARM on x86) | must match | must match | must match |
| Performance | slowest | moderate (translation overhead) | high | high (best today) |
| Examples | QEMU, Bochs | VMware (1999), early VirtualBox | Xen, User-Mode Linux | KVM, Hyper-V, Xen HVM (Intel VT-x, AMD-V) |
| Limit | too slow for production | complex engineering | can't run closed-source OS | needs modern CPU |
- **Emulation ≠ virtualization** `[C:53]`: emulation imitates different hardware (cross-ISA, slow); virtualization shares *the same* hardware among isolated guests (near-native).

## B4. What gets virtualized `[C:54-58]`
- **Server** (VMs; enables IaaS; restore = copy a file) · **Storage** (logical view over physical devices; SAN/NAS) · **Network** (VLAN etc.) · **Memory** (contiguous guest address space not tied to physical RAM, like OS virtual memory) · also desktop, application.

## B5. Overcommitment `[C:59-63, C:3]`
- **CPU overcommit ratio = total vCPUs ÷ total pCPUs.** Works because CPU is **time-sliced**. Guidance: **1:1** latency-sensitive (DBs, real-time); **3:1–5:1** mixed; **>8:1** high risk. Watch **CPU Ready time** (> ~5% ⇒ oversubscribed). Example: 6 VMs × 2 vCPU = 12 on 4 pCPU = **3:1**; but if 3 VMs peg both vCPUs → 6 vCPUs of demand on 4 pCPUs (1.5:1 squeeze for the busy subset) — *the ratio is a heuristic, not a guarantee*. `[C:60]`
- **Memory overcommit ratio = configured memory ÷ physical RAM.** Memory **cannot be time-sliced**, so the hypervisor must reclaim, from least to most invasive: **page sharing (TPS/KSM) → ballooning → compression → host swapping** (last resort; disk ≈ 100,000× slower than RAM). `[C:61-63]`

## B6. Advantages and caveats `[C:64-66]`
- Advantages: consolidation, isolation, availability (migration/restore), flexible provisioning, cost.
- **Points to note `[C:66]`:** **software licensing** (each VM may need its own licence → VM sprawl costs), **IT training**, management complexity, security of the shared host.

---

# C. HYPERVISORS & IaaS ARCHITECTURE

## C1. Hypervisor (VMM) `[C:67-71]`
- **What:** software/firmware/hardware that creates and runs VMs. Host = machine; guest = VM. **Type 1 (bare-metal)** runs directly on hardware (VMware ESXi, Xen, KVM, Hyper-V); **Type 2 (hosted)** runs as an application on a host OS (VirtualBox, VMware Workstation). `[C:67, C:69]`
- **Goals (Popek & Goldberg) `[C:68]`:** **equivalence/fidelity** (guest behaves as on bare metal), **resource control** (VMM has complete control of resources), **efficiency** (most instructions run directly on hardware); practical goals: isolation/fault containment, security boundary, consolidation, portability, operational agility.
- **Levels of virtualization `[C:71]`:** hardware, OS, library, application level — hardware and OS-level give highest performance; hardware and application levels are costliest to implement; user isolation is strongest at hardware level.
- **Type 1 vs 2:** Type 1 = performance, security, used in clouds; Type 2 = convenience for desktops, extra OS layer, slower.

## C2. IaaS architecture and actors `[C:72-73]`
- IaaS = hypervisors + cluster/resource management + storage + network + APIs, producing on-demand VMs.
- **NIST cloud actors `[C:73]`:** **Consumer** (uses), **Provider** (builds/operates), **Broker** (intermediates/adds value), **Auditor** (assesses/verifies), **Carrier** (connects/transports).

---

# D. AWS CORE (the course's reference public cloud)

## D1. Global infrastructure `[C:74-80]`
- AWS = public cloud with 240+ pay-as-you-go services. **Region** = separate, independent geographic area (a naming boundary, not a building). **Availability Zone (AZ)** = isolated location(s) within a region, connected by low-latency links, in lower-risk areas — **spread across AZs to survive zone failure**. `[C:75-77, C:5]`
- Extensions: **Local Zones** (extend a region to a city), **Wavelength** (AWS inside telecom networks → low latency to mobile), **Outposts** (managed AWS infrastructure in *your* site). `[C:78-79]`
- **Shared responsibility `[C:80]`:** AWS = security **of** the cloud; customer = security **in** the cloud.

## D2. IAM `[C:81-86]`
- **Why:** in IaaS every resource is network-reachable — IAM replaces the locked door.
- **Authentication** = who are you (password + MFA/SSO). **Authorization** = what may you do (least privilege). **Role** = permissions assumed temporarily, no long-term keys (avoid hard-coded credentials). **Policy** = JSON rule book attached to users/groups/roles. **Request flow ("IAM tango")** `[C:86]`: authenticate → evaluate policies → allow/deny. Organisation-level **SCPs** say what is *never* allowed. `[C:2]`

## D3. EC2 `[C:87-96]`
- **EC2** = resizable compute; you launch **instances** (VMs). **AMI** = template of initial software state (OS, patches, apps); sources: AWS, Marketplace, existing instances, uploaded VMs `[C:88-89]`.
- **Instance families:** t (burstable), m (balanced), c (compute), r (RAM), x (extreme memory), p (GPU), i (IOPS), d (dense HDD), g (Graviton/ARM). `[C:5]`
- **Lifecycle `[C:90, C:194]`:** pending → running → stopping → stopped (disks kept, no compute charge; restart may change host & public IP) → shutting-down → terminated (final). **Bootstrapping** = code run at launch (user-data/cloud-init).
- **Tenancy `[C:91-92]`:** shared (default, isolated by hypervisor), **Dedicated Host** (whole server; licence compliance). **Placement group `[C:93]`:** logical grouping in one AZ for low-latency 10 Gbps networking. **Instance store** = local disk that disappears with the instance `[C:5]`.
- **VPC `[C:94-96]`:** logically isolated virtual network; components: **subnets, route tables, DHCP option sets, security groups, network ACLs**; optional: **internet gateway, Elastic IP, ENI, endpoints, NAT, VPN**. Default VPC subnets are public.

## D4. Storage and databases `[C:97-108]`
| | **S3** (object) `[C:98-100]` | **EBS** (block) `[C:101-102]` | **EFS** (file) `[C:103]` | **Glacier** `[C:105]` |
|---|---|---|---|---|
| Model | objects by key via HTTP API | persistent volumes for one EC2 | shared file system (NFS) for many servers | archive |
| Scale | unlimited | sized volumes (e.g., GP SSD 1 GB–16 TB) | elastic, virtually unlimited | unlimited |
| Durability/availability | replicated across AZs | replicated **within one AZ** | multi-AZ | very durable |
| Use | static content, data lakes, backups, media | boot volumes, databases, low-latency | shared content, big-data, home dirs | **infrequent access; retrieval hours (3–5 h)** |
- Comparison grid `[C:104]`. **Block** = disk-level, **file** = hierarchy via OS, **object** = flat, API access `[C:97]`.
- **DBs `[C:106-108]`:** **RDS** (managed relational: MySQL, PostgreSQL, MariaDB, Oracle, SQL Server, Db2, Aurora; **Multi-AZ** synchronous standby for automatic failover), **DynamoDB** (key-value NoSQL), **ElastiCache** (in-memory), **Redshift** (analytics warehouse).
- **Case:** Zomato moved Trino/Druid from x86 to **Graviton (Arm) + Spot**: −30% compute cost, +25% query performance, ~$300K/yr saved `[C:109]`.

---

# E. CONTAINERS & DOCKER

## E1. Containers `[C:110-119]`
- **What:** software that **packages application code + dependencies** so it runs the same in any environment; **OS-level virtualization** — containers **share the host kernel** (no guest OS). `[C:110]`
- **How isolation works `[C:111-113, C:5]`:** **Namespaces** give a process its own isolated *view* (process IDs, network, mounts, hostname…) — `ps aux` inside shows a few processes, on host hundreds. **cgroups** *limit and account* CPU, memory, disk I/O. **Namespaces isolate; cgroups limit.**
- **LXC** = Linux Containers tool in user space; **LXD** = manager on top (images, networking, API). `[C:114-115]`
- **OS containers** behave like lightweight VMs running many processes/services (LXC); **application containers** package a single service/process (Docker, Rocket). `[C:116-118]`
- **VM vs container `[C:119]`:** VM = hypervisor + guest OS per VM (GBs, minutes, strong isolation, any OS); container = Docker engine + shared kernel (MBs, seconds, high density, **kernel must match host**, weaker isolation).

## E2. Docker `[C:120-134]`
- **Motivation:** "works on my machine" problems, heavy VMs → containers as lightweight alternative `[C:120-121]`.
- **Architecture `[C:122-124]`:** **client** (CLI) → **daemon (`dockerd`)** does the work (builds, runs, distributes; manages images, containers, volumes, networks) → **registry** (Docker Hub public; private registries). `docker pull/run` fetch from the configured registry.
- **Objects `[C:125-128]`:** **Image** = read-only layered template built from a **Dockerfile** (first instruction `FROM` = base image); **Container** = running instance with a thin writable layer; also volumes, networks. **Why Dockerfile:** reproducible, version-controlled, reviewable; vs `docker commit` (undocumented, unrepeatable).
- **Layers & copy-on-write `[C:129, C:5]`:** each instruction creates a new read-only layer; containers **share image layers** and copy a file to their own writable layer only when modified → fast start, small footprint.
- **Dockerfile order/best practices `[C:130-132]`:** rule-based order (`FROM` first; `WORKDIR`, `ENV`…), **caching-driven order** (copy dependency manifest → install → copy source, so cache survives code changes); small base images, pin versions, don't run as root, one concern per container, `.dockerignore`.
- **Life cycle `[C:133]`:** Dockerfile → **build** → image → **push** to registry → **pull** → **run** container. **Summary `[C:134]`.**
- *Minimal example:* `docker run -d -p 80:80 nginx` → daemon pulls image, creates container, maps port.

---

# F. CONTAINER ORCHESTRATION

## F1. Why orchestration; imperative vs declarative `[C:135-137]`
- One container on one machine is easy (`docker run`); **many containers on many machines** raise: placement, scaling, networking, self-healing, rolling updates, load balancing. `[C:137]` Key features: management of containers, launching sets, rescheduling failed ones, linking, connectivity, firewall/port control, **load balancing, horizontal scaling, rolling upgrades**.
- **Imperative** = script *how* (TV remote). **Declarative** = state *desired result* and let the tool **converge** (thermostat) — the GitOps style. `[C:136, C:5]`

## F2. Docker Swarm `[C:138-146]`
- **What:** Docker's built-in clustering ("swarm mode", since Engine 1.12): `docker swarm init` turns machines into one cluster; you declare "run 3 copies of this image" and Swarm places and maintains them. `[C:138]`
- **Objects `[C:139]`:** **Stack** (whole app) → **Service** (a part, as definition) → **Task** (one scheduled copy) → **Container** (exactly one per task). *Replicated* services run N copies; *global* services one per node.
- **Manager fault tolerance (Raft):** failures tolerated = ⌊(N−1)/2⌋ → 3 managers tolerate 1, 5 tolerate 2; even counts add nothing. `[C:139, C:3]`
- **Networking `[C:141]`:** **overlay network** (virtual network across nodes), **ingress + routing mesh** (published port reachable on *any* node). **Stack file `[C:142]`:** Compose file with `deploy:` keys, used by `docker stack deploy`; images must come from a registry. **Security `[C:143]`:** mutual-TLS node certificates, join tokens, encrypted secrets, autolock. Commands `[C:144]`.
- **Pros `[C:146]`:** easy to learn, nothing extra to install, secure by default, light, essentials built-in (scaling, rolling update/rollback, discovery, LB), quick. **Cons:** **no automatic scaling**, weak for stateful data (volume stuck on one node), all-or-nothing health, small ecosystem, no managed cloud service, uncertain long-term future. `[C:145-146]`

## F3. Kubernetes (K8s) `[C:147-160]`
- **What:** open-source platform that runs containers across a cluster; you describe the **desired state** in files and it **continuously reconciles** reality to match (observe → compare → act, repeat). Name = Greek "helmsman". `[C:147-148]`
- **Control plane `[C:150-151]`:** **kube-apiserver** (front door: authN, authZ, admission; stores in etcd), **etcd** (cluster memory/state), **kube-scheduler** (picks node), **kube-controller-manager** (controllers: the "fixers"). **Worker `[C:152]`:** **kubelet** (node agent), **kube-proxy** (service networking rules), **container runtime** (containerd — Docker Engine not required). **Pod** = smallest unit (one or more containers sharing IP/storage).
- **Add-ons `[C:153]`:** CNI network plugin, CoreDNS, metrics server, CSI storage, ingress controller.
- **Object anatomy `[C:154]`:** `apiVersion`, `kind`, `metadata` (name, namespace, labels), `spec` (desired state). Common kinds: Deployment, StatefulSet, Service (ClusterIP/NodePort), ConfigMap/Secret, PVC, HPA.
- **Pod life cycle `[C:156]`:** ① controller creates Pod object → ② scheduler picks node → ③ kubelet starts containers via runtime → ④ kubelet reports status → ⑤ failure handling: pod crash → kubelet restarts (CrashLoopBackOff if repeated); node failure → controller notices missing heartbeats and **reschedules pods elsewhere**.
- **Why K8s over Swarm `[C:148-149]`:** built-in autoscaling (HPA), readiness vs liveness checks, persistent volumes/StatefulSets/operators, advanced scheduling (requests, limits, affinity, taints), namespaces/RBAC for multi-team, extensible (CRDs), managed offerings (EKS/AKS/GKE), wide adoption. **Swarm** wins on simplicity and light footprint for small teams. Recap comparison `[C:161]`.
- **FreshCart worked example `[C:155, C:158-160]`:** 4 parts — **web** (nginx+React, 2 copies), **api** (Node/Express, 3 copies, autoscale 2–8), **redis** (carts; disposable), **postgres** (must never be lost; 1 + own disk). *Swarm view:* 4 services → 7 tasks, overlay networks `frontend`/`backend`, `db` pinned by label, secret + config + volume. *K8s view:* Deployments (web, api, redis), **StatefulSet** for postgres, Services (web NodePort, api ClusterIP), ConfigMap/Secret/PVC, HPA; control-plane tainted.

---

# G. PaaS, SaaS, SERVERLESS

## G1. Shared responsibility across models `[C:163, C:184]`
| Layer | On-prem | IaaS | PaaS | FaaS | BaaS | SaaS |
|---|---|---|---|---|---|---|
| Application / logic | You | You | You | You | Shared | Vendor |
| Data & configuration | You | You | You | You | You | **You (shared)** |
| Runtime/scaling | You | You | Vendor | Vendor | Vendor | Vendor |
| OS | You | You | Vendor | Vendor | Vendor | Vendor |
| Hardware/virtualisation | You | Vendor | Vendor | Vendor | Vendor | Vendor |
- **Data, identity and access remain your accountability in every model.**

## G2. PaaS `[C:162-171]`
- **What:** you bring **code**; provider runs runtime, OS, servers, scaling. Gives: language runtimes, build/deploy (git push, CI/CD), scaling & load balancing, managed data services, identity/secrets, logs/metrics/tracing. `[C:164]`
- **Between IaaS and serverless `[C:165]`:** IaaS = VM image (you patch OS), **CaaS** = container image (you manage orchestrator/worker nodes: ECS/EKS/AKS/GKE), PaaS = code or container + platform (App Service, Beanstalk, Cloud Run).
- **Why teams choose `[C:166]`:** faster time to market, no hardware capex, platform patching, built-in scaling, consistent environments, shared pipelines. **Costs `[C:167]`:** limited control, forced upgrades, scale/cold-start limits, cost at scale, harder debugging, **lock-in**. **Fits / doesn't `[C:168]`:** web apps/APIs on standard stacks, small teams, variable traffic, dev/test ✔; custom kernels/drivers, special networking/GPU/HPC, steady very high load, strict portability ✘.
- Four PaaS families `[C:170]`; managed services equivalents across AWS/Azure/GCP `[C:171]`; Azure hierarchy (tenant → subscription → resource group → resource; ARM) `[C:169]`.

## G3. SaaS `[C:172-179]`
- **What:** use the provider's app (browser/API); billing by seat/usage. **Inside a SaaS product `[C:173]`:** one codebase, tenant resolved on every request, **tenant ID on every row**, **row-level security**, configuration (metadata) not code, control plane (onboarding, plans, billing, metering).
- **Tenancy models `[C:174]`:** **Silo** (dedicated stack/tenant — strong isolation, easy tuning/compliance; highest cost), **Pool** (shared stack, tenant-ID separation — lowest cost, one upgrade path; **noisy-neighbour** risk), **Bridge** (shared app, tiered data/compute — premium tenants isolated).
- **Benefits `[C:175]`:** customers — low entry cost, no installs, any device, frequent updates; vendors — recurring revenue (MRR/ARR), one codebase, usage data.
- **Subscription vs ownership `[C:176]`:** 200 users × $100/month = $240K/yr vs on-prem $350K + $150K/yr → cumulative Year 4: SaaS 960 vs 950; Year 5: 1,200 vs 1,100 ⇒ **SaaS cheaper for ~4 years, then ownership wins.**
- **Six limits `[C:177]`:** limited customisation, vendor-set upgrades, borrowed availability, integration limits, exit/portability, data/compliance. **Ask the vendor `[C:178]`:** security (isolation, SSO/MFA, SOC 2/ISO 27001), compliance/residency, availability/SLA. **Fits/doesn't `[C:179]`:** CRM/HR/finance ✔; embedded firmware, offline-first, ultra-low latency, strict sovereignty ✘.

## G4. Serverless: BaaS & FaaS `[C:180-183]`
- **Serverless** = no servers to manage, scales with demand **including to zero**, pay for use (servers exist; you never see them).
- **BaaS** = ready-made backend (auth, DB, storage, push) via SDK — Firebase, Supabase, Amplify; fastest to ship; **lock-in**, misconfigured security rules.
- **FaaS** = function runs on an **event** (HTTP via API gateway, upload, queue message, timer); stateless, billed for time; **AWS Lambda** (≤ **15 min** per run; 128 MB–10,240 MB; Firecracker microVMs). **Constraints `[C:183]`:** stateless/short-lived, **cold starts**, concurrency limits, event-source lock-in; keep state outside.

## G5. Five models side by side `[C:184-188]`
| | You deploy | Typical user | Scaling | Billing | Control | Ops effort | Portability | Main risk |
|---|---|---|---|---|---|---|---|---|
| IaaS | VM image | sysadmins | you configure | VM-hour/storage | highest | high | best | misconfiguration, unpatched systems |
| PaaS | code/container | app developers | platform rules | plan/instance-hours | medium-high | medium | medium | deprecation, plan limits |
| FaaS | function | event-driven devs | automatic per event | per invocation | low-medium | low | low | cold start, limits, cost at scale |
| BaaS | (none — SDK) | front-end teams | vendor | usage | low | low | low | lock-in |
| SaaS | (none) | end users | vendor | subscription | lowest | lowest | lowest | availability borrowed, exit |
- **Decision path `[C:187]`:** standard capability needed? → **SaaS**; need backend without building it? → **BaaS**; need OS/kernel/hardware control? → **IaaS**; event-driven/short/spiky? → **FaaS**; else **PaaS** (containers/CaaS when you want PaaS-style ops with portability). **Real systems mix models** (FreshCart `[C:188]`).

---

# H. VM MANAGEMENT, MIGRATION & INFRASTRUCTURE-AS-CODE

## H1. Resource management & VM life cycle `[C:189-197]`
- **Private cloud problem `[C:189-190]`:** finite capacity can't be hidden; the *immediate provisioning* model of public clouds is inadequate; a hosted private cloud ≠ AWS "VPC". **Why resource management `[C:191]`:** multi-cloud scatter, data everywhere. **RM `[C:192]`:** procurement and release of resources — create/place VM per task, release when done to the free pool.
- **VM life cycle `[C:193]`:** request → inspect resource pool, match → provision → configure → run → monitor → release.
- **Provisioning `[C:195-197]`:** by manual OS install, **AMI + user-data/cloud-init**, template, or cloning. Steps: **validate (IAM, quota) → place (scheduler picks host) → image (root disk from AMI) → network → boot → contextualise (cloud-init: hostname, keys, install software)**. *If a person has to install nginx by hand, the cloud's benefit is lost* `[C:197]`.

## H2. VM migration `[C:198-209, C:3, C:6]`
- **What/why `[C:198-199]`:** move a VM's state (CPU, memory, disks, devices, config) between hosts/datastores — **host maintenance/failing hardware, load balancing & consolidation, DR / DC exit, cloud-to-cloud move, cost/performance tuning**.
- **Types:** **hot/live** (running; downtime tens–hundreds of ms; whole migration seconds–minutes), **cold** (powered off; off for entire copy; no CPU-compat check, no shared storage), **live storage migration** (disks move while VM runs), **live migration with storage / shared-nothing**.
- **Xen live (pre-copy) stages `[C:202-205]`:** pre-migration → reservation → **iterative pre-copy** (copy memory in rounds while VM runs; re-send **dirtied** pages) → **stop-and-copy** (suspend, redirect traffic, send CPU state + last pages) → **commitment** (source released) → **activation** (resume; reattach drivers; advertise IP). Failure safety: at least one host always holds a consistent image.
- **Worked numbers `[C:204, C:3]`:** M = 8 GiB, bandwidth B = 1 GiB/s, dirty rate D = 0.2 GiB/s, stop when remainder < 0.3 GiB: rounds 8 → 1.6 → 0.32 → 0.064 GiB; final stop-and-copy 0.064 GiB = **64 ms downtime**; total ≈ 10 s. **Convergence rule: each round shrinks by D/B, so pre-copy only converges when D < B.**
- **Pre-copy vs post-copy `[C:6]`:** pre-copy pushes memory while running; **post-copy** runs on target and pulls faulted pages (🆕 risk: failure mid-way loses VM).
- **Classic vs shared-nothing `[C:206-207]`:** classic needs **shared storage** (only CPU+memory move), low downtime; shared-nothing mirrors disks first then pre-copies memory — more flexible (no SAN), more complex/longer. **Preconditions:** compatible CPU features, same reachable network, enough free capacity; **blockers:** GPU/PCI pass-through, SR-IOV, local USB/CD, very large/busy VMs, affinity rules.
- **Storage migration steps `[C:209]`:** reads/writes to source while copying → writes mirrored to both → synchronised → switch → delete source.

## H3. Infrastructure as Code and governance `[C:210-213]`
- **IaC:** infrastructure described in text files (in Git) that a tool turns into resources — repeatable, reviewable, rollbackable; avoids hand-built "snowflakes". **Declarative** (state what; tool works out how) is the norm; cycle: write → **plan/change set** → apply → **state** recorded. **Idempotent** = applying twice changes nothing. **Drift** = real resources changed outside code. `[C:210-211, C:6]`
- **Tags** (key/value: Owner, Project, Environment, CostCenter) for ownership, cost allocation, automation (stop dev at night), access rules — small mandatory set, enforce at creation. `[C:212]`
- **Quotas** (limits per service/region; many raiseable), **budgets & alerts** (50/80/100%), **guardrails** (SCPs). `[C:213]`

## H4. 6 R's of migration `[C:214, C:6]`
**Rehost** (lift-and-shift), **Replatform** (small optimisations), **Repurchase** (move to SaaS), **Refactor/Re-architect** (cloud-native), **Retire**, **Retain** — choose per application by business value, technical fit, cost, timeline.

---

# I. CAPACITY MANAGEMENT, VIM & WORKLOAD SCHEDULING

## I1. Capacity management `[C:215-219, C:3]`
- **Goal:** keep capacity between **too little** (poor performance, SLA-breach penalties, lost service, impact on other services) and **too much** (wasted resources, VM sprawl, higher licence cost). Industry surveys: much cloud spend goes on idle/oversized resources. `[C:216]`
- **Find the right balance `[C:217-218]`:** application sizing; **shares, limits and reservations**; continuous monitoring/tuning; VM consolidation; power off unused hosts; automation (templates/golden images, resource pools, load balancing, automatic placement/re-balancing); **protect critical apps** (highest priority shares, guaranteed reservations, CPU affinity).
- **Five challenges `[C:215]`:** capacity on demand, automation, tenant isolation & security, scalability, "just-in-case" habit (balance performance, cost, security, responsiveness).
- **Sizing formula `[C:219, C:3]`:** hosts = ⌈ p95 × (1+growth) × (1+headroom) ÷ vCPU per host ⌉ + 1 spare. Example: p95 = 120 vCPU → ×1.15 = 138 → ×1.25 = 172.5 → 173 → ÷48 = 3.6 → **4 hosts + 1 spare (N+1) = 5**. Steps: **measure → forecast → headroom → convert to hosts → add spare**.

## I2. Virtual Infrastructure Manager (VIM) `[C:220-228]`
- **What:** control plane for a pool of hypervisor hosts — decides **where VMs run** and manages their **life cycle**; allocates compute, storage, network on current needs. Needed because the ecosystem is distributed and location matters; early schedulers placed each VM once greedily; modern ones **re-balance continuously**. `[C:221-222]`
- **OpenNebula `[C:223-228]`:** open-source cloud middleware for **heterogeneous distributed data centres**; supports Xen, KVM, VMware via **pluggable drivers** (high-level "start/stop VM" translated to hypervisor commands); private/public/hybrid; features: CLI + XML-RPC/libvirt, **requirement/rank scheduler**, advance reservation (Haizea), image & network management. Benefits: centralised management, placement policies (balance, consolidation), dynamic resizing, isolation, bursting. Platform comparison `[C:228]`.

## I3. Workloads and scheduling `[C:229-248]`
- **Terms `[C:231]`:** **workload** (application + resource demand), **lease** (contract between provider and user for resource use), **application → jobs (work-breakdown) → tasks**. **Resource allocation (RA)** = identify/allocate a resource for a workload type; **task scheduling (TS)** = shuffle/manage tasks efficiently on available resources.
- **Placement is NP-hard → heuristics `[C:232]`:** first-fit, best-fit, worst-fit. Example: hosts of capacity 10; requests 4,5,6,3,2 → **first-fit (arrival order) = 3 hosts** (4+5, 6+3, 2); **first-fit decreasing** (6,5,4,3,2) = **2 hosts** (6+4, 5+3+2). *Tighter packing saves hosts but raises blast radius.* `[C:3]`
- **How schedulers decide `[C:233]`:** **Filter** (remove hosts failing hard constraints: free CPU/memory, AZ, hypervisor/image, affinity, taints) → **Score/weigh** (spread vs pack, balanced load, locality) → pick best. No host passes → K8s pod **Pending**; OpenStack Nova "No valid host". Possible **priority/preemption**. **Overcommit** via statistical multiplexing.
- **VM vs container scheduling `[C:234]`:** VM = fixed size, minutes to start, heavy to move, strong isolation; pod = millicores/MiB, seconds, cheap to reschedule/resize, shared kernel. Scheduler uses **declared requests**, not live usage — wrong declarations waste or cause contention.
- **Amdahl's law `[C:235, C:3]`:** S_max = 1 ÷ [(1−p) + p/s]. p=0.9, s=10 → ≈5.26×; ∞ speed-up caps at 1/(1−p)=10×. **Serial part limits scale-out.**
- **Provisioning models `[C:236-240]`:** **immediate** (public clouds; capacity assumed infinite), **best-effort** (queue and prioritise), **advance reservation** (pre-reserve for a time window; known demand e.g., instrument running 2–4 pm). *Naive reservation causes under-utilisation* (fragmentation); **VMs help** (suspend, migrate, resume without changing the app) but add **preparation overhead** (disk-image transfer) and **runtime overhead** (suspend/resume).
- **Leases `[C:241-242]`:** **Best-Effort Lease (BEL)** — queue, earliest start, backfilling, may use suspend/resume to fit; **Advanced Reservation (AR)** — no queue; starts at requested time or not at all; handles preparation overhead via **EDF** image-transfer scheduling; may pre-empt BE leases.
- **Reservation models in AWS `[C:237]`:** **On-Demand** (best effort), **Spot** (spare capacity, discount, interruptible with **2-minute warning**), **On-Demand Capacity Reservation** (guaranteed in one AZ, pay whether used or not), **Capacity Blocks** (GPU, future date). Savings Plans / Reserved Instances lower price but **do not reserve capacity**.
- **Scaling `[C:243]`:** **vertical (scale up)** — simple, no app change; ceiling, restart, still single point of failure. **Horizontal (scale out)** — no hard ceiling, failure-tolerant, fine-grained cost; needs **stateless design**/shared state, load balancer, coordination overhead. *Scale up for quick relief; design to scale out.*
- **K8s scheduling `[C:244-245]`:** **request** = reserved share (what scheduler uses); **limit** = hard cap — CPU limit → **throttling**, memory limit → **OOM-kill**. **Node allocatable = capacity − system reserved.** Example: 2-vCPU node ≈ 1.9 vCPU allocatable; pods of 500m → **3 fit** (1.9 ÷ 0.5 = 3.8↓); 4th **Pending** even if node idle. 1 CPU = 1000 millicores.
- **Cost-aware `[C:246-248]`:** waste = oversized requests, idle nodes, always-on dev, wrong purchase model. Levers: **rightsizing** (new request ≈ **p95 + 20–30%**, measure ≥ 2 weeks incl. peaks; conservative for memory), scale-in, bin-packing/consolidation, off-hours dev/test, Arm instances. **Triad:** steady base → commit (Savings Plan/reserved); spiky → autoscale; interruptible batch → Spot with checkpointing; unmissable launch → capacity reservation.

---

# J. SLA & HIGH AVAILABILITY

## J1. SLA, SLO, SLI `[C:253, C:256-257, C:2]`
- **SLA** = the *entire agreement*: what service, how supported, times, locations, **costs, performance, availability, responsibilities, penalties**. **SLO** = specific, **measurable target** within it (availability, throughput, response time, e.g., **RTO**). **SLI** = the **actual measurement** compared with the SLO. **KPI** = metric with thresholds that decides if the objective is met.
- **Good SLO properties `[C:253]`:** attainable, repeatable, measurable, understandable, meaningful, controllable, affordable, mutually acceptable.
- **Allowed downtime `[C:3]`:** downtime = (1 − availability) × period; in a 30-day month (43,200 min): **99.9% = 43.2 min; 99.95% = 21.6 min; 99.99% = 4.32 min**.

## J2. High availability (HA) `[C:254-255]`
- HA is driven by the SLA (e.g., net banking 24×7 with published maintenance windows; 99.95% availability).
- **Steps `[C:255]`:** **build for server failure** (redundant servers that can come online) → **build for zone failure** (DR site, switch-over) → **build for cloud failure** (contain errors, **"fire-door" policy** so faults don't cascade) → **automate and test** (test repeatedly; little manual interference). DR terms: **RTO** (time to resume; the deck prints "Return to Operation") `[C:2]`.

## J3. Traditional SLO management `[C:258-260]`
- Early approach: provision for peak load up-front. Tools: **load balancing** (front-end distributes to replicas; **class-agnostic** vs **class-aware** — the latter inspects client/request type), **admission control** (police requests under overload): **request-based** (reject when at capacity; may break multi-request sessions) vs **session-based** (finish admitted sessions, reject new ones).

## J4. SLA types and life cycle `[C:261-270]`
- **Infrastructure SLA** — provider guarantees servers/power/network (e.g., hardware 99% monthly, power 99.99%, network 99.99%); **Application SLA** — in co-hosting, capacity allocated by demand, provider guarantees application-level metrics (response time, throughput).
- **Infra SLA vs capacity management `[C:262]`:** predictable demand → **reservations**; unpredictable/elastic → **capacity planning + continuous optimisation** to meet SLAs and use capacity efficiently.
- **SLA life cycle (5 phases) `[C:263-265]`:** **contract definition** (catalogue of standard templates, customise) → **publication & discovery** → **negotiation** → **operationalisation** (**monitoring**, **accounting**, **enforcement**) → **de-commissioning** (termination terms).
- **Cloud-application SLA life cycle (5 phases) `[C:266-270]`:** **feasibility** → **on-boarding** (move app to hosting platform; **profile runtime characteristics**) → **pre-production** (simulated production to verify and agree SLA) → **production** (monitor; sustained breach → remedial action/renegotiation) → **termination** (transfer data back, retain only essentials).

---

# K. MULTI-TENANCY

## K1. Concept `[C:249-252]`
- **What:** architectural pattern in which **a single instance of software on the provider's infrastructure serves multiple tenants**; unlike plain multi-user, it needs **per-tenant customisation of that one instance**. **Multi-instance** = one copy per tenant.
- **Four maturity levels `[C:251-252]`:** (1) **Custom instances** — own version per customer; hardest to manage. (2) **Configurable instances** — same version, per-tenant config (logos, workflows); one codebase; easier upgrades. (3) **Configurable, multi-tenant-efficient instances** — a *single instance* shared; efficient and simple to manage. (4) **Scalable, configurable, multi-tenant-efficient resources** — level 3 + instances on a cluster scaling with demand; maximum sharing (e.g., Gmail).

## K2. Multi-tenancy in each service layer `[C:271-273]`
| Layer | Options |
|---|---|
| **IaaS** | **VM isolation** (strongest, least efficient), **network isolation** (virtual network per tenant; better efficiency, complex), storage isolation, hybrid |
| **PaaS** | **Application-level** (containers/virtualisation), **tenant-level** (dedicated platform instance), **resource-pool** (dedicated CPU/memory/storage; good performance, less secure), hybrid |
| **SaaS** | **Database-level** (separate schema per tenant: complete isolation, complex DB management), **application-level** (partition data in app: flexible, less secure), hybrid |

## K3. Benefits, risks, characteristics `[C:274-278]`
- **Benefits:** **cost savings** (per-instance memory/CPU overhead shared), **data aggregation** (one schema → easy cross-tenant analytics), easy upgrades.
- **Disadvantages:** **complexity** (customisation, per-tenant metadata), **release risk** (one update can cause downtime for *all* tenants; bugs may appear in other tenants' views; release windows constrained).
- **Characteristics `[C:276]`:** **customisation** — branding (skin), workflow, **data-model extensions**, access control; **quality of service** isolation (a tenant must not degrade others); **security** — secure sharing; **authentication** `[C:278]`: **centralised** (one user DB, tenant admin rights to manage own users) vs **decentralised** (each tenant authenticates its own users).

## K4. Data sharing and customisation `[C:279-281]`
- **Shared tables:** all tenants' rows in one table separated by **tenant ID**; space-efficient; needs views/joins and a **metadata table**; customisation hard. **Dedicated tables per tenant:** one DB instance, separate table copies; easy customisation, more space; only that tenant has privileges.
- **Customisation of shared tables:** **pre-allocated columns** (fixed spare columns: waste if too many, restriction if too few), **name-value pairs**, XML/extension tables `[C:281, C:5]`.

---

# L. APPLICATION DEVELOPMENT & DEPLOYMENT

## L1. Architectural styles `[C:283-285]`
| | Monolith | Microservices | Serverless functions |
|---|---|---|---|
| Idea | one program | many small services each doing one job | small functions on events |
| Deploy | all as one unit | each service independently | each function independently |
| Scale | copy the whole app | scale only the busy service | platform scales per event |
- **Netflix `[C:284]`:** database corruption (Aug 2008) → move to AWS (completed Jan 2016); microservices behind API gateway **Zuul**; still runs its own CDN (**Open Connect**). *Takeaway: a cloud move is a long journey.* **Prime Video `[C:285]`:** a video-monitoring tool moved **from microservices + serverless to a single monolith on EC2/ECS → >90% cost reduction**. *Takeaway: smaller is not always better; fit the style to the workload.*

## L2. 12-factor app `[C:286-287]`
Methodology for portable, robust, scalable SaaS apps: declarative setup, clean contract with OS (portability), cloud-platform friendly, **minimal dev/prod divergence**, scalable without changing practices; config in environment, stateless processes, backing services as attached resources, logs as streams, disposability.

## L3. APIs and integration `[C:288-290]`
- **API** = agreed way for one program to ask another (e.g., S3 web API). **API gateway** = single front door: **authenticates, throttles (rate limits), routes** to the right service. **Synchronous** (caller waits; simple, but stuck if the other service is down) vs **asynchronous/messaging** (queue; absorbs spikes; result later) .

## L4. CI/CD and safe releases `[C:291-293]`
- **Pipeline:** commit → build → test → release → deploy. **CI** merge small changes often; **Continuous Delivery** every passing change is ready; **Continuous Deployment** goes live automatically. Tests: unit → integration → end-to-end (fast/cheap first).
- **Safe release strategies `[C:292]`:** **rolling** (replace a few at a time), **blue-green** (two identical environments; switch traffic; instant rollback), **canary** (small share of traffic first). **Spinnaker** (from Netflix, 2015) supports blue-green & canary. **DORA** metrics: deployment frequency, lead time for changes, change failure rate, time to restore `[C:293, C:6]`.

## L5. Four ways to deploy `[C:294-296]`
| | Custom VMs | Containers | Managed platform | Serverless |
|---|---|---|---|---|
| You manage | OS, runtime, app, scaling | app + image + cluster settings | app code & data | function code |
| Scaling | you build it | orchestrator | platform | automatic per event |
| Control / setup effort | highest / highest | high / medium | medium / low | lowest / lowest |
| Best for | custom OS needs | portable microservices | web apps | event-driven/uneven load |
- **Lambda `[C:296]`:** since Nov 2014; ≤ 15 min; 128 MB–10,240 MB; *suits short event-driven work, not long-running jobs*.

## L6. Observability `[C:297-298]`
**Logs** (what happened), **Metrics** (how much), **Traces** (where — one request across services), **Alerts** (notify when a number crosses a limit). **Four golden signals (Google SRE):** **latency, traffic, errors, saturation**.

## L7. Hybrid vs multi-cloud `[C:299]` — see A5.

---

# M. EMERGING CLOUD TOPICS `[C:300-310]`
- **Evolution `[C:300]`:** S3 (Mar 2006) → EC2 beta (Aug 2006) → Lambda & ECS (2014) → quantum on cloud (IBM, 2016; Braket 2020): *the cloud keeps turning new technology into rentable services*.
- **Edge `[C:301-302]`:** device → edge → cloud region; faster response, less data sent, works with weak links. **Continuum:** device → local edge (Greengrass) → network edge (Wavelength) → cloud region.
- **AI/ML on cloud `[C:303-304]`:** **training** (heavy, hours, many accelerators, occasional) vs **inference** (many small fast requests, always on, cloud or edge). **Lifecycle:** collect → train → evaluate → deploy → monitor; rent accelerators only while training; keep data near compute; managed tooling.
- **Quantum `[C:305]`:** qubit (0/1/superposition); cloud access = write circuit → send to cloud quantum service. **Neuromorphic `[C:306]`:** spiking neurons, event-driven, memory next to compute — energy-efficient. **Photonic `[C:307]`:** light for interconnect (mature) vs light for compute (early research).
- **Accelerators `[C:308]`:** **CPU** (general), **GPU** (thousands of parallel cores; AI), **TPU/ASIC** (custom for ML maths; least flexible), **FPGA** (reprogrammable) — *match the chip to the job*.
- **Green cloud `[C:309-310]`:** data centres ≈ 1.5% of world electricity (2024, ~415 TWh), IEA projects ~945 TWh by 2030; local concentration (Ireland ~1/5). **PUE = total facility energy ÷ IT equipment energy** (1.0 perfect; 2.0 = one extra watt of cooling/losses per computing watt); Google fleet 1.09 (2024) vs industry average 1.54; PUE ignores server efficiency, water use, electricity source. `[C:4]`

---

# N. LIKELY CONCEPT QUESTIONS (inferred — no PYQ archive) → ANSWER FROM

| Likely question | Answer from |
|---|---|
| NIST definition; 5 characteristics; 3 service & 4 deployment models; compare | A1-A5 |
| Cloud vs traditional hosting/web app; challenges & shared responsibility | A6, G1 |
| Why virtualization; x86 challenges; four approaches (compare) | B1-B3 |
| CPU/memory overcommit (calculate, reclaim techniques) | B5 |
| Hypervisor types and goals; IaaS actors | C |
| AWS regions/AZ; IAM terms; EC2 lifecycle/tenancy; VPC; S3 vs EBS vs EFS | D |
| Containers vs VMs; namespaces/cgroups; Docker architecture & layers; Dockerfile | E |
| Orchestration; Swarm vs Kubernetes; K8s components; pod lifecycle | F |
| PaaS vs SaaS vs FaaS vs BaaS; tenancy models; SaaS economics | G |
| VM life cycle/provisioning; live migration (pre-copy, calculation); IaC; 6 R's | H |
| Capacity planning (calculation); VIM/OpenNebula; scheduling/leases; Amdahl; scaling | I |
| SLA/SLO/SLI; HA steps; SLA life cycle; admission control | J |
| Multi-tenancy levels/models/benefits/risks/data sharing | K |
| Monolith/microservices/serverless; 12-factor; API gateway; CI/CD; blue-green/canary; observability | L |
| Edge, AI on cloud, quantum, green cloud/PUE | M |
