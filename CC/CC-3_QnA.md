# Cloud Computing — 50 Scenario-Based Questions & Answers

## Preliminary Analysis (How this question bank was built)

**From the lecture slides:** Heavy emphasis on (a) virtualization mechanics — hypervisor types, x86 privilege rings, sensitive-but-unprivileged instructions, binary translation vs. paravirtualization vs. hardware-assisted, emulation; (b) CPU/memory overcommitment with ratios, CPU Ready Time, and reclamation techniques (page sharing, ballooning, compression, host swapping); (c) the 5-layer IaaS architecture stack and NIST actors; (d) AWS as IaaS reference — EC2 (instance types, AMIs, lifecycle, tenancy, placement groups, instance store), EBS/S3/EFS/Glacier, RDS Multi-AZ vs. Read Replicas, VPC, Regions/AZs/Local Zones/Wavelength/Outposts/Edge Locations, IAM (authentication vs. authorization, roles, policies), and shared responsibility.

**From the sample paper:** Questions demand *justification or no marks*, mix architecture comparison (SaaS vs. PaaS+IaaS), identification + justification (service/deployment models), Docker practicals (Dockerfile, volumes), demerits + mitigation reasoning, and numerical cost decisions (private expansion vs. public cloud). This bank mirrors those patterns — including one cost-calculation and one Dockerfile question.

---

# Section 1 — Medium (Questions 1–10)

### Q1. Choosing a service model for a startup

**Scenario:** A 6-person startup wants to launch a customer web application. It has no infrastructure engineers, expects highly unpredictable traffic, and wants the team to focus purely on writing code. The CTO is deciding between IaaS, PaaS, and SaaS. Which model fits best, and why?

**Answer:** PaaS is the most appropriate model. The team deploys application code onto a provider-managed runtime; the provider handles servers, OS patching, scaling, and capacity planning.

**Why:** The startup's constraints eliminate the alternatives. IaaS would force them to manage virtual machines, OS patching, and scaling — work no one on the team is staffed for. SaaS doesn't apply because they are *building* a custom application, not buying a finished product. PaaS sits exactly at their need: control over code, no control burden over infrastructure.

**Technical reasoning:** The IaaS–PaaS–SaaS spectrum is fundamentally about the division of operational responsibility. As you move up the stack, the provider absorbs more layers (IaaS: provider manages hardware + virtualization; PaaS: provider also manages OS, runtime, and scaling; SaaS: everything including the application itself). Rapid elasticity — a NIST essential characteristic — is delivered automatically in PaaS, which suits unpredictable traffic without capacity planning.

**Trade-offs:** PaaS reduces control: you accept the provider's runtime versions, scaling behaviour, and platform limits, and migration to another provider can be harder due to platform-specific bindings.

**Industry relevance:** Most early-stage startups deploy on managed platforms precisely to convert infrastructure operations into a pay-per-use, provider-managed service, matching the "focus on innovation, not infrastructure" motivation for cloud adoption.

**Key concepts tested:** Service model selection, responsibility split, elasticity, NIST characteristics.

---

### Q2. Deployment model for a regulated fintech

**Scenario:** A payments company must keep certain transaction data within its own data centre due to regulatory data-residency rules, but wants to burst its customer-facing web workload to the cloud during festival-season peaks. Which deployment model should it adopt, and why?

**Answer:** A **hybrid cloud** — private infrastructure for the regulated data, a public cloud for the elastic customer-facing tier, connected so workloads can burst between them.

**Why:** Pure public cloud violates the residency constraint; pure private cloud cannot provide cost-effective seasonal elasticity because the company would have to buy peak capacity that sits idle for most of the year. Hybrid captures the benefit of each.

**Technical reasoning:** NIST defines four deployment models: public, private, community, hybrid. Hybrid clouds bind private and public infrastructure so data and applications can move between them. The design pattern here is "steady-state sensitive workload on private, spiky stateless workload on public" — the private side handles compliance-critical components, while the public side absorbs demand that varies from 1 to 100.

**Trade-offs:** Hybrid increases architectural complexity — the company must manage two environments, consistent networking/identity across them, and integration between the tiers. Poorly integrated hybrids can end up with the costs of both models and the agility of neither.

**Industry relevance:** Banks, insurers, and healthcare providers commonly adopt this exact pattern: compliance-locked core systems on private infrastructure, elastic digital channels on public cloud.

**Key concepts tested:** Deployment models, cloud bursting, elasticity vs. compliance tension.

---

### Q3. Consolidating underutilised servers with virtualization

**Scenario:** An enterprise runs 30 business applications, each on its own physical server. Average CPU utilisation is under 20%, yet the company must keep every application isolated from the others. Infrastructure costs (power, cooling, floor space, hardware refresh) are rising. What technology should it adopt and how would it help? What new risks appear?

**Answer:** Adopt **server virtualization**: place a hypervisor (Type-1 bare-metal) on powerful physical hosts and run the 30 applications inside isolated VMs multiplexed onto the shared hardware.

**Why:** This is the classic consolidation case. The problem is not capacity — it is that one-app-per-server leaves 80% of resources idle. Virtualization partitions one physical machine into multiple logically isolated VMs, so isolation is preserved while utilisation rises sharply.

**Technical reasoning:** The hypervisor creates and runs virtual machines, presenting each guest OS with its own virtual CPU, memory, storage, and network. Three enabling properties apply: **partitioning** (resources divided among VMs, one VM unaware of others), **isolation** (failure in one VM doesn't affect others), and **encapsulation** (each VM state is a file, so backup/recovery is as simple as copying files). Fewer physical servers directly cut space, cooling, cabling, and capital expense.

**Trade-offs / new risks:** VMs need separate OS licenses, so uncontrolled creation causes *server sprawl* and license cost explosion. Staff need virtualization training. Powerful physical hosts are required, so there may be upfront hardware investment. Finally, consolidating many VMs onto one host concentrates failure risk — mitigate by spreading VMs across multiple hosts and planning capacity so one host failure doesn't take down every application.

**Key concepts tested:** Virtualization motivation, hypervisor, partitioning/isolation/encapsulation, virtualization limitations.

---

### Q4. Image vs. container misunderstanding

**Scenario:** A junior developer says: *"I modified the files inside my running Docker container, so the image is now updated — anyone who pulls this image will see my changes."* The team lead disagrees. Explain who is right, and clearly distinguish a Docker image from a Docker container.

**Answer:** The team lead is right. A **Docker image** is a read-only, immutable template composed of stacked layers (base OS, dependencies, application code). A **container** is a running instance of an image with one additional thin **writable layer** on top.

**Why the developer is wrong:** File modifications inside a running container are written to that container's private writable layer only. The underlying image layers are never altered. When the container is removed, the writable layer and its changes are discarded. Other machines pulling the image see the original image, not the container's state.

**Technical reasoning:** `docker run` instantiates a container *from* an image; one image can spawn many identical containers. To persist changes, the developer must either **commit** the container to a new image and push it to a registry, or externalise state into **volumes**. This separation is what makes images portable and reproducible: the same image produces consistent behaviour on every host, which is the core value of containerization.

**Trade-offs:** The immutability model requires discipline — state (databases, uploads) must go into volumes, and configuration into environment variables, or changes will silently vanish on restart.

**Industry relevance:** This image/container distinction underpins production practice: teams build versioned images once, push to a registry, and run identical containers across dev, test, and production.

**Key concepts tested:** Docker image vs. container, layers, writable layer, persistence.

---

### Q5. Selecting an AWS Region

**Scenario:** An Indian fintech is deploying its trading platform on AWS. Its users are almost entirely in India, and regulations require customer transaction data to be stored in India. Which Region should it choose as primary, what other factors should influence the choice, and how should it plan for regional failure?

**Answer:** Choose the **Mumbai Region (ap-south-1)** as primary. Selection should weigh latency to users, data-residency/compliance requirements, cost differences between Regions, and availability of required services. Plan a **secondary Region** (e.g., Hyderabad ap-south-2) for disaster recovery.

**Why:** Latency is driven by physical distance — an Indian user base gets the best response times from the nearest Region. Residency law makes this choice mandatory rather than optional for regulated data.

**Technical reasoning:** Each AWS Region is a separate geographic area that is *completely independent* of other Regions — resources created in one Region exist only there. Any service you launch runs in your current default Region (changeable via console, `AWS_DEFAULT_REGION`, or `--region`). Region-level isolation gives the greatest fault tolerance, so a second Region provides insurance against an outage of the entire primary Region.

**Trade-offs:** Running in two Regions doubles some costs and requires data replication between them; replication of large state across Regions has real latency and bandwidth cost.

**Industry relevance:** Multi-Region design is standard for financial workloads: primary Region for live traffic, warm or cold secondary Region for DR, with regulated data confined to in-country Regions.

**Key concepts tested:** Regions, Region independence, Region selection criteria, DR planning.

---

### Q6. Object vs. block storage for a photo-sharing app

**Scenario:** A photo-sharing app must store 50 million user-uploaded photos that will be retrieved over the web by millions of users. The team is deciding between Amazon S3 and Amazon EBS. Which is appropriate for the photos, which for the application's database, and why?

**Answer:** Store the photos in **Amazon S3** (object storage) and run the relational database on **EBS-backed** storage (block storage).

**Why:** The photos are independent, immutable objects accessed over HTTP by many consumers — exactly the object-storage pattern. The database needs a low-latency block device attached to its EC2 instance with file-system semantics — exactly EBS.

**Technical reasoning:** S3 is server-independent object storage accessed over the Internet via an API built on standard HTTP verbs (PUT/GET whole objects). Buckets hold unlimited objects identified by unique keys, storage is paid per use with no capacity planning, and objects are automatically replicated across multiple devices and facilities within a Region. EBS, by contrast, is persistent *block* storage attached to EC2 instances — replicated within its Availability Zone, sized in volumes (1 GB–16 TB for general-purpose SSD), and only attachable to one instance at a time. Using EBS for 50M photos would mean capacity planning, per-volume limits, and no direct web access.

**Trade-offs:** S3 is slower than EBS and you operate on whole objects — no incremental in-place updates; that's irrelevant for photos but disqualifies it as a database disk.

**Key concepts tested:** Object vs. block storage, S3 properties, EBS properties, workload matching.

---

### Q7. Unexpected bill from "stopped" instances

**Scenario:** A company's finance team notices an AWS bill larger than expected. Investigation shows several EC2 instances are *stopped*, yet charges continue. Two specific causes are found: (1) EBS volumes attached to stopped instances, and (2) Elastic IPs that were released from instances but still reserved. Explain why these continue to bill.

**Answer:** Stopping an instance stops **EC2 compute charges only** — storage and address resources remain billable. EBS volumes persist (by design, so your data and configuration survive a stop) and are charged per GB-month. An Elastic IP that is **not attached to a running instance is still billed hourly** — this is intentional, to discourage hoarding scarce public IPv4 addresses.

**Why:** The billing model follows resource lifecycle, not instance lifecycle. When an instance stops: compute charges = ₹0, EBS volume charges continue, EIP unattached charges continue. When the instance *terminates*, compute is gone permanently, and EBS volumes persist until explicitly deleted.

**Technical reasoning:** This reflects the EC2 lifecycle states — Launching → Running → Stopped → Terminated — where "Stopped" means the instance can be restarted with its disk intact, so the disk must be kept and paid for. AWS addresses and DNS names follow similar logic: public IPs exist only while the instance runs and cannot be transferred, while Elastic IPs are static addresses you reserve independently.

**Trade-offs / lesson:** Cost governance must cover non-compute resources. Set tagging (up to 50 tags per instance, useful for department/cost attribution) and CloudWatch-based monitoring/alerting to catch idle volumes and unattached EIPs.

**Key concepts tested:** EC2 lifecycle and billing, EBS persistence, Elastic IP billing, cost management.

---

### Q8. Type-1 vs. Type-2 hypervisor selection

**Scenario:** A university wants students to run a second OS on their personal laptops for a lab course. The same university's research group needs to host production VMs for a shared cluster with strong performance and isolation. Which hypervisor type fits each situation and why?

**Answer:** Laptops → **Type-2 (hosted) hypervisor** such as Oracle VirtualBox. Production cluster → **Type-1 (bare-metal) hypervisor** such as ESXi, KVM, or Hyper-V.

**Why (laptops):** A Type-2 hypervisor runs as an application on top of the host OS. It is easy to install, needs no dedicated hardware access, and leverages the host OS's device drivers — ideal for desktop/lab use where convenience outweighs peak performance.

**Why (production):** A Type-1 hypervisor runs directly on physical hardware with no host OS underneath. This gives higher performance (no extra OS layer), stronger isolation, and direct control of CPU/memory/IO virtualization — the design used in cloud data centres (the IaaS virtualization layer is explicitly a bare-metal hypervisor such as KVM, Xen, or AWS Nitro).

**Trade-offs:** Type-2 depends on the stability and security of the host OS and adds overhead; Type-1 is harder to set up on a desktop and less convenient for students. The trade is manageability/convenience vs. performance/isolation.

**Key concepts tested:** Hypervisor types, bare-metal vs. hosted architecture, use-case matching.

---

### Q9. Why a container registry is needed in CI/CD

**Scenario:** A team builds Docker images in its CI pipeline on a build server, then manually copies the image files to three production servers whenever a release happens. Releases are inconsistent — sometimes servers run slightly different builds. What infrastructure element is missing, and how does it solve the problem?

**Answer:** A **container registry** — a centralised service that stores and distributes versioned images (Docker Hub for public, or a private registry for proprietary images). The CI pipeline should `push` each build to the registry with a unique tag; every server `pull`s the exact same tagged image.

**Why this solves the problem:** The current flow has no single source of truth — copied files can diverge, and there's no record of which build is on which server. A registry makes the image the immutable artefact of record: the CI pipeline publishes it once, and all environments pull the identical digest.

**Technical reasoning:** Registries provide image storage, versioning via tags/digests, and access control (private registries restrict who can push/pull — essential for proprietary code). They integrate naturally into continuous delivery: the deployment step becomes "run image version X," making rollbacks as simple as re-running a previous tag. This is the same distribution model AWS applies to VMs with AMIs and AWS Marketplace.

**Trade-offs:** The registry becomes critical infrastructure — it must be available and secured; untagged/old images accumulate and need pruning.

**Key concepts tested:** Container registries, image versioning, CD pipeline integration.

---

### Q10. "Access Denied" after successful login

**Scenario:** An intern successfully logs into the cloud console with username, password, and MFA, but receives "Access Denied" when attempting to terminate an EC2 instance. The intern's manager says the system is misconfigured. Is it? Explain the two distinct security mechanisms at play.

**Answer:** No — the system is working correctly. **Authentication** succeeded (the intern's identity was verified), but **authorization** failed: the intern's attached policy does not grant `ec2:TerminateInstance` on that resource, so the request is denied.

**Why:** These are separate questions. Authentication answers *"Who are you?"* — credentials (password + MFA, SSH keys, temporary tokens, SSO) are checked against a trusted store, producing a confirmed identity with no permissions decided yet. Authorization answers *"What are you allowed to do?"* — on every request, policies that explicitly allow or deny specific actions on specific resources are evaluated.

**Technical reasoning:** Policies are the enforceable rulebook: each statement specifies Effect (Allow/Deny), Action, Resource, and optionally Condition. An explicit Deny always overrides an Allow. Unlike authentication (once per session), authorization is evaluated continuously on every API call. This is **least privilege** functioning as designed — each identity gets only the access it needs.

**Industry context:** This pattern (verified identity, denied action) is the single most common security misunderstanding in cloud operations; least privilege intentionally makes unauthorized actions fail even for logged-in users.

**Key concepts tested:** Authentication vs. authorization, policies, least privilege.

---

# Section 2 — Medium-High (Questions 11–35)

### Q11. ERP modernization: SaaS vs. PaaS+IaaS

**Scenario:** A logistics company is modernizing its ERP. **Option A:** a SaaS ERP suite for inventory, billing, and HR with minimal customization. **Option B:** a PaaS-based custom application combined with IaaS-hosted VMs for its legacy warehouse-robot control software.
a) Compare the options on scalability, customization, and operational control.
b) When is each preferred?

**Answer:**

| Factor | Option A (SaaS) | Option B (PaaS + IaaS) |
|---|---|---|
| Scalability | Provider-managed, instant, elastic | Achievable but the company must design and operate scaling |
| Customization | Limited to configuration; cannot change core code | Full — bespoke workflows, custom integrations |
| Operational control | Minimal — provider runs everything | High — control over runtime (PaaS) and full OS/network control (IaaS) |

**b) When A is preferred:** Business processes match standard ERP workflows, the company wants rapid adoption with minimal IT staff, and standardisation itself is a benefit (vendor-maintained upgrades). **When B is preferred:** The company has genuinely unique logistics workflows, must integrate tightly with legacy/industrial systems (the robot-control software has no SaaS equivalent and needs full VM control on IaaS), or faces compliance demands requiring control over the stack.

**Why:** The comparison is the responsibility spectrum. SaaS trades control and customization for near-zero operations; PaaS+IaaS buys control and flexibility at the cost of operating more of the stack. The realistic answer for many enterprises is *both* — the sample of Option B shows IaaS is often retained precisely for legacy components that cannot be SaaS-ified.

**Key concepts tested:** Service model comparison, scalability/customization/control trade-off, legacy system constraints.

---

### Q12. Identifying service and deployment models

**Scenario:** An insurance firm uses a browser-based cloud CRM for its sales team (the firm configures it but never touches servers, OS, or application code). Its IT team separately provisions cloud virtual machines on which it installs and manages custom actuarial applications. Identify the service model in each situation and the overall deployment model, with justification.

**Answer:** The CRM is **SaaS**: the consumer uses the finished application over the network; the provider manages servers, storage, networking, OS, middleware, runtime, data, *and* the application. The actuarial apps on VMs are **IaaS**: the consumer manages the OS and above (applications, data, access policy) while the provider supplies virtualized compute, storage, and network. The overall deployment model is **public cloud** — both services are consumed from a provider's shared, multi-tenant infrastructure and offered to the general customer base. If the firm additionally connects its own on-premises data centre and spans workloads across both, the architecture becomes **hybrid**.

**Why:** The classification criterion is *who manages which layer*. "Accessed via a browser with no management" is the SaaS signature; "IT team installs and manages applications on virtual servers" is the IaaS signature. Deployment model is about *whose infrastructure* serves the workload — a shared commercial provider means public.

**Trade-offs / caution:** Students often mislabel browser-accessed custom apps as SaaS; the test is whether the *consumer* develops/maintains the application. Here the firm only configures the vendor's product → SaaS; it builds its own apps → those are not SaaS.

**Key concepts tested:** SaaS vs. IaaS identification, deployment model determination, justification discipline.

---

### Q13. Slow Docker builds — layer caching

**Scenario:** A team's Dockerfile copies the entire application source first, then runs the dependency installation. Because application code changes hourly while dependencies change monthly, every code change triggers a full multi-minute rebuild. Diagnose the problem and rewrite the instruction order, explaining why it works.

**Answer:** The problem is **layer-cache invalidation order**. Each Dockerfile instruction creates a layer; when an instruction's inputs change, that layer *and every layer after it* are rebuilt. By copying all source code first, every code edit invalidates the COPY layer, forcing the expensive dependency-install layer to re-run.

**Fix:**
```dockerfile
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
```
Copy only the dependency manifest first, install dependencies, *then* copy the application code.

**Technical reasoning:** Dependency manifests (`requirements.txt`, `package.json`) change far less often than source code. Placing `COPY <manifest>` + install before `COPY . .` means code edits invalidate only the final COPY layer — the dependency layer stays cached and the rebuild takes seconds. This is the standard production Dockerfile pattern and was implicitly tested in the sample paper's Dockerfile question (copy `package.json`, install, then copy remaining code).

**Trade-offs:** Requires disciplined file ordering; very occasional dependency updates will still rebuild that layer — which is correct and desirable (dependencies must refresh when the manifest changes).

**Key concepts tested:** Docker image layers, build cache, Dockerfile ordering.

---

### Q14. Write a Dockerfile (practical task)

**Scenario:** Write a Dockerfile for a Python web application that requires Python 3.11, is based on the official Ubuntu 22.04 image, has working directory `/usr/src/app`, copies `requirements.txt` first and installs dependencies, then copies the remaining application code, and exposes port 5000.

**Answer:**
```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python3", "app.py"]
```

**Explanation of each requirement:** `FROM` selects the base image (Ubuntu 22.04 as specified). `RUN` installs the required Python 3.11 toolchain at build time. `WORKDIR` sets the working directory (and creates it if absent) for all subsequent instructions. `COPY requirements.txt .` is deliberately placed *before* copying the rest of the code so the dependency layer stays cached (see Q13). `RUN pip3 install` installs the pinned dependencies into the image. `COPY . .` brings in the application source. `EXPOSE 5000` documents the listening port. `CMD` sets the default container start command (build/run commands are not required per the task, and `CMD` is the image's runtime default, not a build step).

**Why this ordering matters:** dependency installation precedes bulk code copy so routine code changes don't reinstall dependencies.

**Key concepts tested:** Dockerfile instructions (FROM, RUN, WORKDIR, COPY, EXPOSE, CMD), layer order, port declaration.

---

### Q15. Docker volumes for a production database container

**Scenario:** A company runs its MySQL database in a Docker container in production. During an incident, an operator removes and recreates the container — and all customer data disappears. Explain what happened, what role Docker volumes play for container restarts, and how volumes affect portability and efficiency across multiple production containers.

**Answer:** The data lived in the container's **writable layer**, which is destroyed when the container is removed — recreating it starts from the pristine image. **Docker volumes** solve this: a volume is a persistent storage area managed by Docker that exists *independent of the container lifecycle*. Mounting a volume at `/var/lib/mysql` means the database files survive container restart, stop, and removal.

**Role for restarts:** On restart, the container re-attaches to the same volume, so state (data, logs) persists exactly as the AMI/volume model persists VM state beyond instance lifecycle. Without a volume, "restart-safe" is impossible for any stateful container.

**Impact on portability and efficiency:** *Portability* — volumes decouple data from containers, so containers remain ephemeral and replaceable (the cloud-native model), while data is managed explicitly (backed up, snapshotted, migrated). In multi-container production, volumes also let several containers share common data where needed. *Efficiency* — writes to volumes bypass the container's copy-on-write filesystem overlay, avoiding its performance overhead for I/O-heavy workloads like databases.

**Trade-offs:** Volumes introduce their own lifecycle (backup, retention, access control), and bind-mounted host paths can reduce portability if they encode host-specific layout.

**Key concepts tested:** Container writable layer ephemerality, volumes, stateful containers, production data management.

---

### Q16. VMs vs. containers for 100 microservices

**Scenario:** A SaaS company is deploying 100 small, independent microservices. The platform team is deciding between one VM per service and containers on a shared VM fleet. Compare the two approaches and recommend one.

**Answer:** **Containers** are the appropriate choice for 100 lightweight microservices.

| Factor | VM | Container |
|---|---|---|
| Isolation | Strong — separate guest OS kernel per VM | Weaker — containers share the host kernel |
| Overhead | High — each VM carries a full OS (GBs) | Low — MBs; only the app + libraries |
| Startup | Minutes (boot a whole OS) | Seconds (no OS boot) |
| Density | Dozens per host | Hundreds per host |
| Use case | Strong isolation needs, different OSes, legacy apps | Many small, frequently deployed services |

**Why:** 100 services × full guest OS would multiply memory, boot time, patching surface, and licensing cost enormously. Containers share the host kernel, so each service adds only megabytes, starts in seconds, and enables the rapid, independent deployments microservices require.

**Trade-offs:** Container isolation is process-level, not hardware-level; a kernel vulnerability could theoretically affect containers sharing that kernel. Mitigate with hardened hosts, least-privilege settings, and keeping genuinely high-risk or different-OS workloads in VMs. A very common production answer is **containers *on* VMs**: VMs provide the security boundary, containers provide density and agility.

**Key concepts tested:** VM vs. container comparison, kernel sharing, microservice deployment economics.

---

### Q17. Why containers start instantly — and what that implies

**Scenario:** A developer notices that a Docker container starts in ~2 seconds while a VM takes ~2 minutes, and asks: *"Where does the OS come from in a container if it starts so fast?"* Explain the mechanism, and distinguish application containers from system containers.

**Answer:** A container does **not boot an OS at all — it shares the host machine's kernel**. The container image carries user-space only: the application plus its libraries (a base image like Ubuntu provides the file system, tools, and libraries, but *not* a kernel). Startup therefore involves creating isolated processes via namespaces/cgroups, not booting a kernel, which is why it takes seconds instead of minutes. In a VM, by contrast, the guest OS is a real OS with its own kernel that must be fully bootstrapped on virtual hardware — hence the minutes.

**Implications:** (1) Kernel sharing explains both the efficiency (no duplicated kernel, tiny images) and the security boundary difference (container escape = kernel-level concern; VM kernels are separate). (2) Containers cannot run a *different* kernel — e.g., Linux containers on a Windows host require a Linux VM layer beneath, and Linux containers cannot natively run Windows applications.

**Application vs. system containers:** *Application containers* (the Docker model) package a single application/process, are short-lived, and are ideal for microservices. *System containers* (e.g., LXC-style) behave more like lightweight VMs — they run an init system and multiple services inside a more VM-like environment sharing the host kernel.

**Key concepts tested:** Kernel sharing, image contents, startup mechanics, application vs. system containers.

---

### Q18. How containers achieve isolation — namespaces

**Scenario:** A developer runs `ps` inside a container and sees *only* the container's own processes, even though the host is running dozens of other containers. Another developer notes the container has its own hostname, network interfaces, and root filesystem. Which mechanism produces each of these views?

**Answer:** **Linux namespaces.** Each namespace type isolates one *view* of the system: **PID** (the container sees only its own process tree — hence the `ps` result), **NET** (its own network stack: interfaces, IPs, routing), **MNT** (its own filesystem mount tree and root), **UTS** (its own hostname), **IPC** (isolated inter-process communication), and **USER** (mapped user identities). Docker creates a fresh set of these namespaces per container, so every container perceives itself as a standalone machine.

**Why this matters:** Namespaces provide *isolation of visibility*, which is what makes many containers coexist safely and appear independent to their applications — the illusion each microservice needs.

**Critical distinction:** Namespaces do **not** limit resource consumption. A container with isolated views can still consume all host CPU/memory. Resource limits are the job of **cgroups** (control groups). If asked "how do we stop a container from eating all the RAM?", the answer is cgroups, not namespaces — conflating the two is the classic exam trap.

**Key concepts tested:** Namespace types, view isolation vs. resource control distinction.

---

### Q19. A container starving its neighbours — cgroups

**Scenario:** On a shared production host, one container periodically consumes 100% CPU and most memory, degrading every other container. The ops team proposes "adding more namespaces to fix it." Evaluate the proposal and prescribe the correct mechanism.

**Answer:** The proposal is wrong — the fix is **cgroups (control groups)**. Namespaces control *what a container can see*; they say nothing about *how much it can use*. cgroups are the kernel mechanism that limits, accounts for, and prioritises resource usage (CPU time, memory, block I/O) per group of processes.

**Prescription:** Apply resource constraints per container — e.g., `docker run --cpus=1.5 --memory=2g` — which places the container's processes in cgroups with a CPU quota and memory ceiling. A memory limit also triggers controlled behaviour: the kernel kills (OOM) a container exceeding its limit *before* it can destabilise the host.

**Technical reasoning:** Without cgroup limits, the Linux scheduler divides CPU by demand, so a single demanding process naturally starves others. With limits, each container gets a bounded share, making performance predictable for multi-tenant hosts — the same principle a hypervisor applies when partitioning physical resources among VMs.

**Trade-offs:** Limits must be tuned: too low and containers are throttled or OOM-killed under legitimate load; too high and protection is illusory. Combine namespaces (isolation) + cgroups (resource control) for the complete container model.

**Key concepts tested:** cgroups, noisy-neighbour problem, resource management, namespace/cgroup division of labour.

---

### Q20. From scripts to Kubernetes

**Scenario:** A team runs 40 Docker containers across 5 servers using shell scripts. When a server fails, nobody notices until customers complain; scaling means manually starting containers; rolling out updates involves downtime. The team hears Kubernetes is "a container." Correct the misconception and explain what value Kubernetes adds.

**Answer:** Kubernetes is **not a container** — it is a **container orchestrator**: a control system that manages * fleets* of containers across a cluster of hosts.

**Value it adds to this team's exact pain points:** (1) **Declarative desired state** — you declare "I want 6 replicas of service X" and the system continuously reconciles reality to match. (2) **Self-healing** — it detects failed containers/nodes and automatically restarts or reschedules them elsewhere, directly fixing the "nobody notices until customers complain" failure mode. (3) **Automated scheduling** — it decides which host runs each container based on resource requests, replacing manual placement. (4) **Horizontal scaling** — replicas scale up/down on demand without scripts. (5) **Service discovery and load balancing** — stable virtual endpoints distribute traffic across replicas. (6) **Rolling updates and rollbacks** — new versions replace old ones incrementally with no downtime.

**Technical reasoning:** This is the container-era analogue of the IaaS orchestration layer: the scheduler, capacity manager, and lifecycle management that turn pooled resources into a managed service.

**Trade-offs:** Kubernetes adds significant operational complexity and a learning curve; for a handful of containers it can be overkill. It earns its cost at the scale and reliability requirements this team already has.

**Key concepts tested:** Kubernetes role, orchestration, declarative state, self-healing, misconception correction.

---

### Q21. Designing a cloud-native application

**Scenario:** A bank's monolithic core app is tightly coupled, deployed as one unit, manually scaled by adding whole-server capacity, and cannot tolerate frequent releases. The bank wants to redesign it cloud-native. Which design principles apply, and what should change?

**Answer:** Apply the core cloud-native principles: **microservices decomposition** (independent, loosely coupled services), **containerization** (each service packaged as an immutable image), **orchestration** (Kubernetes managing deployment, scaling, self-healing), **statelessness** (session/state externalised so any replica can serve any request), **declarative APIs and configuration**, **CI/CD automation**, and **design for failure** (health checks, retries, replication).

**What changes concretely:** The single deployable unit becomes dozens of independently deployable services; scaling changes from "clone the whole server" to "add replicas of only the hot service"; releases become frequent, small, and reversible; failures are assumed and contained per-service instead of taking down the monolith.

**Why this fits the cloud:** These principles exploit cloud characteristics — elasticity (scale only what's hot), resource pooling, and on-demand provisioning. Statelessness in particular is what makes horizontal scaling and self-healing safe: any instance can be killed and replaced without losing user sessions.

**Trade-offs:** Microservices introduce distributed-systems complexity — network latency between services, eventual consistency, harder debugging, and operational maturity requirements (monitoring, orchestration). Cloud-native is an architecture decision, not a lift-and-shift; the migration should be phased (see Q47).

**Key concepts tested:** Cloud-native design principles, statelessness, microservices, design-for-failure.

---

### Q22. Why x86 virtualization was "impossible" — rings and sensitive instructions

**Scenario:** A cloud architect is explaining to new hires why a standard hypervisor technique that worked on IBM mainframes failed on x86 CPUs for years, and why x86 virtualization only became practical in 1998. Present the explanation.

**Answer:** Two properties of x86 combined to break the standard **trap-and-emulate** technique:

1. **Ring conflict.** x86 defines four privilege rings (0–3). The OS kernel runs in Ring 0 (most privileged, full hardware access); user applications run in Ring 3 (least privileged) and reach hardware only via system calls. A hypervisor must sit *underneath* the OS to intercept privileged operations — but the OS expects Ring 0 for itself. There is no natural room for both.

2. **The fatal subtlety — sensitive-but-unprivileged instructions.** A hypervisor normally handles privileged instructions by *trapping* them and emulating the result. But x86 has a set (~17) of instructions that are *sensitive* (behave differently depending on ring) yet *not privileged* — executed outside Ring 0, they **do not trap**; they fail silently with no error the hypervisor can catch. This violates the Popek–Goldberg virtualizability requirement, so the standard technique had a real gap on x86 specifically — not the ring numbering, but the untrappable instructions.

**Resolution:** VMware (1998/99) made it practical via **binary translation**, rewriting the sensitive instruction stream at runtime so the guest OS could run deprivileged (Ring 1) without knowing it — full virtualization. Later, **paravirtualization** (modified guest OS issues hypercalls) and **hardware-assisted virtualization** (Intel VT-x / AMD-V, 2005–06, adding a privilege level *below* Ring 0 so the CPU itself traps the instructions) closed the gap cleanly.

**Key concepts tested:** x86 rings, privileged vs. sensitive instructions, trap-and-emulate, why virtualization history matters.

---

### Q23. Choosing among the three virtualization generations

**Scenario:** A company runs an older x86 fleet whose CPUs lack Intel VT-x/AMD-V. It wants consolidation, and ~80% of its guests are Linux; the rest are Windows. Compare binary translation (full virtualization), paravirtualization, and hardware-assisted virtualization for this fleet, and recommend a strategy.

**Answer:**

| Dimension | Binary translation (full virt.) | Paravirtualization | Hardware-assisted |
|---|---|---|---|
| Guest OS | Unmodified | Kernel modified (hypercalls) | Unmodified |
| CPU requirement | None — works on old x86 | None, but needs modifiable kernel source | Requires VT-x/AMD-V (2005–06+) |
| Performance | Moderate overhead; translation cost scales with kernel activity | Near-native for CPU-bound work | Best; CPU traps in silicon |
| Practical constraint | Complex engineering; same-ISA only | Historically open-source guests only (Linux, BSD) — Windows not modifiable | None once hardware is new |

**Recommendation:** For the **Linux fleet on existing hardware**, use **paravirtualization** — the kernels can be modified to issue hypercalls, giving near-native performance without new CPUs. For the **Windows guests**, paravirtualization is not viable (closed-source kernel), so use **binary translation** in the interim. Strategically, plan a **hardware refresh to VT-x/AMD-V-capable CPUs**, because hardware-assisted virtualization gives unmodified guests of any OS with the lowest overhead and is the third generation that the whole industry (KVM, Xen HVM, Hyper-V, modern ESXi) standardised on.

**Why:** Each generation made a different trade: software workaround (BT) → guest cooperation (PV) → silicon support (HW-assist). The migration path follows exactly that evolution.

**Key concepts tested:** Three virtualization generations, guest modification, performance trade-offs, ISA constraint.

---

### Q24. Emulation vs. virtualization for cross-platform testing

**Scenario:** A QA team must test an ARM-compiled application but only has x86 laptops. Someone suggests "just virtualize an ARM VM." Explain why true virtualization cannot do this, what technology can, and its cost.

**Answer:** Full virtualization requires the guest and host to **share the same ISA** — it exploits the CPU executing most guest instructions natively and only traps sensitive ones. An ARM guest on an x86 host violates that premise: there is no ARM CPU underneath to run the instructions. The required technology is **emulation**: software reproduces the behaviour of an entirely different machine — CPU instruction set, memory, and I/O — presenting the guest with a virtual hardware profile completely decoupled from the physical host. The guest runs as an ordinary user-mode process on the host, unaware it isn't on real hardware.

**Mechanisms and cost:** Emulation is implemented by **interpretation** (fetch/decode/execute each guest instruction every time — simple, expensive) or **dynamic binary translation** (translate blocks of guest instructions to host instructions once, cache, and reuse — what QEMU's TCG and Apple's Rosetta 2 use). Even DBT-based emulation carries heavy overhead versus native execution — typically **10–100× slower** — so emulation is reserved for cases where ISAs genuinely differ (cross-platform dev/test, legacy hardware support), not for same-ISA cloud virtualization, where trap-and-emulate/hardware-assisted dominates.

**Recommendation:** Accept the performance hit for functional QA testing on emulated ARM (QEMU/Bochs class); never use emulation for production-scale workloads where a native or virtualized path exists.

**Key concepts tested:** Emulation vs. virtualization, ISA constraint, interpretation vs. DBT, performance overhead.

---

### Q25. CPU overcommitment calculation and assessment

**Scenario:** A host has 8 physical cores. An admin deploys 10 VMs, each configured with 4 vCPUs. Compute the overcommit ratio, assess whether it is safe, and state what metric should be watched and the rules of thumb.

**Answer:**
- **Total vCPUs** = 10 × 4 = 40
- **Overcommit ratio** = 40 ÷ 8 = **5:1**

**Assessment:** 5:1 sits at the *upper edge* of the workable zone for **mixed workloads**. Guidance: **1:1** for latency-sensitive workloads (databases, real-time); **3:1–5:1** workable for mixed workloads; **>8:1** high risk by most practitioners. So 5:1 is *acceptable but worth watching* — and risky if any of these VMs is latency-sensitive.

**Metric to watch: CPU Ready Time** — the time a vCPU is ready to run but waiting for a physical core to free up. Rising CPU Ready (commonly flagged above ~5%) signals the host is oversubscribed beyond what current workloads tolerate.

**Why overcommitment works at all:** CPU time is *time-sliced* — the hypervisor scheduler hands out short slices of physical core time exactly as an OS multiplexes processes onto cores, so more vCPUs than pCPUs can be allocated because not all VMs peak simultaneously.

**Remediation if Ready Time climbs:** reduce vCPU counts (right-size), move latency-sensitive DBs toward 1:1 placement, distribute VMs across more hosts, or add capacity.

**Key concepts tested:** Overcommit ratio, CPU Ready Time, workload-dependent safe zones, scheduler time-slicing.

---

### Q26. Memory overcommitment and reclamation techniques

**Scenario:** A host has 128 GB physical RAM. 24 VMs are configured with 8 GB each. Compute the overcommit ratio, and explain the sequence of techniques the hypervisor uses when actual demand exceeds physical memory, in order from least to most invasive.

**Answer:**
- **Configured memory** = 24 × 8 = 192 GB
- **Ratio** = 192 ÷ 128 = **1.5:1**

**Key difference from CPU:** memory *cannot* be time-sliced — a page is either allocated to a VM or it isn't. The hypervisor cannot schedule its way out of a shortfall; it must actively reclaim or deduplicate memory.

**Reclamation sequence (least → most invasive):**
1. **Page sharing (TPS/KSM):** merges identical memory pages (same OS images make many pages identical). Cheapest reclaim; since vSphere 6.0, inter-VM sharing is disabled by default for security — only intra-VM sharing stays on unless re-enabled.
2. **Ballooning:** a guest-side driver asks the guest OS to give back memory; the *guest* decides which of its own processes/pages to evict — the most cooperative and efficient forced reclaim (see Q27).
3. **Compression:** the hypervisor compresses pages instead of writing them to disk — slower than RAM, far faster than disk.
4. **Host-level swapping (last resort):** pages swapped to disk without guest cooperation — disk is roughly **100,000× slower than RAM**, causing sharp latency spikes.

**Assessment:** 1.5:1 is a moderate ratio; it is safe *provided* actual active working sets stay below physical RAM. The ratio alone is not the risk signal — actual peak demand is (see the sample-paper-style logic in Q44's CPU counterpart).

**Key concepts tested:** Memory overcommit ratio, reclamation technique ordering, why memory differs from CPU.

---

### Q27. Ballooning in depth — and what happens without it

**Scenario:** Memory overcommitment is enabled on a host under memory pressure. An operator observes that one VM's guest OS reports plenty of free memory internally, yet the host is reclaiming from it inefficiently and falling back to host-level swapping, causing latency spikes. Diagnose the likely cause and explain how ballooning is supposed to work.

**Answer:** **Likely cause: the balloon driver (guest tools/agent) is not installed or not running inside that VM.** Without it, the hypervisor cannot cooperate with the guest OS and must escalate to compression and ultimately host-level swapping.

**How ballooning works:** A guest-side driver inflates a "balloon" by allocating memory *inside the guest*. This forces the **guest OS itself** to decide which of its own processes' pages to evict to its own swap/page file — precisely the pages it can least afford to lose (or the ones already idle, like the free memory the operator observed). The balloon's pages are then reclaimed by the hypervisor and given to other VMs. Deflating the balloon returns memory to the guest under its own control.

**Why it's the right mechanism:** The guest OS has perfect knowledge of which memory is truly idle versus active; the hypervisor does not. Ballooning converts a blind host-side reclaim into a cooperative, guest-informed one — which is why it precedes compression and host swapping in invasiveness.

**Escalation consequence:** Without ballooning, the host treats the VM's memory as opaque, compression only delays the problem, and **host-level swapping** — disk, ~100,000× slower than RAM — produces exactly the observed sharp latency spikes.

**Fix:** Install/verify guest tools with the balloon driver across the fleet; monitor swap and ballooning activity as capacity-planning signals.

**Key concepts tested:** Ballooning mechanics, guest cooperation, reclamation escalation, diagnosis.

---

### Q28. Demerits of a virtualization technique and their mitigation (sample-paper pattern)

**Scenario:** *(As in the sample paper:)* Pick **one** virtualization technique — full virtualization with VMs **or** software virtualization using containers. State its **two primary demerits** in cloud computing with brief justification, and how each can be mitigated.

**Answer (chosen: full virtualization with VMs):**

**Demerit 1 — Performance overhead.** The hypervisor must intercept and translate the sensitive x86 instructions that don't trap naturally outside Ring 0, plus emulate virtual devices in software. This translation cost scales with kernel activity, so I/O- and context-switch-heavy workloads suffer more than compute-bound ones. **Mitigation:** use hardware-assisted virtualization (Intel VT-x/AMD-V) so the CPU itself traps sensitive instructions in silicon, and use paravirtualized drivers for I/O devices — this is precisely why modern hypervisors combine both.

**Demerit 2 — Resource and operational overhead per VM (heavyweight guests).** Every VM carries a complete guest OS — gigabytes of RAM and disk, minutes of boot time, and a full patching/licensing surface per VM. In a cloud, this inflates cost per workload and enables *server sprawl* (each VM needs its own software license, so uncontrolled VM creation explodes license fees). **Mitigation:** right-size VMs and enforce governance on creation; apply memory overcommitment techniques (page sharing, ballooning, compression) to raise density; and where workloads are lightweight and homogeneous, shift them to **containers**, which share the host kernel and carry only MBs of overhead.

**Why this answer shape:** the sample paper explicitly rewards *demerit + justification + mitigation* — not a list of disadvantages without remedies.

**Key concepts tested:** Full virtualization overheads, mitigation via hardware assist/PV drivers, sprawl, VM vs. container economics.

---

### Q29. Mapping requirements onto the IaaS architecture stack

**Scenario:** A provider is building an IaaS platform. For each requirement, identify the layer of the IaaS architecture stack that fulfils it, and name the cross-cutting concerns involved:
(a) Customers order and manage VMs via a web portal and REST APIs.
(b) A scheduler provisions, migrates, and terminates VMs, metering usage for billing.
(c) vCPUs, virtual disks, and virtual NICs are aggregated into flexible compute/storage/network pools.
(d) A bare-metal hypervisor abstracts physical resources into virtual machines.

**Answer:**
- (a) **Layer 5 — Service/Access Layer (consumer touchpoint):** self-service portal, REST APIs, CLI, SDKs — the only layer consumers interact with directly.
- (b) **Layer 4 — Core Management/Orchestration Layer (cloud manager):** VM lifecycle management (provision/migrate/terminate), scheduler, capacity manager, SLA monitor, and metering & billing engine.
- (c) **Layer 3 — Resource Pool/Abstraction Layer:** pooled virtual resources (compute pool of vCPUs, storage pool of vDisks, network pool of vNICs) enabling flexible, physical-location-independent allocation.
- (d) **Layer 2 — Virtualization Layer:** the hypervisor (Type-1 bare metal — KVM, Xen, AWS Nitro) abstracting physical CPU/RAM/disk/NIC into isolated execution environments. (Layer 1 beneath it is the physical infrastructure: servers, storage arrays, network, power, cooling.)

**Cross-cutting concerns (span every layer):** IAM (identity & access management), network security (VPC, firewalls, security groups), monitoring & logging (metrics, logs, alerts), and billing & chargeback. The stack's flow: users request via the service layer → orchestration allocates from resource pools → hypervisor provisions on physical infrastructure → cross-cutting services assure security, visibility, and cost control at every layer.

**Key concepts tested:** IaaS reference architecture layers, orchestration, cross-cutting concerns.

---

### Q30. IaaS actors in a multi-cloud engagement

**Scenario:** An enterprise adopting multi-cloud engages a consultancy that recommends providers and negotiates SLAs on its behalf, hires an independent firm to verify the provider's security and SLA compliance, and buys dedicated connectivity from a telecom partner. Map each party to the NIST IaaS actor model and state each actor's role.

**Answer:**
- **Cloud Consumer** — the enterprise itself: selects services, negotiates SLAs, pays for usage.
- **Cloud Broker** — the consultancy: an intermediary that helps consumers select, integrate, and manage services across providers. Canonical broker functions: **service intermediation** (negotiating/managing services), **service aggregation** (combining multiple services), and **service arbitrage** (dynamically choosing providers, e.g., by price/performance).
- **Cloud Auditor** — the independent assessor: evaluates security, privacy, performance, compliance, and SLA verification, providing audit reports and assurance.
- **Cloud Carrier** — the telecom partner: provides connectivity and transport between consumers, providers, and data centres (bandwidth, QoS, inter-datacentre links).
- **Cloud Provider** — the platform(s) being consumed: builds, owns, and operates the IaaS infrastructure and is responsible for availability, performance, security, operations, metering, and billing.

**Why this matters:** The actor model clarifies accountability: consumers use, providers build and run, brokers add value, auditors assure trust, carriers connect everything. In multi-cloud practice the broker and auditor roles are frequently commercialised exactly as this scenario shows.

**Key concepts tested:** NIST IaaS actors, broker functions, audit and carrier roles.

---

### Q31. Instance type vs. AMI selection for an in-memory database

**Scenario:** A team must deploy an in-memory analytics database on AWS. They are confused about whether to choose an "r7g" or an Ubuntu AMI. Clarify the two decisions, recommend an instance family, and justify it.

**Answer:** These are the **two independent decisions** required to launch any EC2 instance:
1. **Instance Type** = the *virtual hardware*: vCPUs, memory, storage, network performance. Analogy: **the processor**.
2. **AMI (Amazon Machine Image)** = the *software loaded at launch*: OS, patch level, pre-installed software. Analogy: **the OS**.

You need both: an r7g instance type *and* an AMI (e.g., Ubuntu) — one doesn't replace the other.

**Recommendation: a memory-optimized family (r7g).** Instance types are grouped into families by resource ratios: general-purpose (m — balanced), compute-optimized (c — high compute:memory), **memory-optimized (r — for in-memory databases)**, storage-optimized (i/d), GPU/accelerated (p), burstable (t). An in-memory database's working set lives in RAM, so the memory-heavy ratio of the r-family (with Graviton ARM CPUs, per the "g" suffix) matches the workload. Industry evidence supports this: Zomato migrated analytics clusters (Trino/Druid) to Graviton2 ARM instances with Spot pricing and achieved ~30% compute cost reduction and 25% better query performance.

**Trade-offs:** Memory-optimized instances cost more per hour than general-purpose; Graviton (ARM) requires the software stack to support ARM builds.

**Key concepts tested:** Instance type vs. AMI distinction, instance families, workload-matching.

---

### Q32. Golden AMI from an existing instance

**Scenario:** An enterprise must launch 200 web servers that are identically patched, hardened, monitored, and configured to corporate standard. Currently admins launch from AWS-published Ubuntu AMIs and configure each by hand — servers drift and audits fail. Propose the correct AMI strategy and justify it against alternatives.

**Answer:** Adopt a **golden AMI generated from an existing instance**. Process: launch one instance from a published AMI → configure it fully to corporate standards (patches, security hardening, monitoring agent) → generate a new AMI from the configured instance → use *that* AMI to launch all 200 servers (and all future ones).

**Why this wins:** It delivers **consistent, repeatable standards** — every instance inherits patches, security, and monitoring automatically; it **reduces drift** across projects because non-conforming launches become difficult; and it saves time/cost — configure once, reuse many times. This is the "Generated from Existing Instances" AMI source, described as the most common source of AMIs.

**Against alternatives:** *AWS-published AMIs* give default OS settings requiring immediate patching on every launch — that's exactly the manual, drifting process being replaced. *AWS Marketplace AMIs* are for third-party partner software with cloud-appropriate licensing (useful when you need a pre-configured commercial product, at instance cost + optional per-hour software fee) — not for enforcing *your* internal standard. *Uploaded VM images* (via VM Import/Export from raw/VMDK/VHD/OVA) are for migrating existing on-premises virtual servers — again, a different problem.

**Governance note:** tag the golden AMI pipeline (environment, owner) and re-bake it on a patch cycle so the standard stays current.

**Key concepts tested:** AMI sources, golden image pattern, configuration drift, corporate standards.

---

### Q33. Tenancy options for licensed, compliance-sensitive workloads

**Scenario:** A company runs server-bound commercial software licensed per physical socket, and its auditors require knowing exactly which physical server hosts each workload. On AWS, it must choose between shared tenancy, Dedicated Instances, and Dedicated Hosts. Explain each option and recommend one.

**Answer:**
- **Shared Tenancy (default):** a single physical host may house instances from *different customers*. It is still secure — AWS does not overprovision and fully isolates instances at the hardware/virtualization level — but the customer has no visibility or control over physical placement, so it cannot satisfy socket-based licensing or audit requirements.
- **Dedicated Instances:** instances run on hardware **dedicated to a single customer's account**; other instances in the same account may still run on shared tenancy, isolated at the hardware level. Good for steady-state workloads needing isolation short of a full host, but placement is still AWS-managed.
- **Dedicated Host:** an entire **physical server fully dedicated to one customer**, with visibility and control over instance placement on that host.

**Recommendation: Dedicated Hosts.** They directly address both constraints: (1) **licensing** — server-bound (socket/core-licensed) software requires a known, dedicated physical server, which shared or account-level dedication cannot guarantee; (2) **audit/compliance** — the customer controls and can attest to placement. Dedicated Hosts are explicitly "best for license compliance, BYOL scenarios, regulatory requirements."

**Trade-offs:** Dedicated options cost more than shared tenancy and reduce the flexibility of automatic placement; all tenancy models provide strong isolation, so the choice is driven by compliance/licensing, not baseline security.

**Key concepts tested:** EC2 tenancy options, shared vs. dedicated isolation, licensing-driven placement.

---

### Q34. Placement groups for an HPC cluster

**Scenario:** A research lab runs a tightly coupled weather-simulation cluster on EC2. Nodes exchange massive intermediate data constantly, and current cross-node network throughput is the bottleneck. Which EC2 feature addresses this, how does it work, and what availability trade-off does it introduce?

**Answer:** Use a **placement group** — a logical grouping of instances within a **single Availability Zone** that enables applications to participate in a **low-latency, 10 Gbps network** between members.

**How it works:** Instances in the group are placed close together (same AZ, cluster-style placement) and connected by high-bandwidth, low-latency networking. To fully exploit it, choose instance types that support **enhanced networking** and 10 Gbps performance; network performance otherwise scales with instance size within a family. Common types: **cluster** (HPC, tightly coupled), **spread** (critical apps needing separation across underlying hardware), **partition** (big data like Hadoop/HBase).

**The trade-off:** everything in one AZ means **correlated failure risk** — an AZ-level outage (power, flooding, network) takes the whole cluster down at once. That is the deliberate opposite of the multi-AZ fault-isolation pattern used for web tiers. For HPC the latency/throughput requirement justifies accepting AZ concentration; mitigation is checkpointing to durable storage (e.g., S3) so the cluster can be relaunched elsewhere.

**Why the lab's bottleneck matches:** constant massive inter-node data exchange is exactly the "interdependent workloads" use case placement groups optimise.

**Key concepts tested:** Placement groups, single-AZ concentration vs. fault isolation, enhanced networking.

---

### Q35. Instance store vs. EBS for a database

**Scenario:** A developer launched a database EC2 instance using **instance store** volumes because they were "free and fast." Overnight the instance was stopped for maintenance, and the database was gone in the morning. Explain what happened, why the storage behaved that way, and what the correct storage choice is.

**Answer:** **Instance store** is **ephemeral, temporary block storage** physically attached to the host computer. Its data is lost when: the underlying disk drive fails, **the instance stops** (data persists only across a *reboot*, not a stop), or the instance terminates. The overnight stop therefore wiped the database — the storage behaved exactly as designed.

**Why it exists:** instance stores are included in the instance's cost (cost-effective) and physically attached (very fast), making them appropriate for *temporary* data — buffers, caches, scratch space, replicated data that is rebuilt — never for data that must survive the instance lifecycle.

**Correct choice: Amazon EBS.** EBS provides **persistent block-level storage volumes** for EC2: each volume is **automatically replicated within its Availability Zone** to protect against component failure; volumes survive stop/terminate (until explicitly deleted); multiple volumes can attach to one instance; and types differ in performance/price — General-Purpose SSD (1 GB–16 TB, baseline 3 IOPS/GB capped at 10,000) for boot volumes and small/medium databases, Provisioned IOPS SSD (4 GB–16 TB, highest predictable performance) for I/O-intensive, latency-critical databases.

**Trade-offs:** EBS is billed separately per GB and per volume type; it is also **AZ-bound** (a volume attaches only within its AZ) — relevant later when designing multi-AZ HA (Q36).

**Key concepts tested:** Instance store ephemerality, loss conditions, EBS persistence and types.

---

# Section 3 — High (Questions 36–50)

### Q36. Multi-AZ high availability for an e-commerce platform

**Scenario:** An e-commerce company runs its entire production stack — web servers, application servers, and database — inside a single AWS Availability Zone. During a power event in that AZ, the site is completely down for 40 minutes. Identify the architectural weaknesses and design a corrected multi-AZ architecture.

**Answer:** **Weakness:** a single AZ is a single fault domain. AZs are physically separated facilities (different flood plains, independent power grids, UPS/generators), so an AZ outage — however rare — takes down everything co-located there. Hosting all instances in one affected location means none are available; the design violates fault isolation.

**Corrected design:**
- **Compute:** run web/app instances across **≥2 AZs** (e.g., us-east-1a and us-east-1b), fronted by a load balancer and auto scaling. If instances in one AZ fail, instances in another AZ handle requests — the textbook mitigation.
- **Database:** use **RDS Multi-AZ**: a synchronous standby maintained in a second AZ with **automatic failover**. Note Multi-AZ is an availability/durability feature, not a performance feature.
- **Stateless application tier:** keep sessions/state out of the instances (or externalised) so any instance in any AZ can serve any request — a cloud-native prerequisite for cross-AZ scaling.
- **Storage nuance:** EBS volumes are replicated *within their own AZ* — they protect against component failure, not AZ failure. Static assets should live in **S3**, which is replicated across multiple facilities within the Region.

**Trade-offs:** Cross-AZ data transfer incurs cost; synchronous Multi-AZ writes add commit latency; running 2× capacity across AZs raises compute spend. These are the standard prices of availability.

**Industry relevance:** Multi-AZ is the baseline production posture for any customer-facing platform; the 40-minute outage here is precisely the incident class multi-AZ design eliminates.

**Key concepts tested:** AZ fault isolation, load balancing, RDS Multi-AZ, EBS-AZ scope, statelessness.

---

### Q37. Region vs. AZ vs. Edge Location for a global streaming service

**Scenario:** A video-streaming company serves users worldwide. It must (1) host its application and user data, (2) survive the failure of a data-centre facility, and (3) deliver video files to viewers in hundreds of cities with minimal buffering. Explain which AWS infrastructure construct addresses each requirement and why the three are not interchangeable.

**Answer:**
1. **Application + data hosting → AWS Region.** A Region is a geographic area grouping a cluster of Availability Zones — where virtualized compute, storage, and platform services actually run. The company picks Regions by latency to major user bases, compliance, cost, and service availability.
2. **Surviving facility failure → Availability Zones.** Each AZ is one or more discrete data centres with redundant power, networking, and connectivity, physically separated from other AZs (different flood plains, independent grids) yet connected by low-latency links. Deploying across AZs protects against a single-location service disruption while keeping components close enough for synchronous coordination.
3. **Fast global content delivery → Edge Locations.** Edge locations (CloudFront POPs) are content-caching points placed close to end users; they cache content (videos, images, software updates) for fast delivery. Crucially, an edge location **does not run your compute or store your primary data** — it is a cache, like a luggage drop-off counter at a train station rather than the hotel itself.

**Why not interchangeable:** Regions are for running workloads and holding system-of-record data; AZs are fault-isolation units *within* a Region; edges are distribution caches *outside* Regions, close to users. Placing databases in edge locations or expecting edges to survive regional failure are category errors.

**Architecture:** origin in 2+ Regions with cross-Region replication for resilience; AZ-spread within each Region for HA; CloudFront edges for delivery.

**Key concepts tested:** Region vs. AZ vs. Edge Location, caching vs. hosting, global architecture.

---

### Q38. Selecting the right edge option: Outposts, Local Zones, Wavelength

**Scenario:** Three clients approach a consultancy: (A) a **bank** that must run core transaction processing **on-premises for data residency** but wants to use AWS APIs and management; (B) a **media company** needing very low latency for users in one metro area that lacks a nearby full Region; (C) a **cloud-gaming startup** whose latency budget only works if compute sits **inside 5G carrier networks**. Recommend the right AWS offering for each and justify.

**Answer:**
- **A → AWS Outposts.** Outposts is a fully managed service that extends AWS infrastructure, services, APIs, and tools **to customer premises**. The bank runs applications on-premises using the same programming interfaces as AWS Regions while using local compute/storage for lower latency and **local data processing/data-residency** needs — and AWS operates, monitors, and manages the capacity as part of a Region. This is the textbook Outposts use case (bank core transaction processing on-prem for data residency).
- **B → AWS Local Zones.** A Local Zone is an **extension of an AWS Region in geographic proximity to users**, with its own internet connection, serving local users with low-latency communications (e.g., us-west-2-lax-1 in Los Angeles). Media processing near a metro without a full Region is the designed use case; note opt-in is required and not all services are available in every Local Zone.
- **C → AWS Wavelength Zones.** Wavelength embeds AWS compute and storage **within telecom providers' data centres at the edge of the 5G network**, so traffic reaches application servers **without leaving the carrier's network** — eliminating the multi-hop internet latency that would break a cloud-gaming/ultra-low-latency budget, while using familiar AWS services and APIs.

**Distinguishing principle:** data residency on-prem → Outposts; metro proximity → Local Zones; carrier-network 5G edge → Wavelength. All three exist because one global Region grid cannot satisfy every latency and legal constraint.

**Key concepts tested:** Outposts, Local Zones, Wavelength Zones, latency vs. residency trade-offs.

---

### Q39. VPC design review: public database exposure

**Scenario:** A security review finds that a team's VPC places the web tier *and* the database in a single public subnet with Internet Gateway access. Critique the design and present the corrected topology, explaining the roles of subnets, the Internet Gateway, NAT, and security groups vs. NACLs.

**Answer:** **Critique:** Only resources that must be Internet-reachable belong in a **public subnet**. A database with internet-path exposure is an attack-surface violation — a VPC is *logically isolated* from other networks precisely so you can control what faces the Internet. Databases should never be publicly addressable.

**Corrected topology:**
- **Public subnet:** load balancer / web tier only, with an **Internet Gateway (IGW)** attached to the VPC for Internet communication.
- **Private subnet:** application and database tiers — instances get private IPv4 addresses only, cannot be reached from the Internet, but communicate within the VPC. For outbound needs (OS patching, API calls), a **NAT gateway** in the public subnet lets private instances initiate Internet connections without being reachable inbound.
- **Route tables** steer traffic: private subnets route 0.0.0.0/0 to NAT, not IGW; public subnets route to IGW.
- **Defence in depth — security groups vs. NACLs:** **security groups** are stateful, instance-level firewalls (e.g., DB security group allows port 3306 *only* from the app tier's security group); **NACLs** are stateless, subnet-level filters for coarse-grained control. Use both: SGs for precise service-to-service rules, NACLs for subnet boundary policy.

**Why this matters:** The corrected design follows the VPC principle — public subnet for Internet-connected resources, private subnet for resources that won't be Internet-connected — and layers instance-level and subnet-level controls.

**Key concepts tested:** VPC isolation, public/private subnets, IGW vs. NAT, security groups vs. NACLs, least network exposure.

---

### Q40. Shared storage decision: EFS vs. EBS vs. S3

**Scenario:** A content-management platform runs six web-server instances that must all read and write the same uploaded documents and templates concurrently, with storage that grows automatically. The team is weighing EBS, S3, and EFS. Choose and justify against each alternative.

**Answer:** Choose **Amazon EFS (Elastic File System)** — a file-level, **shared elastic file system** that many servers can utilise **at the same time**, with virtually unlimited, automatic scalability (scales up on demand and back down; pay only for what you use).

| Criterion | S3 (object) | EBS (block) | EFS (file) |
|---|---|---|---|
| Access model | Whole-object GET/PUT via HTTP API | Block device attached to **one** EC2 instance | Standard file system (NFS) mounted by **many** instances |
| Shared concurrent access | No (application-level only) | No | **Yes** |
| Scaling | Unlimited objects | Manual volume sizing | Automatic, elastic |
| Best for | Static assets, backups, distribution | Boot volumes, DBs, transactional | **Shared file storage, web serving, lift-and-shift** |

**Why not EBS:** a volume can be attached to only **one instance at a time** — six web servers cannot share one EBS volume; you'd need per-server copies and a sync nightmare. **Why not S3:** S3 is object storage accessed by PUT/GET of *whole objects* over HTTP — it is not a file system; a CMS expecting POSIX file semantics (partial updates, directory hierarchies, file locks) doesn't fit its flat, whole-object model.

**Trade-offs:** EFS costs more per GB than S3 and has higher latency than local EBS; for purely static, unchanging assets served to end users, S3 remains the right store. The CMS's *working* shared files → EFS; its published static media → S3.

**Key concepts tested:** EFS vs. EBS vs. S3, access-model fit, concurrent sharing, elastic scaling.

---

### Q41. Archival tiering for compliance logs

**Scenario:** A financial firm must retain seven years of audit logs. The logs are written constantly, accessed rarely (roughly once a year for audits), and storage cost is under scrutiny. Current practice keeps everything in S3 Standard. Propose a cost-optimised storage strategy and state the retrieval-behaviour implications.

**Answer:** Introduce lifecycle tiering: keep recent logs in **S3 Standard**, then transition older objects to **Amazon Glacier** (with Glacier Deep Archive for the coldest tail), the extremely low-cost, durable service designed for **data archiving and online backup** of infrequently accessed data.

**Justification:** S3 Standard is priced for frequent access; paying those rates for data touched once a year wastes most of the spend. Glacier trades retrieval speed for cost: Flexible Retrieval typically returns data in **three to five hours** (options from minutes to bulk 12–48 hour classes; minimum storage durations apply, e.g., 90/180 days). For an annual audit, a multi-hour retrieval window is entirely acceptable — auditors can request extracts in advance. Glacier stores an unlimited amount of virtually any data in any format, is automatically replicated within the Region, and is durable (11 nines class).

**Implications / trade-offs:** Retrieval is *not* instant — design reports and DR assumptions around hours, not seconds; deleting objects before the minimum storage duration incurs charges; automated backups for point-in-time recovery still belong in S3/EBS tiers, not Glacier. Lifecycle rules automate the transitions so operations are hands-off.

**Industry relevance:** This is the standard archive pattern for regulated industries — compliance retention at archival cost, with retrieval SLAs matched to audit workflows.

**Key concepts tested:** Glacier purpose, retrieval-time trade-off, lifecycle tiering, cost optimisation.

---

### Q42. RDS Multi-AZ vs. Read Replicas for a read-heavy news site

**Scenario:** A news website's relational database is read-heavy — traffic multiplies during breaking events. The DBA wants both high availability and read scaling but is unsure whether to use Multi-AZ deployments, Read Replicas, or both. Explain the difference between the two features and recommend a configuration.

**Answer:** They solve different problems and are **not substitutes**:
- **Multi-AZ deployment:** maintains a **synchronous standby in a second AZ** with **automatic failover**. Purpose: **availability/durability** — explicitly *not* a performance feature; the standby serves no read traffic in normal operation.
- **Read Replicas:** **asynchronous, read-only copies** (same-Region or cross-Region) used specifically to **scale read-heavy workloads** — up to 5 for most engines, up to 15 for Aurora. They add read throughput but lag the primary (async) and are not automatic failover targets.

| | Multi-AZ | Read Replicas |
|---|---|---|
| Replication | Synchronous | Asynchronous |
| Purpose | Availability (failover) | Read scaling (performance) |
| Serves traffic? | No (standby) | Yes (reads) |

**Recommendation: use both.** Multi-AZ for automatic failover when the primary AZ fails; Read Replicas to absorb event-driven read surges by distributing SELECT load. Add **automated backups** (default on, 1–35 day retention) for point-in-time recovery to the second — a third, distinct mechanism often confused with the other two.

**Trade-offs:** replicas lag (eventual consistency for reads), each replica costs money, and Multi-AZ adds write latency from synchronous commits — the standard price of durability.

**Key concepts tested:** Multi-AZ vs. Read Replicas, sync vs. async replication, availability vs. performance features.

---

### Q43. Private-cloud expansion vs. public cloud — capacity calculation (sample-paper pattern)

**Scenario:** A company may expand its private cloud capacity by **30 units**, with expansion costing a fixed **Rs. 4,500**. The private unit cost remains **Rs. 100** even after expansion. Current private capacity = **50 units**. Public cloud cost = **Rs. 280 per unit**. Budget = **Rs. 13,000** (inclusive of expansion cost if chosen). Should the company expand private capacity to maximise total demand served? Justify with calculations.

**Answer:** **Yes — expand. It serves 85 units vs. 78 without expansion.**

**Case 1 — No expansion:**
- Private: 50 units × 100 = Rs. 5,000
- Remaining budget: 13,000 − 5,000 = Rs. 8,000
- Public units: 8,000 ÷ 280 = 28.57 → **28 units** (Rs. 7,840)
- **Total served = 50 + 28 = 78 units**

**Case 2 — Expand:**
- Expansion fixed cost: Rs. 4,500; remaining for units: 13,000 − 4,500 = Rs. 8,500
- Private units available: 50 + 30 = 90; affordable at 100/unit: 8,500 ÷ 100 = **85 units** (Rs. 8,500 — exactly the remaining budget)
- Remaining for public: Rs. 0
- **Total served = 85 units**

**Decision logic beyond the arithmetic:** the *marginal cost of an expansion unit* = unit cost + (fixed cost ÷ added units) = 100 + 4,500/30 = **Rs. 250**, which is **below the public unit cost of Rs. 280**. Every expanded unit is cheaper than buying from the public cloud, so shifting budget into expansion increases total demand served (85 > 78, i.e., 7 more units).

**Assumptions (state clearly, per exam instructions):** whole units only; expansion capacity is usable within the budget; public capacity is unbounded.

**Key concepts tested:** Hybrid capacity economics, marginal-cost vs. public-price comparison, budget-constrained maximisation.

---

### Q44. The overcommitment subtlety — aggregate ratio vs. instantaneous contention

**Scenario:** A host has **4 physical cores** (no hyperthreading). **6 VMs** are deployed, each with **2 vCPUs**. The admin computes the overcommit ratio and declares the host safe. Later, users report latency spikes on database VMs during peak hours, when three of the six VMs simultaneously peg both vCPUs at 100%. Explain why the aggregate ratio concealed the problem and what should be done.

**Answer:**
- **Aggregate ratio:** total vCPUs = 6 × 2 = 12; ratio = 12 ÷ 4 = **3:1** — within the workable 3:1–5:1 zone for mixed workloads. The admin's conclusion was *reasonable as a planning heuristic*.
- **The subtlety:** ratios assume demand is spread over time. If 3 VMs simultaneously demand 100% on both vCPUs, that's 3 × 2 = **6 vCPUs of must-run-now demand against only 4 pCPUs** — instantaneous contention of **6:4 = 1.5:1 for that busy subset**, regardless of what the idle VMs are doing. **The aggregate ratio is a planning heuristic, not a guarantee.**

**Diagnosis:** monitor **CPU Ready Time** — the time a vCPU is ready but waiting for a physical core. Sustained values above ~5% confirm the host is oversubscribed beyond what the *current coincident workload* tolerates.

**Remediation options:** right-size the DB VMs (fewer vCPUs each, or more pCPUs), apply reservations so critical VMs are guaranteed cores, separate latency-sensitive DBs onto 1:1 hosts (per the guidance that databases want 1:1), and distribute VMs across more hosts to break the coincidence of peaks.

**Lesson:** safe-zone ratios are workload- and *timing*-dependent; the correct engineering answer always ties back to a measured symptom (CPU Ready) rather than the ratio alone.

**Key concepts tested:** Aggregate vs. instantaneous contention, CPU Ready Time, 1:1 guidance for latency-sensitive workloads.

---

### Q45. Replacing embedded credentials with IAM roles

**Scenario:** An analytics application running on an EC2 instance reads files from an S3 bucket. The developer embedded long-term AWS access keys in the application config. After a credential leak, the security team mandates a redesign. Propose the correct pattern and explain why it is superior, referencing least privilege.

**Answer:** Attach an **IAM role** to the EC2 instance and scope it tightly: a policy allowing **only `s3:GetObject` on that specific bucket** (and prefix), and nothing else.

**How it works:** the instance *assumes* the role and automatically receives **temporary security credentials** (via the security token service) that are valid for minutes-to-hours and rotate automatically. No permanent keys are stored anywhere — not in code, not on the instance. Authentication is thus identity-without-a-person: the role is assumable by the trusted service, and the *policy attached to the role* is the enforceable rulebook (Effect/Action/Resource/Condition).

**Why it is superior:**
1. **Eliminates the leak class entirely** — there are no long-lived, exportable keys to steal; stolen temporary credentials expire quickly and are scoped.
2. **Enforces least privilege mechanically** — "interns can't delete anything" becomes a JSON policy checked on every API call; the app literally cannot `DeleteObject` or touch other buckets, and an explicit Deny overrides any Allow.
3. **Removes rotation toil** — leaked/rotated application keys (the original incident's remediation pain) disappear because credentials auto-rotate.

**Trade-offs:** role policies need careful scoping and periodic review; debugging "Access Denied" requires checking role policy evaluation, but that friction is the least-privilege control working.

**Industry relevance:** This is the standard secure pattern for service-to-service access on AWS (same model as Lambda execution roles and cross-account audit roles).

**Key concepts tested:** Roles vs. long-term keys, temporary credentials, policies, least privilege.

---

### Q46. Shared responsibility in an incident

**Scenario:** An incident review finds two problems: (1) a customer's S3 bucket was left publicly readable, exposing files; (2) a customer's EC2 instance was compromised through an unpatched OS vulnerability. Management asks whether AWS failed. Apply the shared responsibility model to allocate responsibility precisely.

**Answer:** **AWS did not fail — both failures fall on the customer's side of the model.**

**The model:** AWS is responsible for security **"of" the cloud** — the hardware/infrastructure layer (Regions, Availability Zones, edge locations), and the software layer it operates (compute, storage, database, networking services). The customer is responsible for security **"in" the cloud** — customer data, platform/applications, identity & access management, **operating system, and network & firewall configuration** (for IaaS), including client/server-side encryption choices and traffic protection.

**Applying it:**
1. **Publicly readable bucket:** S3 *as a service* was secure; the customer misconfigured **access controls and data protection** — a customer responsibility. Fix: bucket policies/least privilege, default-deny posture, encryption at rest, access auditing.
2. **Unpatched OS on EC2:** the instance's guest OS is squarely the customer's layer in IaaS; AWS secures the hypervisor and physical host, not the guest kernel's patch level. Fix: patched/golden AMIs (Q32), automated patching, network restrictions via security groups.

**Why the model matters:** it defines the boundary that makes multi-tenant public cloud auditable — the provider proves infrastructure security, customers prove workload security. Blaming "the cloud" for guest-layer misconfigurations is the most common governance error in cloud incidents.

**Key concepts tested:** Shared responsibility model, security of vs. in the cloud, IaaS customer obligations.

---

### Q47. Phased migration: physical servers to cloud-native (concept integration)

**Scenario:** A retail company runs a monolithic application on aging physical servers at low utilisation, with traffic that varies 5× daily and a single-location data centre. Leadership wants: better utilisation, component isolation, migration to AWS, and eventually a cloud-native architecture. Design a phased roadmap, naming the technology and justification at each phase.

**Answer:**

**Phase 1 — Virtualize (on-prem or as the entry to cloud):** consolidate the low-utilisation physical fleet onto hypervisor-hosted VMs. *Justification:* partitioning/isolation/encapsulation raise utilisation from ~20% dramatically while preserving per-component isolation; VM state as files simplifies backup/migration.

**Phase 2 — Migrate to IaaS (lift-and-shift on AWS):** move the VMs to EC2 in a chosen Region (Mumbai), using EBS for persistent disks and a VPC with public/private subnets. *Justification:* no upfront cost, multi-tier auto-provisioning within hours, elasticity for the 5× traffic swing, SLA-backed availability; IaaS delivers the foundational building blocks while keeping migration risk low.

**Phase 3 — Containerize stateless components:** extract services that don't need OS-level isolation into Docker images (immutable, layered), stored in a registry; apply cgroup resource limits per service. *Justification:* MBs not GBs per workload, seconds not minutes to start — enabling fast, independent deployments of hot components.

**Phase 4 — Orchestrate:** run containers under Kubernetes for scheduling, self-healing, horizontal scaling, and rolling updates across a multi-AZ cluster. *Justification:* the 5× daily swing is exactly what declarative autoscaling absorbs without overprovisioning.

**Phase 5 — Cloud-native + HA:** decompose the monolith into microservices, externalise state (stateless app tier), put data in managed services (RDS Multi-AZ, S3, EFS as fit), spread everything across ≥2 AZs behind load balancing. *Justification:* fault isolation per AZ, scale per service, and failure containment per component — the defining cloud-native properties.

**Cross-cutting cautions:** re-architect incrementally (big-bang decomposition is the classic failure mode); each phase must deliver standalone value; cost, IAM least privilege, and tagging governance apply from Phase 2 onward.

**Key concepts tested:** Virtualization → IaaS → containers → Kubernetes → cloud-native integration; HA design; phased migration reasoning.

---

### Q48. "It works on my machine" — Docker portability troubleshooting

**Scenario:** A Dockerized application runs perfectly on a developer's laptop but fails on the staging server: the database is empty, some configuration differs, and one binary crashes with an architecture error. Identify the Docker concepts that explain each symptom and outline an investigation method.

**Answer:** Three distinct containerization concepts are implicated:

1. **Empty database → writable-layer ephemerality / missing volume.** State written inside a container lives in its ephemeral writable layer and disappears on recreation. If the dev machine had a long-lived local container (or a bind mount) that staging lacks, staging starts from the pristine image. **Fix/investigate:** mount a **volume** for the data path; verify with `docker inspect` (Mounts section) on both hosts.

2. **Configuration differences → config baked into the image.** If endpoints, credentials, or environment URLs were hardcoded (COPY'd config files or built into layers), the image is environment-specific — violating the image-is-portable principle. **Fix/investigate:** externalise configuration to environment variables at run time; compare `docker inspect` Environment sections.

3. **Architecture crash → ISA mismatch.** An image built for one architecture (e.g., Apple-silicon ARM laptop) cannot run on an x86 server without emulation — the same-ISA constraint that separates virtualization from emulation. **Fix/investigate:** build multi-arch images or build on the target architecture (`docker buildx`).

**General method:** compare the exact image identities on both hosts (`docker images --digests`) — a `latest` tag can silently point to different builds; that alone explains "works on my machine." Then diff inspect output (mounts, env, entrypoint). The systematic lesson: containers are portable *only when* state, config, and architecture are handled explicitly — the image must be the single immutable source of truth pulled from the registry.

**Key concepts tested:** Volumes, configuration externalisation, architecture/ISA, image digests, troubleshooting method.

---

### Q49. Namespaces vs. cgroups — a misdiagnosis

**Scenario:** On a shared host, one container's runaway process consumes 90% CPU. An engineer proposes: "Add another network namespace to that container so it stops consuming everyone's CPU." A second engineer proposes enabling a host-level swap storm to relieve pressure. Evaluate both proposals and give the correct fix, clearly separating what namespaces do from what cgroups do.

**Answer:** **Both proposals are wrong.**

**Why the namespace proposal fails:** namespaces isolate *views*, not *consumption*. Adding a NET namespace (or any other namespace — PID, MNT, UTS, IPC, USER) changes what the container can *see*; it does nothing about how much CPU its processes receive. The engineer has confused isolation-of-visibility with resource control — the central distinction in the container model.

**Why the swap-storm proposal fails:** host-level swapping is the **last resort** of memory reclamation — pages to disk at roughly 100,000× slower than RAM, inflicting sharp latency spikes on *every* VM/container. It treats a CPU-control problem with the most invasive memory mechanism, punishing all tenants to contain one.

**Correct fix — cgroups:** place the container's processes under control-group limits: a CPU quota (e.g., `--cpus=1`) and a memory ceiling (`--memory=2g`, with OOM protection). The kernel then enforces a bounded share regardless of the process's appetite, making multi-tenant performance predictable.

**The complete model:** **namespaces** answer *what can the container see?* (its own processes, network, filesystem, hostname); **cgroups** answer *how much can it use?* (CPU, memory, I/O quotas). Production containers require **both** — isolation from cgroups is insecure under load; control without isolation still exposes processes to each other. This pairing is exactly why containers deliver "VM-like" behaviour at process-level cost.

**Key concepts tested:** Namespace vs. cgroup separation, runaway-process remediation, multi-tenant fairness.

---

### Q50. Region-level disaster recovery — and the EBS replication misconception

**Scenario:** A payments application runs entirely in one AZ: EC2 instances with EBS volumes, and a database in the same AZ. After reading AWS documentation, the CTO concludes: *"EBS is automatically replicated and S3 is replicated, so we are resilient even if the whole Region has an issue."* Identify the two errors in this reasoning, explain the actual replication scopes, and design an appropriate resilience strategy.

**Answer:**

**Error 1 — EBS replication does not survive AZ failure.** Each EBS volume is automatically replicated **within its Availability Zone** to protect against *component* failure. If the AZ fails, the volume goes with it. EBS is also AZ-bound — a volume attaches only within its own AZ. So a single-AZ application loses both compute and its disks simultaneously; the design has no fault isolation.

**Error 2 — automatic S3 replication is intra-Region, not inter-Region.** S3 objects are automatically replicated across **multiple devices in multiple facilities within a Region** — i.e., across AZs, giving extreme durability against component and AZ loss. It does **not** automatically copy to another Region. Nothing about intra-Region replication protects against a *Region-level* event.

**Correct strategy (tiered resilience):**
- **AZ level:** spread EC2 across ≥2 AZs behind a load balancer; run the database as **RDS Multi-AZ** (synchronous standby, automatic failover); keep static assets in S3 (naturally multi-AZ).
- **Region level:** replicate critical data to a second Region — **S3 cross-Region replication**, **cross-Region Read Replicas** (async) for the database — and maintain warm/cold standby application infrastructure there, with DNS-level failover. Regions are completely independent by design, which is precisely what makes them the unit for the "greatest possible fault tolerance."
- **Reality check:** Region-level DR costs more (duplicate infrastructure, cross-Region transfer) and cross-Region replicas lag (async) — the business must choose RPO/RTO and pay accordingly. For a payments platform, an active-passive second Region is typically justified; for internal tools, multi-AZ alone may suffice.

**Key concepts tested:** EBS replication scope, S3 replication scope, AZ vs. Region resilience tiers, DR cost trade-offs.

---

# Most Important Concepts to Revise

1. **IaaS/PaaS/SaaS responsibility split** — classification always comes down to *who manages which layer*; justify with layers, never with brand names.
2. **Service model architecture comparisons (SaaS vs. PaaS+IaaS)** — compare on scalability, customization, operational control; state when each is preferred.
3. **Deployment models (public/private/hybrid)** — driven by data residency, compliance, and elasticity needs; hybrid = private + public binding with burst patterns.
4. **Virtualization fundamentals: hypervisor, host, guest, partitioning/isolation/encapsulation** — the enabler of IaaS and consolidation economics.
5. **Type-1 vs. Type-2 hypervisors** — bare-metal (production, performance, isolation) vs. hosted (desktops, convenience).
6. **x86 rings + sensitive-but-unprivileged instructions** — why trap-and-emulate failed on x86; the root cause of the 1998 breakthrough.
7. **Full virtualization vs. paravirtualization vs. hardware-assisted (3 generations)** — unmodified guest vs. hypercalls vs. Ring −1; performance/OS-modification trade-offs; demerits + mitigations (a sample-paper pattern).
8. **Emulation vs. virtualization** — cross-ISA capability vs. same-ISA speed; interpretation vs. DBT; dev/test only.
9. **CPU overcommitment** — ratio = total vCPU ÷ pCPU; CPU Ready Time (~5% flag); 1:1 / 3:1–5:1 / >8:1 guidance; aggregate ratio ≠ instantaneous guarantee.
10. **Memory overcommitment** — ratio = configured ÷ physical; reclamation order: page sharing → ballooning → compression → host swapping (last resort, ~100,000× slower).
11. **VM vs. containers** — kernel sharing, MB vs. GB, seconds vs. minutes, density vs. isolation strength.
12. **Docker images vs. containers; Dockerfiles and layer caching** — immutable layered images, COPY-manifest-first ordering, EXPOSE/WORKDIR/CMD semantics.
13. **Docker volumes** — persistence across restarts/removals; portability and I/O efficiency for stateful production containers.
14. **Namespaces vs. cgroups** — visibility isolation vs. resource limits; never conflate them in troubleshooting answers.
15. **Kubernetes' role** — orchestrator, not a container: declarative state, self-healing, scheduling, scaling, service discovery.
16. **Cloud-native principles** — microservices, statelessness, containers, orchestration, design-for-failure; phased monolith migration.
17. **IaaS architecture stack + NIST actors** — 5 layers (physical → virtualization → resource pools → orchestration → service/access) with cross-cutting IAM/security/monitoring/billing; consumer/provider/broker/auditor/carrier.
18. **EC2 mechanics** — instance type (hardware/"processor") vs. AMI (software/"OS"); instance families; AMI sources (especially golden AMI); tenancy options; placement groups; instance store vs. EBS; lifecycle billing (stopped instance, unattached EIP).
19. **Storage selection** — S3 (object, whole-object, multi-AZ in region) vs. EBS (block, single-instance, AZ-replicated) vs. EFS (shared file, concurrent) vs. Glacier (archive, 3–5 h retrieval); RDS Multi-AZ (sync, availability) vs. Read Replicas (async, read scaling).
20. **Regions, AZs, edge constructs** — Region independence; AZ physical separation + low-latency links; edge locations cache content; Outposts (on-prem residency), Local Zones (metro latency), Wavelength (5G edge); multi-AZ HA and Region-level DR.

---

# Common Exam Traps

1. **Confusing IaaS with PaaS/SaaS** — test with "who manages the OS/runtime/application?", not with whether it's "in the cloud."
2. **Mislabelling deployment models** — consuming two public services is still public; only spanning *your own* infrastructure makes it hybrid.
3. **Treating containers as lightweight VMs** — containers share the *host kernel*; there is no guest OS boot; isolation is process-level, not hardware-level.
4. **Confusing Docker images with containers** — the image is the immutable template; the container is a running instance with an ephemeral writable layer.
5. **Assuming namespaces and cgroups do the same job** — namespaces = what a container sees; cgroups = how much it may use. Resource problems are never fixed by namespaces.
6. **Assuming Kubernetes is itself a container** — it is an orchestrator that manages containers on multiple hosts.
7. **Confusing AWS Regions with Availability Zones** — Region = geographic grouping of AZs; AZ = discrete, physically separated data centres within a Region; edge locations cache, they don't host your compute.
8. **Assuming multiple VMs/instances in one place give high availability** — co-location in one AZ is a single fault domain; HA requires explicit multi-AZ distribution + failover design.
9. **Assuming EBS replication protects against AZ (or Region) failure** — EBS replicates within its AZ only; S3 replicates across AZs in the Region, not across Regions automatically.
10. **Confusing Multi-AZ with Read Replicas** — synchronous standby for availability vs. asynchronous copies for read scaling; neither substitutes for the other.
11. **Ignoring resource constraints in container answers** — "one container eats everything" demands cgroup limits, not more namespaces or host swapping.
12. **Reading overcommit ratios as guarantees** — the sample's 3:1 example shows coincident demand (6 vCPUs on 4 pCPUs) can still starve VMs; always tie the answer to CPU Ready Time and workload type.
13. **Choosing a technology without stating the trade-off** — the paper explicitly awards marks for justification and mitigation; a bare "use containers" answer loses marks.
14. **Forgetting cost mechanics** — stopped EC2 instances still bill EBS; unattached Elastic IPs bill hourly; expansions have a *marginal* unit cost (fixed ÷ added units + unit cost) that must be compared against the public unit price.
15. **Mixing up image sources** — golden AMI from a configured instance (corporate standard) ≠ Marketplace (partner software) ≠ uploaded VMs (on-prem migration) ≠ AWS-published (default settings needing patching).
16. **Blurring authentication, authorization, and roles** — login success ≠ permission; roles provide temporary, no-long-term-key, least-privilege access and are the fix for embedded credentials.

---
---
---

# Cloud Computing — 25 Theoretical Questions & Answers (Exam-Style)

**Built to match the sample paper's patterns:** mark splits like (2+2+2) and (3+2), sub-parts (a)/(b), "justify" requirements, comparison tables, and the *demerit + mitigation* structure of Q1. Answers are written crisp and specific — the way the paper demands ("Answers should be specific and crisp including justification where required").

---

# Module 1 — Cloud Computing Fundamentals

### Q1. [NIST Framework] State the NIST 3-4-5 classification of cloud computing: the three service models, four deployment models, and five essential characteristics. Briefly explain any three essential characteristics. (2+1+3)

**Answer:**

| NIST Category | Items |
|---|---|
| **3 Service Models** | SaaS, PaaS, IaaS |
| **4 Deployment Models** | Public, Private, Community, Hybrid |
| **5 Essential Characteristics** | On-demand self-service, Broad network access, Resource pooling, Rapid elasticity, Measured service |

**Explanation of three characteristics:**
1. **On-demand self-service** — consumers provision computing capabilities (e.g., launch a VM) automatically via a portal/API without human interaction with the provider.
2. **Rapid elasticity** — capacity can be scaled out and in elastically (1 → 10 → 100 → 1000 units), appearing unlimited to the consumer and matching demand dynamically.
3. **Measured service** — resource usage is monitored, controlled, and reported, enabling transparent pay-per-use billing (utility-based computing).

**Why this matters:** NIST is the reference definition used to classify any cloud platform; exams expect the exact 3-4-5 structure.

---

### Q2. Compare IaaS, PaaS, and SaaS in terms of the layers each model places under the consumer's responsibility, and give one real example of each. (6, 2+2+2)

**Answer:** The three models differ by *which layers of the stack the provider manages*:

| Layer | SaaS | PaaS | IaaS |
|---|---|---|---|
| Application | Provider | Consumer | Consumer |
| Data | Provider | Consumer | Consumer |
| Runtime/Middleware | Provider | Provider | Consumer |
| Operating System | Provider | Provider | Consumer |
| Virtualization/Servers/Storage/Network | Provider | Provider | Provider |

- **IaaS (2):** Provider delivers virtualized compute, storage, network; the consumer manages the OS and everything above. *Example:* AWS EC2 / Azure VMs.
- **PaaS (2):** Provider additionally manages OS, runtime, and scaling; the consumer deploys only application code and data. *Example:* managed app platforms (e.g., AWS Elastic Beanstalk–style services).
- **SaaS (2):** The provider manages everything including the application; the consumer merely uses the software, typically via a browser. *Example:* cloud ERP/CRM accessed via browser.

**Justification:** the classification test is always "who manages which layer" — browser access with zero management ⇒ SaaS; deploying your code on a managed runtime ⇒ PaaS; installing your own OS and apps on virtual servers ⇒ IaaS.

---

### Q3. Compare the four cloud deployment models and state, with justification, the situation in which each is preferred. (5)

**Answer:**

| Model | Definition | Preferred when |
|---|---|---|
| **Public** | Services offered over shared, multi-tenant provider infrastructure to the general public | No strict residency/compliance constraints; elasticity and zero CapEx desired (startups, web apps) |
| **Private** | Infrastructure operated exclusively for one organization (on-prem or hosted) | Data sovereignty, regulatory control, or predictable steady workloads (banks, government) |
| **Community** | Shared by organizations with common concerns (mission, security, compliance) | Sector collaboration with shared policy needs |
| **Hybrid** | Binding of private + public clouds allowing data/application portability | Regulated core on private + seasonal elastic burst to public ("cloud bursting") |

**Justification logic:** the choice is driven by **control vs. elasticity**. A payments company with residency law cannot use pure public cloud (compliance fails), but buying peak-only private capacity wastes money for 11 months — hence hybrid captures both benefits.

**Trade-off:** hybrid adds integration complexity (consistent networking, identity, management across two environments); a poorly integrated hybrid bears the cost of both models with the agility of neither.

---

### Q4. (a) State four key motivations/drivers for enterprises adopting cloud computing. (b) State three significant limitations or concerns of cloud computing. (3+2)

**Answer:**

**(a) Drivers/motivations:**
1. **No upfront cost / CapEx→OpEx** — pay-per-use instead of purchasing servers and data-centre space.
2. **Rapid provisioning & agility** — multi-tier environments auto-provisioned within hours instead of weeks.
3. **Scalability & elasticity** — resources provided on demand, matching unpredictable workload swings.
4. **Focus on core business** — infrastructure outsourced so teams innovate rather than operate; backed by SLAs (e.g., 99.9% uptime).

**(b) Limitations:**
1. **Dependency on connectivity** — the application is unavailable if the network link to the provider fails.
2. **Reduced control & data security/privacy concerns** — sensitive data resides on shared, multi-tenant infrastructure outside the organization's physical perimeter; compliance/data-residency obligations may conflict.
3. **Vendor lock-in & cost unpredictability** — platform-specific bindings make migration hard, and pay-per-use costs can grow unexpectedly at scale.

---

# Module 2 — Virtualization, Containers & Cloud-Native Computing

### Q5. (a) "Cloud can exist without virtualization, but it will be difficult and inefficient." Justify this statement. (b) State the three key benefits of server virtualization. (3+2)

**Answer:**

**(a) Justification:** Without virtualization, cloud delivery means one application per physical server — low resource utilization (large idle capacity), hard manual scaling, over-provisioning, high cost, and slow response to demand. The cloud's promises — *pay for what you use* and *infinite on-demand availability* — become practical only if the backend has **flexibility** (scale up/down, provision in minutes) and **efficiency** (maximize utilization, minimize waste). Virtualization provides exactly this: it turns physical hardware into a **flexible, efficient, shared resource pool** (VMs), enabling high utilization, rapid provisioning, scalability, and lower cost — the true cloud experience.

**(b) Server virtualization benefits:**
1. **Partitioning** — multiple OS instances run on one physical machine; physical resources are divided among VMs; one VM is unaware of the others.
2. **Management/Isolation** — failure of one VM does not affect other VMs; per-VM management agents monitor individual performance.
3. **Encapsulation** — the entire VM state is saved in a file; moving or copying a VM is as easy as copying files (simplifying backup and recovery).

*Result:* consolidation onto fewer physical servers reduces space, cooling, cabling, and capital expense.

---

### Q6. Define hypervisor (VMM), host machine, and guest machine. Compare Type-1 and Type-2 hypervisors with one example of each. (5)

**Answer:** A **hypervisor (Virtual Machine Monitor)** is software, firmware, or hardware that creates and runs virtual machines. It presents guests with a **virtual operating platform**, manages execution of guest OSes, and multiplexes (allocates) CPU, memory, storage, and network among them. The **host machine** is the computer on which the hypervisor runs; each VM is a **guest machine**.

| Aspect | Type-1 (Bare-Metal) | Type-2 (Hosted) |
|---|---|---|
| Placement | Runs **directly on physical hardware**; no host OS | Runs **as an application on top of a host OS** |
| Performance | Higher (no intermediate OS layer) | Lower overhead per VM is higher |
| Isolation | Stronger | Depends on host OS stability/security |
| Ease of use | Data-centre oriented; harder setup on desktops | Easy to install; reuses host OS drivers |
| Examples | VMware ESXi, Microsoft Hyper-V, KVM, Xen | Oracle VirtualBox, VMware Workstation |

**Use-case rule:** production cloud infrastructure (IaaS virtualization layer) uses Type-1; student labs/desktops use Type-2.

---

### Q7. Explain the x86 privilege ring architecture and describe the "sensitive but not privileged" instruction problem that historically blocked x86 virtualization. (6, 3+3)

**Answer:**

**(a) Ring architecture (3):** x86 uses four privilege rings (0–3) controlling access to memory and resources. Each process runs in a ring that determines its access level:
- **Ring 0 (kernel mode):** most privileged; full access to memory and I/O; where the OS kernel runs.
- **Ring 1:** device drivers and low-level system components.
- **Ring 2:** historically for system extensions; **not used in modern x86 systems**.
- **Ring 3 (user mode):** least privileged; user applications run here and reach sensitive resources only via **system calls** handled by the Ring-0 kernel.

*Relevance:* an OS assumes it owns Ring 0, but a hypervisor must sit *underneath* the OS — leaving no natural room for both.

**(b) The sensitive-but-unprivileged problem (3):** hypervisors handle privileged instructions by **trapping** them and emulating the result. However, x86 contains a small set (~17) of instructions that are **sensitive** (behave differently depending on ring) yet **not privileged** — executed outside Ring 0, they **do not trap**; they fail *silently*, with no error the hypervisor can catch. This violates the Popek–Goldberg virtualizability requirement, so the standard trap-and-emulate technique (which worked on IBM mainframes) had a real gap on x86. It was resolved by **binary translation** (VMware, 1998–99) and later **hardware-assisted virtualization** (Intel VT-x / AMD-V, 2005–06).

---

### Q8. Choose full virtualization (binary translation) as a virtualization technique. State TWO primary demerits of this technique in cloud computing with brief justification, and how each can be mitigated. (5) *(Sample-paper pattern)*

**Answer:** *Mechanism in brief:* the hypervisor presents each guest OS a complete virtual machine (virtual CPU, memory, devices) running **unmodified**; a binary-translation layer dynamically rewrites the sensitive instructions at runtime while all other instructions execute directly on hardware ("direct execution"); the guest OS is deprivileged (Ring 1) but unaware of interception.

**Demerit 1 — Translation performance overhead.** Rewriting sensitive instructions costs CPU cycles, and the overhead **scales with kernel activity** — I/O-heavy and context-switch-heavy workloads suffer far more than compute-bound ones.
*Mitigation:* use **hardware-assisted virtualization** (Intel VT-x/AMD-V), where the CPU itself traps sensitive instructions in silicon, and use **paravirtualized drivers** for I/O devices.

**Demerit 2 — Heavyweight per-VM footprint and sprawl.** Every VM carries a complete guest OS — gigabytes of RAM/disk, minutes of boot time, and its own software license. Uncontrolled VM creation causes **server sprawl** and license-fee explosion in the cloud.
*Mitigation:* right-size and govern VM creation; apply memory overcommitment techniques (page sharing, ballooning) to raise density; shift lightweight, homogeneous workloads to **containers**, which share the host kernel at MB-scale overhead.

*(Additional demerits if asked: high hypervisor engineering complexity; same-ISA only — no cross-architecture flexibility.)*

---

### Q9. Explain paravirtualization. How does it avoid the overhead of full virtualization? State the two different approaches used in paravirtualized VMs. (5, 3+2)

**Answer:** **Paravirtualization** modifies the **guest OS kernel's architecture-dependent code** so the guest *cooperates directly* with the hypervisor instead of being fooled by it.

**How overhead is avoided (3):** because the guest kernel is modified to know it is virtualized, it **does not issue the original privileged instructions at all** — it replaces them with explicit **hypercalls** to the hypervisor (a specialized API). The VMM therefore no longer needs to trap and translate sensitive instructions at runtime — the core cost of full virtualization/binary translation. The guest kernel runs deprivileged (Ring 1 on 32-bit x86; Ring 3 on x86-64 where isolation relies on separate address spaces), while the hypervisor occupies Ring 0 and executes privileged operations against real hardware on the guest's behalf. Performance is near-native for CPU-bound work.

**Two approaches (2):**
1. **Recompiling the OS kernel** — PV drivers and APIs must reside in the guest kernel; requires the OS to be virtualization-aware (some vendors, e.g., Novell, shipped PV builds; Microsoft did not).
2. **Installing paravirtualized drivers** — where full PV is impossible, PV drivers are applied to **individual devices** (e.g., network cards, graphics) so their I/O instructions are modified before leaving the VM, improving performance without kernel recompilation.

**Key limitation:** source-level access is required, which historically restricted PV to open-source guests (Linux, BSD), not closed-source Windows.

---

### Q10. Compare the four approaches to x86 virtualization — emulation, binary translation (full virtualization), paravirtualization, and hardware-assisted virtualization — on guest-OS modification, ISA constraint, and performance. (6)

**Answer:**

| Dimension | Emulation | Binary Translation (Full Virt.) | Paravirtualization | Hardware-Assisted |
|---|---|---|---|---|
| **Core idea** | Software simulates an entire different CPU/ISA instruction-by-instruction | Hypervisor translates sensitive instructions at runtime; rest run natively | Guest kernel **modified** to call hypervisor via hypercalls | CPU adds a privilege mode below Ring 0 ("Ring −1") that traps sensitive instructions |
| **Guest OS modified?** | No | No | **Yes** (kernel source) | No |
| **Guest ISA vs host** | **Can differ** (e.g., ARM guest on x86 host) | Must match | Must match | Must match |
| **Performance** | Highest overhead — often 10–100× slower | Moderate — better than emulation, worse than HW-assist | Low — near-native for CPU-bound work | Lowest/best; traps handled in silicon |
| **Examples** | QEMU (pure emulation), Bochs | VMware Workstation/ESX (1998–2006 era) | Xen with modified Linux guest | KVM, Xen HVM mode, Hyper-V, modern ESXi |
| **Key limitation** | Too slow for production; used for cross-ISA dev/test | Complex engineering; overhead scales with kernel activity | Closed-source guests (Windows) historically impossible | Requires VT-x/AMD-V CPUs (2005–06 onward) |

**Evolution logic:** software workaround (BT, 1998) → guest cooperation (PV, 2003) → silicon support (HW-assist, 2005–06). HW-assist is today's standard because it needs no guest modification and no translation.

---

### Q11. (a) What is CPU overcommitment? Why is it possible, and which metric warns that it has gone too far? (b) Why is memory overcommitment fundamentally different from CPU overcommitment, and list the four memory reclamation techniques in order from least to most invasive. (3+3)

**Answer:**

**(a) CPU overcommitment (3):** the hypervisor allocates **more vCPUs across VMs than physical cores (pCPUs) exist**. It works because **CPU time is time-sliced** — the scheduler hands out short slices of physical core time, exactly as an OS multiplexes processes onto cores. *Overcommit Ratio = Total vCPUs ÷ Total pCPUs.* The warning metric is **CPU Ready Time** — the time a vCPU is ready to run but waiting for a physical core to free up; values above ~5% signal oversubscription. Guidance: **1:1** for latency-sensitive workloads (DBs, real-time); **3:1–5:1** workable for mixed workloads; **>8:1** high risk.

**(b) Memory is different (3):** memory **cannot be time-sliced** — a page of RAM is either allocated to a VM or it is not, at any instant. The hypervisor cannot schedule its way out of a shortfall; it must **actively reclaim or deduplicate** memory. (*Overcommit Ratio = Configured Memory ÷ Physical RAM.*) Reclamation techniques, least → most invasive:
1. **Page sharing (TPS/KSM)** — merges identical memory pages; since vSphere 6.0, inter-VM sharing is disabled by default for security.
2. **Ballooning** — a guest-side driver reclaims memory from the guest's own processes; the **guest OS decides** what to page out.
3. **Compression** — hypervisor compresses pages instead of writing to disk (slower than RAM, much faster than disk).
4. **Host-level swapping** — last resort; pages swapped to disk without guest cooperation; disk is ~100,000× slower than RAM, causing sharp latency spikes.

---

### Q12. Compare virtual machines and containers across at least five dimensions, and state under what circumstances each is preferable. (5)

**Answer:**

| Factor | Virtual Machine | Container |
|---|---|---|
| Architecture | Full guest OS with its **own kernel** on virtual hardware | **Shares the host kernel**; carries user-space only (app + libraries) |
| Size | GBs (entire OS image) | MBs (app + libs) |
| Startup | Minutes (boot a whole OS) | Seconds (no OS boot) |
| Isolation | Strong — hardware/virtualization-level separation | Weaker — process-level; kernel vulnerability is a shared risk |
| Density | Dozens per host | Hundreds per host |
| Best for | Different OSes/kernels, strict isolation, legacy apps | 100s of lightweight microservices, fast CI/CD deployments |

**When VMs:** workloads needing a different OS (e.g., Windows on Linux host), hard security boundaries, or legacy system software.
**When containers:** many small, independently deployable services where density, seconds-fast startup, and rapid release cycles dominate.

**Trade-off & industry practice:** container isolation is weaker because of kernel sharing; hence production commonly runs **containers *on* VMs** — VMs provide the security boundary, containers provide density and agility.

---

### Q13. What do Linux namespaces and cgroups each provide to containers? Why are both mechanisms necessary? (5)

**Answer:**

**Namespaces — isolation of *visibility*:** each namespace type gives a container its own isolated *view* of the system: **PID** (own process tree — hence `ps` inside a container shows only its processes), **NET** (own network stack: interfaces, IPs, routes), **MNT** (own mount tree/root filesystem), **UTS** (own hostname), **IPC** (isolated inter-process communication), and **USER** (mapped identities). This is what makes every container perceive itself as a standalone machine.

**cgroups — control of *consumption*:** control groups **limit, account for, and prioritise** resource usage (CPU time, memory, block I/O) per group of processes — e.g., `--cpus=1.5 --memory=2g` bounds a container's share, and a memory limit triggers OOM-kill before the host is destabilised.

**Why both are necessary:** they answer different questions — namespaces answer *"what can the container see?"*; cgroups answer *"how much can it use?"*
- Isolation **without** cgroups → a single container can starve all neighbours (noisy-neighbour problem).
- Limits **without** isolation → processes interfere with and observe each other (insecure).

Production containers require the pairing; this is why containers deliver "VM-like" behaviour at process-level cost. *(Common trap: trying to fix a resource-hog container with more namespaces — only cgroups control consumption.)*

---

### Q14. Distinguish a Docker image from a Docker container, and explain the purpose of any five Dockerfile instructions. Why does instruction ordering matter for build performance? (5)

**Answer:**

**Image vs. container:** a **Docker image** is a read-only, immutable **template** composed of stacked layers (base OS, dependencies, application). A **container** is a **running instance** of an image plus one thin **writable layer** on top. Changes made inside a running container go to its writable layer only — underlying image layers are never altered, and the writable layer is discarded when the container is removed. One image spawns many identical containers; persisting changes requires committing a new image or using volumes.

**Five Dockerfile instructions:**
1. **FROM** — selects the base image (e.g., `ubuntu:22.04`).
2. **RUN** — executes commands at build time (e.g., install packages), creating a layer.
3. **WORKDIR** — sets the working directory for subsequent instructions.
4. **COPY** — copies files from the build context into the image.
5. **EXPOSE / CMD** — documents the listening port / sets the default command when the container starts.

**Ordering matters (layer caching):** each instruction is a cached layer; if an instruction's inputs change, that layer **and all layers after it** rebuild. Copying dependency manifests (`package.json`, `requirements.txt`) and installing dependencies **before** copying bulk source code means routine code edits invalidate only the final COPY layer — builds take seconds instead of minutes.

---

### Q15. What role do Docker volumes play for container restarts? How do they impact the overall portability and efficiency when multiple Docker containers are deployed in a production environment? (4, 2+2) *(Sample-paper pattern)*

**Answer:**

**Role in restarts (2):** a **volume** is persistent storage managed by Docker that exists **independent of the container lifecycle**. Data written inside a container's writable layer is destroyed when the container stops/removes/is recreated; a volume mounted at the data path (e.g., `/var/lib/mysql`) survives container restart, stop, and removal — making stateful containers (databases) restart-safe.

**Impact on portability and efficiency (2):**
- **Portability:** volumes decouple *state* from *containers*, so containers remain ephemeral and replaceable while data is explicitly managed (backed up, snapshotted, migrated). Multiple containers can share common data, and replacing a container version does not touch the data.
- **Efficiency:** volume writes **bypass the container's copy-on-write overlay filesystem**, avoiding its performance overhead — significant for I/O-heavy workloads like databases in multi-container production.

**Trade-off:** volumes introduce their own lifecycle management (backup, retention, access control); host-path bind mounts can reduce portability if they encode host-specific layout.

---

### Q16. (a) What is a container registry and why is it essential in a CI/CD pipeline? (b) Distinguish system containers from application containers. (2+2)

**Answer:**

**(a) Registry (2):** a container registry is a **centralised service that stores and versions images** (Docker Hub for public; private registries for proprietary code, with push/pull access control). In CI/CD, the pipeline pushes each build as a uniquely tagged image once, and every environment (dev/test/prod) **pulls the identical tagged image** — the registry becomes the single immutable source of truth, eliminating "slightly different builds on different servers" and making rollback as simple as re-running a previous tag.

**(b) System vs. application containers (2):** *system containers* (LXC-style) behave like lightweight VMs — they run an **init system and multiple services** inside a VM-like environment while still sharing the host kernel. *Application containers* (the Docker model) package **a single application/process**, are typically short-lived/ephemeral, and suit microservices. The choice mirrors workload shape: full machine-like environments → system containers; one-process, horizontally scaled services → application containers.

---

### Q17. (a) Correct the misconception that "Kubernetes is a container" and explain four capabilities it provides. (b) State four cloud-native design principles and explain why statelessness matters. (3+3)

**Answer:**

**(a) Kubernetes (3):** Kubernetes is **not a container — it is a container orchestrator**: a control system that manages *fleets* of containers across a cluster of hosts. Capabilities:
1. **Declarative desired state** — you declare "6 replicas of service X"; the system continuously reconciles reality to match.
2. **Self-healing** — detects failed containers/nodes and automatically restarts or reschedules them elsewhere.
3. **Automated scheduling & horizontal scaling** — places containers based on resource requests and scales replicas on demand.
4. **Service discovery, load balancing, rolling updates/rollbacks** — stable virtual endpoints distribute traffic; new versions deploy incrementally with no downtime.

**(b) Cloud-native principles (3):** (1) **microservices** — independently deployable, loosely coupled services; (2) **containerization** — immutable packaged images; (3) **orchestration** — automated deployment/scaling/healing; (4) **design for failure** — health checks, replication, retries.

**Why statelessness:** if sessions/state are externalised (not held in the instance), **any replica can serve any request** — making horizontal scaling safe and allowing Kubernetes to kill and replace any instance without losing user data. Statefulness is pushed to databases/volumes.

---

# Module 3 — IaaS (AWS as Reference Platform)

### Q18. (a) What does IaaS provide, and what is its key benefit to the user? (b) Describe the five layers of the IaaS architecture stack and the cross-cutting concerns. (3+3)

**Answer:**

**(a) IaaS (3):** IaaS delivers the **foundational building blocks** to run IT workloads in the cloud — **servers (compute), storage, network, and the operating system** — distributed as a service. Characteristics: dynamic scaling (1→10→100), variable cost, multi-tenancy on shared hardware. **Benefit:** instead of *purchasing* servers, software, data-centre space, or network equipment, the user **rents** them as a fully **outsourced, on-demand service** — reducing CapEx, increasing agility, and letting the organization focus on innovation, not infrastructure. The **enabler is virtualization technology**.

**(b) Stack (bottom → top):**
1. **Physical Infrastructure Layer** — actual capital asset base: servers, storage arrays, network switches/routers, power/cooling, data-centre facilities.
2. **Virtualization Layer** — bare-metal (Type-1) hypervisor (KVM, Xen, AWS Nitro) abstracting physical resources into isolated virtual environments (virtual CPU/RAM/disk/NIC).
3. **Resource Pool / Abstraction Layer** — aggregates virtual resources into **compute, storage, and network pools** (vCPUs, vDisks, vNICs) enabling flexible, location-independent allocation.
4. **Core Management / Orchestration Layer (Cloud Manager)** — VM lifecycle management (provision/migrate/terminate), scheduler, capacity manager, SLA monitor, metering & billing engine.
5. **Service / Access Layer (Consumer Touchpoint)** — self-service portal, REST APIs, CLI, SDKs — the only layer consumers interact with directly.

**Cross-cutting concerns (span all layers):** IAM, network security (VPC, firewalls, security groups), monitoring & logging (metrics/logs/alerts), and billing & chargeback.

---

### Q19. Name the five NIST IaaS actors and state each one's role. (4)

**Answer:**
1. **Cloud Consumer** — uses cloud services to run applications/workloads; selects services, negotiates SLAs, pays for usage.
2. **Cloud Provider** — builds, owns, and operates the IaaS infrastructure and delivers services; responsible for availability, performance, security, operations, metering, and billing.
3. **Cloud Broker** — intermediary that helps consumers select, integrate, and manage services across one or more providers; canonical functions: **service intermediation, service aggregation, service arbitrage**.
4. **Cloud Auditor** — independent assessor that evaluates security, privacy, performance, compliance, and SLA verification, providing audit reports and assurance.
5. **Cloud Carrier** — provides **connectivity and transport** between consumers, providers, and data centres (bandwidth, QoS, inter-datacentre links).

**Key takeaway:** consumers use, providers build and run, brokers add value, auditors assure trust, carriers connect everything.

---

### Q20. (a) Distinguish an EC2 Instance Type from an AMI. (b) List the four ways to obtain an AMI and state one characteristic of each. (2+3)

**Answer:**

**(a) Distinction (2):** two concepts are key to launching an EC2 instance:
- **Instance Type** = the amount of **virtual hardware** dedicated to the instance — vCPUs, memory, storage, network performance. *Analogy: the processor.* Types are grouped into families by resource ratios (t = burstable/tiny; m = general-purpose/balanced; c = compute-optimized; r = memory-optimized/in-memory DBs; x = extreme memory; p = GPU; i/d = storage-optimized), and sizes scale up linearly within a family.
- **AMI (Amazon Machine Image)** = the **software state** of the instance at launch — the OS and its configuration, initial patch level, and any pre-installed application/system software. *Analogy: the OS.*

**(b) Four AMI sources (3):**
1. **Published by AWS** — many Linux/Windows distributions with default OS settings; patches must be applied immediately after launch.
2. **AWS Marketplace** — online store of partner software pre-configured to run on EC2; customer need not install software, licenses are cloud-appropriate; cost = instance rate + optional per-hour software fee.
3. **Generated from an existing instance** — launch, configure to corporate standards, then create a new AMI from it ("golden AMI"); all future instances inherit the standard, reducing drift.
4. **Uploaded virtual servers** — VM Import/Export converts existing on-premises VM images in raw, VMDK, VHD, or OVA formats into AMIs.

---

### Q21. Distinguish AWS Regions, Availability Zones, and Edge Locations, and explain how distributing instances across AZs protects an application. (5)

**Answer:**
- **Region (2):** a **separate geographic area** (e.g., Mumbai, Frankfurt) grouping a cluster of nearby AZs. Each Region is **completely independent** — any service you launch runs only in your current default Region (changeable via console, `AWS_DEFAULT_REGION`, or `--region`). Region isolation achieves the greatest possible fault tolerance and stability. Selection factors: latency to users, compliance/data residency, cost, and service availability.
- **Availability Zone (2):** one or more **discrete data centres** within a Region, each with redundant power, networking, and connectivity, housed in **separate facilities** — physically separated (lower-risk flood plains, independent power grids, multiple internet connections) yet connected by **low-latency links**. AZ code = Region code + letter (e.g., us-east-1a).
- **Edge Location (1):** content-delivery points (CloudFront POPs) placed close to users that **cache content** (video, images, software updates) for fast delivery — they do **not** run your compute or store your primary data.

**Protection mechanism:** if instances are **distributed across multiple AZs** and one instance/AZ fails, instances in **another AZ can handle the requests** — placing resources in separate AZs protects the application from a service disruption impacting a single location.

---

### Q22. What is a VPC? List its core components. Distinguish public and private subnets, and security groups from network ACLs. (5)

**Answer:** A **Virtual Private Cloud** is a virtual network **dedicated to your AWS account**, logically isolated from other virtual networks, within which you launch resources. You configure its **IP address range, subnets, route tables, gateways, and security settings**.

**Core components:** subnets, route tables, DHCP option sets, security groups, network ACLs. *(Optional: Internet Gateways, Elastic IPs, ENIs, endpoints, peering, NAT gateways, Virtual Private Gateway/Customer Gateway/VPN.)*

**Public vs. private subnets:** a **public subnet** is for resources that **must be connected to the Internet** (routed via an Internet Gateway — the default VPC's subnets are all public, and instances launched there get both private and public IPv4 addresses). A **private subnet** is for resources that **won't be Internet-connected**; instances get a private IPv4 only and can communicate within the VPC but not with the Internet (a **NAT gateway** enables outbound-only Internet access, e.g., for patching).

**Security groups vs. NACLs:** **security groups** are **stateful, instance-level** firewalls — return traffic is automatically allowed, and only *allow* rules exist (e.g., allow DB port only from the app tier). **NACLs** are **stateless, subnet-level** filters supporting both allow *and* deny rules, evaluated in numbered order. Used together they provide defence in depth at two layers.

---

### Q23. Compare Amazon S3, Amazon EBS, and Amazon EFS across storage type, access model, durability scope, and best-fit use cases. (5)

**Answer:**

| Criterion | Amazon S3 | Amazon EBS | Amazon EFS |
|---|---|---|---|
| **Storage type** | Object storage | Block storage | File storage |
| **Access model** | Server-independent; accessed over the Internet via HTTP-verb API; **whole-object GET/PUT** (no incremental updates); objects with unique keys in flat buckets (no sub-buckets) | Block device attached to EC2; **one volume → one instance at a time** (multiple volumes → one instance allowed) | Shared file system (NFS) mounted by **many servers simultaneously** |
| **Capacity/pricing** | Unlimited objects; pay only for storage used; eliminates capacity planning | Volume-sized (GP SSD 1 GB–16 TB, 3 IOPS/GB capped at 10,000; PIOPS 4 GB–16 TB for sustained IOPS); pay for provisioned capacity | Elastic — scales up/down on the fly; pay for what you use |
| **Durability scope** | Automatically replicated across **multiple devices and facilities within a Region** (multi-AZ) | Replicated **within its Availability Zone** (protects against component failure, not AZ failure) | Redundantly across multiple AZs |
| **Best for** | Backup/archive, content/media distribution, static websites, big-data analytics, disaster recovery | Boot volumes, SQL/NoSQL databases, transaction processing, data warehouses | Shared web content, lift-and-shift applications, home directories, big-data analytics |

**Justification pattern:** match the *access model* to the workload — many-consumer immutable objects → S3; low-latency private disk for one instance → EBS; concurrent shared file access by many servers → EFS. (For rarely accessed archives, **Amazon Glacier** provides extremely low-cost storage where a 3–5 hour retrieval time is acceptable.)

---

### Q24. Distinguish RDS Multi-AZ deployments from Read Replicas. Can they be used together? (4)

**Answer:**

| Aspect | Multi-AZ Deployment | Read Replica |
|---|---|---|
| Replication | **Synchronous** standby maintained in a **second AZ** | **Asynchronous**, read-only copies (same-Region or cross-Region) |
| Purpose | **Availability/durability** — automatic failover; explicitly **not** a performance feature | **Performance** — scale **read-heavy workloads** by distributing SELECT load |
| Serves traffic? | No (standby, promoted only on failover) | Yes (read queries) |
| Limits | One synchronous standby | Up to 5 for most engines, up to 15 for Aurora |

**Together:** yes — they solve different problems and are commonly combined: Multi-AZ gives automatic failover when the primary AZ fails; Read Replicas absorb event-driven read surges. Additionally, RDS **automated backups** (default on, 1–35 days retention) enable point-in-time recovery to the second — a third, distinct mechanism often confused with the other two.

---

### Q25. Explain the AWS shared responsibility model. How do authentication, authorization, roles, and policies together implement the customer's side of it? (6, 3+3)

**Answer:**

**(a) Shared responsibility (3):** **AWS is responsible for security "OF" the cloud** — the hardware/global infrastructure (Regions, Availability Zones, Edge Locations) and the software layer it operates (compute, storage, database, networking services). **The customer is responsible for security "IN" the cloud** — customer data, platform and applications, identity & access management, and (for IaaS) the **operating system, network, and firewall configuration**, including client/server-side encryption choices. *Example allocation:* an unpatched guest OS on EC2 is a **customer** failure (AWS secures the hypervisor, not the guest); a publicly readable S3 bucket is likewise a **customer** misconfiguration.

**(b) IAM mechanisms implementing the customer's side (3):**
- **Authentication ("Who are you?")** — verifies identity via credentials (password + MFA, SSH key pairs, temporary tokens, SSO) against a trusted store; happens once per session and grants *no* permissions by itself.
- **Authorization ("What are you allowed to do?")** — on **every request**, policies are evaluated to allow or deny actions; enforces **least privilege**; an **explicit Deny always overrides an Allow**; unlike authentication, it is evaluated continuously.
- **Roles** — sets of permissions **assumable without permanent credentials** (no long-term password/access key); an identity or service assumes a role and receives **temporary credentials** (minutes–hours, auto-rotating) — eliminating hardcoded keys and enabling secure automation (e.g., EC2 reading a specific S3 bucket via its instance role).
- **Policies** — the actual **enforceable rulebook** (JSON): each statement specifies **Effect (Allow/Deny), Action, Resource, and optional Condition** (e.g., IP or time of day), converting organizational rules into machine-checked decisions at any scale.

---

## Quick Tips for Answering Theory Questions in This Paper

1. **Justify or lose marks** — the paper explicitly says "Without justification no marks." Every identification/classification answer needs a *why* sentence.
2. **Use the layer/responsibility test** — for service models, VPC tiers, and shared responsibility, answers built on "who manages which layer" are always accepted.
3. **Demerits must come with mitigations** — Q1 of the sample paper awards (3+2) only when each demerit is justified *and* remedied.
4. **Learn the numbers that carry marks** — 3-4-5 NIST; rings 0–3; ~17 sensitive instructions; 1998/2005–06; 1:1–3:1–5:1–>8:1; ~5% CPU Ready; 4 reclamation techniques; 3–5 hours Glacier; up to 5 replicas (15 Aurora); 1–35 day backups.
5. **Tables for comparisons, prose for mechanisms** — examiners award structure; a comparison asked as a table (VM vs container, S3/EBS/EFS, Multi-AZ vs replicas) is faster to write and harder to mark down.
6. **State assumptions** — required by the cover instructions ("Assumptions made if any, should be stated clearly"), especially in numerical questions.
