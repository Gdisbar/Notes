# Cloud Computing — 50 Scenario-Based Questions & Answers

---

## Section 1 — Medium (Questions 1–10)

---

### Q1. [Service Model Selection — SaaS vs IaaS]

**Scenario:** A 12-person accounting firm wants to move its email, document collaboration, and customer relationship management (CRM) to the cloud. The firm has no IT staff and wants minimal operational responsibility. A consultant suggests either subscribing to Microsoft 365 and Salesforce (SaaS) or renting virtual machines on AWS and installing open-source alternatives (IaaS).

**Answer:** The firm should choose the **SaaS** option (Microsoft 365 + Salesforce).

**Why:** SaaS delivers fully managed, ready-to-use applications over the internet. The firm does not manage servers, operating systems, patching, or application updates — the provider handles everything. This aligns with the firm's lack of IT staff and desire for minimal operational responsibility.

**Technical reasoning:** Under the NIST model, SaaS provides the consumer with access to the provider's applications running on cloud infrastructure, accessible via a thin client (browser). The consumer manages only user-specific configuration, not the underlying stack. IaaS, by contrast, would require the firm to manage the OS, middleware, runtime, and applications — essentially replicating an on-premises data center in the cloud without the staff to run it.

**Trade-offs:** SaaS offers less customization and control. Data residency and integration with bespoke systems may be constrained. Vendor lock-in is a real risk. However, for a small firm with standard business needs, these trade-offs are acceptable.

**Industry relevance:** This is the classic "no IT staff, standard business apps" scenario where SaaS wins. Most small businesses today run on SaaS for exactly this reason.

**Key concepts tested:** SaaS vs IaaS, operational responsibility, NIST service models.

---

### Q2. [Virtualization — Server Consolidation]

**Scenario:** A manufacturing company runs 40 applications on 40 separate physical servers. Monitoring shows average CPU utilization is 12% and memory utilization is 18%. The company wants to reduce hardware, power, and cooling costs while keeping applications isolated from one another.

**Answer:** The company should adopt **server virtualization** using a Type 1 (bare-metal) hypervisor such as VMware ESXi or KVM.

**Why:** Server virtualization allows multiple virtual machines (VMs) to run on a single physical host, each with its own guest OS and application, isolated from the others. Consolidating 40 underutilized servers onto a few powerful hosts dramatically improves resource utilization and reduces physical footprint.

**Technical reasoning:** A Type 1 hypervisor runs directly on the hardware and multiplexes CPU, memory, storage, and network resources across VMs. Each VM believes it owns the hardware. The hypervisor enforces isolation via privileged instruction trapping and memory partitioning. This directly addresses the 12–18% utilization problem — the same hardware now serves many workloads.

**Trade-offs:** Consolidation creates a single point of failure if not designed with redundancy. Licensing costs for guest OSes and applications still apply per VM. IT staff need virtualization training. Not all legacy applications virtualize well (e.g., those needing specific hardware dongles).

**Industry relevance:** This is the classic "server sprawl" problem that drove VMware's rise in the 2000s and remains the primary driver of data center consolidation projects.

**Key concepts tested:** Server virtualization, Type 1 hypervisor, resource utilization, isolation, consolidation.

---

### Q3. [IaaS — EC2 Instance Type Selection]

**Scenario:** A fintech startup is deploying a high-frequency trading analytics engine on AWS EC2. The workload requires very high memory bandwidth, large in-memory datasets (over 500 GB), and consistent low-latency performance. The team is choosing between the `t4g` (burstable general purpose) and `r7g` (memory optimized) instance families.

**Answer:** The team should choose the **`r7g` memory-optimized** instance family.

**Why:** The `r7g` family is designed for memory-intensive workloads such as in-memory databases and large analytics. The `t4g` family is burstable — it provides a baseline CPU with the ability to burst, but it is not designed for sustained high-memory, low-latency workloads. Using `t4g` would risk CPU throttling and insufficient memory.

**Technical reasoning:** AWS instance families are grouped by the ratio of compute to memory. The `r` prefix stands for RAM, and `r7g` instances use Graviton3 (ARM) processors with a high memory-to-vCPU ratio. For a 500 GB in-memory dataset, the instance must have enough RAM to hold the working set — otherwise the application will swap to disk and latency will spike. The `t4g` family maxes out at much lower memory and is explicitly "burstable," meaning sustained load exhausts CPU credits.

**Trade-offs:** `r7g` is more expensive per hour than `t4g`. If the workload were variable and CPU-light, `t4g` would be more cost-effective. Also, ARM-based instances require the application to be compiled for ARM — a migration consideration.

**Industry relevance:** Choosing the wrong instance family is one of the most common and costly mistakes in cloud deployments. Memory-optimized instances are standard for Redis, SAP HANA, and real-time analytics.

**Key concepts tested:** EC2 instance families, memory optimization, burstable vs sustained performance, workload-instance matching.

---

### Q4. [Dockerfile — Build Instructions]

**Scenario:** A team is containerizing a Node.js web application. The application requires Node.js 14.16.1 and Express 4.16.2. The base image should be Ubuntu 22.04. The working directory must be `/app`, and the `package.json` file should be copied first. After `npm install`, the rest of the code should be copied, and the container should expose port 3232.

**Answer:** A Dockerfile that achieves this is:

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_14.x | bash - && \
    apt-get install -y nodejs

WORKDIR /app

COPY package.json /app/

RUN npm install express@4.16.2

COPY . /app

EXPOSE 3232
```

**Why:** The Dockerfile follows the standard layering principle: start from a base image (`ubuntu:22.04`), install the required runtime (Node.js 14.16.1 via NodeSource), set the working directory, copy `package.json` first to leverage Docker's layer caching, install dependencies, copy the remaining code, and declare the exposed port.

**Technical reasoning:** Copying `package.json` before the rest of the code ensures that if only application code changes, Docker can reuse the cached `npm install` layer. This dramatically speeds up rebuilds. `EXPOSE 3232` documents the intended port but does not actually publish it — that is done at runtime with `-p`. The `RUN` command installs the specific Express version. In practice, `npm install` would read `package.json`, but the question specifies the dependency explicitly.

**Trade-offs:** Using `ubuntu:22.04` as the base produces a large image (~70 MB+ before Node.js). A `node:14.16.1` base image would be more efficient. Also, `COPY . /app` can accidentally copy `node_modules` from the host unless a `.dockerignore` is used.

**Industry relevance:** This is the canonical multi-stage Docker build pattern used in CI/CD pipelines for Node.js applications.

**Key concepts tested:** Dockerfile syntax, layer caching, base images, WORKDIR, COPY, RUN, EXPOSE.

---

### Q5. [Docker Volumes — Persistence]

**Scenario:** A containerized PostgreSQL database is running in a Docker container. The team notices that every time the container is restarted, all database data is lost. The application works fine on the developer's laptop because the developer never restarts the container.

**Answer:** The team must use **Docker volumes** (or bind mounts) to persist data outside the container's writable layer.

**Why:** By default, data written inside a container is stored in the container's writable layer, which is tied to the container lifecycle. When the container is removed or recreated, that layer is destroyed. Docker volumes store data on the host filesystem (or a remote storage backend) and are mounted into the container at a specified path, surviving container restarts and removals.

**Technical reasoning:** A Docker volume is managed by Docker and stored in `/var/lib/docker/volumes/` on Linux. It can be shared between containers and persists independently of any single container. For a database, the volume would be mounted at the database's data directory (e.g., `/var/lib/postgresql/data`). The command would be `docker run -v pgdata:/var/lib/postgresql/data postgres`.

**Trade-offs:** Volumes introduce a dependency on the host's storage. If the host fails, the volume may be lost unless backed by network storage or replicated. Volume drivers are needed for multi-host environments (e.g., cloud storage). Bind mounts offer more direct host control but are less portable.

**Industry relevance:** Stateful containers (databases, message queues) always require persistent volumes. This is a fundamental pattern in containerized production environments.

**Key concepts tested:** Docker volumes, container lifecycle, persistence, stateful containers.

---

### Q6. [IaaS — Regions and Availability Zones]

**Scenario:** A healthcare SaaS company must deploy its application on AWS. Regulatory requirements mandate that patient data must remain within the country. The company also wants the application to survive the failure of a single data center.

**Answer:** The company should deploy in an **AWS Region within the required country** and distribute resources across **multiple Availability Zones (AZs)** within that Region.

**Why:** An AWS Region is a geographic area (e.g., Mumbai, Frankfurt) that contains multiple isolated AZs. Data stored in a Region stays in that Region unless explicitly moved. Deploying across multiple AZs within the same Region provides fault tolerance: if one AZ fails (power, cooling, network), the application can continue running in another AZ.

**Technical reasoning:** Each AZ is one or more discrete data centers with independent power, cooling, and networking. AZs within a Region are connected by low-latency links. By placing instances in multiple AZs and using a load balancer, the application survives an AZ-level outage. For data residency, the Region choice ensures compliance — e.g., `ap-south-1` (Mumbai) for India.

**Trade-offs:** Multi-AZ deployment increases cost (duplicate resources) and complexity (data replication, latency between AZs). Cross-Region replication would add even more cost but provides disaster recovery. The company must balance regulatory constraints with availability requirements.

**Industry relevance:** This is the standard high-availability pattern for regulated industries. AWS explicitly recommends multi-AZ deployments for production workloads.

**Key concepts tested:** AWS Regions, Availability Zones, fault tolerance, data residency, high availability.

---

### Q7. [PaaS — Application Deployment]

**Scenario:** A startup wants to deploy a Python Django web application. The team has no experience managing servers, operating systems, or scaling infrastructure. They want to push code and have the platform handle provisioning, scaling, and patching. They are considering AWS Elastic Beanstalk (PaaS) versus EC2 (IaaS).

**Answer:** The startup should choose **AWS Elastic Beanstalk (PaaS)** .

**Why:** PaaS abstracts away the underlying infrastructure — the team deploys code, and the platform handles capacity provisioning, load balancing, auto-scaling, and application health monitoring. This matches the team's lack of server management experience and desire to focus on code.

**Technical reasoning:** In the NIST model, PaaS provides a platform where the consumer deploys applications created using supported programming languages and tools. The consumer does not manage the underlying network, servers, OS, or storage. Elastic Beanstalk supports Python/Django, automatically provisions EC2 instances, sets up load balancers, and scales based on demand. With IaaS (raw EC2), the team would have to install Python, configure the web server, manage security groups, set up auto-scaling, and patch the OS — all of which they are not equipped to do.

**Trade-offs:** PaaS offers less control over the runtime environment. The team cannot customize the OS or install arbitrary system-level packages. There is also vendor lock-in — migrating off Elastic Beanstalk requires re-architecting. For highly specialized workloads, IaaS may be necessary.

**Industry relevance:** PaaS is the standard choice for startups and teams that want to focus on application development rather than infrastructure operations.

**Key concepts tested:** PaaS vs IaaS, operational responsibility, AWS Elastic Beanstalk, auto-scaling.

---

### Q8. [Virtualization — Type 1 vs Type 2 Hypervisor]

**Scenario:** A software developer wants to run a Linux VM on their Windows 10 laptop to test a cross-platform application. They also have a production server that will host 30 VMs for a private cloud. The developer asks whether the same hypervisor can be used for both.

**Answer:** No — the developer should use a **Type 2 (hosted) hypervisor** on the laptop (e.g., Oracle VirtualBox, VMware Workstation) and a **Type 1 (bare-metal) hypervisor** on the production server (e.g., VMware ESXi, KVM, Hyper-V).

**Why:** Type 2 hypervisors run as an application on top of a host OS, making them easy to install and use on a personal computer. Type 1 hypervisors run directly on the hardware, providing better performance, isolation, and security — essential for production workloads hosting many VMs.

**Technical reasoning:** A Type 2 hypervisor relies on the host OS for device drivers, memory management, and scheduling, which adds overhead and depends on the host OS's stability. A Type 1 hypervisor has its own lightweight kernel and directly manages hardware, so it can efficiently multiplex resources across many VMs. In a private cloud with 30 VMs, the overhead and security risks of a Type 2 hypervisor would be unacceptable.

**Trade-offs:** Type 1 hypervisors require dedicated hardware and are more complex to set up. Type 2 hypervisors are convenient for development and testing but not suitable for production-scale virtualization.

**Industry relevance:** This is a fundamental architectural decision in any virtualization project. Data centers run Type 1; developers run Type 2.

**Key concepts tested:** Type 1 vs Type 2 hypervisors, bare-metal vs hosted, production vs development.

---

### Q9. [Cloud Deployment Models — Hybrid Cloud]

**Scenario:** A bank runs its core transaction processing on-premises for regulatory and latency reasons. However, it wants to use the public cloud for a new customer-facing mobile banking app that experiences unpredictable traffic. The bank wants the app to scale elastically without investing in new hardware.

**Answer:** The bank should adopt a **hybrid cloud** deployment model.

**Why:** A hybrid cloud combines private/on-premises infrastructure with public cloud, allowing workloads to run where they make the most sense. Sensitive core banking stays on-premises; the variable-traffic mobile app runs in the public cloud and scales elastically.

**Technical reasoning:** The hybrid model connects the two environments via secure networking (VPN, dedicated interconnect). The public cloud provides rapid elasticity and pay-per-use for the mobile app, while the private environment maintains control over regulated data. This is sometimes called "cloud bursting" — the ability to extend capacity to the public cloud during demand spikes.

**Trade-offs:** Hybrid cloud introduces integration complexity — networking, identity management, data synchronization, and security policies must span both environments. It requires skilled staff to manage. It is more expensive than a pure public cloud for the cloud portion, but cheaper than building private capacity for peak loads.

**Industry relevance:** Hybrid cloud is the dominant model in banking, healthcare, and government, where data sensitivity and regulatory compliance coexist with the need for cloud agility.

**Key concepts tested:** Hybrid cloud, cloud bursting, deployment models, regulatory constraints.

---

### Q10. [Containers vs VMs — Isolation and Overhead]

**Scenario:** A company is deploying 50 lightweight microservices. Each microservice is a small Node.js application. The company is deciding between running each microservice in its own VM or in its own container. The VMs would each need a full OS, while containers would share the host OS kernel.

**Answer:** The company should use **containers** for the 50 microservices.

**Why:** Containers are far more lightweight than VMs because they share the host OS kernel and do not require a full guest OS per instance. For 50 small microservices, containers provide faster startup, lower memory overhead, and higher density on the same hardware.

**Technical reasoning:** A VM includes a full OS (kernel + user space), consuming gigabytes of storage and hundreds of MB of RAM at idle. A container packages only the application and its dependencies, sharing the host kernel. Containers start in seconds (or milliseconds) versus minutes for VMs. For 50 microservices, containers allow all services to run on a single host or a small cluster, whereas VMs would require significantly more hardware.

**Trade-offs:** Containers provide weaker isolation than VMs — a kernel exploit could affect other containers on the same host. VMs offer stronger security boundaries. Containers are also less suitable for running different OS kernels (e.g., Windows and Linux on the same host without virtualization). For workloads requiring strong isolation or different OS kernels, VMs are preferable.

**Industry relevance:** This is the core driver of the container revolution. Microservices architectures overwhelmingly use containers (Docker, Kubernetes) for exactly this density and startup advantage.

**Key concepts tested:** Containers vs VMs, kernel sharing, density, isolation, microservices.

---

## Section 2 — Medium-High (Questions 11–35)

---

### Q11. [Docker — Image vs Container]

**Scenario:** A developer builds a Docker image for a web application and runs 10 containers from it. The developer then modifies a file inside one of the running containers. The next day, the developer restarts that container and finds the modification is gone. The developer is confused because they thought the image was updated.

**Answer:** The developer has confused a **Docker image** with a **Docker container**. Modifying a file inside a running container does not change the image. When the container is restarted (or recreated), it reverts to the image's state.

**Why:** A Docker image is a read-only template. A container is a running instance of that image with a thin writable layer on top. Changes made inside a container are stored in that writable layer, which is tied to the container's lifecycle. If the container is removed and recreated, the writable layer is discarded.

**Technical reasoning:** Docker images are built from Dockerfiles and consist of immutable layers. When a container runs, Docker adds a writable layer. Any file modification uses copy-on-write: the file is copied from the read-only layer to the writable layer and modified there. The original image remains unchanged. To persist changes, the developer must either commit the container to a new image (`docker commit`) or use volumes for persistent data.

**Trade-offs:** The ephemeral nature of containers is a feature, not a bug — it ensures reproducibility and immutability. However, it requires developers to externalize state (databases, logs, uploads) into volumes or external services.

**Industry relevance:** This is one of the most common beginner mistakes in Docker. Understanding image vs container is fundamental to containerized workflows.

**Key concepts tested:** Docker image, Docker container, writable layer, copy-on-write, immutability.

---

### Q12. [Docker — Port Exposure vs Publishing]

**Scenario:** A developer writes a Dockerfile with `EXPOSE 8080` and runs the container with `docker run myapp`. The application inside the container listens on port 8080, but the developer cannot access it from the host browser at `localhost:8080`. The developer insists the Dockerfile is correct.

**Answer:** The Dockerfile is correct, but the container was run without **publishing** the port. `EXPOSE` only documents the intended port; it does not make it accessible from the host. The developer must run the container with `-p 8080:8080`.

**Why:** `EXPOSE` is metadata — it tells anyone reading the Dockerfile which port the application uses. It does not create a network mapping. To access the container from the host, the port must be published with `-p host_port:container_port`.

**Technical reasoning:** Docker containers have their own network namespace. By default, they are isolated from the host network. Publishing a port creates a NAT rule (via iptables on Linux) that forwards traffic from the host port to the container port. Without this, the container's port is only accessible from other containers on the same Docker network.

**Trade-offs:** Publishing ports exposes the application to the host network. In production, a reverse proxy or load balancer is often used instead of direct port publishing. Over-publishing ports is a security risk.

**Industry relevance:** This is a classic Docker networking mistake. In orchestrated environments (Kubernetes), Services and Ingress replace manual port publishing.

**Key concepts tested:** EXPOSE vs publish, Docker networking, port mapping, container isolation.

---

### Q13. [Namespaces and cgroups — Resource Control]

**Scenario:** A Docker host is running 15 containers. One container running a data processing job suddenly consumes all available CPU and memory, causing the other 14 containers to become unresponsive. The operations team wants to prevent any single container from monopolizing host resources.

**Answer:** The team should use **cgroups (control groups)** to limit CPU and memory for each container. They may also use **namespaces** to ensure process isolation, but cgroups are the mechanism for resource limiting.

**Why:** Namespaces provide isolation (each container sees its own process tree, network, filesystem), but they do not limit resource consumption. cgroups are the Linux kernel feature that limits, accounts for, and isolates resource usage (CPU, memory, disk I/O, network) for groups of processes. Docker uses cgroups under the hood when you specify `--cpus` or `--memory`.

**Technical reasoning:** When a container is started with `--memory="512m"` and `--cpus="1.0"`, Docker creates a cgroup with those limits. The kernel enforces them — if the container exceeds memory, it is OOM-killed; if it exceeds CPU, it is throttled. Without cgroup limits, a container can consume all host resources, starving others.

**Trade-offs:** Setting limits too low can cause legitimate workloads to be throttled or killed. Limits must be tuned to workload requirements. Also, cgroups limit resources but do not guarantee them unless reservations are used (e.g., `--memory-reservation`).

**Industry relevance:** In multi-tenant container platforms, cgroup-based resource limits are mandatory to prevent "noisy neighbor" problems. Kubernetes uses cgroups extensively for pod resource management.

**Key concepts tested:** Namespaces, cgroups, resource isolation, CPU/memory limits, noisy neighbor.

---

### Q14. [Kubernetes — Orchestration Need]

**Scenario:** A company has 20 Docker containers running on a single host. The host fails, and all 20 containers go down. The company wants to ensure that if a container or host fails, the containers are automatically rescheduled and restarted on healthy hardware. They also want to scale the number of container instances based on traffic.

**Answer:** The company should adopt a **container orchestrator** such as **Kubernetes**.

**Why:** Kubernetes manages a cluster of machines and schedules containers across them. It monitors container health, automatically restarts failed containers, reschedules them if a node fails, and can scale the number of replicas up or down based on CPU or custom metrics.

**Technical reasoning:** Kubernetes uses a control plane (API server, scheduler, controller manager, etcd) and worker nodes (kubelet, kube-proxy, container runtime). A Deployment defines the desired state (e.g., 5 replicas). The controller manager ensures the actual state matches. If a node fails, the scheduler reschedules pods onto healthy nodes. Horizontal Pod Autoscaler adjusts replica counts based on load.

**Trade-offs:** Kubernetes adds significant operational complexity — it is a distributed system that itself needs management. For a small number of containers on a single host, Docker Compose or a simpler orchestrator may suffice. Kubernetes is overkill for trivial deployments but essential for production-scale container workloads.

**Industry relevance:** Kubernetes is the de facto standard for container orchestration in production. It underpins cloud-native architectures across all major cloud providers.

**Key concepts tested:** Kubernetes, orchestration, self-healing, auto-scaling, desired state.

---

### Q15. [IaaS — VPC and Security Groups]

**Scenario:** A company has deployed a web application on EC2 instances in a VPC. The web servers must be accessible from the internet on port 443, but the database servers must only be accessible from the web servers on port 5432. The company also wants to ensure that no other traffic reaches the database.

**Answer:** The company should use **Security Groups** to control traffic. The web server security group allows inbound 443 from anywhere; the database security group allows inbound 5432 only from the web server security group.

**Why:** Security Groups are stateful virtual firewalls at the instance level. They control both inbound and outbound traffic. By referencing the web server security group as the source for the database security group, the company ensures that only web servers can reach the database — not the internet or other instances.

**Technical reasoning:** Security Groups operate at the ENI (Elastic Network Interface) level. They are allow-only rules — there are no explicit deny rules; anything not allowed is denied. Referencing another security group as a source is a powerful pattern: it means "any instance with this security group attached." This avoids hardcoding IP addresses and automatically adjusts as instances scale.

**Trade-offs:** Security Groups are stateful — return traffic is automatically allowed. Network ACLs (NACLs) are stateless and operate at the subnet level; they can be used as an additional layer but are less commonly needed. Misconfigured security groups are a leading cause of cloud data breaches.

**Industry relevance:** This is the fundamental network security pattern in AWS. Least-privilege security group rules are a core cloud security practice.

**Key concepts tested:** VPC, Security Groups, stateful vs stateless firewalls, least privilege.

---

### Q16. [Storage — S3 vs EBS vs EFS]

**Scenario:** A media company needs to store and serve millions of video files to users worldwide. The files are accessed via HTTP. The company also needs block storage for its EC2 instances' boot volumes and a shared file system for its rendering cluster. The company is unsure which AWS storage services to use.

**Answer:** The company should use **S3 for video files**, **EBS for EC2 boot volumes**, and **EFS for the shared file system**.

**Why:** S3 is object storage designed for massive scale, durability (99.999999999%), and HTTP access — ideal for media files served to users. EBS provides block storage volumes for EC2 instances, suitable for boot volumes and databases. EFS is a shared, elastic file system that can be mounted by multiple EC2 instances simultaneously — ideal for a rendering cluster that needs shared access to files.

**Technical reasoning:** S3 stores objects in buckets, each identified by a key, and is accessed via REST API/HTTP. It is not a file system and does not support partial updates. EBS volumes are attached to a single EC2 instance at a time and behave like physical disks. EFS is a POSIX-compliant file system that scales automatically and can be mounted on many instances concurrently.

**Trade-offs:** S3 has higher latency than EBS for small, random I/O. EBS is limited to a single AZ and a single instance (for most volume types). EFS is more expensive than S3 for large volumes of data. Choosing the wrong storage type leads to performance or cost problems.

**Industry relevance:** This is the classic "which storage service?" decision in AWS. Media companies use S3 + CloudFront for content delivery; EBS for databases; EFS for shared workloads.

**Key concepts tested:** S3 (object), EBS (block), EFS (file), storage characteristics, use case matching.

---

### Q17. [Cost Optimization — Reserved vs On-Demand]

**Scenario:** A company runs a steady-state production workload on 20 EC2 instances 24/7. The workload has been stable for 18 months and is expected to continue for at least 2 more years. The company is currently paying On-Demand prices and wants to reduce costs.

**Answer:** The company should purchase **Reserved Instances (RIs)** or **Savings Plans** for the steady-state capacity.

**Why:** On-Demand pricing is designed for unpredictable or short-term workloads. For a stable, long-term workload, Reserved Instances offer significant discounts (up to 72%) in exchange for a 1- or 3-year commitment. Savings Plans offer similar discounts with more flexibility across instance families and Regions.

**Technical reasoning:** AWS pricing models include On-Demand, Reserved, Spot, and Dedicated Hosts. Reserved Instances provide a billing discount, not a physical reservation (for regional RIs). The company commits to a certain amount of compute usage and receives a lower hourly rate. For 20 instances running 24/7 for 2+ years, the savings are substantial.

**Trade-offs:** Reserved Instances require a commitment — if the workload changes or the company migrates to different instance types, the reservation may go unused. Convertible RIs allow some flexibility but offer lower discounts. Spot Instances are cheaper but can be interrupted, making them unsuitable for steady-state production unless the workload is fault-tolerant.

**Industry relevance:** Cost optimization is a top priority in cloud operations. Reserved Instances and Savings Plans are standard tools for reducing EC2 costs.

**Key concepts tested:** EC2 pricing models, Reserved Instances, Savings Plans, cost optimization.

---

### Q18. [Cloud-Native — Stateless vs Stateful]

**Scenario:** A team is migrating a monolithic Java application to a cloud-native architecture. The application currently stores user session data in local memory and writes uploaded files to the local disk. The team wants to run the application in containers on Kubernetes and scale it horizontally.

**Answer:** The team must externalize state. Session data should be stored in a distributed cache (e.g., Redis) or a database, and uploaded files should be stored in object storage (e.g., S3). The application should be made **stateless**.

**Why:** Containers are ephemeral — they can be restarted, rescheduled, or scaled at any time. If session data is stored in local memory, a user's session is lost when the container restarts or when traffic is routed to a different replica. Similarly, local disk writes are lost when the container is replaced. Cloud-native applications are designed to be stateless so they can be scaled horizontally and recover from failures automatically.

**Technical reasoning:** In Kubernetes, pods are scheduled dynamically across nodes. A user's request may hit any replica. Without shared state, the application cannot maintain consistency. By moving session state to Redis and files to S3, any replica can serve any request. This is a core principle of the Twelve-Factor App methodology (Factor VI: Processes — execute the app as one or more stateless processes).

**Trade-offs:** Externalizing state adds latency (network calls to Redis/S3) and introduces new dependencies. It requires additional services (Redis, S3) and their management. However, the scalability and resilience benefits far outweigh these costs for cloud-native applications.

**Industry relevance:** This is the fundamental transformation required when moving from monolithic to microservices/cloud-native. Statelessness is a prerequisite for horizontal scaling and self-healing.

**Key concepts tested:** Cloud-native principles, stateless vs stateful, Twelve-Factor App, horizontal scaling, externalized state.

---

### Q19. [Virtualization — Binary Translation vs Hardware-Assisted]

**Scenario:** A company is running a legacy x86 application on a modern hypervisor. The application performs many privileged operations and runs in a VM. The company notices that the VM's performance is significantly lower than expected. The hypervisor is using binary translation. The host CPU supports Intel VT-x.

**Answer:** The company should enable **hardware-assisted virtualization** (Intel VT-x) instead of relying solely on binary translation.

**Why:** Binary translation dynamically rewrites sensitive instructions at runtime, which adds overhead. Hardware-assisted virtualization (Intel VT-x / AMD-V) adds a new privilege level (Ring -1) where the hypervisor runs, allowing the CPU to trap sensitive instructions automatically. This eliminates the need for binary translation for most instructions and delivers near-native performance.

**Technical reasoning:** On x86, certain instructions are "sensitive but not privileged" — they behave differently in different rings but do not trap. Binary translation works around this by rewriting code. Hardware-assisted virtualization solves the problem in silicon: the CPU itself traps these instructions when the guest runs in a non-root mode. The hypervisor can then handle them and return control. Modern hypervisors (KVM, Xen, Hyper-V, ESXi) use hardware assistance by default when available.

**Trade-offs:** Hardware-assisted virtualization requires CPU support (Intel VT-x or AMD-V). Older hardware without these extensions cannot use it. Also, hardware assistance may not eliminate all overhead — I/O and memory virtualization still have costs. However, for CPU-bound workloads with many privileged operations, the improvement is substantial.

**Industry relevance:** Since 2006, hardware-assisted virtualization has become the standard. Binary translation is now largely historical, though it remains relevant for specific use cases (e.g., nested virtualization on older hardware).

**Key concepts tested:** Binary translation, hardware-assisted virtualization, Intel VT-x, AMD-V, x86 virtualization challenges.

---

### Q20. [IaaS — AMI and Instance Launch]

**Scenario:** A company wants to launch 50 identical EC2 instances for a batch processing job. Each instance must have the same OS, the same application code, and the same configuration. The company wants to minimize launch time and ensure consistency across all instances.

**Answer:** The company should create a custom **Amazon Machine Image (AMI)** with the OS, application code, and configuration pre-installed, and launch all 50 instances from that AMI.

**Why:** An AMI captures the entire software state of an instance — OS, patches, applications, and configuration. Launching from a custom AMI ensures consistency across all instances and eliminates the need to install and configure software at launch time. This significantly reduces launch time and human error.

**Technical reasoning:** AMIs can be created from an existing EC2 instance (after configuring it to the desired state), from AWS-published AMIs, from the AWS Marketplace, or from uploaded virtual machine images. When an instance launches from an AMI, it boots with the exact software state captured in the image. For 50 identical instances, a custom AMI is the most efficient approach.

**Trade-offs:** Custom AMIs must be maintained — when the application or OS is updated, a new AMI must be created and instances relaunched. This can be automated with tools like EC2 Image Builder or Packer. Also, AMIs are Region-specific; to launch in multiple Regions, the AMI must be copied.

**Industry relevance:** Custom AMIs are a cornerstone of immutable infrastructure and auto-scaling. They are used extensively in CI/CD pipelines and infrastructure-as-code workflows.

**Key concepts tested:** AMI, instance consistency, immutable infrastructure, launch efficiency.

---

### Q21. [Deployment Models — Community Cloud]

**Scenario:** Five hospitals in the same state want to share a cloud-based electronic health records (EHR) system. They have common regulatory requirements (HIPAA-like), common security needs, and want to share the cost of infrastructure. They do not want to use a public cloud because of data sensitivity, but they cannot afford individual private clouds.

**Answer:** The hospitals should adopt a **community cloud**.

**Why:** A community cloud is shared by multiple organizations with common business, security, compliance, or operational requirements. It provides the cost-sharing benefits of a public cloud while restricting access to the participating organizations — satisfying both the budget and the privacy requirements.

**Technical reasoning:** The community cloud infrastructure may be owned, managed, and operated by the participating organizations or by a third-party provider. It is not open to the general public (unlike a public cloud) and is not dedicated to a single organization (unlike a private cloud). Governance and security standards are agreed upon by the participants.

**Trade-offs:** Community clouds require shared governance, which can be complex — allocating responsibility for security, cost, and operations across multiple independent organizations is challenging. Scalability is capped by the consortium's combined infrastructure. However, for hospitals with common needs, it is an excellent fit.

**Industry relevance:** Community clouds are used in healthcare, government, and education — sectors where organizations share compliance requirements but cannot use public multi-tenant infrastructure.

**Key concepts tested:** Community cloud, deployment models, shared governance, compliance.

---

### Q22. [Docker — Container Registry]

**Scenario:** A development team builds a Docker image on a developer's laptop. The team wants to deploy this image to a production Kubernetes cluster. The image is 2 GB in size. The team is considering copying the image file manually to the production server versus using a container registry.

**Answer:** The team should use a **container registry** (e.g., Docker Hub, Amazon ECR, Google GCR) to store and distribute the image.

**Why:** A container registry is a centralized repository for Docker images. It supports versioning (tags), access control, and efficient distribution. Manually copying image files is error-prone, does not scale, and lacks versioning. Kubernetes can pull images directly from a registry using an image pull secret.

**Technical reasoning:** When a Docker image is pushed to a registry, it is stored as layers. When a node pulls the image, only missing layers are downloaded — this is efficient for large images and updates. Registries also provide vulnerability scanning, image signing, and lifecycle policies. In Kubernetes, the `image:` field in a pod spec references the registry and tag (e.g., `myregistry/myapp:v1.2.3`).

**Trade-offs:** Using a public registry like Docker Hub may have rate limits and privacy concerns. Private registries (ECR, GCR, ACR) offer better control but add cost and configuration. Also, image pull latency depends on network bandwidth and registry proximity.

**Industry relevance:** Container registries are a core component of any containerized CI/CD pipeline. They enable reproducible deployments and rollbacks.

**Key concepts tested:** Container registry, Docker Hub, ECR, image distribution, Kubernetes image pull.

---

### Q23. [IaaS — Auto Scaling]

**Scenario:** An e-commerce website experiences traffic that varies significantly: low at night, high during the day, and extremely high during flash sales. The company currently runs a fixed number of EC2 instances sized for peak load, which is expensive during off-peak hours. The company wants to reduce costs while maintaining performance.

**Answer:** The company should implement **Auto Scaling** with a launch template or configuration that defines the instance type, AMI, and scaling policies.

**Why:** Auto Scaling automatically adjusts the number of EC2 instances based on demand. During low traffic, it terminates unnecessary instances; during high traffic, it launches new ones. This matches capacity to demand and reduces costs during off-peak periods.

**Technical reasoning:** Auto Scaling groups define minimum, maximum, and desired capacity. Scaling policies can be target tracking (e.g., maintain 50% CPU utilization), step scaling (e.g., add 2 instances when CPU > 70%), or scheduled scaling (e.g., scale up at 9 AM). The load balancer distributes traffic across instances. Health checks ensure unhealthy instances are replaced.

**Trade-offs:** Auto Scaling requires the application to be stateless (or to externalize state) so that any instance can handle any request. Scaling takes time (minutes), so it may not react instantly to sudden spikes. For extremely rapid spikes, over-provisioning or faster scaling policies may be needed.

**Industry relevance:** Auto Scaling is a fundamental cloud capability and a primary driver of cost efficiency. It is used by virtually every production web application on AWS.

**Key concepts tested:** Auto Scaling, elasticity, cost optimization, stateless design, load balancing.

---

### Q24. [Storage — Glacier for Archival]

**Scenario:** A legal firm must retain client documents for 10 years for compliance. The documents are rarely accessed — perhaps once a year for audit purposes. The firm wants to minimize storage costs but cannot delete the documents. The documents total 50 TB.

**Answer:** The firm should use **Amazon Glacier** (or S3 Glacier storage classes) for archival storage.

**Why:** Glacier is designed for long-term archival of infrequently accessed data. It offers extremely low storage costs (as low as $0.004/GB/month) but has retrieval times ranging from minutes to hours, depending on the retrieval option. For documents accessed once a year, this is ideal.

**Technical reasoning:** Glacier stores data in archives within vaults. Retrieval options include Expedited (1–5 minutes), Standard (3–5 hours), and Bulk (5–12 hours). Data is encrypted at rest and replicated across multiple facilities. The firm can use S3 Lifecycle Policies to automatically transition objects from S3 Standard to Glacier after a certain period.

**Trade-offs:** Retrieval is not instantaneous — if the firm needs a document urgently, Expedited retrieval costs more. Also, there is a minimum storage duration (90 days for Glacier Flexible Retrieval, 180 days for Deep Archive). Deleting data before the minimum duration incurs early deletion fees.

**Industry relevance:** Glacier is used extensively in healthcare, legal, finance, and media for compliance and archival. It is a key component of cost-effective data retention strategies.

**Key concepts tested:** Glacier, archival storage, retrieval options, lifecycle policies, compliance.

---

### Q25. [Virtualization — Memory Overcommitment]

**Scenario:** A virtualization administrator configures a host with 128 GB of physical RAM to run 20 VMs, each with 8 GB of configured memory (total 160 GB). The administrator enables memory overcommitment. Some VMs begin experiencing performance degradation and swapping.

**Answer:** The administrator should reduce the overcommit ratio or use memory reclamation techniques (ballooning, page sharing, compression) carefully. The overcommit ratio of 160:128 (1.25:1) is causing memory pressure.

**Why:** Memory overcommitment allows the hypervisor to allocate more virtual memory than physical RAM, betting that not all VMs will use their full allocation simultaneously. However, if active demand exceeds physical RAM, the hypervisor must reclaim memory — via ballooning (guest driver releases memory), page sharing (deduplicating identical pages), compression, or host-level swapping. Swapping is extremely slow and causes performance degradation.

**Technical reasoning:** The overcommit ratio is configured memory ÷ physical RAM. A ratio of 1.25:1 is generally considered safe if actual utilization is lower than configured. But if the working set exceeds 128 GB, reclamation kicks in. The administrator should monitor active memory usage and adjust either the number of VMs, their memory allocations, or add more RAM.

**Trade-offs:** Overcommitment improves density and reduces cost, but risks performance degradation under memory pressure. The safe ratio depends on workload characteristics — memory-intensive workloads (databases) should not be overcommitted aggressively. Monitoring is essential.

**Industry relevance:** Memory overcommitment is a standard feature in VMware, KVM, and Hyper-V. Proper capacity planning and monitoring are critical to avoid the "oversubscription" trap.

**Key concepts tested:** Memory overcommitment, ballooning, page sharing, swapping, capacity planning.

---

### Q26. [IaaS — Placement Groups]

**Scenario:** A company is deploying a high-performance computing (HPC) cluster on AWS. The cluster requires low-latency, high-bandwidth (10 Gbps) networking between instances. The instances must be physically close to each other to minimize latency.

**Answer:** The company should use a **Cluster Placement Group**.

**Why:** A Cluster Placement Group is a logical grouping of instances within a single Availability Zone that enables low-latency, high-throughput networking (up to 10 Gbps or higher with enhanced networking). This is ideal for HPC workloads that require tight coupling and fast inter-node communication.

**Technical reasoning:** Placement groups influence how instances are placed on underlying hardware. Cluster placement packs instances close together, often on the same rack or switch, to minimize network hops. Spread placement groups distribute instances across distinct hardware to reduce correlated failures. Partition placement groups divide instances into partitions for large distributed workloads (Hadoop, Cassandra).

**Trade-offs:** Cluster placement groups increase the risk of correlated failure — if the rack or switch fails, all instances are affected. They also may limit the ability to launch new instances if capacity is unavailable. Spread placement groups are better for high availability; cluster groups are better for performance.

**Industry relevance:** Placement groups are used in HPC, financial modeling, and big data workloads where network latency directly impacts job completion time.

**Key concepts tested:** Placement groups, cluster vs spread vs partition, low-latency networking, HPC.

---

### Q27. [Containers — System vs Application Containers]

**Scenario:** A company wants to run a lightweight Linux environment that behaves like a VM — with its own init system, multiple services, and persistent state — but without the overhead of a full VM. The company also wants to run individual microservices in isolated containers.

**Answer:** The company should use **system containers** for the VM-like environment and **application containers** for the microservices.

**Why:** System containers (e.g., LXC, LXD) provide a full OS environment with an init system and multiple processes — they are designed to feel like lightweight VMs. Application containers (e.g., Docker) are designed to run a single process or service. Using the right type for each use case is important.

**Technical reasoning:** System containers share the host kernel but provide a more complete OS user space, including init, systemd, and package management. They are useful for lifting-and-shifting legacy applications or for multi-service environments. Application containers follow the "one process per container" principle, making them ideal for microservices, easy to scale, and simple to manage.

**Trade-offs:** System containers are heavier than application containers and are less portable across different host OSes. Application containers are more lightweight but require the application to be designed for a single-process model. Mixing both on the same host is possible but adds management complexity.

**Industry relevance:** System containers are used in IaaS-like scenarios (e.g., LXD on Ubuntu) and for legacy migration. Application containers dominate in cloud-native and microservices architectures.

**Key concepts tested:** System containers, application containers, LXC, Docker, use case matching.

---

### Q28. [Cloud-Native — Microservices vs Monolith]

**Scenario:** A retail company has a monolithic e-commerce application. During peak season, the entire application must be scaled together, even though only the checkout and catalog modules are under heavy load. The company wants to scale components independently and deploy updates to one module without redeploying the entire application.

**Answer:** The company should migrate to a **microservices architecture**, decomposing the monolith into independently deployable services (e.g., catalog, cart, checkout, user profile).

**Why:** Microservices allow each component to be developed, deployed, and scaled independently. If the catalog service is under heavy load, only that service needs more instances. Updates to the checkout service can be deployed without affecting the catalog. This is a core cloud-native design principle.

**Technical reasoning:** Each microservice runs in its own container, communicates via APIs (REST, gRPC), and can be managed by Kubernetes. This enables independent scaling, fault isolation, and technology diversity (each service can use the best language/framework). The database is typically decomposed as well — each service owns its data.

**Trade-offs:** Microservices introduce distributed system complexity — network latency, partial failures, data consistency challenges, and operational overhead. A monolith is simpler to develop and deploy initially. The migration itself is costly and risky. Microservices are not always the right answer — for small teams or simple applications, a modular monolith may be better.

**Industry relevance:** Microservices are the dominant architectural pattern for large-scale cloud-native applications (Netflix, Amazon, Uber). However, the industry has also seen a backlash against premature microservices adoption.

**Key concepts tested:** Microservices, monolith, independent scaling, fault isolation, cloud-native architecture.

---

### Q29. [IaaS — Elastic IP vs Public IP]

**Scenario:** A company runs a web server on EC2. The server needs a static public IP address that does not change when the instance is stopped and restarted. The company also wants to be able to remap the IP to a different instance if the original fails.

**Answer:** The company should use an **Elastic IP (EIP)** .

**Why:** A standard public IP address is assigned dynamically and is released when the instance is stopped or terminated. An Elastic IP is a static, reserved public IP address that can be associated with an instance and remapped to another instance if needed. This provides a stable endpoint for users and enables failover.

**Technical reasoning:** Elastic IPs are allocated to an AWS account and remain associated with the account until released. They can be associated with instances, network interfaces, or NAT gateways. If an instance fails, the EIP can be quickly reassociated with a standby instance, minimizing downtime. EIPs are billed hourly when not associated with a running instance.

**Trade-offs:** Elastic IPs are a limited resource (default 5 per Region). They are not needed if using a load balancer or Route 53 DNS failover. Overuse of EIPs can lead to management complexity. Also, EIPs are Region-specific.

**Industry relevance:** Elastic IPs are used for static endpoints, failover scenarios, and whitelisting in firewalls. However, modern architectures often use load balancers and DNS instead.

**Key concepts tested:** Elastic IP, public IP, static addressing, failover, EC2 networking.

---

### Q30. [Docker — Multi-Stage Builds]

**Scenario:** A team is building a Docker image for a Go application. The build process requires the Go compiler and dependencies, but the final runtime only needs the compiled binary. The team's current Dockerfile installs the Go toolchain in the final image, resulting in a 1.5 GB image. The team wants to reduce the image size.

**Answer:** The team should use a **multi-stage Docker build**.

**Why:** Multi-stage builds allow the team to use one image for building (with the Go toolchain) and a separate, minimal image for runtime (e.g., `scratch` or `alpine`). Only the compiled binary is copied from the build stage to the final stage. This dramatically reduces the final image size.

**Technical reasoning:** A multi-stage Dockerfile has multiple `FROM` statements. The first stage (`AS builder`) installs Go and compiles the binary. The second stage (`FROM alpine`) copies the binary from the builder stage using `COPY --from=builder`. The final image contains only the binary and its runtime dependencies — not the compiler, source code, or build tools.

**Trade-offs:** Multi-stage builds are more complex to write and debug. They require careful separation of build-time and runtime dependencies. However, the benefits — smaller images, faster pulls, reduced attack surface — are substantial for production.

**Industry relevance:** Multi-stage builds are a best practice for compiled languages (Go, Rust, C++) and for frontend builds (Node.js build stage → Nginx runtime stage).

**Key concepts tested:** Multi-stage builds, Dockerfile optimization, image size, build vs runtime dependencies.

---

### Q31. [IaaS — Shared Responsibility Model]

**Scenario:** A company is using AWS EC2 to host a web application. A security audit reveals that the application's database is publicly accessible because the security group allows inbound traffic from `0.0.0.0/0` on port 3306. The company's CTO argues that AWS should have prevented this because "AWS is responsible for security."

**Answer:** The CTO is incorrect. Under the **Shared Responsibility Model**, AWS is responsible for security **of** the cloud (infrastructure), while the customer is responsible for security **in** the cloud (their data, applications, configurations, and access management).

**Why:** AWS secures the physical data centers, hardware, networking, and virtualization layer. The customer is responsible for the guest OS, applications, data, and — critically — security group configurations, IAM policies, and network access controls. A misconfigured security group is the customer's responsibility.

**Technical reasoning:** The Shared Responsibility Model is a foundational cloud security concept. AWS publishes detailed documentation on the split. For IaaS, the customer manages the OS, middleware, runtime, applications, and data. For PaaS, the customer manages applications and data; AWS manages the runtime, middleware, and OS. For SaaS, AWS manages almost everything; the customer manages data and access.

**Trade-offs:** The model is clear in theory but often misunderstood in practice. Misconfigurations — not infrastructure flaws — are the leading cause of cloud security incidents. Customers must invest in training, tooling, and governance to meet their responsibilities.

**Industry relevance:** This is the most important concept in cloud security. Every cloud practitioner must understand where their responsibility begins and ends.

**Key concepts tested:** Shared Responsibility Model, cloud security, misconfiguration, IAM, security groups.

---

### Q32. [IaaS — IAM Roles vs Access Keys]

**Scenario:** An application running on an EC2 instance needs to read files from an S3 bucket. A developer hardcodes AWS access keys and secret keys in the application's configuration file. The security team flags this as a critical vulnerability.

**Answer:** The developer should use an **IAM role** attached to the EC2 instance instead of hardcoded access keys.

**Why:** Hardcoded credentials are a major security risk — they can be leaked via source control, logs, or compromised instances. IAM roles provide temporary, automatically rotated credentials to the instance via the instance metadata service. The application assumes the role and receives credentials with only the permissions it needs.

**Technical reasoning:** When an IAM role is attached to an EC2 instance, the instance can call the AWS STS (Security Token Service) to obtain temporary credentials. These credentials expire automatically (minutes to hours) and are rotated. The role's policy defines exactly which S3 actions are allowed (e.g., `s3:GetObject` on a specific bucket). No permanent keys are stored anywhere on the instance.

**Trade-offs:** IAM roles are specific to AWS services — they work for EC2, Lambda, ECS, etc. For on-premises applications, other approaches (e.g., IAM Roles Anywhere, Vault) are needed. Roles also require careful policy design to avoid over-permissioning.

**Industry relevance:** This is a fundamental AWS security best practice. Hardcoded keys are a leading cause of cloud breaches (e.g., the Capital One breach involved an SSRF attack that accessed instance credentials).

**Key concepts tested:** IAM roles, temporary credentials, STS, least privilege, EC2 instance profile.

---

### Q33. [Containers — Docker Networking]

**Scenario:** A team runs two containers on the same Docker host: a web application container and a database container. The web application needs to connect to the database. The team is unsure how the containers should communicate.

**Answer:** The team should create a **user-defined bridge network** and attach both containers to it. The web application can then reach the database container by its container name (DNS resolution).

**Why:** On a user-defined bridge network, Docker provides automatic DNS resolution between containers. The web application can connect to `db:5432` (using the database container's name). This is cleaner and more secure than using the default bridge network, which does not provide automatic DNS.

**Technical reasoning:** Docker creates a virtual network interface for each container and connects them to a bridge. On a user-defined network, Docker runs an embedded DNS server that resolves container names to their IP addresses. The default bridge network requires `--link` (deprecated) or manual IP management. User-defined networks also provide better isolation — containers on different networks cannot communicate unless explicitly connected.

**Trade-offs:** User-defined networks add a small amount of configuration overhead. For simple single-host setups, the default bridge may suffice. In Kubernetes, networking is handled by the CNI plugin, and services provide stable DNS names.

**Industry relevance:** Docker networking is fundamental to multi-container applications. Docker Compose creates a default network for all services in a compose file.

**Key concepts tested:** Docker networking, bridge networks, DNS resolution, container communication.

---

### Q34. [IaaS — NAT Gateway vs Internet Gateway]

**Scenario:** A company has a VPC with public and private subnets. The public subnet hosts a load balancer and web servers. The private subnet hosts database servers that should not be directly accessible from the internet. However, the database servers need to download software updates from the internet.

**Answer:** The company should use an **Internet Gateway (IGW)** for the public subnet and a **NAT Gateway** for the private subnet.

**Why:** An Internet Gateway allows resources in a public subnet to communicate with the internet (both inbound and outbound). A NAT Gateway allows resources in a private subnet to initiate outbound connections to the internet (e.g., for software updates) while preventing inbound connections from the internet.

**Technical reasoning:** The IGW is attached to the VPC and routes traffic to/from the internet. The NAT Gateway is placed in a public subnet and has an Elastic IP. The private subnet's route table directs outbound traffic to the NAT Gateway, which translates the source IP to its own EIP. Return traffic is translated back. This provides outbound-only internet access for private resources.

**Trade-offs:** NAT Gateways incur hourly and data processing charges. They are managed services (highly available within an AZ). For multi-AZ redundancy, a NAT Gateway per AZ is recommended. Also, NAT Gateways do not support inbound connections — if the database needs to be accessed from the internet, a different architecture (e.g., VPN, bastion host) is required.

**Industry relevance:** This is the standard VPC architecture for separating public-facing and internal resources. It is a fundamental AWS networking pattern.

**Key concepts tested:** VPC, public vs private subnets, Internet Gateway, NAT Gateway, network security.

---

### Q35. [Virtualization — CPU Overcommitment]

**Scenario:** A virtualization host has 8 physical cores. The administrator has configured 10 VMs, each with 4 vCPUs (total 40 vCPUs). The overcommit ratio is 5:1. The administrator notices that CPU Ready time is increasing and some VMs are experiencing latency.

**Answer:** The administrator has overcommitted CPU beyond what the workloads can tolerate. The 5:1 ratio is at the high end of the acceptable range, and rising CPU Ready time indicates contention. The administrator should reduce the number of vCPUs per VM, add more physical hosts, or migrate some VMs to another host.

**Why:** CPU overcommitment works because vCPUs are time-sliced — the hypervisor schedules them onto physical cores. However, if the total demand for CPU time exceeds physical capacity, VMs must wait for a core to become available. CPU Ready time measures this wait. A CPU Ready above ~5% typically indicates problematic oversubscription.

**Technical reasoning:** The overcommit ratio is total vCPUs ÷ total pCPUs. A 5:1 ratio is generally acceptable for mixed workloads but risky for latency-sensitive workloads (databases, real-time apps). The administrator should monitor CPU Ready and adjust. Options include reducing vCPU count per VM (many VMs are configured with more vCPUs than needed), adding hosts, or using CPU affinity/reservations.

**Trade-offs:** Overcommitment improves density and reduces cost, but at the risk of performance degradation. The safe ratio depends on workload characteristics — CPU-intensive workloads should have lower ratios (1:1 to 3:1), while light workloads can tolerate higher ratios. Monitoring is essential.

**Industry relevance:** CPU overcommitment is a standard feature in VMware, KVM, and Hyper-V. Proper capacity planning and monitoring are critical for performance.

**Key concepts tested:** CPU overcommitment, CPU Ready time, vCPU vs pCPU, time-slicing, capacity planning.

---

## Section 3 — High (Questions 36–50)

---

### Q36. [Integrated — Migration to Cloud-Native]

**Scenario:** A logistics company runs a monolithic Java application on 10 physical servers. The application handles order tracking, inventory, and customer notifications. Traffic is low at night and peaks during business hours. The company wants to migrate to AWS, reduce infrastructure costs, improve availability, and eventually adopt a cloud-native architecture. The company has a skilled DevOps team.

**Answer:** The company should adopt a phased migration:

1. **Lift-and-shift to IaaS:** Migrate the monolith to EC2 instances (or containers on ECS/EKS) in a multi-AZ VPC. Use Auto Scaling to match capacity to demand. This provides immediate cost and availability benefits without re-architecting.
2. **Containerize:** Package the application in Docker containers and deploy on Kubernetes (EKS). This improves portability and resource utilization.
3. **Decompose into microservices:** Gradually split the monolith into microservices (order, inventory, notification) that can be scaled and deployed independently.
4. **Adopt cloud-native services:** Use managed services (RDS, SQS, SNS, Lambda) for databases, messaging, and notifications.

**Why:** A phased approach reduces risk and allows the team to learn. Lift-and-shift is fast but does not fully exploit cloud benefits. Containerization improves density and deployment speed. Microservices enable independent scaling — the inventory service can scale during peak hours without scaling the entire application. Managed services reduce operational burden.

**Technical reasoning:** The migration follows the AWS Migration Acceleration Program (MAP) or similar frameworks. The 6 Rs of migration (Rehost, Replatform, Repurchase, Refactor, Retire, Retain) are relevant: Rehost (lift-and-shift) first, then Replatform (containerize), then Refactor (microservices). Each stage delivers incremental value.

**Trade-offs:** Phased migration takes longer and requires sustained effort. Microservices introduce distributed system complexity. Managed services create vendor lock-in. The company must balance speed, cost, and architectural purity. A "big bang" rewrite is risky and often fails.

**Industry relevance:** This is the typical enterprise cloud migration journey. Most companies start with lift-and-shift, then modernize incrementally. Rushing to microservices without operational maturity often leads to failure.

**Key concepts tested:** Cloud migration strategies, lift-and-shift, containerization, microservices, managed services, phased modernization.

---

### Q37. [Integrated — Multi-Region High Availability]

**Scenario:** A global media streaming company serves users in North America, Europe, and Asia. The company wants its application to remain available even if an entire AWS Region fails. The application stores user profiles, watch history, and video metadata in a database. Video content is stored in S3.

**Answer:** The company should deploy the application in **multiple AWS Regions** (e.g., us-east-1, eu-west-1, ap-southeast-1), with:

- **Route 53** for latency-based routing and failover.
- **DynamoDB Global Tables** (or Aurora Global Database) for multi-Region data replication.
- **S3 Cross-Region Replication** for video content.
- **Auto Scaling** in each Region to handle local demand.
- **Health checks** to detect Regional failures and route traffic to healthy Regions.

**Why:** A multi-Region architecture provides disaster recovery and low-latency access for global users. If one Region fails, Route 53 routes traffic to the next closest healthy Region. Data replication ensures user profiles and watch history are available in all Regions.

**Technical reasoning:** Route 53 latency-based routing directs users to the Region with the lowest latency. Failover routing uses health checks to detect Regional outages. DynamoDB Global Tables provide multi-master replication with eventual consistency. S3 Cross-Region Replication copies objects asynchronously. The application must be designed to handle eventual consistency and conflict resolution.

**Trade-offs:** Multi-Region architecture is complex and expensive — duplicate infrastructure, data transfer costs, and operational overhead. Data consistency across Regions is challenging (especially for relational databases). Not all applications need multi-Region; multi-AZ is sufficient for most. The company must weigh the cost of multi-Region against the business impact of a Regional outage.

**Industry relevance:** Global companies (Netflix, Spotify, Airbnb) use multi-Region architectures for availability and performance. AWS itself recommends multi-AZ as the baseline and multi-Region for critical workloads.

**Key concepts tested:** Multi-Region, disaster recovery, Route 53, DynamoDB Global Tables, S3 replication, latency-based routing.

---

### Q38. [Integrated — Container Resource Management]

**Scenario:** A Kubernetes cluster runs 30 pods across 5 nodes. The operations team notices that some nodes are heavily utilized while others are idle. Some pods are being OOM-killed, while others are underutilized. The team wants to improve resource utilization and prevent pod evictions.

**Answer:** The team should configure **resource requests and limits** for all pods, use the **Horizontal Pod Autoscaler (HPA)** , and consider **node affinity/anti-affinity** and **pod disruption budgets**.

**Why:** Kubernetes schedules pods based on resource requests. Without requests, the scheduler cannot make informed decisions, leading to imbalance. Limits prevent a pod from consuming excessive resources. HPA scales the number of pods based on CPU/memory utilization. Affinity rules control pod placement. Pod disruption budgets ensure availability during voluntary disruptions.

**Technical reasoning:** Requests are used for scheduling — the scheduler places a pod on a node with sufficient allocatable resources. Limits are enforced by cgroups — exceeding memory limits leads to OOM kill; exceeding CPU limits leads to throttling. The HPA controller adjusts replica counts based on metrics from the metrics server. Node affinity can spread pods across nodes; anti-affinity can keep replicas apart for high availability.

**Trade-offs:** Setting requests too high leads to wasted capacity; too low leads to contention. Limits that are too low cause OOM kills; too high allow noisy neighbors. Autoscaling requires accurate metrics and can lag behind sudden spikes. The team must monitor and tune continuously.

**Industry relevance:** Resource management in Kubernetes is a core operational challenge. Misconfigured requests/limits are a leading cause of instability in production clusters.

**Key concepts tested:** Kubernetes resource requests/limits, HPA, scheduling, OOM kill, cgroups, node affinity.

---

### Q39. [Integrated — IaaS vs PaaS vs SaaS Decision]

**Scenario:** A healthcare startup is building a telemedicine platform. It needs:

- A custom application for video consultations (unique to the business).
- A standard email system for staff.
- A managed database for patient records.
- A platform to deploy and scale the custom application.

The startup has a small engineering team and wants to minimize operational overhead.

**Answer:** The startup should use a **combination of service models**:

- **SaaS** for email (e.g., Google Workspace, Microsoft 365).
- **PaaS** for deploying the custom application (e.g., AWS Elastic Beanstalk, Google App Engine, Heroku).
- **DBaaS** (a type of PaaS) for the managed database (e.g., Amazon RDS, MongoDB Atlas).
- **IaaS** only if specific infrastructure control is needed (e.g., for custom video processing).

**Why:** Different workloads have different requirements. Email is a commodity — SaaS is the obvious choice. The custom application is unique — PaaS provides the deployment platform without server management. The database is standard — a managed database service handles backups, patching, and replication. The startup should not manage servers if it doesn't need to.

**Technical reasoning:** The NIST service models are not mutually exclusive — a single organization can use all three. The decision depends on the level of control required, the operational capacity of the team, and the strategic importance of the workload. For a small team, minimizing operational overhead is critical — hence PaaS and managed services over IaaS.

**Trade-offs:** PaaS limits control over the runtime environment. Managed databases may not support all database engines or extensions. SaaS may not integrate with custom applications. The startup must balance control, cost, and operational burden. As the company grows, it may migrate some workloads from PaaS to IaaS for greater control.

**Industry relevance:** This is the typical architecture of a modern startup — SaaS for commodity services, PaaS for custom applications, managed services for data. The "serverless" and "managed services" trend reflects this.

**Key concepts tested:** IaaS vs PaaS vs SaaS, managed services, operational overhead, service model selection.

---

### Q40. [Integrated — Docker + Kubernetes + Cloud-Native]

**Scenario:** A fintech company is building a payment processing platform. The platform must be highly available, scale during peak transaction periods, and support rapid deployment of new features. The company wants to use containers and Kubernetes. The compliance team requires strong isolation between different customers' data.

**Answer:** The company should:

- Use **Kubernetes** with **namespaces** for logical isolation between customers or environments.
- Use **Network Policies** to restrict traffic between pods.
- Use **Pod Security Policies** (or OPA/Gatekeeper) to enforce security standards.
- Use **Persistent Volumes** with encryption for customer data.
- Use **RBAC** for access control.
- Deploy across **multiple AZs** for high availability.
- Use **HPA** for scaling during peak periods.
- Implement **CI/CD** with automated testing and canary deployments.

**Why:** Kubernetes provides the orchestration, scaling, and self-healing required for a payment platform. Namespaces and network policies provide the isolation required by compliance. Multi-AZ deployment ensures availability. HPA handles peak loads. CI/CD enables rapid feature deployment.

**Technical reasoning:** Namespaces in Kubernetes provide logical separation — resources in one namespace are isolated from another by default (though network policies are needed to enforce network isolation). RBAC controls who can access what. Persistent Volumes with encryption (e.g., AWS EBS encryption) protect data at rest. Canary deployments reduce risk by rolling out changes to a small subset of users first.

**Trade-offs:** Kubernetes is complex to operate — it requires expertise in networking, storage, security, and observability. Strong isolation via namespaces is not as strong as VM-level isolation; for very high security requirements, separate clusters or VMs may be needed. Compliance requirements may dictate specific configurations that add cost and complexity.

**Industry relevance:** This is the standard architecture for modern fintech and SaaS platforms. Kubernetes has become the operating system of the cloud.

**Key concepts tested:** Kubernetes, namespaces, network policies, RBAC, persistent volumes, HPA, CI/CD, multi-AZ.

---

### Q41. [Integrated — Cost vs Performance Trade-Off]

**Scenario:** A data analytics company runs a Spark cluster on AWS. The cluster processes large datasets and is used 8 hours a day, 5 days a week. The company wants to minimize costs but cannot tolerate job failures. The jobs are not time-critical — they can be interrupted and resumed.

**Answer:** The company should use a combination of **Spot Instances** for worker nodes and **On-Demand or Reserved Instances** for the master node.

**Why:** Spot Instances offer up to 90% discounts but can be interrupted with 2 minutes' notice. For a Spark cluster where worker tasks can be retried, Spot is ideal for cost savings. The master node (which coordinates the job) should not be interrupted — hence On-Demand or Reserved. The cluster can be spun up when needed and terminated after the job completes.

**Technical reasoning:** Spark is designed to be fault-tolerant — if a worker node is lost, tasks are rescheduled on other nodes. Spot Instances are allocated from spare AWS capacity and can be reclaimed. Using Spot for workers and On-Demand for the master balances cost and reliability. The company could also use AWS Glue or EMR with Spot instances.

**Trade-offs:** Spot Instances can be interrupted, causing job delays. If the Spot capacity is unavailable, the cluster may not launch. The company must design the workload to be interruptible. For time-critical jobs, On-Demand or Reserved is safer.

**Industry relevance:** Spot Instances are widely used for batch processing, big data, and CI/CD workloads. They are a key tool for cost optimization on AWS.

**Key concepts tested:** Spot Instances, On-Demand, Reserved Instances, fault tolerance, cost optimization, Spark.

---

### Q42. [Integrated — Legacy Application Migration]

**Scenario:** A manufacturing company runs a 15-year-old inventory management application on a physical Windows Server 2003 machine. The application cannot be recompiled or modified — the vendor is out of business. The company wants to migrate to the cloud to reduce hardware costs and improve reliability.

**Answer:** The company should use **IaaS** — migrate the application to a VM on AWS (or another cloud) running Windows Server 2003 (or a compatible OS). This is a "lift-and-shift" (rehost) migration.

**Why:** The application cannot be modified, so re-platforming or refactoring is not possible. IaaS provides the closest match to the existing environment — a VM with the same OS and configuration. The company can run the legacy application without changes, gaining cloud benefits (reduced hardware, snapshots, disaster recovery) without application changes.

**Technical reasoning:** AWS supports importing virtual machine images (VMDK, VHD, OVA) via VM Import/Export. The company can create an AMI from the existing physical server (using tools like AWS Server Migration Service or third-party tools) and launch it on EC2. The VM runs in a VPC with security groups controlling access. Snapshots of EBS volumes provide backups.

**Trade-offs:** Running an unsupported OS (Windows Server 2003) in the cloud is a security risk — no security patches are available. The company should isolate the VM in a private subnet with strict security group rules. Also, licensing for legacy OSes in the cloud can be complex. The company should plan to eventually replace the application.

**Industry relevance:** Legacy migration is a common challenge in enterprise cloud adoption. Lift-and-shift is the fastest path, but it does not modernize the application. Companies often run legacy apps in VMs while building replacements.

**Key concepts tested:** Lift-and-shift, IaaS, VM import, legacy applications, security isolation.

---

### Q43. [Integrated — Data Residency and Edge Computing]

**Scenario:** A global IoT company collects sensor data from factories in Europe, Asia, and North America. Each factory requires real-time processing of sensor data for safety alerts (latency < 10 ms). Regulatory requirements mandate that data stays within the region. The company also needs to aggregate data globally for analytics.

**Answer:** The company should use **AWS Local Zones** or **Wavelength Zones** for real-time processing at the edge, **Regional data stores** for data residency, and **cross-Region aggregation** for analytics.

**Why:** Local Zones extend AWS infrastructure closer to end users (e.g., a specific city), providing low-latency access. Wavelength Zones embed AWS compute in telecom providers' 5G networks for ultra-low latency. For real-time safety alerts (<10 ms), processing must happen close to the factory. Data residency requirements mean data must stay in the Region. For global analytics, aggregated (anonymized) data can be moved to a central data warehouse.

**Technical reasoning:** Local Zones are extensions of a Region — they have their own internet connections and support services like EC2, EBS, and VPC. Wavelength Zones are embedded in 5G networks. The company can deploy application instances in Local Zones for real-time processing, store data in the Region's S3/RDS, and use services like AWS Glue or Kinesis to aggregate data across Regions.

**Trade-offs:** Edge deployments add cost and management complexity. Not all AWS services are available in Local Zones. Data synchronization across Regions introduces latency and consistency challenges. The company must design for eventual consistency and conflict resolution.

**Industry relevance:** Edge computing is critical for IoT, autonomous vehicles, and real-time applications. AWS Local Zones and Wavelength are part of the growing edge infrastructure.

**Key concepts tested:** Local Zones, Wavelength Zones, edge computing, data residency, latency, IoT.

---

### Q44. [Integrated — Security and Compliance in Cloud]

**Scenario:** A fintech company is migrating to AWS. It must comply with PCI DSS (Payment Card Industry Data Security Standard). The compliance team requires:

- Encryption of data at rest and in transit.
- Strict access control.
- Audit logging.
- Network segmentation.
- Regular vulnerability assessments.

**Answer:** The company should implement:

- **Encryption:** EBS encryption, S3 SSE, RDS encryption, TLS for data in transit.
- **Access control:** IAM roles, least privilege, MFA, AWS Organizations SCPs.
- **Audit logging:** AWS CloudTrail, Config, VPC Flow Logs.
- **Network segmentation:** VPC, private subnets, security groups, NACLs.
- **Vulnerability management:** Amazon Inspector, GuardDuty, third-party scanners.
- **Compliance:** AWS Artifact for compliance reports, AWS Config rules for PCI DSS.

**Why:** PCI DSS has specific technical requirements that map to AWS services. Encryption protects data confidentiality. IAM and MFA protect access. CloudTrail provides audit trails. Network segmentation limits the scope of PCI DSS assessment. AWS provides a shared responsibility model — AWS is responsible for the infrastructure, the company for its configuration and data.

**Technical reasoning:** AWS offers a PCI DSS compliance package. The company can use AWS Config to continuously monitor compliance. Amazon GuardDuty provides threat detection. AWS WAF protects web applications. The company must also implement its own controls — e.g., code reviews, penetration testing, incident response.

**Trade-offs:** Compliance adds cost and complexity. Encryption key management (KMS) requires careful planning. Audit logging generates large volumes of data. The company must balance security with performance and usability.

**Industry relevance:** Compliance is a major driver of cloud adoption in regulated industries. AWS and other cloud providers offer extensive compliance certifications and tools.

**Key concepts tested:** PCI DSS, encryption, IAM, CloudTrail, VPC, shared responsibility, compliance.

---

### Q45. [Integrated — Serverless vs Containers vs VMs]

**Scenario:** A company is building a new application with three components:

1. A REST API that handles sporadic requests (a few per minute, sometimes bursts).
2. A background job that processes large files every night.
3. A stateful WebSocket server that maintains long-lived connections.

The company wants to minimize operational overhead and cost.

**Answer:** The company should use:

1. **Serverless (AWS Lambda + API Gateway)** for the REST API.
2. **Containers (AWS Fargate or EC2 with Docker)** for the background job.
3. **VMs or containers with persistent storage** for the WebSocket server.

**Why:** Different workloads have different characteristics. The REST API is event-driven and sporadic — serverless is ideal (pay per invocation, no idle cost). The background job is batch-oriented — a container or VM that runs on a schedule is cost-effective. The WebSocket server is long-lived and stateful — it needs persistent connections and state, which serverless does not support well.

**Technical reasoning:** Lambda scales automatically and charges per invocation. It is not suitable for long-running processes or persistent connections (15-minute timeout). Fargate runs containers without managing servers — ideal for batch jobs. For WebSockets, a container or VM with a load balancer that supports sticky sessions (or a Redis-backed session store) is needed.

**Trade-offs:** Serverless has cold start latency and limited execution time. Containers require more management than serverless but offer more control. VMs offer the most control but the most management. The company must match the technology to the workload.

**Industry relevance:** This is the modern "polyglot" architecture — using the right compute model for each workload. Serverless is growing rapidly, but containers and VMs remain essential for certain workloads.

**Key concepts tested:** Serverless, Lambda, containers, Fargate, VMs, WebSocket, workload-compute matching.

---

### Q46. [Integrated — Disaster Recovery Strategies]

**Scenario:** A company has a critical application running on AWS in a single Region. The company wants to implement a disaster recovery (DR) strategy. The RTO (Recovery Time Objective) is 1 hour, and the RPO (Recovery Point Objective) is 15 minutes. The company wants to minimize cost.

**Answer:** The company should implement a **Warm Standby** DR strategy.

**Why:** Warm Standby means running a scaled-down version of the production environment in a second Region. It is always running and can be scaled up to full capacity within minutes. This meets the 1-hour RTO and 15-minute RPO while being less expensive than an active-active (multi-Region) setup.

**Technical reasoning:** DR strategies on AWS include:

- **Backup and Restore:** Lowest cost, highest RTO/RPO (hours to days).
- **Pilot Light:** Core services replicated, rest scaled on demand (RTO: tens of minutes).
- **Warm Standby:** Scaled-down full environment running (RTO: minutes).
- **Multi-Site Active-Active:** Full capacity in multiple Regions (RTO: near-zero, highest cost).

For RTO of 1 hour and RPO of 15 minutes, Warm Standby is appropriate. Data replication (e.g., RDS cross-Region read replicas, S3 cross-Region replication) achieves the RPO.

**Trade-offs:** Warm Standby costs more than Pilot Light but less than Active-Active. It requires automation to scale up the standby environment. Data replication lag must be monitored to ensure RPO is met. The company must test failover regularly.

**Industry relevance:** DR is a critical requirement for production workloads. AWS provides building blocks (Route 53, cross-Region replication, etc.) to implement various DR strategies.

**Key concepts tested:** Disaster recovery, RTO, RPO, Warm Standby, Pilot Light, multi-Region.

---

### Q47. [Integrated — Multi-Tenancy and Isolation]

**Scenario:** A SaaS company provides a project management tool to 500 customers. The company wants to run all customers on shared infrastructure to reduce costs, but some enterprise customers require stronger isolation. The company is considering a multi-tenant architecture on AWS.

**Answer:** The company should adopt a **hybrid multi-tenancy model**:

- **Shared infrastructure** (shared database, shared compute) for small/medium customers.
- **Dedicated infrastructure** (separate database, separate VPC or account) for enterprise customers requiring isolation.

**Why:** Multi-tenancy reduces costs by sharing resources. However, some customers require stronger isolation for security, compliance, or performance reasons. A tiered approach balances cost and customer requirements.

**Technical reasoning:** On AWS, isolation can be implemented at multiple levels:

- **Account-level:** Each tenant gets a separate AWS account (using AWS Organizations).
- **VPC-level:** Each tenant gets a separate VPC.
- **Compute-level:** Each tenant gets separate containers or VMs.
- **Database-level:** Shared database with tenant ID, separate schemas, or separate databases.

The company can use a shared model for the majority and dedicated resources for premium tenants. IAM policies and resource tags enforce access control.

**Trade-offs:** Multi-tenancy introduces complexity in data isolation, noisy neighbor prevention, and billing. Dedicated infrastructure increases cost but satisfies enterprise requirements. The company must design the application to support both models — this requires careful architecture.

**Industry relevance:** Multi-tenancy is a core SaaS design concern. Companies like Salesforce, Slack, and Zoom use tiered multi-tenancy to serve different customer segments.

**Key concepts tested:** Multi-tenancy, isolation, shared vs dedicated, AWS Organizations, VPC, IAM.

---

### Q48. [Integrated — Performance Optimization]

**Scenario:** A web application running on AWS experiences high latency during peak hours. The application uses an RDS MySQL database, EC2 instances for the web tier, and S3 for static assets. Monitoring shows that the database is the bottleneck — read queries are slow, and CPU utilization on the DB instance is high.

**Answer:** The company should implement **RDS Read Replicas** to offload read traffic, and consider **ElastiCache (Redis or Memcached)** for caching frequent queries. For static assets, **CloudFront CDN** should be used to cache content at edge locations.

**Why:** Read replicas allow read-heavy workloads to scale horizontally — the primary database handles writes, and replicas handle reads. ElastiCache caches query results in memory, reducing database load. CloudFront caches static assets at edge locations, reducing latency for users and load on S3/EC2.

**Technical reasoning:** RDS Read Replicas use asynchronous replication — there is a small lag between the primary and replicas. The application must be designed to route read queries to replicas and write queries to the primary. ElastiCache provides sub-millisecond latency for cached data. CloudFront has 200+ edge locations worldwide.

**Trade-offs:** Read replicas introduce replication lag — reads may return stale data. ElastiCache requires cache invalidation logic. CloudFront adds cost but reduces origin load. The company must monitor and tune each component.

**Industry relevance:** This is the standard performance optimization stack for web applications on AWS. Caching, read replicas, and CDN are the three pillars of web-scale performance.

**Key concepts tested:** RDS Read Replicas, ElastiCache, CloudFront, caching, database scaling, latency optimization.

---

### Q49. [Integrated — Container Security]

**Scenario:** A company runs 100 containers in a Kubernetes cluster. A security audit reveals that:

- Some containers run as root.
- Some containers have privileged mode enabled.
- Images are pulled from public registries without scanning.
- Secrets are stored in environment variables.

**Answer:** The company should implement:

- **Non-root containers:** Run containers as a non-root user.
- **Disable privileged mode:** Use security contexts to prevent privilege escalation.
- **Private registry with image scanning:** Use Amazon ECR with image scanning, or a private registry with vulnerability scanning.
- **Secrets management:** Use Kubernetes Secrets (encrypted) or AWS Secrets Manager instead of environment variables.
- **Network policies:** Restrict pod-to-pod communication.
- **RBAC:** Limit who can deploy and access resources.
- **Pod Security Standards:** Enforce baseline or restricted policies.

**Why:** Running as root and privileged mode are major security risks — a compromised container could affect the host. Public images may contain vulnerabilities. Environment variables are not secure for secrets (they can leak in logs, crash dumps, etc.).

**Technical reasoning:** Kubernetes Security Contexts define privilege and access control settings for pods and containers. `runAsNonRoot: true` prevents root execution. `privileged: false` disables privileged mode. `readOnlyRootFilesystem: true` prevents writes to the container filesystem. Network Policies are firewall rules for pods. RBAC controls API access. Secrets should be stored in etcd (encrypted at rest) or an external secret manager.

**Trade-offs:** Security hardening adds complexity and may break applications that expect root or write access. Image scanning adds build-time overhead. Secrets management requires integration with the application. However, the security benefits are essential for production.

**Industry relevance:** Container security is a top concern for organizations adopting Kubernetes. Misconfigured containers are a common attack vector.

**Key concepts tested:** Container security, Kubernetes security contexts, RBAC, network policies, secrets management, image scanning.

---

### Q50. [Integrated — Cloud Adoption Strategy]

**Scenario:** A large enterprise with 5000 employees is planning a cloud migration. The company has:

- 200+ applications (some legacy, some modern).
- A large on-premises data center with 5-year-old hardware.
- A skilled IT team but limited cloud experience.
- Regulatory requirements for data privacy.
- A mandate to reduce IT costs by 30% over 3 years.

The CEO asks for a cloud adoption strategy.

**Answer:** The company should adopt a **phased, multi-year cloud adoption strategy**:

1. **Assessment:** Inventory applications, classify by criticality, compliance, and migration complexity.
2. **Pilot:** Migrate a few non-critical applications to build cloud skills and validate the approach.
3. **Foundation:** Establish landing zones (AWS Organizations, IAM, VPC, security baselines), networking, and governance.
4. **Migration:** Migrate applications in waves — start with lift-and-shift for simple apps, re-platform for databases, and re-architect for strategic apps.
5. **Optimization:** Implement cost management, auto-scaling, reserved instances, and continuous improvement.
6. **Innovation:** Adopt cloud-native services (serverless, AI/ML, IoT) to drive new business value.

**Why:** A phased approach reduces risk and allows the organization to learn. Starting with a pilot builds confidence and skills. Establishing a foundation ensures security and compliance. Migrating in waves manages complexity. Optimization delivers the cost savings mandate. Innovation leverages cloud capabilities for competitive advantage.

**Technical reasoning:** The strategy aligns with AWS's Cloud Adoption Framework (CAF) — Business, People, Governance, Platform, Security, Operations. The 6 Rs of migration (Rehost, Replatform, Repurchase, Refactor, Retire, Retain) guide application-level decisions. Cost optimization uses tools like AWS Cost Explorer, Budgets, and Trusted Advisor. Compliance is addressed through AWS Artifact, Config, and CloudTrail.

**Trade-offs:** Migration is disruptive and requires change management. Not all applications should move to the cloud — some may be retired or retained on-premises. The company must invest in training and hiring. Cost savings may take time to materialize. Regulatory constraints may limit which workloads can move.

**Industry relevance:** This is the typical enterprise cloud journey. Successful migrations are phased, well-governed, and aligned with business objectives.

**Key concepts tested:** Cloud adoption strategy, migration phases, AWS CAF, 6 Rs, governance, cost optimization, compliance.

---

## Most Important Concepts to Revise

| Rank | Concept | Revision Point |
|------|---------|----------------|
| 1 | **IaaS vs PaaS vs SaaS** | Know the NIST definitions, who manages what, and when to choose each. |
| 2 | **Deployment Models** | Public, Private, Hybrid, Community — characteristics, pros/cons, use cases. |
| 3 | **Virtualization** | Type 1 vs Type 2 hypervisors, benefits, limitations, x86 challenges. |
| 4 | **VM vs Containers** | Isolation, overhead, startup, density, use cases. |
| 5 | **Docker** | Images, containers, Dockerfiles, volumes, registries, networking. |
| 6 | **Namespaces vs cgroups** | Namespaces = isolation; cgroups = resource limits. |
| 7 | **Kubernetes** | Orchestration, self-healing, scaling, resource requests/limits. |
| 8 | **AWS Regions & AZs** | Region = geographic area; AZ = isolated data center; multi-AZ for HA. |
| 9 | **EC2 Instance Types** | Families (general, compute, memory, storage, accelerated), use case matching. |
| 10 | **AWS Storage** | S3 (object), EBS (block), EFS (file), Glacier (archive). |
| 11 | **VPC & Security Groups** | Public vs private subnets, IGW, NAT, security groups (stateful), NACLs. |
| 12 | **Shared Responsibility Model** | AWS secures the cloud; customer secures what's in the cloud. |
| 13 | **IAM** | Users, roles, policies, least privilege, temporary credentials. |
| 14 | **Cloud-Native Principles** | Stateless, microservices, CI/CD, immutable infrastructure. |
| 15 | **High Availability** | Multi-AZ, auto-scaling, load balancing, health checks. |
| 16 | **Cost Optimization** | On-Demand vs Reserved vs Spot, right-sizing, auto-scaling. |
| 17 | **Disaster Recovery** | RTO/RPO, Backup & Restore, Pilot Light, Warm Standby, Active-Active. |
| 18 | **Resource Overcommitment** | CPU vs memory overcommitment, CPU Ready, ballooning, swapping. |
| 19 | **Placement Groups** | Cluster (performance), Spread (availability), Partition (distributed). |
| 20 | **Container Registries** | Docker Hub, ECR, GCR; image distribution, versioning. |

---

## Common Exam Traps

1. **Confusing IaaS with PaaS:** IaaS provides raw infrastructure (VMs, storage, network); PaaS provides a platform for deploying applications. If the user manages the OS, it's IaaS. If the provider manages the OS, it's PaaS.

2. **Treating containers as lightweight VMs:** Containers share the host kernel; VMs have their own kernel. This affects isolation, compatibility, and density.

3. **Confusing Docker images with containers:** An image is a read-only template; a container is a running instance. Changes to a container do not affect the image.

4. **Assuming namespaces and cgroups do the same thing:** Namespaces provide isolation (what a process can see); cgroups provide resource limits (how much a process can use).

5. **Assuming Kubernetes is a container:** Kubernetes is an orchestrator — it manages containers, but it is not itself a container.

6. **Confusing AWS Regions with Availability Zones:** A Region is a geographic area (e.g., Mumbai); an AZ is an isolated data center within a Region. Multi-AZ provides HA; multi-Region provides DR and global reach.

7. **Assuming multiple VMs automatically provide high availability:** VMs on the same host or in the same AZ are not highly available. HA requires distribution across AZs and health checks.

8. **Ignoring resource constraints:** Not setting CPU/memory limits on containers leads to noisy neighbors and OOM kills. Not right-sizing VMs leads to waste or performance issues.

9. **Choosing a technology without explaining trade-offs:** Every cloud decision has trade-offs. The exam expects you to discuss what the solution does NOT solve.

10. **Confusing EXPOSE with publishing a port:** `EXPOSE` documents; `-p` publishes. `EXPOSE` alone does not make a port accessible from the host.

11. **Assuming SaaS means no responsibility:** Under the Shared Responsibility Model, the customer is always responsible for their data and access management, even in SaaS.

12. **Forgetting that S3 is not a file system:** S3 is object storage — you GET and PUT whole objects. It does not support partial updates or file system semantics.

13. **Assuming Reserved Instances are physical reservations:** Regional Reserved Instances are billing discounts, not physical capacity reservations. Capacity Reservations are separate.

14. **Confusing Security Groups with NACLs:** Security Groups are stateful, instance-level, allow-only. NACLs are stateless, subnet-level, allow and deny.

15. **Overlooking data residency in multi-Region designs:** Just because you can replicate data globally doesn't mean you should — regulations may prohibit it.

16. **Assuming serverless is always cheaper:** Serverless is cost-effective for sporadic workloads but can be more expensive for steady-state, high-volume workloads.

17. **Ignoring cold start latency in serverless:** Lambda cold starts can add hundreds of milliseconds to response time — problematic for latency-sensitive applications.

18. **Assuming microservices are always better than monoliths:** Microservices add complexity. For small teams or simple applications, a monolith may be more appropriate.

19. **Forgetting that Docker volumes persist beyond container lifecycle:** Data in a container's writable layer is lost when the container is removed. Volumes persist.

20. **Not considering the operational cost of Kubernetes:** Kubernetes is powerful but complex. It requires expertise and tooling. For simple deployments, it may be overkill.

---
---
---
# Cloud Computing — 25 Theoretical Questions & Answers

*(Matching the style, depth, and marks distribution of the sample question paper)*

---

### Q1. Define Cloud Computing as per NIST. State the FIVE essential characteristics of cloud computing and briefly explain any TWO. **(5 Marks: 2+3)**

**Answer:**

**NIST Definition (2 Marks):**
Cloud computing is a model for enabling ubiquitous, convenient, on-demand network access to a shared pool of configurable computing resources (e.g., networks, servers, storage, applications, and services) that can be rapidly provisioned and released with minimal management effort or service provider interaction.

**Five Essential Characteristics (3 Marks):**

1. **On-Demand Self-Service:** Consumers can provision computing capabilities (server time, storage) automatically without human interaction with the provider.
2. **Broad Network Access:** Capabilities are available over the network and accessed through standard mechanisms (mobile, laptop, workstation).
3. **Resource Pooling:** Provider's resources are pooled to serve multiple consumers using a multi-tenant model, with resources dynamically assigned and reassigned.
4. **Rapid Elasticity:** Capabilities can be elastically provisioned and released, scaling rapidly outward and inward with demand.
5. **Measured Service:** Cloud systems automatically control and optimize resource use via metering (storage, processing, bandwidth), providing transparency to both provider and consumer.

**Explanation of any TWO:**

- **Rapid Elasticity:** A web application can automatically scale from 2 instances to 20 during a traffic spike and back down afterward, paying only for what is used. This is enabled by virtualization and auto-scaling groups.
- **Resource Pooling:** Multiple customers (tenants) share the same physical hardware, but each is isolated logically. The provider dynamically allocates CPU, memory, and storage from a shared pool, improving utilization and reducing cost.

---

### Q2. Differentiate between IaaS, PaaS, and SaaS with respect to: (a) Who manages what, (b) Target audience, (c) Examples. **(6 Marks: 2+2+2)**

**Answer:**

| Factor | IaaS | PaaS | SaaS |
|--------|------|------|------|
| **Who manages what** | Provider manages virtualization, servers, storage, networking. Customer manages OS, middleware, runtime, applications, data. | Provider manages OS, middleware, runtime, servers, storage. Customer manages applications and data only. | Provider manages everything — applications, runtime, OS, servers, storage. Customer manages only user-specific configuration and data. |
| **Target audience** | Infrastructure & Network Architects, System Administrators | Application Developers, DevOps Engineers | End Users (business users) |
| **Examples** | Amazon EC2, Google Compute Engine, Azure VMs | Google App Engine, AWS Elastic Beanstalk, Heroku | Salesforce, Microsoft 365, Google Workspace |

**Key Insight:** The level of abstraction increases from IaaS → PaaS → SaaS. As abstraction increases, customer control decreases, but so does operational burden.

---

### Q3. Explain the FOUR cloud deployment models. Under what circumstances would an organization prefer a Hybrid Cloud over a Public Cloud? **(5 Marks: 3+2)**

**Answer:**

**Four Deployment Models (3 Marks):**

1. **Public Cloud:** Infrastructure is owned by a cloud provider and made available to the general public over the internet on a pay-per-use basis. *Example: AWS, Azure, GCP.*
2. **Private Cloud:** Infrastructure is exclusively dedicated to a single organization. It may be on-premises or hosted by a third party. *Example: VMware vSphere on-prem, AWS Outposts.*
3. **Community Cloud:** Infrastructure is shared by multiple organizations with common business, security, or compliance requirements. *Example: Government cloud for multiple agencies.*
4. **Hybrid Cloud:** Combines two or more distinct cloud environments (private, public, or on-premises) that remain separate but are connected to work as a unified environment.

**When to prefer Hybrid Cloud over Public Cloud (2 Marks):**

- **Regulatory/Compliance Requirements:** Sensitive data (e.g., patient records, financial data) must remain on-premises or in a private cloud due to data residency laws.
- **Latency-Sensitive Workloads:** Core transaction processing requires low latency and cannot tolerate public cloud network delays.
- **Cloud Bursting:** The organization wants to use public cloud for peak demand but keep baseline workloads on private infrastructure for cost and control.
- **Legacy Systems:** Some legacy applications cannot be migrated to the public cloud and must remain on-premises, while new applications run in the public cloud.
- **Security & Control:** The organization wants full control over its most critical data and infrastructure while leveraging public cloud for less sensitive workloads.

---

### Q4. What is Virtualization? Explain the difference between Type 1 and Type 2 Hypervisors with a suitable diagram/description. State TWO benefits of virtualization. **(5 Marks: 2+2+1)**

**Answer:**

**Definition (2 Marks):**
Virtualization is a computer architecture technology that allows multiple virtual machines (VMs) to be multiplexed on the same physical hardware (Host). It abstracts hardware resources (CPU, memory, storage, network) and presents them as virtual resources to guest operating systems, enabling multiple OS instances to run concurrently and in isolation on a single physical machine.

**Type 1 vs Type 2 Hypervisors (2 Marks):**

| Feature | Type 1 (Bare-Metal) | Type 2 (Hosted) |
|---------|---------------------|-----------------|
| **Runs on** | Directly on physical hardware | On top of a host operating system |
| **Performance** | Higher (near-native) | Lower (extra layer of host OS) |
| **Use case** | Production data centers, enterprise virtualization | Development, testing, personal use |
| **Examples** | VMware ESXi, KVM, Xen, Hyper-V | Oracle VirtualBox, VMware Workstation |
| **Security** | Stronger isolation | Weaker (depends on host OS security) |

**Diagram (Description):**
- **Type 1:** Hardware → Hypervisor → VMs (Guest OS + Apps). The hypervisor has its own lightweight kernel and directly manages hardware.
- **Type 2:** Hardware → Host OS → Hypervisor → VMs. The hypervisor runs as an application on the host OS.

**Two Benefits of Virtualization (1 Mark):**

1. **Cost Efficiency:** Consolidate multiple workloads onto fewer physical servers, reducing hardware, power, cooling, and space costs.
2. **Isolation & Security:** Each VM is isolated from others; a failure or security breach in one VM does not affect others.

---

### Q5. Explain the role of Docker volumes. How do they differ from bind mounts? State TWO scenarios where volumes are essential. **(5 Marks: 2+2+1)**

**Answer:**

**Role of Docker Volumes (2 Marks):**
Docker volumes are the preferred mechanism for persisting data generated by and used by Docker containers. Unlike the container's writable layer (which is tied to the container lifecycle and destroyed when the container is removed), volumes are stored on the host filesystem (managed by Docker) and persist independently of any container. They can be shared between multiple containers and are the recommended way to persist database data, logs, and application state.

**Volumes vs Bind Mounts (2 Marks):**

| Feature | Docker Volumes | Bind Mounts |
|---------|---------------|-------------|
| **Location** | Managed by Docker (`/var/lib/docker/volumes/`) | Anywhere on host filesystem |
| **Portability** | High — Docker manages the path | Low — depends on host directory structure |
| **Use case** | Production, persistent data, multi-container sharing | Development, sharing source code with container |
| **Backup** | `docker volume` commands | Manual host filesystem backup |
| **Security** | Isolated from host | Direct access to host filesystem |

**Two Scenarios Where Volumes Are Essential (1 Mark):**

1. **Database Containers:** A PostgreSQL container needs to persist data across restarts, upgrades, and container replacements. Without a volume, all data is lost when the container is removed.
2. **Shared Data Between Containers:** A web application container and a logging container need to access the same log files. A shared volume allows both containers to read/write the same data.

---

### Q6. What is the difference between a Docker image and a Docker container? Explain the Docker image layering concept. **(5 Marks: 2+3)**

**Answer:**

**Image vs Container (2 Marks):**

| Feature | Docker Image | Docker Container |
|---------|--------------|------------------|
| **Definition** | A read-only template with instructions for creating a container | A runnable instance of an image |
| **State** | Immutable (cannot be changed) | Ephemeral (writable layer on top of image) |
| **Storage** | Stored in registry or local cache | Running in memory/disk on host |
| **Lifecycle** | Built, pushed, pulled | Created, started, stopped, deleted |
| **Analogy** | Class (in OOP) | Object (instance of class) |

**Docker Image Layering Concept (3 Marks):**

A Docker image is built from a Dockerfile, and each instruction in the Dockerfile creates a **layer**. Layers are stacked on top of each other:

- **Base Layer:** The starting point (e.g., `ubuntu:22.04`).
- **Intermediate Layers:** Each `RUN`, `COPY`, `ADD` instruction creates a new layer.
- **Top Layer:** The final writable layer when a container runs.

**Key Points:**
- Layers are **cached** — if a layer hasn't changed, Docker reuses it, speeding up builds.
- Layers are **shared** between images — if two images use the same base, the base layer is stored only once.
- When a container runs, Docker adds a thin **writable layer** on top. Changes are stored here using **copy-on-write**.
- When the container is deleted, the writable layer is lost, but the image layers remain.

**Example:**
```dockerfile
FROM ubuntu:22.04          # Layer 1
RUN apt-get update          # Layer 2
RUN apt-get install -y nginx # Layer 3
COPY . /app                 # Layer 4
```
Each instruction adds a layer. If only `COPY` changes, Layers 1–3 are cached.

---

### Q7. Explain Namespaces and cgroups in the context of containers. How do they together enable container isolation and resource control? **(5 Marks: 2+3)**

**Answer:**

**Namespaces (2 Marks):**
Namespaces are a Linux kernel feature that provides **isolation** — they control what a process can **see**. Each container gets its own set of namespaces, so it appears to have its own isolated instance of global resources. Types of namespaces include:

- **PID:** Process IDs — container sees only its own processes.
- **NET:** Network interfaces, routing tables, ports.
- **MNT:** Filesystem mount points.
- **UTS:** Hostname and domain name.
- **IPC:** Inter-process communication.
- **USER:** User and group IDs.

**cgroups (2 Marks):**
Control groups (cgroups) are a Linux kernel feature that provides **resource limiting** — they control how much a process can **use**. cgroups limit, account for, and isolate resource usage (CPU, memory, disk I/O, network) for groups of processes. Docker uses cgroups when you specify `--cpus`, `--memory`, etc.

**How They Work Together (1 Mark):**

| Mechanism | Purpose | Analogy |
|-----------|---------|---------|
| **Namespaces** | Isolation — what the container can see | Walls of a room |
| **cgroups** | Resource control — how much the container can use | Size of the room |

Together, they enable containers to run as isolated, resource-bounded processes on a shared host kernel. Namespaces prevent a container from seeing or interfering with other containers' processes, networks, and filesystems. cgroups prevent a single container from monopolizing CPU, memory, or I/O, ensuring fair resource distribution and preventing noisy-neighbor problems.

**Example:** A container running a web server has its own NET namespace (so it can bind to port 80 without conflicting with other containers) and a cgroup limiting it to 512 MB RAM and 1 CPU core.

---

### Q8. What is Kubernetes? Explain any FOUR core components of Kubernetes and their roles. **(5 Marks: 1+4)**

**Answer:**

**What is Kubernetes (1 Mark):**
Kubernetes is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications. It groups containers into logical units (pods) and manages them across a cluster of machines, providing self-healing, auto-scaling, and rolling updates.

**Four Core Components (4 Marks):**

1. **API Server (`kube-apiserver`):** The front-end of the Kubernetes control plane. It exposes the Kubernetes API, which is used by all other components (and users) to interact with the cluster. It validates and processes REST requests.

2. **etcd:** A distributed key-value store that serves as Kubernetes' backing store for all cluster data. It stores the entire state of the cluster (what pods should be running, their configurations, etc.). It is the single source of truth.

3. **Scheduler (`kube-scheduler`):** Watches for newly created pods that have no node assigned and selects a node for them to run on. It considers resource requirements, affinity/anti-affinity rules, and other constraints.

4. **Controller Manager (`kube-controller-manager`):** Runs controller processes that regulate the state of the cluster. For example, the Replication Controller ensures the desired number of pod replicas are running. If a pod fails, the controller creates a new one.

**Additional Components (if needed):**
- **Kubelet:** An agent on each worker node that ensures containers are running in a pod.
- **Kube-proxy:** Maintains network rules on nodes, enabling communication between pods and services.

---

### Q9. Explain the Shared Responsibility Model in Cloud Computing. How does the division of responsibility change across IaaS, PaaS, and SaaS? **(5 Marks: 2+3)**

**Answer:**

**Shared Responsibility Model (2 Marks):**
The Shared Responsibility Model defines the division of security and operational responsibilities between the cloud provider and the cloud customer. In general:

- **Cloud Provider is responsible for security OF the cloud** — physical data centers, hardware, networking, virtualization layer, and the infrastructure that runs the cloud services.
- **Cloud Customer is responsible for security IN the cloud** — their data, applications, identity and access management, and configuration of cloud services.

**Division Across IaaS, PaaS, SaaS (3 Marks):**

| Layer | IaaS | PaaS | SaaS |
|-------|------|------|------|
| **Data** | Customer | Customer | Customer |
| **Applications** | Customer | Customer | Provider |
| **Runtime** | Customer | Provider | Provider |
| **Middleware** | Customer | Provider | Provider |
| **OS** | Customer | Provider | Provider |
| **Virtualization** | Provider | Provider | Provider |
| **Servers** | Provider | Provider | Provider |
| **Storage** | Provider | Provider | Provider |
| **Networking** | Provider | Provider | Provider |

**Key Insight:** As you move from IaaS → PaaS → SaaS, the provider takes on more responsibility, and the customer takes on less. However, the customer is **always** responsible for their data and access management, regardless of the service model.

**Example:** In IaaS (EC2), the customer must patch the guest OS. In PaaS (Elastic Beanstalk), the provider patches the OS. In SaaS (Salesforce), the provider patches everything; the customer only manages user access and data.

---

### Q10. What are AWS Regions, Availability Zones, and Edge Locations? How do they contribute to high availability and low latency? **(5 Marks: 2+3)**

**Answer:**

**Definitions (2 Marks):**

1. **AWS Region:** A geographic area (e.g., Mumbai, Frankfurt, Oregon) that contains multiple, isolated Availability Zones. Each Region is completely independent and isolated from other Regions.

2. **Availability Zone (AZ):** One or more discrete data centers within a Region, each with redundant power, networking, and connectivity, housed in separate facilities. AZs are physically separated (different flood plains, power grids) but connected via low-latency links.

3. **Edge Location:** A site deployed globally (200+ locations) used by Amazon CloudFront (CDN) to cache content closer to end users. Edge locations do not run customer compute; they cache and serve content.

**Contribution to High Availability and Low Latency (3 Marks):**

- **High Availability:** By distributing resources across multiple AZs within a Region, an application can survive the failure of a single AZ. If one AZ loses power or network, the application continues running in another AZ. This is the foundation of fault-tolerant architectures.
- **Low Latency (Compute):** Deploying applications in multiple Regions allows users to be served from the nearest Region, reducing network latency.
- **Low Latency (Content):** Edge locations cache static content (images, videos, CSS) closer to users. When a user requests content, it is served from the nearest edge location rather than the origin server, dramatically reducing latency.
- **Disaster Recovery:** Multi-Region deployments provide resilience against large-scale Regional failures.

**Example:** A global e-commerce site hosts its application in `us-east-1`, `eu-west-1`, and `ap-south-1`. Users are routed to the nearest Region via Route 53. Static assets are served via CloudFront from edge locations. Within each Region, the application runs in 3 AZs for high availability.

---

### Q11. What is the difference between Block Storage, File Storage, and Object Storage? Give an example of each from AWS. **(5 Marks: 2+3)**

**Answer:**

**Comparison (2 Marks):**

| Feature | Block Storage | File Storage | Object Storage |
|---------|--------------|--------------|----------------|
| **Level** | Raw storage device level | OS level | Application level |
| **Data Management** | Fixed-size blocks | Named hierarchy of files/folders | Objects with data + metadata |
| **Protocol** | iSCSI, Fibre Channel | NFS, SMB/CIFS | HTTP/REST API |
| **Use Case** | Databases, boot volumes | Shared file systems, home directories | Backup, media, static websites |
| **Scalability** | Limited (volume size) | Moderate | Virtually unlimited |

**AWS Examples (3 Marks):**

1. **Block Storage — Amazon EBS (Elastic Block Store):**
   - Provides persistent block-level storage volumes for EC2 instances.
   - A volume can be attached to a single EC2 instance at a time.
   - Automatically replicated within its AZ.
   - Use case: Boot volumes, databases, transactional workloads.

2. **File Storage — Amazon EFS (Elastic File System):**
   - Provides scalable, elastic network file storage for EC2 instances.
   - Can be mounted on multiple EC2 instances simultaneously.
   - Automatically scales up/down as files are added/removed.
   - Use case: Shared file systems, content management, home directories.

3. **Object Storage — Amazon S3 (Simple Storage Service):**
   - Stores data as objects in buckets, each identified by a unique key.
   - Accessed via HTTP/REST API.
   - Virtually unlimited scalability, 99.999999999% durability.
   - Use case: Backup, media storage, static website hosting, big data analytics.

---

### Q12. Explain the concept of a VPC in AWS. What are its key components? How does a NAT Gateway differ from an Internet Gateway? **(5 Marks: 2+2+1)**

**Answer:**

**VPC Definition (2 Marks):**
An Amazon Virtual Private Cloud (VPC) is a logically isolated virtual network in the AWS Cloud where you define your own IP address space. It allows you to launch AWS resources in a virtual network that you define, with complete control over networking configuration (subnets, route tables, gateways, security groups).

**Key Components of VPC (2 Marks):**

1. **Subnets:** Segments of a VPC's IP address range where you can place groups of isolated resources. Can be public (routable to internet) or private.
2. **Route Tables:** Rules that determine where network traffic is directed.
3. **Internet Gateway (IGW):** Enables communication between the VPC and the internet.
4. **NAT Gateway:** Enables instances in a private subnet to connect to the internet (outbound only) while preventing inbound connections.
5. **Security Groups:** Stateful virtual firewalls at the instance level.
6. **Network ACLs:** Stateless firewalls at the subnet level.
7. **Elastic IPs:** Static public IP addresses.

**NAT Gateway vs Internet Gateway (1 Mark):**

| Feature | Internet Gateway (IGW) | NAT Gateway |
|---------|------------------------|-------------|
| **Direction** | Inbound and outbound | Outbound only |
| **Attached to** | VPC | Public subnet |
| **Used by** | Public subnet resources | Private subnet resources |
| **Purpose** | Full internet access | Outbound internet access for private resources |
| **IP Translation** | No | Yes (source NAT) |

---

### Q13. What is Auto Scaling? Explain the different types of scaling policies. How does Auto Scaling contribute to cost optimization and high availability? **(5 Marks: 2+2+1)**

**Answer:**

**Auto Scaling Definition (2 Marks):**
Auto Scaling is a cloud feature that automatically adjusts the number of compute instances (e.g., EC2 instances) in a group based on demand. It ensures that you have the right number of instances to handle your application's load, scaling out (adding instances) during high demand and scaling in (removing instances) during low demand.

**Types of Scaling Policies (2 Marks):**

1. **Target Tracking Scaling:** Maintain a target value for a metric (e.g., keep CPU utilization at 50%). Auto Scaling adjusts the number of instances to keep the metric at the target.
2. **Step Scaling:** Scale based on a set of step adjustments. For example, if CPU > 70%, add 2 instances; if CPU > 90%, add 4 instances.
3. **Scheduled Scaling:** Scale based on a schedule (e.g., scale up at 9 AM on weekdays, scale down at 6 PM).
4. **Predictive Scaling:** Uses machine learning to predict future traffic and scale proactively.

**Contribution to Cost Optimization and High Availability (1 Mark):**

- **Cost Optimization:** Auto Scaling ensures you only run the instances you need. During low traffic, it terminates unnecessary instances, reducing costs. During high traffic, it adds instances only when needed.
- **High Availability:** Auto Scaling can distribute instances across multiple AZs. If an instance or AZ fails, Auto Scaling replaces the failed instance in a healthy AZ, maintaining application availability.

---

### Q14. Explain the difference between Horizontal Scaling and Vertical Scaling. Which is more suitable for cloud environments and why? **(5 Marks: 2+3)**

**Answer:**

**Horizontal vs Vertical Scaling (2 Marks):**

| Feature | Horizontal Scaling (Scale Out/In) | Vertical Scaling (Scale Up/Down) |
|---------|-----------------------------------|----------------------------------|
| **Definition** | Adding or removing instances | Adding or removing resources (CPU, RAM) to an existing instance |
| **Direction** | Increases number of machines | Increases size of a single machine |
| **Limit** | Virtually unlimited | Limited by hardware maximum |
| **Downtime** | Typically none (new instances added) | Often requires restart |
| **Cost** | Linear (more instances = more cost) | Exponential (larger instances cost disproportionately more) |
| **Complexity** | Requires load balancing, distributed architecture | Simpler for the application |
| **Fault Tolerance** | High (if one fails, others handle load) | Low (single point of failure) |

**Why Horizontal Scaling is More Suitable for Cloud (3 Marks):**

1. **Elasticity:** Cloud providers allow you to launch hundreds of instances in minutes. Horizontal scaling leverages this elasticity to handle traffic spikes.
2. **Fault Tolerance:** With multiple instances, if one fails, others continue to serve requests. Vertical scaling creates a single point of failure.
3. **Cost-Effectiveness:** Horizontal scaling uses commodity instances, which are cheaper per unit of compute. Vertical scaling requires expensive, high-end hardware.
4. **No Upper Limit:** Cloud providers offer virtually unlimited instances. Vertical scaling is capped by the largest available instance type.
5. **Auto Scaling Integration:** Cloud auto-scaling services (e.g., AWS Auto Scaling) are designed for horizontal scaling, automatically adding/removing instances based on demand.

**Exception:** Vertical scaling is still useful for databases and legacy applications that cannot be distributed.

---

### Q15. What is a Dockerfile? Explain the purpose of the following instructions: `FROM`, `WORKDIR`, `COPY`, `RUN`, `EXPOSE`, `CMD`. **(5 Marks: 2+3)**

**Answer:**

**Dockerfile Definition (2 Marks):**
A Dockerfile is a text file containing a set of instructions used to build a Docker image. It defines the base image, application code, dependencies, environment configuration, and the command to run when a container starts. Docker reads the Dockerfile and executes each instruction in order, creating a layered image.

**Purpose of Instructions (3 Marks):**

| Instruction | Purpose | Example |
|-------------|---------|---------|
| `FROM` | Specifies the base image to start from. Must be the first instruction. | `FROM ubuntu:22.04` |
| `WORKDIR` | Sets the working directory for subsequent instructions (`RUN`, `CMD`, `COPY`, etc.). Creates the directory if it doesn't exist. | `WORKDIR /app` |
| `COPY` | Copies files/directories from the host into the image. | `COPY package.json /app/` |
| `RUN` | Executes a command during the image build process. Used to install packages, run scripts, etc. Each `RUN` creates a new layer. | `RUN npm install` |
| `EXPOSE` | Documents the port on which the container will listen at runtime. Does **not** publish the port — it is metadata. | `EXPOSE 3232` |
| `CMD` | Specifies the default command to run when the container starts. Can be overridden at runtime. Only one `CMD` per Dockerfile. | `CMD ["node", "server.js"]` |

**Key Distinction:** `RUN` executes at build time; `CMD` executes at runtime.

---

### Q16. Explain the concept of Cloud-Native Applications. What are the key principles of cloud-native design? How does it differ from traditional web applications? **(5 Marks: 2+2+1)**

**Answer:**

**Cloud-Native Definition (2 Marks):**
Cloud-native applications are designed specifically to leverage cloud computing capabilities such as elastic scalability, high availability, automation, distributed architecture, and managed cloud services. They are built as loosely coupled microservices, packaged in containers, and orchestrated dynamically.

**Key Principles of Cloud-Native Design (2 Marks):**

1. **Microservices:** Applications are decomposed into small, independent services that communicate via APIs.
2. **Containers:** Each service is packaged in a container for portability and isolation.
3. **CI/CD:** Continuous Integration and Continuous Deployment enable rapid, automated releases.
4. **DevOps:** Collaboration between development and operations teams.
5. **Statelessness:** Services are stateless; state is externalized to databases or caches.
6. **Elasticity:** Services scale horizontally based on demand.
7. **Observability:** Comprehensive logging, monitoring, and tracing.
8. **API-First:** Services expose well-defined APIs.

**Difference from Traditional Web Applications (1 Mark):**

| Feature | Traditional Web App | Cloud-Native App |
|---------|---------------------|------------------|
| **Architecture** | Monolithic | Microservices |
| **Scaling** | Vertical (bigger server) | Horizontal (more instances) |
| **Deployment** | Manual, infrequent | Automated CI/CD, frequent |
| **Infrastructure** | Fixed servers | Dynamic, orchestrated (Kubernetes) |
| **Fault Tolerance** | Limited | Self-healing, highly available |
| **State** | Often stateful | Stateless (state externalized) |

---

### Q17. What is the role of a Container Registry? Explain the difference between Docker Hub and Amazon ECR. **(5 Marks: 2+3)**

**Answer:**

**Role of Container Registry (2 Marks):**
A container registry is a centralized repository for storing, versioning, and distributing Docker images. It allows developers to push images after building them and pull images when deploying containers. Registries support tagging (versioning), access control, and image scanning for vulnerabilities. They are essential for CI/CD pipelines and Kubernetes deployments.

**Docker Hub vs Amazon ECR (3 Marks):**

| Feature | Docker Hub | Amazon ECR |
|---------|-----------|------------|
| **Type** | Public (with private repos) | Private (fully managed) |
| **Ownership** | Docker Inc. | AWS |
| **Access Control** | Docker Hub accounts, organizations | IAM policies, repository policies |
| **Integration** | Broad (any Docker environment) | Deep integration with AWS (ECS, EKS, Lambda) |
| **Image Scanning** | Limited (paid plans) | Built-in vulnerability scanning |
| **Rate Limits** | Yes (pull rate limits for free tier) | No (within AWS) |
| **Encryption** | HTTPS | HTTPS + encryption at rest |
| **Use Case** | Open-source projects, public images | Enterprise, private images, AWS workloads |

**Key Insight:** Docker Hub is convenient for public images and open-source projects. Amazon ECR is preferred for production AWS workloads due to IAM integration, security, and no rate limits.

---

### Q18. Explain the AWS Shared Responsibility Model with a diagrammatic representation. Who is responsible for what in EC2, RDS, and S3? **(5 Marks: 2+3)**

**Answer:**

**Shared Responsibility Model (2 Marks):**
The Shared Responsibility Model defines the security responsibilities of AWS and the customer. AWS is responsible for "security OF the cloud" (infrastructure), while the customer is responsible for "security IN the cloud" (data, applications, configurations).

**Diagrammatic Representation:**

```
+-----------------------------------+
|         CUSTOMER RESPONSIBILITY    |
|  (Security IN the cloud)           |
|  - Data & encryption               |
|  - IAM & access management         |
|  - Applications                    |
|  - OS patching (for IaaS)          |
|  - Security group configuration    |
+-----------------------------------+
|         AWS RESPONSIBILITY         |
|  (Security OF the cloud)           |
|  - Physical data centers           |
|  - Hardware & networking           |
|  - Virtualization layer            |
|  - Managed service infrastructure  |
+-----------------------------------+
```

**Responsibility for EC2, RDS, S3 (3 Marks):**

| Service | AWS Responsibility | Customer Responsibility |
|---------|-------------------|------------------------|
| **EC2** | Physical hardware, hypervisor, network, AZ infrastructure | Guest OS patching, security groups, IAM, data encryption, application security |
| **RDS** | Database engine patching, backups, hardware, replication | Data encryption, IAM access, security group rules, database user management |
| **S3** | Storage infrastructure, durability, encryption at rest (optional) | Bucket policies, IAM access, object encryption (optional), data classification |

**Key Insight:** For managed services (RDS, S3), AWS takes on more responsibility. For IaaS (EC2), the customer takes on more. But the customer is always responsible for their data and access management.

---

### Q19. What is Cloud Bursting? Explain how it works with a suitable example. What are its benefits and challenges? **(5 Marks: 2+2+1)**

**Answer:**

**Cloud Bursting Definition (2 Marks):**
Cloud bursting is a hybrid cloud deployment technique where an application runs in a private cloud or on-premises data center during normal load, but "bursts" into a public cloud when demand spikes beyond the private infrastructure's capacity. The application temporarily uses public cloud resources to handle the overflow.

**How It Works (2 Marks):**

1. **Normal Operation:** The application runs on private infrastructure (on-premises or private cloud).
2. **Demand Spike:** When demand exceeds private capacity (e.g., CPU, memory, network), the system detects the threshold.
3. **Burst to Public Cloud:** Additional application instances are launched in the public cloud.
4. **Load Balancing:** Traffic is distributed between private and public instances.
5. **Scale Down:** When demand decreases, public cloud instances are terminated, and traffic returns to private infrastructure.

**Example:** A retail company runs its e-commerce website on-premises. During Black Friday, traffic exceeds the capacity of its private servers. The company bursts to AWS, launching additional EC2 instances behind a load balancer. After the sale, the EC2 instances are terminated.

**Benefits and Challenges (1 Mark):**

| Benefits | Challenges |
|----------|------------|
| Cost-effective (pay only for burst capacity) | Networking complexity (VPN, latency) |
| Scalability without permanent investment | Data synchronization between environments |
| Handles unpredictable spikes | Security and compliance across environments |
| Improved user experience during peak | Application must be designed for distribution |

---

### Q20. Explain the concept of Resource Pooling and Multi-Tenancy in Cloud Computing. How does virtualization enable these? **(5 Marks: 2+3)**

**Answer:**

**Resource Pooling and Multi-Tenancy (2 Marks):**

- **Resource Pooling:** The cloud provider's computing resources (CPU, memory, storage, network) are pooled together to serve multiple consumers. Resources are dynamically assigned and reassigned according to consumer demand. The consumer generally has no knowledge of the exact physical location of their resources.

- **Multi-Tenancy:** Multiple customers (tenants) share the same physical infrastructure while remaining logically isolated from each other. Each tenant's data and workloads are invisible to other tenants. This is a direct consequence of resource pooling.

**How Virtualization Enables These (3 Marks):**

1. **Abstraction:** Virtualization abstracts physical hardware (CPU, memory, storage) into virtual resources. This allows multiple VMs or containers to run on the same physical host.

2. **Isolation:** The hypervisor (for VMs) or namespaces (for containers) isolate each tenant's environment. A tenant cannot access another tenant's data or processes.

3. **Dynamic Allocation:** Virtualization allows resources to be allocated and deallocated dynamically. If one tenant needs more CPU, the hypervisor can allocate more vCPUs from the pool.

4. **Overcommitment:** Virtualization enables overcommitment (allocating more virtual resources than physical), improving utilization and reducing cost.

5. **Security:** The hypervisor enforces security boundaries. Even though tenants share hardware, they cannot interfere with each other.

**Example:** AWS EC2 uses the Nitro hypervisor to run VMs from thousands of customers on shared physical servers. Each customer's VM is isolated, and resources are dynamically allocated from AWS's massive resource pool.

---

### Q21. What is the difference between a Public IP, Private IP, and Elastic IP in AWS? When would you use each? **(5 Marks: 2+3)**

**Answer:**

**Comparison (2 Marks):**

| Feature | Private IP | Public IP | Elastic IP |
|---------|-----------|-----------|------------|
| **Scope** | Within VPC | Internet | Internet |
| **Assignment** | Automatically assigned | Automatically assigned | Manually allocated |
| **Persistence** | Persists with instance | Changes on stop/start | Static, persists until released |
| **Cost** | Free | Free | Charged when not attached |
| **Remappable** | No | No | Yes (can be moved between instances) |

**When to Use Each (3 Marks):**

1. **Private IP:**
   - Use for communication between instances within the same VPC or across peered VPCs.
   - Use for database servers, application servers, and internal services.
   - Example: A web server communicates with a database server using the database's private IP.

2. **Public IP:**
   - Use for instances that need to be accessible from the internet.
   - Use for web servers, bastion hosts, and NAT instances.
   - Note: Public IP changes when the instance is stopped and started.

3. **Elastic IP:**
   - Use when you need a static, persistent public IP address.
   - Use for failover scenarios (remap EIP to a standby instance).
   - Use for whitelisting in firewalls or DNS records.
   - Example: A company needs a fixed IP for its API endpoint so that partners can whitelist it. If the instance fails, the EIP is remapped to a standby instance.

**Trade-offs:** Elastic IPs are a limited resource (default 5 per Region) and are billed when not attached to a running instance. Overuse should be avoided; load balancers and Route 53 are often better alternatives.

---

### Q22. Explain the concept of Immutable Infrastructure. How does it differ from Mutable Infrastructure? What are its benefits in cloud environments? **(5 Marks: 2+2+1)**

**Answer:**

**Immutable Infrastructure (2 Marks):**
Immutable infrastructure is a paradigm where servers (or containers) are never modified after they are deployed. If a change is needed (e.g., software update, configuration change), a new instance is built from a new image, and the old instance is replaced. The old instance is never patched or updated in place.

**Mutable vs Immutable Infrastructure (2 Marks):**

| Feature | Mutable Infrastructure | Immutable Infrastructure |
|---------|------------------------|--------------------------|
| **Updates** | Patched in place | Replaced with new instance |
| **Configuration Drift** | Common (servers diverge over time) | Eliminated (every instance is identical) |
| **Rollback** | Difficult (undo changes) | Easy (revert to previous image) |
| **Deployment** | In-place updates | Blue-green or canary deployments |
| **Examples** | Traditional servers, pets | Containers, AMIs, Terraform |
| **Tools** | Ansible, Chef, Puppet | Docker, Packer, Kubernetes |

**Benefits in Cloud Environments (1 Mark):**

1. **Consistency:** Every instance is identical, eliminating configuration drift.
2. **Reliability:** Rollbacks are simple — just deploy the previous image.
3. **Scalability:** New instances can be launched quickly from a golden image.
4. **Security:** Vulnerabilities are fixed by replacing instances, not patching.
5. **CI/CD Friendly:** Immutable infrastructure aligns with automated pipelines.

**Example:** Instead of SSH-ing into a server to update Nginx, you build a new Docker image with the updated Nginx version, push it to a registry, and deploy new containers. The old containers are terminated.

---

### Q23. What is the difference between a Security Group and a Network ACL in AWS? How do they work together to secure a VPC? **(5 Marks: 2+3)**

**Answer:**

**Comparison (2 Marks):**

| Feature | Security Group | Network ACL |
|---------|---------------|-------------|
| **Level** | Instance (ENI) level | Subnet level |
| **State** | Stateful (return traffic automatically allowed) | Stateless (return traffic must be explicitly allowed) |
| **Rules** | Allow only | Allow and Deny |
| **Evaluation** | All rules evaluated | Rules evaluated in order (lowest number first) |
| **Default** | Deny all inbound, allow all outbound | Allow all inbound and outbound |
| **Applies to** | EC2 instances, RDS, etc. | All resources in a subnet |

**How They Work Together (3 Marks):**

1. **Layered Security:** Security Groups provide instance-level security, while NACLs provide subnet-level security. Traffic must pass both to reach an instance.

2. **Inbound Traffic Flow:**
   - Traffic enters the subnet → NACL rules are evaluated (allow/deny).
   - If allowed, traffic reaches the instance → Security Group rules are evaluated (allow only).

3. **Outbound Traffic Flow:**
   - Traffic leaves the instance → Security Group rules are evaluated.
   - If allowed, traffic leaves the subnet → NACL rules are evaluated.

4. **Use Case:**
   - Use NACLs to block specific IP ranges (e.g., block a known malicious IP) at the subnet level.
   - Use Security Groups to control which ports and protocols are allowed for specific instances.

**Example:** A VPC has a public subnet with a web server. The NACL allows inbound HTTP/HTTPS from anywhere and denies SSH from a specific IP range. The Security Group allows inbound HTTP/HTTPS from anywhere and SSH only from the bastion host's security group.

---

### Q24. Explain the concept of Serverless Computing. How does it differ from Containers and VMs? What are its advantages and limitations? **(5 Marks: 2+2+1)**

**Answer:**

**Serverless Computing (2 Marks):**
Serverless computing is a cloud execution model where the cloud provider dynamically manages the allocation of machine resources. The customer writes and deploys code (functions) without provisioning or managing servers. The code is executed in response to events (HTTP requests, file uploads, database changes). The customer pays only for the compute time consumed — there is no charge when the code is not running.

**Serverless vs Containers vs VMs (2 Marks):**

| Feature | VMs | Containers | Serverless |
|---------|-----|------------|------------|
| **Unit of Deployment** | VM image | Container image | Function |
| **Management** | Customer manages OS, runtime | Customer manages runtime, container | Provider manages everything |
| **Scaling** | Manual/Auto Scaling | Auto Scaling (pods) | Automatic (per request) |
| **Startup Time** | Minutes | Seconds | Milliseconds (cold start) |
| **Cost Model** | Per hour (running) | Per hour (running) | Per invocation + duration |
| **State** | Stateful | Stateless (typically) | Stateless |
| **Use Case** | Legacy, databases | Microservices, APIs | Event-driven, sporadic workloads |

**Advantages and Limitations (1 Mark):**

| Advantages | Limitations |
|------------|-------------|
| No server management | Cold start latency |
| Pay-per-invocation (cost-effective for sporadic) | Limited execution time (e.g., Lambda 15 min) |
| Automatic scaling | Limited memory/CPU |
| High availability built-in | Vendor lock-in |
| Faster time to market | Difficult to debug locally |

**Example:** AWS Lambda, Azure Functions, Google Cloud Functions.

---

### Q25. What is the role of IAM in AWS? Explain the concepts of Users, Groups, Roles, and Policies. How does IAM enforce the principle of least privilege? **(5 Marks: 2+3)**

**Answer:**

**Role of IAM (2 Marks):**
AWS Identity and Access Management (IAM) is a web service that helps you securely control access to AWS resources. IAM enables you to manage users, groups, roles, and permissions. It answers the questions: "Who can access what?" and "Under what conditions?" IAM is the foundation of AWS security — every API call is authenticated and authorized by IAM.

**Users, Groups, Roles, and Policies (3 Marks):**

1. **Users:** An IAM user is an identity with long-term credentials (password, access keys) that represents a person or application. Each user has a unique name and can be assigned permissions.

2. **Groups:** A collection of IAM users. Permissions are attached to a group, and all users in the group inherit those permissions. Groups simplify permission management (e.g., "Developers" group, "Admins" group).

3. **Roles:** An IAM role is an identity with temporary credentials that can be assumed by a user, application, or AWS service. Roles do not have long-term credentials. They are used for cross-account access, EC2 instance profiles, and Lambda execution roles.

4. **Policies:** A policy is a JSON document that defines permissions — what actions are allowed or denied on which resources under what conditions. Policies are attached to users, groups, or roles.

**Enforcement of Least Privilege:**

- **Least Privilege:** Grant only the minimum permissions necessary to perform a task.
- IAM enforces this by allowing you to write granular policies (e.g., `s3:GetObject` on a specific bucket).
- Explicit Deny always overrides Allow.
- Regular audits (IAM Access Analyzer, CloudTrail) help identify unused permissions.
- Roles provide temporary, scoped credentials instead of long-term keys.

**Example Policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::my-bucket/*"
    }
  ]
}
```
This policy allows reading objects from `my-bucket` only — nothing else.

---

## Quick Revision Summary

| Concept | Key Point |
|---------|-----------|
| NIST 5 Characteristics | On-demand, Broad Network, Pooling, Elasticity, Measured |
| IaaS/PaaS/SaaS | Who manages what; abstraction increases |
| Deployment Models | Public, Private, Hybrid, Community |
| Virtualization | Type 1 (bare-metal) vs Type 2 (hosted) |
| Docker Image vs Container | Image = template; Container = running instance |
| Namespaces vs cgroups | Isolation vs Resource Limiting |
| Kubernetes | Orchestration, self-healing, scaling |
| Shared Responsibility | AWS secures cloud; customer secures in cloud |
| Regions/AZs/Edge | Geographic → Data Center → CDN |
| Storage Types | Block (EBS), File (EFS), Object (S3) |
| VPC Components | Subnets, IGW, NAT, Route Tables, SGs, NACLs |
| Security Group vs NACL | Stateful/Instance vs Stateless/Subnet |
| Auto Scaling | Target tracking, step, scheduled, predictive |
| Horizontal vs Vertical | Scale out vs scale up |
| Cloud Bursting | Private baseline + Public peak |
| Immutable Infrastructure | Replace, don't patch |
| Serverless | No servers, pay per invocation |
| IAM | Users, Groups, Roles, Policies; least privilege |
| Elastic IP | Static public IP, remappable |
| Dockerfile | FROM, WORKDIR, COPY, RUN, EXPOSE, CMD |

---

*End of Question Bank*