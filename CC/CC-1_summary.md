# Cloud Computing — Comprehensive Lecture Summary

---

## Table of Contents

1. Cloud Computing Fundamentals
2. Cloud Service Models (IaaS, PaaS, SaaS)
3. Cloud Deployment Models
4. NIST Cloud Definition Framework
5. Virtualization
6. Hypervisors and VM Architecture
7. Types of Virtualization
8. Containers and Docker
9. Kubernetes and Orchestration
10. IaaS Architecture and AWS
11. AWS Regions, AZs, and Edge Locations
12. AWS Compute (EC2)
13. AWS Storage
14. AWS Networking (VPC)
15. AWS Databases
16. IAM, Authentication, and Authorization
17. Cloud-Native Design Principles
18. Cloud Adoption, Benefits, and Challenges
19. Key Comparisons and Trade-offs
20. Revision Summary

---

## 1. Cloud Computing Fundamentals

### 1.1 Definition (NIST)

Cloud computing is a model for enabling **ubiquitous, convenient, on-demand network access** to a shared pool of configurable computing resources (networks, servers, storage, applications, services) that can be **rapidly provisioned and released** with minimal management effort or provider interaction.

### 1.2 The 3-4-5 Rule

| 3 Service Models | 4 Deployment Models | 5 Essential Characteristics |
|------------------|---------------------|----------------------------|
| IaaS | Public | On-Demand Self-Service |
| PaaS | Private | Broad Network Access |
| SaaS | Hybrid | Resource Pooling |
| | Community | Rapid Elasticity |
| | | Measured Service |

### 1.3 Five Essential Characteristics

1. **On-Demand Self-Service:** Provision resources automatically without human interaction.
2. **Broad Network Access:** Available over the network via standard mechanisms (mobile, laptop).
3. **Resource Pooling:** Multi-tenant model; resources dynamically assigned/reassigned.
4. **Rapid Elasticity:** Scale outward and inward rapidly with demand.
5. **Measured Service:** Automatically control and optimize resource use via metering.

### 1.4 Evolution of Computing

| Era | Technology | Description |
|-----|-----------|-------------|
| 1960s | Terminals | Users access mainframe via simple terminals |
| 1970s | Mainframe | Centralized computing |
| 1980s | Desktop | Applications move to personal computers |
| 1990s | Client-Server | Applications run on servers, accessed by clients |
| 2000s | Internet | Web applications, server farms |
| 2010s | Cloud | Virtualized, software-defined everything |
| 2020s | Cloud-Native | Containers, serverless, AI-enabled |

### 1.5 Distributed Computing Paradigms

- **Parallel Computing:** Many calculations carried out simultaneously.
- **Cluster:** Group of loosely coupled computers working as a single system.
- **Grid:** Most distributed form; computers communicate over the internet.

### 1.6 Key Terms

- **Abstraction:** Details of system implementation hidden from users.
- **Virtualization:** Resources pooled and shared; illusion of sole ownership; scales rapidly.

---

## 2. Cloud Service Models (IaaS, PaaS, SaaS)

### 2.1 IaaS (Infrastructure as a Service)

**Definition:** Provides virtualized computing resources (compute, storage, network) on demand.

**Capabilities:**
- Servers, storage, network, operating systems
- Enables dynamic scaling (1...10...100...)
- Variable costs, multi-tenancy

**Models:** Public, Private, or combination

**Benefits:**
- Reduce CapEx, increase agility
- Pay only for what you use
- Focus on innovation, not infrastructure

**Examples:** Amazon EC2, Eucalyptus, GoGrid, Rackspace Cloud

**Applicability:**
- Need computing power on demand
- Sudden traffic spikes
- Development & testing
- Backup & disaster recovery
- AI/ML & HPC

### 2.2 PaaS (Platform as a Service)

**Definition:** Provides a platform for developers to build, deploy, and manage applications without managing infrastructure.

**Capabilities:**
- Tools to build applications
- Scripting environment
- Database platform
- Runtime, middleware, managed services

**Characteristics:**
- Collaborative platform for application development
- Allows creation of proprietary data or applications
- Fault tolerant design, containerization

**Benefits:**
- Lower costs, faster delivery
- No need to worry about upgrades or licenses
- Focus on innovation

**Examples:** Google App Engine, Azure, Force.com, Heroku

### 2.3 SaaS (Software as a Service)

**Definition:** Provides ready-to-use software applications over the internet on a subscription basis.

**Capabilities:**
- Hosted, finished product
- Subscription to services
- Managed service or monthly subscription

**Characteristics:**
- Not owned, but subscribed
- Multi-tenant
- Customer configuration instead of application configuration
- Centralized management

**Benefits:**
- No upfront software/license purchase
- Lower TCO
- Access from anywhere

**Examples:** Salesforce, Microsoft 365, Google Workspace

### 2.4 Comparison: Who Manages What

| Layer | On-Premise | IaaS | PaaS | SaaS |
|-------|-----------|------|------|------|
| Applications | You | You | You | Vendor |
| Data | You | You | You | Vendor |
| Runtime | You | You | Vendor | Vendor |
| Middleware | You | You | Vendor | Vendor |
| OS | You | You | Vendor | Vendor |
| Virtualization | You | Vendor | Vendor | Vendor |
| Servers | You | Vendor | Vendor | Vendor |
| Storage | You | Vendor | Vendor | Vendor |
| Networking | You | Vendor | Vendor | Vendor |

### 2.5 Which AAS to Choose?

| IaaS | PaaS | SaaS |
|------|------|------|
| Cover infrastructure maintenance | Focus on app development | Create solutions with standardized core |
| Save money vs hardware | Streamline workflows | Develop ecommerce rapidly |
| Flexibility and scalability | Rapidly launch applications | Short-term projects |
| Rapid growth | Multiple developers | Web and mobile access |

---

## 3. Cloud Deployment Models

### 3.1 Public Cloud

- Infrastructure owned by cloud provider
- Available to public on pay-per-use basis
- **Characteristics:** Homogeneous infrastructure, common policies, shared resources, multi-tenant, leased infrastructure, economies of scale
- **Examples:** AWS, Azure, GCP
- **Pros:** No upfront cost, elastic scalability, no maintenance
- **Cons:** Vendor lock-in, less control, compliance risks

### 3.2 Private Cloud

- Infrastructure exclusively dedicated to a single organization
- Can be owned, managed, operated by organization or third party
- Deployed on-premises, colocation, or hosted data center
- **Pros:** Full control, data stays on-premises, no noisy neighbor, security policy control
- **Cons:** High capital expenditure, limited elasticity, requires in-house expertise, idle capacity

### 3.3 Hybrid Cloud

- Combines two or more distinct cloud environments (private + public)
- Remain separate but connected via secure networking
- **Use cases:** Sensitive data on private, customer-facing apps on public; cloud bursting
- **Pros:** Combines public elasticity with private control, supports cloud bursting, disaster recovery
- **Cons:** Networking complexity, unified management required, security across environments

### 3.4 Community Cloud

- Shared by multiple organizations with common requirements
- Access restricted to participating organizations
- **Pros:** Shared infrastructure costs, common compliance, collaboration
- **Cons:** Shared governance, performance reliance, slower to set up, scalability capped

### 3.5 Deployment Model Summary

| Model | Analogy | Primary Strength | Primary Trade-off |
|-------|---------|------------------|-------------------|
| Public | Rent an apartment | Scale & speed | Less infrastructure control |
| Private | Own your house | Control & customization | Higher cost & responsibility |
| Hybrid | Own + rent when needed | Flexibility | Integration complexity |
| Community | Shared facility for group | Common requirements | Shared governance |

### 3.6 Multi-Cloud

- Deliberate use of two or more public cloud providers
- **Why:** Avoid lock-in, best-of-breed services, resilience, data residency
- **Advantages:** Negotiating leverage, best-of-breed, resilience, regulatory flexibility
- **Disadvantages:** Operational complexity, networking & cost overhead, inconsistent security, higher tooling cost

---

## 4. NIST Cloud Definition Framework

```
+-------------------------------------------------------------+
|                     HYBRID CLOUDS                            |
+-------------------------------------------------------------+
|  Deployment Models: Private | Community | Public             |
+-------------------------------------------------------------+
|  Service Models: SaaS | PaaS | IaaS                          |
+-------------------------------------------------------------+
|  Essential Characteristics:                                  |
|  - On Demand Self-Service                                    |
|  - Broad Network Access                                      |
|  - Rapid Elasticity                                          |
|  - Resource Pooling                                          |
|  - Measured Service                                          |
+-------------------------------------------------------------+
|  Common Characteristics:                                     |
|  - Massive Scale        - Resilient Computing               |
|  - Homogeneity          - Geographic Distribution            |
|  - Virtualization       - Service Orientation                |
|  - Low Cost Software    - Advanced Security                  |
+-------------------------------------------------------------+
```

---

## 5. Virtualization

### 5.1 Definition

Virtualization is a computer architecture technology that allows **multiple virtual machines (VMs) to be multiplexed on the same physical hardware (Host)** . It abstracts hardware resources (CPU, memory, storage, network) and presents them as virtual resources to guest operating systems.

### 5.2 Key Terms (Mnemonic: "MY VERY HAPPY MANY HOSTS RUN GUESTS")

| Term | Meaning |
|------|---------|
| **VM** | Virtual Machine — virtual computer created to run OS and applications |
| **VMM** | Virtual Machine Monitor — software layer that manages and controls VMs |
| **Hypervisor** | Another name for VMM — creates, runs, manages VMs |
| **Multiplexed** | Shared use of hardware among many VMs |
| **Host** | System where VMM resides (physical or virtual) |
| **Runs** | Host runs VMM, which runs VMs |
| **Guest** | Virtual Machines created |

### 5.3 Objectives of Virtualization

- **Maximize Resource Utilization:** Consolidate workloads on fewer machines
- **Reduce Costs:** Lower CapEx and OpEx through consolidation
- **Increase Flexibility & Agility:** Rapidly provision, clone, move, or scale VMs
- **Improve Availability & Resilience:** High availability, fault tolerance, workload mobility
- **Simplify Management:** Centralize infrastructure management

### 5.4 What Virtualization Enables

- Run multiple workloads simultaneously
- Isolation & Security
- Portability
- Scalability
- Higher efficiency, lower cost, greater agility, business continuity

---

## 6. Hypervisors and VM Architecture

### 6.1 Hypervisor Definition

A hypervisor (VMM) is computer software, firmware, or hardware that creates and runs virtual machines. It presents guest systems with a virtual operating platform and manages the execution of guest operating systems.

### 6.2 Type 1 vs Type 2 Hypervisors

| Feature | Type 1 (Bare-Metal) | Type 2 (Hosted) |
|---------|---------------------|-----------------|
| **Runs on** | Directly on hardware | On top of host OS |
| **Performance** | Near-native | Lower (extra layer) |
| **Use case** | Production data centers | Development, testing |
| **Examples** | VMware ESXi, KVM, Xen, Hyper-V | VirtualBox, VMware Workstation |
| **Security** | Stronger isolation | Weaker (depends on host OS) |

### 6.3 Hypervisor Goals (Popek-Goldberg Framework)

| Category | Goal | What it means |
|----------|------|---------------|
| **Formal Property** | Equivalence/Fidelity | Guest behaves identically to bare hardware |
| **Formal Property** | Resource Control | VMM retains complete control over physical resources |
| **Formal Property** | Efficiency | Statistically dominant fraction of instructions execute directly |
| **Practical Goal** | Timesharing/Resource Multiplexing | Multiple users share expensive hardware |
| **Practical Goal** | Isolation/Fault Containment | Crash in one VM doesn't affect others |
| **Practical Goal** | Security Boundary | Malicious code contained within VM |
| **Practical Goal** | Hardware Abstraction/Portability | Same VM image runs on different hardware |
| **Practical Goal** | Consolidation/Cost Reduction | Run many underutilized OS instances |
| **Practical Goal** | Operational Agility | Snapshots, live migration, fast provisioning |

### 6.4 x86 Architecture and Virtualization Challenges

**Rings of OS (x86):**

| Ring | Privilege Level | What Runs Here |
|------|----------------|----------------|
| Ring 0 | Most Privileged | OS Kernel (kernel mode) |
| Ring 1 | More Privileged | Reserved for system use |
| Ring 2 | Less Privileged | Reserved for system use |
| Ring 3 | Least Privileged | User Applications (user mode) |

**The Challenge:** x86 OSes assume they fully own the hardware. To virtualize x86, a hypervisor must sit underneath the OS, but the OS expects Ring 0. Some instructions are **sensitive but not privileged** — they behave differently based on ring but don't trap, causing silent errors.

**Solutions:**
1. **Binary Translation (1998, VMware):** Rewrites sensitive instructions in software before execution.
2. **Hardware-Assisted Virtualization (2006, Intel VT-x/AMD-V):** Adds a new privilege level below Ring 0 (Ring -1) for the hypervisor.

### 6.5 Levels of Virtualization

| Level | What it Virtualizes | Examples |
|-------|---------------------|----------|
| Application | Applications and runtime | JVM, .NET CLR |
| Library/API | System libraries and APIs | WINE, WABI |
| Operating System | OS kernel | Jail, Virtual Environment |
| Hardware Abstraction (HAL) | CPU, memory, storage, I/O | VMware, Xen, KVM |
| Instruction Set (ISA) | CPU instruction set | QEMU |

    The Virtualization Layer (QEMU): This is the engine. It takes physical hardware and slices it up into Virtual Machines (VMs).

    The Networking Layer (BIRD): Once those VMs are created, BIRD acts as the traffic controller, using dynamic routing protocols to make sure the VMs can talk to each other and the outside world. Used inside Kubernetes

    The Management Layer (BOSH): BOSH acts as the automated fleet manager. It commands QEMU to spin up hundreds of VMs, tells them what software to run, and monitors their health. Used in AWS EC2

### 6.6 Virtualization Approaches Comparison

| Dimension | Emulation | Binary Translation | Paravirtualization | Hardware-Assisted |
|-----------|-----------|-------------------|-------------------|-------------------|
| **Core Idea** | Software simulates CPU/ISA | Hypervisor translates sensitive instructions | Guest OS modified to call hypervisor | CPU adds privilege mode for trapping |
| **Guest OS Modified?** | No | No | Yes | No |
| **Guest ISA vs Host** | Can differ | Must match | Must match | Must match |
| **Performance** | Highest overhead | Moderate | Low | Lowest overhead |
| **Examples** | QEMU | VMware (1998-2006) | Xen (modified guest) | KVM, Hyper-V |
| **Requires CPU Support?** | No | No | No | Yes (VT-x/AMD-V) |

---


### Type-1: Bare-Metal

**Binary Translation (Full Virtualization)**

```text
+---------------------------------------------------+
|                   Guest Application               |
+---------------------------------------------------+
|               Guest OS (Unmodified)               |
|      (Sensitive instructions are intercepted)     |
+---------------------------------------------------+
|          VMM / Hypervisor (Binary Translator)     |
|   (Translates non-virtualizable ring instructions)|
+---------------------------------------------------+
|                Host Operating System              |
+---------------------------------------------------+
|                    Hardware                       |
|           (CPU, Memory, I/O Devices)              |
+---------------------------------------------------+

```

**Para-Virtualization**

```text
+---------------------------------------------------+
|                   Guest Application               |
+---------------------------------------------------+
|           Guest OS (Explicitly Modified)          |
|      (Replaces sensitive instructions with        |
|               direct Hypercalls)                  |
+---------------------------------------------------+
|         Hypervisor (e.g., Xen / Paravirtualized)  |
|      (Handles Hypercalls directly via API)        |
+---------------------------------------------------+
|                    Hardware                       |
|           (CPU, Memory, I/O Devices)              |
+---------------------------------------------------+

```

**Hardware-Assisted Virtualization**

```text
+---------------------------------------------------+
|                   Guest Application               |
+---------------------------------------------------+
|               Guest OS (Unmodified)               |
+---------------------------------------------------+
|            Virtual Machine Monitor (VMM)          |
|      (Runs in Root Mode / Hardware-Assisted)      |
+---------------------------------------------------+
|                 Hardware Extensions               |
|       (Intel VT-x / AMD-V CPU Modes & MMX)        |
+---------------------------------------------------+
|                Underlying Hardware                |
|           (CPU, Memory, I/O Devices)              |
+---------------------------------------------------+

```

---

### Type-2: Hosted (With a Host OS)

```text
+---------------------------------------------------+
|                   Guest Application               |
+---------------------------------------------------+
|               Guest OS (Unmodified)               |
+---------------------------------------------------+
|         Virtual Machine Monitor (VMM)             |
+---------------------------------------------------+
|              Host Operating System                |
|      (e.g., Windows, Linux, macOS)                |
+---------------------------------------------------+
|        Hardware Extensions (Intel VT-x / AMD-V)   |
+---------------------------------------------------+
|                Underlying Hardware                |
+---------------------------------------------------+

```

---

## 7. Types of Virtualization

### 7.1 Server Virtualization

- Abstracts physical machine; multiple VMs run independently
- **Benefits:** Partitioning, isolation, encapsulation, hardware independence
- **Key:** Single physical machine → multiple VMs, each with own OS

### 7.2 Storage Virtualization

- Provides logical, abstracted view of physical storage
- **Benefits:** Resource optimization, cost reduction, increased availability, improved performance
- **Example:** SAN (Storage Area Network)

### 7.3 Network Virtualization

- Creates multiple isolated virtual networks on shared physical infrastructure
- **Examples:** VLAN, VRF, SDN, NV Overlay
- **Benefits:** Multi-tenancy isolation, better resource utilization, improved security, agility

### 7.4 Memory Virtualization

- Shares physical memory dynamically among VMs
- **Mechanism:** Virtual Address (VA) → Physical Address (PA) → Machine Address (MA)
- **Techniques:** Page sharing, ballooning, compression, host-level swapping

### 7.5 Device Virtualization

- Presents each VM with virtualized version of peripheral devices
- Guest OS interacts with virtual device; hypervisor talks to physical hardware
- **Benefits:** Flexibility, remote access, centralized management

### 7.6 Desktop Virtualization

- **VDI (Virtual Desktop Infrastructure):** Desktop runs on centralized server
- **Hosted Virtual Desktop:** Desktop delivered as a service

---

## 8. Containers and Docker

### 8.1 Containers vs VMs

| Feature | VM | Container |
|---------|-----|-----------|
| **Isolation** | Hardware-level (hypervisor) | OS-level (namespaces) |
| **OS** | Full guest OS per VM | Shares host kernel |
| **Startup** | Minutes | Seconds/milliseconds |
| **Size** | GBs | MBs |
| **Density** | Low (10s per host) | High (100s per host) |
| **Security** | Stronger isolation | Weaker (kernel shared) |
| **Use Case** | Legacy, different OS | Microservices, cloud-native |

### 8.2 Namespaces and cgroups

| Mechanism | Purpose | What it Controls |
|-----------|---------|------------------|
| **Namespaces** | Isolation | What a process can **see** (PID, NET, MNT, UTS, IPC, USER) |
| **cgroups** | Resource Limiting | How much a process can **use** (CPU, memory, I/O) |

**Together:** Namespaces = walls of a room; cgroups = size of the room.

### 8.3 Docker Concepts

- **Docker Image:** Read-only template with instructions for creating a container.
- **Docker Container:** Runnable instance of an image; has writable layer on top.
- **Dockerfile:** Text file with instructions to build an image.
- **Docker Volume:** Persistent storage mechanism; survives container lifecycle.
- **Docker Registry:** Centralized repository for storing/distributing images.
- **Docker Networking:** Bridge, host, overlay networks; DNS resolution between containers.

### 8.4 Dockerfile Instructions

| Instruction | Purpose |
|-------------|---------|
| `FROM` | Base image |
| `WORKDIR` | Set working directory |
| `COPY` | Copy files from host to image |
| `RUN` | Execute command at build time |
| `EXPOSE` | Document intended port |
| `CMD` | Default command at runtime |
| `ENTRYPOINT` | Main command (harder to override) |


Understanding Docker can be tricky because its core concepts sound similar. Think of the relationship using a real-world blueprint analogy: a **Dockerfile** is the architectural blueprint, a **Docker image** is the ready-to-print house plan, a **Docker container** is the actual lived-in house built from that plan, and **Docker Compose** is the master city planner coordinating a neighborhood of multiple houses.

Here is a breakdown of each term, followed by a practical example tying them all together.

---

### 1. Dockerfile

* **What it is:** A text document that contains all the commands a user could call on the command line to assemble an image. It is your recipe or blueprint.
* **Key role:** It defines your base operating system, application code, dependencies, environment variables, and the command to run when the app starts.

### 2. Docker Image

* **What it is:** A read-only, static file template that contains instructions for creating a Docker container.
* **Key role:** Built from a `Dockerfile`, an image is the compiled, packaged version of your application ready to be shared and run anywhere.

### 3. Docker Container

* **What it is:** A runnable and washable instance of a Docker image.
* **Key role:** While images are static storage files, containers are active, running processes isolated in their own environment with their own file system, network, and resource limits. You can run multiple containers from the exact same image.

### 4. Docker Compose

* **What it is:** A tool for defining and running multi-container Docker applications.
* **Key role:** Instead of manually typing long command-line instructions to spin up a web app, database, and cache separately, `docker-compose.yml` lets you configure all services in a single file and start everything with one command (`docker compose up`).

---

### Practical Example: A Simple FastAPI & MongoDB App

Let's build a small application to see how all four pieces work together.

#### Step 1: The Dockerfile

Create a file named `Dockerfile` in project folder to package Python application into a read-only image:

```dockerfile
# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

```

#### Step 2: The Docker Image

When you run the build command, Docker reads the `Dockerfile` and creates a static **Docker image** (let's call it `fastapi-app:latest`).

```bash
docker build -t fastapi-app:latest .

```

*At this point, you have a packaged image stored locally that contains your FastAPI app and all its required modules.*

#### Step 3: The Docker Container

Now, you can launch an isolated, running instance of that image—turning it into a **Docker container**:

This launches an active, isolated container instance of that image—turning it into a **Docker container**. You can visit http://localhost:8000 to see it

```bash
docker run -d -p 8000:8000 --name running-fastapi-app fastapi-app:latest

```

* **`-d`**: Runs the container in the background (detached mode).
* **`-p 8000:8000`**: Maps port 8000 on your computer to port 8000 inside the container. i.e `host-port:container-port`
* **`running-fastapi-app`**: The name of your active **container**. You can now open your browser and visit `http://localhost:8000` to use your app.

#### Step 4: Docker Compose

Real applications require more than just one container (e.g., your FastAPI app + a MongoDB database). Instead of running separate commands for each, you use a `docker-compose.yml` file:

```yaml

version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MONGO_URL=mongodb://db:27017
    depends_on:
      - db

  db:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:

```

With **Docker Compose**, you simply type:

```bash
docker compose up

```
This single command reads the configuration, builds your Dockerfile into an image for the web service, pulls the MongoDB image, creates storage volumes, and spins up **two interactive containers** working together seamlessly.

---

### Summary Comparison Table

| Concept | Analogy | State | Primary Command |
| --- | --- | --- | --- |
| **Dockerfile** | Recipe / Blueprint | Text File | `docker build` |
| **Docker Image** | Packaged Cake Mix | Read-only Template | `docker pull` / `docker build` |
| **Docker Container** | The Baked Cake | Active, Running Process | `docker run` |
| **Docker Compose** | The Bakery Management System | Multi-container Orchestrator | `docker compose up` |



### 8.5 Multi-Stage Builds

- Use multiple `FROM` statements
- Build stage (with toolchain) → Runtime stage (minimal)
- **Benefit:** Smaller final image, reduced attack surface

### 8.6 System vs Application Containers

| Feature | System Containers | Application Containers |
|---------|-------------------|----------------------|
| **Purpose** | Full OS environment | Single process/service |
| **Examples** | LXC, LXD | Docker |
| **Init System** | Yes | No (one process) |
| **Use Case** | Legacy migration, multi-service | Microservices |

---

## 9. Kubernetes and Orchestration

### 9.1 What is Kubernetes?

Open-source container orchestration platform that automates deployment, scaling, and management of containerized applications.

### 9.2 Core Components

| Component | Role |
|-----------|------|
| **API Server** | Front-end; exposes Kubernetes API |
| **etcd** | Distributed key-value store; cluster state |
| **Scheduler** | Assigns pods to nodes |
| **Controller Manager** | Regulates cluster state; ensures desired replicas |
| **Kubelet** | Agent on worker nodes; runs containers |
| **Kube-proxy** | Network rules; enables pod communication |

### 9.3 Key Concepts

- **Pod:** Smallest deployable unit; one or more containers
- **Deployment:** Desired state for pods; manages replicas
- **Service:** Stable network endpoint for pods
- **Namespace:** Logical isolation within cluster
- **ConfigMap/Secret:** Configuration and sensitive data
- **Persistent Volume:** Storage that survives pod lifecycle

### 9.4 Kubernetes Benefits

- Self-healing (restarts failed containers)
- Auto-scaling (HPA)
- Load balancing
- Rolling updates and rollbacks
- Resource management (requests/limits)

---

## 10. IaaS Architecture and AWS

### 10.1 IaaS Architecture Stack

| Layer | Components | Description |
|-------|-----------|-------------|
| **5. Service/Access Layer** | Self-Service Portal, REST APIs, CLI, SDKs | Consumer touchpoint |
| **4. Core Management/Orchestration** | VM Lifecycle, Scheduler, Capacity, SLA, Metering | Manages and orchestrates resources |
| **3. Resource Pool/Abstraction** | Compute Pool, Storage Pool, Network Pool | Aggregates virtual resources |
| **2. Virtualization Layer** | Hypervisor (Type 1), Virtual CPU/RAM/Disk/NIC | Abstracts physical resources |
| **1. Physical Infrastructure** | Servers, Storage, Network, Data Center | Actual capital asset base |

### 10.2 IaaS Actors (NIST SP 500-292)

| Actor | Role |
|-------|------|
| **Cloud Consumer** | Uses services |
| **Cloud Provider** | Builds and operates |
| **Cloud Broker** | Intermediates/adds value |
| **Cloud Auditor** | Assesses/verifies |
| **Cloud Carrier** | Connects/transports |

---

## 11. AWS Regions, AZs, and Edge Locations

### 11.1 Definitions

| Term | Definition |
|------|------------|
| **Region** | Geographic area (e.g., Mumbai, Frankfurt) containing multiple AZs |
| **Availability Zone (AZ)** | One or more discrete data centers with redundant power, networking, connectivity |
| **Edge Location** | Site for CloudFront CDN caching (200+ locations) |
| **Local Zone** | Extension of Region in geographic proximity to users |
| **Wavelength Zone** | AWS compute embedded in 5G networks |
| **Outposts** | AWS infrastructure extended to customer premises |

### 11.2 Hotel Chain Analogy

| AWS Concept | Hotel Analogy |
|-------------|---------------|
| Region | City with multiple hotel properties |
| Availability Zone | Individual hotel property with independent infrastructure |
| Local Zone | Small hotel near airport |
| Wavelength Zone | Hotel kiosk in airport lounge |
| Edge Location | Luggage drop-off counter |

### 11.3 High Availability with Multi-AZ

- Deploy instances in multiple AZs
- If one AZ fails, others continue serving
- Use load balancer to distribute traffic
- **Key:** AZs are isolated (different flood plains, power grids)

---

## 12. AWS Compute (EC2)

### 12.1 EC2 Overview

- Amazon EC2 provides resizable compute capacity in the cloud
- Virtual servers called **instances**
- Pay per hour while running
- Stop instance → no EC2 charges (EBS charges continue)

### 12.2 Instance Types

| Family | Description | Mnemonic |
|--------|-------------|----------|
| t4g | General Purpose (Graviton) — Burstable | t = tiny/turbo |
| m7g | General Purpose (Graviton3, ARM) — Balanced | m = main |
| c7g | Compute Optimized (Graviton3) | c = compute |
| r7g | Memory Optimized (Graviton3) | r = RAM |
| x2gd | Memory Optimized — Extreme | x = extreme |
| p5 | Accelerated Computing (H100 GPU) | p = processing |
| i4g | Storage Optimized (Graviton) | i = IOPS |
| d3en | Storage Optimized — Dense HDD | d = dense |

### 12.3 Amazon Machine Image (AMI)

- Defines initial software state of an instance
- Includes: OS, configuration, patch level, pre-installed software
- **Sources:**
  1. Published by AWS
  2. AWS Marketplace
  3. Generated from existing instance
  4. Uploaded virtual servers (VM Import/Export)

### 12.4 EC2 Tenancy Options

| Option | Description | Best For |
|--------|-------------|----------|
| **Shared Tenancy** | Default; multiple customers on same host | Most workloads |
| **Dedicated Instances** | Hardware dedicated to single customer | Compliance, isolation |
| **Dedicated Host** | Physical server fully dedicated | Licensing, BYOL |

### 12.5 Placement Groups

| Type | Description | Use Case |
|------|-------------|----------|
| **Cluster** | Low-latency, high-bandwidth (10 Gbps) | HPC |
| **Spread** | Distinct hardware; reduce correlated failures | Critical applications |
| **Partition** | Divided into partitions | Hadoop, Cassandra |

### 12.6 Instance Stores

- Temporary block-level storage
- Physically attached to host
- **Data lost when:** disk fails, instance stops, instance terminates
- Included in EC2 cost

### 12.7 EC2 Lifecycle

```
Pending → Running → Stopped → Terminated
           ↑          |
           └── Start ─┘
```

---

## 13. AWS Storage

### 13.1 Storage Types

| Type | Description | AWS Service |
|------|-------------|-------------|
| **Block** | Raw storage device level; fixed-size blocks | Amazon EBS |
| **File** | OS level; named hierarchy of files/folders | Amazon EFS |
| **Object** | Application level; objects with data + metadata | Amazon S3 |

### 13.2 Amazon S3

- Object storage with simple web service interface
- **Concepts:**
  - **Objects:** Fundamental entities (1 byte to 5 TB)
  - **Buckets:** Containers for objects (up to 10,000 per account)
  - **Keys:** Unique identifier for object within bucket
- **Durability:** 99.999999999% (11 nines)
- **Use cases:** Backup, media, static website, big data

### 13.3 Amazon EBS

- Persistent block-level storage for EC2
- Automatically replicated within AZ
- **Types:**
  - General-Purpose SSD (gp2/gp3): 1 GB–16 TB, 3 IOPS/GB, max 10,000 IOPS
  - Provisioned IOPS SSD (io1/io2): 4 GB–16 TB, high performance
- **Use cases:** Boot volumes, databases, big data

### 13.4 Amazon EFS

- File-level storage; shared elastic file system
- Virtually unlimited scalability
- Highly available; can be used by many servers
- **Use cases:** Lift-and-shift, analytics, web serving

### 13.5 Amazon Glacier

- Extremely low-cost storage for archiving
- Retrieval time: minutes to hours
- **Use cases:** Long-term backup, compliance, archival

### 13.6 Storage Comparison

| Feature | S3 | EBS | EFS |
|---------|-----|-----|-----|
| **Type** | Object | Block | File |
| **Pricing** | Pay as you use | Pay for provisioned | Pay as you use |
| **Scalability** | Unlimited | Limited (volume size) | Unlimited |
| **Durability** | 99.999999999% | 99.99% | 99.99% |
| **Access** | Web interface | Block device | File system |
| **Use Case** | Media, backup, static | Boot, databases | Shared file systems |

---

## 14. AWS Networking (VPC)

### 14.1 VPC Components

| Component | Description |
|-----------|-------------|
| **Subnet** | Segment of VPC IP range |
| **Route Table** | Rules for traffic direction |
| **Internet Gateway** | VPC to internet communication |
| **NAT Gateway** | Outbound internet for private subnet |
| **Security Group** | Stateful instance-level firewall |
| **Network ACL** | Stateless subnet-level firewall |
| **Elastic IP** | Static public IP |
| **VPC Peering** | Connect two VPCs |
| **VPN Gateway** | Connect to on-premises |


---

### Quick Comparison Summary

Here is your quick summary checklist of all the core networking components:

### 🌐 Traffic Flow & Connectivity

* **Subnet:** Decides *where* a server lives (partitions your network - Public vs Private).
* **Routing Table:** The map/receptionist deciding *how* packets travel between subnets and outside.
* **IGW (Internet Gateway):** The front gate handling public internet traffic in and out.
* **NAT Gateway:** The mail desk letting private servers reach the internet outbound without letting the internet initiate inbound connections.
* **VPC Peering:** A direct private corridor connecting two separate cloud networks.
* **VPN:** A secure encrypted bridge connecting your on-prem office to the cloud.

### 🛡️ Security Layers

* **Security Group (SG):** Instance-level firewall (**Stateful**—remembers allowed return traffic).
* **Network ACL (NACL):** Subnet-level firewall (**Stateless**—requires explicit inbound and outbound rules).

---

### 14.2 Public vs Private Subnets

| Feature | Public Subnet | Private Subnet |
|---------|---------------|----------------|
| **Internet Access** | Yes (via IGW) | No (via NAT for outbound) |
| **Public IP** | Yes | No |
| **Use Case** | Web servers, load balancers | Databases, app servers |

### 14.3 Security Group vs Network ACL

| Feature | Security Group | Network ACL |
|---------|---------------|-------------|
| **Level** | Instance | Subnet |
| **State** | Stateful | Stateless |
| **Rules** | Allow only | Allow and Deny |
| **Evaluation** | All rules | In order |

---

## 15. AWS Databases

### 15.1 Database Services

| Service | Type | Use Case |
|---------|------|----------|
| **DynamoDB** | NoSQL Key-Value | High-throughput, serverless |
| **RDS** | Relational (SQL) | Web apps, e-commerce |
| **ElastiCache** | In-Memory | Caching, sessions |
| **Redshift** | Data Warehouse | Analytics, BI |

### 15.2 Amazon RDS

- Managed relational database service
- Supports: MySQL, PostgreSQL, MariaDB, Oracle, SQL Server, Db2, Aurora
- **Features:**
  - Multi-AZ (synchronous standby, auto-failover)
  - Read Replicas (asynchronous, read scaling)
  - Automated backups (1–35 days retention)
  - Point-in-time recovery

---

## 16. IAM, Authentication, and Authorization

### 16.1 IAM Overview

- Manages digital identities and controls resource access
- Cross-cutting layer gating access across compute, storage, network, management
- **Examples:** AWS IAM, Azure Entra ID, Google Cloud IAM

### 16.2 Authentication vs Authorization

| Concept | Question | When |
|---------|----------|------|
| **Authentication** | "Who are you?" | Once per session |
| **Authorization** | "What are you allowed to do?" | Every request |

### 16.3 IAM Components

| Component | Description |
|-----------|-------------|
| **User** | Identity with long-term credentials |
| **Group** | Collection of users with shared permissions |
| **Role** | Identity with temporary credentials; assumable |
| **Policy** | JSON document defining permissions |

### 16.4 Policy Elements

| Element | Description |
|---------|-------------|
| **Effect** | Allow or Deny |
| **Action** | What can be done (e.g., s3:GetObject) |
| **Resource** | What resource (e.g., bucket ARN) |
| **Condition** | Under what conditions |

### 16.5 Principle of Least Privilege

- Grant only minimum permissions necessary
- Explicit Deny always overrides Allow
- Use roles for temporary credentials
- Regular audits and monitoring

### 16.6 IAM TANGO (Flow)

```
Authentication → Role Assumed → Authorization → Decision
```

1. **Authentication:** Prove who you are
2. **Role Assumed:** Obtain temporary credentials
3. **Authorization:** Check what you're allowed to do
4. **Decision:** Allow or Deny

---

## 17. Cloud-Native Design Principles

### 17.1 Cloud-Native vs Traditional

| Feature | Traditional Web App | Cloud-Native App |
|---------|---------------------|------------------|
| **Architecture** | Monolithic | Microservices |
| **Scaling** | Vertical | Horizontal |
| **Deployment** | Manual, infrequent | CI/CD, frequent |
| **Infrastructure** | Fixed servers | Dynamic, orchestrated |
| **State** | Often stateful | Stateless |
| **Fault Tolerance** | Limited | Self-healing |

### 17.2 Key Principles

1. **Microservices:** Small, independent services
2. **Containers:** Packaging for portability
3. **CI/CD:** Automated pipelines
4. **DevOps:** Collaboration
5. **Statelessness:** Externalize state
6. **Elasticity:** Scale horizontally
7. **Observability:** Logging, monitoring, tracing
8. **API-First:** Well-defined interfaces

### 17.3 Twelve-Factor App (Relevant Factors)

- Codebase: One codebase, many deploys
- Dependencies: Explicitly declare
- Config: Store in environment
- Backing Services: Treat as attached resources
- Processes: Stateless
- Port Binding: Export via port
- Logs: Treat as event streams

---

## 18. Cloud Adoption, Benefits, and Challenges

### 18.1 Key Benefits

| Benefit | Description |
|---------|-------------|
| **Reduced Costs** | Pay-as-you-go, no upfront hardware |
| **Ease of Utilization** | Low upfront costs, easy adoption |
| **Quality of Service** | SLAs guarantee performance |
| **Reliability & High Availability** | Auto-scaling, fault tolerance |
| **Outsourced IT Management** | Provider manages infrastructure |
| **Simplified Maintenance** | Continuous updates |
| **Low Entry Barrier** | Start small, scale as you grow |

### 18.2 Cloud Challenges

| Challenge | Description |
|-----------|-------------|
| **Security** | Shared responsibility, misconfigurations |
| **Compliance** | Data residency, regulations |
| **Vendor Lock-in** | Proprietary services |
| **Cost Management** | Unpredictable spending |
| **Operational Complexity** | Distributed systems |
| **Skills Gap** | Need for cloud expertise |

### 18.3 Cloud Failures (Examples)

| Date | Provider | Impact |
|------|----------|--------|
| Dec 2024 | AWS us-east-1 | Multiple services |
| Jun 2024 | Google Cloud | Global outage |
| Feb 2024 | Microsoft Azure | Global outage |
| Dec 2022 | AWS us-east-1 | EC2, Lambda, RDS |
| Oct 2021 | Google Cloud | Global outage |
| Jul 2021 | Microsoft Azure | Global outage |

**Lesson:** Redundancy, multi-region architecture, and robust monitoring are key.

---

## 19. Key Comparisons and Trade-offs

### 19.1 VM vs Container

| Factor | VM | Container |
|--------|-----|-----------|
| Isolation | Hardware-level | OS-level |
| Overhead | High | Low |
| Startup | Minutes | Seconds |
| Density | Low | High |
| Use Case | Legacy, different OS | Microservices |

### 19.2 Horizontal vs Vertical Scaling

| Factor | Horizontal | Vertical |
|--------|------------|----------|
| Direction | More machines | Bigger machine |
| Limit | Unlimited | Hardware max |
| Downtime | None | Often required |
| Cost | Linear | Exponential |
| Fault Tolerance | High | Low |

### 19.3 IaaS vs PaaS vs SaaS

| Factor | IaaS | PaaS | SaaS |
|--------|------|------|------|
| Control | High | Medium | Low |
| Management | Customer | Shared | Provider |
| Abstraction | Low | Medium | High |
| Examples | EC2 | Elastic Beanstalk | Salesforce |

### 19.4 Public vs Private vs Hybrid vs Community

| Factor | Public | Private | Hybrid | Community |
|--------|--------|---------|--------|-----------|
| Cost | Low | High | Medium | Medium |
| Control | Low | High | Medium | Medium |
| Scalability | High | Limited | High | Limited |
| Security | Shared | Dedicated | Mixed | Shared |

---

## 20. Revision Summary

### Most Important Concepts

1. **NIST 5 Characteristics:** On-demand, Broad Network, Pooling, Elasticity, Measured
2. **IaaS/PaaS/SaaS:** Who manages what; abstraction increases
3. **Deployment Models:** Public, Private, Hybrid, Community
4. **Virtualization:** Type 1 vs Type 2 hypervisors
5. **x86 Virtualization:** Binary translation vs hardware-assisted
6. **Containers vs VMs:** Isolation, overhead, density
7. **Docker:** Images, containers, Dockerfiles, volumes
8. **Namespaces vs cgroups:** Isolation vs resource limiting
9. **Kubernetes:** Orchestration, self-healing, scaling
10. **AWS Regions/AZs:** Geographic → Data Center
11. **EC2 Instance Types:** General, compute, memory, storage
12. **AWS Storage:** S3 (object), EBS (block), EFS (file)
13. **VPC:** Subnets, IGW, NAT, Security Groups, NACLs
14. **IAM:** Users, Groups, Roles, Policies
15. **Shared Responsibility:** AWS secures cloud; customer secures in cloud
16. **Cloud-Native:** Microservices, stateless, CI/CD
17. **Auto Scaling:** Target tracking, step, scheduled
18. **Disaster Recovery:** RTO/RPO, Warm Standby
19. **Cost Optimization:** Reserved, Spot, On-Demand
20. **High Availability:** Multi-AZ, load balancing

### Common Exam Traps

1. Confusing IaaS with PaaS
2. Treating containers as lightweight VMs
3. Confusing Docker images with containers
4. Assuming namespaces and cgroups do the same thing
5. Confusing AWS Regions with AZs
6. Assuming multiple VMs automatically provide HA
7. Ignoring resource constraints
8. Choosing technology without explaining trade-offs
9. Confusing EXPOSE with publishing a port
10. Assuming SaaS means no responsibility

### Quick Reference Tables

**Service Models:**

| Model | User Manages | Provider Manages |
|-------|-------------|------------------|
| IaaS | OS, runtime, apps, data | Virtualization, servers, storage, network |
| PaaS | Apps, data | OS, runtime, middleware, servers |
| SaaS | Data, access | Everything else |

**Storage:**

| Type | AWS | Use Case |
|------|-----|----------|
| Object | S3 | Media, backup, static |
| Block | EBS | Boot, databases |
| File | EFS | Shared file systems |
| Archive | Glacier | Long-term retention |

**Deployment Models:**

| Model | Description | Use Case |
|-------|-------------|----------|
| Public | Shared infrastructure | Most workloads |
| Private | Dedicated infrastructure | Sensitive data |
| Hybrid | Combined | Regulatory + agility |
| Community | Shared by group | Common compliance |

**Hypervisors:**

| Type | Runs On | Examples |
|------|---------|----------|
| Type 1 | Bare metal | ESXi, KVM, Xen |
| Type 2 | Host OS | VirtualBox, VMware Workstation |

**Virtualization Approaches:**

| Approach | Guest Modified? | Performance | Example |
|----------|----------------|-------------|---------|
| Emulation | No | Lowest | QEMU |
| Binary Translation | No | Moderate | VMware (early) |
| Paravirtualization | Yes | High | Xen |
| Hardware-Assisted | No | Highest | KVM, Hyper-V |

---

*End of Lecture Summary*