# Software Architecture — Comprehensive Lecture Summary

---

# Module 1: Introduction to Software Architecture

## 1.1 What is Software Architecture?

**Definition:** The software architecture of a system is the set of structures needed to reason about the system, which comprise software elements, relations among them, and properties of both.

**Key Points:**
- Architecture is an abstraction — it omits information that has no ramifications outside a single element
- Every system has an architecture, even if undocumented or poorly designed
- Behavior is part of architecture — how elements interact at runtime
- Architecture is not just "early" or "major" design decisions; structures are easier to identify

## 1.2 Architectural Structures

Three categories of structures:

| Structure Type | Elements | Relations | Purpose |
|----------------|----------|-----------|---------|
| **Module** | Modules (code units) | Is-a-submodule-of, uses, inherits-from | Code organization, functional decomposition, modifiability |
| **Component-and-Connector (C&C)** | Runtime components, connectors | Attachment, communication | Runtime behavior, performance, availability, scalability |
| **Allocation** | Software elements, external environments | Allocated-to, migrates-to | Deployment, team assignment, file structure |

**Useful Module Structures:**
- **Decomposition:** Is-a-submodule-of relation; supports project structuring
- **Uses:** Requires correct presence of; supports incremental development
- **Layered:** Provides abstraction; supports portability
- **Class/Generalization:** Inherits-from; supports reuse

**Useful C&C Structures:**
- **Process:** Runs concurrently with; supports scheduling analysis
- **Concurrency:** Runs on same logical thread; identifies resource contention
- **Shared Data:** Produces/consumes data; supports performance and data integrity
- **Client-Server:** Communicates with; supports distributed operation

**Useful Allocation Structures:**
- **Deployment:** Allocated to; supports performance, availability, security
- **Implementation:** Stored in; supports configuration control
- **Work Assignment:** Assigned to; supports project management

## 1.3 Structures vs. Views

- **Structure:** The set of elements itself, as they exist in software or hardware
- **View:** A representation of a coherent set of architectural elements, as written by and read by system stakeholders
- A view is a representation of a structure

## 1.4 Architectural Patterns

**Definition:** Compositions of architectural elements that solve particular problems, documented and disseminated.

**Common Patterns:**
- **Layered:** Strictly unidirectional uses relation
- **Shared-Data (Repository):** Components create, store, access persistent data
- **Client-Server:** Clients and servers communicate via protocols
- **Multi-tier:** Distributes components across hardware/software subsets
- **Competence Center / Platform:** Work assignment patterns

## 1.5 What Makes a "Good" Architecture?

**Important:** There is no inherently good or bad architecture. Architectures are more or less fit for purpose.

**Process Rules of Thumb:**
- Product of single architect or small group with identified leader
- Based on prioritized list of quality attribute requirements
- Documented using views
- Evaluated using formal method (ATAM)
- Lends itself to incremental implementation

**Product Rules of Thumb:**
- Use information hiding and separation of concerns
- Achieve quality attributes using well-known tactics
- Never depend on particular version of commercial tool
- Minimize number of ways components interact
- Separate data producers from data consumers
- Don't expect one-to-one correspondence between modules and components
- Every process should be written so assignment to processor can change
- System should do same things in same way throughout
- Contain specific (and small) set of resource contention areas

## 1.6 Importance of Software Architecture

**13 Reasons Architecture Matters:**
1. Inhibits or enables quality attributes
2. Allows reasoning about and managing change
3. Enables early prediction of system qualities
4. Enhances communication among stakeholders
5. Carrier of earliest, hardest-to-change design decisions
6. Defines constraints on implementation
7. Dictates structure of organization (Conway's Law)
8. Basis for evolutionary prototyping
9. Key artifact for cost and schedule reasoning
10. Transferable, reusable model for product lines
11. Focuses attention on assembly of components
12. Channels creativity of developers
13. Foundation for training new team members

## 1.7 Contexts of Software Architecture

| Context | Focus | Key Factors |
|---------|-------|-------------|
| **Technical** | Role in system | Quality attributes, technical environment, industry practices |
| **Project Life-Cycle** | Relation to development phases | Waterfall, Iterative, Agile, Model-driven |
| **Business** | Effect on business environment | Business goals, market position, customer needs |
| **Professional** | Role of architect | Skills, knowledge, duties, leadership |

**Architecture Influence Cycle:**
Business + Technical + Project + Professional → Stakeholders → Architect → Architecture → System → (cycles back)

## 1.8 Architecture Competence

**Individual Competence:**
- Combines technical knowledge, business knowledge, soft skills (leadership, negotiation)
- SEI framework: Duties, Skills, Knowledge

**Organizational Competence:**
- Standardized processes
- Career paths for architects
- Governance

---

# Module 2: Quality Attributes

## 2.1 Understanding Quality Attributes

**System Requirements Categories:**
- **Functional:** What system must do, how it behaves
- **Quality Attribute:** Annotate functional requirements (how fast, how secure, etc.)
- **Constraints:** Design decisions with zero degrees of freedom

**Functionality vs. Quality Attributes:**
- Functionality: Ability to do intended work
- Quality attributes are orthogonal to functionality
- Multiple architectures can satisfy same functionality
- Quality attributes differentiate architectures

## 2.2 Quality Attribute Scenarios

**Six Parts:**
1. **Source:** Entity generating stimulus
2. **Stimulus:** Condition requiring response
3. **Environment:** Conditions under which stimulus occurs
4. **Artifact:** System or portion stimulated
5. **Response:** Activity undertaken as result
6. **Response Measure:** Measurable outcome

**General vs. Concrete Scenarios:**
- **General:** System-independent, can pertain to any system
- **Concrete:** Specific to particular system under consideration

## 2.3 Availability

**Definition:** Property that software is there and ready to carry out its task when needed; about minimizing service outage time by mitigating faults.

**Availability General Scenario:**

| Portion | Possible Values |
|---------|-----------------|
| Source | Internal/external: people, hardware, software, physical infrastructure |
| Stimulus | Fault: omission, crash, incorrect timing, incorrect response |
| Artifact | Processors, communication channels, persistent storage, processes |
| Environment | Normal, startup, shutdown, repair mode, degraded, overloaded |
| Response | Prevent fault, detect fault, recover from fault |
| Response Measure | Availability %, time to detect, time to repair, degraded mode time |

**Availability Tactics:**

| Category | Tactics |
|----------|---------|
| **Detect Faults** | Ping/Echo, Monitor, Heartbeat, Timestamp, Sanity Checking, Condition Monitoring, Voting, Exception Detection, Self-Test |
| **Recover from Faults (Preparation & Repair)** | Active Redundancy, Passive Redundancy, Spare, Exception Handling, Rollback, Software Upgrade, Retry, Ignore Faulty Behavior, Degradation, Reconfiguration |
| **Recover from Faults (Reintroduction)** | Shadow, State Resynchronization, Escalating Restart, Non-Stop Forwarding |
| **Prevent Faults** | Removal from Service, Transactions, Predictive Model, Exception Prevention, Increase Competence Set |

## 2.4 Performance

**Definition:** About time and software system's ability to meet timing requirements.

**Performance General Scenario:**

| Portion | Possible Values |
|---------|-----------------|
| Source | Internal or external to system |
| Stimulus | Arrival of periodic, sporadic, or stochastic event |
| Artifact | System or components |
| Environment | Normal, emergency, peak load, overload |
| Response | Process events, change level of service |
| Response Measure | Latency, deadline, throughput, jitter, miss rate |

**Performance Tactics:**

| Category | Tactics |
|----------|---------|
| **Control Resource Demand** | Manage Sampling Rate, Limit Event Response, Prioritize Events, Reduce Overhead, Bound Execution Times, Increase Resource Efficiency |
| **Manage Resources** | Increase Resources, Introduce Concurrency, Maintain Multiple Copies of Computations, Maintain Multiple Copies of Data, Bound Queue Sizes, Schedule Resources |

## 2.5 Usability

**Definition:** Concerned with how easy it is for user to accomplish desired task and kind of user support system provides.

**Usability Areas:**
- Learning system features
- Using system efficiently
- Minimizing impact of errors
- Adapting system to user needs
- Increasing confidence and satisfaction

**Usability General Scenario:**

| Portion | Possible Values |
|---------|-----------------|
| Source | End user, possibly specialized role |
| Stimulus | User tries to use system efficiently, learn, minimize errors, adapt, configure |
| Environment | Runtime or configuration time |
| Artifact | System or specific portion |
| Response | Provide features or anticipate needs |
| Response Measure | Task time, errors, tasks accomplished, satisfaction, knowledge gain |

**Usability Tactics:**

| Category | Tactics |
|----------|---------|
| **Support User Initiative** | Cancel, Undo, Pause/Resume, Aggregate |
| **Support System Initiative** | Maintain Task Model, Maintain User Model, Maintain System Model |

## 2.6 Security

**Definition:** Measure of system's ability to protect data and information from unauthorized access while still providing access to authorized.

**CIA Characteristics:**
- **Confidentiality:** Data/services protected from unauthorized access
- **Integrity:** Data/services not subject to unauthorized manipulation
- **Availability:** System available for legitimate use

**Supporting Characteristics:**
- **Authentication:** Verifies identities
- **Nonrepudiation:** Sender cannot deny sending; recipient cannot deny receiving
- **Authorization:** Grants user privileges to perform task

**Security General Scenario:**

| Portion | Possible Values |
|---------|-----------------|
| Source | Human or system, previously identified or unknown |
| Stimulus | Unauthorized attempt to display, change, delete data, access services, reduce availability |
| Artifact | System services, data, components, resources |
| Environment | Online/offline, connected/disconnected, behind firewall/open |
| Response | Transactions protected, data integrity, authentication, nonrepudiation, availability, tracking |
| Response Measure | Compromise extent, detection time, attacks resisted, recovery time, data vulnerability |

**Security Tactics:**

| Category | Tactics |
|----------|---------|
| **Detect Attacks** | Detect Intrusion, Detect Service Denial, Verify Message Integrity, Detect Message Delay |
| **Resist Attacks** | Identify Actors, Authenticate Actors, Authorize Actors, Limit Access, Limit Exposure, Encrypt Data, Separate Entities, Change Default Settings |
| **React to Attacks** | Revoke Access, Lock Computer, Inform Actors |
| **Recover from Attacks** | Audit, (plus Availability recovery tactics) |

## 2.7 Modifiability

**Definition:** About change and cost/risk of making changes.

**Three Questions:**
1. What can change?
2. What is likelihood of change?
3. When is change made and who makes it?

**Modifiability General Scenario:**

| Portion | Possible Values |
|---------|-----------------|
| Source | End user, developer, system administrator |
| Stimulus | Add, modify, delete, or change functionality/quality attribute |
| Artifact | System, specific portion |
| Environment | Runtime, compile time, build time, design time |
| Response | Make modification, test, deploy |
| Response Measure | Cost, time, impact on other functions |

**Modifiability Tactics:**

| Category | Tactics |
|----------|---------|
| **Increase Cohesion** | Increase Semantic Coherence, Encapsulate, Use an Intermediary |
| **Reduce Coupling** | Restrict Dependencies, Refactor, Abstract Common Services |
| **Defer Binding** | Later binding is better (but mechanisms cost more) |

## 2.8 Other Quality Attributes

| Attribute | Definition | Key Concern |
|-----------|------------|-------------|
| **Interoperability** | Degree to which systems can exchange meaningful information | Locate, Manage Interfaces |
| **Testability** | Ease of demonstrating faults through testing | Control/observe state, Limit complexity |
| **Scalability** | Handle growth (horizontal/vertical) | Resources, Load balancing |
| **Portability** | Ease of changing platform | Abstraction, Layering |
| **Deployability** | How executable arrives at host | Packaging, Installation |
| **Mobility** | Movement and affordances of platform | Size, Display, Input, Bandwidth |
| **Monitorability** | Ability of operations staff to monitor | Instrumentation, Logging |
| **Safety** | Ability to avoid damage/injury/loss of life | Prevention, Detection, Recovery |
| **Conceptual Integrity** | Consistency in design | Same thing done same way |
| **Marketability** | Architecture as marketing feature | Service-oriented, Cloud-based |
| **Quality in Use** | Effectiveness, Efficiency, Freedom from risk | User perspective |

**Standard Lists (ISO/IEC 25010):**
- Functional Suitability
- Performance Efficiency
- Compatibility
- Usability
- Reliability
- Security
- Maintainability
- Portability

---

# Module 3: Architecture Requirements and Design

## 3.1 Architecturally Significant Requirements (ASRs)

**Definition:** Requirement that will have a profound effect on architecture of a system; has measurable effect on computer system's architecture.

**Indicators of Architectural Significance:**
- Holds significant business value and/or presents technical risk
- Important to key stakeholder with considerable influence
- Unique — no existing components address it
- QoS/SLA aspects differ from current architecture
- Previously led to budget overruns or client dissatisfaction

**Common Examples:**
- Scalability: Support future growth with minimal effort
- Security: Protect data and prevent unauthorized access
- Availability: Specific uptime or reliability rate

**Examples of ASRs:**
- Audit trail (File or DB? Depends on usage)
- Third party interaction (Communication mechanisms supported)
- Notifications (SMS, Email, MQ; one-way or acknowledgement)

## 3.2 Gathering ASRs

**From Requirements Documents:**
- Follow MoSCoW (Must, Should, Could, Won't)
- Include Quality Attributes per SWEBOK
- Look for BSR (Basic Skill Requirements)
- Absence of clear QA is challenge for architect

**Look for Leads from Design Decision:**

| Design Decision Category | Look for Requirements Addressing |
|--------------------------|----------------------------------|
| Allocation of Responsibilities | Planned evolution, user roles, modes |
| Coordination Model | Timeliness, currency, correctness |
| Data Model | Processing steps, information flows, entities, access rights |
| Management of Resources | Time, concurrency, memory, scalability |
| Mapping among Elements | Team plans, processors, networks |
| Binding Time | Extension, portability, configurations |
| Choice of Technology | Named technologies, changes |

**From Stakeholders (Quality Attribute Workshop - QAW):**

| Step | Activity |
|------|----------|
| 1 | Presentation & Introduction |
| 2 | Business Mission Presentation (1 hour) |
| 3 | Architectural Plan Presentation |
| 4 | Identification of Architectural Drivers |
| 5 | Scenario Brainstorming |
| 6 | Scenario Consolidation |
| 7 | Scenario Prioritization (votes = 30% of scenarios) |

**From Business Goals (PALM Method):**

| Step | Activity |
|------|----------|
| 1 | PALM Overview |
| 2 | Business Drivers Presentation |
| 3 | Architecture Drivers Presentation |
| 4 | Business Goals Elicitation |
| 5 | Identify Quality Attributes from Goals |
| 6 | Assign Pedigree to Quality Attributes |
| 7 | Exercise Conclusion |

**Business Goals Format:** "X" to "Y" by when?
- X = Current State
- Y = Future State
- Example: Market share by 15%

**Business Goal Categories:**
1. Contributing to growth and continuity
2. Meeting financial objectives
3. Meeting personal objectives
4. Meeting responsibility to employees
5. Meeting responsibility to society
6. Meeting responsibility to state
7. Meeting responsibility to shareholders
8. Managing market position
9. Improving business processes
10. Managing quality and reputation
11. Managing change in environmental factors

**Example: BHIM App**
- Business Goal: Speed up cashless transactions
- Quality Attributes: Usability (easy steps), Security (3-factor)

## 3.3 Utility Tree

**Definition:** Hierarchical representation to capture and prioritize key quality attributes, breaking them down into detailed scenarios.

**Structure:**
```
Utility (Root)
├── Quality Attribute (Branch)
│   ├── Attribute Refinement (Sub-branch)
│   │   └── ASR/Scenario (Leaf) [Priority: (Business Value, Technical Risk)]
```

**ASR in Utility Tree:**

| Quality Attribute | Refinement | ASR | Priority |
|-------------------|------------|-----|----------|
| Performance | Transaction response time | User updates patient account under peak load; completes < 0.75 sec | (H, M) |
| Usability | Proficiency training | New hire proficient in < 1 week | (M, L) |
| Security | Confidentiality | Physical therapist sees orthopaedic data but not financial | (H, M) |
| Availability | No Downtime | System supports 24/7 web access | (L, L) |
| Modifiability | Routine changes | Bug fix with < 3 person-days effort | (H, M) |

**Priority Ratings:**
- H = High
- M = Moderate
- L = Low

## 3.4 Attribute-Driven Design (ADD)

**Steps:**
1. Choose module to decompose
2. Refine module based on ASRs
3. Choose architectural tactics
4. Instantiate modules and assign responsibilities
5. Define interfaces
6. Verify and refine
7. Iterate

## 3.5 Agile vs. Traditional Architecture

**Agile Manifesto:**
- Individuals and interactions over processes and tools
- Working software over comprehensive documentation
- Customer collaboration over contract negotiation
- Responding to change over following a plan

**How Much Architecture?**

| Project Size | Sweet Spot (% of schedule) |
|--------------|---------------------------|
| 10 KSLOC | ~0% |
| 100 KSLOC | ~20% |
| 1,000 KSLOC | ~40% |

**Guidelines for Agile Architects:**
- Large, stable requirements: Do significant up-front architecture
- Large, unstable requirements: Start with candidate architecture, evolve
- Small, uncertain requirements: Agree on major patterns, don't over-document

**Agility and Documentation:**
- Write for the reader
- If reader doesn't need it, don't write it
- Remember maintainers and newcomers

**Incremental Commitment Model Principles:**
- Commitment and accountability of success-critical stakeholders
- Stakeholder "satisficing"
- Incremental and evolutionary growth
- Iterative development
- Interleaved definition and development
- Risk management

---

# Module 4: Documenting Software Architecture

## 4.1 Architecture Documentation

**Uses and Audiences:**
- Communication among stakeholders
- Analysis of quality attributes
- Education of new team members
- Decision record
- Conformance baseline

**Key Principle:** "Documenting an architecture is a matter of documenting the relevant views and then adding documentation that applies to more than one view."

**Problem:** Architecture documents do not address concerns of all stakeholders.
**Solution:** Use different notations for several views, each addressing specific set of concerns.

## 4.2 Philippe Kruchten's 4+1 View Model

| View | Purpose | Stakeholders | Quality Attributes | Notation |
|------|---------|--------------|-------------------|----------|
| **Logical** | Object model of design | End users | Functional requirements | Booch/UML |
| **Process** | Concurrency and synchronization | Integrators | Scalability, Concurrency, Performance | Garlan & Shaw |
| **Development** | Static organization in development environment | Programmers, Managers | Modifiability, Reusability | Booch |
| **Physical** | Mapping to hardware | System Engineers | Reliability, Availability, Performance | Informal |
| **Scenarios** | Validation and illustration | All | All | Similar to logical |

**Logical View:**
- Elements: Key abstractions (objects, classes)
- Notation: Booch (simplified)
- Tool: Rational Rose

**Process View:**
- Multiple levels of abstraction
- Process = grouping of tasks forming executable unit
- Major tasks: Architecture relevant
- Minor tasks: Buffering
- Notation: Process, Simplified Process, Periodic process adornment

**Development View:**
- Static organization of software
- Hierarchy of layers
- Software management, reuse, constraints of tools
- Style: Layered

**Physical View:**
- Mapping software to hardware
- Reflects distributed aspect
- Two architectures: Test/development and deployment

**Scenarios:**
- Putting all 4 views together
- Viewer: All users and evaluators
- Considers: System consistency and validity

**View Interconnection:**
- Start with Logical view
- Move to Development / Process view
- Finally go to Physical view

**Two Strategies:**
- **Inside-out:** Starting from Logical structure
- **Outside-in:** Starting from physical structure

**Documenting the Architecture:**
- Software Architecture Document: Follows 4+1 views
- Software Design Guidelines: Captures most important design decisions

**Documentation Package:**

| Section | Content |
|---------|---------|
| Title Page | |
| Change History | |
| Table of Contents | |
| List of Figures | |
| 1. Scope | |
| 2. References | |
| 3. Software Architecture | |
| 4. Architectural Goals & Constraints | |
| 5. Logical Architecture | |
| 6. Process Architecture | |
| 7. Development Architecture | |
| 8. Physical Architecture | |
| 9. Scenarios | |
| 10. Size and Performance | |
| 11. Quality | |
| Appendices | Acronyms, Definitions, Design Principles |

**Limitations:**
- Missing tools to integrate views
- Inconsistency problem
- More tangible in maintenance

## 4.3 Modernizing the 4+1 View

| View | Modern Context |
|------|----------------|
| **Logical** | UML 2.5: Class diagrams, Package diagrams; Microservices: Service API contracts, Bounded Contexts |
| **Process** | Containers (Docker), Serverless (Lambda); Sequence diagrams, Activity diagrams; Focus on latency, deadlocks, eventual consistency |
| **Development** | DevOps, CI/CD; Monorepo vs. Multi-repo; Third-party dependencies (NuGet, NPM, Maven) |
| **Physical** | VMs, Kubernetes; Elasticity, Network topologies; Deployment diagrams |

**Architectural Drift:**
- Problem: Code evolves faster than documentation
- Solution: Architecture as Code (Mermaid.js, PlantUML), Automated checks (fitness functions)
- Goal: Living document, not static PDF

## 4.4 Case Study: Modern E-commerce Platform

| View | Focus | Stakeholders |
|------|-------|--------------|
| Logical | Classes, modules, data model | Developers |
| Process | Runtime components, concurrency, data flow | Integrators |
| Development | Code organization, repositories, build | Developers, Build Engineers |
| Physical | Mapping to hardware/cloud, focus and scalability | System Engineers, Project Managers |

**Stakeholder Mapping:**
- Complex UI? → Prioritize Logical View
- High-performance/traffic? → Prioritize Process View
- Large, distributed team? → Prioritize Development View
- Cloud-native/Complex security? → Prioritize Physical View

---

# Module 5: Architecture Conformance, Testing, Reconstruction, Layered Architecture, ATAM

## 5.1 Ensuring Conformance to Architecture

**Code May Drift Away from Architecture:**
- Not sticking to discipline of layers
- Accessing database directly without data access layer
- Notifying modules one by one instead of publish-subscribe
- Not using common logging mechanism

**Techniques to Keep Code and Architecture Consistent:**

| Technique | Description |
|-----------|-------------|
| **Embed Design in Code** | Architecturally Evident Coding Style: Indicate layer, pattern role |
| **Use Frameworks** | Spring, Hibernate, AUTOSAR |
| **Use Code Templates** | Fault-tolerant template with sections |
| **Update Architecture Documentation** | Mark "No longer applicable"; synchronize at release |
| **Educate New Team Members** | Architecture overview sessions |
| **Code Reviews** | Checklist-based |
| **Folder Structure** | Organize by architectural aspect |
| **Automated Checks** | SonarJ, Lattix, Structure 101 |

**Examples of Frameworks:**
- Spring
- Hibernate
- AUTOSAR (AUTomotive Open System ARchitecture)
- Publish-Subscribe (JMS)
- Workflow framework (SalesForce.com)
- Rules engine (DROOLS)
- Logging framework (Log4J)

**Fault-Tolerant Template Example:**
```
Get event
Case (Event type)
    Normal:
        // Received by primary process
        Process X;
        Send state to backup process;
        Process Y;
        Send state to backup process;
    Update State data:
        // Received by backup process
        Update state data;
    Switch over:
        // Received by backup process
        Notify clients about change in Primary;
End case;
```

## 5.2 Architecture and Testing

**How Architecture Helps Testing:**
1. **Prioritize Test Cases Based on ASRs:**
   - Utility tree identifies high-priority scenarios
   - High business value + high architectural impact = high priority test cases

2. **Create Integration Test Plan:**
   - Architecture tells what modules interact
   - Architecture tells which modules depend on which
   - Helps identify integration test cases and modules needed

3. **Design for Testability:**
   - Ability to switch data sources (Test/Production)
   - Roll back changes
   - Replace components (simulators)

**Work Products for Prioritizing Test Cases:**

| Work Product | How It Helps |
|--------------|--------------|
| Utility Tree | Identifies high-priority scenarios |
| Quality Attribute Scenarios | Provides specific, testable requirements |
| ASR List | Focuses testing |
| Module Views | Shows what to unit test |
| C&C Views | Shows what to integration test |

## 5.3 Architecture Reconstruction

**Purpose:**
- Understand architecture of system with no documentation
- Migrate from old technology to new
- Identify reusable components

**Phases:**

| Phase | Description | Output |
|-------|-------------|--------|
| **1. Raw View Extraction** | Extract from source code, execution traces, build scripts | Classes, files, caller-callee, global data access |
| **2. DB Construction** | Put extracted data into common format | Repository |
| **3. View Fusion** | Combine views from different sources | Consolidated view |
| **4. Architecture Analysis** | Validate correctness, check violations | Reconstructed architecture |
| **5. Iterate** | Refine as needed | Refined architecture |

**Examples of Extracted Elements and Relations:**

| Source Element | Relation | Target Element | Description |
|----------------|----------|----------------|-------------|
| File | includes | File | C preprocessor #include |
| File | contains | Function | Definition of function |
| File | defines_var | Variable | Definition of variable |
| Directory | contains | Directory | Subdirectory |
| Directory | contains | File | File in directory |
| Function | calls | Function | Static function call |
| Function | access_read | Variable | Read access |
| Function | access_write | Variable | Write access |

**Execution Trace:** Tree representation of trace of routine calls

**Case Study: Vanish System**
- Tool: ARMIN (ARchitecture Reconstruction and MINing)
- Reconstructed architecture: Dialogue, Functional_Core_Adapter, Logical_Interaction, Utilities, Functional_Core, Presentation
- Analysis: Not strictly layered

**Tools:**
- Dali
- ARMIN
- Lattix
- SonarJ
- Structure 101

**Architecture Analysis Example:**
- Check conformance to architecture
- Restriction: Layer calls adjacent layers only
- Restriction: All DB access via entity bean only

**Example Violation (Sonar):**
- No portion of application should depend on JUnit
- Sonar detects rule violation

**Example Violation (Dynamic Analysis):**
- All database access managed by entity beans
- Discovered via dynamic analysis

## 5.4 Layered Architecture

**Typical Layers:**

| Layer | Responsibility | Technologies |
|-------|----------------|--------------|
| **Presentation** | UI/UX, user input | React, Angular, Mobile |
| **Business/Logic** | Business rules, validation, processing | Java, .NET, Python |
| **Service** | APIs, orchestration, external integration | REST, SOAP, GraphQL |
| **Data Access** | Persistence, ORM, database connections | Hibernate, JPA, SQL |
| **Cross-Cutting** | Security, logging, communication | Aspects, Frameworks |

**Benefits:**
- Separation of Concerns
- Loose coupling
- Reusable lower layer components
- Exchangeable parts

**Presentation Layer Techniques:**
- Cache frequently used data (client-side vs. server-side)
- Asynchronous communication (AJAX)
- Responsive design for different form factors
- Memcached for distributed caching

**Business Layer Techniques:**
- Application façade to hide internal modules
- Session management (client-based vs. server-based)
- Workflow engine, Rules engine
- Aspect-oriented design for cross-cutting concerns

**Data Layer Techniques:**
- DB connection pool
- Multiple copies of data
- Transactions for atomicity
- Object-Relational mapping (Hibernate)
- Stored procedures
- Parameterized SQL queries (prevent SQL injection)

**Service Layer Techniques:**
- Provide services to external systems
- Handle duplicate requests
- Handle out-of-sequence messages
- Handle communication failures (retry, queue)

**Example: Insurance Policy Processing Workflow:**
Determine Order Complete → Check Record → Determine Premium → Approve/Reject → Notify Applicant

**Aspect-Oriented Design:**
- Encapsulate cross-cutting concerns into aspects
- Examples: Logging, Auditing, Caching, Security
- Modules in different layers call cross-cutting subsystem at different points

**Logging & Instrumentation:**
- Log actionable information (errors)
- Log only essential data (logging is expensive)
- Instrumentation: request rate, error rate, duration, queue length

## 5.5 Architecture Tradeoff Analysis Method (ATAM)

**Definition:** Comprehensive method for evaluating software architectures by having project decision makers and stakeholders articulate precise quality attribute requirements and illuminating architectural decisions relevant to each high-priority scenario.

**Participants:**

| Role | Responsibilities |
|------|------------------|
| **Evaluation Team** | External, unbiased, 3-5 people |
| **Project Decision Makers** | Project manager, customer, architect |
| **Architecture Stakeholders** | Developers, testers, integrators, maintainers, users |

**Phases:**

| Phase | Activity | Participants | Duration |
|-------|----------|--------------|----------|
| **0** | Partnership and preparation | Evaluation team, decision makers | Weeks |
| **1** | Evaluation (Steps 1-6) | Evaluation team, decision makers | 1-2 days |
| **2** | Evaluation (Steps 7-9) | Evaluation team, decision makers, stakeholders | 2 days |
| **3** | Follow-up | Evaluation team | Report generation |

**Steps 1-9:**

| Step | Activity |
|------|----------|
| 1 | Present ATAM |
| 2 | Present Business Drivers |
| 3 | Present Architecture |
| 4 | Identify Architectural Approaches |
| 5 | Generate Utility Tree |
| 6 | Analyze Architectural Approaches |
| 7 | Brainstorm and Prioritize Scenarios |
| 8 | Analyze Architectural Approaches (Round 2) |
| 9 | Present Results |

**Outputs:**

| Output | Description |
|--------|-------------|
| Concise Architecture Presentation | Documented architecture |
| Business Goals | Articulated and captured |
| Prioritized Quality Attribute Scenarios | From utility tree |
| Risks | Decisions that may lead to undesirable consequences |
| Non-Risks | Decisions deemed safe |
| Risk Themes | Overarching systemic weaknesses |
| Sensitivity Points | Decisions affecting specific quality attributes |
| Trade-Off Points | Decisions affecting multiple quality attributes |
| Mapping of Decisions to Requirements | How decisions achieve quality attributes |

**Intangible Outputs:**
- Sense of community among stakeholders
- Open communication channels
- Better overall understanding of architecture
- Often longest-lasting results

**Example Risk/Sensitivity/Trade-Off:**

| Type | Example |
|------|---------|
| **Risk** | Heartbeat frequency affects detection time |
| **Sensitivity Point** | Number of database clients affects transactions/sec |
| **Trade-Off** | Higher heartbeat frequency = better availability but more processing time |

**Lightweight Architecture Evaluation:**
- For smaller, less risky projects
- Single day or half-day meeting
- Internal participants
- Steps: Present ATAM, Business Drivers, Architecture, Identify Approaches, Utility Tree, Analyze, Present Results

**Typical Agenda (4-6 Hours):**

| Step | Time |
|------|------|
| 1. Present ATAM | 0 hours |
| 2. Present Business Drivers | 0.25 hours |
| 3. Present Architecture | 0.5 hours |
| 4. Identify Architectural Approaches | 0.25 hours |
| 5. Generate QA Utility Tree | 0.5-1.5 hours |
| 6. Analyze Architectural Approaches | 2-3 hours |
| 7. Brainstorm Scenarios | 0 hours (omitted) |
| 8. Analyze Architectural Approaches | 0 hours (omitted) |
| 9. Present Results | 0.5 hours |

**Summary:**
- If system is important enough for explicit architecture, it should be evaluated
- Number and extent of evaluations varies
- Designer should evaluate during important decisions
- Lightweight evaluations can be peer reviews
- ATAM is comprehensive method
- Lightweight Architecture Evaluation is inexpensive, low-ceremony

---

# Summary Tables

## Quality Attribute Cheat Sheet

| Quality Attribute | What It Means | Typical Architectural Concern | Example Scenario |
|-------------------|---------------|-------------------------------|------------------|
| **Availability** | System ready when needed | Fault detection, recovery, redundancy | 99.999% uptime |
| **Performance** | Meets timing requirements | Latency, throughput, resource management | < 100ms response |
| **Security** | Protects data from unauthorized access | Authentication, authorization, encryption | HIPAA compliance |
| **Scalability** | Handles growth | Horizontal/vertical scaling, load balancing | Millions of messages/sec |
| **Modifiability** | Ease of making changes | Cohesion, coupling, defer binding | Add feature in 2 weeks |
| **Testability** | Ease of testing | Control/observe state, limit complexity | 85% coverage in 3 hours |
| **Interoperability** | Systems exchange information | Adapters, data transformation | Integrate with EHR |
| **Usability** | Ease of use | User initiative, system initiative | Productive in 2 minutes |

## Important Frameworks / Methods

| Method | Purpose | Key Steps | When to Use |
|--------|---------|-----------|-------------|
| **QAW** | Elicit quality attribute requirements | Introductions → Business Mission → Architectural Plan → Identify Drivers → Brainstorm Scenarios → Consolidate → Prioritize | Early requirements |
| **Architectural Drivers** | Identify key factors | Business goals, constraints, quality attributes | Before design |
| **Utility Tree** | Prioritize ASRs | Root → Branches → Leaves → Priority | ASR identification |
| **ADD** | Systematic design | Choose module → Refine → Choose tactics → Instantiate → Define interfaces → Verify | Design phase |
| **4+1 View** | Document architecture | Logical, Process, Development, Physical, Scenarios | Documentation |
| **Layered Architecture** | Separation of concerns | Presentation, Business, Service, Data | Enterprise applications |
| **ATAM** | Evaluate architecture | Present ATAM → Business Drivers → Architecture → Identify Approaches → Utility Tree → Analyze → Brainstorm → Analyze → Present Results | After design |
| **Conformance** | Ensure implementation matches design | Architecturally evident coding, frameworks, templates, documentation | During implementation |
| **Testing** | Test architectural properties | Prioritize test cases based on ASRs, integration test plan | Testing phase |
| **Reconstruction** | Understand existing systems | Raw view extraction → DB construction → View fusion → Analysis → Iterate | Legacy modernization |

## Common Exam Traps

| Trap | How to Avoid |
|------|--------------|
| Confusing business goals with architectural drivers | Business goals are high-level; architectural drivers are specific factors |
| Treating quality attributes as functional requirements | Quality attributes annotate functional requirements |
| Identifying quality attribute without measurable scenario | Always create scenario with source, stimulus, response, measure |
| Ignoring trade-offs | Every decision has trade-offs; discuss them |
| Assuming highest-performance architecture is best | Best depends on business goals and constraints |
| Confusing architecture views with architectural styles | Views are representations; styles are patterns |
| Treating documentation as one-time activity | Update as architecture evolves |
| Ignoring architecture conformance | Implementation can drift; check conformance |
| Assuming documented architecture = current implementation | Documentation may be outdated; reconstruct if needed |
| Confusing reconstruction with design | Reconstruction is understanding; design is creating |
| Memorizing ATAM steps without understanding application | Focus on applying concepts to scenarios |
| Forgetting to prioritize ASRs | Use utility tree for prioritization |
| Overlooking stakeholder concerns | Identify and engage stakeholders early |
| Ignoring constraints (cost, time, technology) | Constraints are zero-degree-of-freedom decisions |
| Not considering real-time requirements | Real-time systems have hard deadlines |

---

# End of Comprehensive Summary

This summary covers all key concepts from the lecture slides across all five modules, organized for efficient revision and examination preparation.