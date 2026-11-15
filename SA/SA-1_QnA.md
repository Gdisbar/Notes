# Software Architecture — 50 Scenario-Based Questions & Answers

---

## Section 1 — Medium Difficulty (Questions 1–10)

---

### Q1. A mid-sized insurance company is building a new claims processing system. The business wants the system to be available 24/7, but the IT budget is limited. The current monolithic system experiences frequent downtime during peak claim periods. As the architect, how would you identify the architectural drivers and propose an approach?

**Answer:**

**Analysis:** The primary architectural drivers are:
- **Availability** (24/7 operation, especially during peak periods)
- **Cost constraint** (limited budget)
- **Performance** (handling peak claim volumes)

**Relevant Quality Attributes:** Availability, Performance, Cost (constraint).

**Architectural Reasoning:**
The current monolithic system has a single point of failure. To achieve high availability without massive cost, the architect should:
1. Identify the most critical components (e.g., claim submission, approval workflow) that must remain available.
2. Apply **availability tactics** such as:
   - **Active redundancy** for critical components (hot spare)
   - **Passive redundancy** for less critical components (warm spare)
   - **Heartbeat monitoring** to detect failures
   - **Rollback** for failed transactions
3. Use a **layered architecture** with clear separation between presentation, business, and data layers so that failures in one layer don't cascade.

**Trade-Offs:**
- Full active redundancy across all components would exceed budget. Therefore, prioritize redundancy for components with highest business impact.
- Cost vs. availability: accept slightly lower availability for non-critical modules (e.g., reporting).

**Recommendation:**
Adopt a **partially redundant architecture** where the claim submission and approval engine are replicated, while reporting and analytics run on a single instance with graceful degradation. Use heartbeat monitoring and automated failover.

**Industry Relevance:** This is common in insurance and banking where core transaction processing must be highly available but budgets are constrained.

**Key concepts tested:** Availability tactics, architectural drivers, cost-availability trade-off.

---

### Q2. An e-commerce platform experiences slow page loads during flash sales. The business wants to improve performance without changing the user interface. What architectural tactics would you apply, and what trade-offs might arise?

**Answer:**

**Analysis:** The performance bottleneck is likely due to:
- High concurrent user load
- Database contention
- Synchronous processing of requests

**Relevant Quality Attributes:** Performance, Scalability, Usability (response time).

**Architectural Reasoning:**
Apply **performance tactics**:
1. **Control Resource Demand:**
   - Prioritize events (e.g., checkout requests over browsing)
   - Limit event response (queue non-critical requests)
   - Reduce overhead (remove unnecessary intermediaries)
2. **Manage Resources:**
   - Introduce concurrency (thread pools, async processing)
   - Maintain multiple copies of computations (replicas)
   - Cache frequently accessed data (e.g., product catalog) using **client-side and server-side caching** (as discussed in lecture)
   - Use **AJAX** for partial page updates instead of full page reloads

**Trade-Offs:**
- Caching improves performance but introduces **data staleness** (e.g., product price changes may not reflect immediately).
- Concurrency improves throughput but increases **complexity** and risk of race conditions.
- Prioritizing checkout may degrade browsing experience for some users.

**Recommendation:**
Implement a **multi-layer caching strategy** (client-side for user preferences, server-side for product data), use AJAX for dynamic content loading, and introduce an **application façade** in the business layer to orchestrate requests efficiently. Monitor cache hit ratios and adjust TTL.

**Industry Relevance:** Flash sales at e-commerce giants like Amazon, Flipkart require such tactics.

**Key concepts tested:** Performance tactics, caching, AJAX, trade-offs with data consistency.

---

### Q3. A healthcare application stores sensitive patient records. The security team requires encryption of data at rest and in transit, but doctors complain that the system is slow. How would you balance security and performance?

**Answer:**

**Analysis:** The conflict is between **Security** (confidentiality, integrity) and **Performance** (response time).

**Relevant Quality Attributes:** Security, Performance, Usability.

**Architectural Reasoning:**
1. **Security tactics:**
   - Encrypt data at rest (AES-256) and in transit (TLS)
   - Authenticate actors (doctors, nurses) using multi-factor authentication
   - Authorize access based on roles (RBAC)
   - Audit all access to patient records
2. **Performance optimization:**
   - Use **hardware acceleration** for encryption/decryption
   - Cache decrypted data only in memory for active sessions
   - Use **asynchronous logging** for audit trails to avoid blocking
   - Optimize database queries with indexing and stored procedures
3. **Balance:**
   - Encrypt only sensitive fields (e.g., diagnosis, prescription) rather than entire database
   - Use **column-level encryption** for performance
   - Implement **session management** to avoid re-authentication for every request

**Trade-Offs:**
- Full database encryption adds latency; selective encryption reduces overhead.
- Caching decrypted data improves speed but increases risk if memory is compromised.

**Recommendation:**
Adopt **selective encryption** for sensitive fields, use hardware security modules (HSMs) for key management, and implement caching with strict memory protection. Conduct regular security audits.

**Industry Relevance:** Electronic Health Records (EHR) systems like Epic, Cerner face this exact trade-off.

**Key concepts tested:** Security tactics, performance tactics, encryption, trade-off analysis.

---

### Q4. A logistics company wants to migrate its legacy mainframe-based system to a web-based platform. The legacy system has no documentation. How would you approach architecture reconstruction?

**Answer:**

**Analysis:** The goal is to understand the existing architecture before migration.

**Relevant Quality Attributes:** Modifiability, Interoperability, Maintainability.

**Architectural Reasoning:**
Follow **architecture reconstruction** steps (from lecture):
1. **Raw view extraction:**
   - Extract information from source code, execution traces, build scripts
   - Identify classes, files, caller-callee relationships, global data access
2. **DB construction:**
   - Put extracted data into a common format (repository)
3. **View fusion:**
   - Combine static view (source code analysis) with dynamic view (execution traces)
   - Group elements into layers or subsystems based on expert input
4. **Architecture analysis:**
   - Validate correctness of reconstructed architecture
   - Check for violations (e.g., layers calling non-adjacent layers)
5. **Iterate** as needed

**Tools:** ARMIN, Dali, Lattix, SonarJ, Structure 101.

**Trade-Offs:**
- Reconstruction is time-consuming but essential for successful migration.
- Automated tools may miss dynamic dependencies; manual expert review is needed.

**Recommendation:**
Use a combination of automated extraction (ARMIN) and expert workshops to reconstruct the architecture. Document the reconstructed architecture using **module and C&C views**. Identify reusable components (e.g., logging, security) for the new platform.

**Industry Relevance:** Banks, insurance companies, and government agencies frequently migrate mainframe systems to modern platforms.

**Key concepts tested:** Architecture reconstruction, raw view extraction, view fusion, tools.

---

### Q5. A SaaS product needs to support multiple tenants with different customization requirements. How would you design the architecture to balance modifiability and scalability?

**Answer:**

**Analysis:** The system must support **multi-tenancy** with tenant-specific customizations.

**Relevant Quality Attributes:** Modifiability, Scalability, Security.

**Architectural Reasoning:**
1. **Modifiability tactics:**
   - **Increase cohesion:** Separate tenant-specific logic into distinct modules
   - **Reduce coupling:** Use **intermediary** (e.g., API gateway) between tenants and core services
   - **Defer binding:** Use configuration files or rules engines for tenant-specific business rules (as discussed in lecture: workflow engine, rules engine)
2. **Scalability tactics:**
   - **Horizontal scalability** (scaling out): Add more servers to handle tenant load
   - **Vertical scalability** (scaling up): Increase resources for database
   - Use **shared database with tenant ID** or **separate databases per tenant** based on isolation requirements
3. **Security:**
   - Ensure tenant data isolation
   - Authenticate and authorize tenant users separately

**Trade-Offs:**
- Shared database reduces cost but risks data leakage; separate databases increase isolation but raise cost and complexity.
- Customizations increase modifiability but can lead to code fragmentation.

**Recommendation:**
Adopt a **layered architecture** with:
- **Presentation layer:** Tenant-specific UI themes
- **Business layer:** Core services + tenant-specific extensions via **Application Façade** and **rules engine**
- **Data layer:** Shared database with tenant ID for cost efficiency, or separate schemas for large tenants
- **Service layer:** APIs for tenant-specific integrations

**Industry Relevance:** Salesforce, Zoho, and other SaaS providers use similar patterns.

**Key concepts tested:** Multi-tenancy, modifiability tactics, scalability, layered architecture.

---

### Q6. A hospital's patient monitoring system must respond to critical alerts within 2 seconds. The current system sometimes takes 10 seconds. What architectural changes would you recommend?

**Answer:**

**Analysis:** This is a **real-time performance** issue with a hard deadline.

**Relevant Quality Attributes:** Performance, Availability, Reliability.

**Architectural Reasoning:**
1. **Control Resource Demand:**
   - Prioritize critical alerts over routine notifications
   - Bound execution times for alert processing
   - Reduce overhead by removing unnecessary intermediaries
2. **Manage Resources:**
   - Introduce **concurrency** (dedicated thread for alerts)
   - Maintain **multiple copies of computations** (redundant alert processors)
   - Use **scheduling** to prioritize alert tasks
3. **Architecture style:**
   - Use **publish-subscribe** pattern for alert distribution (instead of polling)
   - Implement **heartbeat** and **watchdog** for monitoring
   - Use **active redundancy** for alert processing components

**Trade-Offs:**
- Dedicated resources for alerts increase cost but ensure timing.
- Publish-subscribe improves responsiveness but adds complexity.

**Recommendation:**
Adopt a **component-and-connector architecture** with:
- **Alert Publisher** component that publishes to a message queue
- **Alert Subscriber** components with dedicated threads
- **Priority-based scheduling** in the business layer
- **Real-time monitoring** using heartbeat and timestamp tactics

**Industry Relevance:** Medical devices, industrial control systems, and financial trading platforms require such real-time guarantees.

**Key concepts tested:** Real-time performance, concurrency, publish-subscribe, scheduling.

---

### Q7. A retail chain wants to integrate its online store with its physical store inventory system. The two systems were developed independently and use different data formats. How would you achieve interoperability?

**Answer:**

**Analysis:** The challenge is **interoperability** between heterogeneous systems.

**Relevant Quality Attributes:** Interoperability, Modifiability, Performance.

**Architectural Reasoning:**
1. **Locate tactics:**
   - Use a **directory service** to discover the inventory system
   - Implement **discover service** pattern
2. **Manage interfaces tactics:**
   - **Orchestrate:** Use a control mechanism to coordinate inventory updates
   - **Tailor interface:** Add translation capabilities (e.g., XML to JSON)
3. **Architecture:**
   - Introduce a **Service layer** that provides a unified API for inventory operations
   - Use **message queues** for asynchronous communication
   - Implement **data transformation** components to convert between formats
4. **Design checklist for interoperability:**
   - Ensure responsibilities for detecting, accepting, rejecting, and logging interoperability requests
   - Ensure coordination mechanisms can handle volume, timeliness, and jitter
   - Ensure data abstractions are consistent

**Trade-Offs:**
- Real-time synchronization vs. batch processing: Real-time improves accuracy but requires more infrastructure.
- Adding a service layer improves modifiability but introduces latency.

**Recommendation:**
Build a **Service layer** with:
- **Inventory Service** exposing REST APIs
- **Adapter components** for each system (online store, physical store)
- **Message queue** (e.g., JMS) for asynchronous updates
- **Data transformation** using XSLT or custom mappers

**Industry Relevance:** Omnichannel retail (e.g., Walmart, Target) requires such integration.

**Key concepts tested:** Interoperability tactics, service layer, message queues, data transformation.

---

### Q8. A fintech startup wants to build a payment processing system. The system must be secure, highly available, and easy to modify for regulatory changes. How would you prioritize these requirements using a utility tree?

**Answer:**

**Analysis:** The utility tree helps prioritize quality attributes based on business value and architectural impact.

**Utility Tree Construction:**

| Quality Attribute | Attribute Refinement | ASR (Scenario) | Priority (Business Value, Technical Risk) |
|-------------------|----------------------|----------------|-------------------------------------------|
| **Security** | Confidentiality | Payment data must be encrypted end-to-end | (H, H) |
| | Integrity | Transactions must not be tampered | (H, H) |
| | Authentication | Users must be authenticated via 2FA | (H, M) |
| **Availability** | No Downtime | System must be available 99.99% | (H, H) |
| | Recovery | Failover must occur within 30 seconds | (H, M) |
| **Modifiability** | Regulatory Changes | New tax rules can be added in 1 day | (M, H) |
| | Payment Methods | New payment gateway in 2 weeks | (M, M) |
| **Performance** | Response Time | Transaction completes in < 2 seconds | (H, M) |

**Prioritization:**
- **High Priority (H, H):** Security (confidentiality, integrity), Availability (no downtime)
- **Medium Priority:** Modifiability (regulatory changes), Performance

**Architectural Implications:**
- Security: Encryption, authentication, authorization, audit
- Availability: Active redundancy, heartbeat, failover
- Modifiability: Layered architecture, rules engine, façade
- Performance: Caching, concurrency

**Trade-Offs:**
- Security vs. Performance: Encryption adds latency
- Availability vs. Cost: Redundancy increases cost

**Recommendation:**
Focus first on security and availability (high business value, high technical risk), then modifiability, then performance. Use the utility tree to drive design decisions.

**Industry Relevance:** Fintech companies like Stripe, Square use similar prioritization.

**Key concepts tested:** Utility tree, ASR prioritization, quality attributes.

---

### Q9. A media streaming service wants to ensure smooth playback for users with varying network speeds. What architectural tactics would you use to achieve this?

**Answer:**

**Analysis:** The challenge is **performance** and **usability** under variable network conditions.

**Relevant Quality Attributes:** Performance, Usability, Availability.

**Architectural Reasoning:**
1. **Performance tactics:**
   - **Manage sampling rate:** Adapt video quality based on bandwidth
   - **Maintain multiple copies of data:** Store video at different resolutions
   - **Bound queue sizes:** Buffer video to handle network jitter
2. **Usability tactics:**
   - **Maintain user model:** Remember user preferences
   - **Maintain task model:** Predict what user wants to watch
   - **Cancel/Undo:** Allow users to change quality
3. **Architecture:**
   - Use **client-server** pattern with CDN (Content Delivery Network)
   - Implement **adaptive bitrate streaming** (e.g., HLS, DASH)
   - Use **caching** at edge servers

**Trade-Offs:**
- Higher quality requires more bandwidth; lower quality degrades experience.
- Caching reduces latency but increases storage cost.

**Recommendation:**
Implement **adaptive bitrate streaming** with:
- Multiple video quality levels stored on CDN
- Client-side logic to switch based on bandwidth
- Buffer management to handle network fluctuations
- User controls to override quality

**Industry Relevance:** Netflix, YouTube, Amazon Prime use adaptive streaming.

**Key concepts tested:** Performance tactics, usability tactics, CDN, adaptive streaming.

---

### Q10. A government tax filing system experiences massive load spikes during the last week of the filing deadline. How would you design the architecture to handle this?

**Answer:**

**Analysis:** The system needs **scalability** and **availability** during peak loads.

**Relevant Quality Attributes:** Scalability, Availability, Performance.

**Architectural Reasoning:**
1. **Scalability tactics:**
   - **Horizontal scalability:** Add more servers during peak
   - **Vertical scalability:** Increase database resources
   - **Load balancing:** Distribute requests across servers
2. **Availability tactics:**
   - **Active redundancy:** Multiple instances
   - **Rollback:** Recover from failed submissions
   - **Degradation:** Disable non-critical features during peak
3. **Architecture:**
   - **Layered architecture** with separate presentation, business, and data layers
   - **Queue-based processing** for submissions
   - **Caching** for frequently accessed data (e.g., tax rates)
   - **Database connection pooling**

**Trade-Offs:**
- Auto-scaling increases cost during peak but saves during off-peak.
- Queue-based processing improves reliability but adds latency.

**Recommendation:**
Use a **cloud-based architecture** with:
- **Auto-scaling** for web and application servers
- **Load balancer** to distribute traffic
- **Message queue** for asynchronous processing
- **Database read replicas** for scaling reads
- **Graceful degradation** (e.g., disable chat support during peak)

**Industry Relevance:** Government systems (e.g., IRS, India's Income Tax portal) face such spikes.

**Key concepts tested:** Scalability tactics, load balancing, queue-based processing, degradation.

---

## Section 2 — Medium-High Difficulty (Questions 11–35)

---

### Q11. A banking application frequently becomes unavailable during month-end processing. The business requires 99.99% availability, but the proposed architecture increases infrastructure cost significantly. As the architect, how would you analyse the architectural drivers, identify the relevant quality attribute scenarios, and evaluate the trade-off?

**Answer:**

**Analysis:**
- **Business Goal:** Ensure month-end processing completes without downtime.
- **Architectural Driver:** Availability (99.99%) during peak load.
- **Quality Attribute Scenario:**
  - **Source:** Internal batch process
  - **Stimulus:** Month-end processing request
  - **Artifact:** Transaction processing system
  - **Environment:** Peak load
  - **Response:** System remains available
  - **Response Measure:** 99.99% uptime, no transaction failures
- **Trade-Off:** Availability vs. Cost.

**Architectural Reasoning:**
1. **Identify critical components:** Transaction engine, database, batch processor.
2. **Apply availability tactics:**
   - **Active redundancy** for transaction engine
   - **Passive redundancy** for batch processor
   - **Heartbeat** monitoring
   - **Rollback** for failed transactions
3. **Cost optimization:**
   - Use **cloud auto-scaling** to add resources only during month-end
   - Implement **degradation** for non-critical features (e.g., reporting)
   - Use **caching** to reduce database load

**Trade-Offs:**
- Full redundancy 24/7 is costly; auto-scaling reduces cost but requires cloud infrastructure.
- Degradation affects user experience but maintains core availability.

**Recommendation:**
Adopt a **hybrid approach:**
- Core transaction engine: active redundancy (always on)
- Batch processor: passive redundancy with auto-scaling during month-end
- Non-critical features: degradation during peak
- Use **cloud infrastructure** for cost-effective scaling

**Industry Relevance:** Banks like HSBC, Citibank face month-end processing spikes.

**Key concepts tested:** Availability tactics, quality attribute scenarios, trade-off analysis, cost optimization.

---

### Q12. A healthcare provider wants to build a telemedicine platform. The system must support video consultations, prescription management, and integration with electronic health records (EHR). Identify the ASRs and propose an architecture.

**Answer:**

**Analysis:**
- **Business Goals:** Provide remote healthcare, integrate with EHR, ensure patient data security.
- **ASRs:**
  1. **Security:** HIPAA compliance, encryption of patient data
  2. **Performance:** Video latency < 200ms
  3. **Availability:** 99.9% uptime for consultations
  4. **Interoperability:** Integrate with multiple EHR systems (HL7, FHIR)
  5. **Usability:** Easy for doctors and patients

**Architectural Reasoning:**
1. **Layered Architecture:**
   - **Presentation Layer:** Web/mobile UI for doctors and patients
   - **Business Layer:** Consultation management, prescription workflow
   - **Service Layer:** APIs for EHR integration
   - **Data Layer:** Encrypted database for patient records
2. **Quality Attribute Tactics:**
   - **Security:** Encryption, authentication, authorization, audit
   - **Performance:** WebRTC for video, CDN for static content
   - **Availability:** Active redundancy for consultation servers
   - **Interoperability:** Adapters for HL7/FHIR, service layer APIs
3. **Architecture Patterns:**
   - **Client-server** for video
   - **Publish-subscribe** for notifications
   - **Repository** for EHR data

**Trade-Offs:**
- Security vs. Performance: Encryption adds latency; use hardware acceleration.
- Interoperability vs. Modifiability: Supporting multiple EHR formats increases complexity.

**Recommendation:**
Adopt a **service-oriented layered architecture** with:
- WebRTC for video
- REST APIs for EHR integration
- Encrypted database (AES-256)
- Active redundancy for critical components
- Audit logging for compliance

**Industry Relevance:** Telemedicine platforms like Teladoc, Amwell.

**Key concepts tested:** ASRs, layered architecture, security, interoperability, quality attributes.

---

### Q13. A large e-commerce company wants to migrate from a monolithic architecture to a more modular architecture. The current system is difficult to modify, and changes in one module break other modules. How would you apply Attribute-Driven Design (ADD) to guide this migration?

**Answer:**

**Analysis:**
- **Goal:** Improve modifiability and reduce coupling.
- **ADD Steps:**
  1. **Choose module to decompose:** Identify the monolithic system.
  2. **Refine module based on ASRs:** Identify quality attributes (modifiability, scalability).
  3. **Choose architectural tactics:** Encapsulation, separation of concerns.
  4. **Instantiate modules and assign responsibilities:** Create layers/services.
  5. **Define interfaces:** APIs between modules.
  6. **Verify and refine:** Check against ASRs.

**Architectural Reasoning:**
1. **Identify ASRs:**
   - Modifiability: Changes should be localized
   - Scalability: Support independent scaling
   - Performance: Maintain response time
2. **Apply tactics:**
   - **Increase cohesion:** Group related functionality
   - **Reduce coupling:** Use interfaces, intermediaries
   - **Encapsulate:** Hide internal details
3. **Design modules:**
   - **Presentation Layer:** UI components
   - **Business Layer:** Order management, inventory, pricing
   - **Service Layer:** APIs for external systems
   - **Data Layer:** Database access
4. **Define interfaces:**
   - REST APIs between layers
   - Message queues for asynchronous communication

**Trade-Offs:**
- Modularity improves modifiability but adds latency.
- Migration is risky; need incremental approach.

**Recommendation:**
Use **ADD** to incrementally decompose the monolith:
- Start with the most changed module (e.g., order management)
- Extract it as a separate service
- Define clear APIs
- Repeat for other modules
- Use **strangler pattern** to gradually replace monolith

**Industry Relevance:** Companies like Amazon, Netflix migrated from monoliths to microservices.

**Key concepts tested:** ADD, modifiability tactics, layered architecture, migration.

---

### Q14. A telecom company wants to build a system that processes call detail records (CDRs) in real-time. The system must handle millions of records per hour and provide near-real-time analytics. What architecture would you recommend?

**Answer:**

**Analysis:**
- **Requirements:** High throughput, low latency, real-time analytics.
- **Quality Attributes:** Performance, Scalability, Availability.

**Architectural Reasoning:**
1. **Architecture Style:** **Pipe-and-filter** or **event-driven** architecture.
2. **Components:**
   - **Ingestion:** Message queue (Kafka) for CDRs
   - **Processing:** Stream processing (e.g., Apache Storm, Flink)
   - **Storage:** NoSQL database (e.g., Cassandra) for scalability
   - **Analytics:** In-memory database (e.g., Redis) for real-time queries
3. **Tactics:**
   - **Performance:** Concurrency, load balancing, caching
   - **Scalability:** Horizontal scaling, sharding
   - **Availability:** Active redundancy, heartbeat
4. **Layered Architecture:**
   - **Presentation Layer:** Dashboards
   - **Business Layer:** Analytics engine
   - **Service Layer:** APIs for external systems
   - **Data Layer:** NoSQL + in-memory DB

**Trade-Offs:**
- Real-time processing vs. batch: Real-time is complex but meets business need.
- NoSQL vs. RDBMS: NoSQL scales better but lacks ACID.

**Recommendation:**
Use an **event-driven architecture** with:
- Kafka for ingestion
- Stream processing for real-time analytics
- Cassandra for storage
- Redis for caching
- Active redundancy for critical components

**Industry Relevance:** Telecom companies like Verizon, AT&T process billions of CDRs daily.

**Key concepts tested:** Event-driven architecture, performance, scalability, real-time processing.

---

### Q15. A retail company has a legacy inventory system that is tightly coupled with its order management system. Changes to inventory require changes to order management. How would you reduce coupling using architectural tactics?

**Answer:**

**Analysis:**
- **Problem:** Tight coupling between inventory and order management.
- **Goal:** Reduce coupling, improve modifiability.

**Architectural Reasoning:**
1. **Tactics to reduce coupling:**
   - **Encapsulate:** Create a separate Inventory Service with a well-defined API
   - **Use an intermediary:** Introduce a message queue between systems
   - **Restrict dependencies:** Define allowed interactions
   - **Abstract common services:** Create a shared data model
2. **Architecture:**
   - **Service Layer:** Inventory Service, Order Service
   - **Communication:** Asynchronous messaging (publish-subscribe)
   - **Data Layer:** Separate databases for each service
3. **Design:**
   - Order Service publishes order events
   - Inventory Service subscribes to order events
   - Inventory Service publishes inventory updates
   - Order Service subscribes to inventory updates

**Trade-Offs:**
- Asynchronous communication improves decoupling but adds complexity.
- Separate databases improve autonomy but require data synchronization.

**Recommendation:**
Implement a **publish-subscribe** architecture with:
- Inventory Service and Order Service as independent components
- Message queue (e.g., RabbitMQ, Kafka) for communication
- Event-driven updates
- Clear API contracts

**Industry Relevance:** Retailers like Walmart, Target use such decoupling for agility.

**Key concepts tested:** Coupling reduction, publish-subscribe, service layer, modifiability.

---

### Q16. A financial institution wants to evaluate its proposed architecture for a trading platform. The system must handle high-frequency trades, ensure security, and be highly available. How would you apply ATAM to evaluate this architecture?

**Answer:**

**Analysis:**
- **Goal:** Evaluate architecture using ATAM.
- **Quality Attributes:** Performance, Security, Availability.

**ATAM Steps:**
1. **Present ATAM:** Explain process to stakeholders.
2. **Present Business Drivers:** Trading volume, regulatory requirements.
3. **Present Architecture:** Describe proposed architecture.
4. **Identify Architectural Approaches:** Patterns, tactics used.
5. **Generate Utility Tree:**
   - **Performance:** Trade execution < 10ms (H, H)
   - **Security:** Encrypted transactions (H, H)
   - **Availability:** 99.999% uptime (H, H)
6. **Analyze Architectural Approaches:**
   - **Risk:** Single point of failure in trade engine
   - **Sensitivity Point:** Number of concurrent traders affects latency
   - **Trade-Off:** Encryption adds latency but ensures security
7. **Brainstorm and Prioritize Scenarios:** Stakeholders add scenarios.
8. **Analyze Architectural Approaches (Round 2):** Evaluate new scenarios.
9. **Present Results:** Risks, non-risks, sensitivity points, trade-offs.

**Outputs:**
- Risks: Single point of failure, encryption latency
- Non-risks: Redundant network paths
- Sensitivity Points: Concurrency, encryption
- Trade-Offs: Security vs. Performance

**Recommendation:**
Address risks by adding redundancy to trade engine, using hardware acceleration for encryption. Document trade-offs and monitor.

**Industry Relevance:** Stock exchanges, investment banks use ATAM for critical systems.

**Key concepts tested:** ATAM, utility tree, risks, sensitivity points, trade-offs.

---

### Q17. A logistics company wants to build a system that tracks shipments in real-time. The system must integrate with multiple carriers (FedEx, UPS, DHL) and provide a unified tracking interface. What architecture would you propose?

**Answer:**

**Analysis:**
- **Requirements:** Real-time tracking, multi-carrier integration, unified interface.
- **Quality Attributes:** Interoperability, Performance, Availability.

**Architectural Reasoning:**
1. **Architecture Style:** **Service-oriented** with **adapter pattern**.
2. **Components:**
   - **Carrier Adapters:** One for each carrier (FedEx, UPS, DHL)
   - **Tracking Service:** Aggregates data from adapters
   - **Notification Service:** Sends updates to users
   - **API Gateway:** Unified interface for clients
3. **Tactics:**
   - **Interoperability:** Adapters, data transformation
   - **Performance:** Caching, asynchronous updates
   - **Availability:** Redundancy, failover
4. **Layered Architecture:**
   - **Presentation Layer:** Web/mobile UI
   - **Service Layer:** Tracking API
   - **Business Layer:** Tracking logic
   - **Data Layer:** Database for shipment data

**Trade-Offs:**
- Real-time vs. batch: Real-time improves UX but requires more infrastructure.
- Adapter complexity: Each carrier has different APIs; adapters add maintenance.

**Recommendation:**
Use a **service-oriented architecture** with:
- Adapter for each carrier
- Unified Tracking API
- Message queue for asynchronous updates
- Caching for frequently accessed data

**Industry Relevance:** Logistics platforms like ShipStation, Shippo.

**Key concepts tested:** Interoperability, adapter pattern, service-oriented architecture.

---

### Q18. A hospital wants to build a system that predicts patient readmission risks using machine learning. The system must integrate with existing EHR and provide predictions in real-time. What architectural challenges do you foresee?

**Answer:**

**Analysis:**
- **Requirements:** Real-time predictions, EHR integration, accuracy.
- **Quality Attributes:** Performance, Interoperability, Modifiability, Security.

**Architectural Challenges:**
1. **Data Integration:** EHR systems use different formats (HL7, FHIR).
2. **Real-Time Processing:** ML models need to process data quickly.
3. **Model Updates:** ML models need to be updated without downtime.
4. **Security:** Patient data must be protected.
5. **Scalability:** Handle multiple patients simultaneously.

**Architectural Reasoning:**
1. **Layered Architecture:**
   - **Presentation Layer:** Clinician dashboard
   - **Business Layer:** Prediction engine
   - **Service Layer:** EHR integration APIs
   - **Data Layer:** Data warehouse for training, real-time DB for predictions
2. **Tactics:**
   - **Interoperability:** Adapters for HL7/FHIR
   - **Performance:** Caching, concurrency
   - **Modifiability:** Separate ML model from prediction engine (defer binding)
   - **Security:** Encryption, audit

**Trade-Offs:**
- Real-time predictions vs. batch: Real-time is complex but more useful.
- Model accuracy vs. speed: More complex models are slower.

**Recommendation:**
Use a **layered architecture** with:
- Adapters for EHR integration
- Separate ML model service (can be updated independently)
- Real-time prediction API
- Caching for patient data
- Encryption for security

**Industry Relevance:** Healthcare systems like Epic, Cerner are adding ML capabilities.

**Key concepts tested:** Interoperability, performance, modifiability, security, ML integration.

---

### Q19. A company is developing a mobile banking app. The app must work offline and sync when online. How would you design the architecture to handle offline mode?

**Answer:**

**Analysis:**
- **Requirements:** Offline functionality, data synchronization.
- **Quality Attributes:** Availability, Usability, Performance.

**Architectural Reasoning:**
1. **Architecture Style:** **Client-server** with **local storage** on client.
2. **Components:**
   - **Client:** Mobile app with local database (SQLite)
   - **Sync Service:** Handles data synchronization
   - **Server:** Banking backend
3. **Tactics:**
   - **Availability:** Local caching, offline mode
   - **Usability:** Seamless transition between online/offline
   - **Performance:** Local data access
4. **Design:**
   - Client stores data locally
   - Sync Service detects connectivity
   - Conflict resolution for concurrent updates
   - Queue for offline transactions

**Trade-Offs:**
- Offline mode improves usability but introduces data consistency challenges.
- Local storage increases security risk (data on device).

**Recommendation:**
Implement:
- Local database (SQLite) with encryption
- Sync Service with conflict resolution (e.g., last-write-wins)
- Queue for offline transactions
- Background sync when online
- Secure authentication (biometrics)

**Industry Relevance:** Banking apps like Chase, Bank of America support offline mode.

**Key concepts tested:** Availability, usability, synchronization, security.

---

### Q20. A media company wants to build a content management system (CMS) that supports multiple channels (web, mobile, social media). How would you design the architecture for multi-channel publishing?

**Answer:**

**Analysis:**
- **Requirements:** Multi-channel publishing, content reuse.
- **Quality Attributes:** Modifiability, Scalability, Performance.

**Architectural Reasoning:**
1. **Architecture Style:** **Layered** with **content repository**.
2. **Components:**
   - **Content Repository:** Stores content in structured format (XML/JSON)
   - **Transformation Service:** Converts content for each channel
   - **Publishing Service:** Publishes to channels
   - **Channel Adapters:** APIs for each channel
3. **Tactics:**
   - **Modifiability:** Separation of content from presentation
   - **Scalability:** Horizontal scaling for publishing service
   - **Performance:** Caching, CDN
4. **Layered Architecture:**
   - **Presentation Layer:** Channel-specific UIs
   - **Business Layer:** Content management, workflow
   - **Service Layer:** Publishing APIs
   - **Data Layer:** Content repository

**Trade-Offs:**
- Content reuse vs. channel-specific optimization.
- Real-time publishing vs. batch.

**Recommendation:**
Use a **layered architecture** with:
- Central content repository
- Transformation service for each channel
- Publishing API
- CDN for content delivery
- Caching for performance

**Industry Relevance:** CMS platforms like WordPress, Drupal, Adobe Experience Manager.

**Key concepts tested:** Layered architecture, modifiability, multi-channel publishing.

---

### Q21. An airline wants to build a reservation system that handles millions of bookings. The system must be highly available and scalable. How would you apply the 4+1 view model to document the architecture?

**Answer:**

**Analysis:**
- **Goal:** Document architecture using 4+1 views.
- **Quality Attributes:** Availability, Scalability, Performance.

**4+1 Views:**

1. **Logical View:**
   - **Elements:** Booking, Payment, Inventory, Notification
   - **Relations:** Booking uses Payment, Inventory
   - **Diagram:** Class diagram, package diagram
   - **Stakeholders:** End users, developers

2. **Process View:**
   - **Elements:** Booking process, Payment process, Notification process
   - **Relations:** Concurrency, synchronization
   - **Diagram:** Sequence diagram, activity diagram
   - **Stakeholders:** Integrators, performance engineers

3. **Development View:**
   - **Elements:** Modules, libraries, subsystems
   - **Relations:** Dependencies
   - **Diagram:** Package diagram, layer diagram
   - **Stakeholders:** Developers, project managers

4. **Physical View:**
   - **Elements:** Servers, databases, load balancers
   - **Relations:** Communication paths
   - **Diagram:** Deployment diagram
   - **Stakeholders:** System engineers, operations

5. **Scenarios (Use Cases):**
   - **Scenario:** Book a flight
   - **Diagram:** Sequence diagram
   - **Stakeholders:** All

**Trade-Offs:**
- Documentation overhead vs. clarity.
- Different views for different stakeholders.

**Recommendation:**
Document using 4+1 views:
- Logical view for functionality
- Process view for concurrency
- Development view for code organization
- Physical view for deployment
- Scenarios for validation

**Industry Relevance:** Airlines like Delta, United use similar documentation.

**Key concepts tested:** 4+1 view model, documentation, stakeholders.

---

### Q22. A company has a monolithic application that is difficult to test. How would you improve testability through architectural changes?

**Answer:**

**Analysis:**
- **Problem:** Monolithic architecture is hard to test.
- **Goal:** Improve testability.

**Architectural Reasoning:**
1. **Testability Tactics:**
   - **Control and Observe System State:**
     - Specialized interfaces for testing
     - Record/Playback
     - Localize state storage
     - Abstract data sources
     - Sandbox
     - Executable assertions
   - **Limit Complexity:**
     - Limit structural complexity (reduce cyclic dependencies)
     - Limit non-determinism

2. **Architecture Changes:**
   - Decompose monolith into **layers** (presentation, business, data)
   - Introduce **interfaces** for each layer
   - Use **dependency injection** for testability
   - Separate **business logic** from infrastructure

3. **Design:**
   - **Presentation Layer:** UI components (testable with UI tests)
   - **Business Layer:** Business logic (testable with unit tests)
   - **Data Layer:** Database access (testable with integration tests)
   - **Service Layer:** APIs (testable with API tests)

**Trade-Offs:**
- Decomposition improves testability but adds complexity.
- More interfaces mean more test cases.

**Recommendation:**
Decompose into layers, introduce interfaces, use dependency injection, and create test harnesses. Use **executable assertions** for critical components.

**Industry Relevance:** Legacy systems often need such refactoring for testability.

**Key concepts tested:** Testability tactics, layered architecture, decomposition.

---

### Q23. A company wants to build a system that supports both batch and real-time processing of financial transactions. How would you design the architecture to handle both?

**Answer:**

**Analysis:**
- **Requirements:** Batch (end-of-day) and real-time (immediate) processing.
- **Quality Attributes:** Performance, Availability, Modifiability.

**Architectural Reasoning:**
1. **Architecture Style:** **Lambda architecture** (batch + speed layers).
2. **Components:**
   - **Batch Layer:** Processes historical data (Hadoop, Spark)
   - **Speed Layer:** Processes real-time data (Storm, Flink)
   - **Serving Layer:** Merges results (Cassandra, HBase)
3. **Tactics:**
   - **Performance:** Concurrency, caching
   - **Availability:** Redundancy, failover
   - **Modifiability:** Separate batch and real-time logic
4. **Layered Architecture:**
   - **Presentation Layer:** Dashboards
   - **Business Layer:** Analytics
   - **Service Layer:** APIs
   - **Data Layer:** Batch storage + real-time storage

**Trade-Offs:**
- Complexity vs. flexibility: Lambda architecture is complex but handles both.
- Cost: Maintaining two processing paths is expensive.

**Recommendation:**
Use **Lambda architecture** with:
- Batch layer for historical analysis
- Speed layer for real-time
- Serving layer for unified queries
- Redundancy for availability

**Industry Relevance:** Financial institutions use Lambda architecture for risk analysis.

**Key concepts tested:** Lambda architecture, batch vs. real-time, performance.

---

### Q24. A retail company wants to implement a recommendation engine that suggests products to users. How would you design the architecture to ensure scalability and performance?

**Answer:**

**Analysis:**
- **Requirements:** Real-time recommendations, scalability.
- **Quality Attributes:** Performance, Scalability, Availability.

**Architectural Reasoning:**
1. **Architecture Style:** **Microservices** or **service-oriented**.
2. **Components:**
   - **User Profile Service:** Stores user preferences
   - **Product Catalog Service:** Stores product data
   - **Recommendation Engine:** ML model for recommendations
   - **API Gateway:** Unified interface
3. **Tactics:**
   - **Performance:** Caching, pre-computation
   - **Scalability:** Horizontal scaling, sharding
   - **Availability:** Redundancy
4. **Layered Architecture:**
   - **Presentation Layer:** Web/mobile UI
   - **Service Layer:** Recommendation API
   - **Business Layer:** ML model
   - **Data Layer:** NoSQL for user data, RDBMS for products

**Trade-Offs:**
- Real-time vs. pre-computed: Real-time is accurate but slower; pre-computed is faster but less accurate.
- Complexity: ML models add complexity.

**Recommendation:**
Use a **service-oriented architecture** with:
- Pre-computed recommendations for common cases
- Real-time recommendations for personalized cases
- Caching for frequently accessed data
- Horizontal scaling for recommendation engine

**Industry Relevance:** Amazon, Netflix use similar architectures.

**Key concepts tested:** Scalability, performance, service-oriented architecture.

---

### Q25. A company has a legacy system that is documented poorly. The documentation is outdated and does not match the implementation. How would you reconstruct the architecture?

**Answer:**

**Analysis:**
- **Problem:** Outdated documentation, mismatch with implementation.
- **Goal:** Reconstruct current architecture.

**Architectural Reconstruction Steps:**
1. **Raw View Extraction:**
   - Extract from source code, execution traces, build scripts
   - Identify classes, files, caller-callee relationships
2. **DB Construction:**
   - Store extracted data in a common format
3. **View Fusion:**
   - Combine static and dynamic views
   - Group elements into layers/subsystems
4. **Architecture Analysis:**
   - Validate correctness
   - Check for violations
5. **Iterate:**
   - Refine as needed

**Tools:** ARMIN, Dali, Lattix, SonarJ.

**Trade-Offs:**
- Automated tools vs. manual review: Tools are fast but may miss context.
- Time vs. accuracy: Reconstruction takes time.

**Recommendation:**
Use **ARMIN** for extraction, expert workshops for fusion, and **SonarJ** for violation detection. Document the reconstructed architecture using **module and C&C views**.

**Industry Relevance:** Legacy modernization projects.

**Key concepts tested:** Architecture reconstruction, tools, view fusion.

---

### Q26. A company wants to build a system that integrates with multiple payment gateways (PayPal, Stripe, Square). How would you design the architecture for flexibility?

**Answer:**

**Analysis:**
- **Requirements:** Support multiple payment gateways, easy to add new ones.
- **Quality Attributes:** Modifiability, Interoperability, Security.

**Architectural Reasoning:**
1. **Architecture Style:** **Adapter pattern** + **Service layer**.
2. **Components:**
   - **Payment Service:** Unified interface
   - **Adapters:** One for each gateway
   - **Payment Gateway APIs:** External
3. **Tactics:**
   - **Modifiability:** Encapsulate gateway-specific logic in adapters
   - **Interoperability:** Standard data format (JSON)
   - **Security:** Encryption, tokenization
4. **Layered Architecture:**
   - **Presentation Layer:** Checkout UI
   - **Service Layer:** Payment API
   - **Business Layer:** Payment logic
   - **Data Layer:** Transaction DB

**Trade-Offs:**
- Abstraction vs. gateway-specific features.
- Complexity of maintaining multiple adapters.

**Recommendation:**
Use **adapter pattern** with:
- Unified Payment API
- Adapter for each gateway
- Configuration-driven gateway selection
- Secure tokenization

**Industry Relevance:** E-commerce platforms like Shopify.

**Key concepts tested:** Adapter pattern, modifiability, interoperability, security.

---

### Q27. A company wants to build a system that handles real-time bidding for online ads. The system must respond within 100ms. What architecture would you recommend?

**Answer:**

**Analysis:**
- **Requirements:** Real-time bidding, <100ms response.
- **Quality Attributes:** Performance, Scalability, Availability.

**Architectural Reasoning:**
1. **Architecture Style:** **Event-driven** with **in-memory processing**.
2. **Components:**
   - **Bid Request Handler:** Receives requests
   - **Bid Engine:** Calculates bids
   - **User Profile Service:** Provides user data
   - **Ad Inventory Service:** Provides ad data
3. **Tactics:**
   - **Performance:** In-memory data, caching, concurrency
   - **Scalability:** Horizontal scaling, sharding
   - **Availability:** Redundancy
4. **Design:**
   - Use **Redis** for in-memory data
   - Use **Kafka** for event streaming
   - Use **load balancer** for distribution

**Trade-Offs:**
- In-memory vs. disk: In-memory is faster but more expensive.
- Accuracy vs. speed: Complex bidding algorithms are slower.

**Recommendation:**
Use an **event-driven architecture** with:
- In-memory data stores (Redis)
- Stream processing (Kafka)
- Horizontal scaling
- Redundancy

**Industry Relevance:** Google Ads, Facebook Ads use real-time bidding.

**Key concepts tested:** Real-time performance, event-driven architecture, in-memory processing.

---

### Q28. A company wants to build a system that supports A/B testing for its website. How would you design the architecture to support experimentation?

**Answer:**

**Analysis:**
- **Requirements:** A/B testing, easy to add experiments.
- **Quality Attributes:** Modifiability, Performance, Usability.

**Architectural Reasoning:**
1. **Architecture Style:** **Service-oriented** with **feature flags**.
2. **Components:**
   - **Experiment Service:** Manages experiments
   - **Feature Flag Service:** Controls feature rollout
   - **Analytics Service:** Tracks results
3. **Tactics:**
   - **Modifiability:** Feature flags for toggling features
   - **Performance:** Caching for flag evaluation
   - **Usability:** Easy to configure experiments
4. **Layered Architecture:**
   - **Presentation Layer:** Website UI
   - **Service Layer:** Experiment API
   - **Business Layer:** Experiment logic
   - **Data Layer:** Experiment DB

**Trade-Offs:**
- Feature flags add complexity but enable experimentation.
- Real-time vs. batch analytics.

**Recommendation:**
Use **feature flags** with:
- Experiment Service
- Feature Flag Service
- Analytics Service
- Caching for performance

**Industry Relevance:** Netflix, Amazon use A/B testing extensively.

**Key concepts tested:** Modifiability, feature flags, experimentation.

---

### Q29. A company wants to build a system that handles millions of IoT device messages. The system must be scalable and reliable. What architecture would you recommend?

**Answer:**

**Analysis:**
- **Requirements:** High volume, scalability, reliability.
- **Quality Attributes:** Scalability, Availability, Performance.

**Architectural Reasoning:**
1. **Architecture Style:** **Event-driven** with **message queue**.
2. **Components:**
   - **Ingestion Service:** Receives device messages
   - **Message Queue:** Kafka, RabbitMQ
   - **Processing Service:** Processes messages
   - **Storage Service:** Stores data (NoSQL)
3. **Tactics:**
   - **Scalability:** Horizontal scaling, sharding
   - **Availability:** Redundancy, failover
   - **Performance:** Concurrency, batching
4. **Layered Architecture:**
   - **Presentation Layer:** Dashboards
   - **Service Layer:** Device API
   - **Business Layer:** Processing logic
   - **Data Layer:** NoSQL DB

**Trade-Offs:**
- Real-time vs. batch: Real-time is complex but useful.
- Cost: High volume requires significant infrastructure.

**Recommendation:**
Use an **event-driven architecture** with:
- Kafka for ingestion
- Stream processing for real-time
- NoSQL for storage
- Redundancy for availability

**Industry Relevance:** IoT platforms like AWS IoT, Azure IoT.

**Key concepts tested:** Scalability, availability, event-driven architecture, IoT.

---

### Q30. A company wants to build a system that provides personalized content to users. How would you design the architecture to handle personalization?

**Answer:**

**Analysis:**
- **Requirements:** Personalized content, real-time updates.
- **Quality Attributes:** Performance, Scalability, Modifiability.

**Architectural Reasoning:**
1. **Architecture Style:** **Service-oriented** with **user profile**.
2. **Components:**
   - **User Profile Service:** Stores preferences
   - **Content Service:** Manages content
   - **Personalization Engine:** Generates personalized content
   - **API Gateway:** Unified interface
3. **Tactics:**
   - **Performance:** Caching, pre-computation
   - **Scalability:** Horizontal scaling
   - **Modifiability:** Separate personalization logic
4. **Layered Architecture:**
   - **Presentation Layer:** UI
   - **Service Layer:** Personalization API
   - **Business Layer:** Personalization logic
   - **Data Layer:** User profile DB

**Trade-Offs:**
- Real-time vs. pre-computed personalization.
- Privacy vs. personalization.

**Recommendation:**
Use a **service-oriented architecture** with:
- User Profile Service
- Personalization Engine
- Caching for performance
- Privacy controls

**Industry Relevance:** Netflix, Spotify use personalization.

**Key concepts tested:** Personalization, service-oriented architecture, performance.

---

### Q31. A company wants to build a system that supports multi-factor authentication (MFA). How would you design the architecture to ensure security and usability?

**Answer:**

**Analysis:**
- **Requirements:** MFA, security, usability.
- **Quality Attributes:** Security, Usability, Performance.

**Architectural Reasoning:**
1. **Architecture Style:** **Layered** with **authentication service**.
2. **Components:**
   - **Authentication Service:** Handles MFA
   - **User Service:** Stores user data
   - **Notification Service:** Sends OTP
3. **Tactics:**
   - **Security:** Encryption, tokenization, audit
   - **Usability:** Remember device, biometrics
   - **Performance:** Caching, async notifications
4. **Layered Architecture:**
   - **Presentation Layer:** Login UI
   - **Service Layer:** Auth API
   - **Business Layer:** Auth logic
   - **Data Layer:** User DB

**Trade-Offs:**
- Security vs. Usability: MFA improves security but adds friction.
- Cost: SMS OTP costs money.

**Recommendation:**
Use **adaptive MFA**:
- Risk-based authentication
- Remember device for low-risk
- Biometrics for high-risk
- Fallback to OTP

**Industry Relevance:** Banks, SaaS platforms use MFA.

**Key concepts tested:** Security, usability, MFA, adaptive authentication.

---

### Q32. A company wants to build a system that processes claims for insurance. The system must handle complex workflows. How would you design the architecture?

**Answer:**

**Analysis:**
- **Requirements:** Complex workflows, modifiability.
- **Quality Attributes:** Modifiability, Performance, Availability.

**Architectural Reasoning:**
1. **Architecture Style:** **Workflow engine** + **layered**.
2. **Components:**
   - **Workflow Engine:** Manages claim process
   - **Rules Engine:** Business rules
   - **Claim Service:** Claim data
   - **Notification Service:** Updates
3. **Tactics:**
   - **Modifiability:** Rules engine for business rules
   - **Performance:** Caching, async processing
   - **Availability:** Redundancy
4. **Layered Architecture:**
   - **Presentation Layer:** Claim UI
   - **Business Layer:** Workflow engine, rules engine
   - **Service Layer:** Claim API
   - **Data Layer:** Claim DB

**Trade-Offs:**
- Flexibility vs. complexity: Workflow engines add complexity but enable modifiability.
- Real-time vs. batch processing.

**Recommendation:**
Use **workflow engine** (e.g., BizTalk, Oracle BPEL) with:
- Rules engine for business rules
- Layered architecture
- Async processing for long-running tasks

**Industry Relevance:** Insurance companies like Allstate, Geico.

**Key concepts tested:** Workflow engine, modifiability, layered architecture.

---

### Q33. A company wants to build a system that supports real-time collaboration (like Google Docs). How would you design the architecture?

**Answer:**

**Analysis:**
- **Requirements:** Real-time collaboration, conflict resolution.
- **Quality Attributes:** Performance, Availability, Usability.

**Architectural Reasoning:**
1. **Architecture Style:** **Event-driven** with **operational transformation**.
2. **Components:**
   - **Collaboration Service:** Manages sessions
   - **Document Service:** Stores documents
   - **Sync Service:** Handles updates
   - **WebSocket Server:** Real-time communication
3. **Tactics:**
   - **Performance:** WebSockets, caching
   - **Availability:** Redundancy
   - **Usability:** Conflict resolution
4. **Layered Architecture:**
   - **Presentation Layer:** Editor UI
   - **Service Layer:** Collaboration API
   - **Business Layer:** Conflict resolution
   - **Data Layer:** Document DB

**Trade-Offs:**
- Real-time vs. consistency: Real-time needs conflict resolution.
- Complexity: Operational transformation is complex.

**Recommendation:**
Use **WebSockets** for real-time, **operational transformation** for conflict resolution, and **redundancy** for availability.

**Industry Relevance:** Google Docs, Microsoft Office 365.

**Key concepts tested:** Real-time collaboration, event-driven architecture, conflict resolution.

---

### Q34. A company wants to build a system that supports regulatory compliance (e.g., GDPR). How would you design the architecture to ensure compliance?

**Answer:**

**Analysis:**
- **Requirements:** GDPR compliance, data privacy.
- **Quality Attributes:** Security, Modifiability, Auditability.

**Architectural Reasoning:**
1. **Architecture Style:** **Layered** with **audit service**.
2. **Components:**
   - **Data Service:** Manages data
   - **Consent Service:** Manages user consent
   - **Audit Service:** Logs access
   - **Deletion Service:** Handles right to be forgotten
3. **Tactics:**
   - **Security:** Encryption, access control
   - **Modifiability:** Configurable retention policies
   - **Auditability:** Logging, monitoring
4. **Layered Architecture:**
   - **Presentation Layer:** User portal
   - **Service Layer:** Compliance API
   - **Business Layer:** Compliance logic
   - **Data Layer:** Encrypted DB

**Trade-Offs:**
- Compliance vs. Performance: Audit logging adds overhead.
- Data retention vs. deletion.

**Recommendation:**
Use **audit service**, **consent service**, and **deletion service** with:
- Encryption for data at rest and in transit
- Access control
- Audit logging
- Configurable retention

**Industry Relevance:** All companies handling EU citizen data.

**Key concepts tested:** Security, modifiability, auditability, GDPR.

---

### Q35. A company wants to build a system that handles high-volume financial transactions with ACID compliance. What architecture would you recommend?

**Answer:**

**Analysis:**
- **Requirements:** ACID compliance, high volume.
- **Quality Attributes:** Performance, Availability, Consistency.

**Architectural Reasoning:**
1. **Architecture Style:** **Layered** with **transaction management**.
2. **Components:**
   - **Transaction Service:** Manages transactions
   - **Account Service:** Manages accounts
   - **Ledger Service:** Records transactions
   - **Notification Service:** Alerts
3. **Tactics:**
   - **Performance:** Connection pooling, caching
   - **Availability:** Redundancy, failover
   - **Consistency:** Transactions, locking
4. **Layered Architecture:**
   - **Presentation Layer:** UI
   - **Service Layer:** Transaction API
   - **Business Layer:** Transaction logic
   - **Data Layer:** RDBMS with ACID

**Trade-Offs:**
- ACID vs. Scalability: ACID limits horizontal scaling.
- Consistency vs. Availability (CAP theorem).

**Recommendation:**
Use **RDBMS** with ACID, **connection pooling**, **redundancy**, and **transaction management**. Consider **sharding** for scalability.

**Industry Relevance:** Banks, payment processors.

**Key concepts tested:** ACID, performance, availability, consistency.

---

## Section 3 — High Difficulty (Questions 36–50)

---

### Q36. A global bank wants to build a real-time fraud detection system. The system must analyze transactions in <100ms, integrate with multiple payment systems, and adapt to new fraud patterns. Identify the ASRs, propose an architecture, and evaluate it using ATAM.

**Answer:**

**Analysis:**
- **Business Goals:** Detect fraud in real-time, minimize false positives, adapt to new patterns.
- **ASRs:**
  1. **Performance:** Transaction analysis <100ms
  2. **Interoperability:** Integrate with payment systems (SWIFT, ACH, cards)
  3. **Modifiability:** Add new fraud rules quickly
  4. **Availability:** 99.999% uptime
  5. **Security:** Protect financial data

**Proposed Architecture:**
1. **Event-driven architecture** with **stream processing**.
2. **Components:**
   - **Transaction Ingestion:** Kafka for high-volume ingestion
   - **Fraud Engine:** Stream processing (Flink, Storm)
   - **Rules Engine:** Drools for fraud rules
   - **ML Model Service:** Anomaly detection
   - **Alert Service:** Notifications
3. **Layered Architecture:**
   - **Presentation Layer:** Fraud analyst dashboard
   - **Service Layer:** Fraud detection API
   - **Business Layer:** Fraud engine, rules engine
   - **Data Layer:** NoSQL for transaction history, in-memory for real-time

**ATAM Evaluation:**
1. **Utility Tree:**
   - **Performance:** <100ms analysis (H, H)
   - **Interoperability:** Integrate with 5 payment systems (H, M)
   - **Modifiability:** Add rule in 1 day (M, H)
   - **Availability:** 99.999% (H, H)
2. **Risks:**
   - **Risk:** ML model latency may exceed 100ms
   - **Risk:** Rules engine may become bottleneck
3. **Sensitivity Points:**
   - Number of concurrent transactions affects latency
   - Rule complexity affects performance
4. **Trade-Offs:**
   - **Security vs. Performance:** Encryption adds latency
   - **Accuracy vs. Speed:** Complex models are slower
   - **Modifiability vs. Performance:** More rules slow processing

**Recommendation:**
- Use **in-memory processing** for real-time
- **Parallel processing** for rules
- **Pre-computed models** for ML
- **Hardware acceleration** for encryption

**Industry Relevance:** Banks like JPMorgan, HSBC use similar systems.

**Key concepts tested:** ASRs, ATAM, event-driven architecture, trade-offs.

---

### Q37. A healthcare company wants to build a system that integrates with multiple hospitals' EHR systems. Each hospital uses different standards (HL7, FHIR, proprietary). The system must provide a unified patient view. How would you design the architecture?

**Answer:**

**Analysis:**
- **Requirements:** Integrate heterogeneous EHR systems, unified patient view.
- **Quality Attributes:** Interoperability, Security, Performance.

**Architectural Reasoning:**
1. **Architecture Style:** **Service-oriented** with **adapters**.
2. **Components:**
   - **EHR Adapters:** One for each hospital
   - **Patient Service:** Unified patient data
   - **Consent Service:** Manages patient consent
   - **API Gateway:** Unified interface
3. **Tactics:**
   - **Interoperability:** Adapters, data transformation
   - **Security:** Encryption, consent management
   - **Performance:** Caching, async processing
4. **Layered Architecture:**
   - **Presentation Layer:** Clinician UI
   - **Service Layer:** Patient API
   - **Business Layer:** Patient data aggregation
   - **Data Layer:** Unified patient DB

**Trade-Offs:**
- Real-time vs. batch: Real-time improves care but requires more infrastructure.
- Standardization vs. flexibility: Strict standards improve interoperability but may not fit all.

**Recommendation:**
Use **adapters** for each EHR, **consent service** for privacy, and **caching** for performance. Implement **FHIR** as the canonical standard.

**Industry Relevance:** Health information exchanges (HIEs).

**Key concepts tested:** Interoperability, security, layered architecture, adapters.

---

### Q38. A company wants to build a system that supports both on-premise and cloud deployment. How would you design the architecture for portability?

**Answer:**

**Analysis:**
- **Requirements:** On-premise and cloud deployment, portability.
- **Quality Attributes:** Portability, Modifiability, Performance.

**Architectural Reasoning:**
1. **Architecture Style:** **Layered** with **abstraction**.
2. **Components:**
   - **Deployment Abstraction Layer:** Abstracts infrastructure
   - **Configuration Service:** Environment-specific config
   - **Monitoring Service:** Unified monitoring
3. **Tactics:**
   - **Portability:** Abstraction, containerization
   - **Modifiability:** Configuration-driven
   - **Performance:** Environment-specific optimization
4. **Layered Architecture:**
   - **Presentation Layer:** UI
   - **Service Layer:** APIs
   - **Business Layer:** Business logic
   - **Data Layer:** Abstracted data access

**Trade-Offs:**
- Portability vs. Performance: Abstraction adds overhead.
- Cost: Maintaining two deployment models.

**Recommendation:**
Use **containerization** (Docker, Kubernetes), **abstraction layers**, and **configuration-driven** deployment. Use **Terraform** for infrastructure as code.

**Industry Relevance:** Enterprises with hybrid cloud strategies.

**Key concepts tested:** Portability, modifiability, abstraction, containerization.

---

### Q39. A company wants to build a system that handles real-time stock trading. The system must be highly available, secure, and fast. How would you apply ATAM to evaluate the architecture?

**Answer:**

**Analysis:**
- **Requirements:** High availability, security, low latency.
- **Quality Attributes:** Availability, Security, Performance.

**ATAM Application:**
1. **Business Drivers:** Trading volume, regulatory compliance.
2. **Architecture Presentation:** Event-driven with redundancy.
3. **Utility Tree:**
   - **Performance:** Trade execution <10ms (H, H)
   - **Availability:** 99.999% (H, H)
   - **Security:** Encrypted transactions (H, H)
4. **Risks:**
   - Single point of failure in trade engine
   - Encryption latency
5. **Sensitivity Points:**
   - Concurrency affects latency
   - Encryption strength affects performance
6. **Trade-Offs:**
   - Security vs. Performance
   - Availability vs. Cost

**Recommendation:**
- **Active redundancy** for trade engine
- **Hardware acceleration** for encryption
- **In-memory processing** for speed
- **Regular ATAM evaluations**

**Industry Relevance:** Stock exchanges, investment banks.

**Key concepts tested:** ATAM, availability, security, performance.

---

### Q40. A company wants to build a system that supports multi-tenant SaaS with tenant-specific customizations. How would you design the architecture to balance isolation and cost?

**Answer:**

**Analysis:**
- **Requirements:** Multi-tenancy, customizations, cost efficiency.
- **Quality Attributes:** Security, Modifiability, Scalability.

**Architectural Reasoning:**
1. **Architecture Style:** **Layered** with **tenant isolation**.
2. **Components:**
   - **Tenant Service:** Manages tenants
   - **Configuration Service:** Tenant-specific config
   - **Customization Service:** Tenant-specific logic
3. **Tactics:**
   - **Security:** Tenant isolation, encryption
   - **Modifiability:** Configuration-driven customizations
   - **Scalability:** Shared infrastructure
4. **Isolation Models:**
   - **Shared Database:** Lowest cost, lowest isolation
   - **Separate Schema:** Medium cost, medium isolation
   - **Separate Database:** Highest cost, highest isolation

**Trade-Offs:**
- Isolation vs. Cost: Higher isolation costs more.
- Customization vs. Maintainability: More customizations increase complexity.

**Recommendation:**
Use **hybrid model:**
- Shared database for small tenants
- Separate schema for medium tenants
- Separate database for large/enterprise tenants
- Configuration-driven customizations

**Industry Relevance:** Salesforce, Zoho.

**Key concepts tested:** Multi-tenancy, security, modifiability, scalability.

---

### Q41. A company wants to build a system that handles real-time video processing for surveillance. The system must detect anomalies and alert security. How would you design the architecture?

**Answer:**

**Analysis:**
- **Requirements:** Real-time video processing, anomaly detection, alerts.
- **Quality Attributes:** Performance, Availability, Scalability.

**Architectural Reasoning:**
1. **Architecture Style:** **Event-driven** with **stream processing**.
2. **Components:**
   - **Video Ingestion:** RTSP streams
   - **Video Processing:** GPU-accelerated processing
   - **Anomaly Detection:** ML models
   - **Alert Service:** Notifications
3. **Tactics:**
   - **Performance:** GPU, parallel processing
   - **Availability:** Redundancy
   - **Scalability:** Horizontal scaling
4. **Layered Architecture:**
   - **Presentation Layer:** Security dashboard
   - **Service Layer:** Alert API
   - **Business Layer:** Anomaly detection
   - **Data Layer:** Video storage, metadata

**Trade-Offs:**
- Real-time vs. accuracy: Real-time is faster but less accurate.
- Cost: GPU infrastructure is expensive.

**Recommendation:**
Use **GPU-accelerated processing**, **stream processing**, and **ML models**. Implement **redundancy** for availability.

**Industry Relevance:** Surveillance systems, smart cities.

**Key concepts tested:** Real-time processing, performance, scalability, ML.

---

### Q42. A company wants to build a system that supports real-time collaboration for remote teams. The system must handle document editing, video conferencing, and chat. How would you design the architecture?

**Answer:**

**Analysis:**
- **Requirements:** Real-time collaboration, multiple modes.
- **Quality Attributes:** Performance, Availability, Usability.

**Architectural Reasoning:**
1. **Architecture Style:** **Event-driven** with **WebSockets**.
2. **Components:**
   - **Document Service:** Manages documents
   - **Video Service:** WebRTC for video
   - **Chat Service:** WebSockets for chat
   - **Presence Service:** User presence
3. **Tactics:**
   - **Performance:** WebSockets, CDN
   - **Availability:** Redundancy
   - **Usability:** Real-time updates
4. **Layered Architecture:**
   - **Presentation Layer:** UI
   - **Service Layer:** Collaboration API
   - **Business Layer:** Collaboration logic
   - **Data Layer:** Document DB

**Trade-Offs:**
- Real-time vs. consistency: Conflict resolution needed.
- Complexity: Multiple services add complexity.

**Recommendation:**
Use **WebSockets** for real-time, **WebRTC** for video, **operational transformation** for document sync, and **redundancy** for availability.

**Industry Relevance:** Microsoft Teams, Slack, Google Workspace.

**Key concepts tested:** Real-time collaboration, event-driven architecture, WebSockets.

---

### Q43. A company wants to build a system that supports regulatory reporting for financial transactions. The system must be auditable, secure, and modifiable. How would you design the architecture?

**Answer:**

**Analysis:**
- **Requirements:** Regulatory reporting, auditability.
- **Quality Attributes:** Security, Modifiability, Auditability.

**Architectural Reasoning:**
1. **Architecture Style:** **Layered** with **audit service**.
2. **Components:**
   - **Transaction Service:** Processes transactions
   - **Reporting Service:** Generates reports
   - **Audit Service:** Logs all activities
   - **Compliance Service:** Ensures compliance
3. **Tactics:**
   - **Security:** Encryption, access control
   - **Modifiability:** Configurable reporting rules
   - **Auditability:** Comprehensive logging
4. **Layered Architecture:**
   - **Presentation Layer:** Reporting UI
   - **Service Layer:** Reporting API
   - **Business Layer:** Reporting logic
   - **Data Layer:** Transaction DB, audit DB

**Trade-Offs:**
- Auditability vs. Performance: Logging adds overhead.
- Modifiability vs. Stability: Frequent changes may introduce bugs.

**Recommendation:**
Use **audit service**, **configurable reporting**, and **encryption**. Implement **immutable audit logs**.

**Industry Relevance:** Banks, financial institutions.

**Key concepts tested:** Auditability, security, modifiability, regulatory compliance.

---

### Q44. A company wants to build a system that handles real-time inventory management for a retail chain. The system must integrate with POS, e-commerce, and warehouse systems. How would you design the architecture?

**Answer:**

**Analysis:**
- **Requirements:** Real-time inventory, multi-system integration.
- **Quality Attributes:** Interoperability, Performance, Availability.

**Architectural Reasoning:**
1. **Architecture Style:** **Event-driven** with **adapters**.
2. **Components:**
   - **Inventory Service:** Manages inventory
   - **POS Adapter:** Integrates with POS
   - **E-commerce Adapter:** Integrates with e-commerce
   - **Warehouse Adapter:** Integrates with warehouse
3. **Tactics:**
   - **Interoperability:** Adapters, standard data format
   - **Performance:** Caching, async updates
   - **Availability:** Redundancy
4. **Layered Architecture:**
   - **Presentation Layer:** Inventory dashboard
   - **Service Layer:** Inventory API
   - **Business Layer:** Inventory logic
   - **Data Layer:** Inventory DB

**Trade-Offs:**
- Real-time vs. batch: Real-time improves accuracy but requires more infrastructure.
- Complexity: Multiple adapters add complexity.

**Recommendation:**
Use **event-driven architecture** with **adapters** for each system, **message queue** for async updates, and **caching** for performance.

**Industry Relevance:** Retail chains like Walmart, Target.

**Key concepts tested:** Interoperability, event-driven architecture, adapters.

---

### Q45. A company wants to build a system that supports real-time fraud detection for credit cards. The system must analyze transactions in <50ms and adapt to new fraud patterns. How would you design the architecture?

**Answer:**

**Analysis:**
- **Requirements:** Real-time fraud detection, <50ms, adaptive.
- **Quality Attributes:** Performance, Modifiability, Availability.

**Architectural Reasoning:**
1. **Architecture Style:** **Event-driven** with **stream processing**.
2. **Components:**
   - **Transaction Ingestion:** Kafka
   - **Fraud Engine:** Flink/Storm
   - **Rules Engine:** Drools
   - **ML Model Service:** Anomaly detection
   - **Alert Service:** Notifications
3. **Tactics:**
   - **Performance:** In-memory, parallel processing
   - **Modifiability:** Rules engine
   - **Availability:** Redundancy
4. **Layered Architecture:**
   - **Presentation Layer:** Fraud analyst dashboard
   - **Service Layer:** Fraud API
   - **Business Layer:** Fraud engine
   - **Data Layer:** In-memory DB, NoSQL

**Trade-Offs:**
- Speed vs. accuracy: Faster is less accurate.
- Complexity: ML models add complexity.

**Recommendation:**
Use **in-memory processing**, **parallel rules**, **pre-computed ML models**, and **redundancy**.

**Industry Relevance:** Credit card companies like Visa, Mastercard.

**Key concepts tested:** Real-time performance, event-driven architecture, ML.

---

### Q46. A company wants to build a system that supports real-time analytics for IoT sensor data. The system must handle millions of messages per second. How would you design the architecture?

**Answer:**

**Analysis:**
- **Requirements:** High volume, real-time analytics.
- **Quality Attributes:** Scalability, Performance, Availability.

**Architectural Reasoning:**
1. **Architecture Style:** **Event-driven** with **stream processing**.
2. **Components:**
   - **Ingestion:** Kafka
   - **Processing:** Flink/Spark Streaming
   - **Storage:** Cassandra
   - **Analytics:** Redis
3. **Tactics:**
   - **Scalability:** Horizontal scaling, sharding
   - **Performance:** In-memory, parallel processing
   - **Availability:** Redundancy
4. **Layered Architecture:**
   - **Presentation Layer:** Dashboards
   - **Service Layer:** Analytics API
   - **Business Layer:** Analytics logic
   - **Data Layer:** NoSQL, in-memory

**Trade-Offs:**
- Real-time vs. batch: Real-time is complex but useful.
- Cost: High volume requires significant infrastructure.

**Recommendation:**
Use **Kafka** for ingestion, **Flink** for stream processing, **Cassandra** for storage, **Redis** for caching, and **redundancy** for availability.

**Industry Relevance:** IoT platforms, smart cities.

**Key concepts tested:** Scalability, performance, event-driven architecture, IoT.

---

### Q47. A company wants to build a system that supports real-time collaboration for remote healthcare teams. The system must handle video, chat, and document sharing. How would you design the architecture?

**Answer:**

**Analysis:**
- **Requirements:** Real-time collaboration, healthcare compliance.
- **Quality Attributes:** Performance, Security, Availability.

**Architectural Reasoning:**
1. **Architecture Style:** **Event-driven** with **WebRTC**.
2. **Components:**
   - **Video Service:** WebRTC
   - **Chat Service:** WebSockets
   - **Document Service:** Document sharing
   - **Compliance Service:** HIPAA compliance
3. **Tactics:**
   - **Performance:** WebSockets, CDN
   - **Security:** Encryption, audit
   - **Availability:** Redundancy
4. **Layered Architecture:**
   - **Presentation Layer:** UI
   - **Service Layer:** Collaboration API
   - **Business Layer:** Collaboration logic
   - **Data Layer:** Encrypted DB

**Trade-Offs:**
- Security vs. Performance: Encryption adds latency.
- Compliance vs. Usability: Compliance adds friction.

**Recommendation:**
Use **WebRTC** for video, **WebSockets** for chat, **encryption** for security, and **audit logging** for compliance.

**Industry Relevance:** Telemedicine platforms.

**Key concepts tested:** Real-time collaboration, security, compliance, performance.

---

### Q48. A company wants to build a system that supports real-time pricing for e-commerce. The system must adjust prices based on demand, competition, and inventory. How would you design the architecture?

**Answer:**

**Analysis:**
- **Requirements:** Real-time pricing, multiple factors.
- **Quality Attributes:** Performance, Modifiability, Scalability.

**Architectural Reasoning:**
1. **Architecture Style:** **Event-driven** with **rules engine**.
2. **Components:**
   - **Pricing Service:** Calculates prices
   - **Demand Service:** Tracks demand
   - **Competition Service:** Tracks competitor prices
   - **Inventory Service:** Tracks inventory
   - **Rules Engine:** Pricing rules
3. **Tactics:**
   - **Performance:** Caching, async updates
   - **Modifiability:** Rules engine
   - **Scalability:** Horizontal scaling
4. **Layered Architecture:**
   - **Presentation Layer:** Pricing dashboard
   - **Service Layer:** Pricing API
   - **Business Layer:** Pricing logic
   - **Data Layer:** Pricing DB

**Trade-Offs:**
- Real-time vs. batch: Real-time is complex but responsive.
- Complexity: Multiple factors add complexity.

**Recommendation:**
Use **rules engine** for pricing, **event-driven** for real-time updates, and **caching** for performance.

**Industry Relevance:** Amazon, Walmart use dynamic pricing.

**Key concepts tested:** Real-time pricing, rules engine, modifiability.

---

### Q49. A company wants to build a system that supports real-time risk analysis for trading. The system must calculate risk in <100ms. How would you design the architecture?

**Answer:**

**Analysis:**
- **Requirements:** Real-time risk analysis, <100ms.
- **Quality Attributes:** Performance, Availability, Security.

**Architectural Reasoning:**
1. **Architecture Style:** **Event-driven** with **in-memory processing**.
2. **Components:**
   - **Trade Ingestion:** Kafka
   - **Risk Engine:** In-memory processing
   - **Position Service:** Tracks positions
   - **Market Data Service:** Provides market data
3. **Tactics:**
   - **Performance:** In-memory, parallel processing
   - **Availability:** Redundancy
   - **Security:** Encryption
4. **Layered Architecture:**
   - **Presentation Layer:** Risk dashboard
   - **Service Layer:** Risk API
   - **Business Layer:** Risk engine
   - **Data Layer:** In-memory DB

**Trade-Offs:**
- Speed vs. accuracy: Faster is less accurate.
- Cost: In-memory is expensive.

**Recommendation:**
Use **in-memory processing**, **parallel computation**, **redundancy**, and **encryption**.

**Industry Relevance:** Investment banks, hedge funds.

**Key concepts tested:** Real-time performance, risk analysis, event-driven architecture.

---

### Q50. A company wants to build a system that supports real-time personalized recommendations for a streaming service. The system must handle millions of users and provide recommendations in <200ms. How would you design the architecture?

**Answer:**

**Analysis:**
- **Requirements:** Real-time recommendations, <200ms, millions of users.
- **Quality Attributes:** Performance, Scalability, Availability.

**Architectural Reasoning:**
1. **Architecture Style:** **Service-oriented** with **ML models**.
2. **Components:**
   - **User Profile Service:** Stores preferences
   - **Content Service:** Manages content
   - **Recommendation Engine:** ML models
   - **API Gateway:** Unified interface
3. **Tactics:**
   - **Performance:** Caching, pre-computation
   - **Scalability:** Horizontal scaling
   - **Availability:** Redundancy
4. **Layered Architecture:**
   - **Presentation Layer:** UI
   - **Service Layer:** Recommendation API
   - **Business Layer:** Recommendation engine
   - **Data Layer:** NoSQL, in-memory

**Trade-Offs:**
- Real-time vs. pre-computed: Real-time is accurate but slower.
- Complexity: ML models add complexity.

**Recommendation:**
Use **pre-computed recommendations** for common cases, **real-time** for personalized, **caching** for performance, and **horizontal scaling** for scalability.

**Industry Relevance:** Netflix, Spotify, YouTube.

**Key concepts tested:** Real-time recommendations, performance, scalability, ML.

---

# Final Revision Section

## A. Top 20 Concepts to Revise

| Rank | Concept | Why It Matters | What to Remember |
|------|---------|----------------|------------------|
| 1 | Quality Attributes | Drive architectural decisions | Availability, Performance, Security, Modifiability, Usability, Testability, Interoperability |
| 2 | Quality Attribute Scenarios | Make requirements testable | Source, Stimulus, Environment, Artifact, Response, Response Measure |
| 3 | Architectural Drivers | Key factors influencing architecture | Business goals, constraints, quality attributes |
| 4 | ASRs | Requirements with profound effect on architecture | Identify from requirements, business goals, stakeholders |
| 5 | Utility Tree | Prioritize ASRs | Quality Attribute → Refinement → Scenario → Priority (H/M/L) |
| 6 | ADD | Systematic design method | Choose module, refine, choose tactics, instantiate, define interfaces, verify |
| 7 | ATAM | Evaluate architecture | Business drivers, utility tree, risks, sensitivity points, trade-offs |
| 8 | 4+1 View Model | Document architecture | Logical, Process, Development, Physical, Scenarios |
| 9 | Layered Architecture | Separation of concerns | Presentation, Business, Service, Data layers |
| 10 | Availability Tactics | Ensure uptime | Detect, Recover, Prevent faults |
| 11 | Performance Tactics | Meet timing requirements | Control demand, Manage resources |
| 12 | Security Tactics | Protect data | Detect, Resist, React, Recover |
| 13 | Modifiability Tactics | Reduce change cost | Increase cohesion, Reduce coupling, Defer binding |
| 14 | Testability Tactics | Improve testing | Control/observe state, Limit complexity |
| 15 | Interoperability Tactics | Integrate systems | Locate, Manage interfaces |
| 16 | Architecture Conformance | Ensure implementation matches design | Architecturally evident coding, frameworks, templates |
| 17 | Architecture Reconstruction | Understand existing systems | Raw view extraction, DB construction, View fusion, Analysis |
| 18 | Architecture Testing | Test architectural properties | Prioritize test cases based on ASRs, integration test plan |
| 19 | Agile Architecture | Balance agility and architecture | Sweet spot for up-front work, spikes, incremental |
| 20 | Architecture Documentation | Communicate design | Views, stakeholders, notations |

---

## B. Quality Attribute Cheat Sheet

| Quality Attribute | What It Means | Typical Architectural Concern | Example Scenario |
|-------------------|---------------|-------------------------------|------------------|
| **Availability** | System is ready when needed | Fault detection, recovery, redundancy | 99.999% uptime for trading system |
| **Performance** | Meets timing requirements | Latency, throughput, resource management | <100ms response for fraud detection |
| **Security** | Protects data from unauthorized access | Authentication, authorization, encryption | HIPAA-compliant patient records |
| **Scalability** | Handles growth | Horizontal/vertical scaling, load balancing | Millions of IoT messages per second |
| **Modifiability** | Ease of making changes | Cohesion, coupling, defer binding | Add new payment gateway in 2 weeks |
| **Testability** | Ease of testing | Control/observe state, limit complexity | 85% path coverage in 3 hours |
| **Interoperability** | Systems exchange information | Adapters, data transformation | Integrate with multiple EHR systems |
| **Usability** | Ease of use | User initiative, system initiative | New user productive in 2 minutes |

---

## C. Important Frameworks / Methods

### Quality Attribute Workshop (QAW)
- **Purpose:** Elicit quality attribute requirements from stakeholders
- **Key Steps:** Introductions → Business Mission → Architectural Plan → Identify Drivers → Brainstorm Scenarios → Consolidate → Prioritize
- **When to Use:** Early in requirements gathering

### Architectural Drivers
- **Purpose:** Identify key factors influencing architecture
- **Key Steps:** Identify business goals, constraints, quality attributes
- **When to Use:** Before design begins

### Utility Tree
- **Purpose:** Prioritize ASRs
- **Key Steps:** Root (Utility) → Branches (Quality Attributes) → Leaves (Scenarios) → Priority (H/M/L)
- **When to Use:** During ASR identification and prioritization

### Attribute-Driven Design (ADD)
- **Purpose:** Systematic architecture design
- **Key Steps:** Choose module → Refine based on ASRs → Choose tactics → Instantiate → Define interfaces → Verify
- **When to Use:** During architecture design

### 4+1 View Model
- **Purpose:** Document architecture for different stakeholders
- **Key Views:** Logical, Process, Development, Physical, Scenarios
- **When to Use:** During documentation

### Layered Architecture
- **Purpose:** Separation of concerns
- **Key Layers:** Presentation, Business, Service, Data
- **When to Use:** Enterprise applications

### ATAM
- **Purpose:** Evaluate architecture
- **Key Steps:** Present ATAM → Business Drivers → Architecture → Identify Approaches → Utility Tree → Analyze → Brainstorm → Analyze → Present Results
- **When to Use:** After architecture design

### Architecture Conformance
- **Purpose:** Ensure implementation matches design
- **Key Techniques:** Architecturally evident coding, frameworks, templates, documentation updates
- **When to Use:** During implementation

### Architecture Testing
- **Purpose:** Test architectural properties
- **Key Techniques:** Prioritize test cases based on ASRs, integration test plan
- **When to Use:** During testing

### Architecture Reconstruction
- **Purpose:** Understand existing systems
- **Key Steps:** Raw view extraction → DB construction → View fusion → Analysis → Iterate
- **When to Use:** Legacy modernization

---

## D. Common Exam Traps

| Trap | How to Avoid |
|------|--------------|
| Confusing business goals with architectural drivers | Business goals are high-level; architectural drivers are specific factors influencing architecture |
| Treating quality attributes as functional requirements | Quality attributes annotate functional requirements (how fast, how secure) |
| Identifying a quality attribute without creating a measurable scenario | Always create a scenario with source, stimulus, response, measure |
| Ignoring trade-offs | Every architectural decision has trade-offs; discuss them |
| Assuming the highest-performance architecture is best | Best architecture depends on business goals and constraints |
| Confusing architecture views with architectural styles | Views are representations; styles are patterns (layered, client-server) |
| Treating documentation as a one-time activity | Documentation must be updated as architecture evolves |
| Ignoring architecture conformance | Implementation can drift from architecture; check conformance |
| Assuming documented architecture = current implementation | Documentation may be outdated; reconstruct if needed |
| Confusing architecture reconstruction with architecture design | Reconstruction is understanding existing; design is creating new |
| Memorizing ATAM steps without understanding application | Focus on applying ATAM concepts to scenarios |
| Forgetting to prioritize ASRs | Use utility tree to prioritize based on business value and technical risk |
| Overlooking stakeholder concerns | Identify and engage stakeholders early |
| Ignoring constraints (cost, time, technology) | Constraints are zero-degree-of-freedom decisions |
| Not considering real-time requirements | Real-time systems have hard deadlines; use appropriate tactics |

---
---
---

# Software Architecture — 25 Theoretical Questions & Answers

---

## Section 1 — Core Concepts (Questions 1–6)

---

### Q1. Explain the concept of software architecture as a set of structures. How do module, component-and-connector, and allocation structures differ, and why is it important to consider multiple structures rather than a single architecture?

**Answer:**

**Software Architecture Definition:**
The software architecture of a system is the set of structures needed to reason about the system, which comprise software elements, relations among them, and properties of both.

**Three Categories of Structures:**

| Structure Type | Elements | Relations | Purpose |
|----------------|----------|-----------|---------|
| **Module** | Modules (code units) | Is-a-submodule-of, uses, inherits-from | Code organization, functional decomposition, modifiability |
| **Component-and-Connector (C&C)** | Runtime components, connectors | Attachment, communication | Runtime behavior, performance, availability, scalability |
| **Allocation** | Software elements, external environments | Allocated-to, migrates-to | Deployment, team assignment, file structure |

**Why Multiple Structures Matter:**

1. **Different stakeholder concerns:** Developers care about module structure; integrators care about C&C; system engineers care about allocation.
2. **Different quality attributes:** Modifiability is best analyzed through module structure; performance through C&C; security through deployment.
3. **No single structure is sufficient:** A module view cannot show runtime concurrency; a C&C view cannot show code organization.
4. **Reasoning power:** Each structure provides analytical leverage for specific quality attributes.

**Example:**
A client-server system can be viewed as:
- **Module view:** Client module and Server module (decomposition)
- **C&C view:** Client component and Server component with request-reply connector
- **Allocation view:** Client deployed to user machines, Server deployed to data center

**Key Takeaway:** Architecture is inherently multi-view. Documenting only one view provides an incomplete picture and hinders reasoning about critical quality attributes.

---

### Q2. What is the relationship between functionality and quality attributes in software architecture? Why is it said that functionality and quality attributes are orthogonal?

**Answer:**

**Functionality:** The ability of the system to do the work for which it was intended. It defines what the system does.

**Quality Attributes:** These annotate functional requirements—how fast, how secure, how available, how modifiable, etc.

**Orthogonality Explained:**

Functionality and quality attributes are orthogonal because:
1. **Multiple architectures can satisfy the same functionality:** Given a set of functional requirements, there is no end to the architectures you could create to satisfy that functionality.
2. **Quality attributes differentiate architectures:** Two systems with identical functionality can have vastly different architectures based on their quality attribute requirements.
3. **Functionality does not determine architecture:** A function like "process payment" can be implemented in a monolith, microservices, or layered architecture.

**Example:**

| Functional Requirement | Quality Attribute Annotation | Architectural Implication |
|------------------------|------------------------------|---------------------------|
| "When user presses green button, Options dialog appears" | Performance: Dialog appears < 0.5 seconds | Caching, pre-loading |
| Same | Availability: Function fails < 0.1% of time | Redundancy, failover |
| Same | Usability: Easy to learn in < 2 minutes | Maintain user model, help system |

**Why This Matters:**
- Architects cannot focus only on functionality; they must understand quality attribute requirements.
- Two systems with the same features can have completely different architectures.
- Quality attributes are the primary drivers of architectural decisions.

**Industry Relevance:** A banking app and a social media app may both have "user login" functionality, but the banking app's architecture will be driven by security and availability, while the social media app may prioritize performance and scalability.

---

### Q3. What are Architecturally Significant Requirements (ASRs)? How do they differ from ordinary requirements, and what indicators help identify them?

**Answer:**

**Definition:** An Architecturally Significant Requirement (ASR) is a requirement that will have a profound effect on the architecture of a system. These requirements have a measurable effect on a computer system's architecture.

**Difference from Ordinary Requirements:**

| Aspect | Ordinary Requirements | ASRs |
|--------|----------------------|------|
| Impact | Localized, straightforward implementation | Profound effect on architecture |
| Scope | Single module or component | System-wide structural implications |
| Risk | Low technical risk | High technical risk |
| Business Value | May have moderate value | Holds significant business value |
| Example | "System should create an order" | "System should detect fraud in real-time and block user" |

**Indicators of Architectural Significance:**
1. **Significant business value and/or technical risk**
2. **Important to a key stakeholder with considerable influence**
3. **Unique—no existing components address it**
4. **QoS/SLA aspects differ from current architecture**
5. **Previously led to budget overruns or client dissatisfaction**

**Common Examples:**
- **Scalability:** System should support future growth with minimal effort
- **Security:** System must protect data and prevent unauthorized access
- **Availability:** System should have specific uptime/reliability rate

**Why ASRs Matter:**
- Play a key role in determining overall system structure
- Influence design decisions early in development
- Help architects focus on what truly matters
- Even in critical applications, only 40% of software features are used

**Industry Relevance:** In a healthcare application, "HIPAA compliance" is an ASR because it affects encryption, audit trails, access control, and data storage across the entire system.

---

### Q4. Explain the concept of architectural tactics. How do tactics differ from architectural patterns, and why is it important to understand both?

**Answer:**

**Architectural Tactics:** A collection of primitive design techniques that an architect can use to achieve a quality attribute response. Tactics are the building blocks of design.

**Architectural Patterns:** Packaged strategies for solving some of the problems facing a system. Patterns are complex—they are a bundle of design decisions.

**Key Differences:**

| Aspect | Tactics | Patterns |
|--------|---------|----------|
| Scope | Single quality attribute response | Multiple quality attributes |
| Granularity | Primitive, fine-grained | Complex, coarse-grained |
| Purpose | Achieve specific quality attribute | Solve recurring architectural problem |
| Example | Heartbeat, Active Redundancy | Client-Server, Layered |
| Flexibility | Can be combined | Often need adaptation |

**Why Understand Both:**

1. **Patterns are often difficult to apply as-is:** Architects need to modify and adapt them.
2. **Tactics allow constructing design fragments from first principles:** If no pattern exists, tactics provide a systematic approach.
3. **Tactics make design more systematic:** By cataloguing tactics, architects have a toolkit for achieving quality attributes.
4. **Trade-off analysis:** Understanding tactics helps assess options for augmenting existing patterns.
5. **Choice of tactic depends on trade-offs:** Factors include other quality attributes and cost to implement.

**Example:**
- **Pattern:** Client-Server architecture
- **Tactics within it:** Heartbeat (availability), Concurrency (performance), Encryption (security)

**Industry Relevance:** An architect designing a trading system might use the Client-Server pattern but apply specific tactics like Active Redundancy for availability and Prioritize Events for performance.

---

### Q5. What are the four contexts of software architecture? Why is it important for an architect to understand each context?

**Answer:**

**Four Contexts of Software Architecture:**

1. **Technical Context:**
   - **What:** The technical role architecture plays in the system
   - **Key Factors:** Quality attributes, current technical environment, industry practices
   - **Why Important:** Determines what the architecture can achieve

2. **Project Life-Cycle Context:**
   - **What:** How architecture relates to development phases
   - **Key Factors:** Waterfall, Iterative, Agile, Model-driven development
   - **Why Important:** Architecture activities must align with process

3. **Business Context:**
   - **What:** How architecture affects the organization's business environment
   - **Key Factors:** Business goals, market position, customer needs
   - **Why Important:** Architecture must serve business purposes

4. **Professional Context:**
   - **What:** The role of the architect in the organization
   - **Key Factors:** Skills, knowledge, duties, leadership
   - **Why Important:** Architects need technical and soft skills

**Why Understanding All Contexts Matters:**

1. **Holistic Decision-Making:** Architecture decisions affect and are affected by all contexts.
2. **Stakeholder Communication:** Different stakeholders care about different contexts.
3. **Trade-Off Analysis:** Business context may override technical preferences.
4. **Career Development:** Professional context defines what skills architects need.
5. **Project Success:** Ignoring any context can lead to architectural failure.

**Example:**
A technically elegant architecture that ignores business goals (e.g., cost constraints) will fail. An architecture that meets business goals but ignores technical constraints (e.g., team expertise) will also fail.

**Industry Relevance:** In Agile projects, the project life-cycle context dictates that architecture must evolve incrementally, not be fully defined upfront.

---

### Q6. What makes a "good" architecture? Discuss the process rules and product rules that serve as guidelines for architects.

**Answer:**

**Important Caveat:** There is no such thing as an inherently good or bad architecture. Architectures are either more or less fit for some purpose. They can only be evaluated in the context of specific stated goals.

**Process Rules of Thumb:**

| Rule | Rationale |
|------|-----------|
| Architecture should be product of single architect or small group with identified leader | Gives conceptual integrity and technical consistency |
| Architect should have prioritized list of quality attribute requirements | Ensures focus on what matters |
| Architecture should be documented using views | Addresses concerns of most important stakeholders |
| Architecture should be evaluated using formal method (like ATAM) | Early identification of risks |
| Architecture should lend itself to incremental implementation | Create skeletal system with communication paths |

**Product Rules of Thumb:**

| Rule | Rationale |
|------|-----------|
| Use information hiding and separation of concerns | Encapsulate things likely to change |
| Achieve quality attributes using well-known tactics | Proven approaches |
| Never depend on particular version of commercial tool | Structured for easy change |
| Minimize number of ways components interact | Reduces complexity |
| Separate modules that produce data from those that consume it | Increases modifiability |
| Don't expect one-to-one correspondence between modules and components | Many-to-many mapping |
| Every process should be written so assignment to processor can change | Flexibility in deployment |
| System should do same things in same way throughout | Aids understandability |
| Architecture should contain specific (and small) set of resource contention areas | Clear resolution |

**Why These Rules Matter:**
- Provide guidance without being prescriptive
- Help architects avoid common pitfalls
- Can be adapted to different project contexts
- Balance technical and business concerns

**Industry Relevance:** These rules are applicable across domains—from banking to healthcare to e-commerce.

---

## Section 2 — Quality Attributes & Design (Questions 7–13)

---

### Q7. Explain the concept of a quality attribute scenario. What are its six parts, and why is it important to use scenarios rather than simple quality attribute definitions?

**Answer:**

**Definition:** A quality attribute scenario is a means of characterizing quality attributes in a testable, measurable way.

**Six Parts of a Quality Attribute Scenario:**

| Part | Description | Example (Availability) |
|------|-------------|------------------------|
| **Source** | Entity generating the stimulus | Internal/external: people, hardware, software |
| **Stimulus** | Condition requiring a response | Fault: omission, crash, incorrect timing |
| **Environment** | Conditions under which stimulus occurs | Normal operation, startup, overload |
| **Artifact** | System or portion stimulated | Processors, communication channels, storage |
| **Response** | Activity undertaken as result of stimulus | Detect fault, log, notify, recover |
| **Response Measure** | Measurable outcome | Availability %, time to detect, time to repair |

**Why Scenarios Are Important:**

1. **Testability:** Simple definitions like "system will be modifiable" are meaningless. Scenarios make requirements testable.
2. **Resolving Overlapping Concerns:** Is a denial-of-service attack about availability, performance, security, or usability? Scenarios clarify the specific concern.
3. **Common Vocabulary:** Different attribute communities have different vocabularies. Scenarios provide a unifying structure.
4. **Prioritization:** Scenarios can be prioritized using utility trees.
5. **Architectural Decision-Making:** Scenarios drive architectural decisions by making requirements concrete.

**Example Comparison:**

| Vague Definition | Quality Attribute Scenario |
|------------------|---------------------------|
| "System should be available" | "During normal operations, when a CPU fails, the system detects the fault within 2 seconds and continues operating with 99.999% availability." |
| "System should be secure" | "A disgruntled employee from remote location attempts to modify pay rate table during normal operations. System maintains audit trail and restores correct data within 1 day." |
| "System should be performant" | "A user updates patient account while system is under peak load. Transaction completes in less than 0.75 seconds." |

**Industry Relevance:** In requirements gathering, scenarios help bridge the gap between business stakeholders (who state needs) and architects (who design solutions).

---

### Q8. What are the key trade-offs between effectiveness, efficiency, and satisfaction in usability design? How can designers balance these competing factors?

**Answer:**

**Usability Definition:** Usability is concerned with how easy it is for the user to accomplish a desired task and the kind of user support the system provides.

**Three Key Factors:**

| Factor | Definition | Example |
|--------|------------|---------|
| **Effectiveness** | Whether the system is correct; degree to which user achieves goals | User successfully completes a transaction |
| **Efficiency** | Effort and time required to develop/use the system | Task completed in minimum steps |
| **Satisfaction** | User's comfort and confidence with the system | User feels in control and satisfied |

**Trade-Offs:**

1. **Effectiveness vs. Efficiency:**
   - More validation checks improve effectiveness but increase time (efficiency)
   - Example: Requiring multiple confirmations for a bank transfer

2. **Efficiency vs. Satisfaction:**
   - Streamlined processes (efficiency) may reduce user confidence (satisfaction)
   - Example: Auto-filling data saves time but may make users feel less in control

3. **Effectiveness vs. Satisfaction:**
   - Error prevention (effectiveness) may frustrate users (satisfaction)
   - Example: Blocking certain actions to prevent errors

**Balancing Strategies:**

1. **User Initiative Tactics:**
   - **Cancel:** Allow users to cancel long-running operations
   - **Undo:** Allow users to reverse actions
   - **Pause/Resume:** Allow users to temporarily stop tasks
   - **Aggregate:** Allow grouping of lower-level objects

2. **System Initiative Tactics:**
   - **Maintain Task Model:** Predict what user is trying to do
   - **Maintain User Model:** Represent user's knowledge and behavior
   - **Maintain System Model:** Provide appropriate feedback

3. **Design Principles:**
   - Progressive disclosure (show only what's needed)
   - Sensible defaults (reduce effort)
   - Clear feedback (maintain confidence)
   - Graceful error recovery (minimize impact)

**Example: Mobile Banking App**
- **Effectiveness:** Biometric authentication ensures only authorized access
- **Efficiency:** Quick balance check without full login
- **Satisfaction:** Clear confirmation messages, undo for transactions

**Industry Relevance:** Mobile apps must balance these factors carefully—users expect speed (efficiency) but also security (effectiveness) and confidence (satisfaction).

---

### Q9. How do the Security, Communication, and Reliability views of quality attributes intersect and impact the design of a software system? Provide an example where optimizing one view may compromise another.

**Answer:**

**The Three Views:**

| View | Focus | Key Concerns |
|------|-------|--------------|
| **Security** | Protection from unauthorized access | Confidentiality, Integrity, Availability, Authentication, Authorization |
| **Communication** | Exchange of information between components | Protocols, Message integrity, Timeliness, Reliability |
| **Reliability** | System's ability to perform required functions | Fault tolerance, Recovery, Consistency |

**Intersections:**

1. **Security & Communication:**
   - Encryption of data in transit
   - Authentication of communicating parties
   - Message integrity verification

2. **Security & Reliability:**
   - Audit trails for accountability
   - Secure recovery mechanisms
   - Protection against denial-of-service

3. **Communication & Reliability:**
   - Guaranteed message delivery
   - Ordering of messages
   - Handling communication failures

**Example Where Optimizing One Compromises Another:**

**Scenario:** A financial trading system

- **Optimizing Security:** Implement strong encryption (AES-256) for all communications
  - **Compromises Communication:** Encryption/decryption adds latency, slowing message exchange
  - **Compromises Reliability:** If encryption keys are lost, communication fails

- **Optimizing Communication:** Use lightweight protocols for low latency
  - **Compromises Security:** Less encryption may expose data
  - **Compromises Reliability:** Faster protocols may have less error checking

- **Optimizing Reliability:** Implement redundant communication paths
  - **Compromises Security:** More paths mean more attack surfaces
  - **Compromises Communication:** Redundancy adds complexity and overhead

**Balancing Approach:**

1. **Risk-Based Decision:** For high-value transactions, prioritize security; for low-risk, prioritize speed
2. **Hardware Acceleration:** Use HSMs for encryption to reduce latency
3. **Layered Security:** Apply strongest encryption only to sensitive data
4. **Monitoring:** Continuously monitor all three attributes

**Industry Relevance:** This trade-off is critical in financial services, healthcare, and any system handling sensitive data in real-time.

---

### Q10. Explain the concept of Attribute-Driven Design (ADD). What are its steps, and how does it help architects move from requirements to architecture?

**Answer:**

**Definition:** Attribute-Driven Design (ADD) is a systematic method for designing software architecture based on quality attribute requirements.

**Steps of ADD:**

1. **Choose Module to Decompose:**
   - Start with the system or a subsystem
   - Typically begin with the highest-level system

2. **Refine Module Based on ASRs:**
   - Identify Architecturally Significant Requirements (ASRs)
   - Select a subset of ASRs that are most relevant for this decomposition

3. **Choose Architectural Tactics:**
   - Select tactics that address the chosen ASRs
   - Consider trade-offs among tactics

4. **Instantiate Modules and Assign Responsibilities:**
   - Create modules/components based on chosen tactics
   - Assign functional responsibilities to each

5. **Define Interfaces:**
   - Specify how modules communicate
   - Define APIs, protocols, data formats

6. **Verify and Refine:**
   - Check if design meets ASRs
   - Refine as needed
   - Repeat for next module

**How ADD Helps:**

| Challenge | How ADD Addresses It |
|-----------|---------------------|
| Complexity | Breaks design into manageable steps |
| Requirements traceability | Each step linked to ASRs |
| Quality attribute focus | Tactics chosen based on quality attributes |
| Systematic approach | Repeatable process |
| Trade-off analysis | Forces consideration of alternatives |

**Example: E-commerce System**

1. **Choose Module:** Order Processing
2. **Refine Based on ASRs:** Performance (order < 2 sec), Availability (99.9%)
3. **Choose Tactics:** Caching (performance), Active Redundancy (availability)
4. **Instantiate:** Order Service with cache and redundant instances
5. **Define Interfaces:** REST API for order submission
6. **Verify:** Test if order completes < 2 sec under load

**Industry Relevance:** ADD is used in industry to systematically design architectures for complex systems, ensuring quality attributes are addressed from the start.

---

### Q11. What are the key differences between architecture evaluation by the designer, peer review, and analysis by outsiders? When is each appropriate?

**Answer:**

**Three Forms of Evaluation:**

| Aspect | By Designer | Peer Review | By Outsiders |
|--------|-------------|-------------|--------------|
| **Who** | Architect themselves | Colleagues within project | External experts |
| **When** | During design process | At milestones | After design complete |
| **Perspective** | Self-critical | Constructive | Objective |
| **Cost** | Low | Medium | High |
| **Depth** | Varies | Moderate | Comprehensive |

**1. Evaluation by the Designer:**

- **Purpose:** "Test" part of "generate-and-test" approach
- **When:** Every time a key design decision is made or milestone completed
- **How Much:** Depends on:
  - Importance of decision
  - Number of alternatives
  - Good enough vs. perfect
- **Advantages:** Immediate feedback, low cost
- **Disadvantages:** May miss own mistakes, less objective

**2. Peer Review:**

- **Purpose:** Leverage collective expertise
- **When:** At any point where candidate architecture exists
- **Steps:**
  1. Reviewers determine quality attribute scenarios
  2. Architect presents architecture
  3. For each scenario, designer explains how satisfied
  4. Potential problems captured
- **Advantages:** Multiple perspectives, moderate cost
- **Disadvantages:** May be biased by internal politics

**3. Analysis by Outsiders:**

- **Purpose:** Objective evaluation by experts
- **When:** After architecture is designed, especially for critical systems
- **Advantages:** Specialized knowledge, objective eye, managers listen more
- **Disadvantages:** High cost, requires time, may not understand context

**When to Use Each:**

| Situation | Recommended Evaluation |
|-----------|----------------------|
| Small project, low risk | Designer evaluation only |
| Medium project, some risk | Designer + Peer Review |
| Large project, high risk | Designer + Peer Review + Outsiders |
| Critical system (life, money, security) | Full evaluation including ATAM |

**Industry Relevance:** In practice, most projects use a combination. ATAM is a formal method that incorporates all three perspectives.

---

### Q12. How does the Utility Tree facilitate identification and prioritization of ASRs? Provide an example scenario where the Utility Tree helps refine quality attributes and drive design decisions.

**Answer:**

**Utility Tree Definition:** A hierarchical representation used to capture and prioritize key quality attributes, breaking them down into more detailed scenarios.

**Structure:**

```
Utility (Root)
├── Quality Attribute (Branch)
│   ├── Attribute Refinement (Sub-branch)
│   │   └── ASR/Scenario (Leaf) [Priority: (Business Value, Technical Risk)]
```

**How It Facilitates ASR Identification:**

1. **Systematic Exploration:** Forces consideration of all quality attributes
2. **Concretization:** Converts vague quality attributes into specific scenarios
3. **Prioritization:** Each scenario gets (H/M/L, H/M/L) rating
4. **Focus:** Identifies high-priority scenarios for design attention
5. **Communication:** Provides common language for stakeholders

**Example: Healthcare System (Nightingale)**

| Quality Attribute | Refinement | ASR | Priority |
|-------------------|------------|-----|----------|
| **Performance** | Transaction response time | User updates patient account under peak load; completes < 0.75 sec | (H, M) |
| | Throughput | At peak load, 150 normalized transactions/sec | (M, M) |
| **Usability** | Proficiency training | New hire becomes proficient in < 1 week | (M, L) |
| | Normal operations | Payment officer initiates payment plan without delays | (M, M) |
| **Security** | Confidentiality | Physical therapist sees orthopaedic data but not financial | (H, M) |
| | Integrity | System resists intrusion and reports within 90 sec | (H, M) |
| **Availability** | No downtime | System supports 24/7 web access | (L, L) |
| **Modifiability** | Routine changes | Bug fix distributed with < 3 person-days effort | (H, M) |
| | Extensibility | New product (blood bank) created in 2 person-months | (M, M) |

**How Utility Tree Drives Design Decisions:**

1. **High-Priority Scenarios Get Attention:**
   - Performance (H, M) → Design uses caching, load balancing
   - Security (H, M) → Design uses encryption, access control

2. **Trade-Offs Identified:**
   - Performance vs. Security: Encryption may slow responses
   - Usability vs. Modifiability: Training vs. frequent changes

3. **Architectural Tactics Selected:**
   - For Performance: Manage sampling rate, prioritize events
   - For Security: Authenticate actors, encrypt data

**Industry Relevance:** Utility trees are used in ATAM evaluations to focus on what matters most to stakeholders.

---

### Q13. How do the different views in Philippe Kruchten's 4+1 architectural model support identification and analysis of ASRs? What role do quality attributes play in integrating these views?

**Answer:**

**4+1 View Model:**

| View | Purpose | Stakeholders | Quality Attributes |
|------|---------|--------------|-------------------|
| **Logical** | Functional requirements, key abstractions | End users, designers | Functionality, Usability |
| **Process** | Concurrency, distribution, performance | Integrators, performance engineers | Performance, Scalability, Availability |
| **Development** | Software organization, modules, libraries | Programmers, managers | Modifiability, Reusability |
| **Physical** | Mapping to hardware, distribution | System engineers, operations | Availability, Reliability, Performance |
| **Scenarios** | Validation, illustration | All stakeholders | All quality attributes |

**How Views Support ASR Identification:**

1. **Logical View:**
   - Identifies functional ASRs
   - Shows key abstractions and their relationships
   - Example: "System must support patient record management"

2. **Process View:**
   - Identifies performance and concurrency ASRs
   - Shows processes, threads, synchronization
   - Example: "System must handle 1000 concurrent users"

3. **Development View:**
   - Identifies modifiability ASRs
   - Shows module organization and dependencies
   - Example: "New payment gateway can be added in 2 weeks"

4. **Physical View:**
   - Identifies deployment and availability ASRs
   - Shows hardware mapping and distribution
   - Example: "System must survive data center failure"

5. **Scenarios:**
   - Validate that architecture meets ASRs
   - Provide concrete examples
   - Example: "User books flight, system confirms in < 2 sec"

**Role of Quality Attributes in Integrating Views:**

1. **Common Thread:** Quality attributes run through all views
2. **Trade-Off Analysis:** Views show how optimizing one attribute affects others
3. **Stakeholder Communication:** Different views address different stakeholder concerns
4. **Completeness Check:** All quality attributes should be addressed in at least one view

**Example Integration:**

- **Performance ASR:** "Trade execution < 10ms"
  - Logical View: Trading engine components
  - Process View: Concurrent processing, thread pools
  - Physical View: Co-location of components
  - Development View: Optimized algorithms

**Industry Relevance:** The 4+1 model is widely used in industry for documenting architectures because it addresses multiple stakeholder concerns.

---

## Section 3 — Documentation, Conformance & Evaluation (Questions 14–19)

---

### Q14. Explain the importance of architecture documentation. What are the key uses and audiences, and how should documentation be structured to serve different stakeholders?

**Answer:**

**Importance of Architecture Documentation:**

1. **Communication:** Common abstraction for stakeholders
2. **Analysis:** Enables evaluation of quality attributes
3. **Education:** Introduces new team members to system
4. **Decision Record:** Captures rationale for design choices
5. **Conformance:** Provides baseline for implementation checks

**Uses and Audiences:**

| Audience | Concern | Relevant Views |
|----------|---------|----------------|
| **Developers** | Code organization, interfaces | Module, Development |
| **Integrators** | Component interactions | C&C, Process |
| **System Engineers** | Deployment, hardware | Physical, Allocation |
| **Project Managers** | Work breakdown, schedule | Development, Work Assignment |
| **Testers** | Testable interfaces | Module, C&C |
| **Maintainers** | Change impact | Module, C&C |
| **End Users** | Functionality | Logical, Scenarios |
| **Customers** | Business goals | Scenarios, Business context |

**Documentation Structure:**

1. **Views:** Document relevant views based on stakeholder needs
   - Module views (decomposition, uses, layered, class)
   - C&C views (process, concurrency, shared data, client-server)
   - Allocation views (deployment, implementation, work assignment)

2. **Beyond Views:**
   - Architecture overview
   - Mapping between views
   - Rationale for decisions
   - Glossary

3. **View Templates:**
   - Primary presentation
   - Element catalog
   - Context diagram
   - Variability guide
   - Rationale

**Key Principles:**

1. **Write for the reader:** If reader doesn't need it, don't write it
2. **Document relevant views:** Not all views needed for all projects
3. **Keep it current:** Update as architecture evolves
4. **Use appropriate notation:** UML, informal diagrams, etc.

**Example Documentation Package:**

| Section | Content |
|---------|---------|
| 1. Scope | Purpose, context |
| 2. References | Related documents |
| 3. Architecture | Overview, goals, constraints |
| 4. Logical Architecture | Key abstractions, classes |
| 5. Process Architecture | Concurrency, performance |
| 6. Development Architecture | Modules, layers |
| 7. Physical Architecture | Deployment |
| 8. Scenarios | Use cases |
| 9. Size and Performance | Metrics |
| 10. Quality | Quality attribute analysis |

**Industry Relevance:** Good documentation reduces onboarding time, prevents architectural drift, and facilitates maintenance.

---

### Q15. What is architecture conformance? How can organizations ensure that implementation remains consistent with the designed architecture?

**Answer:**

**Definition:** Architecture conformance is the degree to which the implementation of a system adheres to the designed architecture.

**Why Code Drifts from Architecture:**

1. **Not sticking to layer discipline:** Object in one layer calling object beyond adjacent layer
2. **Direct database access:** Bypassing data access layer
3. **Point-to-point notifications:** Instead of publish-subscribe
4. **Ignoring common mechanisms:** Not using common logging, security

**Techniques to Ensure Conformance:**

| Technique | Description | Example |
|-----------|-------------|---------|
| **Architecturally Evident Coding Style** | Indicate architectural aspects in code | Comments, naming conventions |
| **Frameworks** | Use frameworks that enforce architecture | Spring, Hibernate, AUTOSAR |
| **Code Templates** | Provide templates for common patterns | Fault-tolerant template |
| **Architecture Documentation** | Update when code changes | Mark sections as "no longer applicable" |
| **Code Reviews** | Review for architectural violations | Checklist-based |
| **Folder Structure** | Organize code by architectural aspect | Layer folders, service folders |
| **Education** | Train new team members | Architecture overview sessions |
| **Automated Checks** | Use tools to detect violations | SonarJ, Lattix, Structure 101 |

**Examples of Conformance Techniques:**

**1. Architecturally Evident Coding Style:**
```java
// LAYER: Business
// PATTERN: Publish-Subscribe (Publisher)
public class OrderService {
    // Publishes order events
}

// LAYER: Data Access
// PATTERN: Repository
public class OrderRepository {
    // Data access only
}
```

**2. Templates for Fault Tolerance:**
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

**3. Tools for Detection:**
- **SonarJ:** Detects layer violations
- **Lattix:** Dependency structure matrix
- **Structure 101:** Architecture analysis

**Industry Relevance:** In large projects with multiple teams, conformance checking is essential to prevent architectural decay.

---

### Q16. How does architecture help in testing activities? What work products of architecture design can be used to prioritize test cases and create integration test plans?

**Answer:**

**How Architecture Helps Testing:**

1. **Prioritize Test Cases Based on ASRs:**
   - Architecture identifies ASRs
   - ASRs determine what to test most thoroughly
   - High-priority scenarios become high-priority test cases

2. **Create Integration Test Plan:**
   - Architecture shows module interactions
   - Architecture shows dependencies
   - Helps identify integration test cases

3. **Design for Testability:**
   - Architecture can be designed to support testing
   - Tactics like specialized interfaces, record/playback

**Work Products for Prioritizing Test Cases:**

| Work Product | How It Helps |
|--------------|--------------|
| **Utility Tree** | Identifies high-priority scenarios (high business value + high architectural impact) |
| **Quality Attribute Scenarios** | Provides specific, testable requirements |
| **ASR List** | Focuses testing on what matters most |
| **Module Views** | Shows what to unit test |
| **C&C Views** | Shows what to integration test |

**Example: Utility Tree for Testing**

| Quality Attribute | Scenario | Priority | Test Priority |
|-------------------|----------|----------|---------------|
| Performance | Transaction < 0.75 sec | (H, M) | High |
| Security | Resists intrusion < 90 sec | (H, M) | High |
| Availability | 24/7 access | (L, L) | Low |
| Modifiability | Bug fix < 3 person-days | (H, M) | Medium |

**How Architecture Helps Integration Testing:**

1. **Identifies Modules That Interact:**
   - C&C view shows connections
   - Determines integration points

2. **Identifies Dependencies:**
   - Module view shows uses relations
   - Determines test order

3. **Determines Test Scope:**
   - Which modules need to be integrated
   - What interfaces need testing

**Testability Tactics:**

| Tactic | Purpose |
|--------|---------|
| Specialized Interfaces | Control/capture variable values |
| Record/Playback | Capture information for replay |
| Localize State Storage | Start system in arbitrary state |
| Abstract Data Sources | Substitute test data |
| Sandbox | Isolate system for experimentation |
| Executable Assertions | Indicate faulty state |

**Industry Relevance:** In practice, architects work with testers to ensure testability requirements are met and test cases are prioritized based on architectural significance.

---

### Q17. Explain the Architecture Tradeoff Analysis Method (ATAM). What are its phases, participants, and outputs?

**Answer:**

**ATAM Definition:** A comprehensive method for evaluating software architectures by having project decision makers and stakeholders articulate a precise list of quality attribute requirements and illuminating architectural decisions relevant to each high-priority scenario.

**Participants:**

| Role | Responsibilities |
|------|------------------|
| **Evaluation Team** | External, unbiased, 3-5 people |
| **Project Decision Makers** | Project manager, customer, architect |
| **Architecture Stakeholders** | Developers, testers, integrators, maintainers, users |

**Phases of ATAM:**

| Phase | Activity | Participants | Duration |
|-------|----------|--------------|----------|
| **0** | Partnership and preparation | Evaluation team, decision makers | Weeks |
| **1** | Evaluation (Steps 1-6) | Evaluation team, decision makers | 1-2 days |
| **2** | Evaluation (Steps 7-9) | Evaluation team, decision makers, stakeholders | 2 days |
| **3** | Follow-up | Evaluation team | Report generation |

**Steps 1-9:**

1. **Present ATAM:** Explain process, set expectations
2. **Present Business Drivers:** System overview, constraints, goals
3. **Present Architecture:** Architect presents architecture
4. **Identify Architectural Approaches:** Catalog patterns and tactics
5. **Generate Utility Tree:** Prioritize quality attribute scenarios
6. **Analyze Architectural Approaches:** Map high-priority scenarios to architecture
7. **Brainstorm and Prioritize Scenarios:** Stakeholders contribute scenarios
8. **Analyze Architectural Approaches (Round 2):** Evaluate new scenarios
9. **Present Results:** Risks, non-risks, sensitivity points, trade-offs

**Outputs of ATAM:**

| Output | Description |
|--------|-------------|
| **Concise Architecture Presentation** | Documented architecture |
| **Business Goals** | Articulated and captured |
| **Prioritized Quality Attribute Scenarios** | From utility tree |
| **Risks** | Decisions that may lead to undesirable consequences |
| **Non-Risks** | Decisions deemed safe |
| **Risk Themes** | Overarching systemic weaknesses |
| **Sensitivity Points** | Decisions affecting specific quality attributes |
| **Trade-Off Points** | Decisions affecting multiple quality attributes |
| **Mapping of Decisions to Requirements** | How decisions achieve quality attributes |

**Intangible Outputs:**

- Sense of community among stakeholders
- Open communication channels
- Better overall understanding of architecture
- Often the longest-lasting results

**Example Risk and Trade-Off:**

| Type | Example |
|------|---------|
| **Risk** | Heartbeat frequency affects detection time |
| **Sensitivity Point** | Number of database clients affects transactions/sec |
| **Trade-Off** | Higher heartbeat frequency = better availability but more processing time |

**Industry Relevance:** ATAM has been used for over a decade in domains ranging from automotive to financial to defense.

---

### Q18. What is architecture reconstruction? Explain its phases and the tools that support it.

**Answer:**

**Definition:** Architecture reconstruction is the process of recovering the architecture of an existing system, especially when documentation is missing or outdated.

**Purpose:**

1. **Understand systems with no documentation**
2. **Migrate from old to new technology** (e.g., mainframe to web)
3. **Identify reusable components** (logging, security)

**Phases of Architecture Reconstruction:**

| Phase | Description | Output |
|-------|-------------|--------|
| **1. Raw View Extraction** | Extract information from source code, execution traces, build scripts | Classes, files, caller-callee relationships, global data access |
| **2. DB Construction** | Put extracted data into common format | Repository |
| **3. View Fusion** | Combine views from different sources | Consolidated view |
| **4. Architecture Analysis** | Validate correctness, check violations | Reconstructed architecture |
| **5. Iterate** | Refine as needed | Refined architecture |

**Detailed Phase Descriptions:**

**Phase 1: Raw View Extraction**
- **Sources:** Source code, execution traces, build scripts
- **Extracted Elements:** Classes, files, functions, variables
- **Extracted Relations:** Includes, contains, defines, calls, accesses

**Examples of Extracted Elements and Relations:**

| Source Element | Relation | Target Element | Description |
|----------------|----------|----------------|-------------|
| File | includes | File | C preprocessor #include |
| File | contains | Function | Definition of function in file |
| Function | calls | Function | Static function call |
| Function | access_read | Variable | Read access on variable |
| Function | access_write | Variable | Write access on variable |

**Phase 2: DB Construction**
- Store extracted data in a common format
- Enable querying and manipulation

**Phase 3: View Fusion**
- **View 1:** Source code analysis (static view)
- **View 2:** Execution traces (dynamic view)
- **View 3:** Expert grouping into layers
- **Combine:** Consolidated view

**Phase 4: Architecture Analysis**
- Check conformance to architecture
- Example violations:
  - Layer calls non-adjacent layer
  - Database access not via entity bean

**Tools for Architecture Reconstruction:**

| Tool | Description |
|------|-------------|
| **ARMIN** | Architecture Reconstruction and Mining |
| **Dali** | Architecture analysis |
| **Lattix** | Dependency structure matrix |
| **SonarJ** | Architecture conformance |
| **Structure 101** | Architecture analysis |

**Case Study: Vanish System**

- Tool: ARMIN
- Reconstructed architecture showed:
  - Not strictly layered
  - Dialogue, Functional_Core_Adapter, Logical_Interaction, Utilities, Functional_Core, Presentation
- Analysis: One can notice architecture is not strictly layered

**Industry Relevance:** Used in legacy modernization, merger & acquisition due diligence, and understanding third-party systems.

---

### Q19. How does documentation of software architecture influence identification and prioritization of ASRs? What role does it play in ensuring quality attributes are met?

**Answer:**

**How Documentation Influences ASR Identification:**

1. **Provides Baseline:** Documents current architecture, making gaps visible
2. **Facilitates Communication:** Common reference for stakeholders to discuss requirements
3. **Captures Rationale:** Records why decisions were made, informing future ASRs
4. **Enables Analysis:** Documented views allow systematic analysis of quality attributes

**Documentation Artifacts That Help Identify ASRs:**

| Artifact | How It Helps |
|----------|--------------|
| **Quality Attribute Scenarios** | Concrete, testable requirements |
| **Utility Tree** | Prioritized ASRs |
| **Architecture Views** | Show where quality attributes are addressed |
| **Design Decisions** | Record rationale, constraints |
| **Business Goals** | Link requirements to business needs |

**Role in Ensuring Quality Attributes:**

1. **Traceability:** Link ASRs to architectural decisions
2. **Completeness Check:** Ensure all ASRs are addressed
3. **Trade-Off Analysis:** Document trade-offs between quality attributes
4. **Verification:** Provide basis for evaluating if system meets requirements

**Example: Documentation Impact on ASRs**

| Documentation Element | ASR Identification | Quality Attribute Ensured |
|----------------------|-------------------|--------------------------|
| Logical View | Functional requirements | Usability, Functionality |
| Process View | Performance, concurrency | Performance, Scalability |
| Development View | Modifiability, reuse | Modifiability |
| Physical View | Deployment, availability | Availability, Reliability |
| Scenarios | Validation | All |

**How Documentation Helps When Requirements Change:**

1. **Impact Analysis:** Understand what parts of architecture are affected
2. **Change Propagation:** Trace how changes affect quality attributes
3. **Decision Reuse:** Learn from past decisions
4. **Stakeholder Communication:** Discuss changes with all affected parties

**Best Practices:**

1. **Document Relevant Views:** Only what's needed for stakeholders
2. **Keep Current:** Update when architecture changes
3. **Link to Requirements:** Trace ASRs to design decisions
4. **Use Standard Notation:** UML, informal diagrams
5. **Include Rationale:** Why decisions were made

**Industry Relevance:** In regulated industries (healthcare, finance), documentation is mandatory for compliance and audit.

---

## Section 4 — Layered Architecture, Agile & Real-Time (Questions 20–25)

---

### Q20. Explain the layered architecture pattern. What are the responsibilities of each layer, and what are the benefits and potential pitfalls?

**Answer:**

**Definition:** Layered architecture organizes the system into layers, where each layer provides services to the layer above and uses services from the layer below.

**Typical Layers:**

| Layer | Responsibility | Technologies |
|-------|----------------|--------------|
| **Presentation** | User Interface, User Experience | React, Angular, Mobile |
| **Business/Logic** | Business rules, data validation, processing | Java, .NET, Python |
| **Service** | APIs, orchestration, external integration | REST, SOAP, GraphQL |
| **Data Access** | Persistence, ORM, database connections | Hibernate, JPA, SQL |
| **Cross-Cutting** | Security, logging, communication | Aspects, Frameworks |

**Responsibilities in Detail:**

**Presentation Layer:**
- Handle user input
- Render UI
- Client-side caching
- Asynchronous communication (AJAX)
- Responsive design

**Business Layer:**
- Application façade (hide internal modules)
- Session management
- Workflow engine
- Rules engine
- Aspect-oriented design

**Service Layer:**
- Provide services to external systems
- Handle duplicate requests
- Handle out-of-sequence messages
- Handle communication failures

**Data Layer:**
- DB connection pool
- Multiple copies of data
- Transactions (atomicity)
- Object-Relational mapping
- Stored procedures
- Parameterized queries (SQL injection prevention)

**Benefits:**

| Benefit | Description |
|---------|-------------|
| **Separation of Concerns** | Each layer has distinct responsibility |
| **Loose Coupling** | Layers interact through well-defined interfaces |
| **Reusability** | Lower layer components can be reused |
| **Exchangeable Parts** | Layers can be replaced independently |
| **Maintainability** | Changes localized to specific layers |
| **Scalability** | Layers can be scaled independently |

**Potential Pitfalls:**

| Pitfall | Description | Consequence |
|---------|-------------|-------------|
| **Layer Bypassing** | Presentation layer directly accesses data layer | Tight coupling, maintainability issues |
| **Sinkhole Anti-Pattern** | Requests pass through layers without processing | Performance overhead |
| **Tight Coupling** | Layers know too much about each other | Changes propagate across layers |
| **Wrong Responsibilities** | Business logic in presentation layer | Duplication, inconsistency |
| **Too Many Layers** | Over-engineering | Complexity, performance |

**Example: E-commerce System**

```
┌─────────────────────────────────────┐
│         Presentation Layer          │
│  (Web UI, Mobile UI)                │
├─────────────────────────────────────┤
│         Business Layer              │
│  (Order Processing, Pricing)        │
├─────────────────────────────────────┤
│         Service Layer               │
│  (APIs for Partners)                │
├─────────────────────────────────────┤
│         Data Access Layer           │
│  (Order DAO, Product DAO)           │
├─────────────────────────────────────┤
│         Database                    │
└─────────────────────────────────────┘
```

**Industry Relevance:** Layered architecture is the most common pattern in enterprise applications due to its simplicity and maintainability.

---

### Q21. How does architecture differ in Agile projects compared to traditional projects? What is the "sweet spot" for up-front architecture work?

**Answer:**

**Agile vs. Traditional Architecture:**

| Aspect | Traditional | Agile |
|--------|-------------|-------|
| **Up-front work** | Extensive | Minimal |
| **Documentation** | Comprehensive | Lightweight |
| **Design** | Big design up front | Emergent design |
| **Changes** | Resisted | Welcomed |
| **Architecture** | Fixed | Evolves |

**The "Sweet Spot" for Up-Front Architecture:**

Based on Boehm and Turner's research:

| Project Size | Sweet Spot (% of schedule) | Rationale |
|--------------|---------------------------|-----------|
| **10 KSLOC** | ~0% | Too small for up-front work to pay off |
| **100 KSLOC** | ~20% | Moderate up-front work reduces rework |
| **1,000 KSLOC** | ~40% | Significant up-front work essential |

**Why the Sweet Spot Exists:**

1. **Trade-Off:** Up-front work vs. rework
2. **Small Projects:** Up-front work is waste; can iterate quickly
3. **Large Projects:** Without up-front work, rework costs explode
4. **Complexity:** Million-line projects need architecture to organize effort

**How Agile Architects Work:**

1. **Two Modes:**
   - **Top-down:** Design and analyze architectural structures
   - **Bottom-up:** Analyze implementation constraints

2. **Spikes:** Experiments to make trade-offs
   - Would distributed database impact latency?
   - What scalability from mod_perl vs. standard Perl?
   - How many participants per meeting server?

3. **Incremental Commitment Model:**
   - Commitment and accountability of stakeholders
   - Stakeholder "satisficing"
   - Incremental and evolutionary growth
   - Iterative development
   - Interleaved definition and development
   - Risk management

**Guidelines for Agile Architects:**

| Project Type | Recommendation |
|--------------|----------------|
| Large, stable requirements | Do significant up-front architecture |
| Large, unstable requirements | Start with candidate architecture, evolve |
| Small, uncertain requirements | Agree on major patterns, don't over-document |

**Example: WebArrow (Agile Architecting)**

- Web-conferencing system
- Needed top-down and bottom-up thinking
- Used spikes to answer questions:
  - Distributed database vs. local flat files?
  - mod_perl vs. standard Perl?
  - Ratio of database servers to meeting servers?

**Industry Relevance:** Most modern projects blend Agile and architecture. The key is finding the right balance for the project's size and risk.

---

### Q22. What are the architectural implications of real-time systems? How do timing constraints, reliability, and resource constraints affect design?

**Answer:**

**Real-Time System Definition:** Systems where the correctness of the system depends not only on the logical result but also on the time at which the result is produced.

**Architectural Implications:**

| Constraint | Architectural Implication | Tactics |
|------------|--------------------------|---------|
| **Timing** | Must meet deadlines | Prioritize events, Bound execution times, Schedule resources |
| **Reliability** | Must operate correctly | Active redundancy, Heartbeat, Watchdog |
| **Resource** | Limited CPU, memory | Manage sampling rate, Reduce overhead, Increase resource efficiency |

**Timing Constraints:**

1. **Hard Real-Time:** Missing deadline = system failure
   - Example: Airbag deployment, pacemaker
   - Architecture: Deterministic scheduling, no caching

2. **Soft Real-Time:** Missing deadline = degraded quality
   - Example: Video streaming, online gaming
   - Architecture: Best-effort scheduling, buffering

3. **Firm Real-Time:** Missing deadline = result useless
   - Example: Financial trading
   - Architecture: Prioritization, fast paths

**Performance Tactics for Real-Time:**

| Tactic | Description | Example |
|--------|-------------|---------|
| **Manage Sampling Rate** | Reduce data capture frequency | Lower video resolution |
| **Limit Event Response** | Process only up to max rate | Queue overflow handling |
| **Prioritize Events** | Rank by importance | Critical alerts first |
| **Reduce Overhead** | Remove intermediaries | Direct communication |
| **Bound Execution Times** | Limit processing time | Timeout mechanisms |
| **Increase Resource Efficiency** | Better algorithms | Optimized sorting |
| **Introduce Concurrency** | Parallel processing | Thread pools |
| **Maintain Multiple Copies** | Replicas | Redundant servers |

**Reliability Tactics for Real-Time:**

| Tactic | Description |
|--------|-------------|
| **Active Redundancy** | Hot spare processes identical input |
| **Passive Redundancy** | Warm spare receives periodic state |
| **Heartbeat** | Periodic message to monitor |
| **Watchdog** | Timer that triggers on failure |
| **Rollback** | Revert to known good state |

**Resource Constraints:**

1. **CPU:** Limited processing power
   - Solution: Offload to hardware, optimize algorithms

2. **Memory:** Limited storage
   - Solution: Compression, caching strategies

3. **Network:** Limited bandwidth
   - Solution: Data prioritization, compression

**Example: Industrial Control System**

| Requirement | Architectural Decision |
|-------------|----------------------|
| Response < 10ms | Dedicated real-time OS, no GC |
| 99.999% availability | Active redundancy, heartbeat |
| Limited memory | Static allocation, no dynamic memory |
| Safety-critical | Watchdog, fail-safe mode |

**Industry Relevance:** Real-time systems are critical in automotive, aerospace, medical devices, and financial trading.

---

### Q23. What are the key considerations when documenting architectures that change faster than you can document them? How does this apply to Agile projects?

**Answer:**

**The Problem:** In fast-moving projects, architecture documentation becomes outdated quickly. This is known as architectural drift.

**Key Considerations:**

1. **Document Only What's Necessary:**
   - Write for the reader
   - If reader doesn't need it, don't write it
   - Remember maintainers and newcomers

2. **Use Lightweight Documentation:**
   - Architecture as Code (Mermaid.js, PlantUML)
   - Diagrams in same Git repo as code
   - Automated checks (fitness functions)

3. **Focus on Stability:**
   - Document decisions unlikely to change
   - Mark volatile sections
   - Use "No longer applicable" markers

4. **Synchronize at Release:**
   - At release time, synchronize architecture doc with code
   - Update only what's changed

**Agile Documentation Principles:**

| Principle | Application |
|-----------|-------------|
| **Working software over comprehensive documentation** | Document only what adds value |
| **Responding to change** | Keep documentation lightweight |
| **Customer collaboration** | Document for stakeholders, not for compliance |
| **Simplicity** | Maximize work not done |

**Techniques for Agile Documentation:**

1. **Architecture as Code:**
   - Use PlantUML, Mermaid.js
   - Store in version control
   - Generate diagrams from code

2. **Fitness Functions:**
   - Automated checks for architectural conformance
   - Example: Ensure no cyclic dependencies
   - Example: Ensure layer rules followed

3. **Living Documentation:**
   - Update as part of definition of done
   - Use wikis, not static PDFs
   - Link to code

4. **Just Enough Documentation:**
   - Module views for developers
   - C&C views for integrators
   - Deployment views for operations

**What to Document in Agile:**

| Document | When | Why |
|----------|------|-----|
| **Architecture Overview** | Project start | Common understanding |
| **Key Decisions** | As made | Rationale |
| **Interfaces** | When defined | Integration |
| **Deployment** | When changed | Operations |

**Example: Agile Architecting at WebArrow**

- Used spikes to answer architectural questions
- Documented only what was needed
- Focused on experiments, not documents

**Industry Relevance:** Modern DevOps and CI/CD pipelines enable Architecture as Code, keeping documentation current with minimal effort.

---

### Q24. Explain the relationship between architecture and requirements. How do architects gather ASRs from requirements documents, stakeholders, and business goals?

**Answer:**

**Relationship Between Architecture and Requirements:**

1. **Requirements Drive Architecture:** Functional and quality attribute requirements determine architectural decisions
2. **Architecture Constrains Requirements:** Once architecture is set, it constrains what can be required
3. **Iterative Relationship:** Requirements and architecture co-evolve

**Gathering ASRs from Requirements Documents:**

| Step | Action |
|------|--------|
| 1 | Review requirements for MoSCoW (Must, Should, Could, Won't) |
| 2 | Check for Quality Attributes (SWEBOK recommendations) |
| 3 | Look for BSR (Basic Skill Requirements) |
| 4 | Identify challenging requirements |

**Look for Leads from Design Decision:**

| Design Decision Category | Look for Requirements Addressing |
|--------------------------|----------------------------------|
| Allocation of Responsibilities | Planned evolution, user roles, modes |
| Coordination Model | Timeliness, currency, correctness |
| Data Model | Processing steps, information flows |
| Management of Resources | Time, concurrency, memory, scalability |
| Mapping among Elements | Team plans, processors, networks |
| Binding Time | Extension, portability, configurations |
| Choice of Technology | Named technologies, changes |

**Gathering ASRs from Stakeholders:**

**Quality Attribute Workshop (QAW):**

| Step | Activity |
|------|----------|
| 1 | Presentation & Introduction |
| 2 | Business Mission Presentation |
| 3 | Architectural Plan Presentation |
| 4 | Identification of Architectural Drivers |
| 5 | Scenario Brainstorming |
| 6 | Scenario Consolidation |
| 7 | Scenario Prioritization |

**Gathering ASRs from Business Goals:**

**PALM Method (Pedigreed Attribute eLicitation Method):**

| Step | Activity |
|------|----------|
| 1 | PALM Overview |
| 2 | Business Drivers Presentation |
| 3 | Architecture Drivers Presentation |
| 4 | Business Goals Elicitation |
| 5 | Identify Quality Attributes from Goals |
| 6 | Assign Pedigree to Quality Attributes |
| 7 | Exercise Conclusion |

**Business Goals to Quality Attributes:**

| Business Goal | Quality Attribute |
|---------------|-------------------|
| Speed up cashless transactions | Usability, Security |
| Reduce workload by 8% | Modifiability |
| Regulatory compliance | Security, Auditability |
| Productivity improvement | Performance |
| Automated testing strategy | Testability |

**Example: BHIM App**

- **Business Goal:** Speed up cashless transactions
- **Quality Attributes:** Usability (easy steps), Security (3-factor)
- **Architectural Decisions:** Simple UI, multi-factor authentication

**Tying Methods Together:**

1. **Start with Requirements Documents:** Identify candidates
2. **Interview Stakeholders:** Refine and prioritize
3. **Understand Business Goals:** Link to quality attributes
4. **Build Utility Tree:** Consolidate and prioritize
5. **Iterate:** As requirements evolve

**Industry Relevance:** Effective ASR gathering is critical for project success. Missing an ASR can lead to costly architectural changes later.

---

### Q25. Compare and contrast the roles of architecture evaluation and architecture reconstruction. When would you use each, and how do they complement each other?

**Answer:**

**Definitions:**

| Aspect | Architecture Evaluation | Architecture Reconstruction |
|--------|------------------------|------------------------------|
| **Purpose** | Assess if architecture meets goals | Recover architecture from implementation |
| **When** | During/after design | When documentation missing/outdated |
| **Input** | Architecture documentation | Source code, execution traces |
| **Output** | Risks, trade-offs, recommendations | Reconstructed architecture, violations |
| **Method** | ATAM, peer review | Raw view extraction, view fusion |

**Architecture Evaluation:**

**When to Use:**
- After designing a new architecture
- Before major investment
- When quality attributes are critical
- When multiple stakeholders involved

**Methods:**
- ATAM (Architecture Tradeoff Analysis Method)
- Lightweight Architecture Evaluation
- Peer Review

**Outputs:**
- Risks and non-risks
- Sensitivity points
- Trade-off points
- Risk themes

**Architecture Reconstruction:**

**When to Use:**
- Legacy system modernization
- Merger & acquisition due diligence
- Understanding third-party systems
- When documentation is missing

**Phases:**
1. Raw view extraction
2. DB construction
3. View fusion
4. Architecture analysis
5. Iterate

**Outputs:**
- Reconstructed architecture
- Violations
- Reusable components

**How They Complement Each Other:**

| Scenario | Evaluation | Reconstruction |
|----------|------------|----------------|
| **New System** | Evaluate proposed architecture | Not needed |
| **Legacy System** | Evaluate after reconstruction | Reconstruct first |
| **Migration** | Evaluate target architecture | Reconstruct current |
| **Merger** | Evaluate combined architecture | Reconstruct both |

**Integrated Approach:**

1. **Reconstruct:** Understand current architecture
2. **Evaluate:** Assess if it meets goals
3. **Design:** Create target architecture
4. **Evaluate:** Assess target architecture
5. **Implement:** Build target
6. **Conform:** Ensure implementation matches design

**Example: Legacy Modernization**

| Step | Activity | Method |
|------|----------|--------|
| 1 | Reconstruct current architecture | ARMIN, expert workshops |
| 2 | Identify violations | SonarJ, Lattix |
| 3 | Evaluate current architecture | ATAM |
| 4 | Design target architecture | ADD |
| 5 | Evaluate target architecture | ATAM |
| 6 | Implement and conform | Conformance checks |

**Key Differences:**

| Aspect | Evaluation | Reconstruction |
|--------|------------|----------------|
| **Direction** | Design → Assessment | Implementation → Design |
| **Focus** | Quality attributes | Structure recovery |
| **Stakeholders** | All | Architects, developers |
| **Time** | Days to weeks | Weeks to months |
| **Tools** | ATAM, utility tree | ARMIN, Dali, SonarJ |

**Industry Relevance:** Both are essential for managing architectural health. Evaluation prevents problems; reconstruction diagnoses existing ones.

---

# End of 25 Theoretical Questions

---

## Summary of Coverage

| Section | Topics Covered | Questions |
|---------|----------------|-----------|
| **Core Concepts** | Architecture definition, structures, functionality vs. quality attributes, ASRs, tactics, contexts, good architecture | 1-6 |
| **Quality Attributes & Design** | Quality attribute scenarios, usability trade-offs, security/communication/reliability, ADD, evaluation forms, utility tree, 4+1 views | 7-13 |
| **Documentation, Conformance & Evaluation** | Documentation importance, conformance, testing, ATAM, reconstruction, documentation & ASRs | 14-19 |
| **Layered Architecture, Agile & Real-Time** | Layered architecture, Agile architecture, real-time systems, documentation in Agile, requirements gathering, evaluation vs. reconstruction | 20-25 |

These 25 theoretical questions complement the 50 scenario-based questions and cover the entire syllabus comprehensively. Each answer provides exam-quality reasoning, trade-offs, and industry relevance.