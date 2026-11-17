# Software Architectures (SE ZG651 / SS ZG653) — EC3 Scenario Q&A Bank

**Companion to `SA_EC3_Notes.md`.** Every question below is scenario-based and answered in full 10-mark style: assumptions → concept → mapping → both-side analysis → decision. Marks are indicated so you can calibrate length (60–80 words per mark). Section references like [B3] point to the notes.

---

# How to Answer — the Answer-Writing Guide (memorise this before anything else)

## 1. The reusable answer skeleton (works for almost every EC-3 question)

1. **Assumptions first.** The paper instructs it, and it licenses every later choice. Make them *usable*: "≈2,000 concurrent users at peak", "three feeds available as streams keyed by customer ID" — not "many users".
2. **Name the driver.** Which quality attribute / ASR does the scenario make dominant? Write it as a six-part scenario with numbers when it helps.
3. **Select and map.** Choose the pattern/tactic/technology and **map the scenario's own components onto its elements** ("the fraud-checker is a filter", "the fare service is the server"). Mapping earns marks; reciting definitions does not.
4. **Both sides.** State the quality gained AND the weakness paid — every pattern has benefits and liabilities; every tactic has side effects.
5. **Resolve side effects.** Name the tactic that repairs each liability and stop when what remains is small.
6. **Decide, with a condition.** Close by answering the exact question asked, plus the condition under which the decision would flip ("if load were certain within months, I would instead...").

## 2. Command words → what they demand

| Command word | What is asked | Structure that fits |
|---|---|---|
| Define / state / name | Exact wording or list | Definition, then one consequence |
| Describe | A mechanism | Steps in order, then one example |
| Explain | How and why | Mechanism → link to quality attribute or scenario |
| Analyse | Parts, relationships, effects | Break scenario into elements; effect of each on each attribute; conclude |
| Discuss / evaluate | Both sides + judgement | Gain, price, conclusion tied to the scenario |
| Compare / contrast | Same criteria on both sides | Table, one criterion per mark; end with how they complement |
| Propose / design / construct | A structure for the scenario | Assumptions → labelled diagram → walk-through → justification by forces |
| Justify | A decision with reasons | Decision → 2–3 reasons → the matching liability you accept |

"Explain how you would apply" = map the scenario onto the pattern's elements. "Discuss trade-offs" = both sides. "Specific examples" = numbers or named frameworks/products.

## 3. Marks-to-depth calibration

150 minutes / 40 marks = **3.75 min per mark**. 60–80 words per mark (tables and diagram count). A 10-mark question ≈ **600–800 words**; a 2-mark sub-part ≈ **one precise paragraph** (no preamble); a 1.5-mark sub-part ≈ one labelled diagram + short narrative. Spend no more than ~35 minutes per question; leave 5 minutes to check assumptions and sub-part numbering.

## 4. Four habits that lose marks (avoid)

1. Reciting the definition and never touching the scenario's components.
2. Listing benefits without the price (examiners read one-sided answers as half-answers).
3. Inventing unstated numbers silently — label them "illustrative".
4. Calculations without formula → substitution → arithmetic → unit → interpretation.

---

# THEME 1 — Pipes & Filters and MVC (EC-3 Q1 territory)

### Q1.1 [10 marks — past-paper shape] A retail company wants a data-processing workflow that turns purchases, preferences and demographics into personalised recommendations. Propose a pipes-and-filters architecture, explain how it handles different data types and scaling, and identify limits of the pattern for real-time use with remedies.

**Assumptions (illustrative):** three feeds — order events, preference records, customer profiles — arrive as streams keyed by customer ID; recommendations are served to a web page/app; volume grows continuously.

**(i) Proposed architecture [~3 marks].**
The design has three stages — ingestion, processing, output — with every link a pipe carrying one agreed record format:

```
Purchases ──► [Validate & clean purchases] ──► [Aggregate purchase history] ──┐
Preferences ─► [Normalise preferences] ──────► [Encode preference vector] ───┼─► [JOIN by customer ID]
Demographics ► [Standardise demographics] ───► [Map to customer segment] ────┘        │
                                                                       [Generate candidates]
                                                                       [Score with model]
                                                                       [Rank + business rules]
                                                                                │
                                                  Data sink: recommendation store / API
Second sink (red pipe from every cleaning filter): rejected records, for repair & audit
```

- *Ingestion:* each feed enters a validation/cleaning filter; invalid records leave via a second pipe to a rejected-records sink — pipes carry data, so the error path is a designed flow, not an afterthought.
- *Processing:* a per-type branch turns each feed into features; a **join filter** combines the three streams by customer ID into a complete feature record; candidate generation, model scoring and ranking follow.
- *Output:* the data sink is the recommendation store/API read by web/app/e-mail.
- Pipes are **bounded queues** (capacity, end-of-data marker, when-full/when-empty behaviour), so filters stay exchangeable and a fast producer is slowed to the consumer's speed.

**(ii) Configuration per data type and scaling [~3 marks].**
Each data type gets its own branch because format and cleaning differ: purchases are validated and aggregated; preferences normalised and encoded; demographics standardised and mapped to a segment. **Every branch ends in the same record format keyed by customer ID**, so the join needs no knowledge of the branch — a new data type (browsing history) is added as a new branch without touching existing filters (recombination). Scaling: branches run in parallel (independent filters); a slow filter is **replicated behind a partitioner** — it splits input by customer ID, copies run concurrently, a merger recombines results; bounded pipes apply back-pressure so memory stays bounded. The gain is limited by the **slowest filter** and by **inter-filter data transfer cost**.

**(iii) Limits in real-time use + remedies [~4 marks].**
- *Latency through the whole chain:* results only as fast as the slowest stage; remedy: replicate the slow filter, keep filters incremental (streaming, not batch), pre-compute candidates offline.
- *Missing data:* a customer missing one feed produces no join output; remedy: the join emits partial records flagged incomplete; a downstream filter scores with available features; a repair process consumes the rejected-records sink and re-injects.
- *Recommendation accuracy with streaming cleanup:* cleaning on the fly may distort features; remedy: dual path — accurate nightly batch recompute (MapReduce over history) plus fast approximate pipeline during the day (lambda design).
- *Shared state is expensive:* user profiles needed by several stages must travel in the stream or live in a store; remedy: cache service keyed by customer ID.
- *Error handling:* pipes carry no error information; remedy: side-output pipes + dead-letter sink, as designed above.
- *Alternative architectures:* for strict seconds-level personalisation use **publish-subscribe + real-time scoring service** (event-driven, model served in-memory) or a streaming framework; the pipeline remains for batch feature preparation. Decision: keep pipes-and-filters for the heavy batch feature pipeline; add an online scoring service for real-time recommendations.

### Q1.2 [4 marks — past-paper shape] Redesign an e-commerce platform with MVC to improve maintainability: how do you separate concerns, what is the benefit for a new payment gateway and a modified order workflow, and how does a price-calculation change leave the UI untouched?

**Assumption:** a web storefront serves customers and administrators; every screen reads the same business data.

**Separation of concerns [1 mark].** The **Model** holds catalogue, stock, pricing rules, orders, order workflow and the payment-gateway interface — it knows nothing about screens or input. **Views** (product page, cart, order status, admin report) only display data fetched from the model. **Controllers** translate user events ("add to cart", "pay") into service calls on the model and select the answering view. Change propagation: when the model changes it notifies every registered view, which re-reads the model and redraws.

**Adaptability analysis [1 mark].** A **new payment gateway** changes one component — the payment element inside the model; the checkout controller still issues the same request and views show the same result, so neither is edited. A **modified order workflow** is confined to the model's order logic; views change only if the workflow adds something to display. The same model can also serve a **new view+controller pair** (mobile storefront) — this is why the separation pays: each change has exactly one home, so its blast radius is bounded.

**Price example [1 mark].** The discount algorithm lives in the model's pricing rules. Replacing it is a single change inside the model. The cart controller keeps calling the same price service; when prices change, the model notifies the views, which fetch the new price and redraw. **No view or controller code is touched** — the trace (controller → model → notify → views re-fetch) proves it.

**Maintenance [1 mark].** Each change lands in one component, so the effect of an update is bounded and the **UI is not re-tested for business-rule changes**; regression scope shrinks. The honest cost: MVC adds a registry, an update protocol and three component types — more machinery than a simple interface needs, and every change notifies every registered view (possible update storms; use fine-grained notifications if that hurts).

*(Hedge if the question says "scalability": scalability is not a claimed MVC benefit; an interface-independent model can in principle be hosted/replicated separately, which is the usual route in web platforms.)*

### Q1.3 [10 marks] A hospital tele-monitoring system ingests vitals from 5,000 bedside devices every second, cleans them, detects anomalies, stores them, and pushes alerts. Choose between pipes-and-filters and publish-subscribe for the ingestion layer; justify with the quality attributes at stake; propose a combined design.

**Assumptions:** per-device vitals (pulse, SpO2) every 1 s; anomaly detection is lightweight per record; storage is bulk; alerting is urgent; devices may drop and reconnect.

**Analysis of the two candidates.**
- *Pipes and filters* [B3]: fits the **known, ordered transformation** (clean → detect → store), incremental filters give low latency and parallelism, small filters are reusable and testable; liability: shared state (device registry) is awkward, error handling needs side pipes, and one slow filter paces everything.
- *Publish-subscribe* [B9]: fits the **unknown set of consumers** (alerting, dashboards, storage, future analytics all subscribe to "vitals" events); decouples producers from consumers; liability: weaker ordering/delivery guarantees and added latency; exactly what alerts cannot afford without care.

**Decision — combined design.** Ingestion is a **pipeline**: device-feed → validate-and-clean filter → anomaly-detection filter; the anomaly filter **publishes** "alert" events to a pub-sub bus whose subscribers are the alarm service, the doctor-app push service and the audit logger; the storage filter consumes the cleaned stream directly and writes in bulk. Rationale per attribute: *performance* — pipeline stages scale horizontally per organ/vital type (partition by device ID); *availability* — pub-sub decouples consumers, so a down dashboard never blocks alerts; brokers/queues buffer surges; *modifiability* — a new analytics consumer subscribes without touching ingestion; *testability* — filters tested on recorded streams.

**Side effects resolved:** pub-sub ordering risk for alerts → use a connector with per-device ordering guarantee and at-least-once delivery + idempotent alerting; pipeline error handling → side-output to a dead-letter queue with a replay filter; back-pressure at 5,000 msg/s → bounded queues + partitioned replicas of the cleaning filter.

### Q1.4 [6 marks] Your MVC web application now needs a command-line reporting interface and a partner-facing REST API over the same data. What does MVC predict about this change, and what do you add?

**Answer.** MVC predicts this is cheap: the **model is unchanged** because it has no dependency on presentation or input. Each new interface is a **new view+controller pair**: the CLI is a view rendering text plus a controller translating command-line arguments into model service calls; the REST API is a JSON view plus a controller translating HTTP requests (the URL and verb are the "gestures"). Change propagation notifies any registered views; the CLI view subscribes only when a long-running command watches data. What MVC does not settle, and what must be added: **authentication/authorisation for the partner API** (an intermediary at the API boundary — token validation before the controller runs), **rate limiting** (bound the event rate), and **notification fan-out cost** (the REST layer may prefer polling the model rather than registering every request as an observer). Liability acknowledged: both new views depend on the model's interface — a model-interface change reaches them; keep the interface versioned.

### Q1.5 [4 marks] Why is a compiler front end a good pipes-and-filters case, and why is its parser/scanner part usually NOT implemented as independent filters?

**Answer.** The compiler as a whole is a stream transformation: source text → tokens → parse tree → semantic checks → intermediate code → object code — each stage consumes the previous stage's output, stages are independently reusable, and a pipeline lets the code generator write to a pipe from which a separate interpreter/VM reads (the course example: parser calls scanner and semantic analyser via ordinary calls; the code generator writes to a UNIX pipe; the interpreter reads it). Inside the front end, however, **parser and scanner share state**: the parser pushes back tokens, consults symbol tables mid-parse and needs the scanner's next token immediately — the incremental, stateless interaction of filters would be inefficient and awkward. Pipes and filters pays when neighbours agree only on a data format; here the tight coupling is intentional and cheap because both run in one process. Lesson for answers: apply the pattern at the *boundaries* where independence is worth its transfer cost, not dogmatically everywhere.

### Q1.6 [6 marks] Compare pipes-and-filters and blackboard for a fraud-detection engine over transaction streams, and choose.

**Answer.** *Pipes & filters* suits a **known recipe**: normalise → rule-check → score → alert; benefits are parallelism, reuse, low latency; liability: the recipe hard-codes today's fraud logic, and adding a new detection technique means rewiring the chain. *Blackboard* [B4] suits **no deterministic strategy**: rule-expert, model-expert, graph-expert and human-analyst knowledge sources all inspect a shared case board (hypotheses with confidence), an opportunistic controller decides who acts next, and partial evidence is combined and revised; benefits: easy experimentation and exchange of experts; liabilities: no guarantee of timely termination, harder testing, synchronisation on the board. **Choice:** fraud *scoring on the live stream* needs predictable latency → **pipeline** (rules + model as filters, alerts published). *Investigation of a flagged case* is the open-ended problem → **blackboard** behind an analyst console. The two connect: the pipeline's alert filter creates a case on the board. This split matches each pattern's forces and keeps the latency-critical path free of opportunistic scheduling.

---

# THEME 2 — Broker, SOA/Microservices, Publish-Subscribe & Comparisons

### Q2.1 [10 marks — past-paper shape for patterns] A state disaster-response platform must make an ambulance dispatch service (hospital network), fire-brigade vehicle tracking (legacy mainframe) and a police map service (separate network) usable by call-centre and mobile applications that do not know where the services run; a new district's service must be addable at runtime. Apply the Broker pattern: map components, trace one request, identify liabilities and propose a tactic for each.

**Assumptions:** services expose callable operations; the networks are interconnected but administratively separate; addition of services happens without redeploying clients.

**(i) Component mapping [3 marks].**
- **Client** = call-centre console application and the mobile field app — they implement user functionality and send requests through client-side proxies.
- **Server** = ambulance dispatch service, fire-brigade vehicle tracking (a server-side proxy also fronts the legacy mainframe's protocol), police map service — each **registers with the broker** on start.
- **Broker** = the platform's middleware component: registers/unregisters servers, locates servers, forwards requests, returns results/exceptions, recovers on error.
- **Client-side proxies** inside console and app make remote services look like local objects (hide IPC + marshalling).
- **Server-side proxies** wrap each service (for the mainframe, the proxy also translates its protocol).
- **Bridge** connects the broker to a second broker serving the police network — interoperability across networks with different stacks.
- The "new district's service" is simply **another server registering with the broker** at runtime — no client change, which is exactly the pattern's force of dynamic add/remove.

**(ii) Request trace [2 marks].** Call-centre user requests nearest ambulance → console calls `dispatch(area)` on what looks like a local object → client-side proxy packs parameters and sends the request → broker consults its registry, finds the ambulance service (activating it if inactive), forwards → server-side proxy unpacks, calls the service → dispatch result flows back: service → server-proxy (pack) → broker → client-proxy (unpack) → console. Five hand-offs each way; the console never learned an address.

**(iii) Liabilities + tactics [3 marks].**
1. **Latency/indirection** (five hand-offs per call) → tactic: *increase resources* on the broker tier and allow the **direct-communication variant** — broker locates once, proxies then exchange directly — for high-frequency vehicle-tracking reads.
2. **Broker = single point of failure** → tactic: *maintain multiple copies* of the broker (replicated registry with failover), since every client depends on it.
3. **Mainframe protocol heterogeneity & testing difficulty** (an error may sit in client, proxy, broker, bridge or server) → tactic: *encapsulation/use an intermediary* — keep all mainframe specifics inside its server-side proxy, and test each service against its interface in isolation (the pattern's testing benefit), with end-to-end health checks (ping/echo) so a dead bridge is detected within seconds.

**Close:** the broker buys location transparency and runtime extensibility at the price of an extra hop and a critical intermediary; both prices are mitigated by standard tactics, so the pattern fits the scenario's dominant requirement — services usable without knowing where they run.

### Q2.2 [6 marks] An insurer's monolithic claims system is being decomposed into microservices. Compare the microservices decision with a classic SOA/ESB approach for (a) data ownership, (b) deployment, (c) failure handling — and recommend the decomposition boundary.

**Answer.**
- **(a) Data ownership:** classic SOA often leaves shared enterprise databases accessed through services (or an ESB-mediated store) — services couple through shared schemas. Microservices mandate **database per service**: each service owns its tables; others reach the data only via its API or its published events; the price is **eventual consistency** across services, managed with events and compensating actions (CAP: choose AP where the business allows).
- **(b) Deployment:** SOA's ESB centralises routing/transformation — changing one route touches shared middleware; releases coordinate. Microservices are **independently deployable** units behind lightweight pipes (REST/messaging) plus an API gateway for client-facing policy; CI/CD per service ("you build it, you run it").
- **(c) Failure handling:** ESB-era systems fail when the bus or a shared DB fails (SPOFs). Microservices design failure per service: **circuit breakers** fail fast to fallbacks, **bulkheads** isolate resource pools, graceful degradation (quotes display with cached prices if the pricing service is down), retries with idempotent operations.
- **Recommendation on boundaries:** decompose along **business capabilities/bounded contexts** — Claim Intake, Document Processing, Assessment Rules, Payment, Customer Notification — not along technical layers. Each owns its data; cross-service processes (intake→assessment→payment) are orchestrated by events (publish-subscribe) rather than a distributed transaction. Legacy document-processing code that cannot yet be split stays behind an **anti-corruption layer** and one service façade. Rationale: bounded-context decomposition maximises independent change (modifiability) while limiting the blast radius of failure (availability) — the two attributes that motivated leaving the monolith.

### Q2.3 [5 marks] A stock-trading dashboard receives price updates from 400 sources and feeds 12 different screens (portfolio, alerts, charts, audit). Which pattern organises this and why not the alternatives?

**Answer.** **Publish-subscribe** fits: the number and nature of producers (400 feeds) and consumers (12 screens, more later) are **not predetermined**; components interact through announced events; the connector delivers events to registered subscribers, so screens are added without touching feed adapters. Mapping: feed adapters = publishers (announce "price-tick"); screens = subscribers (listen by topic/symbol); event distributor = the bus. Why not the alternatives: **client-server request/reply** forces each screen to poll each source — coupling in identity and wasted requests; **shared-data** would make 12 screens hammer one store (bottleneck + tight data-model coupling) and lose the event-driven timing; **pipes and filters** fixes a single path, but a tick must fan out to many consumers, which is a tee to many pipes at best — clumsy and static. Liabilities accepted and mitigated: ordering/delivery not guaranteed → per-symbol ordered delivery and at-least-once with idempotent screen updates; latency → distributor sized per message rate; a slow screen → per-subscriber queues so one lagging consumer cannot slow publication (back-pressure per subscriber).

### Q2.4 [6 marks] Two teams disagree: the invoicing module should be a Shared-Data component reading a common billing database, or a service behind a published interface. Decide using pattern theory (elements, constraints, weaknesses) and the modifiability tactics involved.

**Answer.** *Shared-Data reading* [B9]: elements = data accessors + one store; constraint = accessors interact only with the store; strengths = one consistent view of billing data, simple reporting; weaknesses = the store is a bottleneck and SPOF, and **producers/consumers are tightly coupled through the data model** — a schema change is a change to every accessor, which is the exact risk when two teams evolve independently. *Service behind a published interface* (SOA/microservice style) [B8]: elements = provider with a published contract + consumer; weaknesses = build complexity, middleware overhead, service evolution outside consumer control — but the **contract** localises change: the provider can re-implement internals (even move its storage) without touching the consumer. Tactics view: the service option applies *encapsulation* and *use an intermediary* (restrict dependencies), directly serving **modifiability**; shared-data keeps coupling high, serving *performance for reads* instead. **Decision:** two independent teams + an evolving data model → **publish a Billing service** (versioned contract, events for changes); keep a **shared read-only reporting replica** (shared-data as a consumer) for heavy queries, so reporting never becomes a write-side coupling. Side effect: event/replica staleness — bounded with an SLA on lag and a reconciliation job.

### Q2.5 [4 marks] Contrast the Controller of MVC with the Control component of PAC, and the Broker with the ESB.

**Answer.** *MVC Controller vs PAC Control:* an MVC controller mediates **within one triad** — it translates user gestures into model/view requests and selects the view; it does not talk to other MVC triads' components (views share one model; the model's notifications do the synchronising). A PAC **Control** mediates within its own agent AND is the **only channel between agents** — the hierarchy of controls coordinates composite UIs (top/intermediate/bottom agents), so cross-agent communication is by design routed and controlled, at the cost of coordination overhead. *Broker vs ESB:* both are intermediaries that absorb location and heterogeneity. The **broker** locates servers and forwards requests/results for a closed set of registered servers (CORBA-style middleware). The **ESB** additionally **routes and transforms messages** between service providers and consumers in an SOA — protocol/shape mediation and orchestration are its job — and its consumers bind to *published service descriptions* rather than registered objects. Both add a hop and can become bottlenecks/SPOFs; the broker's failure halts its clients, and an ESB's failure can halt the enterprise flows that depend on it — replicate both.

### Q2.6 [8 marks] A national exam-results service must serve 2 million students the evening results are published, then fall to near-zero load. Microservices were chosen for agility. Design the runtime architecture and justify each choice against performance, availability and modifiability.

**Assumptions:** read-dominated workload; results frozen at publication; annual spike, 12-hour duration.

**Design and justification.**
- **API gateway** fronts the services (routing, auth, rate limiting): one entry hides internal topology; rate limiting protects back ends at the spike [D1].
- **Stateless Results service** (reads a replica): stateless → horizontal scale-out by adding instances (increase resources); results are immutable that evening → cache-friendly.
- **Distributed cache / CDN** for result pages keyed by roll number: most requests never touch the DB; staleness is irrelevant because results don't change [D6].
- **Message queue between Ingest service and store:** results loaded once from the board's feed; queue decouples and buffers; ingestion is idempotent per student.
- **Availability:** auto-scaling + replicated service and DB tiers (active pair) remove SPOFs; the gateway is load-balanced/replicated; graceful degradation — if the *certificate PDF* service fails, the marks page still serves (bulkhead between critical and optional services) [D7].
- **Modifiability:** microservices per capability (Results, Certificate, Recheck-request, Notification) with DB-per-service; the Recheck service can evolve yearly without touching Results; new consumer channels (SMS) subscribe to published "results-ready" events — publish-subscribe inside the platform.
- **CAP note:** during the evening, availability dominates (students must see marks); the pipeline is AP — replicas may briefly diverge during the load window, acceptable because data is frozen; the Recheck-request path is CP-ish (each request exactly once — deduplicate by student ID).
- **Elasticity economics:** peak lasts ~12 h/year — serverless or aggressive auto-scaling avoids paying for idle capacity the rest of the year; measured service aligns cost to the spike [D3/D6].

### Q2.7 [4 marks] Why does publish-subscribe weaken delivery guarantees, and how do enterprise messaging systems restore them?

**Answer.** The weakness is structural: publishers **do not know their subscribers**, so they cannot wait for each subscriber's acknowledgement — delivery and ordering guarantees that a direct call takes for granted (the caller observes the callee's result) disappear; the distributor adds latency and its scaling limits delivery-time predictability. Messaging systems restore guarantees by inserting durable machinery: **persistent queues/topics** (message survives broker restart), **acknowledgements per subscriber** with redelivery (at-least-once — consumers must be idempotent), **durable subscriptions** (messages retained for offline subscribers), and **ordered partitions** (per-key FIFO, e.g. per account or device). Price: throughput and latency overhead of persistence, and the semantic burden moves to consumers (deduplication). These restored guarantees are exactly what makes pub-sub usable for orders and payments despite the pattern's default weaknesses [B9].

---

# THEME 3 — Security Patterns in C&C Design (EC-3 Q2 territory)

### Q3.1 [4 marks — past paper] Analyse the role of privilege separation in a component-and-connector design and how it prevents attacks that exploit elevated privileges; give a real-world system with challenges and strategies.

**Role and mechanism [1 mark].** In a C&C view, the components that touch untrusted input (network parsers, decompressors, third-party libraries) are the likeliest to contain exploitable errors, yet a monolithic service runs all of them with the full privilege the service needs for a few operations. **Privilege separation applies least privilege inside the application**: the service becomes a small **privileged monitor** plus one or more **unprivileged slaves**; the slave handles untrusted data and must ask the monitor, over a narrow IPC connector, for anything privileged; the monitor validates each request against its current state and refuses the rest.

**How escalation is prevented [1 mark].** An attacker who exploits an error in the slave inherits only the slave's identity — an unused user ID with an empty (chrooted) root directory — so no secret, key, or protected file is reachable. The remaining avenue, the monitor's interface, is deliberately tiny, state-driven, performs privileged work itself instead of handing out secrets, and terminates the session on any out-of-state request.

**Real system: OpenSSH [1 mark].** The SSH daemon historically needed superuser for the whole session (key-exchange signatures, password database, pseudo-terminals, spawning the user shell). After separation: the network-facing code runs in an unprivileged slave; the monitor answers only informational requests (signatures, password checks), capability requests (pseudo-terminal file descriptors) and the identity change; after authentication the **old slave exports its state and the monitor starts a new slave under the user's ID**. Result: ~two thirds of the code (including third-party crypto and compression libraries) runs without privilege, ~2% of existing lines changed, past vulnerabilities (remote packet-overflow root, post-auth channel bug, library double-free) would all have been contained, and measured login/transfer times showed no significant penalty.

**Challenges → strategies [1 mark].** Identity change has no portable mechanism → export slave state (marshalling + shared memory) and respawn under the user. Monitor interface is an attack surface → small per-state request set, strict validation, terminate on invalid, never export secrets. Platform differences → degrade gracefully where descriptor-passing/shared memory are missing. Residual abuse (resource exhaustion) → per-slave UID, read-only empty root, process/descriptor limits. Liability accepted: the monitor must be correct — it holds the privilege — and DoS inside the slave's rights is not prevented.

### Q3.2 [4 marks — past paper] Two patterns are proposed to you: Secure Logger and Clear Sensitive Information. State their assumed objectives, applicability and trade-offs; then discuss combining them in a payment-processing system with benefits and challenges.

**Secure Logger [1 mark].** From the name: logs must help defenders, not attackers — an attacker who reads the log learns nothing useful for planning attacks, and an attacker who edits it cannot do so unnoticed. Applicability: any system that logs, where log content could guide attacks and where logs are used to detect attacks. Trade-offs: harder for operators to read (need reader/viewer), per-record cost (encrypt/sign), the log store becomes a high-value target and can grow into a SPOF; over-masking destroys diagnostic detail.

**Clear Sensitive Information [1 mark].** Objective: no sensitive data survives in reusable resources (buffers, caches, files) after the program finishes with it, because freeing rarely erases. Applicability: wherever secrets (passwords, keys, card data, tokens) are held. Trade-offs: overwriting costs time; every sensitive item and release point must be found; optimising compilers may elide "useless" overwrites; clearing cannot recall copies already made elsewhere (logs, backups).

**Combined in payment processing [1 mark].** The logger governs the **persistent, widely read path** (audit trail); clearing governs the **transient path** (buffers). One transaction: card number read into buffers → tokenised (token stored, PAN never persisted) → authorisation obtained → **one log event** with timestamp, transaction ID, amount, outcome, token reference and last four digits (no PAN, no track data — matching card-industry rules) → buffers overwritten in a destructor that also runs on error paths → the signed/chained log record ships to a central log server.

**Benefits and challenges [1 mark].** Benefits: both leak paths closed; disputes resolvable from a tamper-evident audit record; compliance aligned. Challenges: debugging must use tokens/IDs instead of raw requests; the logger's own queues are buffers needing clearing; error paths and debug settings are where secrets leak; both add latency and operational cost; the single-logging-interface rule needs discipline because stray writes to stdout bypass everything.

### Q3.3 [6 marks] A web application logs failed logins with full request bodies — including passwords — to a file readable by the whole operations team; the file is also writable by the application account. Diagnose the violations, redesign with the Secure Logger pattern, and name the tactics involved.

**Answer.** *Diagnosis:* two violations — (1) **secrets in logs** (passwords, tokens, card data would land in a file that is copied to collectors/backups/support, so every copy widens the audience); (2) **unprotected store** (application-writable log → an attacker who compromises the app can read it to plan attacks and edit it to hide the intrusion). *Redesign:* (a) one logging interface for the whole system; nothing writes to stdout/stderr or ad-hoc files — an Abstract Factory swaps dev logging for secure logging at release; (b) the **secure logger masks/removes** forbidden fields (OWASP list: session IDs, tokens, passwords, connection strings, keys, cardholder data, sensitive personal data) before storing; (c) each record is **protected** — encrypted per store policy, signed or hash-chained to the previous record so deletion/editing of any record breaks later verification; (d) the store is **append-only**, access limited to the logging subsystem (separate write-only account, separate partition), shipped promptly to a central read-only log server (an attacker with host control can still delete local logs, so shipping is the real defence); (e) events from other trust zones are sanitised (neutralise CR/LF/delimiters) to defeat log injection; (f) authorised readers use the special reader/viewer which verifies and decrypts — a failed verification is an incident. *Tactics:* maintain audit trail (recover category), encrypt data, limit access, verify message integrity — bundled into the pattern; side effect: per-record crypto cost and slower ad-hoc diagnosis (accepted, bounded by sampling debug logs during development only).

### Q3.4 [5 marks] Your payment service parses third-party JSON with a library of uneven quality, decrypts card fields with an HSM-held key, and calls the payment network. Sketch a privilege-separated design and specify the monitor's interface policy.

**Answer.** *Split:* a **slave** process parses requests and runs business logic with no access to the key, a chrooted empty root, and an unused UID; a **monitor** process holds the key/HSM connection and offers only two operations: "decrypt this field for transaction t" and "sign this authorisation". *Interface policy (the marks are here):* (1) the secret never crosses the boundary — the monitor returns decrypted fields to the slave for the current transaction only, never exports the key; (2) the monitor is a **state machine**: it accepts decrypt requests only for transactions currently in progress and refuses others (terminating the session on an invalid request); (3) **rate limits** — a hijacked slave cannot use the interface to decrypt the whole card database (e.g. N fields per minute per session); (4) audit every monitor request via the secure logger (whose store the slave cannot write). *Trace:* slave parses (library bugs hit an unprivileged process) → requests decrypt for txn 4471 → monitor validates state + rate → returns field → slave completes authorisation → monitor signs. *Residual risks:* DoS within the slave (mitigate with process limits and restart supervision) and monitor correctness (keep it tiny and audited). Benefit statement: the exploitable third-party parser now runs at the privilege of "nobody", so the worst case is a crashed session, not a card-database breach.

### Q3.5 [4 marks] Distinguish least privilege, privilege separation and sandboxing; when is each the right response?

**Answer.** *Least privilege* is the **principle**: every program/user operates with the least privilege needed; it applies everywhere and limits damage but does not restructure anything. *Privilege separation* is a **pattern inside one application** that makes least privilege precise when the app legitimately needs a high privilege for a few operations: split into monitor + slaves so the exposed attack surface runs unprivileged and privileged code shrinks to an auditable minimum. *Sandboxing* is **external confinement** of a whole application by the OS/policy engine (restricted token, job object, seccomp-style filters); it cannot know the application's internal state (e.g. "this user just authenticated"), so it must allow everything any authenticated user may do — which is exactly the gap privilege separation closes (the app itself knows the state and confines its own pre-auth phase). Right response: default = least privilege at deployment (run services as unprivileged users); authenticating service with complex parsers = privilege separation; third-party/untrusted code execution = sandbox; the three compose (Chromium: sandboxed renderers + a broker that validates policy).

### Q3.6 [6 marks] A hospital records system must satisfy auditors that access-log evidence cannot be silently altered, while doctors complain that logging slows chart opens. Propose an architecture using the Secure Logger plus tactics, and analyse the performance trade-off.

**Answer.** *Architecture:* chart services call one logging interface; the **secure logger** masks patient-identifiable fields not needed for audit (store user, patient-ID hash, action, timestamp, resource, outcome — not clinical notes); each record is hash-chained (each record's MAC covers content + previous record) and **asynchronously shipped to a central append-only audit store** on separate infrastructure; the application never writes logs directly; readers (auditors) use the verification viewer — any gap or failed chain verification is an alert. *Performance resolution:* logging is moved **off the critical path** — the service appends to an in-memory buffer/queue and returns; a shipper thread batches, signs and forwards (bound the queue; on overflow, drop *verbose* debug logs first, never audit events — prioritise events tactic). Signing cost is batched (sign per batch of records with a running chain hash), cutting per-record latency. *Trade-off analysis:* detection of tampering is delayed by batching seconds-to-minutes (acceptable for audit; a real-time security event path uses a separate immediate channel); a queue outage must fail safe — if the audit path is down, the service degrades (retries locally, alerts ops) rather than silently losing evidence (audit trail must be complete). Doctors see sub-millisecond logging; auditors get a tamper-evident off-host trail — both attributes served by separating the *record* path from the *protect-ship* path.

---

# THEME 4 — Cloud, Integration Strategies, CAP and Failure Management

### Q4.1 [8 marks] An e-learning company hosts exam-prep courses. Traffic is ~10× for 6 weeks around national exams and near-baseline otherwise. The board wants a financial and architectural case for moving from owned servers to a public cloud; include the service-model choice and two cloud-specific risks with mitigations.

**Economic case [3 marks].** Three reasons the cloud wins here: (1) **economies of scale** — the provider's large data centres buy power (15–20% of operating cost) at bulk rates, share racks/switches, get ~30% hardware discounts and staff ~1,000 servers per administrator vs ~150 (fewer admin salaries per server); (2) **utilisation** — the provider co-locates complementary load shapes (the company's exam spike fills other tenants' troughs); (3) **elasticity + measured service** — this workload is *spiky*: owning peak capacity means paying all year for hardware idle 11 months; renting it for 6 weeks converts capital expenditure into per-use cost. The workload pattern (sharp, predictable, recurring peak) and the multi-tenant course-delivery model both support the move.

**Service-model choice [2 marks].** **PaaS** for the application tier (runtime, scaling, managed database — team concentrates on course features; cost: less control of the stack) with **IaaS** only where custom runtimes are needed; **SaaS** for commodity needs (mail, monitoring). Deployment model: public cloud (spiky consumer workload, no compliance barrier) — service model and deployment model are independent axes, chosen separately.

**Risks + mitigations [3 marks].** (1) **Multi-tenant security** — inadvertent information sharing (residual data on reallocated disks; shared tables leaking across tenants) and VM escape/side channels: mitigate with per-tenant keys, row-level tenant filters in one shared data-access layer, patched hypervisor, and isolation levels chosen per data sensitivity. (2) **Auto-scaling response time** — new VMs take minutes to start; a sudden spike degrades service before capacity arrives: mitigate with self-aware applications (queue-depth metrics driving *predictive* scaling before the evening peak), warm pools/pre-warmed images, and bounded queues so overload sheds gracefully. (Also note: availability of the *platform* ≠ availability of *our application* — replicate across availability zones; assume instance failure and design detection/recovery in.)

### Q4.2 [6 marks] The company in Q4.1 also offers a mobile app used on commuter trains. Explain how the app should handle data offline, which operations favour availability and which consistency, and how the backend's 99.95% SLA enters the design.

**Answer.** Offline handling = **store locally, synchronise later** [E5]: the UI reads/writes a local store (SQLite) as the primary source; a sync component transmits the pending-change queue when connectivity returns; transmissions are **idempotent** (safe to repeat), ordered per record, with an explicit **conflict policy** (last-write-wins for progress/notes; server-wins for entitlements; field-level merge where safe). *Availability-favoured operations:* watching downloaded lessons, taking practice tests, note-taking, progress updates — a stale or locally-resolved answer is better than none (AP choice). *Consistency-favoured operations:* purchasing a course, redeeming a voucher, certificate download — a double-charge or duplicate entitlement is unacceptable; these are **refused offline** or routed to a CP path with deduplication keys. *SLA enters the design:* 99.95% = **4.38 hours/year** of permitted downtime — the app must treat backend unavailability as *normal*, not exceptional: everything above works offline by design; reconnects use exponential backoff (no thundering herd when the SLA breach ends); the backend itself replicates across zones because the platform SLA says nothing about our single-instance mistakes.

### Q4.3 [6 marks] Compare file transfer, RPC, messaging, WebSockets and an API gateway as integration strategies for (a) nightly ERP↔data-warehouse exchange, (b) live delivery-driver locations on an operations map, (c) exposing 30 internal microservices to partner developers. Justify each choice.

**Answer.** **(a) File transfer** [D1]: nightly bulk exchange with a legacy ERP is periodic, whole-dataset and needs no instant feedback — files (agreed CSV/XML format) are simple, time-decoupled, restartable at file granularity, and ERP teams understand them; RPC would couple availability windows, and messaging adds infrastructure for no latency gain. **(b) WebSockets**: the map needs **real-time bidirectional push** to browsers — a persistent full-duplex connection avoids per-update HTTP overhead and lets the server push driver positions the moment they arrive; messaging under the hood (queue + fan-out) feeds the WebSocket gateway; REST polling would be wasteful and laggy. **(c) API gateway**: 30 services × partner developers = authentication, rate limiting, versioning, aggregation and protocol translation needed **uniformly at one boundary** — the gateway hides internal topology so services can move/split without breaking partners; direct exposure would scatter policy across 30 services. Cross-cutting justification: choose by *coupling* — file transfer couples least (time, format only), messaging decouples in time with delivery machinery, RPC/REST couple most (both ends up, immediate answer), WebSockets add a long-lived connection's state, and the gateway is a policy chokepoint that itself must be replicated (bottleneck/SPOF mitigation).

### Q4.4 [5 marks] Explain the CAP theorem with the two-node example, classify MongoDB, Redis, Cassandra, CouchDB, and state what a payment system and an activity feed should each give up during a partition.

**Answer.** *Theorem:* a distributed service cannot simultaneously guarantee **Consistency** (all nodes see the same data at the same time), **Availability** (surviving nodes keep operating despite failures) and **Partition tolerance** (operation continues despite network partitions) — **any two, never three**. *Example:* N1 and N2 both hold X=5; a partition separates them; a client writes X=7 at N2; another client reads X at N1. An **AP** system answers 5 (older but available). A **CP** system refuses until the partition heals (consistent but unavailable). A **CA** system would require N1–N2 permanent contact — exactly what a partition denies — so CA is theoretical only. *Classification (course list):* **AP** — CouchDB, Cassandra, ScyllaDB; **CP** — MongoDB, Redis; **CA** — none in practical use. *Payments* give up availability: a partition must **not** let two nodes both approve spending the same balance → CP (prefer refusal + retry over double-spend). *Activity feed* gives up consistency: during a partition, keep serving slightly stale posts → AP + eventual consistency; a missed like is recoverable, a blocked feed is not. Always add: during a partition the real choice is C vs A, since P is not optional across machines.

### Q4.5 [6 marks] A monolithic retail site suffers cascading failures: when the recommendations component hangs, checkout stalls. Diagnose with failure-management principles and prescribe an architecture (tactics named).

**Answer.** *Diagnosis:* the monolith shares one process/resource pool — a hanging recommendations call exhausts worker threads/connections that checkout needs; there is no fault isolation, no timeout discipline, and no graceful degradation: a *non-critical* component's failure becomes a *critical* outage. *Prescription:* (1) **bulkhead** — separate resource pools (connection pools, thread pools) per capability, so recommendation hangs cannot consume checkout's capacity; (2) **timeout + circuit breaker** on every inter-service call — fail fast after N failures, serve a fallback ("popular items") and probe periodically to restore (detect: ping/echo + timestamps; recover: retry with idempotent semantics); (3) **degradation** — the checkout path depends only on cart, pricing and payment; recommendations are optional and served from a distributed cache with a bounded staleness window when the live service is down; (4) **replication + load shedding** — replicate each service (remove SPOF); bound queues and shed excess requests with immediate, honest errors at overload (limit event response); (5) **transactions** where money moves (reserve stock + payment atomic). Each prescription is a named tactic with its side effect acknowledged: circuit breakers can flap under transient faults (hysteresis/half-open probing mitigates); shedding drops some customers by design — the overflow behaviour must be specified and communicated.

### Q4.6 [5 marks] The board asks: "Why not just buy one big server instead of distributing?" Answer with utilisation, single-node limits and CAP reasoning.

**Answer.** Three arguments. (1) **Utilisation and scale economics:** co-locating varied workloads (time-of-day, CPU-heavy beside I/O-heavy) raises utilisation on shared infrastructure that a provider amortises over 100k+ servers — power, hardware discounts (up to 30%), and ~1,000 servers per administrator make per-unit cost lower than any single purchase; elasticity matches spend to demand instead of sizing for a peak. (2) **Single-node limits:** one machine is a hard SPOF (no availability engineering possible — maintenance = downtime) and cannot be partition-tolerant; it also hits practical ceilings (RAM, I/O) exactly where web-scale data lives; replication across machines is the only route to both availability and read scalability. (3) **CAP reasoning:** the moment you add a second machine for availability you accept the CAP trade — two copies can disagree (lose strict consistency) or the system must sometimes refuse (lose availability); a single node avoids the dilemma only by giving up both distribution benefits. The honest middle path for a small system: one node + mirror gives consistency + partition tolerance but not high availability — acceptable when downtime is tolerable and cost dominates. So the answer is workload-dependent: buy big for modest, ACID-centric workloads; distribute when growth, availability and elasticity dominate, and manage the CAP choice consciously.

---

# THEME 5 — Mobile Architecture (EC-3 Q3 territory)

### Q5.1 [5 marks — past paper] An app needs GPS and camera on both Android and iOS. Analyse cross-platform vs native on performance, security, UX, development efficiency, code reuse and platform-specific optimisation; decide with frameworks and a method.

**Assumption:** record location + take photographs on both platforms; limited team/budget.

| Attribute | Effect of choosing cross-platform |
|---|---|
| **Performance** | An extra layer (bridge/web view/engine) sits between app and platform — fine for forms/lists/maps; weakest for continuous camera processing and sustained tracking, where native's direct API calls win |
| **Security** | Secure storage, biometrics, permissions reached via **plug-ins**; each third-party plug-in widens the attack surface; web-in-shell code is easier to tamper with; native has platform key stores, biometric prompts, certificate pinning directly |
| **User experience** | Native gives platform controls/gestures and new OS features immediately; cross-platform is near-native on common screens but may miss conventions |
| **Development efficiency** | One code base + one skill set replace two (Swift/Kotlin) → faster, cheaper delivery |
| **Code reusability** | Business logic and most screens shared; native shares almost nothing |
| **Platform-specific optimisation** | Limited — optimising for one platform needs native modules, reintroducing per-platform code |

**GPS and camera:** a hybrid shell reaches camera, accelerometer, GPS through the platform's native capabilities — basic capture and location are feasible cross-platform.

**Decision:** build screens and logic with a cross-platform tool — **React Native or Xamarin (.NET MAUI)** (alternatives: **Flutter** for framework-rendered UI, **Kotlin Multiplatform** for shared logic with native UI) — with **native modules for the location and camera components**. If the requirement were real-time camera analysis or continuous background tracking, go native for those parts or the whole app. **Method:** build a short **spike** of the GPS+camera feature in the chosen framework on both platforms, measure against the performance target, record the decision, iterate in agile sprints with device testing in the pipeline.

### Q5.2 [5 marks — past paper] Analyse a case where a company implements responsive design in its mobile web app; discuss improvements in page loading time, navigation and overall interaction, and explain the role of responsive design.

**Answer.** *Case:* a newspaper publisher (in the manner of the Boston Globe) replaces desktop-site + reduced mobile-site with **one responsive website** that adapts to screen size and orientation. *Role/mechanism:* **flexible grids** — media queries determine screen size and column widths are percentages of the screen; **flexible images** — image width a percentage of its column; **layout choice** — per screen size, columns sit side by side or stack (CSS/HTML shrink, hide or move content).

| Factor | Improvement | Responsive mechanism |
|---|---|---|
| Page loading time | Images scale to the column instead of fixed desktop size; secondary content hidden/moved on small screens; no separate mobile site or redirect to load; one code base cached once | Flexible images; hide/move content; single website |
| Navigation | Multi-column desktop page becomes one vertical column; compact menu; large tap targets; no sideways scrolling or pinch-zoom | Flexible grid; layout chosen by screen size |
| Overall interaction | Text readable without zooming; controls sized for fingers; layout follows rotation between portrait and landscape | Media queries on size and orientation |

*Outcome:* readers see the **full site content** (not a reduced version) and the publisher maintains **one site**. *(Strengthen with mechanics:* viewport meta tag, breakpoints, mobile-first base style, `srcset` responsive images, ~44–48 unit touch targets. *Honest limit:* CSS-hiding content does not stop the browser downloading it — serve device-sized images or you save layout effort but not bandwidth.)*

### Q5.3 [6 marks] A courier-delivery company's drivers work in basements and industrial estates with no signal. Design the app architecture around the store-locally pattern, including the sync design and conflict policy.

**Assumption:** Android devices; tasks = view assigned packages, mark delivered (photo + timestamp), sync to central server.

**Architecture (map to Android components [E6]).** *Activity* = package list screen + "mark delivered" action; *Content provider (SQLite)* = the local store — the **only** thing the UI reads/writes (application availability no longer equals network availability); *Service* = the synchroniser, started on connectivity events; *Broadcast receiver* = listens for connectivity-restored broadcasts (and server push when online) to trigger sync; photo files stored in app storage, referenced by the queue. This is exactly the courier decomposition of the course (Activity → provider → Service → REST API), now made offline-first.

**Sync design.** A **pending-changes queue** (SQLite table) records every delivery event in order; the Service drains it when online; each message carries a **client-generated UUID** so the server can **deduplicate** (idempotent receive — a connection may die after the server applied the event but before the ack); exponential backoff; server responses update local status (sent/acknowledged). Server assigns official delivery records; the app updates its mirror.

**Conflict policy.** Two devices cannot legally deliver the same package (assignment is per driver), so most conflicts are *staleness*, not writes: last-write-wins with server authority for status; reassignments while offline (dispatcher reassigns a package) are resolved **server-wins** on next sync — the app shows an updated assignment and drops the stale event with an operator-visible note. Payments/POD signatures that cannot tolerate conflicts are **refused offline** (warn the driver) rather than queued — the CAP choice per operation: availability for views/notes, consistency for money/signatures.

**Side effects named:** local store is a security surface (device theft) → encrypt at rest, authenticate users, keep nothing but operational data locally; queue growth under long outages → cap queue size with visible "unsent deliveries" counter and oldest-first priority.

### Q5.4 [4 marks] Decompose the Uber-like ride-hailing app into Android components and justify each mapping; state where intents are directed vs broadcast.

**Answer.** Per the course decomposition: **Activity** — the book-a-cab UI (map, pickup/drop, fare quote); **Service** — during the journey, the service sends the driver's GPS to the backend continuously (background work must survive the user switching apps); **Broadcast receiver** — receives the *driver's location updates pushed from the backend* (delivered as system/app broadcasts when the app is backgrounded) and forwards them to the UI for map refresh; **Database/content provider** — configuration and user data (stored credentials, recent trips cache). *Intents:* **directed** — Activity → Service ("start location updates"), Activity ↔ Activity (booking → ride-in-progress screens), Service → Activity updates; **broadcast** — driver-location events (any registered receiver may act: the map UI, a notification builder) and connectivity/system events. Justification: each component keeps to one role — UI, background processing, event reaction, storage — which keeps the app within mobile's "few layers" constraint and lets the location Service live independently of the Activity's lifecycle.

### Q5.5 [5 marks] The team proposes a separate mobile-optimised website (m.site) instead of responsive design. Compare the two on cost, UX, SEO/maintenance and performance; recommend.

**Answer.** *m.site:* a **second product** — separate template set, reduced feature menu, separate URL; pros: can be tuned purely for small screens (lighter pages, explicit mobile UX); cons: feature divergence (users notice "the mobile site has less"), double maintenance and QA, split SEO/link equity, redirect latency, and every new feature ships twice. *Responsive:* one code base, full content on all devices, one URL (SEO-friendly), one maintenance stream; cons: heavier initial engineering (grids, media queries, responsive images), and careless implementations ship desktop-sized payloads to phones (CSS-hidden content still downloads — must use responsive images/conditional loading). *Performance:* responsive can match or beat m.site **if** images are device-sized (`srcset`) and breakpoints trim payloads; a lazy responsive port that only restyles is worse than a good m.site. *Recommendation:* responsive for a content/transaction site with one feature set (the newspaper case: full content everywhere, one cached code base, no redirect); choose m.site only when mobile and desktop are genuinely different products (e.g. a stripped kiosk experience) — and then budget for the permanent second product. Decision condition: if analytics show mobile users need the same content and flows (typical), responsive wins on total cost of change.

### Q5.6 [4 marks] Where does a cross-platform framework sit in the mobile platform stack, and what does that imply for hardware access?

**Answer.** The platform stack is Hardware → Kernel & drivers → Core → Framework (split native/web) → Applications, with the **HAL** exposing device capabilities to the framework layer. A cross-platform framework inserts itself **at the framework/application layer**: it ships its own rendering/runtime (Flutter draws its own UI; React Native instantiates the platform's native widgets via a JS bridge; Cordova/Ionic run web code in a shell) and reaches hardware **through the platform's native framework/HAL APIs** — that is why GPS, camera and accelerometer *work* cross-platform (the capability path exists), while sustained hardware work pays the bridge/hand-off overhead on every call and often waits for plug-in updates when the OS adds new features (the framework layer lags the platform). Implication: design hardware-heavy paths as **native modules** that the shared code calls, keeping the per-platform surface minimal and audited.

---

# THEME 6 — Big Data, Hadoop/MapReduce, NoSQL (EC-3 Q2c territory and beyond)

### Q6.1 [2 marks — past paper] A distributed system must process a massive dataset of user logs from a social-media platform to extract behaviour insights. Explain how you would apply Map-Reduce for scalable processing, and discuss the trade-offs of batch Map-Reduce versus real-time parallel processing.

**Application [1 mark].** Logs are split into file portions across nodes (3Vs: volume → distribution; variety of log formats → parse in map; velocity → batch windows). Each **map** instance, stateless and independent, reads its portion, filters irrelevant records and emits `<key, value>` pairs — e.g. `<user, 1>` or `<hashtag, 1>` per post/view/like. The **infrastructure** deploys instances, partitions and merges pairs by key (shuffle = the distributed sort), and recovers from node failure. Each **reduce** instance combines the values for its keys into totals — engagement per user, per topic — which are the insights. Scalability follows because adding nodes adds map capacity, maps never wait on each other, and map runs where the data lives (only compact pairs cross the network).

**Trade-off [1 mark].** *Batch MapReduce:* high throughput over the **whole historical dataset**, simple fault tolerance (re-run a failed task; data is in files), cheap per GB; but results appear only **after a job finishes** — insight is delayed; overhead is wasteful on small data; uneven partitions create stragglers; multi-stage reduces are awkward. *Real-time (stream) processing:* results within **seconds** of each event (trending topics, abuse detection), but weaker ordering/exactly-once guarantees, complex state handling (checkpoints), and higher always-on operating cost. *Recommendation:* batch for daily behavioural analytics on full history; add a streaming path (or lambda design — batch recomputes accurate results, stream serves fast approximations) if the product needs seconds-level reaction.

### Q6.2 [6 marks] Walk through the hashtag-counting MapReduce job on three shards; verify the result; explain where a combiner helps.

**Answer.** *Given:* shard A = "#Tech is fun #AI", "#Tech news", "#Sports night"; shard B = "#AI rocks", "#Tech #AI", "#Music live"; shard C = "#Sports #Tech", "#Music", "#Tech".
**Step 1 — Map** (each mapper scans its shard, emits one pair per occurrence): Mapper A: (#Tech,1), (#AI,1), (#Tech,1), (#Sports,1). Mapper B: (#AI,1), (#Tech,1), (#AI,1), (#Music,1). Mapper C: (#Sports,1), (#Tech,1), (#Music,1), (#Tech,1). (12 pairs total.)
**Step 2 — Shuffle and sort:** the framework groups all pairs by key and routes each key to one reducer; with two reducers assume #Tech and #Music → reducer 1, #AI and #Sports → reducer 2. Reducer 1 receives #Tech:[1,1,1,1,1], #Music:[1,1]; reducer 2 receives #AI:[1,1,1], #Sports:[1,1].
**Step 3 — Reduce:** sum each list → #Tech = 5, #Music = 2, #AI = 3, #Sports = 2.
**Verification:** 5+2+3+2 = 12 = pairs emitted — no occurrence lost or double-counted. **Combiner:** an optional local reduce on each mapper *before* the shuffle would compress mapper A's two (#Tech,1) into (#Tech,2), cutting network traffic; it is safe here because summation is associative, and it does not change the final result.

### Q6.3 [6 marks] A telecom wants to reprice calling plans from one month of call records; the fraud team wants to block suspicious transactions *during* the call/payment. Which platforms for each, and why can one warehouse not serve both?

**Answer.** *Repricing* = whole-month batch analysis, minutes-to-hours latency acceptable, volume large → **Hadoop/MapReduce over HDFS** (or Spark): cheap per GB, fault tolerance by task re-run, the data already sits in files; metrics like per-plan averages follow the weighted-average map/reduce shape (carry counts!). *Fraud blocking* = a decision **inside the transaction window** → real-time stream processing/online scoring (seconds or less); Hadoop's job model cannot answer before approval, and its batch latency is the wrong shape entirely. *Why one warehouse cannot serve both:* the platforms optimise opposite corners of the same triangle — batch optimises **throughput on stored data** (files, re-run on failure, high latency); streaming optimises **latency on arrival** (state in memory, checkpoints, always-on cost); an MPP/SQL warehouse serves fast declarative queries on structured data but is neither cheap at petabyte scan scale nor event-driven. Hence the standard **lambda-ish architecture**: a stream path scores events live and writes features; a batch path recomputes accurate models/summaries nightly over all history; the serving tier merges both (stream = fast approximation, batch = correction). Name the constraint that forces the split: Hadoop's input-exists-as-files + stateless-maps constraints [B11] are incompatible with event-by-event decisions.

### Q6.4 [6 marks] "NoSQL is just a cache; we should stay on MySQL." Evaluate for (a) a product catalogue with rapidly changing attributes, (b) a general ledger, (c) IoT telemetry. Use NoSQL strengths/omissions and CAP.

**Answer.** The claim conflates "NoSQL" with "non-durable" — NoSQL stores are replicated, persistent databases whose *omissions* (ACID, joins) are deliberate trade-offs for scale, speed and availability [F5].
- **(a) Product catalogue:** attributes change per iteration (new fields per campaign); relational forces schema migrations per change; **document store** accepts new fields at once and keeps one product = one document (no joins to render a page). Strengths used: flexibility, simplicity; omission (no ACID) harmless — a product row is independent. NoSQL justified.
- **(b) General ledger:** debit+credit must be one atomic, isolated unit; joins for reporting are mature in SQL. Missing ACID rules a typical NoSQL store out — **stay relational** (scale up/replicate with CP semantics), or use NoSQL only for derived, rebuildable aggregates.
- **(c) IoT telemetry:** huge write volume, time-ordered, queried by column ranges; **column-family store** (Cassandra/HBase) shards writes across nodes, AP availability during partitions (a sensor drop must not stall the fleet), and analytics read few columns — the perfect fit; ledger-grade transactions are not needed.
*CAP framing:* catalogue/telemetry tolerate eventual consistency (AP); the ledger must not (CP/relational). Conclusion: "stay on MySQL" is right for the ledger and wrong for the catalogue and telemetry — the decision is per workload, by access pattern and consistency need, not by fashion.

### Q6.5 [5 marks] Name and characterise the four NoSQL data models, each with a fitting workload; then state the two design risks NoSQL introduces.

**Answer.**
1. **Document** (MongoDB, CouchDB): self-describing JSON/BSON documents addressed by key; nested records; suits *read-one-entity* workloads — user profiles, product pages, content management.
2. **Key-value** (Redis, Riak): opaque value per key, hash-based access, value never interpreted; suits *massive simple lookups* — sessions, caches, shopping carts.
3. **Column-oriented/column-family** (Cassandra, HBase — Bigtable-style): values of one field of many records stored together; suits *analytics over few columns, time-series, write-heavy logs*.
4. **Graph** (Neo4j): nodes + directed edges with properties; suits *relationship traversal* — social networks, recommendations, fraud rings.
**Two design risks:** (1) **shard-key choice** — a key concentrating popular items creates a hot spot no added node relieves; choose keys that spread load evenly; (2) **application-side responsibilities** — with no transactions/joins, the *application* must detect interference between concurrent writers, spell names consistently (no schema guard), and assemble joined data itself (or denormalise read-together data into one record); plus eventual consistency must be surfaced honestly to users where it is visible.

### Q6.6 [5 marks] Give the HDFS write-path (sunny-day) in steps and explain how the design delivers fault tolerance; why are millions of tiny files a problem?

**Answer.** *Write path (8 steps):* (1) application writes as to any file system (unaware of HDFS); (2) client code buffers to a block; (3) client asks the **NameNode** to write a new block; (4) NameNode returns **three DataNodes**; (5) client sends the block to the first DataNode, telling it the other two replicas; (6) the copies travel as a **pipeline** — DN1 writes and forwards to DN2, DN2 to DN3; (7) each DataNode reports completion to the client; (8) the client **commits** the write to the NameNode after all three acknowledge. *Fault tolerance:* blocks replicated (default 3) so a node's death loses nothing; **heartbeats** let the NameNode detect dead nodes; metadata in NameNode memory identifies surviving copies so lost blocks are **re-replicated** automatically (self-healing); a rebalancer evens distribution. *Tiny-file problem:* **all metadata lives in NameNode memory** — every file, block and replica is an in-memory record; millions of small files exhaust NameNode RAM long before storage fills (each 1 KB file still costs a full block + metadata). Mitigations to name: sequence files/concatenation, HAR archives, or moving small-object workloads to object stores — the architectural lesson is that *metadata placement* is as decision-critical as data placement.

---

# THEME 7 — Economics: CBAM, Utility Curves, Opportunity Cost (EC-3 Q4 territory)

### Q7.1 [5 marks — past paper] You are the lead developer choosing between Strategy S (prioritises scalability, increases development time) and Strategy F (accelerates development, may compromise scalability). Analyse with opportunity costs, stakeholder factors, and decide under limited resources and a tight timeline.

**Opportunity-cost framing [1 mark].** Cost each strategy as its outlay **plus the best alternative it forgoes**: S's opportunity cost = the features/earlier release the time would have bought; F's opportunity cost = the scalability utility surrendered and the rework needed if load grows. Sunk costs are excluded.

**Analysis [1 mark].** Express as scenarios: a scalability scenario (peak users supported) and a time-to-market scenario (release date). Elicit worst/current/desired/best responses; stakeholders assign utilities 0–100; weight scenarios by vote (100 votes each stakeholder; top scenario W=1.0). Benefit per strategy `B = Σ(b×W)` with negative gains counted (F sacrifices scalability utility; S sacrifices schedule utility). Estimate costs relatively ("if S costs x, F ≈ 0.6x") and compare `VFC = B/C` — the higher ratio wins, not the larger benefit.

**Stakeholders [1 mark].** Developers — workload, code quality, rework; Management — deadline, budget, market timing; End-users — response and availability under load. Each group's votes encode their priorities; the analysis structures (not replaces) their decision.

**Factors under limited resources [1 mark].** Probability/timing of growth (growth within months favours S); cost of adding scalability later (cheap if the design stays modular — favours F); penalty for a missed deadline; reversibility of the choice; stakeholder risk tolerance; **the hybrid** — F plus cheap scalability-enabling tactics (stateless, separable components) that keep the later door open.

**Conclusion [1 mark].** With uncertain demand and a firm deadline, **F with a cheap scalability path** likely has the better VFC; with certain imminent growth, the forgone scalability is the larger loss and S wins. The numbers are checked against intuition (CBAM step 9) and the decision rests with the stakeholders.

### Q7.2 [5 marks — past paper] A system must balance security, scalability and cost. Describe how to evaluate trade-offs between architectural strategies and explain how utility-response curves inform the evaluation, with specific examples.

**Method [2 marks].** Write a scenario per concern with a response measure — security (share of attacks blocked), scalability (concurrent users served), cost (spend). Elicit worst/current/desired/best levels; stakeholders assign utilities 0–100; vote-weights per scenario; each candidate strategy maps to expected responses; interpolate utilities; compute `B = Σ(b×W)`; rank by `VFC = B/C`; select within budget; confirm against intuition (step 9).

**Role of the curves [2 marks].** The curve converts each attribute's response into the common currency of utility, so unlike attributes compare. Shapes matter: **security is S-shaped** — little value until a compliance threshold, rapid gain in the band, saturation above; **scalability shows diminishing returns** — the first capacity increase matters most; **cost is a falling straight line** — each unit of spend lowers utility equally. Hence: spending beyond the security threshold or beyond needed capacity is poor value; falling short of the security threshold is expensive (you sit on the flat zero part).

**Example (illustrative figures) [1 mark].** Strategy A = managed identity service with encryption + auto-scaling: security 50→90 (gain 40), scalability 40→85 (gain 45), weights security 1.0/scalability 0.8 → B = 40×1 + 45×0.8 = **76**, cost 200, VFC 0.38. Strategy B = lean in-house gateway: gains 20 and 35 → B = 20 + 28 = **48**, cost 80, VFC **0.60**. A has the larger benefit, B the better VFC — but B's security utility (70) may sit below the S-curve's threshold, so step 9 may favour A, or **B plus a targeted security improvement**. The curves, not raw numbers alone, drive that reading.

### Q7.3 [6 marks] Perform the CBAM arithmetic: two scenarios, two strategies. S1 (download response time): worst 10 s/10, current 4 s/40, desired 1.5 s/80, best 0.5 s/90. S2 (availability): worst 99.0%/10, current 99.5%/40, desired 99.9%/85, best 99.99%/95. Weights: S1 = 1.0, S2 = 0.6. Strategy A (CDN + auto-scaling) expects 2.0 s and 99.7%, cost 30 person-months. Strategy B (geo-replication) expects 3.0 s and 99.9%, cost 50. Budget = 60. Compute utilities by interpolation, benefits, VFCs; select; state what step 9 does.

**(i) Interpolation [1 mark].** S1 for A: expected 2.0 s lies between desired (1.5, 80) and current (4, 40):
`y = 80 + (40−80)×(2.0−1.5)/(4.0−1.5) = 80 − 40×0.2 = 72`.
S1 for B: expected 3.0 s: `y = 80 + (40−80)×(3.0−1.5)/(2.5) = 80 − 24 = 56`.
S2 for A: expected 99.7% lies between current (99.5, 40) and desired (99.9, 85):
`y = 40 + (85−40)×(99.7−99.5)/(99.9−99.5) = 40 + 45×0.5 = 62.5`.
S2 for B: expected 99.9% = desired level → utility **85** exactly.

**(ii) Benefits [1.5 marks].** Gains: A: S1 72−40 = 32; S2 62.5−40 = 22.5 → B_A = 32×1.0 + 22.5×0.6 = 32 + 13.5 = **45.5**. B: S1 56−40 = 16; S2 85−40 = 45 → B_B = 16×1.0 + 45×0.6 = 16 + 27 = **43**.

**(iii) VFC and selection [1 mark].** VFC_A = 45.5/30 = **1.52**; VFC_B = 43/50 = **0.86**. Rank: A first, B second. Budget 60: take A (30, remaining 30) → B costs 50 → does not fit; select **A alone**, delivering 45.5 benefit points. (If the budget were 80, both would fit.)

**(iv) Step 9 [0.5 mark].** Confirm results with intuition: check the chosen set against business goals — here, does an exam-prep platform's goal (serve students on result eve) really accept 3 s downloads and 99.7% availability from A? If stakeholders feel the availability risk contradicts a stated business goal (exams fortnight), iterate the analysis (re-refine S2, or re-price B); the numbers support, not replace, this check.

### Q7.4 [4 marks] Explain what the CBAM delivers beyond a ranking, and three limits visible in its real-world application.

**Answer.** *Delivers:* (1) the VFC ordering of architectural strategies within budget/schedule; (2) a **structural social benefit** — architecture discussions are normally unstructured (requirements and strategies mixed, stimuli and response goals unarticulated); CBAM forces scenarios to be clarified in advance, utilities assigned to specific response goals, and priorities made explicit — which itself clarifies requirements; (3) **utility as the common coin** across unlike attributes (seconds vs % vs person-days), enabling honest trade-off conversation instead of shouting matches. *Limits (NASA ECS case):* inputs are stakeholder judgements (the ECS team even replaced individual voting with consensus) — output is only as good as that consensus; the ranking is **sensitive to utility values** (small utility changes reorder strategies); and the numbers **support but do not replace** step 9 — the case validated its ranking against the stakeholders' intuition, and inconsistencies in the source tables (utility readings, ROI naming) show real analyses need cross-checking.

### Q7.5 [4 marks] The CTO says: "CBAM is too heavy for us; we'll just decide by gut." When is CBAM proportionate, what is its lightweight form, and what does the gut miss?

**Answer.** *Proportionate:* CBAM is reserved for the **most basic, strategy-setting decisions** — the ones whose cost/benefit spans multiple quality attributes and whose reversal is expensive (platform choices, redundancy levels, security architecture). It is disproportionate for fine-grained design choices (a cache TTL) where elicitation cost exceeds decision value. *Lightweight form:* the same skeleton with fewer cycles — collate the handful of live scenarios, refine each to four response levels, one round of votes, interpolate utilities for the two or three candidate strategies, divide by relative costs, and sanity-check; a half-day workshop rather than a multi-phase study. *What gut misses:* (1) **comparability** — gut compares "faster" with "safer" without a common currency; utility curves expose where each marginal unit is actually valued (diminishing returns, thresholds); (2) **the ratio discipline** — stakeholders naturally chase the largest benefit; VFC exposes the option that delivers most value per cost (the NASA strategies with modest benefit but tiny cost out-ranked glamorous ones); (3) **accountability and reuse** — documented scenarios, weights and utilities make the decision reviewable and reusable next cycle, and disagreements become parameter disputes (votes, utilities) rather than personality disputes.

---

# THEME EXTENSIONS — More Scenario Questions with Full Model Answers

### X1 [8 marks] Layers in practice — a tax-filing web application has three identical app servers behind a load balancer. Explain, per layer, the techniques that make each of these work: (1) a citizen's partly completed payment survives when successive requests hit different servers; (2) a five-step assessment workflow (some steps manual) whose order changes when rules change; (3) a search page vulnerable to SQL injection.

**Answer.** *(1) Session/state — business + data layers.* Make the **business tier stateless** and move session state out of any single server: store the in-progress payment in the **shared data tier** (DB or distributed cache keyed by session ID) — any server can resume it. Alternative (sticky sessions) couples the load balancer to state and breaks on server loss; the stateless-and-shared design also serves scale-out. Tactic: *maintain multiple copies* of the data with the cache bounded by a staleness window for non-critical state; payments read from the DB of record.
*(2) Workflow — business layer.* Model the assessment as a **workflow/work-process component**: steps are declarative entries (id, actor, guard, action) interpreted by an engine, not hard-coded call chains; manual steps become tasks in a work queue assigned to officers. When rules change the *order*, you edit workflow configuration, not code — *defer binding (configuration time)* + *encapsulation* of the process. Each step's transitions are auditable for the tax authority.
*(3) Search — data layer.* Replace SQL-string concatenation with **parameterised queries/prepared statements** in the DAO/repository layer so typed input can never become SQL; validate and escape at the presentation boundary (defence in depth: *validate inputs* tactic); least-privilege DB account for the search path so even an injected query cannot read other tables. Closing line: each fix lives in the layer that owns the concern — state in business/data, process in business, queries in data — which is exactly the modifiability argument for layering.

### X2 [6 marks] A hospital information system utility tree has leaves: "blood-bank donor product added in 2 person-months (M,M)", "intrusion resisted and reported within 90 s (H,M)", "24/7 patient access (L,L)", "patient update at peak < 0.75 s (H,M)". Which do you analyse first in an ATAM, and why? What findings might the security leaf produce?

**Answer.** Analyse **(H, M) leaves first** — high business importance, medium architectural difficulty: the 90-second intrusion detection and the 0.75 s peak update; the (M,M) extensibility leaf follows; (L,L) is never analysed (ATAM is deliberately not exhaustive — it spends effort where importance and difficulty intersect, which is what "determines ATAM's focus"). *Security leaf findings (sketch):* decisions — DMZ + firewall, authenticated sessions, audit logging of record reads; a **risk**: "audit store on the same host as the DB — an attacker with DB rights can edit evidence"; a **nonrisk**: "parameterised queries throughout the data layer, verified in code review"; a **sensitivity point**: "audit-log verbosity setting affects intrusion-detection latency"; a **tradeoff point**: "session timeout — shorter improves security, longer improves usability for clinicians mid-procedure." Produce at least one of each finding type with the reason — that is the vocabulary the examiner wants.

### X3 [5 marks] Your client insists a P2P design for a document-collaboration product will remove all servers and all risk. Give the honest counter-analysis.

**Answer.** What P2P genuinely buys: no central SPOF, scale with membership, censorship resistance. What the scenario will hit: (1) **consistency** — concurrent edits on disconnected peers need protocol-level resolution (CRDT-like rules or conflict copies); (2) **security** — no centre to authenticate against, key distribution and revocation are hard; a compromised peer is hard to expel; (3) **availability of structure** — search/indexing requires special peers (supernodes) or flooding, which reintroduces hierarchy; (4) **management** — backup, recovery and compliance (audit trails) are distributed problems; (5) small networks underperform — with few peers, redundancy is thin. Recommendation: a **hybrid** — P2P-style direct sync between a user's own devices, plus a light coordination/backup service (identity, key directory, relay when NATs block direct links). The honest line: P2P trades *centre risks* for *protocol complexity*; with legal/compliance needs, pure P2P usually loses.

### X4 [6 marks] A streaming service must show a personalised home page within 2 s at peak (1 M concurrent). Name the dominant attribute, write the scenario, choose tactics and patterns, and resolve one side effect.

**Answer.** *Attribute + scenario (six-part):* performance; source = 1 M concurrent users; stimulus = home-page request; artifact = personalisation pipeline; environment = evening peak; response = render personalised page; measure = **≤ 2 s at 99th percentile**. *Tactics:* **maintain multiple copies of data** (pre-computed home-page fragments in a distributed cache), **increase resources** (auto-scaling read tier), **bound queue sizes** (shed beyond capacity with cached fallback), scheduling policy for personalisation jobs. *Patterns:* API gateway → stateless page service → cache → recommendation service (itself fed by a MapReduce batch pipeline + stream updates); CDN for static assets. *Mapping:* catalogue/profile services are providers; cache is the intermediary absorbing peak reads. *Side effect + resolution:* pre-computed fragments go stale when the user's tastes change mid-session → bounded staleness window (e.g. 10 min) + immediate invalidation on explicit user actions; cache hot keys → consistent hashing and key salting. Decision: 2 s at peak is achievable with cache-first serving; the recommendation service sits behind the cache deadline so a slow model never blocks the page.

### X5 [5 marks] Distinguish scalability and performance with a scenario each, and name two tactics that serve both and one that serves only performance.

**Answer.** *Performance* = how fast/leniently the system responds **at the current load** (latency, throughput); *scalability* = how gracefully response holds **as load grows** (adding resources should add capacity roughly linearly). Scenario (performance): "checkout page renders in 1.8 s at today's 5,000 concurrent users". Scenario (scalability): "at 50,000 concurrent users, checkout still renders in ≤ 2.5 s after adding 10 app-server instances". Serves both: *maintain multiple copies (of data/computation)* — cuts per-request work and spreads load; *increase resources* with stateless components — improves latency now and capacity later. Only performance: *reduce computational overhead* (algorithmic optimisation) — makes each request cheaper but does nothing about capacity growth. The distinction matters commercially: performance fixes today's pain; scalability defers the next one.

### X6 [6 marks] A bank's card-authorisation service (CP, relational) must also emit an analytics feed. Design the split between the transactional path and the analytics path, and justify the storage and integration choices.

**Answer.** *Transactional path:* relational DB, **CP semantics** — a partition refuses rather than risks double-spend; ACID transactions across account rows; authorisation served from the DB of record with parameterised queries; availability engineered by synchronous replication within a region (consistent copies) + failover. *Analytics path:* the service **publishes authorisation events** (asynchronous messaging, at-least-once, per-account ordering) to a queue; a consumer lands events in **HDFS/NoSQL** (column-family for time-series; document for case files) where **MapReduce/Spark** batch jobs compute fraud models, spend patterns, plan analytics — eventually consistent by design. *Justification:* the two paths have opposite requirements — money movement needs immediate consistency (CAP: CP; NoSQL's missing ACID disqualifies it here), analytics needs cheap petabyte scans and tolerates minutes of lag (batch strengths); the queue decouples them so analytics load can never slow authorisation (bulkhead), and replaying the queue rebuilds the warehouse (auditability). Side effect named: event lag — bounded by monitoring queue depth; alerts if lag exceeds the fraud model's freshness SLA.

### X7 [5 marks] "Our new microservice will call three other services per request." What failure-handling design must accompany it, and which tactic does each piece implement?

**Answer.** Per dependency: (1) **timeout on every call** — never wait unbounded (fault detection: condition monitoring by deadline); (2) **circuit breaker** — after N consecutive failures, fail fast to a fallback and probe periodically (recover: retry/shadow; prevents cascading saturation); (3) **retry with idempotency** — only for idempotent operations, with exponential backoff + jitter (recover: retry); (4) **fallback** — degraded but useful response (stale cache, default list) so the request still completes (graceful degradation); (5) **bulkhead** — separate connection/thread pools per dependency so a hung dependency consumes only its own pool (prevents ripple: restrict resource coupling); (6) **correlation IDs + audit trail** for diagnosis across services. Close: with three downstream calls the failure probability compounds (if each is 99.5%, the chain succeeds ≈ 98.5% before mitigation) — hence the design treats partial failure as the *normal* case, not the exception.

### X8 [4 marks] Explain "begin with service definition" as cloud strategy, and connect it to the automation-levels figure.

**Answer.** Recommendation order: (1) **identify the cloud-related IT services** to offer/procure (e.g. "billable compute for season spikes", "managed customer DB"); (2) **document the internal processes** those services touch (provisioning, change, billing); (3) **map applications/workloads** onto the services — only then choose technologies (IaaS/PaaS/serverless). Rationale: the automation-levels figure shows business value rising from **IT task automation** (provisioning, patching — functionally specific, fixed sequence) through **IT service automation** to **business service automation** (cross-silo, situational, coordination-oriented); cloud services span all three, so defining them as *services tied to business outcomes* captures value that task-level cloud adoption (lifting VMs 1:1) misses. An answer that starts "we need Kubernetes" has skipped the strategy the material teaches.

---

# THEME 8 — Evaluation, Utility Trees and Documentation (EC-3 comprehensive-coverage questions)

### Y1 [8 marks] A ministry will evaluate the architecture of a new tax platform before build. Design the evaluation: participants, the exact ATAM phases/steps you would run in a 3-day window, the utility-tree fragment you would expect, and the outputs you would demand.

**Answer.** *Participants:* evaluation team (team leader, evaluation leader, scenario scribe, proceedings scribe, quality questioners) + project decision makers (manager, customer, architect) + (day 3) wider stakeholders.
*3-day plan (compressed ATAM):* **Day 1 — Phase 1 steps 1–6:** present ATAM; present business drivers (functions, constraints, goals, ASRs); architect presents the architecture (views, patterns, tactics); evaluation team identifies approaches (layers, broker, pub-sub, encryption, replication…); **build the utility tree**; analyse the top-ranked scenarios, tagging risks/nonrisks/sensitivity/tradeoff points. **Day 2 — steps 7–8 with stakeholders:** brainstorm scenarios from the stakeholder floor, prioritise by votes (100 each), analyse the newly high-ranked scenarios against the architecture. **Day 3 — step 9 + follow-up:** present results: prioritised scenarios, risks → mitigation plan, risk themes, decision-to-requirement mapping; agree owners and dates.
*Utility-tree fragment expected (two attributes suffice):* Performance → transaction latency → "assessment submission < 2 s at peak (H,M)"; Security → confidentiality → "citizen record visible only to the owning assessor (H,M)"; Modifiability → new tax-rule change → "rule change deployed in ≤ 1 working day, no code (H,L)". Leaf ranks pick the analysis order.
*Outputs demanded:* the seven ATAM artefacts — architecture presentation; articulated business goals; prioritised scenarios in six-part form; risks/nonrisks; risk themes; decision→requirement mapping; sensitivity/tradeoff points — plus the mitigation plan with owners. *Honesty note:* ATAM analyses only what the top scenarios touch — say so rather than promising exhaustive certification.

### Y2 [5 marks] During that evaluation, two decisions are recorded: (1) a warm-standby database synchronised every 60 s; (2) a heartbeat from the monitor to the external payment interface every second. Classify each as risk, nonrisk, sensitivity or tradeoff point, with reasons.

**Answer.** *(1) Warm standby @60 s:* a **tradeoff point** and a **risk**. Tradeoff: recovery-point objective (staleness ≤ 60 s) is bought with availability/cost — shorter sync windows improve data safety but add overhead (performance/complexity). Risk: a failover loses up to 60 s of confirmed transactions; whether that is acceptable is unproven until a failover test under peak load is run — the decision is potentially problematic and not yet validated. *(2) Heartbeat @1 s:* a **tradeoff point** and a **sensitivity point**. Sensitivity: detection time of a dead gateway is directly governed by the heartbeat interval — it is the decision that determines that availability response. Tradeoff: faster detection costs bandwidth/processing every second (performance), and a flood of pings is also a security surface (ping-flood response). *Vocabulary check:* sensitivity = affects one attribute markedly; tradeoff = affects two+ oppositely; risk = unproven problematic decision; a decision can legitimately carry two tags — the reason line is what earns the mark.

### Y3 [6 marks] A 12-year-old property-tax system must be replaced but has no architecture documents. Explain how architecture reconstruction applies: purposes, phases, what you extract, and how the result feeds migration planning.

**Answer.** *Purposes (all three apply here):* understand an undocumented system; **migrate** it to the new technology; identify **reusable components** (e.g., the tax-calculation library, audit routine). *Phases:* (1) **raw view extraction** from source code, execution traces and build scripts — classes, files used, caller–callee relations, global data accessed (uninterpreted facts); (2) **database construction** — common format; (3) **view fusion** — merge static views (who-calls-whom), dynamic views (traces of the nightly batch), and **expert grouping** (senior maintainers name the functional areas) into consolidated views; (4) **architecture analysis** — validate fused elements against rules, display, and **iterate until trustworthy**. *Feeding migration:* the fused view shows which clusters are self-contained enough to migrate first, which modules bypass the layered design (drift — and therefore refactoring risk), which routines are reused as-is (the audit routine), and which shared data stores make coupling expensive. The reconstruction is also the **baseline for conformance rules** in the new system: the same extraction runs on the new code in CI to detect drift early. Limit to name: reconstruction recovers *what is*, not intent — the rationale still needs the architects who remember it; interview them before they leave.

### Y4 [6 marks] The team asks: "Why document views at all if the code exists?" Give the documentation argument, the view template you would enforce, and one quality-attribute view for this tax platform.

**Answer.** *Argument:* the architecture exists in the system, but only documentation makes it **communicable, analysable and checkable** — stakeholders (developers, testers, operators, security, new joiners) ask different questions, each answered by a specific view; without views, every such question costs archaeology, and conformance cannot be checked because "what is supposed to be" is unstated. *View template:* primary presentation (the diagram) → element catalogue (elements + properties: responsibility, interface, protocol) → context (what lies outside) → variability guide (what can vary per deployment) → rationale (why, and rejected alternatives). *Quality-attribute view for the tax platform:* a **security view** — trust boundaries (internet → DMZ → internal), authentication flows (citizen login via identity provider; assessor roles), data classification (citizen records), encryption points (TLS hops, field-level encryption), audit-path (secure logger → central store), and the privilege boundaries of the entitlement service; audience: security reviewers and auditors. Also acceptable: communication view (protocols per hop) or reliability view (replication/failover paths). Close with the discipline rule: stale views must be marked or updated at release — a wrong view is worse than none, because it teaches distrust.

---

# THEME 9 — Foundations & Tactics in Scenario Form (comprehensive-paper insurance)

### Z1 [6 marks] A startup's system grew without deliberate architecture; the founder says "we have no architecture." Assess the claim and its practical consequences.

**Answer.** The claim is **false in principle**: every system comprises elements and relations, so it *has* an architecture — whether or not anyone described it. What the startup lacks is a **described** architecture. Consequences to enumerate: (1) nothing to communicate to new engineers — onboarding archaeology; (2) nothing to analyse — quality problems (slow responses, fragile changes) are discovered in production instead of predicted; (3) nothing to check conformance against — drift is invisible until painful; (4) rediscovery cost — exactly what reconstruction tools exist for. *Remedy:* reconstruct (raw extraction → database → view fusion with expert grouping → analysis) to recover the *de facto* structures, then decide deliberately which structures to keep, fix, or redesign — the cheapest moment to impose architecture is while the system is still small; the founder's real debt is **undocumented architecture**, not missing architecture.

### Z2 [5 marks] For a new e-commerce system, which module structures would you document first and why — decomposition, uses, layers, class or data model?

**Answer.** *Decomposition* first — it partitions the system into buildable units (Catalogue, Orders, Payments, User Management, Promotions), drives work assignment and integration planning, and localises likely changes (promotions change weekly). *Uses* second — it decides what can ship as an increment: if Orders *uses* Payments but Catalogue does not use Orders, a catalogue-first release is possible; uses is the structure that engineers incremental delivery and subsets. *Layer* third if the team adopts a layered discipline (allowed-to-use rules prevent a UI class from querying the DB). *Data model* alongside — e-commerce lives on entities (Product, SKU, Cart, Order, Customer) and their relations; it is the contract the shared-data components impose. *Class/generalisation* last — useful only where genuine inheritance/reuse exists (payment-provider adapters), otherwise ceremony. Justification line: document the structures that let you **reason about the attributes that matter** — modifiability and incrementality for a startup — not all five for their own sake.

### Z3 [5 marks] The word "architecture" is used for both "the set of structures" and "the big early decisions". Reconcile the two views in an answer an examiner would reward.

**Answer.** The structural definition is operational: structures (elements + relations + properties) that let you reason about the system. The decision-based view ("early, major, hard-to-change decisions") captures a *benefit*: architectural decisions are those whose reversal is expensive. They reconcile like this: **the decisions that matter are precisely those that fix structures** — choosing layers, choosing a broker, splitting services — because structures are what constrain later change; conversely, many "early" decisions (naming conventions, folder layouts) fix no reasoning structure and are not architectural, and some architectural decisions arrive late (adding a cache in year four). So: architecture *is* the set of reasoning-enabling structures; "early and major" describes where many of its decisions sit, not what it is. Examiner marks: exact definition (1), the three objections to "early/major" (1–2), the reconciliation (2), one example of each direction (1).

### Z4 [6 marks] A team must double capacity next month. Their app tier is stateful (in-memory sessions) and their data tier is a single relational node. Propose the tactics, in order, with the modifiability/performance reasoning.

**Answer.** *Step 1 — make the app tier stateless* (modifiability tactic: **encapsulation** of session state behind an interface; then **maintain multiple copies** of state in a shared store): sessions move to a distributed cache/DB keyed by session ID; any instance can serve any request; horizontal scale-out becomes a deployment action, not a redesign. *Step 2 — read replicas for the data tier* (**maintain multiple copies of data**): most e-commerce reads go to replicas; writes stay on the primary; accept bounded staleness on product pages (availability-favoured), keep inventory checks on the primary (consistency-favoured). *Step 3 — caching* (reduce resource demand): catalogue pages cached/CDN'd; hot queries in the distributed cache with invalidation on write. *Step 4 — bound queues + load shedding* at the gateway so overload degrades honestly instead of melting. *Ordering rationale:* statelessness first because it is the enabler — replicas and caching multiply capacity per instance only when any instance can serve any request. Side effects named: cache staleness (bounded windows + invalidation), replica lag (per-operation consistency choice), shed customers (specified overflow behaviour).

### Z5 [4 marks] Contrast ADD's "generate design" step with trial-and-error; name the inputs ADD consumes and the artefact it produces.

**Answer.** ADD's step is **tactic/pattern-driven**: for the module being designed, take its ASRs (six-part scenarios), select the tactics and patterns known to achieve those responses (e.g., availability → heartbeat + active redundancy; modifiability → encapsulation + intermediary), instantiate them as the module's structure, and record what was chosen and why. Trial-and-error instead proposes structures and tests them late — quality emerges (or not) from implementation, expensively. *Inputs:* the prioritised ASRs (from QAW/utility tree), constraints, and the functional responsibilities of the module. *Artefact:* the module's design — its structure (elements/relations), the tactics/patterns used, rationale, and the child modules to iterate on. The exam line: ADD makes quality attributes the *driver* of decomposition, not a property hoped for after the fact.

### Z6 [5 marks] "We do Scrum; architecture is waterfall thinking." Give the agile-architecture answer.

**Answer.** The premise confuses *up-front completeness* with *architecture work*. Agile architecture guidance: identify the ASRs early because their reversal is expensive (you cannot refactor availability into a shipped single-node design cheaply); implement them as the first increments (walking skeleton, spikes for risky technology); keep documentation minimal but sufficient — the views the stakeholders actually need, updated at release; let lower-risk structure emerge with refactoring; evaluate lightweight (mini-ATAM per quarter instead of a 4-phase study). The **sweet spot** argument: cost of architecture effort vs cost of *not* having it — both extremes (no architecture; ivory-tower full documentation) are losses, and the sweet spot sits where the marginal documentation/evaluation cost stops buying risk reduction. Artifacts that survive agile: the utility tree, the decision log (which pattern/tactic, why), and the conformance rules enforced in CI. Exam close: architecture in agile is *continuous and incremental*, not absent — the deliverable is a system whose expensive-to-reverse structures were chosen deliberately first.

---

# THEME 10 — Integration & Storage Decisions (mixed scenario questions)

### W1 [8 marks] A food-delivery platform must integrate: (a) 3,000 rider phones streaming locations; (b) a legacy warehouse system with nightly stock files; (c) restaurant POS systems with flaky connectivity pushing order-acknowledgements; (d) a partner map provider's REST API called on every address search. Choose the integration strategy for each and defend against the next-best alternative.

**Answer.** *Assumptions:* orders are money-critical; map lookups are latency-critical reads.
*(a) Messaging (pub-sub) + WebSocket gateway:* rider phones publish "location" events to a topic (messaging absorbs bursts and decouples consumer sets: ETA service, ops map, analytics); the ops map consumes through a **WebSocket** gateway for 1-second push to many viewers. Next best rejected: REST polling — per-update HTTP overhead × 3,000 riders saturates servers and still lags.
*(b) File transfer:* nightly bulk, whole-dataset, legacy partner — agreed CSV/XML on SFTP with checksums, record counts and a quarantine-and-replay for rejected rows. Next best rejected: shared database — couples two organisations' schemas; messaging — pointless for a nightly batch.
*(c) Messaging with store-and-forward at the POS edge + idempotent consumer:* restaurants with flaky links queue acknowledgements locally and sync when reachable (the store-locally pattern at POS scale); the platform consumer is idempotent per order ID. Next best rejected: synchronous RPC — a down POS must never block an order; file transfer is too slow for order status.
*(d) REST via an API gateway with caching:* synchronous reads on demand; the **gateway caches** frequent address→geo lookups (map results change slowly) and rate-limits/breakers protect against provider failures; fallback = last-known geometry from a local store. Next best rejected: messaging — search needs the answer now; direct integration scatters policy across services.
*Closing line:* four strategies in one system is not indecision — each boundary chose by latency need, availability of the other end, and change coupling.

### W2 [6 marks] The platform's order data now exceeds a single relational node. Options: read replicas, sharding the orders DB, or moving orders to a NoSQL store. Compare on consistency, complexity, and growth; recommend.

**Answer.** *Read replicas:* writes still hit one primary — consistent reads if reading primaries/strict replicas; simplest; growth ceiling = write throughput of one node; replica lag must be surfaced per operation (cart reads tolerate it; inventory doesn't). *Sharding:* orders partitioned by order/customer key — writes scale linearly with nodes; complexity: shard-key choice (hot spots), cross-shard queries/transactions become application work; rebalancing is disruptive. *NoSQL (document/column) store:* horizontal scale + flexible records; loses ACID across orders — a payment+order pair needs it, so only derived/immutable data (events, history) fits comfortably. *Recommendation:* keep the **order-of-record relational but shard by customer ID** once single-node writes saturate, with replicas for reads; move **event streams and click history to a column/document store** for analytics (that data is append-only and CAP-tolerant). Rationale: money movement keeps ACID (CP), analytics takes AP at scale; the decision is per data class, not per platform fashion. Note the CAP sentence: sharding keeps consistency but concentrates availability risk per shard — mitigate with replica sets per shard.

### W3 [5 marks] An interviewer asks: "Where does a distributed cache belong in our layered architecture, and what can go wrong?" Answer precisely.

**Answer.** *Where:* (1) **presentation/web tier** — output/page-fragment caching of expensive rendered pages; (2) **business tier** — memoisation of hot computed results (taxes, FX rates) with short TTLs; (3) **data tier boundary** — query/attribute caching between business and data layers to cut DB load; (4) **session state** for stateless scale-out. *Placement rule:* cache at the read path boundary that absorbs the most load per byte cached — never cache inside the transactional write path. *What goes wrong (name 3+):* **staleness** (bound with TTL/invalidation on write); **hot keys** (salt/split keys); **thundering herd** on expiry (stagger TTLs, jitter); **cache stampede** after restarts (warm-up scripts); **security** (cached PII — encrypt at rest, tenant-scope keys, purge on tenant deletion); **consistency traps** (read-your-own-writes — route the user's subsequent reads to the primary for N seconds); **cache as SPOF** (replicate the cache cluster; degrade to DB on cache loss — the app must survive a cold cache by design).

### W4 [6 marks] Two senior engineers argue about serverless for the ETA service: one says "no servers to manage, perfect fit"; the other says "we need sub-100 ms and tight control". Decide with criteria and name the middle option.

**Answer.** *Criteria:* (1) **latency sensitivity** — serverless adds cold-start latency (hundreds of ms for first hit after idle); sub-100 ms steady-state needs pre-warmed, memory-resident services; (2) **cost shape** — serverless wins for spiky/variable load (scale-to-zero; the CrowdCast economics), loses for constant high-volume traffic (per-invocation pricing exceeds reserved containers); (3) **state** — ETA computation needs live traffic features (in-memory state, nearby-vehicle indexes) — awkward across stateless function invocations; (4) **operational control** — execution-time limits, provider coupling, harder local testing; (5) **scaling granularity** — serverless scales per request (excellent for bursts), containers scale per node (predictable). *Decision:* the *event-driven parts* (ping ingestion, notification dispatch, report generation) → serverless; the *hot path* (ETA scoring at request time) → containerised service with pre-warmed instances behind the API gateway, auto-scaled on queue-depth/request-rate signals. Middle option named: **warm serverless / provisioned concurrency** (provider keeps instances warm — costs approach reserved capacity) or a "serverless first, containers where latency is proven to matter" rule with the spike evidence (the mobile-decision method applied to infrastructure). Close: neither engineer is wrong — the criterion is latency-sensitivity × load shape × state locality per component.

### W5 [4 marks] State the API-gateway-vs-direct-exposure decision for a B2B partner that needs 50,000 calls/minute to one endpoint.

**Answer.** High-volume, single-endpoint, well-authenticated partner traffic argues for a **direct path with dedicated infrastructure** (separate hostname/pool, mTLS or OAuth client-credentials, its own rate-limit contract) with the gateway reserved for policy on heterogeneous client traffic — because routing 50k/min through a general gateway couples the partner's SLA to shared gateway capacity (bottleneck risk) and adds a hop to a latency-sensitive contract. If policy parity matters more than the hop, keep the gateway but deploy a **dedicated gateway instance** for the partner (replication removes the SPOF). Either way: state the rule — the gateway is for policy breadth, not for being the largest pipe in the system; SLO-critical bulk flows deserve their own lane.

---

# THEME 11 — Two Integrated Mega-Questions (full 10-mark rehearsals with several syllabus parts combined)

### M1 [10 marks] "QuickLoan" is a microservices lending platform: Application service, Credit-score service (calls an external bureau), Decision service (rules), Contract service, Notification service. Rules change weekly; credit-score outages are common; auditors require tamper-evident decision records; load peaks on salary days. Design the architecture covering (a) decomposition and data ownership, (b) the decision pipeline and its audit, (c) failure handling for the bureau, (d) elasticity, and (e) the security boundary for citizen PII.

**Assumptions:** each loan application is processed end-to-end within minutes; PII regulations require minimal retention and encryption; the bureau SLA is 99% with occasional multi-minute outages.

**(a) Decomposition and data ownership [2 marks].** Services along bounded contexts — Application, Credit-score, Decision, Contract, Notification — each with **its own database**; cross-service reads only via APIs or published events. The application record (PII) lives solely in the Application service's encrypted store; other services hold references (application ID) plus the attributes they legitimately need (Decision keeps the rule inputs/outputs, not the raw payslips). Contracts between the teams are consumer-driven, tested in CI, so weekly rule changes in Decision never ripple into Contract.

**(b) Decision pipeline and audit [3 marks].** Application service publishes `application-submitted` events (messaging: temporal decoupling, load-levelling on salary days). Decision service consumes, fetches score, evaluates **externalised rules** (weekly changes = configuration, not code — *defer binding to configuration time*), publishes `decision-made`. **Audit:** every decision event is written by the Decision service to the **Secure Logger**: masked PII (hash of citizen ID, application ID, rule version, inputs as categories, outcome, timestamp), each record signed/hash-chained and shipped to an append-only central audit store the services cannot rewrite — auditors verify the chain and replay decisions from rule version + inputs. *Liability:* debug needs tokens not raw PII; masking tuned with the compliance team.

**(c) Bureau failure handling [2 marks].** Circuit breaker around the bureau client: open after N failures → **fallback** = cached score if the applicant was scored before (≤ 30 days) or queue for retry; requests take an explicit "decision pending-bureau" state instead of hanging — the pipeline degrades rather than stalls. Retries are idempotent (bureau call keyed by application ID + score-version); timeouts ≤ 3 s; bulkhead ensures bureau stalls consume only their pool. *Trace:* bureau outage → breaker opens in seconds → applications park in "pending" with customer-visible status → bureau recovers → half-open probe → queue drains in arrival order.

**(d) Elasticity [1 mark].** Stateless services behind an API gateway; scale on queue depth and request rate (proactive before salary-day peaks — auto-scaling response is late if reactive); Notification and Decision scale independently (Decision is CPU-light, Notification is I/O-heavy — separate pools). Contract generation is batched and scheduled off-peak.

**(e) PII security boundary [2 marks].** Trust boundaries: public internet → gateway (mTLS termination, OAuth/OIDC for users) → internal services; PII fields **encrypted at field level** with keys outside the application service's process (key-service monitor pattern — privilege separation: the Application service's parser runs unprivileged; only the monitor touches keys, rate-limited and audited). **Clear Sensitive Information** on the raw-document buffers after OCR/tokenisation (destructors cover error paths); retention minimised (purge raw documents after contract signing); DMZ perimeter + internal segmentation; every service account least-privilege scoped to its own store only. Close: five sub-designs share one theme — *limit the blast radius*: of a rule change (config), a bureau outage (breaker+fallback), a load spike (elasticity), a compromise (privilege separation), an audit query (tamper-evident events).

### M2 [10 marks] "GreenGrid" monitors solar installations: 200,000 inverters report every 30 s; engineers need fault alerts within 15 s; management wants monthly performance reports over all history; the board asks you to justify running it on a public cloud and to price one reliability strategy using CBAM-lite. Cover (a) ingestion and storage design, (b) alert vs report paths, (c) the cloud justification with two mitigations, (d) a CBAM-lite comparison of "replicate the ingestion tier" (cost 30, raises fault-alert availability utility by 25 on a scenario weighted 0.8) versus "add staff to fix alerts manually" (cost 10, raises the same utility by 8), (e) one CAP statement for the telemetry store.

**Assumptions:** telemetry is append-only and time-series; an inverter outage is not money-critical but SLA-critical; reports tolerate minutes of lag.

**(a) Ingestion and storage [2 marks].** Inverters publish to an IoT gateway → **messaging ingestion** (pub-sub, per-device key ordering, store-and-forward at the gateway for offline links — the device-scale store-locally pattern). Storage: **column-family/time-series NoSQL** (write-heavy, few-column reads, AP availability during partitions) sharded by site ID; shard key chosen to spread load evenly (avoid hot sites). 200k×2/min ≈ 6,700 writes/s baseline — scale-out ingestion by partitioning the topic; name the 3Vs: volume (PB-scale history), velocity (real-time alerts), variety (telemetry + events + firmware states).

**(b) Alert vs report paths [2 marks].** *Alert path:* stream processing on the bus — windowed rules (no output for 3 intervals, voltage anomalies) → alert events → notification service; **15 s budget** forces in-memory state, no batch involvement. *Report path:* nightly **MapReduce/Spark over history** — map emits `<site, (kwh, count)>`, reduce aggregates; weighted averages carried with counts; batch is cheap, fault-tolerant and complete — reports do not need seconds. The two paths share storage but not compute (bulkhead): a bad report job cannot delay alerts. This is the lambda shape — stream for now, batch for truth.

**(c) Cloud justification + mitigations [2 marks].** Economies of scale (power 15–20% of cost, ~1,000 servers/admin, hardware discounts), utilisation via co-located complementary loads, and **elasticity + measured service** for the bursty ingest/report profile (month-end reports, summer peaks). Mitigations: (1) multi-tenant risks → per-tenant keys/row-level scoping, patched hypervisor; (2) auto-scaling lag → proactive scaling on queue depth; assume instance failure — heartbeats + re-replication like HDFS.

**(d) CBAM-lite [3 marks].** Benefit: strategy R (replicate) = 25 × 0.8 = **20**; strategy S (staff) = 8 × 0.8 = **6.4**. VFC: R = 20/30 ≈ **0.67**; S = 6.4/10 = **0.64**. Rank: R first; with a small budget both fit, but R delivers ~3× the utility per future unit of spend after the first. *Step 9:* intuition check — replication also fixes the ingestion SPOF for reports, which staff cannot; confirm R. *Opportunity-cost line:* S's cost is recurring headcount that buys no structural improvement; R's forgone alternative (10 units not spent on hiring) buys a permanent capability — R wins on both ratio and reversibility.

**(e) CAP statement [1 mark].** Telemetry store is **AP**: during a network partition, keep ingesting and alerting on local partitions (an alert 30 s stale beats no monitoring), reconcile on heal; the billing-relevant settlement path, if any, reads from the batch-consistent warehouse (CP-ish) — per-operation choice, stated explicitly.

---

# THE CASE-STUDY PACK — Five Questions Modelled on the Course's Named Cases

### C1 [10 marks] Netflix-style — "StreamFlix" suffers evening outages: one database instance serves catalogue, recommendations and billing; a failed nightly batch blocks checkouts; a bad deploy takes everything down. Diagnose architecturally and prescribe the target architecture with named tactics and patterns.

**Answer.** *Assumptions:* 24×7 streaming; peak in evenings; catalogue mostly read-only; recommendations personalisable; billing money-critical.
*Diagnosis:* (1) **shared database = bottleneck + SPOF** for three unrelated workloads (Shared-Data's classic weaknesses — store bottleneck, SPOF, coupling through the data model); (2) **no bulkheads** — batch and interactive workloads share resources, so a batch overrun starves checkouts; (3) **no deployment isolation** — a monolithic deploy couples all features' risk; (4) no degradation path — every fault becomes a total outage.
*Target architecture:* decompose into **microservices per capability** (Catalogue, Recommendations, Playback-session, Billing) with **database-per-service** (modifiability + blast-radius isolation); Catalogue content on **CDN + distributed cache** (most requests never hit the DB); Recommendations served from **pre-computed fragments** refreshed by batch/stream pipelines (read/write split); **Billing isolated as a CP path** (relational, transactions, synchronous replication) behind a bulkhead so analytics/batch can never consume its pool. Failure handling: **circuit breakers** on every inter-service call with fallbacks (cached recommendations; "trending now" grid), **timeouts everywhere**, **active replication across zones** for each service (no SPOF), **bounded queues + load shedding** at the gateway at peak, **chaos/game-day drills** to verify degradation. Deployment: independent CI/CD per service, canary releases. *Tactics named:* maintain multiple copies, increase resources, limit event response, ping/echo health checks, graceful degradation, retry+idempotency. *Pattern names:* microservices, publish-subscribe for events (view-started → recommendations), API gateway. *Close:* each original failure maps to a named mechanism; the architecture's purpose is that the next failure costs one degraded feature, not the whole evening.

### C2 [6 marks] Salesforce-style — "CloudDesk" delivers a helpdesk SaaS to 200 companies. Compare multi-instance/single-tenant with single-instance/multi-tenant on cost, customisation, upgrade agility and security; recommend a per-tier tenancy split for: file uploads (heavy), ticket workflows (standard), audit logs (confidential).

**Answer.** *Multi-instance single-tenant:* per-company instance — strong isolation, free customisation, simple per-customer scaling/upgrade scheduling; costs: N deployments, N upgrade runs, wasted idle capacity — economy collapses as N grows. *Single-instance multi-tenant:* one instance for all — cheapest to run and upgrade (one upgrade for 200 companies), single version to develop; costs: noisy neighbours, customisation via configuration/metadata rather than code forks, and shared-resource security risk (a tenant-filter omission leaks another company's rows; hypervisor-level risks remain). *Per-tier recommendation:* **file uploads = multi-instance** (heavy, spiky processing justifies dedicated capacity so one tenant's burst cannot slow others); **ticket workflows = multi-tenant** (standard, utilitarian workload — highest economy, metadata-driven customisation); **audit logs = multi-tenant but logically isolated** — dedicated store/partition with per-tenant keys and append-only access (confidentiality without full instance duplication; full single-tenancy only for tenants whose compliance demands it). Rule to quote: **tenancy degree follows workload value, volatility and security, and may differ per tier**.

### C3 [5 marks] CIGNA/TripAdvisor-style — an insurance aggregator must quote from 14 insurers, each with a different interface (SOAP, REST, an SFTP mainframe feed, one email-based). Propose the integration architecture and its governance, naming the pattern and the connector per insurer.

**Answer.** *Pattern:* **SOA** — publish a single internal `QuoteService`; each insurer is a **provider** behind an **adapter service** that translates the aggregator's canonical quote request/response into the insurer's protocol: SOAP connector for SOAP providers, REST connector for HTTP APIs, **file-transfer integration** for the SFTP mainframe (nightly rate tables into a local quote provider), **messaging** for async confirmation callbacks, and a human/workflow step for the email-based insurer (orchestration assigns and tracks it). A **registry** holds provider endpoints; an **orchestration server** composes the fan-out to all 14 with per-insurer timeouts and partial-response policy (return the best quotes available within 5 s rather than blocking on the slowest — bounded event response). *Governance:* versioned canonical schema; adapters owned per insurer so a rate-interface change touches one adapter; monitoring per provider (latency, error rate) feeding fallback ranking. *Weaknesses accepted:* middleware overhead and provider-evolution risk outside your control (SOA's named weaknesses) — mitigated by adapter isolation and contract tests.

### C4 [6 marks] Danske Bank-style — a 10-year-old claims monolith releases quarterly; two squads want weekly releases. The CTO proposes "just add more servers." Argue why scaling is not the answer, then outline the migration path (first two steps only) with the risks each step manages.

**Answer.** *Why not scaling:* the pain is **release coupling**, not capacity — all squads edit one code base, so every release integrates everyone's risk and requires whole-system regression and coordination; more servers shorten response time, not integration time. The constraint is organisational ( Conway's law in effect): architecture must decouple change. *Migration step 1 — strangler façade:* put an **API gateway/facade** in front of the monolith and extract the highest-change capability first (claims intake) as a service with its own store; route by path; the monolith keeps the rest. Risk managed: big-bang rewrite risk — traffic can shift incrementally, rollback is trivial (flip the route). *Migration step 2 — anti-corruption + events:* wrap the monolith's shared database behind the façade (no new service writes the monolith's tables directly — an **anti-corruption layer** preserves the old schema); publish domain events (claim-created, claim-assessed) on a bus so the new service and the monolith stay eventually consistent, with a reconciliation job during transition. Risks managed: data-model corruption of the legacy system; hidden coupling surfacing as event-schema churn — versioned events + consumer-driven contract tests. Only after both steps does "more servers" make sense, because now instances are per-service and stateless enough to scale.

### C5 [5 marks] Boston Globe-style — a government portal serves forms, notices and payment on desktop and mobile. The mobile team wants a separate app; the web team wants responsive design. Decide for three user segments (citizen paying a fee, field officer with forms, notice reader) and justify.

**Answer.** *Citizen (occasional, payment):* **responsive web** — findable by search, no install friction, full content everywhere, one code base; payment UX works with large tap targets; store-presence adds little for a once-a-year task. *Field officer (daily, forms+data capture):* **app (native or cross-platform)** — needs offline forms (store-locally), camera for documents, push for assignments, background sync — capabilities the mobile web handles poorly; cross-platform (React Native/Flutter) fits a small team. *Notice reader (browse-only):* responsive web — zero-install reading; ensure responsive images so phones don't download desktop payloads. *Justification pattern:* decide per segment by **task frequency × hardware need × install friction tolerance**, not by one global choice; the portal ships both — one responsive site plus one task-focused app — with shared back-end services (API gateway + REST) serving both clients, so the second client's marginal cost is low.

---

# PRACTICE PAPER B (self-test — no answers shown; marking scheme follows)

**Time: 90 minutes. Attempt closed-book. 4 questions × 10 marks.**

**P1.** A logistics firm ingests 8,000 GPS pings/second from trucks, validates them, computes ETAs, stores history, and serves a live map to operations staff. (a) Propose a pipes-and-filters design with filters/pipes/source/sinks and one scaling mechanism. [3] (b) The live map needs 1-second updates to many simultaneous viewers — which pattern serves the fan-out and why; name one liability. [2] (c) Give three failure scenarios in this pipeline and the tactic for each. [3] (d) Where would you place a shared-data component, and what risk does it introduce? [2]

**P2.** A government benefits portal stores citizen data. An audit finds: full citizen IDs in application logs; log file writable by the app account; the entitlement-decision service runs as root and parses citizen XML with a third-party library. (a) Identify three violations and their risks. [3] (b) Propose the privilege-separated design for the decision service (participants, phases, interface policy). [4] (c) State the logging redesign with the Secure Logger, including what must never appear in a log line. [3]

**P3.** "EduTutor" runs quiz practice on mobile devices for school students; connectivity in schools is poor; the team is 4 developers fluent in JavaScript; the app needs the camera for scanning worksheets. (a) Choose the app type with the six-attribute justification. [4] (b) Design offline quiz-taking and sync with a conflict policy. [3] (c) A stakeholder demands "responsive web instead" — give the two strongest arguments against, and one for. [2] (d) Name the Android component that downloads worksheet packs in the background. [1]

**P4.** "TicketHub" must choose: **Strategy V** — virtual waiting-room + elastic cloud (cost 50; expected: 95% of fans through within 10 minutes, 2 s page loads); **Strategy Q** — fixed capacity sold in waves (cost 20; expected: 60% served per wave, 1 s loads, others queue for hours). Votes: fan-throughput weight 1.0; launch-early weight 0.5. Utilities — fans served (worst 40%/10, current 60%/30, desired 95%/85, best 100%/95); launch-early (0 mo/10, 0/10, 2/60, 4/90). Budget = 60. (a) Interpolate utilities for each expected response. [2] (b) Compute benefits, VFCs; select within budget. [3] (c) State each strategy's opportunity cost and one factor that flips the choice. [3] (d) Name and apply the method's final step. [2]

### Marking scheme (bullet answers — compare your attempt honestly)

- **P1(a):** source = GPS gateway; filters = validate/dedup → enrich (road segments) → ETA compute → store split (history sink + live sink); pipes = bounded queues keyed by vehicle ID; scaling = partition+replicate the ETA filter. *(1 per correct element class + walk-through)*
- **P1(b):** publish-subscribe: map servers subscribe to position events; decouples producers/viewers, unknown consumer set; liability: ordering/delivery guarantees weaker — mitigate per-vehicle ordered delivery. *(pattern + reason + liability)*
- **P1(c):** any three of: filter crash → replicate + re-run task (active redundancy); slow consumer → bounded queue + shed/aggregate; network partition → store-and-forward at gateway (retry, idempotent); data corruption → sanity checking + dead-letter sink. *(tactic named per scenario)*
- **P1(d):** truck/status metadata as a shared-data store for enrichment filters; risks: bottleneck, SPOF, tight data-model coupling — cache or replicate it.
- **P2(a):** secrets in logs (identity theft/attack planning + exposure copies); app-writable logs (tampering hides attacks — no evidence); root + third-party parser (any parser bug = full compromise). *(risk named per violation)*
- **P2(b):** monitor (root, state machine) + slave (unused UID, chroot, no FS) parsing XML; pre-auth phase: parse + request "check entitlement for citizen X" informational requests only, no identifiers/keys exported; post-auth: state-export respawn under service account; policy: terminate on invalid, rate-limit, audit via secure logger. *(participants 1, phases 1, policy 2)*
- **P2(c):** one logging interface; mask citizen IDs (hash/token), no full IDs/names/benefits amounts; sign+chain records; append-only store + central shipping; reader/viewer for auditors. *(interface+masking+integrity+store)*
- **P3(a):** cross-platform (React Native/Flutter) + native module for camera scanning; six attributes addressed with camera-heavy caveat; spike validation. *(decision 1, table 2, caveat+spike 1)*
- **P3(b):** SQLite local store primary; quiz session fully offline; queue answers with UUIDs; idempotent sync; conflicts: quiz answers = availability (LWW or both-kept), purchases/enrolments = consistency (refuse offline). *(store+queue+policy)*
- **P3(c):** against: offline requirement (web needs network for shell/data), camera integration weaker, app-store trust for parents/schools; for: instant updates, one code base, no install friction. *(2+1)*
- **P3(d):** **Service** (background download; receivers only react, Activities are UI).
- **P4(a):** V: fans 95% = desired → 85; early 0 mo = current → 10. Q: fans 60% = current → 30; early 3 mo: 60 + (90−60)×(3−2)/2 = **75**.
- **P4(b):** B_V = (85−30)×1.0 + (10−10)×0.5 = 55; B_Q = (30−30)×1.0 + (75−10)×0.5 = 32.5. VFC_V = 55/50 = 1.1; VFC_Q = 32.5/20 = 1.625. Budget 60: take Q (20, remaining 40) → V costs 50 → **Q alone** (32.5).
- **P4(c):** V's opportunity cost = earlier launch/features forgone (25 mo) + higher spend; Q's = fans turned away at every sale (reputation/refunds/revenue) + later rework. Flip factors: sale frequency (weekly peaks → V), penalty per unserved fan, growth certainty.
- **P4(d):** confirm with intuition: does turning away 40% per wave contradict the business goal? If yes, iterate — re-vote or price a hybrid (Q + bounded elastic burst).

---

# MOCK EC-3 EXAMINATION (attempt in one sitting: 2.5 hours, 40 marks)

**Instructions (as in the real paper):** all parts of a question answered consecutively; each answer starts on a fresh page; **assumptions stated clearly at the start of the answer**; marks per sub-part in brackets; calculations show formula, substitution, units. All four questions are scenario-based and require analysis and justification.

---

**Q1 [10 marks].** Suppose you are tasked with designing the data pipeline for "AgriSense", a service that ingests soil-sensor readings from 40,000 farms every 5 minutes, cleans them, computes per-field moisture trends, and pushes irrigation advice to farmers' mobile apps. The company expects new analytics (pest-risk, yield prediction) to be added every quarter.
a) Propose a pipes-and-filters architecture for the workflow, labelling filters, pipes, source and sink, and walk one reading through the pipeline. **[3]**
b) Explain how the design accommodates (i) the quarterly addition of new analytics and (ii) scaling to 3× farm growth, naming one pattern property and one tactic you rely on. **[2]**
c) Identify three limitations of pipes-and-filters in this scenario and propose a remedy for each. **[3]**
d) The advice push must reach phones within 30 seconds of a reading. Explain why you would place a publish-subscribe connector between the trend filter and the notification service, and one liability you accept. **[2]**

**Q2 [10 marks].** Suppose you are the architect of "PayCircle", a peer-to-peer payment app (mobile front end, backend services, a ledger database) that must pass a payment-card industry audit.
a) The auditor asks how you contain an exploitable bug in the third-party JSON parser that processes every incoming payment request. Propose a privilege-separated design: participants, the two phases, and the monitor's interface policy. **[4]**
b) The auditor next examines logging and memory hygiene. Explain the objectives, applicability and trade-offs of the Secure Logger and Clear Sensitive Information patterns as you would apply them to one payment transaction (card number → authorisation → receipt), and how the two combine. **[4]**
c) State one attack the combined design still does not prevent, and the tactic that limits it. **[2]**

**Q3 [10 marks].** Suppose you are tasked with the mobile architecture of "MediReach", a rural health-worker app: workers record patient visits (photos, symptoms, prescriptions) in areas with intermittent connectivity; the same app must run on Android and iOS; the team has six developers with JavaScript skills; management wants store presence for trust.
a) Choose between native, mobile web and hybrid/cross-platform development; justify against performance, security, user experience, development efficiency, code reuse and platform-specific optimisation; name frameworks and a validation method. **[4]**
b) Design the offline behaviour: where data lives, how synchronisation works, and a conflict policy distinguishing prescription records from visit notes. **[3]**
c) The backend moves to a public cloud with a 99.95% SLA. Calculate the permitted downtime per 30-day month and state one design consequence for the app. **[1]**
d) Name one Android component that must NOT be used to run the photo-upload synchronisation and why. **[2]**

**Q4 [10 marks].** Suppose you are the lead architect of "CrowdCast", an events-ticketing platform. Load peaks at on-sale moments (100,000 users within minutes), then collapses. Two architectural strategies are proposed: **S1** — a scalable cloud design with auto-scaling and CDN (cost 40 person-months; expected: 98% of peak users served, response 1.2 s); **S2** — a fixed-capacity design shipping 3 months earlier (cost 15 person-months; expected: 70% of peak users served, response 0.9 s for those served, others time out). Stakeholder votes: serve-peak-users scenario weight 1.0; ship-early scenario weight 0.4. Utility tables (worst/current/desired/best):
- Peak users served: 50%/utility 10, 70%/30, 98%/85, 100%/95.
- Ship-early (months gained): 0/10, 0/10, 2/50, 4/90. (Current: 0 months gained.)
The budget is 45 person-months.
a) Compute, by interpolation, the utility of each expected response. **[2]**
b) Calculate the total benefit and the value-for-cost of each strategy; select within the budget. **[3]**
c) Frame the choice with opportunity costs (state each strategy's forgone alternative) and list three factors that could reverse the selection. **[3]**
d) State what the final step of the method requires and how it applies here. **[2]**

---
---

# MOCK EXAMINATION: MODEL ANSWERS

*Written to the marks shown, ~60–80 words per mark. Cross-references [ ] point to notes sections.*

## Question 1 — AgriSense pipeline

**(a) Architecture + walk-through [3].** *Assumptions:* readings arrive as streams keyed by farm/field ID; advisories delivered to apps via a notification service.
```
Source: sensor gateway (40,000 farms, every 5 min)
  → [Validate & clean reading] ──(side pipe)→ Sink 2: rejected readings (audit/repair)
  → [Calibrate & unit-normalise]
  → [Trend computation per field] (windowed aggregation)
  → [Advice rule engine / model scoring]
  → Sink 1: advice store → Notification service → farmer apps
Pipes: bounded queues, agreed record format (farmId, fieldId, timestamp, values, quality flag)
```
*Walk-through:* a reading of 32% moisture from farm 881 arrives; the cleaning filter rejects it only if malformed (a spike of 320% passes calibration with a quality flag rather than rejection); calibration converts raw units; the trend filter updates field 12's 24-hour window; the advice engine compares against crop thresholds and emits "irrigate 6 mm tonight" to the advice store, from which the notification service pushes.

**(b) Extensibility + scaling [2].** (i) *Recombination*: because neighbours agree only on the pipe's record format, a new analytics (pest-risk) is **another consumer of the cleaned stream** — a new filter branch tapping the same pipe, without modifying existing filters (pattern property: flexibility by recombination). (ii) Scaling to 3×: replicate the slowest filters behind **partitioners by farm ID** (maintain multiple copies tactic), bounded pipes apply back-pressure, gateway shards by region; the pattern's independent filters make replicas safe (stateless steps carry state in the stream).

**(c) Three limitations + remedies [3].** (1) *Slowest filter sets the pace* → partition + replicate the trend filter; monitor stage latencies (ping/echo). (2) *Error handling is awkward (pipes carry no error information)* → side-output pipe to a rejected/dead-letter sink + replay filter, as drawn. (3) *Shared state across stages is expensive* (field metadata needed by several filters) → a cached metadata service keyed by field ID, or carry the needed metadata in the record. (Also acceptable: data-format conversion overhead → keep one canonical format; real-time limits → see (d).)

**(d) Pub-sub connector + liability [2].** The notification service is **one of several consumers of advice events** (advice store, dashboards, future channels) and their number is not predetermined — publish-subscribe decouples the trend/advice pipeline from consumers: the push service can be down for maintenance without stalling the pipeline, and new consumers subscribe without touching filters. *Liability accepted:* ordering and delivery are not guaranteed by the pattern — mitigate with per-farm ordered delivery and at-least-once semantics with idempotent notifications; latency is added (bounded by an in-memory broker at this volume).

## Question 2 — PayCircle security

**(a) Privilege-separated design [4].** *Participants:* a **monitor** process holding the ledger key/HSM connection at full privilege; a **slave** process (unused UID, chrooted empty root, no file rights) that runs the third-party parser and request handling; a **user-privileged child** for post-authentication work. *Phase 1 (pre-auth):* the slave parses the untrusted request (library bugs land at zero privilege) and asks the monitor for privileged results over a narrow socket-pair IPC. *Phase 2 (post-auth):* on successful authentication the **slave exports its state** (keys, counters, buffered data via marshalling/shared memory) and exits; the **monitor spawns a new slave under the user's identity**. *Monitor interface policy:* state-machine driven (valid request set per phase); only informational requests ("sign this authorisation for txn t", "debit check for account a") — **the key never crosses the boundary**; terminate on any invalid/out-of-state request; rate-limit decrypt/sign operations so a hijacked slave cannot drain the ledger. Trace: parser bug → attacker controls the slave → obtains an identity that can do nothing; monitor refuses malformed requests and ends the session.

**(b) Secure Logger + Clear Sensitive Information in one transaction [4].** *Secure Logger* — objective: logs help defenders, not attackers; an attacker reading the log learns nothing usable, and one editing it is detected. Applicability: payment systems log transactions and use logs for dispute/audit. Trade-offs: operators need the special reader/viewer; per-record signing/encryption costs time; the store becomes a high-value target. *Clear Sensitive Information* — objective: no secret survives in reusable resources (buffers, caches) after use; freeing does not erase. Applicability: the card number's buffers. Trade-offs: every sensitive item and release point must be found; compilers may elide overwrites; copies made before clearing cannot be recalled. *Combined on one transaction:* PAN read into buffers → **tokenised immediately** (token stored; PAN never persisted) → authorisation obtained → **one log event**: timestamp, transaction ID, amount, outcome, token reference, last four digits — no PAN, no track data (matches card-industry rules) → record signed/chained and shipped to the central log server → **buffers overwritten in a destructor that also runs on error paths**. The logger protects the persistent, widely read path; clearing protects the transient path — together both leak paths close.

**(c) Residual gap + tactic [2].** The design does not stop **denial of service within the slave's rights** (resource exhaustion, request floods) — privilege separation contains compromise, not abuse the slave may legitimately perform. Tactic: **limit exposure / resource limits** — per-session process, descriptor and request-rate limits plus bounded queues at the gateway, so exhaustion degrades one session instead of the service.

## Question 3 — MediReach mobile

**(a) Type decision [4].** *Assumptions:* photos + forms dominate; both platforms needed; six JS developers; store presence wanted. **Decision: hybrid/cross-platform** (single code base, store distribution, device access via native shell) — **React Native** (JS team skills; native widgets) or **Flutter** (framework-rendered; consistent UI), with **native modules** only if a hardware path proves heavy. Attribute analysis: *performance* — form/photo capture is well within cross-platform capability; the extra bridge layer is acceptable because no continuous camera/AR processing is required; *security* — reach platform key stores/biometrics via plug-ins; minimise plug-ins (each widens the attack surface); encrypt data at rest; secrets on server; *UX* — near-native on forms/lists; large buttons and simple flows satisfy rural usability; *development efficiency* — one code base and one skill set (JS) vs two (Swift + Kotlin) — decisive with six developers; *code reuse* — business logic and screens ~90% shared; *platform-specific optimisation* — limited, mitigated by native modules for camera compression if profiling shows need. *Validation method:* build a **spike** of the photo-capture-and-upload path on both platforms, measure against the target (e.g. upload of a 3 MB photo < 10 s on 3G), record the decision, iterate in sprints with device-matrix testing.

**(b) Offline design [3].** *Where data lives:* **local store first** — SQLite content provider holds visits, symptoms, prescriptions; photos in app storage; the UI reads/writes only the local store, so availability ≠ connectivity. *Synchronisation:* a **Service** drains a pending-changes queue when connectivity returns (triggered by a Broadcast receiver on connectivity-restored); messages carry client UUIDs → server **deduplicates** (idempotent); exponential backoff; photos upload after the record (resumable chunks); server is authoritative. *Conflict policy:* **visit notes** — availability-favoured: last-write-wins (or field-level merge) is acceptable; **prescription records** — consistency-favoured (a patient receiving two contradicting prescriptions is a safety event): prescriptions are edited only in one place (server authority), the app submits new versions rather than editing, and conflicting edits surface for pharmacist review rather than auto-merging.

**(c) SLA arithmetic + consequence [1].** Downtime per 30-day month = (1 − 0.9995) × 30 × 24 h = 0.0005 × 720 = **0.36 h ≈ 21.6 minutes**. *Consequence:* the app must treat backend unavailability as normal — full offline operation (as in (b)) plus retry/backoff on reconnect; never block data capture on a network call.

**(d) Wrong component [2].** A **Broadcast receiver** must not run the synchronisation: it is meant to **react** to events and hand work to a Service — Android may kill idle receivers, receivers have no execution lifetime for long uploads, and heavy work in a receiver blocks/ANRs the main path. The upload belongs in a **Service** (background, survives screen-off), triggered *by* the receiver. (Content provider is storage access; Activity is UI — equally wrong for background sync, but the receiver is the classic misuse since it *seems* to be woken by connectivity events.)

## Question 4 — CrowdCast economics

**(a) Interpolation [2].** S1 (peak served): expected 98% lies between desired (98, 85) and best (100, 95):
`y = 85 + (95−85)×(98−98)/(100−98) = 85`. Utility = **85**.
For S2, expected 70% = current level → utility = **30**.
Ship-early scenario: S1 gains 0 months → utility = current = **10**; S2 gains 3 months, between desired (2, 50) and best (4, 90):
`y = 50 + (90−50)×(3−2)/(4−2) = 50 + 20 = 70`. Utility = **70**.

**(b) Benefits, VFC, selection [3].** Gains: S1: peak 85−30 = 55; early 10−10 = 0 → B_S1 = 55×1.0 + 0×0.4 = **55**. S2: peak 30−30 = 0; early 70−10 = 60 → B_S2 = 0×1.0 + 60×0.4 = **24**.
VFC: S1 = 55/40 = **1.375**; S2 = 24/15 = **1.6**. Rank: S2 first by VFC, S1 second. Budget 45: take **S2 (15, remaining 30)** → S1 costs 40 → does not fit → **select S2 alone** (24 benefit points), which is also consistent with its higher ratio despite lower total benefit — exactly the case the ratio exists to expose (55 is not worth 2.7× the price here).

**(c) Opportunity costs + reversal factors [3].** *S1's opportunity cost:* the earlier market entry and the 25 person-months not spent on capacity (features/marketing forgone); *S2's opportunity cost:* the peak-service utility forgone — 30% of on-sale users unserved at every event (reputation, refunds, lost ticket revenue), plus later rework when growth arrives. *Reversal factors:* (1) on-sale frequency/duration — if peaks are rare and short, unserved users cost less (S2 holds); if weekly, S1 dominates; (2) growth trajectory — certain rapid growth makes S2's forgone scalability the larger loss; (3) reputation/refund penalties per unserved user — raising S2's effective opportunity cost may flip the VFC ranking; (4) modularity of the S2 design — if cheap scalability-enabling tactics (stateless tiers) are included, the hybrid "S2 now + S1 path" becomes the better buy.

**(d) Final step [2].** **Confirm results with intuition** (CBAM step 9): check the selected strategy against the organisation's business goals; if the outcome contradicts them, the analysis missed something and is iterated. Here: does shipping early while 30% of buyers are turned away align with CrowdCast's goal ("sell every ticket the market demands")? If stakeholders say no, revisit the utilities (especially the serve-peak scenario's weight) or price a hybrid (S2 + bounded auto-scaling), then re-rank. The numbers support, not replace, this judgement.

---

# RAPID-FIRE RECALL BANK (closed-book drill — cover the right column)

| # | Prompt | Answer to produce |
|---|---|---|
| 1 | Definition of software architecture | Set of structures needed to reason about the system: software elements, relations among them, properties of both |
| 2 | Three structure families | Module, Component-and-Connector, Allocation |
| 3 | Structure vs view | Structure = elements+relations as they exist; view = documented representation (architect designs structures, documents views) |
| 4 | Six scenario parts | Source, Stimulus, Artifact, Environment, Response, Response measure |
| 5 | Availability formula | A = MTTF/(MTTF+MTTR); series multiply; pair: 1−(1−A)² |
| 6 | Availability detect tactics | Ping/echo, heartbeat, monitoring, timestamp, sanity check, condition monitoring, voting |
| 7 | Performance tactic categories | Manage resource demand; manage resources |
| 8 | Security tactic categories | Detect, resist, react, recover (audit trail) |
| 9 | Modifiability tactics (5 core) | Increase semantic coherence, encapsulate, use intermediary, restrict dependencies, defer binding |
| 10 | Utility tree 4 levels | Utility → attributes → refinements → scenarios (ranked importance/difficulty) |
| 11 | ATAM 9 steps (gist) | Present ATAM; business drivers; architecture; identify approaches; utility tree; analyse; brainstorm; analyse again; present |
| 12 | ATAM 4 findings | Risk, nonrisk, sensitivity point, tradeoff point |
| 13 | Pattern = ? | Relationship between context, problem, solution; solution = element types + connectors + topology + constraints |
| 14 | 4 pattern categories | Mud-to-Structure, Distributed, Interactive, Adaptable |
| 15 | Pipe vs filter | Filter transforms (component); pipe only conveys, preserves order (connector) |
| 16 | P&F 4 liabilities | Shared state; transfer overhead beats parallel gain; format conversion; error handling |
| 17 | Scaling a slow filter | Partitioner → replicated instances → merger (by key) |
| 18 | Blackboard parts | Blackboard (data), knowledge sources (independent), control (opportunistic loop) |
| 19 | Broker 6 parts | Client, server, broker, client-proxy, server-proxy, bridge |
| 20 | MVC 5 interactions | View queries model; controller changes model; model notifies views; gestures view→controller; controller selects view |
| 21 | MVC 2 weaknesses (C&C form) | Complexity may not pay for simple UIs; abstractions may not fit toolkits |
| 22 | PAC = ? | Hierarchical agents, each Presentation+Abstraction+Control; controls are the only cross-agent channel |
| 23 | SOA 5 components | Provider, consumer, ESB, registry, orchestration server |
| 24 | SOA 3 connectors | SOAP, REST, asynchronous messaging |
| 25 | Microservices vs SOA (3 points) | Fine-grained + DB-per-service + lightweight pipes (no ESB), decentralised governance |
| 26 | Pub-sub weaknesses | Latency; ordering/delivery not guaranteed; scalability limits |
| 27 | Shared-data weaknesses | Store bottleneck; SPOF; tight coupling through the data model |
| 28 | P2P weakness | Security/consistency/backup harder without a centre |
| 29 | MapReduce 3 constraints | Input as files; stateless maps; only <key,value> communication |
| 30 | MapReduce 3 weaknesses | Small-data overhead; straggler partitions; multi-stage reduces complex |
| 31 | Batch vs stream | Batch: throughput + fault tolerance, delayed results; stream: seconds latency, complex state |
| 32 | HDFS roles | NameNode = metadata in memory; DataNodes = replicated blocks (3); heartbeats detect failure |
| 33 | Why tiny files hurt HDFS | All metadata in NameNode memory grows with file count |
| 34 | NoSQL 5 strengths | Scalable (sharding), fast (in-memory), available (replication), semi-structured, schema-flexible |
| 35 | NoSQL 2 omissions | ACID transactions; joins |
| 36 | 4 NoSQL models | Document, key-value, column, graph (+1 fitting workload each) |
| 37 | CAP | Consistency+Availability+Partition tolerance: pick 2; AP stale-but-up; CP refuses; CA theoretical |
| 38 | AP/CP examples | AP: Cassandra, CouchDB; CP: MongoDB, Redis |
| 39 | Cloud 7 characteristics | On-demand, ubiquitous access, pooling, location independence, elasticity, measured service, multi-tenancy |
| 40 | IaaS/PaaS/SaaS consumer | IaaS/PaaS: developer/admin; SaaS: end user |
| 41 | 3 cloud economics reasons | Economies of scale; higher utilisation; multi-tenancy |
| 42 | Hypervisor role | Creates/schedules/isolates VMs; host page table points to VM page tables (isolation) |
| 43 | Multi-tenant risks | Inadvertent sharing; VM escape; side channels; DoS by tenant |
| 44 | Containers vs VMs | Containers share host kernel (fast, dense, weaker isolation); VMs full OS (strong isolation, heavy) |
| 45 | Serverless trade-offs | No admin + scale-to-zero vs cold starts, time limits, state outside, provider coupling |
| 46 | 99.95% per month | 0.05% × 720 h = 0.36 h ≈ 21.6 min |
| 47 | Least privilege | Every program/user operates with least privilege needed (Saltzer-Schroeder) |
| 48 | Privilege separation parts | Monitor (privileged, validating state machine) + slaves (unprivileged) + narrow IPC; identity change via state export/respawn |
| 49 | OpenSSH numbers | ~2/3 code unprivileged; ~2% lines changed; historical bugs contained; no significant slowdown |
| 50 | Secure Logger intent | Logs unreadable/unusable to attackers + tampering detectable; one logging interface; no stdout/stderr |
| 51 | Never log (OWASP) | Session IDs, tokens, passwords, connection strings, keys, card data, sensitive personal data |
| 52 | Clear Sensitive Information | Overwrite secrets in reusable resources before release; destructors cover error paths; compiler elision risk |
| 53 | OpenID vs OAuth | OpenID = authentication (who you are); OAuth = authorisation (delegated access tokens) |
| 54 | TLS vs IPSec | TLS transport layer (channel + certs); IPSec network layer (site-to-site) |
| 55 | Mobile 5 considerations | Simple UI, responsive, compact code, few layers, store-locally/sync-later |
| 56 | App types pros/cons | Native fast/full access/one platform; web instant updates/no hardware; hybrid single code base/slower |
| 57 | Cross-platform frameworks | React Native (JS→native widgets), Flutter (Dart, own rendering), Xamarin→.NET MAUI, Kotlin MP |
| 58 | Android 4 components | Activity (UI), Service (background), Content provider (storage API), Broadcast receiver (events); intents wire them |
| 59 | Store-locally pattern | Local store is primary; sync later; idempotent updates; conflict policy (LWW/server-wins/merge) |
| 60 | Responsive 3 ingredients | Flexible grids (media queries), flexible images, per-size layout choice |
| 61 | 5 mobile UI patterns | Action bars, social login, large buttons, notifications, discoverable controls |
| 62 | Big data 3Vs | Volume, variety, velocity — each drives distribution/schema-flexibility/batch-vs-stream |
| 63 | CBAM 9 steps | Collate→Refine→Prioritise(votes)→Utility→Map strategies→Interpolate→Benefit→VFC select→Confirm intuition |
| 64 | CBAM formulas | b = U_exp − U_cur; B = Σ(b×W); VFC = B/C; y = ya + (yb−ya)(x−xa)/(xb−xa) |
| 65 | Curve shapes | Diminishing returns; value-near-top; S-shaped (threshold); linear; step function |
| 66 | Opportunity cost | Value of best forgone alternative (sunk cost ≠ opportunity cost) |
| 67 | Utility anchors | Best case 100, worst case 0 (convention loosely applied) |
| 68 | API gateway role | Single entry: routing, auth, rate limiting, aggregation, versioning |
| 69 | Circuit breaker | Fail fast after N failures; fallback; probe to restore (failure handling for microservices) |
| 70 | Bulkhead | Isolate resource pools so one failing component cannot exhaust others |

---

# EXTENDED DRILL BANK (40 more — cover the right column, then check)

| # | Prompt | Answer to produce |
|---|---|---|
| 1 | Why is a box-and-line drawing not an architecture? | Names elements but states no relations/properties; readers imagine behaviour that may be wrong; architecture = elements + relations + properties |
| 2 | Every system has an architecture — implication? | Exists even undocumented; documentation enables communication, analysis, conformance checking |
| 3 | "Early/major decisions" objection | Some architectural decisions are late; some early ones aren't architectural; "major" is unjudgeable |
| 4 | Decomposition structure supports? | Modifiability by localising change; basis of work assignment, docs, integration plans |
| 5 | Uses structure enables? | Extension and functional subsets → incremental development |
| 6 | Layer structure gives? | Portability — change the underlying platform by changing one layer |
| 7 | Deployment structure reasons about? | Performance, data integrity, security, availability of distributed systems |
| 8 | Concurrency structure finds? | Parallelism opportunities and resource-contention locations |
| 9 | Work assignment structure | Maps modules to teams; determines team communication pathways |
| 10 | 4+1 views | Logical, Process, Development, Physical + Scenarios binding them |
| 11 | Why scenarios are "+1" | One use case traced across all four views proves consistency |
| 12 | QAW steps (gist) | Present; business drivers; brainstorm scenarios; consolidate; prioritise; refine |
| 13 | PALM | Purpose, Anticipated benefits, Losses, Metrics — business-goal elicitation |
| 14 | ADD 3 steps | Choose module; identify ASRs; generate design (tactics/patterns); iterate |
| 15 | Agile architecture guidance | Just enough, iterate, spikes for risk, minimal documentation stakeholders need |
| 16 | Ping/echo vs heartbeat | Both fault detection; ping/echo = round-trip probe, heartbeat = periodic presence signal |
| 17 | Active vs passive redundancy | All copies sync + take over instantly vs standby updated periodically, slower failover |
| 18 | Graceful degradation | Drop optional features under fault; keep core service |
| 19 | Limit event response | Bound queues/execution so overload fails fast instead of growing waits |
| 20 | Anticipate change | Localise predicted changes into dedicated modules (modifiability tactic) |
| 21 | Broker known uses | CORBA, IBM SOM/DSOM, OLE 2.x, WWW |
| 22 | Broker variants | Direct communication, message passing, trader, adapter, callback |
| 23 | Name service & marshalling | Name→object resolution within a name space; machine-independent data encoding |
| 24 | MVC known uses | Smalltalk VisualWorks; MFC (Document-View); ET++ |
| 25 | Document-View variant | Merges view+controller for platforms binding display/events; loses pluggable controllers |
| 26 | Blackboard known uses | HEARSAY-II speech, HASP/SIAP sonar, CRYSALIS protein, SUS |
| 27 | Microkernel definition | Minimal core (mechanisms, IPC, resources) + internal/external servers + adapters + clients |
| 28 | Reflection definition | Meta level (metaobjects) modifies base level via metaobject protocol at runtime |
| 29 | Multi-tier constraint | Each component belongs to exactly one tier |
| 30 | Map-reduce relations | Deploy-on (instance↔processor); instantiate-monitor-control (infrastructure↔instances) |
| 31 | HDFS write steps (short) | Buffer → NameNode → 3 DataNodes → pipeline copy → acks → commit |
| 32 | Document vs key-value | Query inside self-describing records vs opaque get/put by key |
| 33 | Column store suits | Few-column analytics, time-series, write-heavy logs (Cassandra/HBase) |
| 34 | Graph store suits | Traversals: social, recommendations, fraud rings |
| 35 | Tesco/Ryanair/Marriott/Gannett/GE | Retail catalogue / mobile backend / reservations / CMS / industrial IoT — NoSQL breadth |
| 36 | Big data uses (any 4) | Fraud detection, targeting, sentiment, call-centre analytics, plant monitoring, weather |
| 37 | AWS 4 differences | Programmable resources; global capacity; managed services; built-in security |
| 38 | AWS principles (any 5) | Scalability, disposable resources, automation, loose coupling, services-not-servers, right DB, no SPOF, cost, caching, security-everywhere |
| 39 | Tailspin tenancy options | Multi-instance single-tenant vs single-instance multi-tenant; choose per tier |
| 40 | Service mesh vs ESB | Uniform sidecar policy for east-west microservice traffic vs central routing/transformation hub |

# SELF-TEST ESSAY PROMPTS (write 8–12 lines each, then check against the notes)

1. "Patterns are molecules built from atoms." Prove it on Publish-Subscribe and repair one of its weaknesses with a named tactic.
2. Trace one request through a Broker system, naming every proxy operation, then state where latency and fragility both come from.
3. A teammate says: "MVC automatically makes apps scalable." Refute precisely, then state what MVC *does* buy.
4. Design the monitor interface of a privilege-separated payment service: request types, state machine, rate limits.
5. One transaction, one paragraph: card number → receipt, showing Secure Logger + Clear Sensitive Information cooperating.
6. Explain to a manager why CAP means the payments path and the activity feed must be architected differently.
7. A customer demands 99.999% availability. Convert that to hours/year, then list the tactics you would price, and say why you would first negotiate.
8. Run CBAM steps 5–8 on two strategies of your own invention, showing interpolation, benefit, VFC and the budget stop.
9. Justify storing a rural health app's prescriptions differently from its visit notes, using the CAP vocabulary.
10. Explain why Hadoop is wrong for fraud-blocking and right for monthly plan repricing — name the constraints that decide.
11. Decompose a food-delivery app into Android components with intents; mark each intent directed or broadcast.
12. Your CTO asks: "Why not one big server?" Answer with utilisation, SPOF and CAP arguments in five sentences.

---

# EXTENDED DRILL BANK 2 (30 more rows — rapid closed-book recall)

| # | Prompt | Answer to produce |
|---|---|---|
| 1 | Utility vs response measure | Utility = stakeholder valuation 0–100; response = physical measure; interpolation response→utility |
| 2 | Rank-weight scheme | Top scenario W=1.0, others relative to top votes (100 votes/stakeholder) |
| 3 | Benefit formula with side effects | b = U_expected − U_current (negative counts); B = Σ b×W |
| 4 | VFC selection rule | Rank by B/C; take top-ranked until budget/schedule exhausted; ties → stakeholder judgement |
| 5 | CBAM step 9 | Confirm with intuition against business goals; iterate if misaligned |
| 6 | Five curve shapes | Diminishing returns; value-near-top; S-curve; linear; step |
| 7 | Security S-curve lesson | Little value below a compliance threshold; saturation above — buy to the band |
| 8 | Opportunity cost of speed-first | Scalability utility forgone + later rework if growth arrives |
| 9 | Sunk cost rule | Exclude from opportunity-cost analysis (already unrecoverable) |
| 10 | ATAM phase 1 vs 2 | Steps 1–6 with decision makers; steps 7–9 with stakeholders added |
| 11 | ATAM is exhaustive? | No — scenario-driven; analyses only what top scenarios touch |
| 12 | Lightweight evaluation | Fewer scenarios/steps, same skeleton, for budget-constrained or agile cadence |
| 13 | Raw view extraction collects | Classes, files used, caller-callee, global data accessed — uninterpreted |
| 14 | View fusion merges | Static + dynamic (trace) views + expert grouping into consolidated views |
| 15 | Reconstruction purposes | Understand undocumented; migrate; identify reusable components |
| 16 | Conformance by construction | Frameworks (Spring MVC, JMS, Hibernate) force structure |
| 17 | Fault-tolerant template sections | Normal process (+send state), state update, switch-over — cannot be omitted |
| 18 | Folder-structure conformance | File layout mirrors architecture so violations are visible at commit time |
| 19 | Drift definition | Code gradually departs from documented architecture via local fixes |
| 20 | REST verbs for mobile web | GET/POST/HEAD/PUT/DELETE + URL over HTTPS; JSON replies |
| 21 | Mobile web vs hybrid difference | Container: browser vs native shell (device APIs + store install) |
| 22 | Progressive web app | Web technology delivering app-like behaviour (offline, installable) |
| 23 | Screen target sizes | ~44–48 density-independent touch units; no hover-dependent functions |
| 24 | Broadcast-intent benefit | Sender doesn't know receivers — publish-subscribe inside one device |
| 25 | Uber component mapping | Activity=booking UI; Service=sends location; Broadcast receiver=receives cab location; DB=config/user data |
| 26 | HAL role | Standard interfaces exposing hardware capabilities to the API framework |
| 27 | Tizen/Android stack order | Apps → framework (native+web) → core → kernel/drivers (HAL between) |
| 28 | Edge computing drivers | Latency, bandwidth cost, data locality/compliance, offline resilience |
| 29 | Service mesh provides | mTLS, retries/timeouts, traffic shifting, observability via sidecars |
| 30 | Wasm value | Portable sandboxed near-native execution; plugin isolation; complements JS |

# FINAL 60-SECOND REVISION SCRIPT (read aloud the night before)

1. Architecture = structures (elements + relations + properties) to reason about the system; three families; design structures, document views.
2. Scenario = source, stimulus, artifact, environment, response, measure — numbers or it isn't a scenario.
3. Tactics are atoms; patterns are molecules; every plus has a minus; resolve with the next tactic until small.
4. Patterns: layers (isolate), pipes-filters (stream, slowest-stage limit), blackboard (no recipe), broker (locate+forward, 5 hand-offs), MVC (model notifies; change lands in one home), PAC (agent tree), SOA/microservices (published contracts; DB-per-service; breakers), C-S/P2P/pub-sub/shared-data (the four C&C primitives), map-reduce (stateless maps, key-grouped reduce), multi-tier (deployment grouping).
5. Security: least privilege → privilege separation (monitor/slave, state machine, OpenSSH 2/3 unprivileged); Secure Logger (no secrets, tamper-evident, one interface); Clear Sensitive Information (overwrite incl. error paths); TLS channel, OpenID who, OAuth what, DMZ perimeter.
6. Cloud: 7 characteristics; IaaS/PaaS/SaaS; cheaper by scale + utilisation + tenancy; hypervisor isolates via page tables; HDFS NameNode/DataNodes/3 replicas/heartbeats; auto-scaling is late — be proactive; multi-tenant risks = leakage, escape, side-channels, DoS.
7. CAP: pick two; AP stale-but-up (Cassandra), CP refuses (MongoDB/Redis), CA theoretical; payments CP, feeds AP.
8. Mobile: 5 considerations; native vs cross-platform = the six attributes + hardware caveat; store-locally/sync-later with idempotent queue and conflict rules; Android = Activity/Service/Provider/BroadcastReceiver wired by intents; responsive = grids+images+queries.
9. Big data: 3Vs; Hadoop = HDFS + MapReduce; map emits pairs, shuffle groups, reduce sums (carry counts!); batch = cheap+tolerant+late; stream = fast+complex; NoSQL = scale/speed/availability minus ACID/joins; four models by access pattern; shard key decides.
10. CBAM: collate → refine (4 levels) → vote → utility → map strategies → interpolate → benefit Σb×W → VFC=B/C within budget → intuition; opportunity cost = best forgone alternative; utility curves decide where a marginal improvement is worth money.
11. Answer skeleton: assumptions → attribute/scenario → select+map → both sides → resolve side effects → decision with a flip condition.
12. Budget: 3.75 min/mark; 60–80 words/mark; 10-mark answer ≈ 600–800 words + a labelled diagram; calculation = formula → substitution → result → interpretation.

---

# DEFINITION DICTATION DRILL (write each from memory, then compare — exact wording earns the first mark)

1. **Software architecture** — the set of structures needed to reason about the system, comprising software elements, relations among them, and properties of both.
2. **Structure** — a set of elements held together by a relation.
3. **View** — a representation of a coherent set of architectural elements as written by and read by stakeholders.
4. **Architectural pattern** — a relationship between a context, a generalised problem and an abstracted solution, described by element types, connectors, topology and semantic constraints.
5. **Tactic** — a design decision that influences a single quality-attribute response.
6. **Layer** — a grouping of modules offering a cohesive set of services through a managed (public) interface; allowed-to-use is unidirectional.
7. **Filter / Pipe** — a component that transforms its input stream into its output stream incrementally / a connector that conveys data in order without altering it.
8. **Change propagation (MVC)** — the model maintains a registry (attach/detach/notify) of dependent views and controllers; every change triggers update, and each view re-fetches and redraws.
9. **Broker** — a component that coordinates communication between clients and servers: locating servers, forwarding requests, transmitting results and exceptions.
10. **Marshalling** — machine-independent encoding of data for transmission (unmarshalling = the reverse).
11. **SOA** — a collection of distributed components that provide and/or consume services; consumers depend only on published service descriptions.
12. **Microservice** — a fine-grained, independently deployable service owning its own data, interacting through lightweight protocols.
13. **Blackboard** — a central data store of solution-space and control data, updated by independent knowledge sources scheduled opportunistically by a control component.
14. **Microkernel** — a minimal core of mechanisms (including IPC and resources) with internal servers, external servers, adapters and clients built around it.
15. **Reflection** — a meta level of metaobjects that represents the system itself and can change base-level behaviour through a metaobject protocol.
16. **Tier** — a logical grouping of components for deployment; each component belongs to exactly one tier.
17. **MapReduce** — parallel analysis of file-based data: stateless map functions emit <key,value> pairs; the framework groups pairs by key; reduce combines each key's values.
18. **HDFS** — a distributed, replicated, self-healing file system: a NameNode holds metadata in memory; DataNodes hold replicated blocks; heartbeats detect failures.
19. **CAP theorem** — a distributed service cannot simultaneously guarantee consistency, availability and partition tolerance; any two, never all three.
20. **Eventual consistency** — a guarantee that replicas of data will agree after some delay.
21. **NoSQL database** — a non-relational database designed for scale, speed, availability and semi-structured data, typically without ACID transactions or joins.
22. **Multi-tenancy** — a single application instance serving multiple consumers unaware of each other.
23. **Least privilege** — every program and user operates with the least privilege necessary to complete the job.
24. **Privilege separation** — dividing an application into a privileged monitor and unprivileged slaves so errors in exposed code cannot yield privilege; the monitor validates each request against its current state.
25. **Utility / Utility-response curve** — the stakeholder benefit of a quality-attribute response (0 = worst acceptable, 100 = best useful) / the graph relating response measures to utilities.
26. **Value for cost** — VFC = total benefit ÷ cost; rank-order strategies by it within budget and schedule constraints.
27. **Opportunity cost** — the value of the best forgone alternative; sunk costs are excluded.
28. **ATAM** — a scenario-driven architecture evaluation method producing risks, nonrisks, sensitivity points and tradeoff points in nine steps across four phases.
29. **ASR / Utility tree** — a requirement with a profound architectural effect / the hierarchy utility→attributes→refinements→ranked scenarios that concretises ASRs.
30. **Store locally, synchronise later** — the mobile pattern where the device holds the primary copy and transmits to the server when connectivity permits.

**Scoring rule:** write each from memory; if the structure (subject + mechanism + qualifier) is right but wording differs, that is a pass — in the exam, examiners reward the complete mechanism, not the publisher's exact words.

---

# CLOSING CHECKLIST (tick before entering the hall)

- [ ] Six scenario parts recite without notes (S-S-A-E-R-R).
- [ ] ATAM 9 steps and 4 finding types recite.
- [ ] CBAM 9 steps + B, VFC, interpolation formulas written from memory.
- [ ] OpenSSH numbers (two-thirds, 2%) and 99.95% → 4.38 h/year available.
- [ ] Pipes-and-filters: 6 implementation steps, 4 liabilities, scaling trick.
- [ ] MVC: 5 interactions + the price-change trace.
- [ ] MapReduce: map/shuffle/reduce + hashtag example + why counts travel.
- [ ] Native vs cross-platform: the six attributes in table form.
- [ ] Android: 4 components + 2 intent kinds + Uber decomposition.
- [ ] CAP: two-node example + AP/CP products.
- [ ] NoSQL: 5 strengths, 2 omissions, 4 models.
- [ ] HDFS: write-path 8 steps + tiny-file weakness.
- [ ] Answer skeleton memorised: assumptions → attribute → select+map → both sides → resolve → decide.
- [ ] Two mock papers attempted under time; Practice Paper B marking scheme self-applied.
- [ ] 60-second revision script (Drill Bank 2 end) read aloud twice.

**You are ready. The paper rewards analysis, comparison and justification — practise the shape, and the knowledge will follow the skeleton.**

---

# TWENTY EXAM-DAY CLOSING MOVES (small tactics that collect stray marks)

1. **Underline the command words** in the question before writing one word — "analyse" ≠ "define" ≠ "discuss".
2. Write **assumptions in a numbered list** — the marker sees instruction-compliance in three seconds.
3. Every pattern answer gets a **mini-diagram** — even a 6-line box-and-arrow drawing with labels beats pure prose (marks: element types + connectors visible).
4. Name **benefit AND liability in the same paragraph** — "X buys Y at the price of Z" is the highest-density sentence form in this subject.
5. Always **map, never recite**: replace "the pattern has elements…" with "the fraud-checker service is the filter…".
6. For every **liability named, add the tactic** that repairs it — that is the molecules-and-atoms discipline made visible.
7. Calculations: **formula → substitution → result → interpretation**, four lines, every time.
8. Clamp interpolations at the **bracketing points** — never extrapolate beyond the elicited curve; say "utility = 85 (capped at desired level)".
9. In CAP answers, **state the two-node example** — it converts a memorised fact into demonstrated understanding.
10. In mobile answers, answer **under the question's own headings** (performance, security, UX, efficiency, reuse, optimisation) in that order.
11. **Tables for comparisons, prose for justifications** — a 5-row table replaces 300 words and is easier to mark.
12. Close every design answer with **"Decision + flip condition"**: "I choose X; if load growth were certain within months, I would instead choose Y."
13. In CBAM answers, state **which filter keeps what** (top-third scenarios; top-50% votes; VFC strategies) — the three-filter line is a frequent 1-mark item.
14. Use the **stored scenario library** (Part Z2) for response measures — never invent a number you cannot defend as illustrative.
15. If time runs short, **write the skeleton in full and shrink prose** — an examiner marks mapped points, not paragraphs.
16. Number your sub-answers **exactly as the paper does** — (a)(i), (a)(ii)…; part answered consecutively, fresh page per question.
17. For "justify your decision", give **two reasons and the accepted liability** — three items, then stop.
18. In security answers, begin with **least privilege and the trust boundary** — the paper's framing — before the pattern.
19. Quote **one number per claimed benefit** (OpenSSH two-thirds; 99.95% = 4.38 h/year; admin ratio 1,000 vs 150) — numbers anchor credibility.
20. Leave **5 minutes at the end** to re-read assumptions vs conclusions — a conclusion that contradicts a stated assumption is the most expensive silent error in a closed-book paper.

---

# FINAL NOTE — HOW TO STUDY THIS FILE IN THE LAST 10 DAYS

- **Days 10–7:** read Parts A–G once; write each Part R pattern card from memory; drill Part M/V calculations (10 minutes daily).
- **Days 6–4:** attempt Theme Q&A questions closed-book (one theme per day, 45 minutes, 600–800 words per question); compare with model answers; mark yourself honestly against the command-word table.
- **Days 3–2:** full Mock Examination under 2.5-hour timing, then Practice Paper B; memorise Part I mnemonics and the 60-second script (Drill Bank 2 end).
- **Day 1:** Part Z verbatim definitions + Z2 scenario library + Z3 assumption bank + the twenty closing moves; sleep.
- **In the hall:** budget 35 minutes per question; assumptions first; map, price, decide — and remember the examiner rewards *justified judgement*, not recalled prose. Every "both sides" sentence and every "decision with a flip condition" is a mark the pure-recall candidate leaves on the table — those marks are yours because you practised the shape, not just the content.
