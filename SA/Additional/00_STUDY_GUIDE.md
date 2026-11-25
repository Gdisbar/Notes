# STUDY GUIDE — what to study now that you carry every slide

*The reframe:* you can look **anything** up. Marks go to what cannot be looked up — **choosing the right idea for the situation, justifying it with trade-offs, and structuring the answer.** So study *judgement*, not content.

---

## 1. The three tiers

### Tier 1 — Understand deeply (cannot be copied from a slide)
| # | Skill | What you must be able to do | Practise with |
|---|---|---|---|
| 1 | **Write a quality-attribute scenario** | turn any vague requirement into the six-part scenario with a measure | `01` B1; `02` §0 |
| 2 | **Reason from stimulus to tactic category** | say *why* a category (detect / recover / prevent; control demand / manage resources; detect / resist / react / recover; reduce coupling / defer binding…) fits, and its side-effects | `03` D1.1–D1.8 |
| 3 | **Prioritise with a utility tree** | rate (importance, difficulty), pick the top-right first | `01` D3 |
| 4 | **Select a pattern and defend it** | match problem → pattern, state liability, patch it with a tactic | `03` §2 |
| 5 | **Run an ATAM-style analysis** | for a scenario: approach → risk / non-risk / sensitivity / trade-off point | `01` H2; `02` §11 |
| 6 | **Make consistency/availability/isolation choices** | CAP per service; multi-tenancy per tier; IaaS/PaaS/SaaS; cloud failure handling | `03` §7; `02` §9 |
| 7 | **Apply layered techniques to a case** | place techniques in the right layer; service-layer failure handling | `02` §10 |
| 8 | **Identify mobile components** | map functions to Activity / Service / Content provider / Broadcast receiver | `02` §8 |
| 9 | **Do the CBAM arithmetic** | weights → benefit → VFC → ranking, and explain the result | `02` §14; `03` §8 |
| 10 | **Structure a case-study answer** | business goals → ASRs → patterns/tactics → views → deployment → trade-offs → evaluation | `04` §1 |

### Tier 2 — Know exactly where it is (look up, don't memorise)
Tactic **names** for each attribute · the seven decision categories and every checklist question · structure/steps/variants of each pattern · the relations tables for structures · 4+1 view details · ATAM nine steps · HDFS write steps · Android component definitions · Hadoop/Spark/NoSQL details · TLS/OAuth/OpenID definitions · PMBOK matrix · CBAM step details · conformance techniques. → use the tab list in §5.

### Tier 3 — Skim only
Why architecture matters (list) · the four contexts · history and introductions · ISO quality lists · credits/summary slides. One reading is enough.

---

## 2. Where your time should go

| Question type (from your guideline) | Share of study time | What to do |
|---|---|---|
| **Concept** | **15%** | read `01` once; make sure you can explain every Tier-1 skill in your own words |
| **Scenario** | **30%** | do the scenario drills (below); learn the 8-step algorithm in `02` §0 |
| **Design decision** | **30%** | learn the comparison tables in `03` by *reasoning*, not by heart; practise the five-line decision answer |
| **Case study** | **25%** | do two timed cases from `04` using the blueprint |

> If time is short, **do Tier-1 skills 1, 2, 4, 6 and 10 first** — they appear in almost every analysis question.

---

## 3. Drills (on paper, timed)

| # | Drill | Time | Success check |
|---|---|---|---|
| D1 | Write a six-part scenario for each of the seven attributes for one system (e.g., online ticket booking) | 20 min | every scenario has a measurable response measure |
| D2 | For availability, performance and security: take one scenario → name tactic categories → name one side-effect of each | 20 min | each tactic has a "costs…" statement |
| D3 | Ten mini-problems: *"which pattern?"* (use `03` §2.1 only after answering) | 20 min | ≥8 correct with a stated liability |
| D4 | Attempt the four layered exercises **without looking** `[M:111-115]`, then compare | 30 min | components in the right layers; service-layer failure handled |
| D5 | Write a utility tree (6 scenarios) for a hospital/banking system and rate them | 15 min | at least two (H,H) with measures |
| D6 | Mini ATAM on one scenario: approach, risk, non-risk, sensitivity, trade-off | 15 min | one explicit trade-off point |
| D7 | CBAM: 3 strategies × 2 scenarios; compute benefit, VFC, rank | 15 min | correct ranking and one sentence of explanation |
| D8 | CAP: decide C vs A for five services of an e-commerce site | 10 min | decision justified per service |
| D9 | Multi-tenancy: choose single/multi per tier for two SaaS cases | 10 min | tied to data confidentiality and processing load |
| D10 | Full case: courier/health/e-commerce/GST — blueprint headings + one diagram | 40 min | all 10 blueprint headings present |

---

## 4. Self-test questions (inferred; answer on paper, then check against the notes)
1. A bank's payment service must survive a data-centre crash — write the availability scenario and choose tactics; state costs.
2. Page load is slow at peak — diagnose and fix; what do you give up?
3. A partner needs to book your rooms through an API — design the service layer and failure handling.
4. Compare Broker and Publish-Subscribe for an event-driven system.
5. Why might Layers be wrong for a platform that must run customer-specific plug-ins?
6. Run a lightweight ATAM on a chosen scenario — what are the outputs?
7. Native, web or hybrid for a field-service app with poor connectivity?
8. Choose between single-instance multi-tenant and one instance per tenant for a GST filing product.
9. Click-stream analytics: batch or stream? Why?
10. Rank three strategies by VFC given a table.
11. How much up-front architecture for a 100-KSLOC project with changing requirements?
12. The code has drifted from the design — how do you find out and fix it?

---

## 5. Tab list — flag these pages in your PDF
### Part 1 — pages 1–76 (exact)
| Topic | Pages |
|---|---|
| Quality-attribute scenario, tactics, 7 decision categories | 22-28 |
| Availability (tactics) | 28-35 |
| Performance | 35-40 |
| Usability | 40-44 |
| Other quality attributes | 45-48 |
| Security | 48-53 |
| Modifiability | 53-58 |
| Interoperability | 58-63 |
| Testability | 63-68 |
| Agile & architecture | 71-76 |
| Structures & views (what is SA) | 11-21 |

**Design checklists (the seven-category checklists you asked to keep):**

| Attribute | Design-checklist pages |
|---|---|
| Availability | 33-34 |
| Performance | 38-39 |
| Usability | 43-44 |
| Security | 51-53 |
| Modifiability | 56-58 |
| Interoperability | 61-63 |
| Testability | 67-68 |

### Part 2 — pages 77–136 (±1 page)
| Topic | Pages |
|---|---|
| ASR, QAW, business goals, utility tree | 77-85 |
| Views, 4+1 model | 85-104 |
| Layered techniques | 104-113 |
| Layered / hotel / logistics exercises | 111-115 |
| Evaluation, ATAM | 116-125 |
| Conformance | 124-129 |
| Architecture & testing | 128-131 |
| Reconstruction | 130-136 |

### Part 3 — pages 137–224 (±1 page)
| Topic | Pages |
|---|---|
| Patterns intro | 137-142 |
| Layers | 140-147 |
| Pipes & Filters | 146-151 |
| Blackboard | 150-154 |
| Broker | 154-163 |
| MVC | 164-172 |
| PAC | 170-179 |
| SOA | 183-186 |
| Microkernel | 186-192 |
| Reflection | 191-196 |
| SEI patterns (client-server, P2P, pub-sub, shared data, map-reduce, multi-tier) | 195-204 |
| Tactics & patterns interactions | 202-207 |
| Cloud architecture (NIST, HDFS, IaaS, NoSQL) | 206-219 |
| CAP theorem | 222-224 |

### Part 4 — pages 225–298 (±1 page)
| Topic | Pages |
|---|---|
| Mobile architecture | 225-235 |
| Cloud issues, multi-tenancy, deployment options | 235-240 |
| Big data, Hadoop, Spark, NoSQL, streaming | 241-263 |
| Security technologies | 262-270 |
| Machine learning | 270-275 |
| Project management & governance | 275-286 |
| CBAM (economics) | 286-298 |

---

## 6. Exam-day workflow
1. **Tag** each question (Concept / Scenario / Decision / Case) — 2 min.
2. **Plan** with the skeleton: concern → scenario → tactics/patterns → trade-off → where it lives → validation.
3. **Look up only lists** (tactic names, checklist questions, steps) using the tab list.
4. **Write** short, structured points with the vocabulary; add *because / at the cost of / boundary*.
5. **Leave time** to add one diagram (layers, deployment, or component) wherever the question allows.
