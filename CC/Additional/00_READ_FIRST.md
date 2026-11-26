# Cloud Computing (CSI ZG527) — EC3 Open-Book Notes (v3: you carry every slide)
*How to use these notes in the exam hall*

---

## 1. What you have

| File | Purpose | Use it when… |
|---|---|---|
| `00_STUDY_GUIDE.md` | **What to study and what to just look up**, drills, time split, tab list | planning your preparation |
| `01_Concepts.md` | Lean understanding of each topic: the idea, how topics connect, trade-offs, vocabulary | "Explain / define / differentiate…" |
| `02_Scenario_Based.md` | Situation → diagnose → mitigate → recover, edge cases, **worked calculations** | "A host/pod/zone fails… a workload is slow… what do you do?" |
| `03_Design_Decisions.md` | Choosing between options: criteria, trade-offs, boundaries | "Which model/service/orchestrator would you choose and why?" |
| `04_Case_Studies.md` | End-to-end cloud architecture/migration for a business problem | "Design / migrate / size the solution for…" |

## 2. REF key — finding a slide in your compressed PDF

You carry **`Cloud_Computing_Slides.pdf`** (320 pages, **two slides stacked per page**, every slide present). A reference looks like **`[M:100]`** or **`[M:42-43]`**:

> **`[M:page]`** = page of that PDF. Each page shows two slides: the upper one, then the lower one.

**Layout**
- Pages **1–5**: table of contents · pages **5–8**: **glossary** (abbreviations, **formulas with worked numbers on pages 6–7**, key concepts) · pages **9–320**: the 623 slides in course order.
- **Slide *n* is on page 8 + ⌈n ÷ 2⌉.** This matters because the glossary quotes original slide numbers — for example "slide 156" → page 86.
- Each reference points to where a topic **starts**; a long topic may continue on the following pages.

*Do not use these notes with the earlier 310-page or 150-page Cloud PDFs; their page numbers differ.*

## 3. Honest limits

- **No previous-year paper archive was available to me.** "Likely questions" are inferred from the worked examples in the slides (CPU/memory overcommit, live-migration rounds, capacity sizing, bin-packing, SaaS vs on-premises cost, Kubernetes allocatable) and from the question style you described.
- Because you carry every slide, these notes **do not re-list** what is on the slides (service names, commands, object kinds, step lists). They give the *reasoning* and where to look the details up.
- Product facts (for example Lambda's 15-minute limit, Spot's two-minute notice) are as printed in the deck; the deck itself says to check current provider documentation.
- 🆕 = standard textbook content that the slides imply but do not spell out.

## 4. Universal answer skeleton

1. **Frame the need** — which cloud characteristic or business goal is at stake (elasticity, cost, availability, isolation, compliance, time-to-market)?
2. **Name the mechanism** — hypervisor, namespaces/cgroups, overlay network, control plane, lease, SLO…
3. **Explain how it works** — three to five steps or a small diagram.
4. **Give the cost or limit** — overhead, lock-in, complexity, noisy neighbour, cold start.
5. **Compare** with the nearest alternative (VM vs container, IaaS vs PaaS, Swarm vs K8s…).
6. **Close with a decision rule or a number** — "choose X when…", or a short calculation.

> **Marks-winning habit:** end every design answer with *"because … at the cost of …; not suitable when …"*, and add one number or formula where the topic allows it.

## 5. Power vocabulary

*abstraction · virtualization · hypervisor (Type 1/2) · trap-and-emulate · sensitive vs privileged instructions · binary translation · paravirtualization (hypercall) · hardware-assisted · overcommit ratio · ballooning · live migration (pre-copy, stop-and-copy, post-copy) · shared-nothing · namespaces / cgroups · image layers / copy-on-write · declarative vs imperative · desired state · reconciliation · control plane · overlay network · Raft quorum · requests vs limits · horizontal vs vertical scaling · shared responsibility · tenancy (silo / pool / bridge) · noisy neighbour · SLA / SLO / SLI · RTO · lease (best-effort / advance reservation) · bin-packing · IaC (idempotent, drift) · 12-factor · blue-green / canary · golden signals · PUE.*

## 6. In the exam hall
- First 5 minutes: read every question and tag it **Concept / Scenario / Decision / Case**; jump to the matching file.
- For a numeric question, open the **glossary formulas (pages 6–7)** first and copy the *step layout*.
- For a case, write the blueprint headings first (`04_Case_Studies.md` §1).
- Use the PDF for *lists and numbers*; write sentences from understanding.
