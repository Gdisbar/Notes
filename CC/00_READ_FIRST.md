# Cloud Computing (CSI ZG527) — EC3 Open-Book Study Notes
*How to use these notes in the exam hall*

---

## 1. What you have

| File | What it is | Use it when the question says… |
|---|---|---|
| `01_Concepts.md` | Understanding of every topic: what / why / how / where it fits / cost / minimal example | "Explain…", "Define…", "Differentiate…", "Discuss…" |
| `02_Scenario_Based.md` | Situation → diagnose → mitigate → recover, plus edge cases and **worked calculations** | "A VM/host/pod/zone fails… a workload is slow… what do you do?" |
| `03_Design_Decisions.md` | Choosing between options: criteria, trade-offs, boundaries | "Which model/service/hypervisor/orchestrator would you choose and why?" |
| `04_Case_Studies.md` | End-to-end cloud architecture/migration for a business problem | "Design / migrate / size the cloud solution for…" |

## 2. REF key — how to find the slide quickly

You will carry **one** condensed PDF: `Cloud_Computing_Exam_Condensed.pdf` (310 pages). A reference looks like **`[C:96]`**:

> **`[C:page]`** = page number **in the condensed PDF** (page 1 = first glossary page). Ranges: `[C:30-34]`.

- **Pages 1–7 are the Glossary** (abbreviations, **worked formulae**, key concepts). It is your fastest lookup:
  - `[C:1-2]` abbreviations · **`[C:3]` formulae & worked calculations** (overcommit, capacity sizing, live-migration pre-copy, Amdahl, Swarm quorum, availability) · `[C:4-7]` key-concept definitions.
- Pages 8–310 are the slides in original order; each page keeps its original slide title (so you can also search by title).
- The glossary's own "slide NNN" numbers refer to the **original** deck, *not* to your condensed pages — use `[C:…]` numbers from these notes instead.
- **⚠ NOT-CARRIED** = in the original lecture notes but not in your condensed PDF. The note itself holds enough to answer.
- **🆕** = standard textbook content that the slides imply but don't spell out. Safe to use.

## 3. Honest limits

- **No previous-year paper archive was available to me.** All "likely questions" lists are inferred from (a) the **worked examples in the slides** (CPU/memory overcommit, live-migration rounds, capacity sizing, bin-packing, SaaS vs on-prem cost, K8s node allocatable — these are the strongest hints of numeric questions) and (b) your description of the exam (analysis / experience, not recall).
- Content is built from the **original** deck (639 pages), not only the condensed one.
- The 6 R's of migration slides are images; the list in the notes is the standard AWS one (as the glossary also states `[C:6]`).
- Product facts (AWS limits such as Lambda's 15 minutes, Spot's 2-minute notice) are as printed in the deck; the deck itself says to check current provider documentation.

## 4. Universal answer skeleton

1. **Frame the need** — which cloud characteristic or business goal is at stake (elasticity, cost, availability, isolation, compliance, time-to-market)?
2. **Name the mechanism** — use textbook names (hypervisor, namespaces/cgroups, overlay network, control plane, lease, SLO…).
3. **Explain how it works** — 3–5 steps or a small diagram (host → hypervisor → VMs; client → daemon → registry; API server → scheduler → kubelet).
4. **Give the cost / limit** — every option has a price (overhead, lock-in, complexity, noisy neighbour, cold start).
5. **Compare** with the nearest alternative in a 2–3 row table (VM vs container, IaaS vs PaaS, Swarm vs K8s…).
6. **Close with a decision rule or a number** — "choose X when…", or a short calculation (ratio, hosts, downtime minutes).

> **Marks-winning habit:** end every design answer with *"because … at the cost of … ; not suitable when …"*. Add **one number or formula** where the topic allows it — numeric reasoning stands out in an open-book paper.

## 5. Power vocabulary

*abstraction · virtualization · hypervisor (Type 1/2) · trap-and-emulate · sensitive vs privileged instructions · binary translation · paravirtualization (hypercall) · hardware-assisted (VT-x/AMD-V, "Ring −1") · overcommit ratio · ballooning / page sharing / swapping · live migration (pre-copy, stop-and-copy, post-copy) · shared-nothing · namespaces / cgroups · image layers / copy-on-write · declarative vs imperative · desired state · reconciliation · control plane · overlay network / ingress · Raft quorum · requests vs limits · horizontal vs vertical scaling · shared responsibility · tenancy (silo / pool / bridge) · noisy neighbour · SLA / SLO / SLI · RTO · lease (best-effort / advance reservation) · bin-packing · IaC (idempotent, drift, state) · 12-factor · blue-green / canary · golden signals · PUE.*

## 6. Time strategy

- First 5 min: read all questions, tag each **Concept / Scenario / Decision / Case** and go to the right file.
- For a numeric question, open **`[C:3]`** first (formulae with worked numbers) and copy the *step layout*.
- For a case, write the blueprint headings (`04_Case_Studies.md` §1) first; fill after.
- Don't copy slide text. Use the REF to check the exact *list or definition*; your sentences come from understanding (the ✍ lines).
