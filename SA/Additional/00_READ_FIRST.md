# Software Architecture (SEZG651 / SSZG653) — EC3 Open-Book Notes (v3: you carry every slide)
*How to use these notes in the exam hall*

---

## 1. What you have

| File | Purpose | Use it when… |
|---|---|---|
| `00_STUDY_GUIDE.md` | **What to study and what to just look up**, drills, time split, tab list | planning your preparation |
| `01_Concepts.md` | Lean understanding of each topic: the idea, how topics connect, trade-offs, vocabulary | "Explain / define / compare…" |
| `02_Scenario_Based.md` | Situation → diagnose → mitigate → recover, edge cases, worked calculations | "A system fails / is slow / is attacked… what do you do?" |
| `03_Design_Decisions.md` | Choosing between options: criteria, trade-offs, boundaries | "Which pattern/tactic/technology and why?" |
| `04_Case_Studies.md` | End-to-end architecture for a business problem | "Design the architecture for…" |

## 2. REF key — finding a slide in your merged PDF

You carry **`26S1-SS-ZG653-SA-merged.pdf`** (298 pages, **four slides per page**, every slide present). A reference looks like **`[M:136]`** or **`[M:116-124]`**:

> **`[M:page]`** = page of the merged PDF. Each page shows four slides in reading order (top-left, top-right, bottom-left, bottom-right). The **original slide number is printed on each slide**, so you can find the exact one in seconds.

**Where each part starts**

| Part | Pages | Content |
|---|---|---|
| Part 1 | 1–76 | foundations, quality attributes with tactics, all design checklists, agility |
| Part 2 | 77–136 | ASR, views, layered techniques & exercises, ATAM, conformance, reconstruction |
| Part 3 | 137–224 | patterns, cloud architecture, CAP |
| Part 4 | 225–298 | mobile, big data, security technologies, ML, management, CBAM |

- Each reference points to the page where the topic **starts**; a topic may spill onto the next page, so check the neighbouring page if you do not see the end of a list.
- Page ranges for Parts 2–4 in the tab list are accurate to about ±1 page.
- *Do not use these notes with the earlier 220-page or four per-part PDFs; their page numbers differ.*

## 3. Honest limits

- **No previous-year paper archive was available to me.** "Likely questions" are inferred from the exercises and worked examples inside the slides and from the question style you described.
- Because you carry every slide, these notes deliberately **do not re-list** what is on the slides (tactic names, checklist questions, step-by-step procedures, structures of each pattern). They give the *reasoning* — and where to look the lists up.
- 🆕 = standard textbook content (Bass–Clements–Kazman) that the slides only imply; safe to use.

## 4. Universal answer skeleton

1. **Name the concern** — which attribute/decision, and why it matters to the business.
2. **Express it as a scenario** — source · stimulus · artifact · environment · response · response measure.
3. **Apply tactics/patterns by their textbook names** — look the exact names up in the PDF; group them by category.
4. **State the trade-off** — every tactic helps one attribute and costs another.
5. **Show where it lives** — module / C&C / allocation view; layer; tier.
6. **Close with validation** — ATAM scenario, utility-tree priority, test, metric, or CBAM value-for-cost.

> **Marks-winning habit:** attach *"because…"* and *"at the cost of…"* to every design choice, and add a boundary — *"this would be wrong if…"*.

## 5. Power vocabulary

*ASR · quality attribute scenario · stimulus / response / response measure · tactic vs pattern · the seven design-decision categories · module / C&C / allocation structures · view vs structure · separation of concerns · information hiding · coupling & cohesion · single point of failure · fault vs failure · hot / warm / cold redundancy · graceful degradation · attack surface · defence in depth · sensitivity point · trade-off point · risk / non-risk / risk theme · utility tree · architectural drift · technical debt · CAP · eventual consistency · multi-tenancy · horizontal vs vertical scaling · value for cost (VFC).*

## 6. In the exam hall

- First 5 minutes: read every question and tag it **Concept / Scenario / Decision / Case**; jump to the matching file.
- Use the PDF **only for lists** (tactic names, checklist questions, steps); write sentences from understanding.
- For a case, write the blueprint headings first (`04_Case_Studies.md` §1).
