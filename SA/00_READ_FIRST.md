# Software Architecture (SEZG651 / SSZG653) — EC3 Open-Book Study Notes
*How to use these notes in the exam hall*

---

## 1. What you have

| File | What it is | Use it when the question says… |
|---|---|---|
| `01_Concepts.md` | Understanding of every topic: what / why / how / where it fits / cost / mini-example | "Explain…", "Define…", "Discuss…", "Compare…" |
| `02_Scenario_Based.md` | Situation → diagnose → mitigate → recover, plus edge cases | "A system has failed/slow/attacked… what will you do?" |
| `03_Design_Decisions.md` | Choosing between options: criteria, trade-offs, boundaries | "Which pattern/tactic/technology would you choose and why?" |
| `04_Case_Studies.md` | End-to-end architecture for a business problem | "Design the architecture for…" |

## 2. REF key — how to find the slide quickly

You will carry the four condensed PDFs. A reference looks like **`[2:61]`**:

> **`[PartFile : page-in-that-PDF]`** → `[2:61]` = page 61 of `SA_Part2_Exam_Condensed.pdf`. Ranges: `[1:49-57]`.

- Page numbers are **PDF page numbers of the condensed files** (page 1 = first page of that file), *not* original slide numbers.
- Pages carry the original slide title, so you can also search the title text.
- **⚠ NOT-CARRIED** = the topic was in the original lecture notes but is *not* in your condensed PDFs (e.g. "why architecture matters", the four contexts). The note itself has enough to answer.
- **🆕** = standard textbook content (Bass–Clements–Kazman) that the slides only imply. Safe to use; examiners expect it.

## 3. Honest limits

- **No previous-year paper archive was available to me.** The "likely questions" lists are inferred from (a) the exercises and worked examples inside the slides (these are the strongest hints), and (b) the question style you described (analysis / experience-based, not recall).
- Content comes from the **original** lecture notes (Parts 1–4), not only the condensed ones.
- Coverage: all four parts of the syllabus — foundations & quality attributes, ASR/documentation/layering/evaluation, patterns & cloud, mobile/big data/security tech/ML/management/economics (CBAM).

## 4. Universal answer skeleton (use for almost any question)

1. **Name the concern** — one line: which quality attribute / decision is at stake, and *why it matters to the business*.
2. **Express it as a scenario** — Source · Stimulus · Artifact · Environment · Response · Response measure (makes your answer testable).
3. **Apply tactics / patterns** — name them *by their textbook names*, grouped by category (e.g. "detect faults → heartbeat; recover → passive redundancy").
4. **State the trade-off** — every tactic helps one attribute and costs another ("redundancy ↑ availability, ↑ cost; intermediary ↑ modifiability, ↓ performance").
5. **Show where it lives** — which view / layer / tier / component (module vs C&C vs allocation).
6. **Close with validation** — how you would check it: ATAM scenario, utility tree priority, test, metric, or CBAM value-for-cost.

> **Marks-winning habit:** whenever you write a design choice, add *"because…" + "at the cost of…"*. Slide-recall answers lack this; analysis answers are graded on it.

## 5. Power vocabulary (drop these terms naturally)

*ASR · quality attribute scenario · stimulus/response/response measure · tactic vs pattern · design decision categories (allocation of responsibilities, coordination model, data model, resource management, mapping, binding time, technology) · module / C&C / allocation structures · view vs structure · separation of concerns · information hiding · coupling & cohesion · single point of failure · fault vs failure · redundancy (active/passive/spare) · graceful degradation · attack surface · defence in depth · sensitivity point · trade-off point · risk / non-risk / risk theme · utility tree · architectural drift · technical debt · CAP (consistency, availability, partition tolerance) · eventual consistency · multi-tenancy · horizontal vs vertical scaling · value for cost (VFC).*

## 6. Time strategy for the hall

- Spend the **first 5 minutes** reading all questions and tagging each as Concept / Scenario / Design-decision / Case study → jump to the matching file.
- For a case study, write the **blueprint headings first** (see `04_Case_Studies.md` §1), then fill — headings alone earn structure marks.
- Don't copy slides; the REF is for *checking the exact list* of tactics/steps, while your **sentences come from understanding** (the "✍" lines in each note).
