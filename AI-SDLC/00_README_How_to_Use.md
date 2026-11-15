# AI-Augmented SDLC (SEZG534) — EC3 Study Notes

**Prepared for:** Comprehensive Exam (EC3), BITS WILP — used as closed-book preparation notes
**Sources:** Course handout (8 modules), CS9 Requirements & Design deck, CS14 Software Delivery Metrics deck, Lab Sheet 2 (Coding Agent Harness), Sample Questions paper, plus your 16-question analysis bank (expanded to 49 questions with model answers).

---

## 1. What is in this pack

| File | Contents | Use it for |
|---|---|---|
| `01_Concept_Notes.md` | Part A: all 8 modules, one concept per section, each ending with an exam-ready skeleton; Part B: 8 memorizable text flows | Building understanding + memorizing answer skeletons |
| `02_Question_Bank.md` | 49 questions in your 6 categories, all with full model answers | Practising reproduction of answers under time pressure |
| `03_Last_Day_Cheat_Sheet.md` | Definitions, mnemonics, formulas, number anchors, dispatch table | Final 24 hours before the exam |
| `EC3_All_In_One.md` | Everything above in one file | Searching across the whole pack |

---

## 2. Your question-pattern analysis → where it is covered

| Question category (your analysis) | What examiners ask | Covered in |
|---|---|---|
| Conceptual / descriptive | SDLC evolution, AI fundamentals, prompting | QBank Category A; Modules 1–2 |
| Scenario-based application | Apply AI tools to a given SDLC scenario | QBank Category B; Modules 3–6 scenarios |
| Compare and justify | Compare AI workflows, tools, validation strategies | QBank Category C; comparison tables in every module |
| Diagram / workflow | AI-augmented SDLC workflow, HITL, CI/CD | QBank Category D; Part B flows (arrow notation only) |
| Critical analysis | Governance, hallucinations, circular validation | QBank Category E; Modules 5, 7, 8 |
| Case-study based | Legacy modernization, AI-assisted development labs | QBank Category F; Module 4 + Lab Sheet 2 section |

**Pattern insight from the sample paper:** every question is a short scenario (4–6 lines) followed by 2–3 directive verbs (Evaluate / Analyze / Explain / Identify), each worth ~3 marks. Your answers must therefore: (1) name the concept, (2) define it in one line, (3) apply it to the scenario's specifics, (4) give one example or number. Three sentences per mark is a safe density.

---

## 3. Closed-book strategy (because you are effectively writing it closed-book)

1. **Memorize skeletons, not paragraphs.** Every concept section in `01_Concept_Notes.md` ends with an "Exam skeleton" — a numbered list of 5–8 memory points. In the exam, reproduce the skeleton first, then expand each point into 2–3 sentences.
2. **Memorize numbers as anchors.** Numbers make answers concrete and are hard to invent under pressure: 4 DORA metrics + rework rate, 5 pillars of the Developer AI Impact Framework, 2 generation attempts in the harness, 30/90-day code turnover windows, 3 J-curve forces, temperature 0.1 in the live harness. Full list in the cheat sheet.
3. **Memorize one comparison table per module.** Compare-and-justify questions are almost guaranteed (your pattern analysis says so). Each module has one master table; if you can reproduce the table you can answer any comparison question.
4. **Diagrams: write flows, do not draw.** Per your instruction, all diagrams are text-only arrow flows (A → B → C). In the exam, write the flow on one line, then describe each stage in words. That earns diagram-question marks without drawing.
5. **Practise the 3-mark rhythm.** Sample paper parts are ~3 marks each: definition (1) + application to scenario (1) + example/justification (1). Practise writing 120–180 word answers against the clock.
6. **Use the dispatch table** (cheat sheet, last section): "If the question says X → open with Y." It is your fastest route into any answer when the clock starts.

---

## 4. Suggested 10-day revision plan

| Days | Task |
|---|---|
| 1–2 | Modules 1–2 concept notes + QBank A |
| 3–4 | Modules 3–4 + QBank B1–B5, C1–C3 |
| 5 | Modules 5–6 + QBank E1–E3, D2 |
| 6 | Modules 7–8 + QBank E4–E8, C4–C7 |
| 7 | Part B flows + QBank D (all workflow questions) |
| 8 | QBank F case studies + Lab Sheet 2 concepts |
| 9 | Attempt 6 random questions from the bank, closed-book, timed (25 min each) |
| 10 | Cheat sheet only; rewrite all skeletons from memory once |

---

## 5. Syllabus coverage check (against the course handout)

- Module 1 (Evolution, velocity, risk, economics) — Sections 1.1–1.6, QBank A1, B1, B8, E4, E5
- Module 2 (AI fundamentals, prompting, agents) — Sections 2.1–2.7, QBank A2–A7, B4, B10
- Module 3 (Requirements, SDD, ADRs, threat modeling) — Sections 3.1–3.9, QBank B2, B3, B9, B11, C2, C3, D7, F7
- Module 4 (Coding, refactoring, guardrails) — Sections 4.1–4.6, QBank A8, B10, C1, C5, D6, E6, F1–F4
- Module 5 (Testing, QA, circular validation) — Sections 5.1–5.6, QBank E1, E3, C6, C7, F2
- Module 6 (CI/CD, release) — Sections 6.1–6.5, QBank D2, D5, B7
- Module 7 (Governance, HITL, liability) — Sections 7.1–7.4, QBank E2, D5, F3
- Module 8 (Feedback loops, metrics, health) — Sections 8.1–8.7, QBank B5, B6, C4, D3, E5, E8, F6
- Lab Sheet 2 harness — Part A sections 2.5, 4.5, 4.6 + Flows F3/F4 + QBank D4, F2, E7
- Sample paper questions — QBank E4 (Q1) and B8 (Q2)
