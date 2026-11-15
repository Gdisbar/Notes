# EC3 Revision Notes — AI-Augmented SDLC (SEZG534)

**Scope (strict):** These notes are built **only** from the course files you supplied for sessions 8–14:

| Source file | What it covers | Where it lands in the notes |
|---|---|---|
| `Merged_Presentation_2026-11-15.pptx` | CS8 AI-Native SDLC, CS9 Spec-Driven Development, CS10 Design & Architecture, CS11 Coding (two decks), CS13 Software Testing, CS14 Software Delivery Metrics | `01_Concept_Notes.md` Parts I–VI |
| `Spec_Driven_Development_GitHub_Spec_Kit_Lab_Sheet.pdf` (Lab 1) | SDD hands-on with GitHub Spec Kit in VS Code | `01_Concept_Notes.md` Part VII |
| `Coding Agent_Harness_Lab_Sheet.pdf` (Lab 2) | Feedforward/feedback harness for coding agents | `01_Concept_Notes.md` Part VII + question bank case studies |
| Your 16-question analysis | Question **pattern** only (6 categories) | `02_Question_Bank.md` — every question re-grounded in the files above |

Nothing from sessions 1–7 (SDLC evolution history, AI/ML fundamentals, prompting theory, RAG internals, DevOps basics, etc.) is included. Where a sample question touched an out-of-scope topic (e.g., "foundation models and transformer architecture"), the question was **rewritten to the same pattern using only in-scope content** (e.g., non-determinism and context windows), and the mapping index at the end of the question bank shows exactly what happened to each of your 16 questions.

## Files

| File | Use it for |
|---|---|
| `01_Concept_Notes.md` | First pass. 7 parts, every concept with a definition, mechanism, and an **Exam skeleton** you must be able to reproduce from memory. |
| `02_Question_Bank.md` | Second pass. 45 questions in 6 categories (A–F), each with a full model answer. Write answers by hand before reading the model. |
| `03_Last_Day_Cheat_Sheet.md` | Day-before and morning-of. One-liners, mnemonics, formulas, number anchors, dispatch table. |
| `EC3_All_In_One.md` | Single file with everything, for search/phone. Do not study from it linearly. |

## Closed-book strategy (even though it is nominally open-book)

1. **Skeleton-first, not prose-first.** Every concept ends with a 5–8 line skeleton. If you can reproduce the skeleton from memory, you can rebuild the paragraph around it in the exam. Prose is decoration; the skeleton is the answer.
2. **Active recall loops.** For each part: read once → close notes → write the skeleton → check → mark misses. A miss gets a red star; red-starred skeletons are your day-before list.
3. **Question-bank discipline.** Do not read model answers first. Attempt, then compare, then note which skeleton lines you missed. 45 questions × ~10 minutes each ≈ 8 sessions.
4. **Anchor on numbers and named artifacts.** Examiners reward specifics: "15–20% churn acceptable", "max 2 repair attempts", "Wave 1 = no dependencies", "`requirements.md / design.md / tasks.md`". The cheat sheet collects every number once.
5. **Flows as text arrows.** Every workflow answer must be written as `A → B → C` first, then explained line by line. Practise redrawing the six master flows: bottleneck shift, Spec Kit command chain, harness loop (Lab 2), cybernetic loop (SEI), circular-vs-independent validation, operations feedback loop.

## Ten-day plan (≈ 3 h/day; compress to 6 days at 5 h if needed)

| Day | Work | Output you must be able to do |
|---|---|---|
| 1 | Part I–II (CS8+CS9) | Draw spec spectrum; list 9 Spec Kit commands in order; 4 SDD risks + mitigations |
| 2 | Part III (CS10) | Deep vs shallow, sink vs pipe, harness vs orchestrator, 3-layer governance — from memory |
| 3 | Part IV first half (CS11: patterns, guardrails, Lattice) | 5 patterns + failure modes; 4 guardrail types; I-C-O layers |
| 4 | Part IV second half (CS11: SEI, non-determinism, review) | Cybernetic loop; actual vs apparent; P-E-C-E → E-C-E; 8 review steps |
| 5 | Part V (CS13) | 6 testing uses; churn & complexity numbers; circular validation remedies; agent-testing triad |
| 6 | Part VI (CS14) | DORA 4 + status under AI; hierarchy A-E-P-Q-B; 5 pillars; ROI + J-curve formulas |
| 7 | Part VII (Labs) | Lab 1 artifact chain; Lab 2 feedforward files, sensor order, bounded repair; verification matrix |
| 8 | Question bank categories A, B (22 Qs) | Attempt 11, mark misses |
| 9 | Question bank categories C, D, E, F (23 Qs) | Attempt all; redraw every flow by hand |
| 10 | Cheat sheet + red-star skeletons + 5 hardest questions re-attempted | Recite all 50 one-liners |

## Question-category → study-section map

| Category (your pattern) | Weight in bank | Primary sections |
|---|---|---|
| A. Conceptual / descriptive | 12 Qs | Parts I–VII (all skeletons) |
| B. Scenario-based application | 10 Qs | Part I–II (requirements), IV (guardrails/patterns), V (testing), VI (metrics), Lab 2 |
| C. Compare and justify | 7 Qs | Spec spectrum (II), deep/shallow + sink/pipe + harness/orchestrator (III), testing comparisons (V), frameworks (VI) |
| D. Diagram / workflow | 6 Qs | Bottleneck shift (I), Spec Kit chain (II), validation loop (V), harness loop (VII) |
| E. Critical analysis | 5 Qs | Governance (III–IV), SDD critique (II), DORA critique + J-curve (VI) |
| F. Case-study / lab | 5 Qs | Lab 1 (VII), Lab 2 (VII), SEI case (IV), Uber case (V) |

## Exam answer template (works for every category)

1. **Define** (1–2 lines, exact terms from the notes).
2. **Flow** (the `A → B → C` arrow line for whatever is asked).
3. **Mechanism** (3–5 numbered points, one named artifact or number each).
4. **Risk / trade-off** (2–3 failure modes with the named guardrail or metric that controls them).
5. **Verdict / recommendation** (one sentence that answers the exact verb in the question: recommend, justify, critique).

If a question says "compare", write a two-column mini-table in the answer script; if it says "design", the flow line comes before any prose.
