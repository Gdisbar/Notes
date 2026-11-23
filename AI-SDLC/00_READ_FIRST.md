# AI-Augmented SDLC — EC3 Open-Book Study Notes
*How to use these notes in the exam hall*

---

## 1. What you have

| File | What it is | Use it when the question says… |
|---|---|---|
| `01_Concepts.md` | Understanding of every topic: what / why / how / where it fits / cost / minimal example | "Explain…", "Define…", "Differentiate…", "Discuss…" |
| `02_Scenario_Based.md` | Situation → diagnose → mitigate → recover, plus edge cases and **worked calculations** | "An AI agent/pipeline/metric behaves badly… what do you do?" |
| `03_Design_Decisions.md` | Choosing between options: criteria, trade-offs, boundaries | "Which approach/tool/control/metric would you choose and why?" |
| `04_Case_Studies.md` | End-to-end AI-augmented delivery or governance strategy | "Propose / design / evaluate the AI-augmented approach for…" |

## 2. REF key — how to find the slide quickly

You will carry **one** condensed PDF: `AI_SDLC_Exam_Condensed.pdf` (129 pages). A reference looks like **`[A:48]`**:

> **`[A:page]`** = page number **in the condensed PDF** (page 1 = first page of that file). Ranges: `[A:34-35]`.

- Each page keeps its original slide title, so you can also search by title.
- **⚠ NOT-CARRIED (orig. slide N)** = the topic is in the original lecture deck but was **not** kept in your condensed PDF. The note itself has enough to answer. *Several high-value items fall in this group* (e.g., **code churn & cyclomatic complexity formulas**, MCP architecture, NIST trustworthy-AI characteristics, SDLC-specific governance failure modes, guides/sensors) — read those parts of the notes carefully.
- **🆕** = standard textbook/industry content that the slides imply but don't spell out. Safe to use.

## 3. Honest limits

- **No previous-year paper archive was available to me.** All "likely questions" are inferred from the slides' worked examples, tables and activities (the **Cost-of-Change Mapping activity**, the **ROI formula**, **churn/complexity formulas**, **decision-class gates**, **risk-mitigation table** are the strongest hints) and your description of the exam (analysis / experience, not recall).
- Content comes from the **original** 269-slide deck, not only the condensed one.
- The deck is a fast-moving topic; figures such as adoption statistics, tool names (Spec Kit, Kiro, Lattice) and framework versions are **as printed in the slides**.
- Several slides are infographics (e.g., maturity phases, Spec Kit commands, decision-class tables). I read their text/table content; where an image carried only a diagram, the note says so.

## 4. Universal answer skeleton

1. **Frame it** — what changed because AI makes *code cheap but verification, judgement and accountability expensive*.
2. **Name the mechanism** — use textbook names (agent, harness, context window, RAG, SDD, guardrail, evals, HITL/HOTL, decision gate…).
3. **Explain how it works** — 3–5 steps or a small loop (Spec → Harness → Loop; Perceive → Reason → Act → Evaluate).
4. **State cost/limit/risk** — hallucination, drift, verification tax, token cost, rubber-stamp fatigue.
5. **Compare** with the traditional approach in a 2–3 row table.
6. **Close with a control + a metric** — "we gate it with … and measure it with …".

> **Marks-winning habit:** every AI-SDLC answer should contain **a benefit, a risk, a control (guardrail/gate/sensor), and a measure**. Mention that **humans remain accountable**.

## 5. Power vocabulary

*AI-augmented vs AI-native SDLC · intent-to-spec · agentic workflow · perceive-reason-act-evaluate · harness (guides/feedforward, sensors/feedback; computational vs inferential) · context engineering · prompt engineering (zero/few-shot, CoT) · RAG (ingest-retrieve-augment-generate) · context window / attention dilution / "lost in the middle" · tokenization · tokenomics · MCP (host/client/server) · spec-driven development (spec-first / spec-anchored / spec-as-source) · constitution / plan / tasks · ADR · deep vs shallow modules · sinks vs pipes · invariants · blast radius · cost-of-change · knowledge priming · design-first · context anchoring · feedback flywheel · guardrail (input/context/output; action) · non-determinism (emergent/concurrent/exceptional) · circular validation · mutation testing · self-healing tests · defect prediction (churn, complexity) · test-impact analysis · LLM-as-judge · evals · drift · DORA (+ rework rate) · AI code share · code turnover · complexity-adjusted throughput · J-curve · verification tax · HITL / HOTL / autonomous · reversible vs irreversible · decision gate · provenance · EU AI Act · NIST AI RMF (Govern-Map-Measure-Manage) · ISO/IEC 42001.*

## 6. Time strategy

- First 5 min: tag each question **Concept / Scenario / Decision / Case** and jump to the right file.
- Numeric questions (churn, complexity, ROI, tokens): open `02_Scenario_Based.md` §10 — formulas with worked numbers.
- For a case, write the blueprint headings (`04_Case_Studies.md` §1) first.
- Don't copy slides; use the REF to verify a *list or definition*, and write your own sentences from the ✍ guidance.
