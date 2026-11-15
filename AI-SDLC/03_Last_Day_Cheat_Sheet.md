# 03 — Last-Day Cheat Sheet

## 1. Fifty one-liners (recite until fluent)

**CS8 — AI-Native SDLC**
1. Traditional SDLC is process-heavy for accountability and control at each step.
2. AI collapses build to agent speed; plan, review/test, deploy stay human speed — the bottleneck moves there.
3. Line-by-line review can't scale once agents write most of the diff.
4. Governance costs rise: exceptions still route through weekly/monthly committees.
5. AI-native SDLC = automated handover + triggering of subsequent plays.
6. SDD = guardrails, requirements, constraints, acceptance criteria, edge cases defined up front, then AI generates from shared context.
7. The spec is the connective tissue across the lifecycle.
8. SDD benefits: less ambiguity, shared-truth alignment, faster generation, predictable delivery.
9. Spec Kit = Specify CLI + templates/scripts (Microsoft, open-source).
10. Kiro: Requirements → Design → Tasks → `requirements.md`, `design.md`, `tasks.md`.
11. Kiro waves: sequential across waves, concurrent within a wave.
12. Kiro steering files: `product.md`, `tech.md`, `structure.md`.
13. Playbook: six stages Plan-Design-Build-Test-Deploy-Maintain; each stage ends writing version control, next begins reading it.
14. Early stages use .md artifacts; from Build onward the artifact is code; the commit chain is the audit trail.
15. SDD challenges: brownfield hard, false sense of control, reviewing markdown over code.

**CS9 — Spec-Driven Development**
16. Elicitation → synthesis → management; elicitation is the least automated stage.
17. User story: "As a [role], I want [action], So that [benefit]"; acceptance criteria: Given/When/Then (BDD).
18. MoSCoW = Must/Should/Could/Won't; backlog re-prioritized by the Product Owner; time fixed, scope flexes.
19. AI synthesis scores ambiguity against INCOSE guidelines and EARS syntax.
20. Spec declares intent; code realizes it; edit the spec, AI regenerates code and tests.
21. SDD risks: dependency/API hallucinations, shallow edge/NFR coverage, spec-to-code drift, cascading errors.
22. Mitigations: simplicity rules in constitution; plan.md as senior-review PR gate; CI fails drift; reference files in context.
23. AD = justified choice for an architecturally significant requirement; ASR = requirement with measurable architectural effect; ADR = one AD + rationale.
24. ADR benefits: history, communication, maintenance, no repeated debates, change management; adrkit drafts/retrieves/checks.
25. Risk-control activities: requirements validation, architecture risk analysis, security & threat modelling, task decomposition + dependency validation.

**CS10 — Design & Architecture**
26. Agents have zero institutional memory — they reset every hour/day and live in the gap between mental model and code.
27. Deep module = narrow interface, big hidden functionality, minimal caller load; shallow = large interface, little hidden work.
28. Pipe leaks execution across boundaries; sink receives, works, terminates within its boundary; sink = deep module.
29. Pipe → agent traces the whole chain → context exhaustion, attention dilution, hallucination; sink → black box, contained blast radius.
30. The architecture is the prompt; the codebase outweighs the prompt; context engineering is a system design problem.
31. The architect authors the constraint: versioning policy, behavioral constraints, dependency rationale, exception logic.
32. Review changes from catching violations to confirming declared behavior.
33. Harness: LLM owns control flow (exploration + human steering); orchestrator: code owns it (unattended, predictable failure).
34. Guiding rule: the less human oversight, the more control flow must live in deterministic code.
35. Governance layers: Constraints (cursorrules/ADLs) → Conformance (fitness functions, gates, linting) → Knowledge (repo maps, ADRs from traces).
36. AI made CRUD cheap; invariants, data topology, non-deterministic boundaries stayed expensive.
37. Frame architecture decisions by blast radius, auditability, latency budgets, invariant protection — not coding time.

**CS11 — Coding**
38. Frustration loop: generate → review → not aligned → regenerate → heavy edit/abandon; caused by missing context and generic patterns.
39. Five patterns K-D-C-E-F: knowledge priming, design-first, context anchoring, encoding standards, feedback flywheel → shared mental model.
40. Guardrails: action / code-quality / security / workflow, enforced by software outside the model; layers Input → RAG → Output; implementations: rules, model-judge, middleware, HITL.
41. Lattice: skills over prompts, atoms→molecules, .lattice/ living context; refiners overlay/override; collaborative judgment presents trade-offs.
42. SEI: RevAPI (static) finds breaking changes; LLM writes recipes; engines apply; build+tests feed back until clean build; write once, roll out everywhere.
43. AI code = apparent non-determinism; spans Emergent (primary), Concurrent, Exceptional of the four sources P-E-C-E.
44. Review AI code in 8 steps — tests and static analysis first, verify intent, quality, dependencies, AI pitfalls, collaborative, automate, improve.

**CS13 — Testing**
45. Six AI testing uses T-R-S-D-V-L: test generation, risk-based prioritization, self-healing, defect prediction, visual AI, log parsing.
46. Churn: absolute = added+deleted+edited; 15–20% acceptable. Cyclomatic: independent paths; <10 low, 11–20 moderate, >20 refactor.
47. Circular validation = code + tests from the same assumptions; break it with spec-grounded tests, TDD, verification gates + mutation testing.
48. Evals: cosine/ROUGE/BLEU, LLM-as-a-judge with rubrics, thresholds (95%+ semantic alignment); agents judged on tool-calling, goal/plan execution, memory drift (component → trajectory → E2E tiers).
49. Continuous observability: data/model drift monitoring; shadow testing compares trajectory success rates safely.
50. DORA: deployment frequency, lead time, MTTR, change failure rate; AI-era: MTTR holds, CFR partial, DF/LT misleading; add rework rate (2025 fifth metric), AI code share, durability, complexity-adjusted output, innovation rate.

## 2. Mnemonics

| Mnemonic | Expands to |
|---|---|
| **A-A-F-P** | SDD benefits: Ambiguity, Alignment, Faster, Predictable |
| **P-D-B-T-D-M** | Playbook stages: Plan, Design, Build, Test, Deploy, Maintain |
| **R-D-T** | Kiro files: requirements, design, tasks |
| **P-T-S** | Kiro steering: product, tech, structure |
| **First / Anchored / Source** | Spec spectrum — human code-editing power decreases → |
| **C-S-C-P-C-T-A-I-C** | Spec Kit chain: constitution, specify, clarify, plan, checklist, tasks, analyze, implement, converge |
| **MoSCoW** | Must / Should / Could / Won't |
| **RAST** | Risk-control: Requirements, Architecture risk, Security/threat modelling, Task decomposition |
| **P-C-S-A-M** | ADR benefits: Preserve, Communicate, Support, Avoid, Manage |
| **K-D-C-E-F** | Five patterns: Knowledge priming, Design-first, Context anchoring, Encoding standards, Feedback flywheel |
| **A-Q-S-W** | Guardrail types: Action, Quality, Security, Workflow |
| **I-C-O** | Guardrail layers: Input (pre-LLM), Context (RAG), Output (post-LLM) |
| **Rules · Judge · Hooks · Human** | Guardrail implementation approaches |
| **P-S-A** | SEI roles: Providers formalize, Sustainers synthesize, App developers apply |
| **P-E-C-E → E-C-E** | Non-determinism sources; AI spans Emergent, Concurrent, Exceptional |
| **F-C-Q-D-P-C-A-I** | 8 review steps: Functional, Context, Quality, Dependencies, Pitfalls, Collaborative, Automate, Improve |
| **T-R-S-D-V-L** | AI testing uses |
| **T-G-M** | Agent testing: Tool-calling, Goal/plan, Memory drift |
| **C-T-E** | Agent test tiers: Component, Trajectory, End-to-end |
| **D-L-C-M** | DORA: Deployment frequency, Lead time, Change failure rate, MTTR |
| **S-D-C-I** | What DORA misses: Share, Durability, Complexity-adjusted, Innovation |
| **A-E-P-Q-B** | Hierarchy: Adoption, Engagement, Productivity, Quality, Business value |
| **A-C-V-Q-R** | Pillars: Adoption, Code share, Velocity (CAT), Quality, ROI |
| **L-V-P** | J-curve forces: Learning curve, Verification tax, Pipeline adaptation |
| **U-I-C** | DX AI measurement: Utilization, Impact, Cost |

## 3. Formula box

```
Absolute churn      = lines added + lines deleted + lines edited
                      (add 100, delete 30, edit 20 → 150 LOC)
Relative churn (%)  = churn / file-or-repo size × 100        [15–20% acceptable]

Cyclomatic (McCabe) = number of linearly independent paths through the CFG
Counting rule       ≈ decision points + 1
                      (if, else if, while, for, case, catch, &&, ||)
Risk bands          <10 low | 11–20 moderate | >20 high → refactor

ROI (%)             = (Value − Investment) / Investment
Value               = new revenues + cost savings
Investment          = direct hard costs (licenses, infra/compute, training)
                      + J-curve cost
J-curve cost        = staff size × fully loaded salary × productivity drop × dip duration

Net ROI multiplier  = (Productive Value of Time Saved − Rework Cost from Code Turnover)
                      ────────────────────────────────────────────────────────────
                                        Total AI Tool Cost
Time Saved Value    = hours saved/engineer/week × loaded engineering cost
Total AI Tool Cost  = seat licenses + token/usage + implementation overhead
                      ("not just the license fee — the actual spend")
```

## 4. Number anchors

| Number | Anchor |
|---|---|
| **15–20%** | Acceptable churn rate |
| **<10 / 11–20 / >20** | Cyclomatic complexity risk bands |
| **20–30%** | Selenium/Cypress suite failing per biweekly release |
| **2 engineer-days/sprint** | Locator repair burn before self-healing |
| **70–90%** | CI run-time reduction with AI-driven TIA |
| **95%+** | Semantic alignment threshold example for eval assertions |
| **2** | Max generation attempts in the Lab 2 harness |
| **0.1** | Temperature in the live harness call; 3 retries on HTTP 429/503 |
| **30 / 90 days** | Code-turnover measurement windows (AI vs human) |
| **~2× (roughly doubled)** | Code churn growth in the AI era |
| **20%** | CFR-rise example the DORA dashboard can't attribute |
| **5** | Metrics: DORA's 4 + rework rate (2025 DORA report) |
| **70% WAU / 10% share** | Shallow-adoption example (high usage, low AI code share) |
| **Dozens of versions** | How far behind OSS libraries sit (FY22 NDAA Section 835 study) |
| **6 / 4 / 3 / 2** | Playbook stages / guardrail types / guardrail layers / max repair attempts |
| **9** | Spec Kit commands |
| **5** | Hierarchy levels = pillars = AWS metric categories = DX Core 4 dimensions |
| **Wave 1 = no dependencies** | Kiro concurrency rule (waves sequential, tasks within wave concurrent) |

## 5. "See the question, write the skeleton" dispatch table

| If the question says… | Write first |
|---|---|
| "How does AI change the SDLC / velocity risks" | Bottleneck-shift flow (D1) → plan/review-test/deploy human speed → churn/debt/drift → spec + sensors + gates + DORA-plus metrics |
| "Requirements from unstructured input / elicitation" | Elicit → synthesize (INCOSE/EARS, stories+GWT) → manage (MoSCoW, traceability) → HITL sign-off → must-not-do guardrails |
| "Validate an AI-generated architecture" | RAST activities → ADRs → authored constraints → plan PR gate with senior review → conformance layer |
| "Compare two AI workflows/tools" | 2-column table → control-flow ownership → guiding rule (oversight ↓ ⇒ deterministic code ↑) → recommendation |
| "Spec vs code authority" | Spectrum First/Anchored/Source → who edits what → drift risk + CI check |
| "Agent generates code AND tests" | Circular validation: same-assumption loop → spec-oracle, TDD, gates+mutation → independent integration tests + human checkpoint |
| "Harness / feedforward / sensors" | 4 feedforward files → static → dynamic → findings-only repair ≤2 → audit → human review |
| "Test maintenance pain / flaky suites" | Self-healing (DOM, fallback locators, background) + TIA (70–90%) + QA-role shift |
| "Defect prediction" | Inputs churn+complexity+history → RF/XGBoost/GNN per PR → actions → humans still merge |
| "DORA under AI / 40% faster but defects up" | Producing≠verifying → MTTR holds/CFR partial/DF+LT mislead → add S-D-C-I + rework rate → durability evidence |
| "ROI / economics of AI" | Cost (licenses+overhead) + J-curve cost formula vs time-saved value + revenue → Net ROI subtracting turnover rework |
| "Governance / who is liable" | Humans own outcomes → guardrail inventory (A-Q-S-W) → HITL pause points → shift-left conformance → audit trail |
| "Legacy migration / library upgrade" | Decompose → RevAPI facts → LLM recipes → engine applies → build/test loop → write once roll out everywhere → approvals |
| "Non-deterministic outputs" | Apparent non-determinism → E-C-E mapping → deterministic pipeline: constrain, schema-validate, spec-grounded tests, static+dynamic, human, gates |
| "Anything from the labs" | Artifact chain + exact command/file names + verification step + audit evidence |

## 6. Lab evidence checklist (if asked to "design/justify a harness")

- [ ] Contract first: `openapi.yaml` as single source of truth
- [ ] Feedforward: style-guide.md + harness-rules.md + generation-task.md
- [ ] Exact assertions required; broad (`>= 400`) forbidden
- [ ] Output schema validated before files written (exactly `appJs`/`testJs`, no fences)
- [ ] Static sensor → dynamic sensor (dynamic skipped until static clean)
- [ ] Findings-only, coded, machine-readable repair input
- [ ] Max 2 attempts; escalate to human
- [ ] Audit: sensor-report.json + attempt-log.json (+ manual-attempt-log.json in Part A)
- [ ] Untrusted until both sensors pass; human review still recorded
- [ ] Extend = contract → rules → sensor patterns change together
