# Software Product Management — Summary (CS01–CS07)
---

# PART 1 — Foundations

## 1. What is Product Management?

**Definition:** Product Management is the strategic process of identifying customer problems, deciding what product or features to build, guiding development, and continuously improving value to achieve business outcomes and customer satisfaction.

**The Four Foundational Questions (memorize verbatim):**
1. **Who** are we building for?
2. **What** should we build?
3. **Why** should we build it?
4. **How** do we know it is successful? (How do we *measure* success?)

⚠️ **Common trap:** Substituting "How do we build it?" for the third question. The correct third pillar is **Why** — a business-rationale question, not an execution question.

**Worked example (Uber):**
- WHO: People who can't reliably hail a cab
- WHAT: Tap to ride, price agreed up front, driver on a map
- WHY: Removes haggling, waiting, cash
- HOW MEASURED: Completed trips, ETA accuracy, cancellation rate

## 2. PM vs Project Manager vs Product Owner

| Dimension | Product Manager | Project Manager | Product Owner |
|---|---|---|---|
| Primary focus | **Why & What** — value, strategy, customer pain, PMF | **When & How** — execution, timeline, budget, risk | **Backlog detail** — tactical Scrum execution, story grooming |
| Key metrics | Adoption, retention, NPS, CSAT, revenue | On-time delivery, budget adherence | Sprint velocity, story-point throughput |
| Horizon | Long-term (months–years) | Medium-term (project lifecycle) | Short-term (sprint to sprint) |

**Exam insight:** Shipping a project on time does **not** guarantee product success. Amazon Fire Phone shipped on time with impressive engineering and still failed — because nobody validated customer demand. On-time delivery = an **output**; product success = an **outcome**.

## 3. The Product Development Lifecycle (PDLC)

**Definition:** The end-to-end process of transforming a customer problem or market opportunity into a successful product, and continuously improving it based on customer feedback and business outcomes.

**The 10 steps (in a loop — not a one-time line):**

| # | Step | Key question |
|---|---|---|
| 1 | Identify the Customer Problem | What problem, how often, how severe, is it worth solving? **Starts with a problem, not a feature.** |
| 2 | Market Research | Interviews, surveys, competitor analysis, usage analytics, support tickets — validate the problem is real |
| 3 | Generate Ideas | Brainstorm **multiple** solutions to the same problem before picking one |
| 4 | Prioritize Features | Customer value, business impact, technical effort, cost, timing |
| 5 | Design Solution | Wireframes, mockups, user flows, prototypes |
| 6 | Development | Engineering builds it; PM ensures the team understands **why** it matters |
| 7 | Testing | Functional, integration, performance, security, UAT |
| 8 | Release | Deployment, release notes, documentation, customer communication |
| 9 | Collect Customer Feedback | Surveys, tickets, reviews, sales input, analytics |
| 10 | Measure Success | CSAT, NPS, feature adoption, revenue, churn, time saved → loop back to Step 1 |

**Key takeaway:** Everything starts with a problem — not a feature, not a solution, not a JIRA ticket.

## 4. Project Business vs Product Business

| Dimension | Project Business | Product Business |
|---|---|---|
| Revenue & Ownership | Milestone billing, time-and-material contracts; **client owns the IP** | Recurring subscription revenue (SaaS ARR); **vendor owns the IP** |
| Scope & Customization | Bespoke, per-client customization | Standardized, configurable core; **no custom code forks** |
| Risk Profile | Delivery / resource-allocation risk (will we hit the milestone?) | Discovery / market risk (will anyone actually buy it?) |
| Duration | Temporary — defined start and end | Ongoing — evolves continuously |
| Success metric | Delivered on time, on budget, to spec | Adoption, retention, revenue growth, CSAT |

⚠️ **Exam trap:** When asked for "differences," always cover **all three** dimensions (Revenue/Ownership + Scope/Customization + Risk Profile). Naming only one costs roughly two-thirds of the marks. Then add a **recommendation justified with case facts** (e.g., for HelpMeLearn: escape linear scaling, mitigate margin pressure, monetize an internal R&D asset).

**Productizing an internal tool** (e.g., HelpMeLearn, Zerodha's Kite, Freshworks' helpdesk origin) means: generalizing beyond the original one-off need, long-term support, security/compliance hardening, configurability for multiple customers, and a dedicated roadmap.

---

# PART 2 — : Problem Space vs Solution Space

**This is one of the most-tested distinctions in the course. Know it cold.**

**Problem Space** = the customer's underlying pain, workflow friction, or goal.
**Solution Space** = the specific implementation built to address it.

**Core insight:** Customers report **solutions**, not problems — because solutions are easier to describe than pain.

- Customer says: *"Give us a bulk CSV upload button."* ← **Solution Space** (a solution they invented)
- Real problem: *"It takes my team 4 hours every Monday to manually enter 300 rows."* ← **Problem Space**
- Better solution the customer never imagined: automated ERP/HRMS API sync that eliminates manual entry entirely.

**Why explore the Problem Space first?** It opens up multiple possible solutions (CSV upload vs API sync vs self-service invite links) so the team can pick the most cost-effective and scalable one — instead of being locked into whatever the customer happened to ask for. Jumping straight to the requested solution is called the **"solution trap."**

**The diagnostic move (exam-critical):** When a scenario gives you a customer's stated solution, you must write the **actual root-cause question you would ask**, not merely say "I would investigate further."

✅ Good answer: *"Why do you need this uploaded daily — is it for shift-change handoff, compliance record-keeping, or reference?"*
❌ Weak answer: *"I would dig deeper into their needs."* (partial credit at best)

**Same idea appears across the course:** CS01 (PDLC Step 1), CS03 (formal vocabulary), and CS02's "What Best Product Teams Do" (the bulk-upload example). The elite team asks **why**, prototypes lightweight options, and measures the *outcome* (e.g., % reduction in onboarding time) — not whether the button shipped.

---

# PART 3 — : Product Discovery vs Product Delivery

**Course catchphrase (memorize):**

> **Discovery = Build the RIGHT product.**
> **Delivery = Build the product RIGHT.**

| Aspect | Discovery | Delivery |
|---|---|---|
| Goal | Decide **what** to build | Build it **correctly** |
| Primary question | Are we solving the **right problem**? | Are we **delivering** the solution well? |
| Focus | Customer needs, market demand, business viability | Engineering execution |
| Activities | Interviews, surveys, prototypes, MVP testing, prioritization | Sprint planning, coding, testing, deployment |
| Success measure | Customers **want** the solution | Customers **receive a reliable** solution |
| Risk mitigated | Market / demand risk | Execution / build risk |

**Why the separation matters — the "build trap":** Skip discovery and jump straight to delivery and you spend months of engineering effort on features validated by nobody. Real-world delivery-perfect / discovery-failed cases: **Amazon Fire Phone** ($170M write-down), **Quibi** ($1.75B, shut down in ~6 months), **Google Glass**, **Netflix Qwikster** (lost ~800,000 subscribers).

**Best practice (CS03 refinement):** Run discovery and delivery **concurrently and continuously** — weekly customer touchpoints *while* shipping sprint increments. Never run discovery only once at kickoff, then go heads-down for months ("big-bang, stale discovery").

---

# PART 4 — User Stories, INVEST & Acceptance Criteria

## 4.1 User Story Format

> **As a <type of user>, I want <goal>, so that <benefit>.**

Example: *As a Front Desk Receptionist, I want to search guests by phone number, so that I can check them in quickly.*

The **"so that"** clause is the most-skipped and most-important part — it tells the team *why*, enabling good judgment on edge cases without constant PM escalation.

## 4.2 The INVEST Checklist (verbatim)

| Letter | Criterion | Meaning |
|---|---|---|
| I | **Independent** | Can be built without waiting on other stories |
| N | **Negotiable** | Describes the need, not the exact implementation |
| V | **Valuable** | Someone (user or business) is measurably better off |
| E | **Estimable** | Team can size it; if not, it needs clarification first |
| S | **Small** | Fits inside one sprint |
| T | **Testable** | You can say in advance how you'd verify it works |

**Spotting broken stories (exam pattern):**
- *"As a user, I want a better dashboard, so that I can see my data"* → vague, untestable, no real persona
- *"As a PM, I want the team to migrate to PostgreSQL"* → technical task masquerading as a user story
- *"Check in guest + take payment + print key card + email folio"* → violates **Small**; that's an epic

## 4.3 Acceptance Criteria — Given / When / Then

A user story says **what** the user wants; acceptance criteria say **how you know it's done**.

> GIVEN I am a support agent with timezone set to IST
> WHEN I open a ticket created at 09:00 UTC
> THEN the timestamp reads 14:30 IST, with the timezone label visible

**Rules for testable criteria:**
- Must be a concrete, checkable condition — never "the feature should be user-friendly" or a restatement of the goal ("usability improves" ❌).
- Include **unhappy paths**: unregistered email, expired links, empty states, timeouts, offline behavior.

⚠️ **Confusion alert:** INVEST's **Estimable** = can the team *size* the story? **Testable** = can you verify it works? Different letters, different checks.

---

# PART 5 — : Agile Manifesto, Sprint, Sprint Goal, Sprint Planning & Velocity

## 5.1 The Agile Manifesto — 4 Values (memorize verbatim, including "over")

1. **Individuals and interactions** over processes and tools
2. **Working software** over comprehensive documentation
3. **Customer collaboration** over contract negotiation
4. **Responding to change** over following a plan

**The word "over" is the operative concept:** the right-side items still have value — they are simply valued *less*. Documentation is not forbidden; working increments are just the primary measure of progress.

**Memory device — the 4P's:**
- Left side (the human/adaptive side): **People, Product, Partnership, Pivot**
- Right side (the 4P's of rigidity): **Processes, Paperwork, Paper-contracts, Plans**
- Say it: *"We value People over Process, Product over Paperwork, Partnership over Paper, and Pivoting over Plans."*

## 5.2 Sprint

**Definition:** A **fixed-length timebox (1–4 weeks)** producing a **usable, shippable product increment**, driven by a clear **Sprint Goal**.

- ⚠️ **Sprint ≠ Release.** A Release is what ships to customers; a Sprint is the internal work cycle that produces it.

## 5.3 Sprint Goal & Sprint Planning

**Sprint Goal:** the single, clear objective that gives the sprint its purpose and coherence — the "why" behind the selected work. It is agreed by the whole Scrum Team (PO + developers + Scrum Master) and guides trade-off decisions mid-sprint ("does this work serve the goal?").

**Sprint Planning:** the ceremony where the team decides **what** will be delivered in the upcoming sprint and **how** it will be done:
1. **Why / What:** select backlog items that serve the Sprint Goal — the highest-priority, highest-value work the team believes it can complete
2. **How:** break items into tasks; the developers own the "how," estimating effort (story points)

**Key roles:** PO presents priorities and clarifies value; **the team (developers) decides how much work it can commit to** — the sprint backlog is a team commitment, not a management target.

## 5.4 Velocity

**Definition:** Story points completed per sprint, calculated as a **rolling average across 3–5 sprints**.

- **Used strictly as a capacity-forecasting tool** — e.g., "our velocity is ~38 points, so we can likely commit ~38 points next sprint."
- **Never** an individual productivity metric and **never** a mandatory target.

**Goodhart's Law (exam-favorite scenario):** *"When a measure becomes a target, it ceases to be a good measure."*

> Scenario pattern: a manager sets a mandatory 50-point velocity target for Sprint 4 "to increase productivity." The team simply **inflates story-point estimates** to hit the number — estimate inflation without any increase in real delivered value. That is a Goodhart's Law violation.

**Example calc:** sprints of 34, 40, 38 points → rolling average velocity = 112/3 ≈ **37 points**.

⚠️ **Confusion alert:** Velocity must be a *rolling average (3–5 sprints)*, not a single-sprint number.

---

# PART 6 — : MVP and Pilot Launch

## 6.1 MVP Definition (memorize verbatim)

> **MVP = the smallest possible increment of a product that allows a team to collect the maximum amount of validated learning about customers with the least effort.**

**MVP vs "half-baked product":**
- MVP = deliberately minimal but **fully functional for its core purpose** — a complete, usable learning loop.
- Half-baked = incomplete/broken, with no intentional learning design.
- The difference is **intentionality**: MVP = "smallest complete loop"; half-baked = "unfinished."

**Build Phase Principle (Lean Startup):** The goal of the Build phase is **not** to build the final product — it is to build the minimum artifact required to start the learning feedback loop. Six months building a full AI recommendation engine before showing anyone = violation. Use **low-code mockups** or a **manual "concierge" service** instead to test demand in days.

## 6.2 Pilot Launch

A pilot = a controlled, limited rollout to a small set of friendly customers (e.g., HelpMeLearn's **2-month free pilot with 3–5 friendly clients**) used to gather **qualitative** retention and engagement feedback before scaling. Pilot-stage success is judged **qualitatively** — not by ARR/MRR (those are Growth-stage metrics) and not by NPS/LTV (Maturity-stage metrics).

## 6.3 Three MVP-Testing Techniques (know what each measures)

| Technique | What it measures | Example |
|---|---|---|
| **User-tests** (clickable wireframes/prototypes) | **Usability** — can people navigate the flow? (NOT feasibility!) | Figma prototype walkthrough with L&D managers |
| **Live-data / Fake-door** | **Value/interest** — real clicks on a non-existent feature | "Enroll now" button that only logs clicks |
| **Landing-page MVP** | **Inbound interest** — sign-ups/CTR on a product that doesn't exist yet | Mock pricing page measuring click-through |

**Why it matters:** testing prototypes iteratively uncovers design flaws in **days** instead of discovering failure post-launch after months of engineering.

⚠️ **Critical confusion:** User-tests measure **Usability**, not **Feasibility**. Feasibility is assessed separately via **engineering spikes** (timeboxed technical investigations) — user-tests are not one of the feasibility-checking mechanisms.

## 6.4 Low-Cost Validation Techniques (for "suggest a 2–3 week action" questions)

1. **Fake-door / smoke test** — landing page or clickable prototype with a "Sign up / Request access" button; measures real interest without building anything.
2. **Concierge MVP** — manually deliver the outcome to a few customers (Dunzo's WhatsApp group origin; Zappos buying shoes at retail; Airbnb founders photographing listings themselves). Zero engineering cost, real demand signal.
3. **Letter of Intent (LOI) / soft commitment** — ask target customers to sign a non-binding pilot agreement. A signed LOI is a far stronger signal than a polite verbal "yes."
4. **Clickable prototype demo (Figma)** — walk customers through a mockup, observe reactions, ask for a concrete next step.

**Why action beats opinion:** asking "would you use this?" gets weak, biased answers. Asking for an **action** — a sign-up, a signed LOI, a calendar commitment for onboarding — reveals real demand and directly reduces **Value/Demand Risk** before expensive engineering is committed.

## 6.5 Kano Model (classifies what belongs in the MVP)

| Category | Meaning | Customer reaction | HelpMeLearn example |
|---|---|---|---|
| **Basic (Threshold)** | Absence causes immediate rejection; presence is taken for granted, never praised | Neutral if present; extreme dissatisfaction if absent | Secure login, multi-tenant data isolation, core course tracking |
| **Performance** | "More is better" — linear satisfaction | Proportional satisfaction/dissatisfaction | Bulk user uploads, dashboard report speed |
| **Delighter (Exciter)** | Unexpected; creates excitement; nobody demanded it | Delight if present; **no** dissatisfaction if absent | AI-assisted skill recommendations |

**MVP inclusion rule:** Basics **must** be in the MVP (non-negotiable entry gate). Performance features are included as resources allow (more = better). Delighters are optional — deferred to later releases.

⚠️ The middle category's exact name is **Performance** — not "functional."

**Extra insight:** Today's delighter is next year's basic (live order tracking was magic in 2015; its absence is now a support ticket).

---

# PART 7 — : The Four Big Risks (Cagan's Framework)

> **Memory hook: Value = they WANT it → Usability = they CAN use it → Feasibility = WE can build it → Viability = it WORKS for our business.**
> (Customer-facing risks first, then internal capability/business-model risks.)

| Risk | One-line definition | Mitigation | HelpMeLearn example |
|---|---|---|---|
| **Value Risk** | Will customers buy/use it — especially vs. free alternatives? (e.g., Moodle) | Discovery interviews, pre-sales pilots, fake-door tests, LOIs | Enterprises may prefer free tools → mitigate via pre-sales pilots |
| **Usability Risk** | Can users figure out how to use it? | Usability testing on low-fi prototypes, user-tests | L&D managers may find admin dashboards too complex → usability testing |
| **Feasibility Risk** | Can we build/scale it with our tech, skills, time? | Timeboxed **engineering spikes** | Architecture may not handle secure multi-tenant BFSI deployment |
| **Business Viability Risk** | Does it work for our business model — legal, financial, sales? | LTV vs. CAC modeling, legal review, sales-cycle analysis | Subscription margins may not cover multi-region compliance costs |

**Why Value Risk is the MOST dangerous:** you can build a flawless, secure, beautiful product and still fail completely if nobody wants it. The other three risks ask "can we build/sustain it?"; Value Risk asks "**should it exist at all?**"

**Business Viability — three sub-checks:**
1. **Unit economics** — LTV must significantly exceed CAC
2. **Legal & compliance** — data residency, security regulations, without prohibitive liability
3. **Sales cycles** — 6–12 month enterprise procurement cycles factored into cash-flow forecasts

> 🚨 **THE single most-repeated error in this course:** when a question asks for the "risk" or "concern," answer with **what could go wrong** — never a justification for why the idea will succeed.
> ✅ *"The Feasibility Risk is that building HIPAA-compliant, low-latency video infrastructure is genuinely hard for our team."*
> ❌ *"Yes, it will work because it's basically a call-scheduling feature."* ← reassurance, not a risk. **Practice the mental flip: risk is a pessimist's question.**

Also note the naming bridge: **Demand Risk = Value Risk**; **Design Risk = Usability Risk** — the exam may use either vocabulary.

---

# PART 8 — : RICE Scoring

## 8.1 The Formula (memorize verbatim)

> **Score = (Reach × Impact × Confidence) ÷ Effort**

| Factor | Meaning | Scale |
|---|---|---|
| **Reach** | Users affected per quarter — *count them, don't feel them* | Raw number (e.g., 40,000) |
| **Impact** | How much it helps each user | 3 = massive, 2 = high, 1 = medium, 0.5 = low, 0.25 = minimal |
| **Confidence** | How sure you are | 100% (hard data), 80% (some evidence), 50% (a hunch) |
| **Effort** | Person-weeks from the people doing the work | Number of weeks |

**Higher score = higher priority.** Effort is a **denominator** — high effort tanks a score even when reach/impact look strong (RICE punishes both uncertainty and cost).

## 8.2 Worked Example (memorize this pattern)

| Feature | Reach | Impact | Confidence | Effort | Score = (R×I×C)÷E | Rank |
|---|---|---|---|---|---|---|
| Checkout crash fix | 40,000 | 3 | 100% | 1 wk | (40,000×3×1.0)÷1 = **120,000** | 1st |
| Search speed <1s | 90,000 | 1 | 80% | 2 wks | (90,000×1×0.8)÷2 = **36,000** | 2nd |
| AI chat assistant | 20,000 | 2 | 50% | 5 wks | (20,000×2×0.5)÷5 = **4,000** | 3rd |

**Insight:** the AI assistant has decent reach and impact, but low confidence + high effort crush its score.

## 8.3 Twist Recalculation (exam favorite — the PCI example)

| Feature | Reach | Impact | Confidence | Effort | Score |
|---|---|---|---|---|---|
| A: Guest checkout | 120,000 | 2 | 80% | 4 wks | (120,000×2×0.8)÷4 = **48,000** |
| B: Saved payment cards | 60,000 | 3 | 100% | 3 wks | (60,000×3×1.0)÷3 = **60,000** |
| C: One-tap reorder | 25,000 | 3 | 80% | 2 wks | (25,000×3×0.8)÷2 = **30,000** |

**Original ranking:** B (60,000) → A (48,000) → C (30,000).
**Twist:** saved cards needs a PCI review adding 3 weeks → new Effort = 6 → (60,000×3×1.0)÷6 = **30,000**.
**New ranking:** A (48,000) → B and C tied (30,000). **The twist flipped the priority** — recalculate before committing.

**Interpretation line for exams:** *"RICE does not make the decision; it forces you to state your Reach assumption out loud in a room where someone can disagree with it — which is where the real argument was hiding."*

---

# PART 9 — : MoSCoW and the 60% Rule

## 9.1 The Four Categories

| Category | Meaning | Diagnostic question |
|---|---|---|
| **Must have** | Without it, no point releasing at all (or illegal/unsafe) | "If we ship without this, do we cancel the release?" — if no, it's **not** a Must |
| **Should have** | Hurts to leave out, but we could cope (with a workaround) | "Could we live without this for one release?" — if yes, it's a Should |
| **Could have** | Nice; do it if there's room — the safety margin | "Would anyone complain if this slipped?" — if no, it's a Could |
| **Won't have** | Explicitly agreed as out of scope for this release | "Have we told the requester, and written it down?" |

**Party analogy (intuition builder):** Musts = music, ice & drinks, seating · Shoulds = main course, dietary options · Coulds = themed decorations, party favors · Won'ts = a rigid 3-course sit-down dinner.

## 9.2 The 60% Rule (a precisely-graded exam fact — the numbers matter)

> **Must-Haves must not exceed 60% of total release capacity, preserving the remaining 40% (Should/Could/Won't) as a safety buffer against unexpected delays.**

**The arithmetic, understood once:** if Musts are packed to 60% and run 25% over budget, you absorb the overrun by cutting Coulds and still deliver on time (60 × 1.25 = 75, comfortably under 100). If Musts are packed to 80%+ with no buffer, any overrun breaks the deadline entirely — there is nothing left to cut (80 × 1.25 = 100, zero margin).

**Exam signal:** if more than 60% of a release ends up tagged "Must," that itself means **nobody has actually made a prioritization decision** — everything cannot be equally essential.

⚠️ **Trap:** never re-derive a MoSCoW priority that **contradicts a fact explicitly stated in the scenario** (e.g., calling something the case calls "a Must for pilot clients" a "Should").

---

# PART 10 — Other Prioritization & Sequencing Frameworks

## 10.1 Opportunity Scoring

Survey real users two questions per problem: (1) How **important** is this to you? (1–10) (2) How **happy** are you with it today? (1–10)

> **Score = Important + (Important − Happy)** — if Happy > Important, the bracket floors at 0 (never go negative).

Reading the score (1–20 scale): **>15 = big opportunity** · 10–15 = worth a look · **<10 = leave it alone.**

**Worked examples:** Important 9, Happy 2 → 9 + 7 = **16** → build it. Important 9, Happy 8 → 9 + 1 = **10** → already fine, low priority. Important 8, Happy 3 → 8 + 5 = **13** → investigate, not an automatic top priority.

## 10.2 Cost of Delay & CD3 (WSJF)

**Cost of Delay** = what you lose (in value) **for every week** a feature is not built.

**Core insight — order matters, not just size:** doing the "biggest loss" job first is not always optimal, because it blocks smaller jobs behind it in the queue.

> **CD3 Score = (money lost per week) ÷ (weeks of work required)** — do the **highest score first.**

**Supermarket-queue analogy:** let the person with 2 items go ahead of the full trolley — everyone's total wait drops, even though the full-trolley shopper individually "loses more."

**Worked example:**

| Job | Loses/week | Takes | CD3 Score | Order |
|---|---|---|---|---|
| A | ₹10 | 2 wks | 10÷2 = **5** | **1st** |
| B | ₹12 | 4 wks | 12÷4 = **3** | 2nd |
| C | ₹8 | 4 wks | 8÷4 = **2** | 3rd |

⚠️ **Trap:** B loses the MOST money per week (₹12) but goes **2nd**, not 1st — the score (loss ÷ duration), not the raw loss, determines order. Comparing X (₹15/wk, 3 wks → 5) vs Y (₹8/wk, 1 wk → 8): **Y goes first.**

**WSJF (Weighted Shortest Job First):** the same logic when you can't get real money figures — Score = (Value + Urgency + Risk Reduction) ÷ Job Size, using relative points (1, 2, 3, 5, 8, 13).

## 10.3 Which Framework When (great exam table)

| Framework | Best used when |
|---|---|
| **RICE** | You have (or can estimate) reach, impact, confidence, effort numbers |
| **MoSCoW + 60%** | Fixed release date; need bounded scope |
| **Opportunity Scoring** | You can survey real users on importance/satisfaction |
| **CD3 / WSJF** | Sequencing multiple jobs to minimize total value lost while waiting |
| **Value vs Effort 2×2** | Quick triage (do now / plan / fill gaps / say no) |

## 10.4 Stakeholder Prioritization Games (CS07)

- **Product Tree:** trunk = existing core; branches = last release's features; leaves = new ideas customers pin on; roots = infrastructure. Reveals **where customers want the product to grow** and which old branch nobody defends (a candidate to cut) — but gives **no cost/effort** data.
- **Buy a Feature:** price features by build cost; give each stakeholder limited fake money (total ≈ ⅓ of total price) so nobody can afford their favorite alone and people must negotiate. Reveals **real willingness to trade off** — what people will *pay* for, not what they'd *say* in a survey (where everything "matters").

---

# PART 11 — : Product Roadmap

## 11.1 Definition

> A Product Roadmap is a **high-level plan** that communicates the product's **vision, direction, priorities, and major initiatives over time.**

It answers: Where is the product going? What major features will be delivered? Why are they important? When (roughly) are they expected? It aligns management, engineering, sales, marketing, customer success, and customers.

**Strategic hierarchy:** Vision (long-term aspiration) → Strategy (pillars to achieve it) → **Roadmap** (themes over time) → Release (feature bundle delivered) → Sprint (1–4 week execution cycle).

**Dream-home analogy:** Vision = "a modern family home" · Roadmap = Foundation → Structure → Interiors → Landscaping · Project plan = daily construction schedule · Release = the day the family moves in.

## 11.2 The Now / Next / Later Structure

| Horizon | Timeframe | Certainty | Focus |
|---|---|---|---|
| **NOW** | This quarter (0–3 months) | High — committed | Specific features, exact problem resolution, bug fixes |
| **NEXT** | Next quarter (3–6 months) | Medium — validated direction | Strategic initiatives under discovery/design |
| **LATER** | 6–12+ months | Low — exploratory vision | Long-term possibilities, future tech adoption |

**The crucial nuance (exam-critical):** a roadmap should communicate **problems to solve, organized as Now/Next/Later — NOT features pinned to fixed dates.** Dates attached to unvalidated features become **broken promises** — a roadmap with a date against every item is "a project plan wearing a costume," and it is the reason nobody trusts roadmaps.

## 11.3 Roadmap vs Release vs Sprint

| | Roadmap | Release | Sprint |
|---|---|---|---|
| Scope | Strategic direction | A product version delivered | Short dev cycle |
| Duration | Months–years | Weeks–months | 1–4 weeks |
| Focus | Outcomes | Deliverables | Implementation |
| Owner | Product Manager | Shared (Product, Eng, QA, DevOps) | Scrum Team |

**Release types:** Major (large features) · Minor (smaller enhancements) · Patch (bug fixes/security) · Hotfix (urgent production fix — cannot wait for a feature flag).

**Safe-release mechanics (cite when a launch goes wrong):** feature flags (ship code dark, switch on separately) · canary / 1% rollout before full release · backwards-compatible migrations · dashboards & alerts before the flag flips · a **written rollback step someone has actually run**.

**Failure case studies:** Flipkart Big Billion Day 2014 (traffic crash within hours; rebuilt around load testing and app-first) · Apple Maps 2012 (shipped as iOS 6 default replacing a working product; Tim Cook's public apology) · Snapchat 2018 (see Part 13).

## 11.4 Backlog vs Roadmap

- **Backlog = a queue, not a pile.** One ordered list where **position = priority, no ties**; refined continuously (~10% of team time); lower items described more coarsely than top items.
- **Roadmap = problems over time** (Now/Next/Later).
- ⚠️ Common mistake: building a roadmap around a fixed feature list with committed dates before those features are validated.

---

# PART 12 — North Star Metric, Working Backwards, Opportunity Solution Tree

## 12.1 North Star Framework / North Star Metric

**Core idea:** the framework centers on a **single, most important metric — the North Star Metric** — that best captures the value your product delivers to customers. It is **complemented by a number of key inputs** that drive it. The team works to impact these key inputs with everything it does every day.

**How it works:** the North Star sits at the top; beneath it, a small tree of **key input metrics** that the team can directly influence; each input can decompose further into operational metrics.

**Illustrative structure:**
- **North Star Metric: Monthly Revenue**
  - Revenue per User ← Purchases, AOV (average order value)
  - Active Users ← Signups, Sessions per User

**Why it matters for exams:** it gives the whole org **one shared measure of success** and forces every initiative to justify itself against that metric and its inputs — connecting daily work to business outcomes.

**Contrast with guardrail thinking:** the North Star tells you where to steer; **guardrail metrics** (see Part 13) tell you what you're breaking while steering.

## 12.2 Working Backwards (Amazon's method)

**Core idea:** this framework **literally reverses the product development process**, starting with what usually comes *later*. The team begins by **imagining that the product has already been built** to identify whether the customer's problem has actually been solved.

**The process (in order):**
1. **Start with the customer — draft the press release.** Write an internal press release (and FAQ) for the finished product as if it were launching today: who is the customer, what problem does it solve, why is it compelling. If the press release isn't compelling, **stop** — don't build it.
2. **Evaluate the opportunity** — is it genuinely compelling for the customer?
3. **Find solutions** — get approvals; work backwards from the press release to the solution.
4. **Build the roadmap** — themes, etc., derived from the validated narrative.
5. **Create the backlog** — stories and tasks that deliver the roadmap.

**What the press-release artifact contains (worth citing):** the customer problem, the existing solution's shortcomings, the new solution's benefits, a customer quote, and the **FAQ** (which exposes hard questions early — pricing, adoption, operations).

**Why it works:** it forces clarity of customer value *before* any resource commitment, kills weak ideas cheaply at the narrative stage, and keeps the customer's voice as the anchor throughout delivery — the mirror image of building first and hoping marketing can explain it later.

## 12.3 Opportunity Solution Tree (Teresa Torres)

**Core idea:** a **visual representation of how you aim to attain your desired outcome.** The diagrammatic tool was designed to **make the discovery process simpler** and to enable the **proper prioritization of opportunities**. It helps product teams **know and focus on "the big picture"** in their work.

**The structure (top to bottom):**

```
                 OUTCOME  (the measurable result you want, e.g., "increase pilot-to-paid conversion")
                    │
        ┌───────────┼───────────┐
   OPPORTUNITY   OPPORTUNITY   OPPORTUNITY   ← customer needs/pains discovered through continuous discovery
        │             │             │
   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
 SOLUTION  SOLUTION SOLUTION SOLUTION SOLUTION SOLUTION   ← ways to address each opportunity
   │
 ASSUMPTION TESTS  (experiments: interviews, fake-door tests, prototypes) ← cheap evidence
```

**Key mechanics to state in an exam:**
1. Start with a **single, clear desired outcome** at the top — not a feature, not a solution.
2. **Discover opportunities** (unmet needs, pains) through continuous customer discovery — continuous interviewing keeps the tree fed with real opportunities rather than internal opinions.
3. **Generate multiple solutions per opportunity** — this directly combats the solution trap: teams habitually fall in love with one idea; the tree forces breadth.
4. **Test assumptions with cheap experiments** before committing engineering.
5. **Prioritize properly:** the tree makes it visible that you should first pick the most promising *opportunity*, then the best *solution* under it — rather than jumping from outcome straight to a pet solution.

**Why it matters:** it connects everything in this course — outcome thinking (roadmaps), problem space vs solution space (opportunities are problem space; solutions are solution space), continuous discovery, MVP testing, and prioritization — into one visual map.

---

# PART 13 — Measuring Product Success: Outputs, Outcomes & Metrics

## 13.1 Outputs vs Outcomes (memorize the table)

| Output (what we did) | Outcome (what changed) |
|---|---|
| Feature released | Customers actively use it |
| Mobile app launched | More bookings through mobile |
| Dashboard created | Managers make faster decisions |
| AI chatbot deployed | Support tickets reduced by 40% |

> **"We shipped it" is not success. "It created value" is success.**

## 13.2 Key Metrics (name the precise terms)

- **CSAT** (Customer Satisfaction): 1–5 star rating
- **NPS** (Net Promoter Score): "How likely are you to recommend us?" 0–10 scale → **Promoters (9–10), Passives (7–8), Detractors (0–6)**; **NPS = %Promoters − %Detractors** (range −100 to +100)
  - Worked example: 500 users → 250 promoters (50%), 150 passives (30%, ignored), 100 detractors (20%) → **NPS = 50 − 20 = +30** (positive, healthy advocacy; >0 good, >50 excellent)
- **DAU/MAU**: engagement and stickiness (DAU/MAU ratio)
- **Feature adoption, retention, churn rate, revenue growth, time saved, support ticket volume, performance metrics** (e.g., API response < 2 sec)

## 13.3 Vanity vs Guardrail Metrics

- **Vanity metrics** feel good but prove nothing: total signups, initial logins, downloads.
- **Actionable metrics** reflect true engagement: **cohort retention rate**, DAU/MAU, task success, return behavior.
- **Guardrail rule:** pair every success metric with a guardrail metric — adoption with uninstalls, engagement with complaints, speed with error rate. *"A success metric with no guardrail beside it is a metric you can win while losing."*

**Snapchat 2018 case (the canonical guardrail failure):** the redesign shipped on time and its target metric (Discover engagement, over-35 engagement) genuinely **improved** — but no guardrail watched existing power users. Result: a **1.2M-signature petition**, and Snapchat's first-ever quarterly DAU decline (~3M users lost). Delivery was excellent; the outcome was a failure because one guardrail metric was missing.

## 13.4 Funnel Analysis (worked pattern)

100,000 carts → 82,000 address → 74,000 payment method → 71,000 payment attempted → 44,000 orders.
- Biggest drop: **payment attempted → placed** (61.97% conversion; 27,000 failures).
- Next step: **segment by payment method and error code** (don't just "redesign the cart page").
- Metric to report upward: **checkout completion = 44%**.

## 13.5 Critical Success Factors (CSFs) — four categories, specific numbers

| Category | Named KPIs |
|---|---|
| Adoption & Engagement | MAU, course completion >75%, CSAT >4.5 |
| Financial Viability | ARR growth, churn <5%, favorable **LTV:CAC** |
| Operational & Quality | Low ticket volume, API response <2s, zero critical vulnerabilities, successful security audits |

A product can look great on one axis (high MAU) while quietly failing on another (poor LTV:CAC) — CSFs force a check on all fronts before a board decision.

---

# PART 14 — User vs Buyer Personas (confirmed 10-mark question pattern)

**The distinction:** the person who *uses* the product daily is often a different person from the one who *authorizes paying* for it.

| | User Persona | Buyer Persona |
|---|---|---|
| Who | Uses the product day-to-day | Decides to purchase / holds the budget |
| Cares about | Ease of use, features, speed, engagement | ROI, cost, security, compliance, integration risk |
| Won over with | Great UX, convenience, time saved | Business case, compliance proof, pricing, references |

**HelpMeLearn personas (know these):**
- **Users:** Learners (bite-sized mobile videos, auto-reminders) · L&D Managers/Admins (bulk enrollments, tracking, dashboards)
- **Buyers:** CTOs (HRMS integration, security compliance) · Procurement (low total cost of ownership)

**The imbalance-consequence table (memorize precisely — this exact framing is tested):**

| If you prioritize... | Result |
|---|---|
| **Buyer only** | Secure, feature-heavy software that employees hate using → **low adoption, high churn at contract renewal** |
| **User only** | Engaging app lacking compliance/procurement approval → **the deal never closes in the first place** |

⚠️ **Trap:** don't mix up which failure follows which imbalance. Buyer-only → churn *at renewal*; User-only → deal *never closes*.

**PM reconciliation strategy:** satisfy **buyer constraints as mandatory entry gates** (non-negotiable just to get in the door), while delivering exceptional user engagement to drive organic adoption after the sale.

---

# PART 15 — Product-Market Fit (PMF)

**Full definition (memorize):** PMF is the point where a product satisfies **strong market demand**, resulting in **organic retention, enthusiastic customer advocacy, and sustainable unit economics.**

**Signals of PMF (quantitative + qualitative):** high retention (stable DAU/MAU) · low churn · organic word-of-mouth · **inbound** sales inquiries without aggressive outbound.

**Danger of scaling marketing spend before PMF — name all three specific consequences:**
1. **Burns capital rapidly**
2. **Amplifies customer churn** (you're now bringing in more people who will also leave)
3. **Exposes underlying product deficiencies to a much larger, more visible audience — faster and more publicly**

A vague "it would be risky" earns partial credit; the three named dangers earn full marks. Red-flag scenario: 30% monthly churn + sales wants 5× the marketing budget → **do not approve**; none of the PMF signals are present.

**LPP's five layers (structured path to PMF, most abstract → most concrete):** Target Customer → Underserved Needs → Value Proposition → Feature Set → UX.

⚠️ **Trap:** LPP Layer 1 "Target Customer" is a pure **segmentation** question (*who*, specifically — "mid-sized IT firms, 500–5,000 employees") — NOT a value/willingness-to-pay question (that's **Value Risk**, a different framework).

---

# PART 16 — Build–Measure–Learn, Iterate vs Pivot

## 16.1 The Loop (Ries's Lean Startup)

**BUILD** the minimum artifact (not the final product) → **MEASURE** actionable metrics via cohort retention (not vanity) → **LEARN:** Pivot or Persevere based on **validated learning** against the original hypothesis.

**Pilot-scenario pattern:** sales celebrates 500 logins in week 1 (vanity); PM points to MAU retention dropping 80% after week 2 (actionable). Instrument **cohort retention curves** — track whether users return *voluntarily* after mandatory/initial sessions.

## 16.2 Iterate vs Pivot

| | Iteration | Pivot |
|---|---|---|
| Nature | Minor tweak optimizing existing strategy | Substantive, structured correction testing a **new fundamental hypothesis** |
| Trigger | Friction is **isolated** to a minority/subset | Friction **blocks adoption across the whole segment** |

**Pivot types (name the specific type):** **Customer Segment Pivot** (shift target market) · **Value Proposition Pivot** (shift core promise). They can **combine** in one decision when both the audience and the promise change (e.g., HelpMeLearn pilot: BFSI → mid-sized IT firms + reposition around seamless HRMS integration).

⚠️ **Trap:** recommending a pivot for an isolated minority complaint (2 of 40 clinics with rural video lag → iterate with a low-bandwidth mode, don't pivot).

**Companion case — Design Within Reach (Steve Blank):** validated demand via a cheap **mail-order catalog before opening stores** (a real-world fake-door test), then executed a true **Customer Segment Pivot** (trade interior designers → direct-to-consumer). **Kate Arnold/Netflix:** rigorous **A/B testing** — user behavior, not executive opinion, decides every UI/recommendation change.

---

# PART 17 — Crossing the Chasm & Early-Adopter Segment Selection

**The Chasm:** the gap between **visionaries** (early adopters — buy on futuristic promise, tolerate bugs) and **pragmatists** (early majority — buy on proven references, stability, security compliance, complete end-to-end solutions). Many products die crossing it.

**3-step GTM strategy (in order):**
1. Target a **specific niche** segment first (not "everyone")
2. Build a **100% complete "whole product"** solving that niche's exact pain
3. Secure **bulletproof case studies/references** for pragmatist risk-aversion

**Six traits of a good early-adopter segment (confirmed exam question type — "which segment first?"):**
1. Urgent, well-recognized, painful problem (not a "nice to have")
2. Willing to try/pay before the solution is polished
3. Small and accessible enough to engage deeply and iterate fast
4. Existing relationship or trust (lowers sales friction)
5. Tolerant of rough edges — values solving the problem over polish
6. Can act as a strong reference/case study to unlock the next segment

**Trade-off logic:** at the early stage, **speed-of-learning and trust beat segment size.** Razorpay targeted small tech startups first; CRED launched only to high-credit-score users; in the HelpMeLearn case, **existing BFSI/healthcare clients** are the right first segment (trust already exists, workflows already understood, real felt pain, easy 2-month pilot offer, strong future references).

---

# PART 18 — Common Confusions Master List (read the night before the exam)

1. **"Risk" questions → answer with the CONCERN, never the reassurance.** (The #1 repeated error in the course.)
2. PM's four questions: Who / What / **Why** / How-measured — not "how to build it."
3. Project vs Product: always **all three** dimensions (Revenue/Ownership, Scope/Customization, Risk Profile).
4. **Release ≠ Sprint**: release ships to customers; sprint is the internal work cycle.
5. LPP "Target Customer" = segmentation (who), NOT willingness-to-pay (that's Value Risk).
6. Enterprise alignment: **Engineering builds the security; Legal writes the contract** — distinct jobs.
7. Kano's middle category = **Performance**, not "functional."
8. User-tests measure **Usability**, not Feasibility (feasibility = engineering spikes).
9. Drucker's **"The Unexpected"** (unplanned success) ≠ **"New Knowledge"** (stumbling on new technical/scientific knowledge).
10. **Lead User Analysis** studies advanced *customers*, not competitors (that's competitive analysis).
11. **UVP** (promise to the customer — "why choose this") ≠ **Unfair Advantage** (why competitors can't copy — moat).
12. Never re-derive a MoSCoW priority that contradicts a fact stated in the scenario.
13. CD3: highest **score** (loss÷duration) goes first — NOT the biggest raw loss.
14. **Design Sprint** (fixed 5 days: Map/Sketch/Choose/Prototype/Test, Jake Knapp) ≠ **Design Thinking** (cyclical 5 stages: Observe/Generate/Prototype/Test/Iterate). Don't merge their step names.
15. Acceptance criteria must be **testable/checkable** — never a restatement of the goal.
16. **Buyer-only** imbalance → churn at renewal; **User-only** imbalance → deal never closes.
17. Estimable (INVEST) = team can *size* it; Testable = you can *verify* it.
18. Velocity = **rolling average (3–5 sprints)**, forecasting tool only; making it a mandatory target = Goodhart's Law (estimate inflation).
19. North Star Metric = **one** metric + key inputs that drive it; don't list ten "north stars."
20. Working Backwards starts with the **press release/FAQ as if the product were done** — the narrative artifact comes *before* any roadmap or backlog.

---

# PART 19 — Rapid-Fire Cheat Sheet (last 10 minutes before the exam)

- **PM = Who / What / Why / How-measured.** Discovery = right product; Delivery = product right; run both **concurrently**.
- **PDLC = 10 looped steps**, starting from the **problem**.
- **Linear (services) vs non-linear (product) scaling**; 3 growth drivers: Cloud/SaaS, standardization demand, democratization of tech.
- **Agile Manifesto:** People>Process, Product>Paperwork, Partnership>Paper, Pivot>Plans (the "over" matters).
- **Sprint = 1–4 week timebox → shippable increment, driven by a Sprint Goal.** Velocity = rolling avg (3–5 sprints), forecast only.
- **MVP = smallest complete learning loop** — minimal but fully functional; not half-baked. Pilot = small controlled rollout, judged qualitatively.
- **MVP tests:** user-tests → usability · fake-door → value · landing page → interest.
- **Four Big Risks:** Value/Usability/Feasibility/Viability — answer with what could go wrong; Value Risk is the most dangerous.
- **RICE = (R×I×C)÷E**, higher = better; Effort is a denominator — recompute when effort changes (PCI twist).
- **MoSCoW: Musts ≤ 60% of release capacity**, 40% buffer.
- **Opportunity Score = Important + (Important − Happy)**; >15 = big opportunity.
- **CD3 = (loss/week) ÷ (weeks)**; highest score first, not biggest loss first.
- **Roadmap = problems, Now/Next/Later — never features-on-dates.** Backlog = ordered queue (position = priority).
- **North Star = one metric + driving inputs.** Working Backwards = press release first, then roadmap, then backlog. Opportunity Solution Tree = Outcome → Opportunities (problem space) → Solutions (solution space) → assumption tests.
- **Kano:** Basic (reject if absent) / Performance (more=better) / Delighter (unexpected).
- **PMF signals:** retention, low churn, word-of-mouth, inbound demand. Premature scaling = capital burn + amplified churn + exposed flaws.
- **Outputs ≠ Outcomes; pair every success metric with a guardrail** (Snapchat = the failure case).
- **Buyer ≠ User:** buyer-only → churn at renewal; user-only → deal never closes.
- **Iterate = isolated friction; Pivot = segment-wide friction** (Segment and/or Value Proposition pivots).
- **Chasm GTM:** niche → whole product → case studies. Early adopter = urgent pain + trust + tolerance + referenceable.
- **CRISPNESS:** bullet points, named terms, formulas verbatim, anchored to case facts — 25–35% of marks live here.

---

*Compiled from: CS01–CS07 lecture slides & study guides (BITS WILP, Madhu Venkat), Important Topics + Solved Past Papers (EC-2 HelpMeLearn / EC-3 NexaCore), and faculty-flagged supplementary frameworks (North Star, Working Backwards, Opportunity Solution Tree).*
