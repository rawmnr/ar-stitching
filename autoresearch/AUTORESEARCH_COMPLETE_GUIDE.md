# The Complete Autoresearch Guide

> Everything you need to apply the Karpathy Loop to optimize **anything** — prompts, code, configs, APIs, marketing copy, infrastructure, or any process with a measurable outcome.

---

## ⚡ TL;DR for Agents — Start Here

**If you are an AI agent reading this document as context, you do NOT need to read all 2000+ lines.** Here's what to read based on what you need:

| I need to... | Read these sections |
|---|---|
| **Create a program.md for a new project** | [Section 6: Universal Template](#6-the-universal-template) → fill in the blanks |
| **See working examples** | [Section 14: Ready-To-Use Examples](#14-ready-to-use-examples) → 5 copy-paste examples |
| **Understand the loop mechanics** | [Section 7: Experiment Loop](#7-the-experiment-loop--step-by-step) → the core cycle |
| **Build the eval script** | [Section 8a: Eval Harness Cookbook](#8a-eval-harness-cookbook) → full working eval scripts |
| **Handle noisy/flaky metrics** | [Section 8b: Metric Noise](#8b-handling-metric-noise-and-non-determinism) → statistical robustness |
| **Optimize parameters intelligently** | [Section 16: Search Strategy](#intelligent-parameter-search-strategy) → 4-phase protocol |
| **Debug issues** | [Section 18: Troubleshooting](#18-troubleshooting) → common failures |
| **Quick copy-paste commands** | [Section 19: Cheat Sheet](#19-quick-reference-cheat-sheet) → minimal reference |
| **Figure out what metric to use** | [Section 8c: Problem Decomposition](#8c-problem-decomposition--how-to-think-about-your-metric) → choosing metrics |
| **Validate and land results** | [Section 16a: Post-Loop Playbook](#post-loop-playbook--validating-and-landing-results) → after the loop ends |

**The 30-second version:**
1. Copy the [Universal Template](#6-the-universal-template) into a `program.md`
2. Fill in: target file, eval command, metric name, constraints
3. Create a frozen eval script (see [Cookbook](#8a-eval-harness-cookbook))
4. `git init && git add . && git commit -m "initial"`
5. Tell the agent: "Read program.md and kick off a new experiment"
6. The agent loops forever: modify → run → measure → keep/revert → repeat

---

## Table of Contents

1. [What Is Autoresearch](#1-what-is-autoresearch)
2. [The Three Primitives](#2-the-three-primitives)
3. [Architecture Deep Dive](#3-architecture-deep-dive)
4. [The program.md File — The Brain](#4-the-programmd-file--the-brain)
5. [How To Write program.md For Any Domain](#5-how-to-write-programmd-for-any-domain)
6. [The Universal Template](#6-the-universal-template)
7. [The Experiment Loop — Step By Step](#7-the-experiment-loop--step-by-step)
8. [The Evaluation System](#8-the-evaluation-system)
   - 8a. [Eval Harness Cookbook](#8a-eval-harness-cookbook) ← full working eval scripts
   - 8b. [Handling Metric Noise and Non-Determinism](#8b-handling-metric-noise-and-non-determinism)
   - 8c. [Problem Decomposition — How to Think About Your Metric](#8c-problem-decomposition--how-to-think-about-your-metric)
9. [State Management and Git As Memory](#9-state-management-and-git-as-memory)
10. [Best Practices](#10-best-practices)
    - 10a. [Pre-Flight Checklist](#10a-pre-flight-checklist) ← run before starting
11. [Writing Effective Constraints](#11-writing-effective-constraints)
    - 11a. [Multi-File Targets](#11a-multi-file-targets--when-one-file-isnt-enough)
12. [Scaling and Parallelization](#12-scaling-and-parallelization)
13. [Real-World Applications](#13-real-world-applications)
14. [Ready-To-Use Examples](#14-ready-to-use-examples)
15. [Limitations and Pitfalls](#15-limitations-and-pitfalls)
16. [Advanced Techniques](#16-advanced-techniques)
    - 16a. [Post-Loop Playbook](#post-loop-playbook--validating-and-landing-results) ← after the loop ends
17. [Community Ecosystem](#17-community-ecosystem)
18. [Troubleshooting](#18-troubleshooting)
19. [Quick Reference Cheat Sheet](#19-quick-reference-cheat-sheet)
20. [Getting Started — Hello World Example](#20-getting-started--hello-world-example)
21. [Agent Setup Instructions](#21-agent-setup-instructions)
22. [Platform and Hardware Notes](#22-platform-and-hardware-notes)
23. [The insights.md Lifecycle](#23-the-insightsmd-lifecycle)
24. [Additional Files in the Original Repo](#24-additional-files-in-the-original-repo)

---

## 1. What Is Autoresearch

Autoresearch is a pattern for **autonomous, iterative optimization** created by Andrej Karpathy (released March 7, 2026). An AI coding agent continuously:

1. **Modifies** a constrained artifact (one file)
2. **Runs** an evaluation (fixed command)
3. **Measures** the result (one scalar metric)
4. **Keeps** improvements, **reverts** regressions
5. **Repeats** indefinitely without human intervention

The canonical implementation is ~1020 lines of Python (631 in `train.py` + 390 in `prepare.py`) plus 115 lines of markdown instructions. But the **pattern itself is domain-agnostic** — it works on anything with a number to optimize.

### The Key Insight

> "You're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org." — Karpathy

The human writes English instructions. The agent writes code. The metric decides what survives.

### Why It Works

- **Constraint is the innovation.** Autoresearch succeeded where AutoGPT failed by embracing limitations: one file, one metric, one time budget.
- **Git is the memory.** Every experiment is a commit. Improvements accumulate. Failures are reverted. The branch history IS the experiment log.
- **The agent cannot cheat.** The evaluation function is frozen — the agent literally cannot modify its own scoring.
- **It runs while you sleep.** ~12 experiments/hour = ~100 experiments overnight.

### Origin

Karpathy ran the agent for 48 hours, producing **700 experiments** and discovering **20 optimizations** that improved ML training. When applied to a larger model, these yielded an **11% speedup**. The repo gained 32,800 stars and 4,400 forks in 8 days.

---

## 2. The Three Primitives

Every autoresearch loop requires exactly three things:

| Primitive | What It Is | Example (Original) | Example (General) |
|-----------|-----------|--------------------|--------------------|
| **One Editable Asset** | The single file the agent modifies | `train.py` | Any file: prompt, config, code, copy |
| **One Scalar Metric** | An objectively measurable number | `val_bpb` (lower = better) | Any number: accuracy %, latency ms, score |
| **One Fixed Time Budget** | Constant duration per experiment | 5 minutes wall-clock | Any fixed duration: 30s, 2min, 5min |

### Why These Three?

- **One file** = keeps diffs reviewable, prevents scope creep, maintains agent coherence
- **One metric** = binary keep/revert decisions, no ambiguity about "better"
- **One budget** = makes all experiments directly comparable, prevents runaway compute

If your problem has these three properties, autoresearch will work on it.

---

## 3. Architecture Deep Dive

### The Three-File System

```
program.md      ← The BRAIN     (human writes, agent reads)
target_file     ← The TARGET    (agent writes, eval reads)
eval_script     ← The JUDGE     (nobody modifies, ever)
```

#### File Roles and Permissions

| File | Who Creates | Who Reads | Who Modifies | Purpose |
|------|------------|-----------|--------------|---------|
| `program.md` | Human | Agent | Human only | Strategy, constraints, rules |
| Target file(s) | Human (initial) | Agent + Eval | Agent only | The thing being optimized |
| Eval script | Human | Agent (runs it) | Nobody | Produces the metric |
| `results.tsv` | Agent | Agent + Human | Agent (append-only) | Experiment ledger |
| `run.log` | Agent (redirected output) | Agent | Overwritten each run | Raw output capture |
| `insights.md` | Agent (optional) | Agent | Agent | Memory of what worked/failed |

#### Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    program.md (Human)                     │
│  "Modify X. Run Y. Keep if Z improves. Never stop."     │
└───────────────────────┬─────────────────────────────────┘
                        │ reads instructions
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    AI Agent (Claude/Codex)                │
│  1. Reads program.md for strategy                        │
│  2. Modifies target file with experimental idea          │
│  3. git commit                                           │
│  4. Runs eval command → run.log                          │
│  5. Extracts metric from run.log                         │
│  6. Decides: keep or revert                              │
│  7. Logs to results.tsv                                  │
│  8. GOTO 2                                               │
└──────┬──────────────────────────────┬───────────────────┘
       │ modifies                      │ runs
       ▼                               ▼
┌──────────────┐              ┌────────────────────┐
│ Target File  │───input──────│ Eval Script (FROZEN)│
│ (agent edits)│              │ (produces metric)   │
└──────────────┘              └────────┬───────────┘
                                       │ outputs
                                       ▼
                              ┌────────────────────┐
                              │ run.log → metric    │
                              │ (grep to extract)   │
                              └────────────────────┘
```

#### The Frozen Evaluator Principle

This is the **most critical architectural constraint**:

> The agent must NEVER be able to modify the evaluation function.

If the agent can rewrite both the code AND the scoring, the loop becomes meaningless — it's optimization without a fixed reference point, producing drift rather than progress. This is a miniature alignment constraint.

**In the original repo**, the evaluator (`evaluate_bpb`) is bundled inside `prepare.py` alongside data loading, tokenization, and constants — it's not a separate "eval script." This works because the entire file is marked read-only. For your own projects, you have options:

**How to enforce it:**
- Bundle the eval function inside a multi-purpose utility file and mark the whole file read-only (the Karpathy approach)
- Put the eval in its own dedicated file marked read-only in `program.md`
- Use an external command or tool the agent cannot modify
- Run the eval in a different directory/container
- State explicitly: "Do NOT modify [eval file]. This is the ground truth."

---

## 4. The program.md File — The Brain

`program.md` is the single most important file in the system. It contains everything the agent needs to run autonomously. Think of it as "programming the programmer" — you write English, the agent writes code.

### The Five Sections

Every effective `program.md` has five sections:

#### Section 1: Setup
Tells the agent how to initialize the experiment.

```markdown
## Setup
1. Agree on a run tag (e.g. `mar20`)
2. Create branch: `git checkout -b autoresearch/<tag>`
3. Read context files: [list files and their purpose]
4. Verify prerequisites exist
5. Initialize results.tsv with header row
6. Confirm and go
```

**Key principles:**
- List ALL files the agent should read for context
- Specify which files are read-only vs editable
- Include verification steps (does data exist? do dependencies work?)
- End with explicit "confirm and go" trigger

#### Section 2: Experimentation Rules
Defines the CAN/CANNOT boundaries. This is the most critical section.

```markdown
## Experimentation

**What you CAN do:**
- Modify `[target_file]` — everything is fair game: [list scope]

**What you CANNOT do:**
- Modify `[eval_script]` — read-only, contains ground truth metric
- Install new packages or add dependencies
- [Any other constraints]

**The goal: [METRIC] — [DIRECTION] is better.**

**Simplicity criterion**: [When is a small improvement worth keeping?]
```

**Key principles:**
- Be explicit about scope — "everything is fair game" within ONE file
- Hard-lock the evaluator
- State the metric and direction (lower/higher) unambiguously
- Include a simplicity criterion to prevent codebase degradation
- Add soft constraints (e.g., "memory usage should not blow up dramatically")

#### Section 3: Output Format
Tells the agent exactly what output to expect and how to parse it.

```markdown
## Output Format
The eval produces output like:
```
score: 85.3
latency_ms: 142
memory_mb: 512
```

Extract the key metric:
```
grep "^score:" run.log
```
```

**Key principles:**
- Show exact example output
- Provide the exact `grep` pattern to extract the metric
- The agent MUST know how to parse the number — ambiguity here breaks the loop

#### Section 4: Logging
Defines the experiment ledger format.

```markdown
## Logging
Log to `results.tsv` (tab-separated, NOT comma-separated).

Header: commit	metric_value	status	description

Statuses: `keep`, `discard`, `crash`

Do NOT commit results.tsv to git.
```

**Key principles:**
- Tab-separated (commas break in descriptions)
- Short 7-char commit hashes
- Include crash status for failed experiments
- Keep untracked in git (local experiment log)

#### Section 5: The Experiment Loop
The actual autonomous cycle. This section makes or breaks the system.

```markdown
## The Experiment Loop

LOOP FOREVER:

1. Check git state (current branch/commit)
2. Modify [target_file] with an experimental idea
3. git commit (with description)
4. Run: `[EVAL_COMMAND] > run.log 2>&1`
5. Extract result: `grep "[PATTERN]" run.log`
6. If grep empty → crash. Read `tail -n 50 run.log` for traceback.
7. Log to results.tsv
8. If metric improved → KEEP (commit stands)
9. If metric equal/worse → REVERT (`git reset --hard HEAD~1`)
10. Repeat

**Timeout**: Kill after [2x normal duration]. Treat as crash.

**Crashes**: Fix trivial errors (typos, imports). Skip fundamentally broken ideas.

**NEVER STOP**: Do NOT pause to ask if you should continue.
The human may be asleep. You run until manually interrupted.
If you run out of ideas, think harder.
```

**Key principles:**
- `LOOP FOREVER` and `NEVER STOP` are non-negotiable directives
- Redirect ALL output to `run.log` — never let it flood the agent's context
- Specify timeout policy (2x normal is a good default)
- Include crash recovery guidance
- Give the agent escape hatches for when it's stuck ("re-read files, combine near-misses, try radical changes")

---

## 5. How To Write program.md For Any Domain

### Step 1: Define Your Three Primitives

Ask yourself:

1. **What file does the agent modify?** (must be exactly one, or a tightly scoped set)
2. **What command produces the metric?** (must output a parseable number)
3. **How long does one experiment take?** (should be under 10 minutes ideally)

### Step 2: Build the Evaluation Harness

The eval script must:
- Be **immutable** — the agent cannot touch it
- Be **deterministic** (or as close as possible)
- Output a **parseable scalar** (e.g., `score: 85.3`)
- Run in a **fixed time budget** (or naturally terminate)
- **Genuinely reflect quality** — avoid metrics that can be gamed

Types of evaluators:
| Evaluator Type | Best For | Example |
|----------------|----------|---------|
| **Script with test suite** | Code optimization | `python test_suite.py` → pass rate |
| **Benchmark tool** | Performance | `k6 run benchmark.js` → p95 latency |
| **External tool** | Frontend | `npx lighthouse` → performance score |
| **LLM-as-judge** | Text/prompt quality | `python judge.py` → quality score 0-100 |
| **Binary test suite** | Correctness | `pytest` → pass/fail count |
| **Real-world metric** | Marketing/business | API call → conversion rate, reply rate |

### Step 3: Write the program.md

Use the template from Section 6 below. Fill in:
- File names and their roles
- Exact run commands
- Exact grep patterns
- Scope boundaries (what CAN and CANNOT change)
- The metric name and direction
- Crash handling and timeout policy

### Step 4: Set Up Git

```bash
git init  # (if not already a repo)
git add .
git commit -m "initial baseline"
git checkout -b autoresearch/<tag>
```

### Step 5: Create results.tsv

```
commit	metric	status	description
```
(Tab-separated header, one line)

### Step 6: Point the Agent and Walk Away

```
Hi, read program.md and let's kick off a new experiment! Let's do the setup first.
```

---

## 6. The Universal Template

Copy this template and fill in the bracketed sections for ANY domain.

```markdown
# [PROJECT NAME] — Autonomous Optimization

This is an experiment to autonomously optimize [WHAT YOU'RE OPTIMIZING].

## Setup

To set up a new experiment:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar20`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `[TARGET_FILE]` — the file you modify. [BRIEF DESCRIPTION OF CONTENTS].
   - `[EVAL_SCRIPT]` — frozen evaluation harness. Do not modify.
   - `[CONTEXT_FILE_1]` — [purpose]. Do not modify.
   - `[CONTEXT_FILE_2]` — [purpose]. Do not modify.
4. **Verify prerequisites**: [CHECK THAT DATA/DEPS/CONFIGS EXIST].
   If not, tell the human to run `[SETUP_COMMAND]`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
   The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

[DESCRIBE THE EXPERIMENT CONTEXT — what the system does, what matters].

**What you CAN do:**
- Modify `[TARGET_FILE]` — this is the only file you edit.
  Everything is fair game: [LIST WHAT CAN BE CHANGED —
  e.g., parameters, algorithms, structure, prompts, config values, etc.]

**What you CANNOT do:**
- Modify `[EVAL_SCRIPT]`. It is read-only.
  It contains the fixed evaluation and ground truth metric.
- Modify `[OTHER_FROZEN_FILES]`. They are read-only.
- Install new packages or add dependencies.
  You can only use what's already available.
- [ANY OTHER CONSTRAINTS — e.g., "Do not change the API contract",
  "Do not remove existing test cases", etc.]

**The goal is simple: get the [HIGHEST/LOWEST] [METRIC_NAME].**

[EXPLAIN WHY THIS METRIC MATTERS AND WHAT IT MEASURES]

**[RESOURCE_CONSTRAINT]** is a soft constraint. Some increase is acceptable
for meaningful [METRIC_NAME] gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better.
A small improvement that adds ugly complexity is not worth it.
Conversely, removing something and getting equal or better results
is a great outcome — that's a simplification win. When evaluating
whether to keep a change, weigh the complexity cost against the
improvement magnitude.

**The first run**: Your very first run should always be to establish
the baseline, so you will run the evaluation as-is without modifications.

## Output format

Once the evaluation finishes it prints a summary like this:

```
---
[METRIC_NAME]:    [EXAMPLE_VALUE]
[SECONDARY_1]:    [EXAMPLE_VALUE]
[SECONDARY_2]:    [EXAMPLE_VALUE]
```

You can extract the key metric from the log file:

```
grep "^[METRIC_NAME]:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv`
(tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	[METRIC_NAME]	[RESOURCE_METRIC]	status	description
```

1. git commit hash (short, 7 chars)
2. [METRIC_NAME] achieved (e.g. [EXAMPLE]) — use [DEFAULT] for crashes
3. [RESOURCE_METRIC] (e.g. memory_gb, duration_s) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	[METRIC]	[RESOURCE]	status	description
a1b2c3d	[VAL1]	[VAL2]	keep	baseline
b2c3d4e	[VAL1]	[VAL2]	keep	[DESCRIPTION]
c3d4e5f	[VAL1]	[VAL2]	discard	[DESCRIPTION]
d4e5f6g	0.000	0.0	crash	[DESCRIPTION]
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/<tag>`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `[TARGET_FILE]` with an experimental idea.
3. git commit
4. Run the experiment: `[EVAL_COMMAND] > run.log 2>&1`
   (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^[METRIC_NAME]:" run.log`
6. If the grep output is empty, the run crashed.
   Run `tail -n 50 run.log` to read the error and attempt a fix.
   If you can't fix it after a few attempts, give up on that idea.
7. Record the results in the tsv
   (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If [METRIC_NAME] improved ([LOWER/HIGHER]),
   you "advance" the branch, keeping the git commit
9. If [METRIC_NAME] is equal or worse,
   you git reset back to where you started

**Timeout**: Each experiment should take ~[EXPECTED_DURATION].
If a run exceeds [MAX_DURATION], kill it and treat as failure.

**Crashes**: If a run crashes, use your judgment:
If it's something easy to fix (typo, missing import), fix and re-run.
If the idea is fundamentally broken, skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup),
do NOT pause to ask the human if you should continue.
Do NOT ask "should I keep going?" or "is this a good stopping point?".
The human might be asleep, or gone from a computer and expects you to
continue working *indefinitely* until you are manually stopped.
You are autonomous. If you run out of ideas, think harder — re-read the
in-scope files for new angles, try combining previous near-misses,
try more radical changes. The loop runs until interrupted, period.

## Strategy hints (optional)

[DOMAIN-SPECIFIC GUIDANCE FOR THE AGENT]:
- [WHAT KINDS OF CHANGES TEND TO WORK]
- [WHAT TO TRY FIRST VS LATER]
- [COMMON PITFALLS IN THIS DOMAIN]
- [REFERENCES TO READ FOR INSPIRATION]

## Search strategy (include when target file has tunable parameters)

The target file has these tunable parameters:
- [PARAM_1] = [current_value] — [what it controls]
- [PARAM_2] = [current_value] — [what it controls]
- ...

Search priority (most impactful → least):
1. [PARAM_A] — [why it likely matters most]
2. [PARAM_B] — [why]
...

**Phase 1** (first ~20%): Sensitivity scan — 2-3 values per top param
**Phase 2** (next ~30%): Range search on the 3-5 most impactful
**Phase 3** (next ~30%): Combine winners, test interactions
**Phase 4** (final ~20%): Code-level improvements beyond parameters

Rules:
- For numeric params: try 0.5x and 2x first, then bisect toward optimal
- For categorical params: test all options, pick best, move on
- Record crash boundaries (e.g., "batch_size > 512 = OOM") — never exceed
- If 3+ consecutive tweaks to same param show diminishing returns, lock it
- After finding good individual values, stack them and test combinations
- If tuning numbers stopped helping, look for algorithmic/structural changes
- You may discover new variables not in this list — add them if promising
```

---

## 7. The Experiment Loop — Step By Step

### The Core Cycle

```
┌──────────────────────────────────────────────┐
│  1. READ current state (git branch, commit)  │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│  2. MODIFY target file with experimental idea │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│  3. COMMIT changes to git                     │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│  4. RUN eval: command > run.log 2>&1          │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│  5. EXTRACT metric: grep "^metric:" run.log   │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│  6. DECIDE: improved?                         │
│     YES → KEEP commit, advance branch         │
│     NO  → REVERT (git reset --hard HEAD~1)    │
│     CRASH → fix or skip, log "crash"          │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│  7. LOG to results.tsv                        │
└──────────────────┬───────────────────────────┘
                   ▼
              GOTO step 1
```

### Critical Rules

1. **Always establish baseline first.** The very first run is unmodified — it sets the reference point.
2. **Redirect ALL output.** Use `> run.log 2>&1`. Never let stdout/stderr flood the agent's context window.
3. **Commit BEFORE running.** This ensures clean git state for revert if needed.
4. **Do NOT commit results.tsv.** It's a local experiment log, untracked by git.
5. **Binary decisions only.** Improved = keep. Not improved = revert. No "maybe."
6. **Never stop.** The agent runs until the human interrupts.

### Handling Edge Cases

| Situation | Action |
|-----------|--------|
| Run crashes with traceback | Read `tail -n 50 run.log`, fix if trivial, skip if fundamental |
| Run exceeds timeout (2x normal) | Kill process, treat as crash, revert |
| Metric is same as baseline | Revert (same = no improvement) |
| Tiny improvement + lots of complexity | Revert (simplicity criterion) |
| No improvement but simpler code | Keep (simplification win) |
| Agent runs out of ideas | Re-read files, combine near-misses, try radical changes |
| Multiple consecutive crashes | Step back, try a completely different direction |
| Metric seems suspiciously good | Double-check the change — could be measurement artifact |

---

## 8. The Evaluation System

### Designing a Good Metric

The metric is the **most important design decision** in the entire system. Get this wrong and nothing else matters.

#### Requirements

| Property | Why | Example |
|----------|-----|---------|
| **Scalar** | Binary keep/revert decisions require a single number | `accuracy: 85.3` not `{precision: 0.82, recall: 0.89}` |
| **Deterministic** | Same input should produce same (or very similar) output | Fixed seed, fixed test data, fixed environment |
| **Honest** | Must genuinely reflect quality, not a gameable proxy | `val_bpb` measures real language modeling quality |
| **Fast** | Each eval should complete in minutes, not hours | 30s-5min is ideal |
| **Parseable** | Agent must extract it with a simple grep | `score: 85.3` on its own line |

#### Metric Direction Convention

State explicitly in program.md:
- "**Lower is better**" (latency, error rate, loss, cost)
- "**Higher is better**" (accuracy, throughput, score, coverage)

#### Dealing With Non-Numeric Outcomes

For subjective quality (text, design, UX), use **LLM-as-judge**:

```python
# eval.py — frozen, agent cannot modify
def judge(output):
    response = llm.evaluate(
        rubric="Rate 0-100 on clarity, relevance, accuracy",
        output=output
    )
    return response.score  # scalar number
```

**Critical rule**: The judge LLM should be a different (ideally stronger) model than the agent. Otherwise the agent learns to write outputs that please itself.

#### Multi-Metric Approaches

When you care about more than one number:

| Strategy | How | When |
|----------|-----|------|
| **Primary + soft constraint** | Optimize metric A, constraint: metric B must stay below X | Most common. E.g., optimize accuracy while memory < 8GB |
| **Weighted composite** | `score = 0.7 * accuracy + 0.3 * (1 / latency)` | When trade-offs are well-understood |
| **Tiered decision** | Optimize A; if A tied, prefer better B | Good for tiebreaking |
| **Pareto keep** | Keep if better on A without regressing on B | Preserves optionality, harder to manage |

### Output Format Best Practices

Make your eval script output in this pattern:

```
---
primary_metric:    85.3000
secondary_1:       142.5
secondary_2:       512.0
```

- Each metric on its own line
- Metric name followed by colon and spaces
- Consistent decimal precision
- Preceded by `---` separator (easy to find in logs)

---

## 8a. Eval Harness Cookbook

The eval script is where most people get stuck. Here are **full, working, copy-paste-ready** eval scripts for the most common patterns. Each one outputs a parseable metric on its own line.

### Pattern 1: Wrap Any Shell Command (Generic)

Use this when your metric comes from running any command and parsing its output.

```bash
#!/bin/bash
# eval.sh — FROZEN. Do not modify.
# Generic eval wrapper. Runs a command, extracts a metric, handles errors.

set -euo pipefail

TIMEOUT_SECONDS=300  # 5 minute max
TARGET_COMMAND="python target.py"  # What to run
METRIC_PATTERN="score"  # What to grep for

# Run with timeout
if timeout "$TIMEOUT_SECONDS" bash -c "$TARGET_COMMAND" > run.log 2>&1; then
    # Extract metric
    RESULT=$(grep "^${METRIC_PATTERN}:" run.log | awk '{print $2}')
    if [ -z "$RESULT" ]; then
        echo "---"
        echo "${METRIC_PATTERN}: ERROR"
        echo "error: metric not found in output"
        exit 1
    fi
    echo "---"
    echo "${METRIC_PATTERN}: ${RESULT}"
else
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "---"
        echo "${METRIC_PATTERN}: TIMEOUT"
        echo "error: exceeded ${TIMEOUT_SECONDS}s"
    else
        echo "---"
        echo "${METRIC_PATTERN}: CRASH"
        echo "error: exit code ${EXIT_CODE}"
        tail -20 run.log
    fi
    exit 1
fi
```

### Pattern 2: Python Benchmark (Speed / Performance)

```python
# eval_benchmark.py — FROZEN. Do not modify.
"""Benchmark a Python function for execution speed.
Runs N iterations, reports median to reduce noise."""

import time
import statistics
import sys
import traceback

NUM_RUNS = 5
TIMEOUT_PER_RUN = 60  # seconds

def run_benchmark():
    try:
        # Import the target module (agent modifies this file)
        import importlib
        import target_module
        importlib.reload(target_module)  # force fresh import

        times = []
        for i in range(NUM_RUNS):
            start = time.perf_counter()
            result = target_module.run()  # the function to benchmark
            elapsed = time.perf_counter() - start

            if elapsed > TIMEOUT_PER_RUN:
                print("---")
                print(f"duration_s: TIMEOUT")
                print(f"error: run {i+1} exceeded {TIMEOUT_PER_RUN}s")
                sys.exit(1)

            times.append(elapsed)

        median_time = statistics.median(times)
        stdev_time = statistics.stdev(times) if len(times) > 1 else 0

        # Correctness check (customize this)
        is_correct = target_module.verify()

        print("---")
        print(f"duration_s: {median_time:.6f}")
        print(f"stdev_s: {stdev_time:.6f}")
        print(f"correctness: {'PASS' if is_correct else 'FAIL'}")
        print(f"runs: {NUM_RUNS}")

        if not is_correct:
            sys.exit(1)

    except Exception as e:
        print("---")
        print(f"duration_s: CRASH")
        print(f"error: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_benchmark()
```

### Pattern 3: LLM-as-Judge (Subjective Quality)

```python
# eval_judge.py — FROZEN. Do not modify.
"""Use a stronger LLM to judge output quality.
Handles retries, cost tracking, and variance via multiple judgments."""

import json
import os
import sys
import time
import traceback

# Use any LLM client — anthropic, openai, etc.
try:
    import anthropic
    client = anthropic.Anthropic()
    MODEL = "claude-sonnet-4-6"  # judge should be STRONGER than the agent
except ImportError:
    import openai
    client = openai.OpenAI()
    MODEL = "gpt-4o"

RUBRIC = """Rate the following output on a scale of 0-100.
Consider: clarity, relevance, accuracy, completeness, conciseness.
Respond with ONLY a JSON object: {"score": <number>, "reason": "<brief reason>"}"""

NUM_JUDGMENTS = 3  # average multiple judgments to reduce variance
MAX_RETRIES = 3
TEST_CASES_FILE = "test_cases.json"  # frozen test inputs

def judge_once(output_text: str) -> float:
    """Get a single judgment score."""
    for attempt in range(MAX_RETRIES):
        try:
            if hasattr(client, 'messages'):
                # Anthropic
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=200,
                    messages=[{"role": "user", "content": f"{RUBRIC}\n\n---\nOUTPUT:\n{output_text}"}]
                )
                text = response.content[0].text
            else:
                # OpenAI
                response = client.chat.completions.create(
                    model=MODEL,
                    max_tokens=200,
                    messages=[{"role": "user", "content": f"{RUBRIC}\n\n---\nOUTPUT:\n{output_text}"}]
                )
                text = response.choices[0].message.content

            parsed = json.loads(text)
            return float(parsed["score"])

        except (json.JSONDecodeError, KeyError, Exception) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)  # exponential backoff

def evaluate():
    try:
        # Load frozen test cases
        with open(TEST_CASES_FILE) as f:
            test_cases = json.load(f)

        # Import and run the target (agent modifies this file)
        import target_module
        import importlib
        importlib.reload(target_module)

        total_score = 0
        total_cost = 0

        for i, test_input in enumerate(test_cases):
            output = target_module.generate(test_input)

            # Multiple judgments, averaged
            scores = [judge_once(output) for _ in range(NUM_JUDGMENTS)]
            avg_score = sum(scores) / len(scores)
            total_score += avg_score

        final_score = total_score / len(test_cases)

        print("---")
        print(f"quality_score: {final_score:.2f}")
        print(f"test_cases: {len(test_cases)}")
        print(f"judgments_per_case: {NUM_JUDGMENTS}")

    except Exception as e:
        print("---")
        print(f"quality_score: CRASH")
        print(f"error: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    evaluate()
```

### Pattern 4: Test Suite Pass Rate

```python
# eval_tests.py — FROZEN. Do not modify.
"""Run a test suite and extract pass rate as the metric."""

import subprocess
import sys
import re

TEST_COMMAND = ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "-q"]
TIMEOUT = 120

def run_tests():
    try:
        result = subprocess.run(
            TEST_COMMAND,
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )

        output = result.stdout + result.stderr

        # Parse pytest output: "X passed, Y failed, Z errors"
        passed = len(re.findall(r"PASSED", output))
        failed = len(re.findall(r"FAILED", output))
        errors = len(re.findall(r"ERROR", output))
        total = passed + failed + errors

        if total == 0:
            print("---")
            print("pass_rate: CRASH")
            print("error: no tests found")
            sys.exit(1)

        pass_rate = (passed / total) * 100

        print("---")
        print(f"pass_rate: {pass_rate:.2f}")
        print(f"passed: {passed}")
        print(f"failed: {failed}")
        print(f"errors: {errors}")
        print(f"total: {total}")

        # Also dump the full output to run.log for debugging
        with open("run.log", "w") as f:
            f.write(output)

    except subprocess.TimeoutExpired:
        print("---")
        print("pass_rate: TIMEOUT")
        print(f"error: tests exceeded {TIMEOUT}s")
        sys.exit(1)
    except Exception as e:
        print("---")
        print(f"pass_rate: CRASH")
        print(f"error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
```

### Pattern 5: HTTP API Benchmark

```python
# eval_api.py — FROZEN. Do not modify.
"""Benchmark an HTTP API for latency and throughput.
Handles server startup/teardown, warmup, and statistical aggregation."""

import subprocess
import time
import statistics
import sys
import json
import signal
import os

SERVER_CMD = ["python", "server.py"]
SERVER_HOST = "localhost"
SERVER_PORT = 8080
WARMUP_REQUESTS = 5
BENCHMARK_REQUESTS = 50
STARTUP_TIMEOUT = 10  # seconds to wait for server
MAX_REQUEST_TIMEOUT = 5  # seconds per request

def wait_for_server(host, port, timeout):
    """Wait for server to accept connections."""
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.create_connection((host, port), timeout=1)
            sock.close()
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.5)
    return False

def make_request(host, port, path="/"):
    """Make a single HTTP request and return latency in ms."""
    import urllib.request
    url = f"http://{host}:{port}{path}"
    start = time.perf_counter()
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=MAX_REQUEST_TIMEOUT) as resp:
        resp.read()
    return (time.perf_counter() - start) * 1000  # ms

def benchmark():
    server_proc = None
    try:
        # Start server
        server_proc = subprocess.Popen(
            SERVER_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )

        if not wait_for_server(SERVER_HOST, SERVER_PORT, STARTUP_TIMEOUT):
            print("---")
            print("p95_latency_ms: CRASH")
            print(f"error: server failed to start within {STARTUP_TIMEOUT}s")
            stderr = server_proc.stderr.read().decode() if server_proc.stderr else ""
            if stderr:
                print(f"server_stderr: {stderr[:500]}")
            sys.exit(1)

        # Warmup (discard these)
        for _ in range(WARMUP_REQUESTS):
            try:
                make_request(SERVER_HOST, SERVER_PORT)
            except Exception:
                pass

        # Benchmark
        latencies = []
        errors = 0
        for _ in range(BENCHMARK_REQUESTS):
            try:
                latency = make_request(SERVER_HOST, SERVER_PORT)
                latencies.append(latency)
            except Exception:
                errors += 1

        if not latencies:
            print("---")
            print("p95_latency_ms: CRASH")
            print(f"error: all {BENCHMARK_REQUESTS} requests failed")
            sys.exit(1)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95_idx = int(len(latencies) * 0.95)
        p95 = latencies[min(p95_idx, len(latencies) - 1)]
        p99_idx = int(len(latencies) * 0.99)
        p99 = latencies[min(p99_idx, len(latencies) - 1)]
        throughput = len(latencies) / (sum(latencies) / 1000)  # req/s
        error_rate = errors / BENCHMARK_REQUESTS

        print("---")
        print(f"p95_latency_ms: {p95:.2f}")
        print(f"p50_latency_ms: {p50:.2f}")
        print(f"p99_latency_ms: {p99:.2f}")
        print(f"throughput_rps: {throughput:.1f}")
        print(f"error_rate: {error_rate:.4f}")
        print(f"requests: {len(latencies)}")

        if error_rate > 0:
            sys.exit(1)  # any errors = fail

    except Exception as e:
        print("---")
        print(f"p95_latency_ms: CRASH")
        print(f"error: {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        # Always kill the server
        if server_proc:
            try:
                if os.name != 'nt':
                    os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
                else:
                    server_proc.terminate()
                server_proc.wait(timeout=5)
            except Exception:
                server_proc.kill()

if __name__ == "__main__":
    benchmark()
```

### Pattern 6: Composite Metric (Multiple Objectives)

```python
# eval_composite.py — FROZEN. Do not modify.
"""Combine multiple metrics into a single weighted score.
Use when you care about multiple things (e.g., speed AND accuracy)."""

import subprocess
import sys
import re

# Define weights (must sum to 1.0)
WEIGHTS = {
    "accuracy": 0.6,    # most important
    "speed": 0.3,       # important
    "simplicity": 0.1,  # nice to have
}

# Hard constraints (instant fail if violated)
HARD_CONSTRAINTS = {
    "accuracy": (">=", 0.70),   # must be at least 70%
    "memory_mb": ("<=", 1024),  # must stay under 1GB
}

def run_eval():
    try:
        result = subprocess.run(
            ["python", "target.py", "--eval"],
            capture_output=True, text=True, timeout=120
        )

        # Parse individual metrics from output
        output = result.stdout
        metrics = {}
        for line in output.split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                try:
                    metrics[key.strip()] = float(val.strip())
                except ValueError:
                    pass

        # Check hard constraints
        for metric_name, (op, threshold) in HARD_CONSTRAINTS.items():
            if metric_name not in metrics:
                print("---")
                print(f"composite_score: CRASH")
                print(f"error: missing required metric '{metric_name}'")
                sys.exit(1)

            val = metrics[metric_name]
            if op == ">=" and val < threshold:
                print("---")
                print(f"composite_score: 0.00")
                print(f"constraint_violation: {metric_name}={val} (requires {op}{threshold})")
                sys.exit(1)
            elif op == "<=" and val > threshold:
                print("---")
                print(f"composite_score: 0.00")
                print(f"constraint_violation: {metric_name}={val} (requires {op}{threshold})")
                sys.exit(1)

        # Calculate weighted composite
        composite = 0.0
        for metric_name, weight in WEIGHTS.items():
            if metric_name in metrics:
                # Normalize each metric to 0-100 range
                val = metrics[metric_name]
                composite += weight * val

        print("---")
        print(f"composite_score: {composite:.4f}")
        for name, val in metrics.items():
            print(f"{name}: {val}")

    except subprocess.TimeoutExpired:
        print("---")
        print("composite_score: TIMEOUT")
        sys.exit(1)
    except Exception as e:
        print("---")
        print(f"composite_score: CRASH")
        print(f"error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_eval()
```

### Choosing the Right Eval Pattern

| Your situation | Use Pattern |
|---|---|
| Optimizing code speed | **Pattern 2** (Python Benchmark) |
| Optimizing prompt/text quality | **Pattern 3** (LLM-as-Judge) |
| Improving test coverage/pass rate | **Pattern 4** (Test Suite) |
| Optimizing API/server performance | **Pattern 5** (HTTP API Benchmark) |
| Multiple objectives at once | **Pattern 6** (Composite Metric) |
| Anything else | **Pattern 1** (Generic Shell Wrapper) |

### Adapting Any Eval Script

Every eval script follows the same contract:

1. **Run the target** (import it, call a subprocess, hit an API)
2. **Measure the metric** (time it, parse output, call a judge)
3. **Print in parseable format** (`metric_name: value` on its own line)
4. **Exit 0 on success, exit 1 on failure** (crash/constraint violation)
5. **Handle all errors** (timeouts, crashes, missing output — never hang)

---

## 8b. Handling Metric Noise and Non-Determinism

When your metric fluctuates between runs (API latency, LLM-as-judge scores, benchmark times on busy machines), the raw keep/revert logic produces false positives and false negatives.

### The Problem

```
Run 1: score = 82.3  (baseline)
Run 2: score = 83.1  → KEEP? (real improvement or noise?)
Run 3: score = 81.9  → REVERT? (real regression or noise?)
```

Without handling noise, you'll keep ~50% false improvements and revert ~50% real improvements.

### Solution 1: Multiple Runs + Median (Simplest)

Add this to your eval script:

```python
# Inside your eval script — run N times, report median
import statistics

NUM_RUNS = 5  # odd number to avoid ties

results = [run_single_eval() for _ in range(NUM_RUNS)]
median_score = statistics.median(results)
stdev = statistics.stdev(results)

print(f"score: {median_score:.4f}")
print(f"stdev: {stdev:.4f}")
```

Add this to your `program.md`:

```markdown
**Noise handling**: The eval runs 5 times and reports the median.
An improvement only counts if the new median exceeds the previous
median by more than 1× the standard deviation.
If stdev > 10% of the metric value, the measurement is too noisy —
try a different approach rather than fine-tuning.
```

### Solution 2: Significance Threshold

Add this to your `program.md`:

```markdown
**Minimum improvement threshold**: Only count an experiment as
"improved" if the metric changed by more than [THRESHOLD].
Changes smaller than [THRESHOLD] are noise — treat as "no improvement"
and REVERT.

Thresholds by domain:
- Speed benchmarks: > 2% improvement (ignore smaller)
- LLM-as-judge scores (0-100): > 2 points improvement
- Test pass rate: > 1 test more passing
- API latency: > 5ms improvement
```

### Solution 3: Confirmation Run

Add this to your experiment loop:

```markdown
**Confirmation rule**: When an experiment shows improvement:
1. Run the eval a SECOND time (confirmation run)
2. If both runs show improvement → KEEP
3. If the second run does NOT confirm → REVERT (was noise)
This costs 2x per experiment but eliminates most false positives.
```

### Solution 4: Fixed Seeds Everywhere

```markdown
**Reproducibility**: Use fixed random seeds in ALL stochastic components:
- Set `PYTHONHASHSEED=42` before running
- Use `random.seed(42)` in any random generation
- Use `numpy.random.seed(42)` if applicable
- Use `torch.manual_seed(42)` for ML
- Use fixed test data (never generate random test inputs at eval time)
```

### When to Use Which

| Noise level | Best solution |
|---|---|
| **Low** (< 2% variance, e.g., test pass rate) | No special handling needed |
| **Medium** (2-10% variance, e.g., benchmarks) | Solution 1 (multiple runs + median) |
| **High** (10-30% variance, e.g., LLM-as-judge) | Solution 1 + Solution 2 (threshold) |
| **Very high** (> 30% variance) | Solution 3 (confirmation run) or redesign the eval |

---

## 8c. Problem Decomposition — How to Think About Your Metric

When you have a goal but aren't sure what metric to use or how to make it measurable, follow this thinking process.

### The Decomposition Walkthrough

**Start with**: "I want to optimize X"

```
Step 1: What is X concretely?
├── "Improve my cold email reply rate"
├── "Make my API faster"
├── "Improve my trading algorithm returns"
└── "Make my documentation clearer"

Step 2: Can I measure X directly with a command?
├── YES → Use that command as your eval. Done.
│   Example: "Make my API faster" → python benchmark.py → prints p95_latency_ms
│
└── NO → I need a proxy metric. Go to Step 3.

Step 3: What proxy correlates with X?
├── Can an LLM judge it? → Use LLM-as-judge (Pattern 3)
│   Example: "Improve email reply rate" → LLM scores on persuasiveness,
│   clarity, relevance (proxy for actual reply rate)
│
├── Can I create a test suite? → Use test pass rate (Pattern 4)
│   Example: "Make documentation clearer" → create 20 comprehension
│   questions, LLM answers them using only the docs, score accuracy
│
├── Can I simulate it? → Write a simulation as your eval
│   Example: "Improve trading returns" → backtest on historical data,
│   report sharpe_ratio
│
└── Can I measure a component of it? → Decompose into sub-metrics
    Example: "Improve user experience" → measure load_time_ms +
    error_rate + LLM-judge-UX-score → composite (Pattern 6)

Step 4: Validate the proxy
├── Does improving the proxy actually improve the real thing?
│   If not → choose a different proxy
├── Can the agent game the proxy without real improvement?
│   If yes → add constraints or choose a harder-to-game proxy
└── Is the proxy fast enough to evaluate in < 5 minutes?
    If not → find a faster approximation or subsample
```

### Common Decomposition Examples

| Goal | Not directly measurable because... | Proxy metric | Eval approach |
|---|---|---|---|
| Higher email reply rate | Can't send real emails in a loop | LLM-judge persuasiveness score (0-100) | Pattern 3 with email-specific rubric |
| Better documentation | "Better" is subjective | Comprehension test pass rate | Create Q&A test set, LLM answers from docs |
| Faster page loads | Need a browser | Lighthouse performance score | Pattern 1 wrapping `npx lighthouse` |
| More engaging social posts | Can't measure real engagement | LLM-judge engagement score | Pattern 3 with engagement rubric |
| More secure code | Security is multi-dimensional | Static analysis findings count | Pattern 1 wrapping `semgrep` or `bandit` |
| Better trading returns | Need historical data | Sharpe ratio on backtest data | Custom Python eval with frozen dataset |
| Cleaner code | "Clean" is subjective | Linting score + complexity metrics | Pattern 1 wrapping `pylint --score` |

### The Proxy Validation Test

Before committing to a proxy metric, run this sanity check:

```markdown
1. Take your current artifact (prompt, code, config)
2. Make a change you KNOW is genuinely better
3. Run the eval — does the metric go up?
4. Make a change you KNOW is worse
5. Run the eval — does the metric go down?
6. Make a change that's meaningless (whitespace, comments)
7. Run the eval — does the metric stay the same?

If all three pass → good proxy. Use it.
If any fail → bad proxy. Choose a different metric.
```

---

## 9. State Management and Git As Memory

### Git Is Non-Negotiable

Git provides three critical functions:

1. **Audit trail** — every experiment is a commit with a description
2. **Rollback mechanism** — `git reset --hard HEAD~1` reverts a failed experiment
3. **Inter-experiment memory** — the agent can read git log to see what was tried

### Branch Strategy

```
master (stable baseline)
  └── autoresearch/mar20 (experiment branch)
        ├── commit: baseline
        ├── commit: increase learning rate ← kept
        ├── commit: switch to GeLU ← reverted (not in branch anymore)
        ├── commit: reduce batch size ← kept
        └── ... (accumulating improvements)
```

### results.tsv — The Experiment Ledger

```
commit	metric	resource	status	description
a1b2c3d	85.3	512.0	keep	baseline
b2c3d4e	87.1	520.0	keep	add retry logic
c3d4e5f	84.9	510.0	discard	switch to async
d4e5f6g	0.0	0.0	crash	double thread pool (OOM)
e5f6g7h	88.2	515.0	keep	optimize query batching
```

**Rules:**
- Tab-separated (never comma-separated)
- 7-char commit hashes
- Untracked by git (do NOT commit this file)
- Append-only (agent never deletes rows)
- Contains ALL experiments — kept, discarded, and crashed

### insights.md — Optional Agent Memory

For long-running sessions (50+ experiments), maintain an `insights.md` file:

```markdown
## What Worked
- Increasing batch size from 64→128 improved throughput 15%
- Async processing reduced latency by 22%

## What Failed
- Thread pool >16 causes OOM on this machine
- GeLU activation consistently worse than ReLU^2 here

## Dead Ends — Do Not Retry
- Reducing vocab size below 4096 (crashes tokenizer)
- Window pattern "SSSS" (no global attention = divergence)

## Promising Directions
- Combining batch size increase with async might compound
- Haven't tried gradient checkpointing yet
```

This prevents the agent from re-exploring dead ends after context compression.

---

## 10. Best Practices

### The Golden Rules

| Rule | Why |
|------|-----|
| **One file, one metric** | Prevents scope creep; keeps diffs reviewable |
| **Lock the evaluator** | Agent cannot fake improvements |
| **Fixed time budget** | Makes all experiments directly comparable |
| **NEVER STOP directive** | Prevents agent from asking permission mid-loop |
| **Redirect to run.log** | Don't flood agent's context window |
| **Simplicity criterion** | Prevents codebase degradation over 100+ iterations |
| **Baseline first** | Establishes the reference point before any changes |
| **Git as memory** | Every experiment is a commit; full audit trail |
| **Explain WHY constraints exist** | Agent makes better edge-case decisions with rationale |

### Writing Effective program.md

1. **Be numerically explicit.** Not "make it better" but "get accuracy above 90%."
2. **Spell out mechanics.** Exact run commands, exact grep patterns, exact file paths.
3. **Include the simplicity criterion.** Always. Without it, code grows ugly over 100 iterations.
4. **Add domain knowledge.** Tell the agent what kinds of changes tend to work in your domain.
5. **Set resource constraints.** "Memory should not exceed 8GB" prevents OOM spirals.
6. **Provide strategy hints.** "Try hyperparameter changes first, then architectural changes."
7. **Include references.** "See the README for background on how the system works."

### Operational Best Practices

1. **Start with a working baseline.** The target file must run successfully before you begin.
2. **Test the eval command manually.** Run it yourself once to verify it produces parseable output.
3. **Use `uv run` or equivalent.** Ensure dependencies are locked and reproducible.
4. **Monitor the first few iterations.** Watch the agent's first 2-3 experiments to catch instruction misunderstandings.
5. **Review results.tsv periodically.** Spot-check that improvements are genuine, not artifacts.
6. **Plan for plateaus.** After 50-100 experiments, update program.md with new strategies.
7. **Keep experiments short.** Under 5 minutes each is ideal. Faster = more experiments = more discoveries.

### What NOT To Do

| Anti-Pattern | Why It's Bad |
|-------------|-------------|
| Multiple files with no clear scope | Agent gets confused, diffs become unreviewable |
| Ambiguous metric ("make it better") | Agent can't make keep/revert decisions |
| No simplicity criterion | Code becomes a mess after 50 iterations |
| Letting output flood context | Agent loses track of instructions, runs slowly |
| Mutable evaluator | Agent learns to game the metric |
| No timeout | A stuck experiment blocks all future experiments |
| Committing results.tsv | Clutters git history with non-code changes |
| Asking agent to "be careful" | Causes timidity — agents should experiment boldly |

---

## 10a. Pre-Flight Checklist

Run through this list **before launching the loop**. Skipping any item risks wasting the first 30+ minutes debugging setup.

```markdown
## Pre-Flight Checklist (copy into your setup process)

Before starting the experiment loop, verify ALL of the following:

### Environment
- [ ] Git is initialized (`git status` works)
- [ ] You're on the correct branch (or create one: `git checkout -b autoresearch/<tag>`)
- [ ] Working directory is clean (`git status` shows nothing unexpected)
- [ ] All dependencies are installed (`pip install`, `npm install`, etc. — done BEFORE loop starts)
- [ ] Required environment variables are set (API keys, paths, configs)
- [ ] If eval needs a server: port is free, server starts cleanly

### Target File
- [ ] Target file exists and is valid (no syntax errors)
- [ ] Target file runs successfully as-is (baseline works before any changes)
- [ ] Target file is committed to git (so the first revert has something to revert to)

### Eval Script
- [ ] Eval script exists and is marked read-only in program.md
- [ ] Running eval manually produces output: `[EVAL_COMMAND] > run.log 2>&1`
- [ ] Output contains the metric in parseable format: `grep "^[METRIC]:" run.log` returns a number
- [ ] Eval completes within the expected time budget (not hanging)
- [ ] Eval produces deterministic or near-deterministic results (run twice, compare)

### Logging
- [ ] `results.tsv` header row exists (or will be created by setup step)
- [ ] `results.tsv` is NOT tracked by git (add to `.gitignore`)
- [ ] `run.log` is NOT tracked by git (add to `.gitignore`)

### program.md
- [ ] Metric direction is stated ("higher is better" or "lower is better")
- [ ] Eval command is complete and correct (copy-paste-runnable)
- [ ] Grep pattern matches the eval output exactly
- [ ] Timeout/time budget is specified
- [ ] NEVER STOP directive is included
- [ ] Simplicity criterion is included
```

### Quick Validation Command

Run this one-liner to verify your setup works end-to-end:

```bash
# Replace [EVAL_COMMAND] and [METRIC] with your actual values
[EVAL_COMMAND] > run.log 2>&1 && grep "^[METRIC]:" run.log && echo "✓ READY" || echo "✗ FIX SETUP"
```

If it prints `✓ READY`, launch the agent. If not, fix the issue before proceeding.

---

## 11. Writing Effective Constraints

### The Three Types of Constraints

#### 1. Hard Boundaries (MUST obey)
```markdown
**What you CANNOT do:**
- Modify eval.py — it is the ground truth
- Install new packages
- Change the API contract
```

#### 2. Soft Constraints (prefer, but can trade off)
```markdown
**Memory** is a soft constraint. Some increase is acceptable
for meaningful gains, but should not blow up dramatically.
```

#### 3. Taste Constraints (qualitative judgment)
```markdown
**Simplicity criterion**: A 0.001 improvement that adds 20 lines
of hacky code? Probably not worth it. A 0.001 improvement from
deleting code? Definitely keep.
```

### Constraint Design Principles

1. **State what's forbidden, not just what's allowed.** Agents are creative — they'll find loopholes in positive-only constraints.
2. **Explain the reason.** "Do not modify eval.py BECAUSE it contains the ground truth metric" is better than just "do not modify eval.py."
3. **Be specific about boundaries.** "Memory should not exceed 24GB" is better than "don't use too much memory."
4. **Include escape hatches.** "If you can't fix a crash after 3 attempts, skip and move on."
5. **Prevent metric gaming explicitly.** "Random seed changes that produce tiny improvements are not genuine improvements."

### Example: Tight vs Loose Constraints

**Too loose (agent wastes time):**
```markdown
Try to make the code faster.
```

**Too tight (agent can't explore):**
```markdown
Only change the BATCH_SIZE constant on line 47.
Change it to values between 32 and 256 in powers of 2.
```

**Just right:**
```markdown
Modify server.py to reduce response latency.
Everything is fair game: algorithms, data structures,
caching, query optimization, connection pooling.
Do NOT modify the API contract (endpoints, request/response schemas).
Do NOT modify benchmark.sh or test_data/.
Goal: lowest p95_latency_ms.
```

---

## 11a. Multi-File Targets — When One File Isn't Enough

The original pattern says "one file." But many real-world tasks require editing multiple files. Here's the rule:

### The Rule

**Multiple editable files are fine IF:**

1. **They're tightly scoped** — they form one logical unit (e.g., all files in `src/components/`)
2. **They're explicitly listed** — program.md names every editable file, not "files in src/"
3. **Diffs stay reviewable** — after 100 experiments, `git diff baseline...HEAD` should be readable
4. **The agent treats them as one change** — modify multiple files in a single commit, revert all or none

**Multiple files are dangerous IF:**

1. Files are in unrelated parts of the codebase
2. Changes in one file can break another in non-obvious ways
3. The total scope is so large the agent can't hold it all in context

### How to Declare Multi-File Targets in program.md

```markdown
## Experimentation

**What you CAN modify** (all of these files form a single logical unit):
- `src/pipeline/extract.py` — data extraction logic
- `src/pipeline/transform.py` — transformation rules
- `src/pipeline/load.py` — loading strategy
- `src/pipeline/config.yaml` — pipeline configuration

These files are ONE system. When you change one, consider if related
files need coordinated changes. Always commit all related changes together.

**What you CANNOT modify:**
- Anything outside `src/pipeline/`
- `tests/` — frozen test suite
- `eval.py` — frozen evaluator
```

### When to Split Into Separate Loops

If your files are truly independent, run **separate autoresearch loops** — one per file/subsystem. Each gets its own `program.md`, eval, and branch. This prevents the combinatorial explosion of multi-file mutations.

---

## 12. Scaling and Parallelization

### Single Agent (Default)

- ~12 experiments/hour (5 min each)
- ~100 experiments overnight
- Greedy hill-climbing — one change at a time
- Simple, no coordination overhead

### Multi-GPU Parallel (SkyPilot Approach)

SkyPilot ran 16 GPUs in parallel and achieved:
- **910 experiments in 8 hours** (vs ~96 sequential)
- **9x wall-clock speedup**
- **Cost**: ~$300 total (~$9 API + ~$260 compute)

Key insight: With 16 GPUs, the agent ran **factorial grids** instead of greedy search. In one wave it tested 6 aspect ratios simultaneously, catching interaction effects that sequential search would miss.

The agent even developed hardware-aware strategies without being told — screening hypotheses cheaply on H100s, then confirming winners on H200s.

### Git Worktree Parallelism

Run multiple agents on the same repo without conflicts:

```bash
git worktree add ../experiment-a feature/experiment-a
git worktree add ../experiment-b feature/experiment-b
```

Each agent gets its own working directory on a separate branch, all pointing to the same `.git` directory. Instant setup, shared history, minimal disk overhead.

### The Practical Ceiling

5-7 concurrent agents before rate limits and merge conflicts consume gains. Beyond that, you need coordination infrastructure (shared experiment log, deduplication).

### Distributed Collaboration (autoresearch@home)

Karpathy's vision: "SETI@home style" — hundreds of agents collaborating:

- Coordination via shared memory (e.g., Ensue)
- Agents claim experiments via semantic deduplication (no duplicates even with different phrasing)
- Claims auto-expire after 15 minutes
- Every result includes complete source for reproduction
- First coordinated run: 20+ agents, 1000+ experiments, 54 hours, 3.2% improvement

---

## 13. Real-World Applications

### Proven Applications (People Are Doing This Now)

| Domain | Editable Asset | Metric | Time Budget | Result |
|--------|---------------|--------|-------------|--------|
| **ML Training** | `train.py` (architecture, optimizer, hyperparams) | `val_bpb` (lower=better) | 5 min | 11% speedup over 700 experiments |
| **Template Engine** | Shopify Liquid source code | Benchmark time (lower=better) | Build+test | 53% faster after decades of human optimization |
| **Cold Email** | Subject line + body copy | Reply rate (higher=better) | 24-48h feedback cycle | Significant lift in response rates |
| **Landing Pages** | Page HTML/CSS/copy | Conversion rate (higher=better) | A/B test window | Measurable conversion improvements |
| **Ad Creative** | Creative assets + targeting | CPA / ROAS (lower/higher) | Campaign window | Reduced cost per acquisition |
| **Voice Agents** | System prompt | Task success rate (higher=better) | ~30s eval | 25% → 100% success rate over 20 iterations |
| **CUDA Kernels** | Triton/CUDA kernel code | Execution time (lower=better) | ~40 experiments/hour | Faster kernels found automatically |
| **Agent Quality** | Agent instruction file | LangSmith eval score (higher=better) | Single eval run | Self-improving agents |
| **SEO** | Page content + meta tags | AI-driven traffic (higher=better) | Crawl cycle | 920% average lift reported |
| **Frontend Perf** | CSS/JS bundle | Lighthouse score (higher=better) | Build+audit ~60s | Lighthouse 39 → 92 |

### Untapped Applications (The Opportunities)

| Domain | Editable Asset | Possible Metric | Notes |
|--------|---------------|-----------------|-------|
| **Database Queries** | Query config / index hints | p95 latency | Benchmark against production-like load |
| **API Performance** | Server config + handler code | Response time / throughput | k6 or wrk load test |
| **Test Coverage** | Test files | Coverage % | `pytest --cov` output |
| **Infrastructure** | Terraform / K8s configs | Cost / compliance score | Apply + validate |
| **Compiler Flags** | Build configuration | Binary size / execution speed | Compile + benchmark |
| **Resume/CV** | Resume content | ATS match score | LLM-as-judge against job description |
| **Documentation** | Doc files | Readability score / completeness | LLM-as-judge rubric |
| **Recommendation System** | Algorithm / weights | Diversity + engagement score | Backtest against dataset |
| **Pricing Strategy** | Pricing config | Revenue / conversion composite | Simulation against historical data |
| **UI/UX** | Component code | Task completion time / error rate | Automated user simulation |
| **Education** | Tutoring prompt/content | Student score improvement | Pre/post test evaluation |
| **Legal Contracts** | Contract clauses | Risk score / compliance % | LLM-as-judge with legal rubric |
| **Recipe Formulation** | Ingredient ratios / instructions | Taste/nutrition composite score | LLM-as-judge or formula |

### The Universal Rule

**If there's a number, it loops.** — Udit Goenka's generalized autoresearch variant

Any domain with these three properties is a candidate:
1. A file that can be modified
2. A command that produces a number
3. A direction (higher or lower = better)

---

## 14. Ready-To-Use Examples

### Example A: Optimize a System Prompt

```markdown
# Prompt Optimization — Autonomous Loop

## Setup
1. Create branch: `git checkout -b autoresearch/<tag>`
2. Read these files:
   - `system_prompt.txt` — the file you modify. Contains the system prompt.
   - `eval.py` — frozen evaluation harness. Do not modify.
   - `test_cases.json` — frozen test dataset. Do not modify.
3. Verify API key is set: `echo $ANTHROPIC_API_KEY`
4. Initialize results.tsv with header row
5. Confirm and go

## Experimentation

We are optimizing a system prompt for a customer support chatbot.
The eval sends 50 test queries through the prompt and scores responses.

**What you CAN do:**
- Modify `system_prompt.txt` — everything is fair game:
  instructions, tone, examples, formatting, constraints, persona.

**What you CANNOT do:**
- Modify `eval.py` — read-only ground truth.
- Modify `test_cases.json` — fixed test dataset.
- Change the model being called (it's hardcoded in eval.py).

**The goal: get the highest `accuracy` score.**

**Token usage** is a soft constraint. Some increase is acceptable
for meaningful accuracy gains, but prompts over 2000 tokens are too expensive.

**Simplicity criterion**: Shorter prompts that achieve the same accuracy are preferred.
If removing a section doesn't hurt accuracy, remove it.

**First run**: Establish baseline with unmodified system_prompt.txt.

## Output format

```
---
accuracy:      72.0
avg_tokens:    1847
avg_latency:   2.3
```

Extract: `grep "^accuracy:" run.log`

## Logging results

Tab-separated results.tsv:
```
commit	accuracy	avg_tokens	status	description
```

## The experiment loop

LOOP FOREVER:
1. Read current system_prompt.txt and results.tsv
2. Modify system_prompt.txt with an experimental idea
3. git commit
4. Run: `python eval.py > run.log 2>&1`
5. Extract: `grep "^accuracy:" run.log`
6. If improved → KEEP. If equal/worse → REVERT.
7. Log to results.tsv

**Timeout**: Kill after 10 minutes.
**NEVER STOP.**

## Strategy hints
- Try restructuring instructions for clarity first
- Add specific examples for common failure cases
- Experiment with persona/tone changes
- Try chain-of-thought instructions
- Remove unnecessary constraints that might confuse the model
```

### Example B: Optimize API Response Time

```markdown
# API Performance — Autonomous Loop

## Setup
1. Create branch: `git checkout -b autoresearch/<tag>`
2. Read these files:
   - `server.py` — the file you modify. Contains API handlers and config.
   - `benchmark.sh` — frozen load test. Do not modify.
   - `test_data/` — frozen test fixtures. Do not modify.
3. Verify server starts: `python server.py &` then `curl localhost:8080/health`
4. Initialize results.tsv
5. Confirm and go

## Experimentation

We are optimizing a Python API server for response latency under load.

**What you CAN do:**
- Modify `server.py` — algorithms, data structures, caching,
  query optimization, connection pooling, async patterns, batch processing.

**What you CANNOT do:**
- Modify `benchmark.sh` or anything in `test_data/`.
- Change the API contract (endpoints, request/response schemas must stay the same).
- Install new packages.

**The goal: get the lowest `p95_latency_ms`.**

**Memory** is a soft constraint. Stay under 2GB.

**Simplicity criterion**: Simpler code achieving equal latency wins.

**First run**: Baseline with unmodified server.py.

## Output format

```
---
p95_latency_ms:   142.5
p99_latency_ms:   287.3
throughput_rps:    1250
memory_mb:        512
error_rate:       0.00
```

Extract: `grep "^p95_latency_ms:" run.log`

## Logging results

```
commit	p95_latency_ms	memory_mb	status	description
```

## The experiment loop

LOOP FOREVER:
1. Modify server.py with optimization idea
2. git commit
3. Start server, run benchmark, stop server:
   `python server.py &
   sleep 2
   bash benchmark.sh > run.log 2>&1
   kill %1`
4. Extract: `grep "^p95_latency_ms:" run.log`
5. If p95 decreased → KEEP. If equal/increased → REVERT.
6. Log to results.tsv

**NEVER STOP.**
```

### Example C: Optimize Frontend Performance

```markdown
# Frontend Performance — Autonomous Loop

## Setup
1. Create branch: `git checkout -b autoresearch/<tag>`
2. Read these files:
   - Files in `src/` — the files you modify.
   - `lighthouse.config.js` — frozen config. Do not modify.
   - `package.json` — do not add/remove dependencies.
3. Verify build works: `npm run build`
4. Initialize results.tsv
5. Confirm and go

## Experimentation

**What you CAN do:**
- Modify any file in `src/` — components, styles, bundling config,
  lazy loading, image optimization, code splitting.

**What you CANNOT do:**
- Modify `lighthouse.config.js`.
- Add or remove npm packages.
- Remove any user-facing functionality.

**The goal: get the highest Lighthouse `performance` score.**

**Bundle size** is a soft constraint. Prefer smaller bundles at equal performance.

**First run**: Baseline with `npm run build && npx lighthouse`.

## Output format

```
---
performance:    67
accessibility:  92
best_practices: 88
seo:            91
bundle_kb:      342
```

Extract: `grep "^performance:" run.log`

## The experiment loop

LOOP FOREVER:
1. Modify source files
2. git commit
3. Run: `npm run build && npx lighthouse http://localhost:3000 \
   --output=json --chrome-flags="--headless" | \
   python parse_lighthouse.py > run.log 2>&1`
4. Extract: `grep "^performance:" run.log`
5. If score increased → KEEP. If equal/decreased → REVERT.
6. Log to results.tsv

**NEVER STOP.**
```

### Example D: Optimize Test Coverage

```markdown
# Test Coverage — Autonomous Loop

## Setup
1. Create branch: `git checkout -b autoresearch/<tag>`
2. Read these files:
   - Files in `tests/` — the files you modify.
   - Files in `src/` — read-only, the code being tested.
3. Verify tests pass: `pytest --cov=src`
4. Initialize results.tsv
5. Confirm and go

## Experimentation

**What you CAN do:**
- Add, modify, or restructure test files in `tests/`.
- Create new test files.

**What you CANNOT do:**
- Modify any source code in `src/`.
- Install new test dependencies.
- Delete existing passing tests.

**The goal: get the highest `coverage_pct`.**

**Test count** is a secondary metric. Fewer tests achieving same coverage is preferred.

**First run**: Baseline coverage.

## Output format

```
---
coverage_pct:   64.2
tests_passed:   47
tests_failed:   0
duration_s:     12.3
```

Extract: `grep "^coverage_pct:" run.log`

## The experiment loop

LOOP FOREVER:
1. Modify test files
2. git commit
3. Run: `python run_coverage.py > run.log 2>&1`
4. Extract: `grep "^coverage_pct:" run.log`
5. IMPORTANT: Also check `grep "^tests_failed:" run.log` — if any fail, REVERT.
6. If coverage increased (and all pass) → KEEP.
7. If coverage equal/decreased → REVERT.
8. Log to results.tsv

**NEVER STOP.**
```

### Example E: Optimize a Configuration File

```markdown
# Config Optimization — Autonomous Loop

## Setup
1. Create branch: `git checkout -b autoresearch/<tag>`
2. Read these files:
   - `config.yaml` — the file you modify.
   - `benchmark.py` — frozen benchmark. Do not modify.
   - `README.md` — context on what each config key does.
3. Initialize results.tsv
5. Confirm and go

## Experimentation

**What you CAN do:**
- Modify any value in `config.yaml` — cache sizes, timeouts,
  thread counts, batch sizes, retry limits, connection pool sizes.

**What you CANNOT do:**
- Modify `benchmark.py`.
- Add config keys that the application doesn't support.
- Set values outside documented ranges (see README).

**The goal: get the highest `throughput_rps` (requests per second).**

**Error rate** must stay at 0.00. Any config that causes errors → REVERT.

**First run**: Baseline with default config.

## Output format

```
---
throughput_rps:   1250
p95_latency_ms:   142.5
error_rate:       0.00
memory_mb:        512
```

Extract: `grep "^throughput_rps:" run.log`

## The experiment loop

LOOP FOREVER:
1. Modify config.yaml
2. git commit
3. Run: `python benchmark.py > run.log 2>&1`
4. Extract: `grep "^throughput_rps:\|^error_rate:" run.log`
5. If error_rate > 0 → REVERT immediately.
6. If throughput increased → KEEP.
7. If throughput equal/decreased → REVERT.
8. Log to results.tsv

**NEVER STOP.**
```

---

## 15. Limitations and Pitfalls

### Fundamental Limitations

#### Greedy Hill-Climbing
On a single agent, the loop is stuck doing greedy search — try one thing, check, repeat. It misses parameter interaction effects that only surface when testing combinations. Multi-agent parallelism partially addresses this.

#### Time Budget Blindness
Fixed time budget creates blindness to changes that only reveal their value over longer runs. Quick-to-show improvements get found; slow-burn insights remain invisible.

#### Goodhart's Law
> "When a measure becomes a target, it ceases to be a good measure."

Documented gaming behaviors:
- **Seed hacking**: Changing random seeds for marginal gains without genuine improvement
- **Throughput gaming**: Optimizing for more iterations rather than better iterations
- **Proxy mismatch**: Optimizing for fast convergence that doesn't hold at longer training

**Defense**: The frozen evaluator is the primary defense. The simplicity criterion is the secondary defense. Human review of winners is the final defense.

#### The Multiple Comparisons Problem
Running 700 experiments at 5% false positive rate guarantees some false positives. Without statistical correction, some "improvements" are noise.

**Defense**: Periodically re-run the full accumulated stack from scratch. If the compound gain holds, it's real.

#### Complexity Degradation
Without the simplicity rule, code grows complex enough that the agent's understanding degrades over successive sessions. Code becomes unreadable after 100+ mutations.

#### Agent Timidity
Karpathy noted: "The models feel very 'cagy' and 'scared' when given problems." Agents tend to make conservative, incremental changes rather than bold experimental leaps.

**Defense**: Include in program.md: "If you've been making small changes for 10+ iterations with minimal improvement, try something radically different."

### Operational Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Output floods context | Agent slows down, loses instructions | Always redirect to run.log |
| No baseline established | No reference for keep/revert | Always run unmodified first |
| Mutable evaluator | Agent "improves" by changing scoring | Hard-lock eval file |
| No timeout | Single stuck run blocks everything | Kill after 2x expected duration |
| results.tsv committed to git | Messy git history | Keep untracked |
| Ambiguous metric direction | Agent oscillates | State explicitly: "lower is better" |
| No simplicity criterion | Code becomes unreadable | Always include this rule |
| Agent asks permission | Loop stalls overnight | Explicit NEVER STOP directive |

### Platform-Specific Results

Optimizations found on one platform may not transfer to another. The 5-minute budget optimizes for YOUR hardware specifically. Results are not portable.

### Security Consideration

AI-generated code contains **2.74x more vulnerabilities** than human code. Every agent-generated artifact needs human security review before production deployment.

---

## 16. Advanced Techniques

### Intelligent Parameter Search Strategy

When the target file has tunable parameters (hyperparameters, config values, thresholds, etc.), do NOT search blindly. Use a structured, intelligent approach:

#### The Four-Phase Search Protocol

Include this in your `program.md` to make the agent search smart:

```markdown
## Search Strategy

You have access to parameters in [TARGET_FILE]. Search INTELLIGENTLY, not randomly.

### Phase 1: Sensitivity Scan (first ~20% of experiments)

Test each parameter independently with 2-3 widely spaced values to find
which ones actually move the metric. Keep baseline values for everything else.

Example: If there are 20 parameters, try 2 values each = ~40 quick experiments.
Result: Rank parameters by impact. Identify the 3-5 that matter most.
The rest can be ignored or fine-tuned later.

### Phase 2: Focused Range Search (next ~30% of experiments)

For each high-impact parameter (from Phase 1), narrow the range:
- If doubling it helped, try 1.5x, 2x, 2.5x, 3x
- If it helped up to a point then hurt, binary search for the sweet spot
- Use bisection: try midpoints between best-so-far and last-failed value

### Phase 3: Combination Stacking (next ~30% of experiments)

Stack the best individual values found in Phase 2.
Test 2-parameter combos first, then 3-parameter, etc.
Watch for interactions:
- If A+B together is WORSE than A alone → they conflict, don't combine
- If A+B together is BETTER than A or B alone → synergy, keep both

### Phase 4: Discovery and Refinement (final ~20% of experiments)

- Read the code deeply — find variables/logic not in the parameter list
  that could be parameterized or changed
- Try structural/algorithmic changes beyond parameter tuning
- Simplification pass: remove unnecessary parameters/code
- Fine-tune the best configuration with small adjustments (±5-10%)
```

#### Smart Behaviors to Encode in program.md

Add these directives to make the agent handle edge cases intelligently:

```markdown
## Intelligent Search Rules

**Range detection**: When testing a numeric parameter:
- Try 0.5x, 1x (baseline), 2x first to find the direction
- If 2x is better, try 3x, 4x until it gets worse, then bisect
- If 0.5x is better, try 0.25x, 0.1x until it gets worse, then bisect
- Track the "ceiling" and "floor" for each parameter in insights.md

**Crash boundary mapping**: When a value causes a crash (OOM, overflow, etc.):
- Record the crash boundary in insights.md (e.g., "batch_size > 512 causes OOM")
- Never try values above that boundary again
- The sweet spot is often just below the crash boundary

**NaN / infinity detection**: If the metric returns NaN, infinity, or
an unreasonable value:
- Revert immediately
- The parameter region that caused it is dangerous — avoid nearby values
- Try approaching from a different direction (change a different parameter)

**Diminishing returns**: If the last 3 adjustments to a parameter each
improved the metric by less than the previous one:
- You've found the approximate optimum for that parameter
- Lock it and move to the next most impactful parameter
- Come back only during Phase 4 fine-tuning

**Parallel parameter changes**: When you've established that two parameters
are independent (changing one doesn't affect the optimal value of the other),
you MAY change both in a single experiment to save iterations.
But if the combined change makes things worse, revert BOTH and test separately.

**Batch experiments for discrete choices**: When testing categorical options
(e.g., algorithm type, activation function, strategy variant):
- Test all options in rapid succession (one experiment each)
- Keep the best, move on — no need to revisit
- These are fast to resolve since there's no range to search

**Discovery mode**: Beyond the given parameters, you should:
- Read the code to understand what each parameter actually controls
- Look for hardcoded values that could be made configurable
- Look for algorithmic choices (sorting method, data structure, caching)
  that could be swapped for better alternatives
- Look for unnecessary computation that could be removed
- Consider adding new parameters if you see an opportunity
```

#### Parameter Search Decision Tree

```
Is this parameter numeric or categorical?
├── CATEGORICAL (e.g., algorithm="A"|"B"|"C")
│   → Test all options, pick best, move on
│
└── NUMERIC (e.g., learning_rate=0.01)
    │
    ├── Step 1: Try 0.5x and 2x of current value
    │   ├── Both worse → current is near-optimal, skip
    │   ├── 2x better → explore upward (3x, 4x, ...)
    │   ├── 0.5x better → explore downward (0.25x, 0.1x, ...)
    │   └── Both better → unusual, try 0.25x and 4x to find range
    │
    ├── Step 2: Find the boundary
    │   ├── Keep going until metric gets worse or crashes
    │   └── The boundary is: last_good_value
    │
    ├── Step 3: Bisect
    │   └── Try midpoint between best and boundary
    │       Repeat 2-3 times for precision
    │
    └── Step 4: Lock and move on
        Record optimal value in insights.md
```

#### Example: Trading Algorithm With 20 Parameters

```markdown
## Strategy for trading_agent.py

This algorithm has 20 parameters. DO NOT test randomly.

### Step 1: Categorize parameters by likely impact
HIGH impact (test first):
- lookback_period, position_size, entry_threshold, stop_loss
  (these directly control trade decisions)

MEDIUM impact (test second):
- trailing_stop_pct, profit_target, max_positions, rebalance_freq
  (these control risk and portfolio management)

LOW impact (test last, if time permits):
- log_level, warmup_periods, slippage_model, fee_override, ...
  (operational parameters, unlikely to improve sharpe_ratio)

### Step 2: Sensitivity scan on HIGH impact params
For each HIGH param, try 3 values: 0.5x, 1x (baseline), 2x
Expected: ~12 experiments to identify the 2-3 that matter most

### Step 3: Focused search on top 2-3 parameters
Binary search for optimal value of each
Expected: ~15-20 experiments

### Step 4: Combination stacking
Combine best individual values
Test for interactions (some params may conflict)
Expected: ~10-15 experiments

### Step 5: Discovery
Read the code for algorithmic improvements beyond parameter tuning
Can we add a filter? Change the signal? Modify the execution logic?
Expected: ~10-20 experiments

### Total: ~50-70 experiments to reach near-optimal configuration
```

#### Template: Adding Search Strategy to Any program.md

Add this block to the Strategy Hints section of your program.md:

```markdown
## Search strategy

The target file has the following tunable parameters:
[LIST PARAMETERS WITH CURRENT VALUES AND BRIEF DESCRIPTIONS]

Search priority (most likely to impact the metric → least likely):
1. [PARAM_A] — [why it matters]
2. [PARAM_B] — [why it matters]
3. [PARAM_C] — [why it matters]
...
N. [PARAM_N] — [probably doesn't matter much]

**Phase 1** (experiments 1-15): Sensitivity scan — 2 values per top-10 param
**Phase 2** (experiments 16-35): Range search on the 3-5 most impactful
**Phase 3** (experiments 36-55): Combine winners, test interactions
**Phase 4** (experiments 56+): Code-level improvements beyond parameters

**Crash boundaries**: Record any value that causes crashes.
Never exceed that boundary again.

**When stuck**: Change strategy, not parameter. If tuning numbers
stopped helping, look for algorithmic/structural improvements.
```

---

### Phase-Based Campaigns

Instead of one continuous loop, structure experiments in phases:

```markdown
## Strategy

Phase 1 (iterations 1-30): Explore hyperparameters freely.
Phase 2 (iterations 31-60): Lock best hyperparameters. Explore architecture.
Phase 3 (iterations 61-80): Lock best architecture. Fine-tune everything.
Phase 4 (iterations 81-100): Simplification pass — remove anything unnecessary.
```

### Plateau Detection

Add explicit plateau-breaking instructions:

```markdown
**Plateau rule**: If 5 consecutive experiments show no improvement:
1. Re-read all in-scope files from scratch
2. Review results.tsv for patterns — what categories of changes worked?
3. Try combining two previous near-miss ideas
4. Attempt one radical change (completely different approach)
5. If still stuck after 3 more attempts, start a simplification pass
```

### Insights Compression

Maintain a running summary to prevent context rot:

```markdown
**After every 10 experiments**, update `insights.md` with:
- What worked (and approximate improvement magnitude)
- What failed (and why, if known)
- Dead ends (do not retry these)
- Promising unexplored directions
```

### Meta-Optimization

The human can iterate on program.md itself:

1. Run overnight with program.md v1 → observe 100 experiments
2. Analyze: Was the agent too conservative? Too reckless? Stuck in a rut?
3. Tighten/loosen constraints, add domain knowledge, refine strategy hints
4. Run overnight with program.md v2 → compare convergence speed and final metric
5. Repeat

Track which program.md versions produced faster convergence and better results.

### Multi-Agent Configurations

#### Independent Researchers
Multiple agents, each with their own GPU/branch, exploring independently.
Combine winners manually.

#### Hierarchical
One "chief scientist" agent delegates experiment ideas to worker agents.
Workers run experiments and report back. Chief decides strategy.

#### Specialist Agents
Different agents focus on different aspects:
- Agent A: Hyperparameter tuning
- Agent B: Architecture changes
- Agent C: Optimization/efficiency
Combine winning changes across agents.

### LLM-as-Judge for Subjective Quality

When your metric isn't a clean number:

```python
# eval.py — FROZEN
import anthropic

def judge(output, rubric):
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",  # stronger than agent model
        messages=[{
            "role": "user",
            "content": f"""Score this output 0-100 based on this rubric:

RUBRIC:
{rubric}

OUTPUT:
{output}

Respond with ONLY the numeric score."""
        }]
    )
    return float(response.content[0].text.strip())
```

**Critical rules for LLM-as-judge:**
- Judge model should be stronger than the agent model
- Use specific, unambiguous rubrics with examples
- Calibrate against human judgments before deploying
- Run multiple judge calls and average (reduces variance)
- The agent must NEVER modify the judge

### Combining Near-Misses

When an experiment improves one metric but slightly regresses another, it's a "near-miss." These are valuable:

```markdown
**Near-miss rule**: If an experiment improves val_bpb by >0.005 but
slightly increases memory (by <5%), keep it marked as "near-miss" in results.tsv.
After 20 experiments, review near-misses and try combining the top 2-3.
```

---

### Post-Loop Playbook — Validating and Landing Results

The loop generates improvements. But before you trust them and merge to main, you need to validate, review, and land properly.

#### Step 1: Verify Compound Gains Are Real

The multiple-comparisons problem means some "improvements" are statistical noise. After 100 experiments, expect ~5 false positives at a 5% rate.

```bash
# Reset to baseline and replay ALL kept changes at once
git stash  # save current state
git checkout main
git checkout -b validation/full-rerun

# Apply the accumulated diff
git diff main...autoresearch/<tag> -- [TARGET_FILE] | git apply

# Run the eval from scratch — multiple times
for i in 1 2 3 4 5; do
  [EVAL_COMMAND] > "validation_run_$i.log" 2>&1
  grep "^[METRIC]:" "validation_run_$i.log"
done

# If all 5 runs show improvement over baseline → gains are REAL
# If some runs don't → some improvements were noise, investigate
```

#### Step 2: Review the Accumulated Diff

```bash
# See everything the agent changed
git diff main...autoresearch/<tag> -- [TARGET_FILE]

# Check: is it readable? Is it maintainable?
# The simplicity criterion should prevent mess, but verify.
```

**Review checklist:**
- [ ] Code is readable by a human (not just the agent)
- [ ] No unintended side effects (security, breaking changes)
- [ ] No hardcoded magic numbers without explanation
- [ ] Changes make logical sense (not just random seed hacking)
- [ ] No unnecessary complexity added for marginal gains

#### Step 3: Write a Summary of Findings

After a long run, document what the agent discovered:

```markdown
## Autoresearch Summary — [PROJECT] — [DATE]

### Starting metric: [BASELINE_VALUE]
### Final metric: [FINAL_VALUE]
### Improvement: [PERCENTAGE]%
### Experiments run: [N]
### Experiments kept: [K] (K/N = [HIT_RATE]%)

### Key discoveries:
1. [MOST IMPACTFUL CHANGE] — contributed ~[X]% of total improvement
2. [SECOND MOST IMPACTFUL] — contributed ~[Y]%
3. [THIRD] — contributed ~[Z]%

### Dead ends (don't retry in future):
- [THING THAT DIDN'T WORK AND WHY]
- [ANOTHER DEAD END]

### Potential future improvements:
- [IDEA THE AGENT DIDN'T TRY OR COULDN'T ACHIEVE]
```

#### Step 4: Merge or Cherry-Pick

**Option A: Merge the whole branch** (if all changes are clean)
```bash
git checkout main
git merge autoresearch/<tag> --no-ff -m "autoresearch: [summary of improvements]"
```

**Option B: Cherry-pick specific improvements** (if some changes are questionable)
```bash
git checkout main
# Find the commits that actually improved things
git log --oneline autoresearch/<tag> | head -20
# Cherry-pick the good ones
git cherry-pick <commit1> <commit2> <commit3>
```

**Option C: Squash into one clean commit** (for clean history)
```bash
git checkout main
git merge --squash autoresearch/<tag>
git commit -m "autoresearch: [X]% improvement in [METRIC] over [N] experiments"
```

#### Step 5: Clean Up

```bash
# Delete the experiment branch
git branch -d autoresearch/<tag>

# Archive results
cp results.tsv "results/results_<tag>_$(date +%Y%m%d).tsv"
cp insights.md "results/insights_<tag>_$(date +%Y%m%d).md"

# Clean working files
rm -f run.log results.tsv insights.md
```

---

## 17. Community Ecosystem

### Notable Projects

| Project | Stars | What It Does |
|---------|-------|-------------|
| **karpathy/autoresearch** | 32,800 | Original — ML training optimization |
| **pi-autoresearch** | 1,377 | Session persistence, dashboard, confidence scoring, branch-aware tracking |
| **autoresearch-mlx** | 701 | Apple Silicon support via MLX framework |
| **autokernel** | 608 | CUDA/Triton kernel optimization, ~40 experiments/hour |
| **autoresearch (domain-agnostic)** | 216 | Generalized for any metric: test coverage, SEO, bundle size |
| **autoresearch-agents** | 72 | Agent-optimizing-agent via LangSmith eval scores |
| **agent-factory** | 70 | Scrapes real problems, builds and deploys agents overnight |
| **autoresearch@home** | — | Distributed SETI@home-style, 20+ agents, 1000+ experiments |
| **n-autoresearch** | — | Multi-GPU with adaptive search (explore/exploit/combine/ablation) |

### pi-autoresearch Features Worth Knowing

- **Session persistence**: `autoresearch.jsonl` + `autoresearch.md` survive restarts
- **Confidence scoring**: After 3+ runs, computes confidence = |best_improvement| / MAD
- **Dashboard**: Status widget, expanded table, fullscreen overlay with vim navigation
- **Branch awareness**: Each git branch has independent session files
- **Backpressure checks**: Optional `autoresearch.checks.sh` for correctness validation

### Key Community Findings

- **Shopify**: Ran autoresearch on 20-year-old Liquid template engine → 53% faster
- **SkyPilot**: 16 GPUs, 910 experiments, 8 hours, $300 total
- **Meta REA**: Multi-day autonomous ML lifecycle, 2x model accuracy, 5x engineering productivity
- **Hyperspace**: 35 agents, 333 experiments, peer-to-peer gossip protocol, CPU-only agents developed different strategies than GPU agents

---

## 18. Troubleshooting

### Common Issues and Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| Agent stops and asks permission | Missing NEVER STOP directive | Add explicit "NEVER STOP" section to program.md |
| Agent modifies the eval script | Constraints not explicit enough | Add "CANNOT modify [eval file]" with explanation |
| Metric not parseable | Output format unclear | Provide exact example output and grep pattern in program.md |
| All experiments crash | Baseline doesn't work | Fix the baseline first — it must run successfully |
| Agent keeps reverting everything | Metric too noisy / threshold too strict | Use averaged metrics or relax the improvement threshold |
| Context window fills up | Output not redirected | Use `> run.log 2>&1`, never use `tee` |
| Agent re-explores dead ends | No memory across context resets | Maintain insights.md with failed approaches |
| Codebase becomes unreadable | No simplicity criterion | Add simplicity rules to program.md |
| Agent makes only tiny changes | Timidity | Add "try radical changes" prompt in program.md |
| False improvements | Measurement noise / overfitting | Re-run accumulated stack from scratch to verify |

### Debugging the Loop

1. **Check results.tsv**: Is the agent making progress? Is it mostly crashing?
2. **Check run.log**: What errors are occurring?
3. **Check git log**: Are commits accumulating? Or is everything getting reverted?
4. **Check program.md**: Are instructions clear? Are constraints specific enough?
5. **Watch 2-3 iterations live**: Catch misunderstandings early.

### When to Restart

- After 20+ consecutive non-improving experiments → rewrite program.md with new strategy
- After major crashes that corrupt state → reset branch, start fresh
- When insights.md shows all major directions have been explored → new phase needed

---

## 19. Quick Reference Cheat Sheet

### Minimum Viable Autoresearch

You need exactly:
1. One file to optimize
2. One command that outputs a number
3. A program.md that says what to do

### Setup Commands

```bash
# Initialize
git checkout -b autoresearch/$(date +%b%d | tr '[:upper:]' '[:lower:]')

# Create results.tsv header (tab-separated, 5 columns)
printf "commit\tmetric\tresource\tstatus\tdescription\n" > results.tsv

# Prompt the agent
"Read program.md and let's kick off a new experiment!"
```

### program.md Skeleton (Minimum)

```markdown
# [Name]

## Setup
1. Create branch. Read [files]. Init results.tsv. Go.

## Rules
- CAN modify: [target file]
- CANNOT modify: [eval file]
- GOAL: [highest/lowest] [metric] from `[command]`

## Loop
LOOP FOREVER:
1. Modify → commit → run `[command] > run.log 2>&1`
2. grep "^[metric]:" run.log
3. Improved → keep. Else → revert. Log to results.tsv.
NEVER STOP.
```

### Key Directives (Copy-Paste These)

```markdown
**NEVER STOP**: Do NOT pause to ask the human if you should continue.
The human might be asleep. You run until manually interrupted.
```

```markdown
**Simplicity criterion**: All else being equal, simpler is better.
A small improvement that adds ugly complexity is not worth it.
```

```markdown
**Redirect output**: `[command] > run.log 2>&1`
Do NOT let output flood your context.
```

```markdown
**Crashes**: Fix trivial errors (typos, imports).
Skip fundamentally broken ideas. Log "crash" and move on.
```

### The Decision Tree

```
Run completed?
├── NO (crash/timeout) → Fix if trivial, else skip. Log "crash".
└── YES
    └── Metric improved?
        ├── YES → Is the code simpler or reasonably clean?
        │   ├── YES → KEEP ✓
        │   └── NO (too hacky for tiny gain) → REVERT ✗
        └── NO
            └── Is the code significantly simpler?
                ├── YES (same metric, less code) → KEEP ✓
                └── NO → REVERT ✗
```

### Metric → eval command examples

| What You Want to Optimize | Eval Command |
|---------------------------|-------------|
| Python script speed | `python -c "import time; start=time.time(); exec(open('script.py').read()); print(f'duration_s: {time.time()-start:.3f}')"` |
| Test pass rate | `pytest --tb=no -q 2>&1 \| python parse_pytest.py` |
| API latency | `k6 run benchmark.js --out json 2>&1 \| python parse_k6.py` |
| Lighthouse score | `npx lighthouse URL --output=json \| python parse_lighthouse.py` |
| LLM response quality | `python eval_with_judge.py` (calls stronger LLM to score) |
| Bundle size | `npm run build 2>&1 && echo "bundle_kb: $(du -sk dist \| cut -f1)"` |
| Memory usage | `python -c "import tracemalloc; tracemalloc.start(); exec(...); print(f'peak_mb: {tracemalloc.get_traced_memory()[1]/1e6:.1f}')"` |
| Compile time | `time gcc -O2 program.c -o program 2>&1` |
| Accuracy on dataset | `python eval.py --dataset test_set.json` |

---

## 20. Getting Started — Hello World Example

The fastest way to understand autoresearch is to run a trivial example end-to-end.

### The Task: Optimize a Sorting Function for Speed

We have a Python file that sorts a list. We want to make it as fast as possible.

#### Step 1: Create the target file

**`sort_algo.py`** — the file the agent modifies:
```python
def sort_data(data):
    """Sort a list of integers. Agent: optimize this for speed."""
    return sorted(data)
```

#### Step 2: Create the frozen eval script

**`eval_sort.py`** — the agent must NEVER modify this:
```python
import time
import random

from sort_algo import sort_data

random.seed(42)
test_data = [random.randint(0, 1_000_000) for _ in range(500_000)]

# Verify correctness
result = sort_data(test_data[:1000])
expected = sorted(test_data[:1000])
if result != expected:
    print("correctness: FAIL")
    exit(1)

# Benchmark (average of 5 runs)
times = []
for _ in range(5):
    data_copy = test_data.copy()
    start = time.perf_counter()
    sort_data(data_copy)
    elapsed = time.perf_counter() - start
    times.append(elapsed)

avg_time = sum(times) / len(times)
print("---")
print(f"duration_s: {avg_time:.4f}")
print(f"correctness: PASS")
```

#### Step 3: Create program.md

**`program.md`**:
```markdown
# Sort Optimization

## Setup
1. Create branch: `git checkout -b autoresearch/<tag>`
2. Read `sort_algo.py` (file you modify) and `eval_sort.py` (frozen eval)
3. Initialize results.tsv. Confirm and go.

## Rules
- CAN modify: `sort_algo.py` — algorithm, data structures, anything
- CANNOT modify: `eval_sort.py` — frozen ground truth
- CANNOT: install new packages
- GOAL: lowest `duration_s` from `python eval_sort.py`
- Correctness is mandatory — if eval prints FAIL, revert immediately

## Output
Extract: `grep "^duration_s:" run.log`

## Loop
LOOP FOREVER:
1. Modify sort_algo.py → git commit → `python eval_sort.py > run.log 2>&1`
2. If duration_s decreased → KEEP. If equal/increased → REVERT.
3. Log to results.tsv (commit, duration_s, status, description)
NEVER STOP.
```

#### Step 4: Run it

```bash
git init && git add . && git commit -m "initial"
# Then launch your agent and say:
# "Read program.md and kick off a new experiment!"
```

That's it. In under 5 minutes you'll see the agent trying different algorithms, data structures, and optimizations — keeping what works, reverting what doesn't.

---

## 21. Agent Setup Instructions

### How to Launch the Loop With Different Agents

The README says: *"Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions)."* Here's what that means concretely:

#### Claude Code (Recommended)

```bash
# Navigate to your project directory
cd /path/to/your/project

# Launch Claude Code
claude

# Then type this prompt:
# "Read program.md and let's kick off a new experiment! Let's do the setup first."
```

**Permission settings**: For autonomous operation, the agent needs permission to:
- Read and write files
- Run shell commands (the eval command)
- Execute git commands

You can either approve each action manually for the first few iterations (recommended for learning), or configure auto-accept for trusted commands.

#### Codex CLI

```bash
codex --model codex-mini-latest \
  "Read program.md and kick off a new experiment"
```

**Note**: Codex has been reported to sometimes ignore the "NEVER STOP" directive. Claude handles autonomous loops more reliably.

#### Cursor / VS Code with AI

Open the project in Cursor, open the chat, and paste:
```
Read program.md in this repo and let's kick off a new autoresearch experiment.
```

#### General Pattern (Any Agent)

The agent just needs:
1. File system access (read + write)
2. Shell access (to run eval commands and git)
3. A starting prompt: "Read program.md and begin"

### Security: "Disable All Permissions" Means

- The agent should NOT need internet access (unless your eval requires API calls)
- The agent should NOT need to install packages
- The agent should be sandboxed to the project directory
- Review the first 2-3 iterations manually to verify behavior
- AI-generated code contains ~2.74x more vulnerabilities than human code — **always review winners before deploying to production**

---

## 22. Platform and Hardware Notes

### For the Original ML Use Case

The original `train.py` requires:
- **NVIDIA GPU** (tested on H100, works on RTX 3090/4090 with adjusted settings)
- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **CUDA toolkit** with Flash Attention support

For non-NVIDIA platforms, use community forks:
- **macOS**: [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (Apple Silicon via MLX)
- **macOS**: [autoresearch-macos](https://github.com/miolini/autoresearch-macos)
- **Windows**: [autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx)
- **AMD**: [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (ROCm support)

### For Non-ML Use Cases (No GPU Needed)

When you apply the autoresearch pattern to non-ML tasks (prompt optimization, code performance, config tuning, etc.), there are **no special hardware requirements**. You only need:
- A machine that can run your eval command
- Git installed
- An AI agent (Claude Code, Codex, Cursor, etc.)
- An API key if your eval uses an LLM-as-judge

### Cost Estimation

| Use Case | Cost Per Experiment | 100 Experiments |
|----------|-------------------|-----------------|
| Code optimization (local eval) | ~$0.02-0.05 agent API | ~$2-5 |
| Prompt optimization (LLM eval) | ~$0.05-0.20 agent + eval API | ~$5-20 |
| ML training (GPU) | ~$0.01 agent + $0.50-2.00 GPU | ~$50-200 |
| Parallel (16 GPU cluster) | ~$0.01 agent + $2-5 GPU/experiment | ~$200-500 |

Agent API costs scale with context length and number of tool calls per iteration. Shorter program.md and cleaner eval output reduce costs.

#### Cost Estimation Formula

Use this formula to estimate before committing:

```
total_cost = (N × agent_cost_per_experiment) + (N × eval_cost_per_experiment)

Where:
  N = planned number of experiments

  agent_cost_per_experiment =
    (input_tokens + output_tokens) × token_price
    ≈ (program.md_size + target_file_size + run.log_tail + tool_overhead) × price
    Typical: $0.02-0.10 per experiment for Sonnet-class models

  eval_cost_per_experiment =
    local eval: $0 (just compute time)
    LLM-as-judge: (tokens_per_judgment × num_judgments × num_test_cases) × token_price
    GPU compute: (eval_duration_hours × GPU_hourly_rate)

Example calculations:
  100 experiments × $0.05 agent + $0 local eval       = $5
  100 experiments × $0.05 agent + $0.15 LLM-judge      = $20
  100 experiments × $0.05 agent + $1.00 GPU/experiment  = $105
```

**Quick estimate**: Before running, do 3 experiments manually, check your API dashboard for cost, multiply by planned total.

### External Dependencies in Evals

When your eval requires external resources (APIs, servers, databases), handle these explicitly:

#### API-Dependent Evals (LLM-as-judge, external services)

```markdown
## External dependencies

This eval calls an external API. Handle these scenarios:

**Rate limits**: If the API returns 429 (rate limited):
- Wait 30 seconds and retry (up to 3 retries)
- If still rate limited, log "rate_limited" and skip this experiment
- Do NOT count rate-limited experiments as failures

**API failures**: If the API returns 5xx or times out:
- Retry once after 10 seconds
- If still failing, log "api_error" and skip
- Do NOT revert the change — it wasn't tested, skip it

**Cost guard**: Track cumulative API spend in insights.md.
If total eval API cost exceeds $[BUDGET], STOP and alert the human.
```

#### Server-Dependent Evals (API performance, web apps)

```markdown
## Server management

**Startup**: Before each eval run:
1. Check if port [PORT] is free: `lsof -i :[PORT]` (Unix) or `netstat -an | find "[PORT]"` (Windows)
2. If port is busy, kill the process or use a different port
3. Start server: `[SERVER_COMMAND] &`
4. Wait for server to respond: poll `http://localhost:[PORT]/health` every 1s, up to 15s
5. If server doesn't start within 15s → log "server_start_failed", skip experiment

**Teardown**: After each eval run:
1. Kill the server process (store PID at startup)
2. Verify port is freed
3. This MUST happen even if the eval crashed — use a finally/trap block

**Port conflicts**: If the port is already in use from a previous crashed run:
1. Find and kill the orphaned process
2. Wait 2 seconds for port release
3. Then proceed with startup
```

#### Database-Dependent Evals

```markdown
## Database management

**Before each experiment**: Reset database to known state
(run seed script, restore snapshot, or use in-memory DB).
**Never** let experiments accumulate state in the database across runs.

**If database connection fails**: Retry once, then skip experiment.
Do NOT treat connection failures as experiment failures.
```

---

## 23. The insights.md Lifecycle

### What It Is

An optional file the agent maintains as running memory of what worked, what failed, and what to try next. It prevents re-exploring dead ends after context compression or across sessions.

### Where It Lives

- In the project root directory, alongside `results.tsv`
- **Untracked by git** (add to `.gitignore`) — it's operational state, not code
- Read by the agent at the start of each iteration (or at least when stuck)

### How to Instruct the Agent

Add this to your `program.md`:

```markdown
## Insights memory

Maintain an `insights.md` file in the project root (untracked by git).

**After every 10 experiments**, update it with:
- **What worked**: Changes that improved the metric (with approximate magnitude)
- **What failed**: Changes that hurt the metric (with reason if known)
- **Dead ends**: Approaches that should NOT be retried
- **Promising directions**: Untested ideas worth exploring

**When stuck** (5+ consecutive non-improving experiments):
1. Re-read `insights.md` from top to bottom
2. Avoid anything listed under "Dead ends"
3. Try combining ideas from "What worked"
4. Explore ideas from "Promising directions"
```

### Example insights.md

```markdown
## What Worked
- Batch size 64→128: +3.2% throughput (experiment 4)
- Connection pooling: -18ms p95 latency (experiment 7)
- Query caching for repeated lookups: -22ms p95 (experiment 12)

## What Failed
- Async handlers: +5ms latency, more complexity (experiment 5)
- Reducing thread pool to 4: -8% throughput (experiment 9)

## Dead Ends — Do Not Retry
- Thread pool >32: causes OOM on this machine
- gzip compression on responses: adds latency, not worth it

## Promising Directions
- Haven't tried read-ahead buffering yet
- Could combine connection pooling + query caching
- Consider request batching for bulk endpoints
```

---

## 24. Additional Files in the Original Repo

The original autoresearch repo contains several files beyond the core three:

| File | Purpose |
|------|---------|
| `prepare.py` | Frozen: data download, tokenizer training, dataloader, `evaluate_bpb()`, constants |
| `train.py` | Agent-editable: model architecture, optimizer, training loop (631 lines) |
| `program.md` | Human-editable: agent instructions (115 lines) |
| `pyproject.toml` | Locked dependency list — defines what packages are available |
| `analysis.ipynb` | Jupyter notebook for post-hoc analysis of experiment results |
| `.gitignore` | Ignores `results.tsv`, `run.log`, `worktrees/`, `results/`, `queue/`, `CLAUDE.md`, `AGENTS.md` |
| `progress.png` | Visualization of experiment progress (shown in README) |

**Notable `.gitignore` entries:**
- `results.tsv` and `run.log` — confirms these are local, not committed
- `worktrees/` and `queue/` — infrastructure for multi-agent parallel runs
- `CLAUDE.md` and `AGENTS.md` — per-session generated agent prompt files
- `results/` — directory for storing detailed experiment outputs

The original `program.md` is kept deliberately bare-bones — it has **no strategy hints section** and no insights.md instructions. The README says this is intentional: *"The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time."*

---

## Appendix: The Philosophy

### From Karpathy's README

> *"One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of 'group meeting'. That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies."*

### The Human's New Job

You are no longer the coder. You are the **constraint designer**. Your job is to:

1. Choose the right metric (avoid Goodhart's Law)
2. Write clear instructions (program.md)
3. Set appropriate boundaries (what can/cannot change)
4. Review results (verify improvements are genuine)
5. Iterate on strategy (update program.md when the agent plateaus)

### The Core Equation

```
Better program.md → Better agent behavior → Better results
```

The quality of your results is directly proportional to the quality of your instructions. The agent is as good as the constraints you give it.

---

*This guide was compiled from analysis of the autoresearch repository, extensive online research including Fortune, VentureBeat, The New Stack, Hacker News, SkyPilot, Hybrid Horizons, MindStudio, and community discussions. Last updated: 2026-03-20.*
