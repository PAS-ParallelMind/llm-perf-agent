---
name: deployment_planning
description: Structured 4-stage workflow for sizing an LLM deployment given a service requirement
when_to_use: User describes a service they want to deploy (purpose, users, latency target) and wants hardware / software guidance
---

# Deployment-planning workflow

You are now in **planning mode**. The user wants a structured deployment
recommendation, not a free-form Q&A. The aim is to converge on the optimal
**hardware + software** configuration for a given serving workload. Follow
the four stages below in order. Persist a structured artifact at the end of
each stage to `stages/0N_<name>.yaml` so the run is auditable and re-entrant.
Announce each stage as you enter it (e.g. `**Stage 1 — Service requirement**`).

**Mode override**: in planning mode, **sweeping over candidates is the
point** of Stage 2. The base system-prompt rule "converge, don't sweep"
does NOT apply here.

## Mental model — what the workflow is doing

- **Stage 1** characterises the workload (estimates, past experience, or
  request traces) into concrete knobs.
- **Stage 2** runs the workload through each hardware candidate. The
  simulator yields a **theoretical upper bound** per candidate. The
  workflow then forks into two paths:
  - **Path A — fix the software** (vLLM, SGLang, etc. as-is). The agent
    applies a historical efficiency factor to the theoretical numbers to
    project the **actual** performance of the baseline stack on each
    candidate, and the user picks hardware on those projected numbers.
    Stage 3 then reports the real measurement against the projection and
    the workflow ends.
  - **Path B — optimize the software**. The user picks hardware on the
    theoretical numbers, accepting the implementation gap as something
    they'll close. Stage 3 measures the baseline against theory and
    diagnoses *where* the gap is. Stage 4 (parked) would close it.

Stage 2 reports **both** paths' Pareto tables and the user picks. Only
the chosen plan is persisted.

## STEP 0 — REQUIRED road-map (write this FIRST, do not skip)

**This step is mandatory.** Your very next user-facing reply after
invoking this skill MUST present the workflow road-map. Do NOT silently
jump into Stage 1's tool calls — the user must see the structure first.

Concretely: the assistant message in which you make your first tool
call for Stage 1 must have **content** that begins with this road-map,
*then* the Stage 1 header (`**Stage 1 — Service requirement**`), and
*then* whatever brief setup text Stage 1 needs. Include the road-map
verbatim or lightly paraphrased — the four numbered bullets and the
closing "let me know" line are required:

> You want to deploy a model — here's how I'll work through it:
> 1. **Workload profile** — translate your service description into
>    concrete numbers (request rate, input/output length, latency target).
> 2. **Hardware sweep** — simulate each candidate to get the theoretical
>    upper bound, and also project the actual performance of a baseline
>    software stack (vLLM/SGLang) using historical measurements. Two
>    paths to pick from: **(A) fix the software** — pick hardware on the
>    projected-actual numbers; **(B) optimize the software** — pick
>    hardware on the theoretical numbers and treat the gap as something
>    to close. You decide.
> 3. **Measure baseline performance** — Path A reports the real numbers
>    against the projection; Path B measures the baseline + diagnoses
>    where the implementation gap is.
> 4. *(future work)* Optimisation suggestions for the bottleneck (Path B
>    only).
>
> Let me know if you'd rather just chat about specific numbers instead.

You do NOT need to wait for the user to acknowledge — emit this in the
same assistant message that starts Stage 1, then continue.

**Checklist before Stage 1**: did your reply contain the four numbered
road-map bullets? If not, you skipped Step 0 — fix it before any
Stage-1 tool call.

## Stage 1 — Service requirement → Workload profile

**Gate**: before reading anything else in this section, confirm the
Step-0 road-map is in your reply for this turn. If it isn't, stop —
prepend it before any Stage-1 work.

**Input**: the user's natural-language service description.

**Process**: Translate it to concrete workload knobs using the rules in
the system prompt (archetype table, Little's Law for users → concurrency,
reasoning-budget for output_len, `num_requests` sizing rule). Fill any
missing field with an *explicitly stated assumption*. Do **not** stall to
ask the user — proceed with stated assumptions and let them override.

**Output artifact** → write to `stages/01_workload.yaml`:

```yaml
model: <PRESET_MODELS key>
request_rate: <float>              # req/s, Poisson arrivals (NOT concurrency)
input_len: <int>
output_len: <int>                  # incl. reasoning budget if model reasons
num_requests: <int>                # max(200, ~10 × request_rate) for stable percentiles
max_num_batched_tokens: <int>      # vLLM --max-num-batched-tokens (e.g. 8192)
max_concurrent_requests: <int>     # vLLM --max-num-seqs server cap (e.g. 1024)
range_ratio: <float>               # 0.0 = fixed lengths (clean modeled-vs-measured
                                   # comparison); 0.1 = ±10% per-request jitter
target_request_latency_s: <float>  # end-to-end per-request seconds
assumptions:
  archetype: <chat|RAG|code|summarization|agentic>
  think_time_s: <float>
  peak_active_fraction: <float>
  reasoning_budget_tokens: <int|0 if non-reasoning model>
  notes: "<anything else worth surfacing>"
```

Concurrency is **not** a workload knob in this schema — it's a *result*
of running the workload through the server (the simulator and benchmark
both report observed peak / mean in-flight). The deployer's lever is
the arrival rate.

**Reply to the user**: a tight summary — the workload knobs you chose, the
assumptions you made, and a one-line invitation to override.

## Stage 2 — Workload + candidates → Deployment plan

**Input**: WorkloadProfile from Stage 1.

**Process**:

1. **Candidates**:
   - **If the user named hardware**, validate each name against the
     catalog (`list_gpus` keys). For any name **not** in the catalog,
     tell the user plainly: *"I can't evaluate `<name>` — it isn't in
     my preset catalog. I can use the closest match (`<key>`), or you
     can drop it."* Do NOT attempt to run `pareto_sweep` with an
     unknown name; the modeling tools key on preset keys. Proceed only
     with the validated subset (plus any closest-match substitutions
     the user accepts).
   - **If the user didn't name hardware**, ask them what they're
     considering — *don't* paste the catalog list. Their actual
     hardware may not even be in the catalog, and offering a menu
     biases them toward your presets. Only after they answer do you
     validate as above.

2. **Theoretical sweep — Table A**: call
   `pareto_sweep(workload_file="stages/01_workload.yaml",
   candidates=[...])`. The result is the theoretical Pareto frontier
   (cost vs latency, saturation-aware). Keep this tool theoretical —
   do NOT extend it with measurement lookups; the projection happens
   in step 3 below.

3. **Projected-actual sweep — Table B** (agent-driven):
   - For each candidate that fits and isn't saturated, call
     `lookup_measurements(model=<workload model>, gpu=<candidate>)`. Each
     returned record carries an `efficiency` dict with keys
     `output_throughput`, `total_throughput`, `tpot`, `ttft` — defined
     as `measured / theory` for throughput and `theory / measured` for
     latency, so values <1.0 always mean "measured is worse than
     theory".
   - **Pick a projection factor** by judgement: prefer the most recent
     record at a similar operating point (close `request_rate`,
     `input_len`, `output_len`). If multiple records exist, you can
     average, take the most recent, or weight by operating-point
     proximity — your call. State which record(s) you used.
   - Apply to that candidate's Table-A row:
     - `projected_output_throughput = theoretical × efficiency.output_throughput`
     - For projected **request latency**: the store keeps per-token
       efficiency, not end-to-end. Approximate with `efficiency.tpot`
       for output-heavy workloads (large `output_len`), or
       `efficiency.ttft` for prefill-heavy ones. Or rebuild from raw
       fields: `measured_e2e ≈ ttft_ms/1000 + tpot_ms × output_len /
       1000` and the same for theory, then use the ratio.
     - Recompute `projected_$/Mtok` from projected throughput.
   - If **no record exists** for (model, gpu), flag the candidate in
     Table B with "—" and a note — do NOT fabricate an efficiency
     factor or copy theoretical numbers across.
   - Build Table B with the same columns as Table A and mark its own
     Pareto frontier on projected numbers.

4. **Present both tables side by side** and name two recommendations:
   - **Path A — fix the software**: cheapest candidate that meets
     `target_request_latency_s` on **projected actual** numbers.
   - **Path B — optimize the software**: cheapest candidate that meets
     the target on **theoretical** numbers.
   - If a candidate appears in only one table (no historical data for
     Table B), say so plainly. If neither table has a candidate meeting
     the target, recommend the closest miss in each.

5. **Ask the user which path** they want to take. Do NOT pre-decide —
   the choice depends on whether they want to ship on a known stack
   (A) or invest in software optimisation (B). Once they pick, restate
   the chosen plan.

6. **Single-GPU only** for now — the modeling tools don't model TP/PP/DP
   scaling. Output `n_gpus: 1` and a parallelism stub.

**Output artifact** → write **only the chosen plan** to
`stages/02_plan.yaml`:

```yaml
candidate_set: [<gpu keys>]
chosen_path: A | B                   # A = fix software / project & verify
                                     # B = optimize software / diagnose gap
recommended:
  gpu: <key>
  n_gpus: 1
  parallelism: {tp: 1, pp: 1, dp: 1}
  rationale: "<why this point under the chosen path>"
meets_target: <true|false>
projection:                          # Path A only — null for Path B
  factor_throughput: <float>         # what you applied to project actual
  factor_latency: <float>
  source_record: "<measurement id / timestamp>"
  projected_throughput_tps: <float>
  projected_latency_s: <float>
```

**Reply to the user**: the two tables (or compact summaries), the two
recommendations, the path question. After they answer, confirm the
chosen plan and tell them "next: Stage 3 measures the baseline."

## Stage 3 — Measure baseline performance

**Input**: WorkloadProfile + DeploymentPlan (with `chosen_path`).

The goal: measure the **actual** performance of the deployed system.
What we do with that measurement depends on the path the user picked in
Stage 2. Branch accordingly.

### Path A — fix the software

**Goal**: report the real performance against the Stage-2 projection.
No verdict — present numbers and the delta; the user decides whether
the delta is acceptable.

**Process**:

1. **Ask about a running server.** Tell the user the recommended config
   and ask: *"Do you have an OpenAI-compatible server running this
   config? Give me the `base_url` and the served `model` name and I'll
   measure it."*
   - **If no** → tell them they need a deployed server to compare
     against the projection, and to come back when one exists. Don't
     fall back to anything — Path A's whole point is the projection vs.
     measurement comparison.

2. **Measure**: call `benchmark_serving(base_url=...,
   workload_file="stages/01_workload.yaml", gpu=<recommended preset>,
   tensor_parallel=1)`. Override `model` if the server serves under a
   different id than the workload's `model`.

3. **Compare**: present measured throughput / latency next to the
   `projection` block from `stages/02_plan.yaml`. Show the delta as
   percentages. Report numbers; do **not** say "pass" / "fail" — the
   user decides what's acceptable.

**Output artifact** → write to `stages/03_report.yaml`:

```yaml
path: A
projected:
  output_throughput_tps: <float>
  request_latency_s: <float>
measured:
  output_throughput_tps: <float>
  request_latency_s: <float>
delta:                               # measured / projected
  throughput: <float>
  latency: <float>
source_record: "<measurement id used for projection>"
```

**Reply to the user**: measured vs projected (numbers + delta %), the
source record used for the projection, and one line on whether the
delta is large enough that they may want to revisit Stage 2 with a
different candidate. End the workflow here.

### Path B — optimize the software

**Goal**: measure baseline performance, compare to theoretical, and
identify the per-op bottleneck — sets up Stage 4 (parked).

The measured numbers come from `benchmark_serving` against the running
server. The theoretical numbers come from `simulate_serving` and serve
two purposes: (a) the reference to compute the implementation-efficiency
gap against, and (b) the per-op breakdown that diagnoses *where* the
gap is coming from (the real benchmark gives end-to-end numbers but no
per-op detail).

**Process**:

1. **Ask about a running server.** Same as Path A.
   - **If yes** → run the measurement path (steps 2–5).
   - **If no** → theoretical-only fallback at the end. Tell the user
     that's a roofline reference, not actual performance, and that they
     should come back to Stage 3 after deploying.

2. **Measure** (primary): `benchmark_serving(base_url=...,
   workload_file="stages/01_workload.yaml", gpu=<recommended preset>,
   tensor_parallel=1)`. Single-GPU only for now. Auto-records to the
   measurement store.

3. **Theoretical reference**: `simulate_serving(
   workload_file="stages/01_workload.yaml", gpu=<recommended preset>)`.
   Same workload, hardware filled in. Use for the roofline comparison
   and the per-op `Bottleneck: <op> — N% of step (BOUND, avg M
   tokens/batch)` line.

4. **Historical cross-check**: `lookup_measurements(model,
   recommended_gpu)` for any prior measurements (other framework
   versions, nearby operating points). Use as a sanity check on the
   fresh measurement, not as the primary source.

5. **Compute the gap and identify the bottleneck**:
   - efficiency = measured / theoretical (throughput) and
     theoretical / measured (latency) — values < 1.0 always indicate
     measured is worse than the roofline (kernel / framework /
     scheduling headroom).
   - Per-op bottleneck from simulation: name the op (NOT
     "decode" / "prefill" — continuous batching mixes them), state its
     bound (COMPUTE/MEMORY/BALANCED), and explain why the gap looks the
     way it does (e.g. "FFN is memory-bound theoretically at 50% of
     step; measured down_proj likely under-utilises HBM bandwidth on
     this framework version").

**Theoretical-only fallback** (no running server):
- Skip step 2.
- Still run step 3 (`simulate_serving`) and step 4
  (`lookup_measurements`).
- Artifact: `measured: null`, `efficiency: null`.
- Reply: this is a *theoretical roofline*, not actual performance —
  come back with `base_url` once deployed.

**Output artifact** → write to `stages/03_report.yaml`:

```yaml
path: B
theoretical:
  output_throughput_tps: <float>
  request_latency_s: <float>
  ttft_ms: <float>
  tpot_ms: <float>
  cost_per_mtok_output: <float>
measured:                            # or null
  output_throughput_tps: <float>
  request_latency_s: <float>
efficiency:                          # null if no measurement
  output_throughput: <float>
  request_latency: <float>
bottleneck:
  op: <qkv_proj | attn_core | o_proj | up_gate_proj | down_proj | lm_head>
  pct_of_step: <float>
  avg_tokens: <float>
  bound: <COMPUTE | MEMORY | BALANCED>
gap_explanation: "<source: <record_ts>, basis: ...>"   # or null
```

**Reply to the user**: lead with the **measured** numbers (or
theoretical, if no server was available, clearly labelled), then the
efficiency gap as a percentage of theoretical, then the bottleneck op
and a one-line interpretation of where the gap comes from. One-line
lead-in to Stage 4 (parked) — that's where closing the gap would
happen.

## Stage 4 — Performance optimization (FUTURE WORK)

This stage is **parked** and only relevant on **Path B**. The intended
scope: given the Stage 3 bottleneck and gap, suggest software-level
optimisations (kernel selection / fused kernels, scheduler tweaks,
prefix caching, speculative decoding, MoE expert-parallel layout,
quantisation-friendly kernels). **The tools to drive it don't exist yet.**

If the user asks for Stage 4: say plainly that it's not yet implemented;
summarise what *would* be done in this stage; offer to re-run Stages 1–3
with different workload knobs or different candidates if they want to
explore the design space further.

## Workflow style

- **Announce each stage** as a markdown header before running it
  (`**Stage 1 — Service requirement → Workload profile**`).
- **Persist each artifact** with `write_file` before moving on. Use
  YAML so they're easy to read and re-load.
- **Don't loop back** automatically. If the user wants to revise an
  earlier stage, do *just that stage* and propagate downstream.
- **Be concise** in the prose replies. The artifacts are the structured
  output; the chat reply is a human-readable summary, not a re-dump.
