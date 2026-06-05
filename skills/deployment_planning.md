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
  - **Baseline performance** (vLLM, SGLang, etc. as-is). The agent
    pulls a **closed-loop calibration** record (per-pass kernel
    efficiency at a fixed in-flight N) from the measurement store and
    feeds it to `simulate_serving(..., efficiency_factor=k)`. The
    simulator runs at the deployment's open-loop rate but scales each
    forward pass by `1/k`, so the equilibrium concurrency shifts up
    naturally and the projected TPOT / TTFT / E2E reflect the kernel
    cost. The user picks hardware on those projected numbers. Stage 3
    captures (or refreshes) the closed-loop calibration on the real
    server, then runs an open-loop capacity probe and reports
    measured-vs-projected.
  - **Potential performance (after optimization)**. The user picks hardware on the
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
>    implementation (vLLM/SGLang) using historical measurements. Two
>    paths to pick from: **baseline performance** — pick hardware on the
>    projected-actual numbers (what the baseline implementation will
>    deliver today); **potential performance (after optimization)** —
>    pick hardware on the theoretical numbers (what you could reach if
>    you close the implementation gap). You decide.
> 3. **Measure performance** — the baseline-performance path reports
>    the real numbers against the projection; the potential-performance
>    path measures the running implementation and diagnoses where the
>    gap is.
> 4. *(future work)* Optimisation suggestions for the bottleneck
>    (potential-performance path only).
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
num_requests: <int>                # max(200, ~100 × request_rate) — ~100s of arrivals so the queue reaches steady state
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

3. **Projected-actual sweep — Table B** (agent-driven, simulator-applied):
   - For each candidate that fits and isn't saturated, call
     `lookup_measurements(model=<workload model>, gpu=<candidate>,
     mode='closed')`. **The `mode='closed'` filter is critical**:
     only closed-loop calibration records give apples-to-apples per-
     forward-pass kernel efficiency. Open-loop records' efficiency
     factors are *biased* by the equilibrium-shift effect (slow
     measured → more concurrency → bigger batches → not the same
     operating point theory was computed at) and must not drive
     projection.
   - **Pick the calibration record** by judgement: prefer the most
     recent at a similar `max_concurrency` and similar lengths.
     State which record you used and its `efficiency.tpot`.
   - **Project actual performance via the simulator** — DO NOT scale
     theoretical numbers manually. Call:
     ```
     simulate_serving(
         workload_file="stages/01_workload.yaml",
         gpu=<candidate>,
         efficiency_factor=<calibration.efficiency.tpot>,
     )
     ```
     The simulator applies the efficiency to per-forward-pass wall
     time; the equilibrium concurrency naturally shifts up (slower
     per-pass → longer in-flight → larger batches) and the reported
     TPOT / TTFT / E2E latency are the projected actual numbers at the
     deployment open-loop rate. Read those into Table B's row for this
     candidate.
   - For the `$/1M tok` column, use the projected output throughput
     (`request_rate × output_len` at sub-saturation, or the simulator's
     served-rate × output_len when saturated).
   - If **no closed-loop record exists** for (model, gpu), flag the
     candidate in Table B with "—" and a note that *calibration is
     missing*. Do NOT fall back to open-loop efficiency or to copying
     theoretical numbers — both are biased. Surface this to the user
     so Stage 3 can capture the calibration first.
   - Build Table B with the same columns as Table A and mark its own
     Pareto frontier on projected numbers.

4. **Present both tables side by side** and name two recommendations:
   - **Baseline performance**: cheapest candidate that meets
     `target_request_latency_s` on **projected actual** numbers.
   - **Potential performance (after optimization)**: cheapest candidate that meets
     the target on **theoretical** numbers.
   - If a candidate appears in only one table (no historical data for
     Table B), say so plainly. If neither table has a candidate meeting
     the target, recommend the closest miss in each.

5. **Ask the user which path** they want to take. Do NOT pre-decide —
   the choice depends on whether they want to size against
   **baseline-performance** numbers (what the baseline implementation
   will deliver today) or against **potential-performance** numbers
   (what they could reach by optimising the implementation). Once they
   pick, restate the chosen plan.

   **If the user picks the baseline-performance path on a candidate
   with no historical record**: you cannot compute a real projection —
   there's no efficiency factor to apply. Do NOT copy the theoretical
   numbers into the projection block and pretend it's a projection (the
   resulting "delta" in Stage 3 will be meaningless). Instead, tell the
   user plainly: *"I don't have a historical measurement for this
   (model, gpu) pair, so I can't project the baseline implementation's
   actual performance. Two options: (a) switch to the
   potential-performance path and we measure + diagnose against theory;
   (b) proceed to Stage 3 to capture the first measurement for this
   pair — future runs will use it for projections."* Let the user pick.

6. **Single-GPU only** for now — the modeling tools don't model TP/PP/DP
   scaling. Output `n_gpus: 1` and a parallelism stub.

**Output artifact** → write **only the chosen plan** to
`stages/02_plan.yaml`:

```yaml
candidate_set: [<gpu keys>]
chosen_path: baseline | optimized    # baseline = size on projected-actual / measure & compare
                                     # optimized = size on theoretical / diagnose gap
recommended:
  gpu: <key>
  n_gpus: 1
  parallelism: {tp: 1, pp: 1, dp: 1}
  rationale: "<why this point under the chosen path>"
meets_target: <true|false>
projection:                          # baseline only — null for optimized
  source_record: "<closed-loop measurement id / timestamp>"
  source_max_concurrency: <int>      # the N the calibration was probed at
  efficiency_tpot: <float>           # closed-loop per-pass efficiency applied
  efficiency_ttft: <float>           # informational; mostly mirrors tpot at saturation
  projected_ttft_ms: <float>         # from simulate_serving(..., efficiency_factor=tpot)
  projected_tpot_ms: <float>
  projected_request_latency_s: <float>
```

**Reply to the user**: the two tables (or compact summaries), the two
recommendations, the path question. After they answer, confirm the
chosen plan and tell them "next: Stage 3 measures the baseline."

## Stage 3 — Measure performance

**Input**: WorkloadProfile + DeploymentPlan (with `chosen_path`).

The goal: measure the **actual** performance of the deployed system.
What we do with that measurement depends on the path the user picked in
Stage 2. Branch accordingly.

### Baseline performance

**Goal**: capture (or refresh) the **closed-loop calibration** that
drives Stage 2's projection, then run an **open-loop capacity** probe
at the deployment rate and report measured-vs-projected.

Two distinct measurements happen here:
- **Calibration probe** (closed-loop, `request_rate=.inf`, fixed N).
  Produces the per-pass kernel efficiency factor that
  `simulate_serving(..., efficiency_factor=...)` consumes for
  projection. Apples-to-apples kernel comparison by construction.
- **Capacity probe** (open-loop, the deployment `request_rate`).
  Reports the real SLO compliance and request latency at the
  expected user load.

**Process**:

1. **Ask about a running server.** Tell the user the recommended
   config and ask: *"Do you have an OpenAI-compatible server running
   this config? Give me the `base_url` and the served `model` name and
   I'll measure it."*
   - **If no** → tell them they need a deployed server, and to come
     back when one exists. Don't fall back to anything.

2. **Calibrate** (closed-loop probe). Skip this step *only* if Stage
   2 already used a recent closed-loop record at a similar
   `max_concurrency` and similar lengths; otherwise run it now.
   - Write a calibration yaml (e.g. `stages/01_workload_cal.yaml`)
     with the same `model` / `input_len` / `output_len` as the
     deployment, plus `request_rate: .inf`,
     `max_concurrent_requests: 16` (or 32 for very fast hardware),
     `num_requests: 500`.
   - Call `benchmark_serving(base_url=..., workload_file=<cal yaml>,
     gpu=<recommended>, tensor_parallel=1)`. The record auto-tags as
     `mode: closed` and stores the kernel efficiency.
   - If Stage 2's projection used a *different* efficiency (proxy
     hardware, stale record, no record), **re-run Stage 2's
     projection** with the fresh efficiency before continuing — the
     projection number you compare against should reflect this
     hardware's actual kernels.

3. **Measure capacity** (open-loop, at the deployment rate). Call
   `benchmark_serving(base_url=..., workload_file=
   "stages/01_workload.yaml", gpu=<recommended preset>,
   tensor_parallel=1)`. Override `model` if the server serves under a
   different id than the workload's `model`. Record auto-tags as
   `mode: open`.

4. **Compare**: present measured request latency (with TTFT / TPOT
   breakdown) next to the `projection` block from
   `stages/02_plan.yaml`. Show the delta as percentages. Report
   numbers; do **not** say "pass" / "fail" — the user decides what's
   acceptable.

   Throughput isn't compared here — at sub-saturation it just mirrors
   `request_rate × output_len` on both sides, so the delta is
   uninformative. Latency (and its TTFT / TPOT decomposition) is the
   meaningful signal.

   If the projection misses by a lot, the most likely causes are
   (a) the calibration record came from a different operating point
   or older framework version, or (b) the deployment rate pushes the
   system into a regime the calibration didn't cover. Surface the
   suspected cause; don't just report the delta.

**Output artifact** → write to `stages/03_report.yaml`:

```yaml
path: baseline
calibration:                         # the closed-loop probe used / refreshed
  record_ts: "<measurement timestamp>"
  max_concurrency: <int>
  efficiency_tpot: <float>
  efficiency_ttft: <float>
projected:                           # from simulate_serving + efficiency_factor
  request_latency_s: <float>
  ttft_ms: <float>
  tpot_ms: <float>
measured:                            # from open-loop benchmark_serving
  request_latency_s: <float>
  ttft_ms: <float>
  tpot_ms: <float>
delta:                               # measured / projected
  request_latency: <float>
  ttft: <float>
  tpot: <float>
```

**Reply to the user**: measured vs projected (latency numbers + delta
%, with TTFT / TPOT breakdown), the calibration record used, and one
line on whether the delta is large enough that they may want to
revisit Stage 2 with a different candidate. End the workflow here.

### Potential performance (after optimization)

**Goal**: measure the running implementation's actual performance,
compare to the theoretical upper bound, and identify the per-op
bottleneck — sets up Stage 4 (parked).

The measured numbers come from `benchmark_serving` against the running
server. The theoretical numbers come from `simulate_serving` and serve
two purposes: (a) the reference to compute the implementation-efficiency
gap against, and (b) the per-op breakdown that diagnoses *where* the
gap is coming from (the real benchmark gives end-to-end numbers but no
per-op detail).

**Process**:

1. **Ask about a running server.** Same as the baseline-performance path.
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
path: optimized
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

This stage is **parked** and only relevant on the
**potential-performance** path. The intended
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
