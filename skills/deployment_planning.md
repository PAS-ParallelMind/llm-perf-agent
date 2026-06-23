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
- **Stage 2** runs the workload through each hardware candidate under
  two latency models, forking into two paths:
  - **Baseline performance** (vLLM, SGLang, etc. as-is). The agent
    calls `simulate_serving(..., latency_source="baseline")` — the
    microbench-calibrated realistic projection. No measurement lookup
    needed; the per-GPU microbench grid *is* the calibration. The user
    picks hardware on these projected numbers. Stage 3 runs an
    open-loop capacity probe on the real server and reports
    measured-vs-baseline-projection.
  - **Potential performance (after optimization)**. The agent calls
    `simulate_serving(..., latency_source="theoretical")` — the
    analytic peak ceiling (FLOPs/bytes × theoretical peaks,
    efficiency=1.0 everywhere). The user picks hardware on that
    ceiling, accepting the implementation gap as something they'll
    close. Stage 3 measures the deployment against the ceiling and
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
> 2. **Hardware sweep** — for each GPU you have, enumerate every
>    (tp, dp) split that fits and simulate it twice: once for
>    **what the baseline software stack (vLLM today) will actually
>    deliver**, and once for **the analytic ceiling — what the
>    hardware could reach with optimised kernels**. I'll then give
>    you two recommendations in plain terms:
>    *"If you deploy with vLLM as-is, X is the best choice. But Y has
>    the highest potential if you can optimise the implementation."*
>    You pick which one to take forward.
> 3. **Measure performance** — run benchmarks on the deployed server
>    to map its real capacity (max concurrency, max sustainable
>    request rate). If you picked the optimisation route, also
>    compare measured vs theoretical to diagnose where the gap is.
> 4. *(future work)* Optimisation suggestions for the bottleneck.
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

**Essentials** — these five workload fields are load-bearing decisions
that the user should own; the agent must not silently default them:

- `model` — the preset key being deployed
- `request_rate` — req/s the server will see (or `.inf` for closed-loop)
- `input_len`, `output_len` — workload shape (incl. reasoning budget
  if the model reasons; state the split if so)
- `target_request_latency_s` — the SLO

**Process**:

1. **Check the essentials.** For each missing essential, decide
   between two responses:
   - **Ask** when the user's description doesn't constrain it enough
     to make a defensible assumption (e.g. "model" never stated, or
     SLO unspecified with no archetype hint).
   - **Assume + emphasize** when the archetype gives a clear default
     (e.g. user said "code completion in IDE" → archetype lookup gives
     `input_len` / `output_len` ranges + a typical TTFT). Lead with
     the assumption in a **bold "Assumptions" block** and invite
     correction in the same reply. Don't bury it in a footnote.
   - Either way: do not run Stage 1 tool calls with a missing essential
     left blank — proceed only after the user answers or you've
     explicitly stated every assumption.

2. **Derive non-essentials** from the system-prompt rules (`num_requests`
   sizing, `max_num_batched_tokens` / `max_concurrent_requests`
   defaults, `range_ratio`). These get an *explicit assumption line*
   in the Assumptions block but don't need to block on the user.

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

1. **Candidates** — also an essential the user must own:
   - **Ask the user what GPUs they have available** — both *which*
     (preset name) and *how many* each. Don't infer from context
     ("you mentioned an H100 earlier") and don't silently sweep the
     catalog. The tool enumerates every valid (tp, dp) split within
     that count, so the user doesn't pick parallelism by hand.
   - For each named GPU, validate against the catalog (`list_gpus`
     keys). For any name not in the catalog, tell the user plainly:
     *"I can't evaluate `<name>` — it isn't in my preset catalog. I can
     use the closest match (`<key>`), or you can drop it."* Don't
     silently substitute.
   - Heads up: if a GPU has no microbench grid under
     `agent/tools/modeling/configs/hw_profiles/<gpu>/`, `baseline`
     mode will raise on it. Flag this before running the sweep
     ("`pareto_sweep` can't run baseline mode on `<gpu>` — its
     microbench grid is missing"). The user either picks a different
     GPU, runs a microbench sweep, or proceeds with theoretical-only
     for that one.
   - **Do not run the pareto_sweep tool without a confirmed candidate
     list** — proceed only after the user names the GPUs and counts.

2. **Baseline projection sweep — Table A**: call
   ```
   pareto_sweep(
       workload_file="stages/01_workload.yaml",
       candidates=[{"gpu": "<key>", "count": <int>}, ...],
       latency_source="baseline",
   )
   ```
   The microbench-calibrated realistic projection. Cost per Mtok is
   the deployment cost (`tp × dp × per-GPU $/h`), so heavier configs
   are correctly penalised — multi-GPU only earns its keep when it
   meets the latency target a smaller config cannot.

3. **Theoretical ceiling sweep — Table B**: call the same tool with
   `latency_source="theoretical"`. The analytic peak ceiling. Use it
   to scope the implementation-optimisation headroom: where the
   theoretical pareto frontier shifts vs the baseline frontier tells
   you how much faster the system *could* run if every kernel hit
   peak.

4. **Pick winners on each table.** From Table A (baseline), find the
   cheapest config that meets `target_request_latency_s`. From Table B
   (theoretical), the same. Call these **A★** and **B★**. If neither
   table has a meeter, pick the closest miss in each and say what
   would need to change (lower rate, larger GPU budget, more
   parallelism than available).

5. **Present in plain terms** — do NOT use the internal labels
   "baseline-software path" / "optimized-software path" in your reply
   to the user. Lead with both tables (compact form is fine), then
   the recommendations in this template:

   > **If you deploy with vLLM (the baseline software today), `<A★ gpu
   > tp=X dp=Y>` is the best choice** — meets your `<N>s` SLO at
   > `$<a>/Mtok` on `<n_gpus>` GPU(s).
   >
   > **But `<B★ gpu tp=X dp=Y>` has the highest potential if you can
   > optimise the implementation** — the analytic ceiling is `<M>s` at
   > `$<b>/Mtok`, which would <undercut A★ / be the cheapest meet /
   > be the only meet> if you close the kernel gap.
   >
   > Which way do you want to go — `<A★>` (deploy today on vLLM) or
   > `<B★>` (commit to optimising)?

   Adjust the framing if A★ == B★ ("the same config wins both — vLLM
   already gets close to the ceiling here") or if one table has no
   meeter at all ("nothing meets your SLO under vLLM today; only
   `<B★>` could meet it after optimisation").

   The framework name "vLLM" is the default baseline assumption —
   swap if the user named a different framework explicitly.

**Output artifact** → write **only the chosen plan** to
`stages/02_plan.yaml`:

```yaml
candidate_set:                       # what the user told you they have
  - {gpu: <key>, count: <int>}
chosen_path: baseline | optimized    # baseline = baseline latency_source / measure to verify
                                     # optimized = theoretical latency_source / measure to diagnose gap
recommended:
  gpu: <key>
  parallelism: {tp: <int>, dp: <int>}
  n_gpus: <tp * dp>
  rationale: "<why this point under the chosen path>"
meets_target: <true|false>
projected:                           # from the chosen path's table row
  source: "simulate_serving latency_source=<baseline|theoretical>"
  ttft_ms: <float>
  tpot_ms: <float>
  request_latency_s: <float>
  cost_per_mtok: <float>
```

**Reply to the user**: the two tables (or compact summaries), the two
recommendations, the path question. After they answer, confirm the
chosen plan and tell them "next: Stage 3 measures the deployment."

## Stage 3 — Measure performance

**Input**: WorkloadProfile + DeploymentPlan (with `chosen_path`).

**Goal**: map the deployed system's actual capacity — the maximum
sustained request rate it can serve and the maximum concurrency it
can handle before SLO breaches. Both numbers come from
`benchmark_serving` against the running server. Whether we *also*
compare against the theoretical ceiling depends on `chosen_path`.

**Common preflight** (both paths):

1. **Ask about a running server.** Tell the user the recommended
   config (`gpu`, `tp`, `dp`) and ask: *"Do you have an OpenAI-
   compatible server running this config? Give me the `base_url` and
   the served `model` name and I'll measure it."*
   - **If no** → tell them they need a deployed server, and to come
     back when one exists. Don't fall back to anything.

2. **Max sustained request rate** (open-loop). Drive a rate sweep
   upward until the SLO breaches or the server saturates. The
   benchmark records each step automatically as `mode: open`.
   - Start at the workload's `request_rate` and sweep up in small
     steps (e.g. ×1.25 per step) until either (a) mean TTFT or E2E
     exceeds the user's target, or (b) `served_rate` falls below
     ~95% of the offered rate.
   - The maximum sustained rate is the last step where SLO still held
     and the server tracked the offered rate.
   - For each step, run with `num_requests ≈ 100 × rate` so the queue
     reaches steady state (shorter runs systematically under-report
     TPOT / E2E near saturation).

3. **Max concurrency at SLO** (closed-loop). Probe the server at
   fixed in-flight N to find where per-request latency starts to
   diverge. Write a calibration YAML with the same `model` /
   `input_len` / `output_len` as the deployment, plus
   `request_rate: .inf`, `num_requests: 500`. Run for a small ladder
   of `max_concurrent_requests` (e.g. 4, 8, 16, 32, 64) and stop when
   TPOT exceeds the target. Each run records as `mode: closed`.

### Branch — if the user chose to deploy today on the baseline software

**Goal**: confirm what the deployment *actually* delivers and how its
capacity compares to the Stage 2 baseline projection.

**Process**:

4a. **Compare to projection.** Present measured TTFT / TPOT / request
    latency at the workload's `request_rate` next to the `projected`
    block from `stages/02_plan.yaml`. Show the delta as percentages.
    Report numbers; do **not** say "pass" / "fail" — the user decides
    what's acceptable.

    If the projection misses by > 20% on TPOT, surface the most likely
    cause: (a) the microbench grid for this (gpu, tp, dp) is stale
    relative to the framework version the server is running, or (b)
    the deployment is in a regime the grid doesn't cover well (SWA
    prefill, atypical batch shapes). Note "may need a microbench
    refresh" but don't loop back automatically.

### Branch — if the user chose to commit to software optimisation

**Goal**: confirm the deployment delivers and quantify how much
implementation headroom there is to the theoretical ceiling.

**Process**:

4b. **Theoretical reference.** Call `simulate_serving(workload_file=
    "stages/01_workload.yaml", gpu=<recommended>, tp=<X>, dp=<Y>,
    latency_source="theoretical")`. Use the per-op breakdown to name
    the bottleneck operation.

5b. **Compute the gap and identify the bottleneck**:
    - efficiency = measured / theoretical (throughput) and
      theoretical / measured (latency) — values < 1.0 always indicate
      measured is worse than the ceiling (kernel / framework /
      scheduling headroom).
    - Per-op bottleneck from the theoretical simulation: name the op
      (NOT "decode" / "prefill" — continuous batching mixes them) and
      explain why the gap looks the way it does (e.g. "FFN dominates
      the step at 65%; measured down_proj likely under-utilises HBM
      bandwidth on this framework version").

**Output artifact** → write to `stages/03_report.yaml`:

```yaml
path: baseline | optimized
deployment:
  gpu: <key>
  parallelism: {tp: <int>, dp: <int>}
  n_gpus: <tp * dp>
capacity:                              # both paths report this
  max_sustained_rate_rps: <float>      # last open-loop step that met SLO
  max_concurrency_at_slo: <int>        # last closed-loop N that met SLO
measured_at_workload:                  # at workload_file's request_rate
  request_latency_s: <float>
  ttft_ms: <float>
  tpot_ms: <float>
# Baseline path only:
projected:                             # from stages/02_plan.yaml
  request_latency_s: <float>
  ttft_ms: <float>
  tpot_ms: <float>
delta_vs_projected:                    # measured / projected
  request_latency: <float>
  ttft: <float>
  tpot: <float>
# Optimized path only:
theoretical:
  request_latency_s: <float>
  ttft_ms: <float>
  tpot_ms: <float>
efficiency:
  request_latency: <float>             # theoretical / measured
  throughput: <float>                  # measured / theoretical
bottleneck:
  op: <qkv | attn_decode | attn_prefill | o_proj | ffn_or_moe | comm>
  pct_of_step: <float>
```

**Reply to the user**: lead with **capacity** (max sustained rate, max
concurrency at SLO) — that's the headline. Then branch-specific
detail: if they chose to deploy today, show measured vs Stage-2
projection (delta %); if they chose to optimise, show the efficiency
gap to the theoretical ceiling + the bottleneck op and a one-line
interpretation, then a one-line lead-in to Stage 4 (parked). Same
"plain terms" rule as Stage 2 — don't echo internal labels like
`chosen_path: optimized` to the user; describe the branch in the
language they used when they picked it.

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
- **Plan the stage before executing it.** Immediately after the stage
  header, write a short 2-4 bullet plan listing the tool calls (and
  questions, if any essentials are missing) you intend to make in
  this stage. THEN start running them. Don't dive straight from the
  header into a tool call — the user should see what's coming.
- **Persist each artifact** with `write_file` before moving on. Use
  YAML so they're easy to read and re-load.
- **Don't loop back** automatically. If the user wants to revise an
  earlier stage, do *just that stage* and propagate downstream.
- **Be concise** in the prose replies. The artifacts are the structured
  output; the chat reply is a human-readable summary, not a re-dump.
