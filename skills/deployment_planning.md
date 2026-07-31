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

## HARD RULES (apply to every stage; do NOT skip)

- **Provide tool-call narration.** Before each tool invocation, briefly explain what you are about to do. After the tool returns, summarize the key findings, including the most important metrics and their units, followed by a concise interpretation.

  Examples:

  * Before: "Checking memory requirements for the selected deployment."
  * After: "Estimated memory requirement is 67.2 GB. The model fits on an 80 GB GPU with moderate headroom."

  Do not silently execute tool calls, chain multiple tools without explanation, or present raw results without interpretation. Detailed tables should only be shown when requested.
- **Persist artifacts.** Each stage writes its YAML artifact to
   `stages/0N_*.yaml` via `write_file` before declaring the stage done.
- **One-stage corrections.** If the user revises an earlier stage,
   do just that stage and propagate downstream — don't loop back
   silently.
- **Plain terms in replies.** Never use the words "baseline" or
   "theoretical" in user-facing text — they're internal mode names
   and confuse the reader. Say "what vLLM delivers today" (or the
   actual serving framework the user named) for the realistic
   projection, and "the hardware's analytic ceiling" for the peak.
   Never echo internal labels like "Table A", "chosen_path", or
   `latency_source` to the user either.
- **Present result, then proceed.** Each stage must end with a
   clear summary of what it produced — the workload profile for
   Stage 1, the two recommendations + headroom for Stage 2, the
   actual performance + verification + headroom for Stage 3 — before
   you announce the next stage or fire any of its tool calls. Do
   NOT chain straight from Stage N's last tool call into Stage N+1's
   plan and tools in the same turn without a result section in
   between. The user should be able to read each stage's output as
   a self-contained answer before the next stage starts.

## Mental model — what the workflow is doing

- **Stage 1 — Understand the workload.** Turn the user's service
  description into concrete knobs (request rate, in/out length, SLO).
- **Stage 2 — Evaluate every deployment option and pick the
  platform.** For each viable (gpu, tp, dp), predict performance
  under what vLLM delivers today AND under the analytic ceiling.
  Recommend the best-performance and best-cost configs (both from
  the vLLM projection), and show per-platform headroom so the user
  picks with the long-horizon potential in view.
- **Stage 3 — Report the actual performance of the chosen platform.**
  The primary deliverable is what the deployed server actually does:
  capacity (max sustained request rate, max concurrency at SLO) and
  per-request latencies (TTFT, TPOT, E2E). Two supporting analyses
  sit alongside: **verification** (does the measurement track the
  Stage-2 vLLM projection? — gives the user confidence the prediction
  is trustworthy) and **headroom** (how much room is left between
  measured performance and the hardware's analytic ceiling? — sets up
  Stage 4).
- **Stage 4 — optimize the software implementation.** Given the
  Stage 3 headroom and the bottleneck op, suggest software-level
  changes (kernel choice, scheduler tuning, prefix caching,
  speculative decoding, etc.) to close the gap. **Currently parked**
  — the tools to drive it don't exist yet; the agent can list what
  *would* be done but can't execute.

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
> 1. **Understand the workload** — translate your service description
>    into concrete numbers (request rate, input/output length, SLO).
> 2. **Evaluate deployment solutions** — for each GPU
>    you have, simulate every viable (tp, dp) split twice: once for
>    **what vLLM will actually deliver today**, and once for **the
>    hardware's analytic ceiling**. I'll recommend the best-performance
>    and best-cost configs from the vLLM numbers, and show per-platform
>    headroom so you can see which platforms have the most room to
>    grow as software matures. You pick one to deploy.
> 3. **Report the actual performance of the chosen platform** — run
>    benchmarks on the deployed server and tell you what it actually
>    delivers: max sustained request rate, max concurrency at SLO,
>    and per-request TTFT / TPOT / E2E. Alongside the headline, I'll
>    also (a) verify the measurement against the Stage-2 vLLM
>    projection (does the prediction hold up?) and (b) show the
>    headroom to the hardware's analytic ceiling (how much room for
>    optimisation).
> 4. *(future work)* **optimize the software implementation** —
>    suggest kernel / scheduler / serving-stack changes to close the
>    Stage-3 headroom. Currently parked; I can describe what would
>    be done but can't run it.
>
> Let me know if you'd rather just chat about specific numbers instead.

You do NOT need to wait for the user to acknowledge — emit this in the
same assistant message that starts Stage 1, then continue.

**Checklist before Stage 1**: did your reply contain the four numbered
road-map bullets? If not, you skipped Step 0 — fix it before any
Stage-1 tool call.

## Stage 1 — Understand the workload

**Gate**: before reading anything else in this section, confirm the
Step-0 road-map is in your reply for this turn. If it isn't, stop —
prepend it before any Stage-1 work.

**Input**: the user's natural-language service description.

**Essentials** — these five workload fields are load-bearing decisions
that the user should own; the agent must not silently default them:

- `model` — the preset key being deployed
- `request_rate` — req/s the server will see (or `.inf` for closed-loop)
- `input_len`, `output_len` — the avg input, output context length of the requests
- `target_request_latency_s` — the SLO. Stage 2 evalutes whether a hardware configuration candidate meet the requirement.

**Process**:

1. **Check the essentials.** For each missing essential, decide
   between two responses:
   - **Ask** when the user's description doesn't constrain it enough
     to make a defensible assumption (e.g. `model` never stated, or
     SLO unspecified with no archetype hint).
   - **Assume + emphasize** when the archetype gives a clear default
     (e.g. user said "code completion in IDE" → archetype lookup gives
     `input_len` / `output_len` ranges + a typical TTFT). Lead with
     the assumption in a **bold "Assumptions" block** and invite
     correction in the same reply. Don't bury it in a footnote.
   - Either way: do not run Stage 1 tool calls with a missing essential
     left blank — proceed only after the user answers or you've
     explicitly stated every assumption.

2. **Derive non-essentials** from the system-prompt rules
   (`max_num_batched_tokens` / `max_concurrent_requests` defaults,
   `range_ratio`). These get an *explicit assumption line* in the
   Assumptions block but don't need to block on the user.
   `num_requests` is auto-derived by the tools — do NOT write it
   into the workload YAML.

3. **When the user supplied every essential explicitly**, do NOT
   invent assumptions. The Assumptions block is for values *you*
   chose because the user didn't state them. If every essential came
   straight from the user and the non-essentials all match the
   system-prompt defaults, **omit the assumptions block entirely**
   from both the artifact and the reply — there's nothing to flag.

**Output artifact** → write to `stages/01_workload.yaml`:

```yaml
model: <PRESET_MODELS key>
request_rate: <float>              # req/s, Poisson arrivals (NOT concurrency)
input_len: <int>
output_len: <int>                  # user-supplied; reasoning budget is already folded in
max_num_batched_tokens: <int>      # vLLM --max-num-batched-tokens (e.g. 8192)
max_concurrent_requests: <int>     # vLLM --max-num-seqs server cap (e.g. 1024)
range_ratio: <float>               # 0.0 = fixed lengths (clean modeled-vs-measured
                                   # comparison); 0.1 = ±10% per-request jitter
target_request_latency_s: <float>  # end-to-end per-request seconds
# OPTIONAL — only present when the agent had to assume one or more
# values. One bullet per assumption: <field>: <value> — <why>.
# Omit the whole `assumptions:` key if everything was user-supplied.
assumptions:
  - "<field>: <value> — <why this default was chosen>"
```

Concurrency is **not** a workload knob in this schema — it's a *result*
of running the workload through the server (the simulator and benchmark
both report observed peak / mean in-flight). The deployer's lever is
the arrival rate.

**Reply to the user** (HARD RULE #5 — present-then-proceed): write
this as the final content of the Stage 1 turn, before any Stage 2
announcement. A tight summary of the workload knobs you captured.
**Only** include an "Assumptions" section if step 3's condition is
false (i.e. you actually had to assume one or more values); if every
essential came from the user, skip it. Close with a one-line
invitation to override. Stage 2 starts in the **next** turn, after
the user either confirms or asks for changes.

## Stage 2 — Evaluate deployment solutions

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
     ("`evaluate_all` can't run baseline mode on `<gpu>` — its
     microbench grid is missing"). The user either picks a different
     GPU, runs a microbench sweep, or proceeds with theoretical-only
     for that one.
   - **Do not run the evaluate_all tool without a confirmed candidate
     list** — proceed only after the user names the GPUs and counts.

   **Do NOT pre-flight a fit check with `estimate_memory` before
   `evaluate_all`.** Reasons:
   - `evaluate_all`'s `_evaluate` step already gates each candidate
     on weights-fit (per-replica VRAM after TP sharding) and surfaces
     KV-bound saturation as a row note. Anything `estimate_memory`
     would tell you, the sweep already tells you per row.
   - `max_concurrent_requests` in the workload YAML is a **server-
     policy cap** (vLLM's `--max-num-seqs`), NOT the steady-state
     in-flight count. Calling `estimate_memory(concurrency=
     max_concurrent_requests)` produces a worst-case VRAM number that
     is almost always pessimistic — actual concurrency is determined
     by `request_rate` and per-request service time (Little's Law),
     and the simulator's admission control derives it dynamically.
     A worst-case fit check at the policy cap will tell you "nothing
     fits" even for workloads where every candidate is comfortable.
   - Pre-flighting any extra step that wasn't in the announced plan
     also violates HARD RULE #1. If the sweep is in the plan, run
     the sweep — don't insert a fit gate first.

2. **vLLM projection sweep — Table A**: call
   ```
   evaluate_all(
       workload_file="stages/01_workload.yaml",
       candidates=[{"gpu": "<key>", "count": <int>}, ...],
       latency_source="baseline",
   )
   ```
   `latency_source="baseline"` is the API string for the microbench-
   calibrated mode — it projects what vLLM actually delivers today.
   Cost per Mtok is the deployment cost (`tp × dp × per-GPU $/h`), so
   heavier configs are correctly penalised — multi-GPU only earns
   its keep when it meets the latency target a smaller config
   cannot.

3. **Analytic ceiling sweep — Table B**: call the same tool with
   `latency_source="theoretical"` (API string). The analytic peak
   ceiling, used here as **context for the headroom story**, not as
   an alternative recommendation. For each candidate, the ratio
   `vllm_latency / ceiling_latency` is how much faster the hardware
   *could* run if its kernels matured.

4. **Pick TWO recommendations from Table A (vLLM)**:

   - **Performance winner** — the config with the *lowest
     `request_latency_s`* that **meets `target_request_latency_s`**.
     Lowest latency = best deploy if latency is the binding concern.
   - **Cost winner** — the config with the *lowest `$/1M tok`* that
     **meets `target_request_latency_s`**. `$/Mtok` is the hardware
     dollar cost per million output tokens actually served, so it
     captures both the GPU class (`per-GPU $/h`) and how productively
     the config uses that hardware. Use lowest `request_latency_s`
     as the tiebreaker when two configs share the same `$/Mtok`.

   Note on ranking:
   - Use `request_latency_s` and `$/1M tok` from the table as-is —
     they're the predicted latency and cost for completed requests,
     which is what the user cares about. Do NOT filter by
     `served (r/s)` or invent a saturation gate; the cost column
     already reflects served throughput.

   If no config in your GPU budget meets the SLO, recommend the
   closest miss and be honest: "no config in your GPU budget meets
   the `<N>s` SLO under vLLM today — closest miss is `<X>` at
   `<Y>s`. Options: lower the rate, bigger GPU budget, or accept
   the slower latency."

   When the same config wins both axes, call that out explicitly
   ("the cheapest config is also the fastest — nothing to trade off").

5. **Present in plain terms** — re-read HARD RULE #4: the words
   "baseline" and "theoretical" must NOT appear anywhere in your
   user-facing reply. Use "vLLM" (or the actual framework name) for
   the realistic projection and "analytic ceiling" for the peak.

   **Show simulation results for the two best configurations only.**
   Don't dump every (gpu, tp, dp) row — the user wants to see the
   numbers behind the recommendations, not the whole sweep. Pick
   the performance winner and the cost winner from the vLLM table
   (per step 4), then surface each pick's full numbers (TTFT, TPOT,
   request latency, $/Mtok) and its analytic-ceiling counterpart so
   the user can see the headroom right next to the recommendation.

   Use this template:

   > **Best performance** — `<gpu_p> tp=X dp=Y>` on `<n_gpus_p>` GPU(s)
   > (~`$<cost_per_mtok_p>/Mtok`):
   >
   > | metric | vLLM today | analytic ceiling |
   > |---|---:|---:|
   > | request latency | `<lat_p>s` | `<lat_p_ceiling>s` |
   > | TTFT (mean) | `<ttft_p>ms` | `<ttft_p_ceiling>ms` |
   > | TPOT (mean) | `<tpot_p>ms` | `<tpot_p_ceiling>ms` |
   > | meets `<N>s` SLO | ✓/✗ | — |
   >
   > Headroom: **`<latency_p / latency_p_ceiling>×`** between vLLM
   > and the ceiling.
   >
   > **Best cost** — `<gpu_c> tp=X dp=Y>` on `<n_gpus_c>` GPU(s)
   > (~`$<cost_per_mtok_c>/Mtok`):
   > _[same table shape as above]_
   >
   > Headroom: **`<latency_c / latency_c_ceiling>×`**.

   Hardware cost in the header is the `$/1M tok` column from the
   evaluate_all table — dollars per million output tokens served on
   owned hardware (MSRP amortised + electricity, see `list_gpus`).
   It captures both GPU-class price and how productively the config
   uses the hardware.
   >
   > Both recommendations are sized against what vLLM delivers
   > today. The headroom column shows how much room each platform
   > has if the kernel ecosystem matures — useful for long-horizon
   > planning, but optimising kernels is hard so we don't size
   > against the ceiling.
   >
   > Which one do you want to take forward — performance, cost, or
   > something else?

   Collapse to one config (and drop the "which one?" question) when
   performance and cost picks are the same. When no config meets
   SLO, replace the cost block with one honest line ("nothing in
   your GPU budget meets the `<N>s` SLO under vLLM today — closest
   miss is `<X>` at `<Y>s`; options: lower the rate, bigger GPU
   budget, or accept the slower latency") and still show the
   performance winner's table.

   The framework name "vLLM" is the default assumption for what's
   running on the server — swap if the user named a different
   framework explicitly (SGLang, TensorRT-LLM, etc.).

**Output artifact** → write to `stages/02_plan.yaml`:

```yaml
candidate_set:                       # what the user told you they have
  - {gpu: <key>, count: <int>}
recommendations:                     # both from Table A (vLLM projection)
  performance:                       # lowest request_latency_s that meets SLO
    gpu: <key>
    parallelism: {tp: <int>, dp: <int>}
    n_gpus: <tp * dp>
    cost_per_mtok: <float>           # $/1M output tokens from the table
    request_latency_s: <float>
    ttft_ms: <float>
    tpot_ms: <float>
    meets_target: <true|false>
  cost:                              # lowest $/Mtok that meets SLO; null if none
    gpu: <key>
    parallelism: {tp: <int>, dp: <int>}
    n_gpus: <tp * dp>
    cost_per_mtok: <float>
    request_latency_s: <float>
    ttft_ms: <float>
    tpot_ms: <float>
headroom:                            # ratio vllm/ceiling per candidate
  - {gpu: <key>, tp: <int>, dp: <int>, ratio: <float>}
  - ...
```

**Reply to the user** (HARD RULE #5 — present-then-proceed): write
this as the final content of the Stage 2 turn, before any Stage 3
announcement. The two tables (or compact summaries), the two
recommendations, the headroom block, and the "which one?" question
(skip if perf == cost). Stage 3's input is whichever recommendation
the user picks; record their pick before moving on. **Wait for the
user's pick — do NOT start Stage 3 tool calls in the same turn.**

## Stage 3 — Report the actual performance of the chosen platform

**Input**: WorkloadProfile + the user's pick from Stage 2's two
recommendations (performance or cost) in `stages/02_plan.yaml`.
Record the pick in the Stage 3 artifact so the trace shows which
config was actually deployed.

**Goal**: the **headline deliverable** is the actual performance of
the deployed configuration — capacity (max sustained request rate,
max concurrency at SLO) plus per-request latencies (TTFT, TPOT, E2E)
measured against the real server. This is what the user takes away
from Stage 3.

Two supporting analyses are presented alongside the headline numbers
(not as separate top-level deliverables):

  (a) **Verification** — measured vs the Stage-2 vLLM projection.
  Tells the user the prediction is trustworthy (or where it drifts).

  (b) **Headroom** — measured vs the hardware's analytic ceiling.
  Gives a concrete number for how much Stage 4 could buy and names
  the dominant op so the user knows *where* to optimise.

**Process**:

1. **Confirm the deployed config.** If Stage 2 produced two
   different recommendations and the user hasn't picked one yet, ask
   first: *"Are you deploying the performance pick (`<X>`) or the
   cost pick (`<Y>`)? I'll measure whichever is running."* Then ask
   for the server: *"Do you have an OpenAI-compatible server running
   `<picked config>`? Give me the `base_url` and I'll measure it."*
   The served model id is auto-detected from `/v1/models` —
   `benchmark_serving` reads it directly, so don't ask the user for it.
   - **If no server is running** → tell them they need a deployed
     server, and to come back when one exists. Don't fall back to
     anything.

2. **Max sustained request rate** (open-loop). Drive a rate sweep
   upward until the SLO breaches or the server saturates. The
   benchmark records each step automatically as `mode: open`.
   - Start at the workload's `request_rate` and sweep up in small
     steps (e.g. ×1.25 per step) until either (a) mean TTFT or E2E
     exceeds the user's target, or (b) `served_rate` falls below
     ~95% of the offered rate.
   - The maximum sustained rate is the last step where SLO still held
     and the server tracked the offered rate.
   - `benchmark_serving` auto-derives `num_requests` from the rate,
     so each step gets enough samples for steady state without you
     setting it.

3. **Max concurrency at SLO** (capacity probe under fixed in-flight
   N — not a calibration; that's a future-mode concern). Find where
   per-request latency starts to diverge as N grows. Write a probe
   YAML with the same `model` / `input_len` / `output_len` as the
   deployment, plus `request_rate: .inf`. Run for a small ladder of
   `max_concurrent_requests` (e.g. 4, 8, 16, 32, 64) and stop when
   TPOT exceeds the target. `num_requests` is auto-derived (500 for
   the closed-loop case). The records are
   tagged `mode: closed` automatically — that's just the benchmark's
   schema label for "closed-loop arrivals", not a signal that this
   is calibration.

4. **Verify against the Stage-2 vLLM projection.** Present measured
   TTFT / TPOT / request latency at the workload's `request_rate`
   next to the `projected` block from `stages/02_plan.yaml`. Show
   the delta as percentages. Report numbers; do **not** say "pass" /
   "fail" — the user decides what's acceptable.

   If the projection misses by > 20% on TPOT, surface the most likely
   cause: (a) the microbench grid for this (gpu, tp, dp) is stale
   relative to the framework version the server is running, or (b)
   the deployment is in a regime the grid doesn't cover well (SWA
   prefill, atypical batch shapes). Note "may need a microbench
   refresh" but don't loop back automatically.

5. **Quantify the headroom to the analytic ceiling.** Call
   `simulate_serving(workload_file="stages/01_workload.yaml",
   gpu=<recommended>, tp=<X>, dp=<Y>, latency_source="theoretical")`
   to get the ceiling numbers for the deployed config. Compute the
   headroom ratios:
   - `headroom_tpot = measured_tpot / ceiling_tpot`
   - `headroom_latency = measured_request_latency / ceiling_request_latency`
   - `headroom_throughput = ceiling_throughput / measured_throughput`

   Also surface the **dominant bottleneck op** from the theoretical
   per-op breakdown (e.g. "FFN dominates the step at 65% of theoretical
   time — that's where optimisation effort would pay off most"). The
   pair — headroom number + bottleneck op — is the Stage 4 hand-off:
   it tells the user how much there is to gain and *where* to start.

   If the measurement is already within ~10% of the ceiling on TPOT,
   headroom is effectively zero — say so plainly ("you're already at
   the hardware's ceiling; no software-side optimisation is going to
   move this meaningfully") and skip naming a bottleneck.

**Output artifact** → write to `stages/03_report.yaml`:

```yaml
deployment:
  picked: <performance | cost>         # which Stage-2 recommendation the user took
  gpu: <key>
  parallelism: {tp: <int>, dp: <int>}
  n_gpus: <tp * dp>
capacity:
  max_sustained_rate_rps: <float>      # last open-loop step that met SLO
  max_concurrency_at_slo: <int>        # last closed-loop N that met SLO
measured_at_workload:                  # at workload_file's request_rate
  request_latency_s: <float>
  ttft_ms: <float>
  tpot_ms: <float>
verification:                          # measured vs Stage-2 vLLM projection
  projected_request_latency_s: <float>
  projected_ttft_ms: <float>
  projected_tpot_ms: <float>
  delta_request_latency: <float>       # measured / projected ratio
  delta_ttft: <float>
  delta_tpot: <float>
headroom:                              # measured vs analytic ceiling
  ceiling_request_latency_s: <float>
  ceiling_tpot_ms: <float>
  headroom_tpot: <float>               # measured_tpot / ceiling_tpot (1.0 = at ceiling)
  headroom_latency: <float>
  bottleneck_op: <qkv | attn_decode | attn_prefill | o_proj | ffn_or_moe | comm | null>
  bottleneck_pct_of_step: <float | null>
```

**Reply to the user** (HARD RULE #5 — present-then-proceed): write
this as the final content of the Stage 3 turn. Stage 3 is the last
stage that runs today (Stage 4 is parked), so this is the workflow's
closing summary. Lead with **the actual performance of the deployed
config** — that's the headline. Two short sections:

- **Capacity & latencies (measured)**: max sustained request rate,
  max concurrency at SLO, and TTFT / TPOT / E2E at the workload's
  `request_rate`. This is what the server actually delivers.

Then the two supporting analyses, framed as context not headline:

- **Verification vs the Stage-2 projection**: measured / projected
  deltas as percentages. One-line interpretation: tracking, slightly
  off, or stale-grid.
- **Headroom to the analytic ceiling**: measured vs ceiling (ratio
  or "X% of ceiling"). Name the dominant bottleneck op. One-line
  "Stage 4 would target `<op>`" hand-off, or "already at the
  ceiling" if headroom is < ~10%.

## Stage 4 — optimize the software implementation (FUTURE WORK)

**Parked.** The intended scope: given a Stage 3 bottleneck reading,
suggest software-level optimisations (kernel selection / fused kernels,
scheduler tweaks, prefix caching, speculative decoding, MoE
expert-parallel layout, quantisation-friendly kernels). The tools to
drive it don't exist yet.

If the user asks for Stage 4: say plainly that it's not yet implemented;
summarise what *would* be done in this stage; offer to re-run Stages 1–3
with different workload knobs or different candidates if they want to
explore the design space further.

## Workflow style

- **Announce each stage** as a markdown header before running it
  (`**Stage 1 — Understand the workload**`).
- **Narrate before each tool call** (HARD RULE #1) — one short
  sentence saying what you're about to do, right before the tool
  fires. No upfront stage plan needed.
- **Persist each artifact** with `write_file` before moving on. Use
  YAML so they're easy to read and re-load.
- **Don't loop back** automatically. If the user wants to revise an
  earlier stage, do *just that stage* and propagate downstream.
- **Be concise** in the prose replies. The artifacts are the structured
  output; the chat reply is a human-readable summary, not a re-dump.
