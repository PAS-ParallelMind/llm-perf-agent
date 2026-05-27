---
name: deployment_planning
description: Structured 4-stage workflow for sizing an LLM deployment given a service requirement
when_to_use: User describes a service they want to deploy (purpose, users, latency target) and wants hardware / software guidance
---

# Deployment-planning workflow

You are now in **planning mode**. The user wants a structured deployment
recommendation, not a free-form Q&A. Follow the four stages below in
order. Persist a structured artifact at the end of each stage to
`stages/0N_<name>.yaml` so the run is auditable and re-entrant. Announce
each stage as you enter it (e.g. `**Stage 1 — Service requirement**`).

**Mode override**: in planning mode, **sweeping over candidates is the
point** of Stage 2. The base system-prompt rule "converge, don't sweep"
does NOT apply here.

## Before Stage 1 — give the user a road map

Before launching Stage 1, write a tight overview of the workflow so the
user knows what to expect. Something like:

> You want to deploy a model — here's how I'll work through it:
> 1. **Workload profile** — translate your service description into
>    concrete numbers (concurrency, input/output length, latency target).
> 2. **Hardware sweep** — compare GPU candidates on cost vs. latency
>    and recommend a Pareto-frontier point.
> 3. **Measure baseline performance** — if you have a server running the
>    recommended config, I'll benchmark it against the workload for real
>    numbers; otherwise I'll fall back to a theoretical estimate and you
>    can come back here after deploying.
> 4. *(future work)* Optimisation suggestions for any bottleneck.

Then proceed straight into Stage 1 — don't wait for confirmation unless
the user explicitly pushes back or asks to switch to conversational mode.

## Stage 1 — Service requirement → Workload profile

**Input**: the user's natural-language service description.

**Process**: Translate it to concrete workload knobs using the rules in
the system prompt (archetype table, Little's Law for users → concurrency,
reasoning-budget for output_len, `num_requests` sizing rule). Fill any
missing field with an *explicitly stated assumption*. Do **not** stall to
ask the user — proceed with stated assumptions and let them override.

**Output artifact** → write to `stages/01_workload.yaml`:

```yaml
model: <PRESET_MODELS key>
concurrency: <int>
input_len: <int>
output_len: <int>                  # incl. reasoning budget if model reasons
num_requests: <int>                # 20× concurrency (short out) or 5× (long)
max_num_batched_tokens: <int>
target_request_latency_s: <float>  # end-to-end per-request seconds
assumptions:
  archetype: <chat|RAG|code|summarization|agentic>
  think_time_s: <float>
  peak_active_fraction: <float>
  reasoning_budget_tokens: <int|0 if non-reasoning model>
  notes: "<anything else worth surfacing>"
```

**Reply to the user**: a tight summary — the workload knobs you chose, the
assumptions you made, and a one-line invitation to override.

## Stage 2 — Workload + candidates → Deployment plan

**Input**: WorkloadProfile from Stage 1.

**Process**:
1. **Candidates**: if the user named hardware, use that set. Otherwise
   call `list_gpus` to see options and **ask the user** which to
   evaluate — don't silently sweep all GPUs.
2. **Sweep**: call `pareto_sweep(model, candidates, concurrency,
   input_len, output_len, num_requests, max_num_batched_tokens,
   target_request_latency_s)`. The result has fit checks, $/1M-token
   cost, request latency, and Pareto-frontier markers.
3. **Recommend**: pick one Pareto-frontier point matching the user's
   stated preference. If they didn't state one, default to the **cheapest
   candidate that meets the latency target** and offer the
   highest-throughput alternative as runner-up. If no candidate meets the
   target, say so and recommend the closest miss.
4. **Single-GPU only** for now — the modeling tools don't model TP/PP/DP
   scaling. Output ``n_gpus: 1`` and a parallelism stub.

**Output artifact** → write to `stages/02_plan.yaml`:

```yaml
candidate_set: [<gpu keys>]
sweep_result: |
  <paste the pareto_sweep text or summarise it>
recommended:
  gpu: <key>
  n_gpus: 1
  parallelism: {tp: 1, pp: 1, dp: 1}
  rationale: "<why this point — cheapest meeting latency / fastest / etc.>"
runner_up:
  gpu: <key>
  rationale: "<alternative tradeoff>"
meets_target: <true|false>
```

**Reply to the user**: the recommendation, the runner-up, one line on the
tradeoff, and "next: Stage 3 will check the bottleneck."

## Stage 3 — Measure baseline performance

**Input**: WorkloadProfile + DeploymentPlan.

The goal here is to **evaluate the actual performance** of the deployed
system and quantify **how close it comes to the theoretical roofline**.
The measured numbers come from `benchmark_serving` against the running
server. The theoretical numbers come from `simulate_serving` and serve
two purposes: (a) the reference to compute the implementation-efficiency
gap against, and (b) the per-op breakdown that diagnoses *where* the
gap is coming from (the real benchmark gives end-to-end numbers but no
per-op detail).

**Process**:

1. **Ask about a running server.** Tell the user the recommended config
   from Stage 2 and ask: *"Do you have an OpenAI-compatible server
   running this config? Give me the `base_url` and the served `model`
   name and I'll measure it."*
   - **If yes** → run the measurement path (steps 2–5).
   - **If no** → take the theoretical-only fallback at the end. Tell the
     user that's a roofline reference, not actual performance, and that
     they should come back to Stage 3 after deploying.

2. **Measure** (primary): call `benchmark_serving(base_url=...,
   model=<served name>, concurrency, input_len, output_len, num_requests,
   gpu=<recommended preset>, tensor_parallel=1)` with the workload from
   Stage 1 (single-GPU for now, so `tensor_parallel=1`). The tool drives
   a real probe via `vllm bench serve` against the user's server and
   auto-records the result to the measurement store.

3. **Theoretical reference**: call `simulate_serving(model,
   recommended_gpu, ...)` with the same workload. Use it for the
   roofline comparison and for the per-op `Bottleneck: <op> — N% of step
   (BOUND, avg M tokens/batch)` line.

4. **Historical cross-check**: call `lookup_measurements(model,
   recommended_gpu)` for any prior measurements (other framework
   versions, nearby operating points). Use them as a sanity check on
   the fresh measurement, not as the primary source.

5. **Compute the gap**: implementation-efficiency = measured ÷
   theoretical for throughput, and theoretical ÷ measured for latency
   (so values < 1.0 always indicate measured is worse than the roofline,
   meaning there's kernel / framework / scheduling headroom). The per-op
   bottleneck from step 3 is the *diagnostic lens* — name the op (NOT
   "decode" / "prefill"; continuous batching mixes them), state its
   compute/memory bound, and use it to explain why the gap looks the way
   it does (e.g. "FFN is memory-bound theoretically at 50% of step;
   measured down_proj likely under-utilises HBM bandwidth on this
   framework version").

**Theoretical-only fallback** (no running server available):
- Skip step 2 (`benchmark_serving`).
- Still run step 3 (`simulate_serving`) and step 4 (`lookup_measurements`).
- In the artifact, set `measured: null` and `efficiency: null`.
- In your reply, say plainly that this is a *theoretical roofline*, not
  actual performance, and that the user should re-run Stage 3 with
  `base_url` once they have the recommended config deployed.

**Output artifact** → write to `stages/03_report.yaml`:

```yaml
theoretical:
  output_throughput_tps: <float>
  request_latency_s: <float>
  ttft_ms: <float>
  tpot_ms: <float>
  cost_per_mtok_output: <float>
measured:                            # or null
  output_throughput_tps: <float>
  request_latency_s: <float>
efficiency:                          # null if no calibration data
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
and a one-line interpretation of where the gap comes from. Keep it
tight — they read the artifact for detail.

## Stage 4 — Performance optimization (FUTURE WORK)

This stage is **parked**. The intended scope: given the Stage 3
bottleneck and gap, suggest software-level optimisations
(kernel selection / fused kernels, scheduler tweaks, prefix caching,
speculative decoding, MoE expert-parallel layout, quantisation-friendly
kernels). **The tools to drive it don't exist yet.**

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
