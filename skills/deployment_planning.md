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

## Stage 3 — Workload + plan → Performance report (baseline + bottleneck)

**Input**: WorkloadProfile + DeploymentPlan.

**Process**:
1. **Theoretical**: call `simulate_serving(model, recommended_gpu,
   concurrency, input_len, output_len, num_requests,
   max_num_batched_tokens)`. The report already includes a one-line
   `Bottleneck: <op> — N% of step (BOUND, avg M tokens/batch)` summary.
2. **Calibrate**: call `lookup_measurements(model, recommended_gpu)`. If
   one or more measured records exist for this operating point, derive a
   rough implementation-efficiency factor (measured ÷ theoretical) and
   report **both** numbers, clearly labelled. If no record exists, say
   plainly that the estimate is purely theoretical.
3. **Bottleneck**: the per-op bottleneck the sim surfaced — name the op
   (NOT "decode" / "prefill"; continuous batching mixes them), its
   percentage of step time, its compute/memory bound, and the average
   tokens reaching it.

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

**Reply to the user**: headline numbers (theoretical AND measured if
known), the bottleneck op + bound, the efficiency gap if calibration
exists. Keep it tight — they read the artifact for detail.

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
