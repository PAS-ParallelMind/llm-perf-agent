# Evaluate.py Reliability Changes

This update improves the integrated evaluator in `eval/evaluate.py`, with a
focus on validation reliability and speedup measurement stability.

## Validation Changes

- Added `--repeat-validation N`.
- Each generated validation input still runs the reference implementation once.
- The candidate implementation can now be run multiple times on the same input.
- Every candidate repeat must pass validation.
- If repeated candidate outputs differ for the same input, the case is marked as
  `FAIL_NONDETERMINISTIC`.
- Validation summaries now include:
  - `fail_nondeterministic`
  - `missing_checker`

## LLM Judge Changes

- Added `--allow-llm-judge`.
- LLM judge is now disabled by default for official correctness decisions.
- `--no-llm` is still supported as a compatibility flag.
- For non-byte-deterministic problems without a checker:
  - Byte mismatches now produce `MISSING_CHECKER` by default.
  - LLM fallback is only used when `--allow-llm-judge` is explicitly set.

This makes missing checkers visible instead of allowing the LLM judge to
silently decide correctness for cases where byte comparison is insufficient.

## Speedup Changes

- Added `--speedup-inputs N`.
- Speedup measurement can now run across multiple generated inputs.
- Each speedup input records:
  - `ref_wall_ms`
  - `ref_compute_ms`
  - `cand_wall_ms`
  - `cand_kernel_ms`
  - `speedup_e2e`
  - `speedup_kernel`
- The top-level speedup fields are kept for backward compatibility.
- Top-level speedup values now summarize the per-input measurements using the
  median.
- Per-input speedup results are stored in `speedup["cases"]`.

## Worker and GPU Scheduling Changes

- Validation and speedup are now separated into two phases when speedup is
  enabled.
- Validation can still use `--workers` for parallel execution.
- Speedup measurement always runs serially after validation.
- This avoids multiple workers competing for the GPU during timing, which makes
  speedup numbers more stable.

## pass@k Metric Changes

- Added `--pass-at-k`, defaulting to `1,5,10`.
- pass@k is computed after validation when validation is enabled.
- Multiple entries with the same problem `id` are treated as multiple samples
  for that problem.
- A sample is counted as correct only when its validation summary fully passes.
- The metric uses the unbiased estimator:
  `1 - comb(n - c, k) / comb(n, k)`.
- Problems with fewer than `k` samples are excluded from that specific
  aggregate.
- The original evaluation output remains a list, and aggregate pass@k metrics
  are written to a sidecar file named `<out_stem>.summary.json`.

## Compatibility Notes

- `--repeat-validation` defaults to `1`, so existing validation behavior remains
  fast by default.
- `--speedup-inputs` defaults to `1`, preserving the old single-input speedup
  behavior unless explicitly changed.
- `--pass-at-k` defaults to `1,5,10`.
- The main `--out` JSON shape is unchanged; pass@k is stored in a sidecar
  summary file.
- Existing top-level speedup JSON keys are still present.
- The legacy scripts `eval/validate.py` and `eval/measure_speedup.py` were not
  changed.
