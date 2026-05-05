# scripts/

Per-benchmark preprocessing scripts go here. Each script's job is to
walk a benchmark suite's source tree and emit a unified `problems.json`
in the format expected by `python -m agent.batch --config run.yaml`
(see the top-level `README.md` and `SPEC.md` §4 for the schema).

The harness itself is benchmark-agnostic — it doesn't know about
ParEval, HeCBench, or any other suite. Anything suite-specific lives
in this folder.

A script in here typically:

1. Reads the benchmark's native files (e.g.
   `ParEval/prompts/generation-prompts.json`,
   `HeCBench/src/<name>-omp/main.cpp`, etc.).
2. Renders each problem's prompt (whatever template it likes).
3. Inlines any required scaffolding files (headers, references) into
   `seed_files`.
4. Writes a flat list of `{id, prompt, seed_files?, metadata?}`
   entries to `problems.json`.

See `eval/build_problems_json.py` (in this repo) for a 30-line
reference implementation that converts ParallelMind's own
`benchmarks.json` into the harness format.
