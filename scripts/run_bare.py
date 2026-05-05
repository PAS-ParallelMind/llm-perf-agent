#!/usr/bin/env python3
"""Bare-model baseline runner — same I/O contract as agent.batch.

Reads `cfg.io.input` (problems.json) and writes `cfg.io.output` in the
same shape `agent.batch` produces, so downstream evaluators can consume
either interchangeably. The agent loop is bypassed entirely: each
problem is a single chat completion call with no tools.

Fields on each output entry:
  id, code, submitted, steps, elapsed_s, error, metadata?

  - submitted: True iff a non-empty code block was extracted.
  - steps:     always 1 (single-shot).
  - elapsed_s: wall-clock time of the chat completion call.

run.yaml fields used:
  model.*           — name, base_url, api_key, temperature, max_tokens
  agent.workers     — concurrent calls
  io.input/output   — same as batch mode
  system_prompt[_file]?

Ignored: agent.max_steps / agent.time_budget / io.workspace_root
(no per-task workspace is created — bare mode has nothing to write).

Usage:
  python scripts/run_bare.py --config /path/to/run.yaml
                             [--limit N] [--skip-existing]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.config import RunConfig
from agent.engine import Engine

_FENCED_RE = re.compile(r"```(?:cuda|cu|cpp|c\+\+|c)?\s*\n?(.*?)```", re.S | re.I)


def extract_code(text: str) -> str:
    """Return the LAST fenced code block, falling back to raw text."""
    if not text:
        return ""
    blocks = _FENCED_RE.findall(text)
    return (blocks[-1] if blocks else text).strip()


def _load_problems(path: str) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: top level must be a list")
    seen: set[str] = set()
    for i, p in enumerate(raw):
        for k in ("id", "prompt"):
            if k not in p:
                raise ValueError(f"{path}[{i}]: missing key {k!r}")
        if p["id"] in seen:
            raise ValueError(f"{path}[{i}]: duplicate id {p['id']!r}")
        seen.add(p["id"])
    return raw


def _export(results: list[dict], output_path: str) -> None:
    Path(output_path).write_text(json.dumps(results, indent=2))
    p = Path(output_path)
    code_path = p.parent / (p.stem + ".code.json")
    code_only = [{"id": r["id"], "code": r["code"]}
                 for r in results if r["submitted"] and r["code"]]
    code_path.write_text(json.dumps(code_only, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Bare-model baseline runner")
    ap.add_argument("--config", required=True,
                    help="run.yaml — see RunConfig in agent/config.py")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N problems")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip ids already submitted in --output")
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    problems = _load_problems(cfg.io.input)
    print(f"{len(problems)} problems loaded from {cfg.io.input}")

    # Bare mode sends only `prompt` — anything in seed_files is ignored,
    # since there's no workspace and no tool calls. Warn so the user
    # doesn't silently lose context the agent run would have seen.
    with_seeds = [p["id"] for p in problems if p.get("seed_files")]
    if with_seeds:
        head = ", ".join(with_seeds[:5]) + ("..." if len(with_seeds) > 5 else "")
        print(
            f"WARNING: {len(with_seeds)}/{len(problems)} problems have "
            f"seed_files; in bare mode the model sees only `prompt` and "
            f"these files are NOT delivered. Affected: {head}",
            file=sys.stderr,
        )

    out_path = Path(cfg.io.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.skip_existing and out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            done_ids = {e["id"] for e in existing if e.get("submitted")}
        except Exception:
            pass

    todo = [p for p in problems if p["id"] not in done_ids]
    if args.limit:
        todo = todo[:args.limit]

    engine = Engine(
        model=cfg.model.name,
        base_url=cfg.model.base_url,
        api_key=cfg.model.api_key,
        temperature=cfg.model.temperature,
        max_tokens=cfg.model.max_tokens,
    )

    system_prompt: str | None = cfg.system_prompt
    if system_prompt is None and cfg.system_prompt_file:
        system_prompt = Path(cfg.system_prompt_file).read_text()

    results: list[dict] = []
    lock = threading.Lock()

    def _run_one(p: dict) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": p["prompt"]})

        t0 = time.monotonic()
        try:
            msg = engine.chat(messages)
            text = msg.content or ""
            error = None
        except Exception as e:
            text = ""
            error = f"{type(e).__name__}: {e}"
        elapsed = time.monotonic() - t0

        code = extract_code(text)
        entry: dict = {
            "id":         p["id"],
            "code":       code,
            "submitted":  bool(code),
            "steps":      1,
            "elapsed_s":  round(elapsed, 2),
            "error":      error,
        }
        if p.get("metadata"):
            entry["metadata"] = p["metadata"]
        return entry

    def _wrap(idx: int, p: dict) -> dict:
        entry = _run_one(p)
        with lock:
            results.append(entry)
            _export(results, str(out_path))
        print(f"[{idx+1}/{len(todo)}] {p['id']} "
              f"submitted={entry['submitted']} "
              f"elapsed={entry['elapsed_s']}s"
              + (f" err={entry['error']!r}" if entry['error'] else ""))
        return entry

    if cfg.agent.workers <= 1:
        for i, p in enumerate(todo):
            _wrap(i, p)
    else:
        with ThreadPoolExecutor(max_workers=cfg.agent.workers) as pool:
            futures = {pool.submit(_wrap, i, p): p["id"]
                       for i, p in enumerate(todo)}
            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"{pid} failed: {e}")

    _export(results, str(out_path))
    print(f"wrote {len(results)} entries to {out_path}")


if __name__ == "__main__":
    main()
