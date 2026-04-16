"""Bare-model baseline generator for ParEval (omp subset).

Hits an OpenAI-compatible vLLM endpoint with a one-shot completion per
problem, producing a JSON in ParEval schema so drivers/run-all.py can
score it against the same rubric used for the agent run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


SYSTEM_TEMPLATE = (
    "You are a helpful coding assistant.\n"
    "You are helping a programmer write a C++ function. "
    "Write the body of the function and put it in a markdown code block.\n"
    "Do not write any other code or explanations.\n"
)

PROMPT_TEMPLATE = (
    "Complete the C++ function {function_name}. "
    "Only write the body of the function {function_name}.\n\n"
    "```cpp\n{prompt}\n```\n"
)

GPU_FN_RE = re.compile(r"__global__ void ([a-zA-Z0-9_]+)\(")
CPU_FN_RE = re.compile(r"\s*[a-zA-Z_][a-zA-Z0-9_:<>,\s\*&]*\s([a-zA-Z0-9_]+)\(")


def get_function_name(prompt: str, model: str) -> str:
    last = prompt.rstrip().splitlines()[-1]
    rx = GPU_FN_RE if model in ("cuda", "hip") else CPU_FN_RE
    m = rx.match(last)
    if not m:
        raise ValueError(f"no function name in: {last!r}")
    return m.group(1)


_CODE_BLOCK = re.compile(r"```(?:cpp|c\+\+)?\s*\n?(.*?)```", re.S)


def postprocess(original_prompt: str, text: str) -> str:
    text = text or ""
    m = _CODE_BLOCK.search(text)
    body = m.group(1) if m else text
    body = body.strip()
    if body.startswith(original_prompt.strip()):
        body = body[len(original_prompt.strip()):].lstrip()
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-url", default="http://140.112.90.38:8001/v1")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--parallelism", default="omp")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--num-samples-per-prompt", type=int, default=1)
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()

    prompts = json.loads(Path(args.prompts).read_text())
    subset = [p for p in prompts if p.get("parallelism_model") == args.parallelism]
    print(f"[gen] {len(subset)} {args.parallelism} prompts")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    t0 = time.monotonic()
    for p in tqdm(subset, desc="bare"):
        fn = get_function_name(p["prompt"], p["parallelism_model"])
        user = PROMPT_TEMPLATE.format(function_name=fn, prompt=p["prompt"])
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_TEMPLATE},
                    {"role": "user", "content": user},
                ],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                n=args.num_samples_per_prompt,
                stream=False,
            )
            raws = [c.message.content or "" for c in resp.choices]
        except Exception as e:
            print(f"[err] {p['name']}: {e}", file=sys.stderr)
            raws = [""] * args.num_samples_per_prompt

        outs = [postprocess(p["prompt"], r) for r in raws]
        entry = dict(p)
        entry["temperature"] = args.temperature
        entry["top_p"] = args.top_p
        entry["do_sample"] = args.temperature > 0.0
        entry["max_new_tokens"] = args.max_new_tokens
        entry["prompted"] = True
        entry["outputs"] = outs
        entry["raw_outputs"] = raws
        results.append(entry)

        out_path.write_text(json.dumps(results, indent=2))

    elapsed = time.monotonic() - t0
    print(f"[gen] wrote {len(results)} entries to {out_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
