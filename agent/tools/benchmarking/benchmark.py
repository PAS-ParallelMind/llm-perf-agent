"""Measured serving benchmark — wraps ``vllm bench serve``.

Drives a synthetic ``random`` workload through a *running* OpenAI-compatible
inference server and reports MEASURED TTFT / TPOT / ITL / throughput.

This is the measured counterpart to ``simulate_serving`` (which is purely
analytical): the parameters mirror it one-for-one, so the agent can put a
modeled estimate and a real measurement side by side and reason about the
gap. The heavy lifting is delegated to ``vllm bench serve`` in a
subprocess — that keeps vLLM's torch/CUDA import out of the agent process
(only paid when the tool actually runs) and tracks vLLM's stable CLI.

Exposed as the ``benchmark_serving`` tool; also runnable as a CLI.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ..base import tool
from ..modeling.report import ReportBuilder
from ...workspace import get_root

# vLLM bench serve builds the request URL as ``base_url + endpoint`` but also
# hits ``base_url + /v1/models`` and ``base_url + /metrics`` on its own, so
# base_url must be the server *root*. Agents (and chat.yaml) usually carry the
# OpenAI-style ``.../v1`` base — strip a trailing /v1 so both forms work.
_V1_SUFFIXES = ("/v1", "/v1/")


def _normalize_base_url(base_url: str) -> str:
    u = base_url.rstrip()
    for suffix in _V1_SUFFIXES:
        if u.endswith(suffix):
            return u[: -len(suffix)]
    return u.rstrip("/")


def _fetch_served_model_name(base_url: str) -> str:
    """Best-effort: the id the server lists at /v1/models (its first model).
    Used to fill --served-model-name when it differs from the HF `model` the
    benchmark tokenizes with. Returns '' on any failure (caller falls back)."""
    import urllib.request

    url = _normalize_base_url(base_url) + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.load(resp)
    except Exception:
        return ""
    ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
    return ids[0] if ids else ""


def _build_command(
    *,
    base_url: str,
    model: str,
    request_rate: str,
    input_len: int,
    output_len: int,
    num_requests: int,
    max_concurrency: int,
    range_ratio: float,
    endpoint: str,
    served_model_name: str,
    seed: int,
    ignore_eos: bool,
    trust_remote_code: bool,
    result_dir: str,
    result_filename: str,
) -> list[str]:
    backend = "openai-chat" if "chat" in endpoint else "openai"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", backend,
        "--base-url", _normalize_base_url(base_url),
        "--endpoint", endpoint,
        "--model", model,
        "--dataset-name", "random",
        "--num-prompts", str(num_requests),
        "--request-rate", str(request_rate),
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--random-range-ratio", str(range_ratio),
        "--seed", str(seed),
        # Report mean/median + p99 for every latency metric, incl. E2E.
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "99",
        "--save-result",
        "--result-dir", result_dir,
        "--result-filename", result_filename,
    ]
    if max_concurrency > 0:  # 0 = no cap (pure open-loop at request_rate)
        cmd += ["--max-concurrency", str(max_concurrency)]
    if served_model_name:
        cmd += ["--served-model-name", served_model_name]
    if ignore_eos:
        cmd.append("--ignore-eos")
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def _parallelism_str(
    tensor_parallel: int,
    pipeline_parallel: int,
    data_parallel: int,
    expert_parallel: bool,
) -> str:
    """Human-readable parallelism descriptor, e.g. 'TP=2 PP=1 DP=1 EP=off'."""
    return (f"TP={tensor_parallel} PP={pipeline_parallel} "
            f"DP={data_parallel} EP={'on' if expert_parallel else 'off'}")


def _n_gpus(tensor_parallel: int, pipeline_parallel: int, data_parallel: int) -> int:
    """Total GPUs a vLLM deployment spans = TP × PP × DP (EP reuses TP×DP)."""
    return tensor_parallel * pipeline_parallel * data_parallel


def _latency_rows(data: dict) -> list[list[str]]:
    rows: list[list[str]] = []
    for key, label in (
        ("ttft", "TTFT"),
        ("tpot", "TPOT"),
        ("itl", "ITL"),
        ("e2el", "E2EL"),
    ):
        mean = data.get(f"mean_{key}_ms")
        if mean is None:
            continue
        rows.append([
            label,
            f"{mean:.3f}",
            f"{data.get(f'median_{key}_ms', 0.0):.3f}",
            f"{data.get(f'p99_{key}_ms', 0.0):.3f}",
        ])
    return rows


def _render_report(
    *,
    base_url: str,
    endpoint: str,
    request_rate: str,
    max_concurrency: int,
    input_len: int,
    output_len: int,
    num_requests: int,
    range_ratio: float,
    gpu: str,
    served_as: str,
    tensor_parallel: int,
    pipeline_parallel: int,
    data_parallel: int,
    expert_parallel: bool,
    wall_s: float,
    data: dict,
) -> str:
    rb = ReportBuilder(width=76)
    rb.banner("MEASURED SERVING BENCHMARK (vllm bench serve)").line()

    rb.heading("Deployment")
    rb.kv("Server", _normalize_base_url(base_url))
    rb.kv("Model", str(data.get("model_id", "?")))
    rb.kv("Served as", served_as)
    rb.kv("Endpoint", f"{endpoint} ({data.get('backend', '?')})")
    n_gpus = _n_gpus(tensor_parallel, pipeline_parallel, data_parallel)
    rb.kv("GPU", f"{n_gpus} × {gpu or 'unknown'}")
    rb.kv("Parallelism",
          _parallelism_str(tensor_parallel, pipeline_parallel,
                           data_parallel, expert_parallel))
    rb.line()

    rb.heading("Workload")
    rb.kv("Request rate", f"{request_rate} req/s (Poisson arrivals)")
    rb.kv("Max concurrency cap",
          "no cap" if max_concurrency <= 0 else f"{max_concurrency} in-flight")
    observed_peak = int(data.get("max_concurrent_requests") or 0)
    if observed_peak:
        rb.kv("Observed peak in-flight", f"{observed_peak}")
    rb.kv("Input / Output len", f"{input_len} / {output_len} tokens (±{range_ratio:.0%})")
    rb.kv("Requests", f"{num_requests}")
    rb.line()

    rb.heading("Results")
    completed = data.get("completed", 0)
    failed = data.get("failed", 0)
    if not completed:
        rb.line("** WARNING: 0 requests completed — the server likely rejected "
                "or could not be reached. **")
        rb.line("Check base_url / model / endpoint and that the server is up. "
                "Metrics below are meaningless.")
        rb.line()
    rb.kv("Completed", f"{completed}" + (f"  ({failed} failed)" if failed else ""))
    rb.kv("Benchmark duration", f"{data.get('duration', 0.0):.2f} s")
    rb.kv("Input / Output tokens",
          f"{data.get('total_input_tokens', 0):,} / {data.get('total_output_tokens', 0):,}")
    rb.rule()
    rb.kv("Request throughput", f"{data.get('request_throughput', 0.0):.3f} req/s")
    rb.kv("Output throughput", f"{data.get('output_throughput', 0.0):.3f} token/s")
    rb.kv("Total throughput", f"{data.get('total_token_throughput', 0.0):.3f} token/s")
    rb.line()

    rb.heading("Latency (ms)")
    rows = _latency_rows(data)
    if rows:
        rb.table(
            headers=["Metric", "Mean", "Median", "P99"],
            rows=rows,
            col_widths=[8, 14, 14, 14],
        )
    else:
        rb.line("(no latency metrics — did any request complete?)")
    rb.line()
    rb.line(f"[measured in {wall_s:.1f}s of wall clock; "
            "compare against simulate_serving for the modeled estimate]")
    rb.rule("=")
    return rb.build()


def run_benchmark(
    *,
    base_url: str,
    workload_file: str,
    model: str,
    request_rate: str,
    input_len: int,
    output_len: int,
    num_requests: int,
    max_concurrency: int = 0,
    range_ratio: float = 0.0,
    endpoint: str = "/v1/completions",
    gpu: str = "",
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    data_parallel: int = 1,
    expert_parallel: bool = False,
    seed: int = 0,
    ignore_eos: bool = True,
    trust_remote_code: bool = True,
    timeout: int = 1200,
) -> str:
    """Run ``vllm bench serve`` and return a rendered report (or an ERROR string)."""
    workspace = get_root()
    result_filename = f"bench_serving_{time.strftime('%Y%m%d-%H%M%S')}.json"
    # `model` is the canonical HF id (tokenizer + theory key). The server may
    # serve under a different id (e.g. a local path) — auto-detect it so the
    # API request names match, without burdening the caller with a 2nd arg.
    served = _fetch_served_model_name(base_url)
    if served and served != model:
        served_model_name = served
        served_as = f"{served}  (auto-detected; requests use this id)"
    elif served:  # server serves under the same id we passed
        served_model_name = ""
        served_as = model
    else:  # couldn't reach /v1/models — proceed but make it visible
        served_model_name = ""
        served_as = (f"{model}  (WARNING: could not detect served id at "
                     "/v1/models; sending `model` as-is — requests will fail "
                     "if the server serves under a different name)")
    cmd = _build_command(
        base_url=base_url,
        model=model,
        request_rate=request_rate,
        input_len=input_len,
        output_len=output_len,
        num_requests=num_requests,
        max_concurrency=max_concurrency,
        range_ratio=range_ratio,
        endpoint=endpoint,
        served_model_name=served_model_name,
        seed=seed,
        ignore_eos=ignore_eos,
        trust_remote_code=trust_remote_code,
        result_dir=str(workspace),
        result_filename=result_filename,
    )

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workspace),
        )
    except subprocess.TimeoutExpired:
        return (
            f"ERROR: benchmark timed out after {timeout}s. The server may be "
            "unreachable or overloaded; lower num_requests/request_rate or raise "
            "the timeout."
        )
    wall_s = time.monotonic() - start

    result_path = Path(workspace) / result_filename
    if not result_path.exists():
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        return (
            f"ERROR: vllm bench serve produced no result file (exit "
            f"{proc.returncode}). Check that the server at {base_url!r} is "
            f"running and serving model {model!r}.\n--- output tail ---\n{tail}"
        )

    try:
        data = json.loads(result_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return f"ERROR: could not parse benchmark result {result_path.name}: {e}"

    report = _render_report(
        base_url=base_url,
        endpoint=endpoint,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        input_len=input_len,
        output_len=output_len,
        num_requests=num_requests,
        range_ratio=range_ratio,
        gpu=gpu,
        served_as=served_as,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        data_parallel=data_parallel,
        expert_parallel=expert_parallel,
        wall_s=wall_s,
        data=data,
    )
    return f"{report}\n\n[raw result saved to {result_filename}]"


@tool(
    "Benchmark a RUNNING OpenAI-compatible inference server (e.g. vLLM) with "
    "a synthetic random workload and report MEASURED TTFT / TPOT / ITL / "
    "throughput. Wraps `vllm bench serve`. The measured counterpart to "
    "`simulate_serving` — its workload comes from the same YAML so modeled "
    "vs. measured numbers compare directly. Requires a reachable, "
    "already-running server; takes real wall-clock time to run.",
    base_url="Server base URL, e.g. 'http://localhost:8000' (a trailing "
             "'/v1' is stripped automatically).",
    workload_file="Workspace-relative path to a WorkloadProfile YAML "
                  "(e.g. 'stages/01_workload.yaml'). Must contain `model`, "
                  "`request_rate`, `input_len`, `output_len`. `num_requests` "
                  "is auto-derived from `request_rate`. `max_concurrent_requests` "
                  "(optional) maps to `--max-concurrency`. `range_ratio` "
                  "(optional, default 0.0) sets per-request length jitter. "
                  "Pass `request_rate: inf` in the YAML for closed-loop mode.",
    endpoint="API path appended to base_url; '/v1/completions' (default) or "
             "'/v1/chat/completions'.",
    gpu="GPU the server runs on (prefer a PRESET_GPUS name). Used in the "
        "report header so the measured numbers are unambiguously tagged to "
        "hardware. Blank = labelled 'unknown'.",
    tensor_parallel="The server's tensor-parallel size (TP). Descriptive "
                    "metadata — the server is already running with this "
                    "layout; the benchmark does not set it. Default 1.",
    pipeline_parallel="The server's pipeline-parallel size (PP). Metadata. "
                      "Default 1.",
    data_parallel="The server's data-parallel size (DP). Metadata. Default 1. "
                  "Total GPUs = TP×PP×DP.",
    expert_parallel="Whether the server uses expert parallelism (EP) for an "
                    "MoE model. Metadata. Default False.",
    ignore_eos="Force exactly output_len tokens per request (ignore EOS) so "
               "measured lengths match the modeled assumption. Default True.",
)
def benchmark_serving(
    base_url: str,
    workload_file: str,
    endpoint: str = "/v1/completions",
    gpu: str = "",
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    data_parallel: int = 1,
    expert_parallel: bool = False,
    ignore_eos: bool = True,
) -> str:
    try:
        from ...workspace import resolve
        import yaml
        wf = yaml.safe_load(resolve(workload_file).read_text()) or {}
    except Exception as e:
        return f"ERROR: could not load workload_file {workload_file!r}: {type(e).__name__}: {e}"

    model = wf.get("model", "")
    rr = wf.get("request_rate")
    request_rate = str(rr) if rr is not None else ""
    input_len = wf.get("input_len", 0)
    output_len = wf.get("output_len", 0)
    max_concurrency = wf.get("max_concurrent_requests", 0)
    range_ratio = float(wf.get("range_ratio", 0.0))

    if not model:
        return f"ERROR: workload {workload_file!r} is missing `model`."
    if not request_rate:
        return f"ERROR: workload {workload_file!r} is missing `request_rate`."

    # num_requests is hard-coded — see modeling.serving.auto_num_requests.
    from ..modeling.serving import auto_num_requests
    try:
        num_requests = auto_num_requests(float(rr))
    except (TypeError, ValueError):
        num_requests = auto_num_requests(0.0)  # falls through to closed-loop default

    return run_benchmark(
        base_url=base_url,
        workload_file=workload_file,
        model=model,
        request_rate=request_rate,
        input_len=input_len,
        output_len=output_len,
        num_requests=num_requests,
        max_concurrency=max_concurrency,
        range_ratio=range_ratio,
        endpoint=endpoint,
        gpu=gpu,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        data_parallel=data_parallel,
        expert_parallel=expert_parallel,
        ignore_eos=ignore_eos,
    )


if __name__ == "__main__":
    import yaml as _yaml
    from pathlib import Path as _Path
    parser = argparse.ArgumentParser(
        description="Measured serving benchmark (wraps vllm bench serve). "
                    "Workload knobs (model, request_rate, input/output_len, "
                    "max_concurrency) come from a WorkloadProfile YAML. "
                    "num_requests is auto-derived from request_rate."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--workload-file", required=True,
                        help="Path to a WorkloadProfile YAML. Must contain "
                             "model, request_rate, input_len, output_len. "
                             "max_concurrent_requests (optional) maps to "
                             "--max-concurrency. range_ratio (optional) "
                             "controls length jitter. num_requests is "
                             "auto-derived from request_rate.")
    parser.add_argument("--endpoint", type=str, default="/v1/completions")
    parser.add_argument("--gpu", type=str, default="",
                        help="GPU preset name. Labels the measurement in the report header.")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--pipeline-parallel", type=int, default=1)
    parser.add_argument("--data-parallel", type=int, default=1)
    parser.add_argument("--expert-parallel", action="store_true")
    parser.add_argument("--no-ignore-eos", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    wf = _yaml.safe_load(_Path(args.workload_file).read_text()) or {}
    rr = wf.get("request_rate")
    if rr is None:
        raise SystemExit(f"workload {args.workload_file!r} missing request_rate")
    from ..modeling.serving import auto_num_requests
    try:
        cli_num_requests = auto_num_requests(float(rr))
    except (TypeError, ValueError):
        cli_num_requests = auto_num_requests(0.0)

    print(run_benchmark(
        base_url=args.base_url,
        workload_file=args.workload_file,
        model=wf["model"],
        request_rate=str(rr),
        input_len=wf["input_len"],
        output_len=wf["output_len"],
        num_requests=cli_num_requests,
        max_concurrency=wf.get("max_concurrent_requests", 0),
        range_ratio=float(wf.get("range_ratio", 0.0)),
        endpoint=args.endpoint,
        gpu=args.gpu,
        tensor_parallel=args.tensor_parallel,
        pipeline_parallel=args.pipeline_parallel,
        data_parallel=args.data_parallel,
        expert_parallel=args.expert_parallel,
        ignore_eos=not args.no_ignore_eos,
        timeout=args.timeout,
    ))
