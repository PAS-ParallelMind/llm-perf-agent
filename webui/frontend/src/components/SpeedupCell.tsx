import type { EvalEntry } from "../api/client";

export function fmtSpeedup(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  if (x >= 100) return `${x.toFixed(0)}×`;
  if (x >= 10)  return `${x.toFixed(1)}×`;
  if (x >= 0.1) return `${x.toFixed(2)}×`;
  if (x > 0)    return `${x.toFixed(3)}×`;
  return `${x.toFixed(2)}×`;
}

export function geomean(xs: number[]): number | null {
  const ys = xs.filter((x) => Number.isFinite(x) && x > 0);
  if (ys.length === 0) return null;
  return Math.exp(ys.reduce((a, b) => a + Math.log(b), 0) / ys.length);
}

export default function SpeedupCell({ entry }: { entry: EvalEntry | undefined }) {
  const sp = entry?.speedup;
  if (!sp) return <span className="text-slate-400">—</span>;
  const e2e  = fmtSpeedup(sp.speedup_e2e);
  const kern = fmtSpeedup(sp.speedup_kernel);
  const isDim = (s: string) => s === "—";
  return (
    <span className="font-mono text-xs">
      <span className={isDim(e2e) ? "text-slate-400" : ""}>{e2e}</span>
      <span className="text-slate-400"> / </span>
      <span className={isDim(kern) ? "text-slate-400" : ""}>{kern}</span>
    </span>
  );
}
