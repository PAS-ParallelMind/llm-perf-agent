import type { EvalEntry } from "../api/client";

export type StatusKind = "PASS" | "FAIL" | "BUILD" | "PARTIAL" | "NONE";

export function classify(entry: EvalEntry | undefined): {
  kind: StatusKind;
  rate: string;
  passingCount: number;
  totalCount: number;
} {
  if (!entry || !entry.submitted || !entry.validation) {
    return { kind: "NONE", rate: "", passingCount: 0, totalCount: 0 };
  }
  const s = entry.validation.summary;
  const total = s.total;
  const passing = s.pass_byte + s.pass_checker + s.pass_llm;
  if (total <= 1) {
    // synthetic single-case rows = build/run abort
    const first = entry.validation.cases[0];
    const isBuild = first && first.status.includes("BUILD");
    return {
      kind: isBuild ? "BUILD" : "FAIL",
      rate: "",
      passingCount: passing,
      totalCount: total,
    };
  }
  const rate = `${passing}/${total}`;
  if (passing === total) return { kind: "PASS", rate, passingCount: passing, totalCount: total };
  if (passing === 0)     return { kind: "FAIL", rate, passingCount: passing, totalCount: total };
  return { kind: "PARTIAL", rate, passingCount: passing, totalCount: total };
}

const CLASSES: Record<StatusKind, string> = {
  PASS:    "bg-pass    text-slate-900",
  FAIL:    "bg-fail    text-slate-900",
  BUILD:   "bg-build   text-slate-900",
  PARTIAL: "bg-partial text-slate-900",
  NONE:    "bg-slate-200 text-slate-500",
};

const LABELS: Record<StatusKind, string> = {
  PASS: "PASS", FAIL: "FAIL", BUILD: "BUILD", PARTIAL: "PART", NONE: "—",
};

export default function StatusPill({
  kind,
  onClick,
}: {
  kind: StatusKind;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      disabled={!onClick}
      onClick={onClick}
      className={`inline-block px-2.5 py-0.5 text-xs font-bold rounded ${CLASSES[kind]} ${
        onClick ? "cursor-pointer hover:opacity-80" : "cursor-default"
      }`}
    >
      {LABELS[kind]}
    </button>
  );
}
