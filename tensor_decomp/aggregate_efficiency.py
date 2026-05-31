"""
Aggregate per-sample efficiency JSONL records into FlashVID Table 6 style
per-video averages.

Each JSONL line is a per-sample record written by EfficiencyRecorder. We
average across samples to produce, per backend:

  Vision Encoding (ms)        = mean(vision_encoding_ms)
  Compression (ms)            = mean(stage2_compression_ms)        [DySeg-DPP]
  LLM Forward / Prefill (ms)  = mean(llm_prefill_ms)
  Prefill Total (ms)          = Compression + LLM Forward
  TTFT (ms)                   = Vision Encoding + Prefill Total

Missing keys (e.g. baseline has no compression) render as "-".

Usage:
    python aggregate_efficiency.py PATH [PATH ...] [--out summary.csv]

PATH can be a single .jsonl file, a directory (which will be globbed for
*.jsonl), or a glob pattern. Multi-rank files (foo_rank0.jsonl, foo_rank1.jsonl)
are merged automatically.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _iter_jsonl(paths: Iterable[str]) -> Iterable[Dict]:
    seen = set()
    for raw in paths:
        for path in sorted(glob.glob(raw)) or [raw]:
            p = Path(path)
            if p.is_dir():
                files = sorted(p.rglob("*.jsonl"))
            elif p.is_file():
                files = [p]
            else:
                continue
            for f in files:
                key = str(f.resolve())
                if key in seen:
                    continue
                seen.add(key)
                with f.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except Exception:
                            continue


def _mean(values: List[float]) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _fmt(v: Optional[float], unit: str = "ms") -> str:
    if v is None:
        return "-"
    if unit == "ms":
        return f"{v:.1f}"
    return f"{v:.3f}"


def aggregate(records: Iterable[Dict]) -> Dict[str, Dict[str, Optional[float]]]:
    by_backend: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Dict[str, int] = defaultdict(int)

    for rec in records:
        backend = str(rec.get("backend", "unknown"))
        counts[backend] += 1
        for key in (
            "vision_encoding_ms",
            "stage2_compression_ms",
            "llm_prefill_ms",
            "llm_prune_ms",
            "generate_ms",
            "input_tokens",
            "output_tokens",
        ):
            if key in rec and rec[key] is not None:
                by_backend[backend][key].append(rec[key])

    summary = {}
    for backend, bucket in by_backend.items():
        vision = _mean(bucket.get("vision_encoding_ms", []))
        compression = _mean(bucket.get("stage2_compression_ms", []))
        llm_forward = _mean(bucket.get("llm_prefill_ms", []))
        prune = _mean(bucket.get("llm_prune_ms", []))
        generate = _mean(bucket.get("generate_ms", []))
        in_tok = _mean(bucket.get("input_tokens", []))
        out_tok = _mean(bucket.get("output_tokens", []))

        prefill_total = None
        if compression is not None and llm_forward is not None:
            prefill_total = compression + llm_forward
        elif llm_forward is not None:
            prefill_total = llm_forward

        ttft = None
        if vision is not None and prefill_total is not None:
            ttft = vision + prefill_total

        summary[backend] = {
            "n_samples": counts[backend],
            "vision_encoding_ms": vision,
            "stage2_compression_ms": compression,
            "llm_prefill_ms": llm_forward,
            "llm_prune_ms": prune,
            "prefill_total_ms": prefill_total,
            "ttft_ms": ttft,
            "generate_ms": generate,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }
    return summary


def render_table(summary: Dict[str, Dict[str, Optional[float]]]) -> str:
    # Order backends to match Table 6 layout where possible.
    order = ["baseline", "flashvid", "tensor_decomp"]
    backends = [b for b in order if b in summary] + [b for b in summary if b not in order]

    headers = [
        "Backend",
        "N",
        "Vision (ms)",
        "Compression (ms)",
        "LLM Forward (ms)",
        "Prefill Total (ms)",
        "TTFT (ms)",
        "In Tok",
        "Out Tok",
    ]
    rows = [headers]
    for b in backends:
        s = summary[b]
        rows.append([
            b,
            str(s["n_samples"]),
            _fmt(s["vision_encoding_ms"]),
            _fmt(s["stage2_compression_ms"]),
            _fmt(s["llm_prefill_ms"]),
            _fmt(s["prefill_total_ms"]),
            _fmt(s["ttft_ms"]),
            _fmt(s["input_tokens"], unit="raw") if s["input_tokens"] is not None else "-",
            _fmt(s["output_tokens"], unit="raw") if s["output_tokens"] is not None else "-",
        ])

    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    lines = []
    for idx, row in enumerate(rows):
        line = " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if idx == 0:
            lines.append("-+-".join("-" * w for w in widths))
    return "\n".join(lines)


def write_csv(summary: Dict[str, Dict[str, Optional[float]]], out_path: str) -> None:
    import csv
    fields = [
        "backend",
        "n_samples",
        "vision_encoding_ms",
        "stage2_compression_ms",
        "llm_prefill_ms",
        "llm_prune_ms",
        "prefill_total_ms",
        "ttft_ms",
        "generate_ms",
        "input_tokens",
        "output_tokens",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for backend, s in summary.items():
            row = {"backend": backend}
            row.update({k: ("" if v is None else v) for k, v in s.items()})
            w.writerow(row)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="JSONL file(s), directory, or glob pattern.")
    p.add_argument("--out", default=None, help="Optional CSV output path.")
    args = p.parse_args(argv)

    records = list(_iter_jsonl(args.paths))
    if not records:
        print("No JSONL records found.", file=sys.stderr)
        return 1

    summary = aggregate(records)
    print(render_table(summary))
    print()
    print(f"Total samples: {sum(s['n_samples'] for s in summary.values())}")

    if args.out:
        write_csv(summary, args.out)
        print(f"\nCSV written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())