
#!/usr/bin/env python3
"""
Filter subsets where `single_pass` outperforms `just_query`, then recompute weighted
"global" metrics across those subsets for *multiple* algorithms.

Inputs
------
- Two required runs for selection:
    --single_pass path/to/single_pass_output.txt
    --just_query  path/to/just_query_output.txt
- Any number of additional runs to evaluate on the selected subsets:
    --run name=path/to/other_algo_output.txt
  (you can repeat --run many times; names must be unique)

Selection Logic
---------------
- Choose subsets where single_pass is clearly better than just_query on a `primary_metric`
  with both absolute and relative thresholds.
- Optional: require improvements on a set of metrics (`compare_metrics`) and/or require
  all of them to be better.

Recomputation
-------------
- For *all* runs (others + the two used for selection), compute query-count-weighted averages
  of each metric across the selected subsets.
- This is equivalent to your "Weighted-avg over subsets" using #queries as weights, which is
  a practical proxy for a true "merge then evaluate" global metric when only per-subset aggregates
  are available.

Outputs
-------
- Console print:
    * Selected subsets with improvement details
    * Weighted metrics per run (aligned by the complete metric-name set)
- JSON file with full results
- Optional CSV table (--out_csv) with rows=runs, columns=metrics

Usage
-----
python filter_and_recompute.py \
  --single_pass path/to/single_pass.txt \
  --just_query  path/to/just_query.txt  \
  --run restgpt=path/to/restgpt.txt \
  --run toolbench=path/to/toolbench.txt \
  --primary_metric NDCG@10 \
  --abs_thresh 0.02 \
  --rel_thresh 0.10 \
  --compare_metrics NDCG@10 MAP@10 Recall@10 \
  --require_all_metrics_better False \
  --out_json report.json \
  --out_csv  report.csv
"""

import argparse
import csv
import json
import re
from typing import Dict, Any, Tuple, List, Set

PerSubset = Dict[str, Dict[str, Any]]

SUBSET_LINE_RE = re.compile(
    r"""^
    (?P<name>[^\s]+)                           # subset name (no whitespace)
    \s+
    (?P<metrics>\{.*\})                        # dict-like metrics
    \s*
    \(\#queries=(?P<n>\d+)\)                   # (#queries=NN)
    """,
    re.VERBOSE
)

def parse_per_subset_metrics(text: str) -> PerSubset:
    """
    Parse the per-subset section from a run's output text.
    Returns: { subset_name: {"metrics": {metric: float, ...}, "n_queries": int} }
    """
    data: PerSubset = {}
    start = text.find("=== Per-subset metrics ===")
    if start == -1:
        raise ValueError("Could not find '=== Per-subset metrics ===' in text.")
    # Take everything until next "===" section or end
    tail = text[start:]
    # Stop before next major section header if present
    next_idx = len(tail)
    for marker in ["=== Global metrics", "=== Weighted-avg", "=== Weighted-avg over subsets"]:
        idx = tail.find(marker)
        if idx != -1:
            next_idx = min(next_idx, idx)
    block = tail[:next_idx]

    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        m = SUBSET_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        metrics_str = m.group("metrics")
        n_queries = int(m.group("n"))
        # Parse dict allowing single quotes
        try:
            metrics = json.loads(metrics_str.replace("'", '"'))
        except Exception:
            import ast
            metrics = ast.literal_eval(metrics_str)
        metrics = {str(k): float(v) for k, v in metrics.items()}
        data[name] = {"metrics": metrics, "n_queries": n_queries}
    return data

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def choose_subsets(
    single_pass: PerSubset,
    just_query: PerSubset,
    primary_metric: str = "NDCG@10",
    abs_thresh: float = 0.02,
    rel_thresh: float = 0.10,
    compare_metrics: List[str] = None,
    require_all_metrics_better: bool = False,
) -> List[Tuple[str, float, float, float]]:
    if compare_metrics is None:
        compare_metrics = [primary_metric]

    selected: List[Tuple[str, float, float, float]] = []
    for subset, sp_info in single_pass.items():
        if subset not in just_query:
            continue
        jq_info = just_query[subset]
        sp_m = sp_info["metrics"]
        jq_m = jq_info["metrics"]
        if primary_metric not in sp_m or primary_metric not in jq_m:
            continue
        sp_val = sp_m[primary_metric]
        jq_val = jq_m[primary_metric]
        abs_impr = sp_val - jq_val
        rel_impr = abs_impr / (jq_val if jq_val != 0 else 1e-9)

        primary_ok = (abs_impr >= abs_thresh) and (rel_impr >= rel_thresh)

        extra_ok = True
        if compare_metrics:
            checks = []
            for met in compare_metrics:
                if met in sp_m and met in jq_m:
                    checks.append(sp_m[met] > jq_m[met])
            if require_all_metrics_better:
                extra_ok = all(checks) if checks else True
            else:
                extra_ok = any(checks) if checks else True

        if primary_ok and extra_ok:
            selected.append((subset, sp_val, jq_val, rel_impr))
    return selected

def weighted_metrics_over_subsets(run: PerSubset, subsets: List[str]) -> Dict[str, float]:
    metric_names: Set[str] = set()
    for s in subsets:
        if s in run:
            metric_names.update(run[s]["metrics"].keys())

    results: Dict[str, float] = {}
    for met in sorted(metric_names):
        num = 0.0
        den = 0.0
        for s in subsets:
            if s not in run: 
                continue
            m = run[s]["metrics"]
            n = run[s]["n_queries"]
            if met in m:
                num += m[met] * n
                den += n
        if den > 0:
            results[met] = num / den
    return results

def parse_runs_arg(run_args: List[str]) -> Dict[str, str]:
    runs = {}
    for item in run_args or []:
        if "=" not in item:
            raise ValueError(f"--run expects name=path, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"--run expects name=path, got: {item}")
        if name in runs:
            raise ValueError(f"Duplicate run name: {name}")
        runs[name] = path
    return runs

def print_table(weighted_by_run: Dict[str, Dict[str, float]]):
    # Collect all metric names
    all_metrics: Set[str] = set()
    for metrics in weighted_by_run.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)

    # Header
    col_width = 13
    header = ["run".ljust(18)] + [m.ljust(col_width) for m in all_metrics]
    print("\nWeighted metrics over SELECTED subsets (all runs):")
    print("".join(header))
    print("-" * (18 + col_width * len(all_metrics)))
    for run_name, metrics in sorted(weighted_by_run.items()):
        row = [run_name.ljust(18)]
        for m in all_metrics:
            val = metrics.get(m, float("nan"))
            cell = f"{val:.5f}" if val == val else "NaN"
            row.append(cell.ljust(col_width))
        print("".join(row))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_pass", required=True, help="Path to single_pass output text")
    ap.add_argument("--just_query", required=True, help="Path to just_query output text")
    ap.add_argument("--run", action="append", help="Additional runs as name=path (repeatable)")
    ap.add_argument("--primary_metric", default="NDCG@10")
    ap.add_argument("--abs_thresh", type=float, default=0.02)
    ap.add_argument("--rel_thresh", type=float, default=0.10)
    ap.add_argument("--compare_metrics", nargs="*", default=["NDCG@10"])
    ap.add_argument("--require_all_metrics_better", type=lambda x: str(x).lower() == "true", default=False)
    ap.add_argument("--out_json", default="filtered_recomputed_report.json")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    # Load required runs
    with open(args.single_pass, "r", encoding="utf-8") as f:
        sp_text = f.read()
    with open(args.just_query, "r", encoding="utf-8") as f:
        jq_text = f.read()

    sp = parse_per_subset_metrics(sp_text)
    jq = parse_per_subset_metrics(jq_text)

    # Select subsets
    selected = choose_subsets(
        sp, jq,
        primary_metric=args.primary_metric,
        abs_thresh=args.abs_thresh,
        rel_thresh=args.rel_thresh,
        compare_metrics=args.compare_metrics,
        require_all_metrics_better=args.require_all_metrics_better,
    )
    selected_names = [s for (s, _, _, _) in selected]

    print("Selected subsets (where single_pass clearly outperforms just_query):")
    if not selected:
        print("  (None selected under current thresholds.)")
    else:
        for (s, sp_val, jq_val, rel_impr) in selected:
            n = sp[s]["n_queries"] if s in sp else "??"
            print(f"  - {s}: {args.primary_metric} single_pass={sp_val:.5f}, just_query={jq_val:.5f}, "
                  f"rel_impr={rel_impr*100:.2f}% (#queries={n})")

    # Parse additional runs
    other_runs = parse_runs_arg(args.run)
    # Always include the two selection runs in the comparison set
    run_specs = {"single_pass": args.single_pass, "just_query": args.just_query, **other_runs}

    # Load & compute weighted metrics for each run
    weighted_by_run: Dict[str, Dict[str, float]] = {}
    per_subset_by_run: Dict[str, PerSubset] = {
        "single_pass": sp,
        "just_query": jq
    }
    for name, path in other_runs.items():
        text = load_text(path)
        per_subset_by_run[name] = parse_per_subset_metrics(text)

    for name, data in per_subset_by_run.items():
        weighted_by_run[name] = weighted_metrics_over_subsets(data, selected_names)

    print_table(weighted_by_run)

    # JSON report
    report = {
        "config": {
            "primary_metric": args.primary_metric,
            "abs_thresh": args.abs_thresh,
            "rel_thresh": args.rel_thresh,
            "compare_metrics": args.compare_metrics,
            "require_all_metrics_better": args.require_all_metrics_better,
        },
        "selected_subsets": [
            {
                "subset": s,
                "single_pass": float(sp_val),
                "just_query": float(jq_val),
                "relative_improvement": float(rel_impr),
                "n_queries": int(sp[s]["n_queries"]) if s in sp else None,
            }
            for (s, sp_val, jq_val, rel_impr) in selected
        ],
        "weighted_metrics_over_selected": weighted_by_run
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON report to: {args.out_json}")

    # Optional CSV
    if args.out_csv:
        # Collect the complete metric-name set.
        all_metrics: Set[str] = set()
        for metrics in weighted_by_run.values():
            all_metrics.update(metrics.keys())
        all_metrics = sorted(all_metrics)

        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["run"] + all_metrics)
            for run_name in sorted(weighted_by_run.keys()):
                row = [run_name] + [f"{weighted_by_run[run_name].get(m, float('nan')):.6f}" if m in weighted_by_run[run_name] else "" for m in all_metrics]
                writer.writerow(row)
        print(f"Saved CSV table to: {args.out_csv}")

if __name__ == "__main__":
    main()
