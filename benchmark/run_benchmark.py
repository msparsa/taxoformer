"""
Run all 4 inference methods over the stratified sample and score each against the
ground-truth lineage. Emits a per-sequence CSV and a summary table (overall + per
stratum), and recommends the best method.

Usage:
    python run_benchmark.py                 # full sample_5k.tsv
    python run_benchmark.py --limit 50      # quick pipeline check
"""
import csv
import time
import argparse
from collections import defaultdict

import numpy as np

import common

METHODS = ["greedy", "min_edit", "beam", "leaf_reconstruct"]
METRIC_KEYS = ["lineage_acc", "lca_distance", "name_recall", "validity", "pred_depth"]
SAMPLE = "sample_5k.tsv"
OUT_CSV = "results/method_comparison.csv"


def load_sample(path, limit=None):
    csv.field_size_limit(10**7)
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", default=SAMPLE)
    args = ap.parse_args()

    print("Loading model...", flush=True)
    common.init_model()
    rows = load_sample(args.sample, args.limit)
    print(f"Scoring {len(rows)} sequences x {len(METHODS)} methods...", flush=True)

    import os
    os.makedirs("results", exist_ok=True)
    fout = open(OUT_CSV, "w", newline="")
    writer = csv.writer(fout)
    header = ["entry", "organism", "stratum", "gt_depth"]
    for m in METHODS:
        for k in METRIC_KEYS:
            header.append(f"{m}_{k}")
        header.append(f"{m}_secs")
    writer.writerow(header)

    # aggregates[(method, stratum_or_ALL)][metric] -> list
    agg = defaultdict(lambda: defaultdict(list))
    secs = defaultdict(list)
    t0 = time.time()

    for i, r in enumerate(rows):
        seq = r["sequence"].strip()
        gt = r["gt_lineage"].split(" | ")
        strat = r["stratum"]
        res = common.run_methods(seq)
        line = [r["entry"], r["organism"], strat, len(gt)]
        for m in METHODS:
            sc = common.score(gt, res[m]["names"])
            for k in METRIC_KEYS:
                v = sc[k]
                agg[(m, "ALL")][k].append(v)
                agg[(m, strat)][k].append(v)
                line.append(round(v, 4) if isinstance(v, float) else v)
            secs[m].append(res[m]["secs"])
            agg[(m, "ALL")]["secs"].append(res[m]["secs"])
            line.append(round(res[m]["secs"], 4))
        writer.writerow(line)
        if (i + 1) % 25 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(rows)}  ({rate:.2f} seq/s, "
                  f"eta {(len(rows)-i-1)/rate/60:.1f} min)", flush=True)
    fout.close()

    # ---- summary ----
    def fmt(method, group):
        a = agg[(method, group)]
        n = len(a["lineage_acc"])
        if n == 0:
            return None
        return {
            "n": n,
            "lineage_acc": np.mean(a["lineage_acc"]),
            "lca_dist": np.mean(a["lca_distance"]),
            "recall": np.mean(a["name_recall"]),
            "validity": np.mean(a["validity"]),
            "depth": np.mean(a["pred_depth"]),
            "secs": np.mean(a["secs"]) if a["secs"] else float("nan"),
        }

    print("\n" + "=" * 92)
    print(f"OVERALL  (n={len(rows)})")
    print(f"{'method':18s} {'lineage_acc':>11s} {'lca_dist':>9s} {'recall':>7s} "
          f"{'validity':>8s} {'depth':>6s} {'sec/seq':>8s}")
    summary = {}
    for m in METHODS:
        s = fmt(m, "ALL")
        summary[m] = s
        print(f"{m:18s} {s['lineage_acc']:>11.3f} {s['lca_dist']:>9.3f} {s['recall']:>7.3f} "
              f"{s['validity']:>8.3f} {s['depth']:>6.1f} {s['secs']:>8.3f}")

    strata = sorted({r["stratum"] for r in rows})
    for grp in strata:
        ng = sum(1 for r in rows if r["stratum"] == grp)
        print("\n" + "-" * 92)
        print(f"STRATUM: {grp}  (n={ng})")
        print(f"{'method':18s} {'lineage_acc':>11s} {'lca_dist':>9s} {'recall':>7s} {'validity':>8s} {'depth':>6s}")
        for m in METHODS:
            s = fmt(m, grp)
            if s:
                print(f"{m:18s} {s['lineage_acc']:>11.3f} {s['lca_dist']:>9.3f} "
                      f"{s['recall']:>7.3f} {s['validity']:>8.3f} {s['depth']:>6.1f}")

    # ---- recommendation ----
    print("\n" + "=" * 92)
    best_acc = max(METHODS, key=lambda m: summary[m]["lineage_acc"])
    best_recall = max(METHODS, key=lambda m: summary[m]["recall"])
    fully_valid = [m for m in METHODS if summary[m]["validity"] > 0.999]
    print(f"Highest lineage accuracy: {best_acc} ({summary[best_acc]['lineage_acc']:.3f})")
    print(f"Highest name recall     : {best_recall} ({summary[best_recall]['recall']:.3f})")
    print(f"Always-valid trees      : {', '.join(fully_valid) if fully_valid else 'none'}")
    print(f"\nPer-sequence CSV written to {OUT_CSV}")

    try:
        make_figure(summary)
    except Exception as e:
        print(f"(figure skipped: {e})")


def make_figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [("lineage_acc", "Lineage accuracy ↑"),
               ("recall", "Name recall ↑"),
               ("validity", "Tree validity ↑")]
    for ax, (key, title) in zip(axes, metrics):
        vals = [summary[m][key] for m in METHODS]
        ax.bar(METHODS, vals, color=["#888", "#4C9", "#49C", "#C84"])
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    fig.suptitle("TaxoFormer inference-method comparison (human-associated SwissProt)")
    fig.tight_layout()
    fig.savefig("results/comparison.png", dpi=130)
    print("Figure written to results/comparison.png")


if __name__ == "__main__":
    main()
