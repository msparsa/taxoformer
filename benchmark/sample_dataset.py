"""
Build a diverse, stratified ~5k sample from the human-associated SwissProt TSV.

Maximizes organism diversity (round-robin across distinct organisms) while balancing
coarse strata (Viruses / Bacteria / Archaea / Eukaryota), preferring reviewed entries.
Writes benchmark/sample_5k.tsv with columns:
    entry, organism, stratum, reviewed, gt_lineage (' | '-joined), sequence
"""
import os
import gzip
import csv
import random
import argparse
from collections import defaultdict

import common

# Source UniProt TSV(.gz) with columns including 'Taxonomic lineage' and 'Sequence'.
TSV = os.environ.get("TAXOFORMER_TSV", "uniprotkb_human_len1-300.tsv.gz")
OUT = "sample_5k.tsv"


def build(n_target=5000, per_org_cap=2, seed=42, min_len=20):
    random.seed(seed)
    csv.field_size_limit(10**7)
    # pool[stratum][organism] = list of rows (prefer reviewed first)
    pool = defaultdict(lambda: defaultdict(list))
    n_rows = 0
    with gzip.open(TSV, "rt") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            n_rows += 1
            seq = (row.get("Sequence") or "").strip()
            gt = common.parse_gt_lineage(row.get("Taxonomic lineage", ""))
            if not gt or len(seq) < min_len:
                continue
            strat = common.stratum_of(gt)
            if strat in common.DROP_STRATA:
                continue  # uninformative ground-truth lineage
            org = row.get("Organism", "?")
            bucket = pool[strat][org]
            reviewed = row.get("Reviewed", "") == "reviewed"
            if len(bucket) < per_org_cap:
                bucket.append({
                    "entry": row.get("Entry", ""),
                    "organism": org, "stratum": strat,
                    "reviewed": "reviewed" if reviewed else "unreviewed",
                    "gt_lineage": " | ".join(gt), "sequence": seq,
                    "_rev": reviewed,
                })
                # keep reviewed rows at the front of each per-organism bucket
                bucket.sort(key=lambda r: not r["_rev"])
    print(f"Scanned {n_rows:,} rows. Strata: "
          + ", ".join(f"{s}={sum(len(v) for v in pool[s].values())}" for s in pool))

    # round-robin: cycle strata, within a stratum cycle organisms, take one row each pass
    strata = list(pool.keys())
    # per-stratum list of (organism, [rows]) with shuffled organism order
    stratum_orgs = {}
    for s in strata:
        orgs = list(pool[s].items())
        random.shuffle(orgs)
        stratum_orgs[s] = orgs

    # Per stratum, build a de-duplicated, diversity-ordered queue: all organisms'
    # first row (shuffled), then all second rows, etc. (round-robin over organisms).
    stratum_queue = {}
    for s in strata:
        orgs = stratum_orgs[s]
        q = []
        for depth in range(per_org_cap):
            for _org, rows in orgs:
                if depth < len(rows):
                    q.append(rows[depth])
        stratum_queue[s] = q
    avail = {s: len(stratum_queue[s]) for s in strata}
    print("Available per stratum (deduped): "
          + ", ".join(f"{s}={avail[s]}" for s in strata))

    # Round-robin across strata, taking the next unused row from each, until we hit
    # the target or every stratum is exhausted. Balances as far as supply allows.
    selected = []
    cursor = {s: 0 for s in strata}
    while len(selected) < n_target:
        progressed = False
        for s in strata:
            if cursor[s] < len(stratum_queue[s]):
                selected.append(stratum_queue[s][cursor[s]])
                cursor[s] += 1
                progressed = True
                if len(selected) >= n_target:
                    break
        if not progressed:
            break  # all strata exhausted
    random.shuffle(selected)

    cols = ["entry", "organism", "stratum", "reviewed", "gt_lineage", "sequence"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in selected:
            w.writerow({c: r[c] for c in cols})

    by_strat = defaultdict(int)
    for r in selected:
        by_strat[r["stratum"]] += 1
    n_org = len({r["organism"] for r in selected})
    n_rev = sum(1 for r in selected if r["reviewed"] == "reviewed")
    print(f"Wrote {len(selected)} rows to {OUT}: {n_org} distinct organisms, "
          f"{n_rev} reviewed. Per-stratum: {dict(by_strat)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--per_org_cap", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(n_target=args.n, per_org_cap=args.per_org_cap, seed=args.seed)
