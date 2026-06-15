"""
Smoke test: load the model and run all 4 inference methods on a handful of
diverse sequences from the TSV, printing predictions next to ground truth.
Confirms the checkpoint loads, decoding uses slot_token_map, and GT alignment
looks sane before running the full benchmark.
"""
import os
import gzip
import csv
import sys

import common

TSV = os.environ.get("TAXOFORMER_TSV", "uniprotkb_human_len1-300.tsv.gz")


def load_examples(n_per_stratum=2):
    """Grab a few rows spanning different strata (virus / eukaryote / bacteria)."""
    picked = {}
    with gzip.open(TSV, "rt") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            gt = common.parse_gt_lineage(row["Taxonomic lineage"])
            seq = row["Sequence"].strip()
            if not gt or not seq or len(seq) < 20:
                continue
            strat = common.stratum_of(gt)
            picked.setdefault(strat, [])
            if len(picked[strat]) < n_per_stratum:
                picked[strat].append((row["Entry"], row["Organism"], seq, gt))
            if sum(len(v) for v in picked.values()) >= n_per_stratum * 4:
                if all(len(v) >= n_per_stratum for v in picked.values()) and len(picked) >= 3:
                    break
    return [item for v in picked.values() for item in v]


def main():
    print("Loading model (ESM-650M + decoder)...", flush=True)
    common.init_model()
    print("Model loaded.\n", flush=True)

    examples = load_examples()
    print(f"Running {len(examples)} examples through all 4 methods.\n")

    for entry, organism, seq, gt in examples:
        print("=" * 100)
        print(f"{entry}  |  {organism}  |  len={len(seq)}  |  stratum={common.stratum_of(gt)}")
        print(f"GT   ({len(gt)}): {' > '.join(gt)}")
        res = common.run_methods(seq)
        for m in ["greedy", "min_edit", "beam", "leaf_reconstruct"]:
            names = res[m]["names"]
            sc = common.score(gt, names)
            print(f"{m:17s}({len(names):2d}) prefix={sc['prefix_len']} "
                  f"recall={sc['name_recall']:.2f} valid={sc['validity']:.2f} "
                  f"{res[m]['secs']:.2f}s: {' > '.join(names)}")
        print()

    # vocab coverage sanity: how many GT taxa exist in the model vocab?
    tok2id = common._engine._tokenizer.token2id
    vocab = {common.norm(k) for k in tok2id}
    all_gt = [common.norm(x) for _, _, _, gt in examples for x in gt]
    cov = sum(1 for x in all_gt if x in vocab) / (len(all_gt) or 1)
    print(f"GT-token vocab coverage on these examples: {cov:.1%} "
          f"({sum(1 for x in all_gt if x in vocab)}/{len(all_gt)})")


if __name__ == "__main__":
    main()
