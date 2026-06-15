"""
Shared utilities for the TaxoFormer inference-method benchmark.

Parsing/scoring helpers are import-light (used by sample_dataset.py without loading
the model). The model is loaded lazily by init_model(), which wraps the public
`taxoformer.TaxoFormer` API. Point it at a local model dir or a Hub repo via the
TAXOFORMER_MODEL env var (default: "msparsa/taxoformer").
"""
import os
import re
from typing import List

MODEL_REPO = os.environ.get("TAXOFORMER_MODEL", "msparsa/taxoformer")

_model = None          # taxoformer.TaxoFormer
_engine = None         # taxoformer._engine (for is_valid_edge)


def init_model(device=None):
    global _model, _engine
    if _model is not None:
        return
    from taxoformer import TaxoFormer
    from taxoformer import _engine as engine
    _model = TaxoFormer.from_pretrained(MODEL_REPO, device=device)
    _engine = engine


# ----------------------------- GT lineage parsing -----------------------------
_RANK_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")


def parse_gt_lineage(raw: str) -> List[str]:
    """'Viruses (no rank), Riboviria (realm), ...' -> ['Viruses', 'Riboviria', ...]."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    out = []
    for p in (s.strip() for s in raw.split(",")):
        name = _RANK_SUFFIX.sub("", p).strip()
        if name:
            out.append(name)
    return out


def stratum_of(gt_lineage: List[str]) -> str:
    """Superkingdom bucket. 'Unclassified'/'Unknown' mark uninformative GT to drop."""
    if not gt_lineage:
        return "Unknown"
    top = gt_lineage[0].lower()
    if "virus" in top:
        return "Viruses"
    if top.startswith("cellular organisms"):
        if len(gt_lineage) > 1:
            t1 = gt_lineage[1].lower()
            for k in ("bacteria", "archaea", "eukaryota"):
                if t1.startswith(k):
                    return k.capitalize()
        return "cellular (other)"
    for k in ("bacteria", "archaea", "eukaryota"):
        if top.startswith(k):
            return k.capitalize()
    return "Unclassified"


DROP_STRATA = {"Unclassified", "Unknown"}


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


# ----------------------------- method runners ---------------------------------
def run_methods(sequence: str) -> dict:
    """Run all four inference methods on one sequence. Returns {method: {names, secs}}."""
    import time
    out = {}
    t = time.time(); greedy = _model.greedy(sequence)
    out["greedy"] = {"names": greedy, "secs": time.time() - t}
    t = time.time()
    out["leaf_reconstruct"] = {"names": _model._leaf_reconstruct(greedy), "secs": time.time() - t}
    for name, fn in [("min_edit", _model.min_edit), ("beam", _model.beam)]:
        t = time.time()
        try:
            out[name] = {"names": fn(sequence), "secs": time.time() - t}
        except Exception as e:
            out[name] = {"names": [], "secs": time.time() - t, "error": str(e)}
    return out


# ----------------------------- metrics ----------------------------------------
def ordered_prefix_len(gt: List[str], pred: List[str]) -> int:
    n = 0
    for a, b in zip(gt, pred):
        if norm(a) == norm(b):
            n += 1
        else:
            break
    return n


def score(gt: List[str], pred: List[str]) -> dict:
    gtn = {norm(x) for x in gt}
    predn = [norm(x) for x in pred]
    plen = ordered_prefix_len(gt, pred)
    gt_len = len(gt) or 1
    valid_edges = total_edges = 0
    for i in range(1, len(pred)):
        total_edges += 1
        if _engine.is_valid_edge(pred[i - 1], pred[i]):
            valid_edges += 1
    return {
        "lineage_acc": plen / gt_len,
        "lca_distance": 1.0 - plen / gt_len,
        "prefix_len": plen,
        "name_recall": len(gtn & set(predn)) / (len(gtn) or 1),
        "pred_depth": len(pred),
        "gt_depth": len(gt),
        "validity": (valid_edges / total_edges) if total_edges else 1.0,
    }
