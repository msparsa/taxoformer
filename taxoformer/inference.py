"""
Public inference API for TaxoFormer.

    from taxoformer import TaxoFormer
    model = TaxoFormer.from_pretrained("msparsa/taxoformer")   # or a local dir
    model.predict("MGNKWSK...", method="leaf_reconstruct")

Four decoding methods are exposed (see README for the benchmark that compares them):
  - greedy            : plain autoregressive argmax (fast baseline)
  - min_edit          : greedy, then repair invalid parent->child edges (always a valid tree)
  - beam              : constrained beam search over valid edges (always a valid tree)
  - leaf_reconstruct  : anchor on the model's deepest in-tree node, reconstruct the canonical
                        lineage to the root (always valid; best accuracy in our benchmark)
"""
import os
import math
from typing import List, Optional, Dict, Set

from . import _engine

_REQUIRED = ["config.json", "model.safetensors", "phylo2_mapping.json",
             "parent_to_child_mapping.csv"]


class TaxoFormer:
    def __init__(self):
        self._child_to_parents: Optional[Dict[str, Set[str]]] = None
        self._known_nodes: Optional[Set[str]] = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, repo_or_dir: str, device: Optional[str] = None) -> "TaxoFormer":
        model_dir = repo_or_dir
        if not (os.path.isdir(repo_or_dir) and
                all(os.path.exists(os.path.join(repo_or_dir, f)) for f in _REQUIRED)):
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(repo_id=repo_or_dir, allow_patterns=_REQUIRED)
        _engine.init(model_dir, device=device)
        obj = cls()
        obj._build_tree_index()
        return obj

    # ------------------------------------------------------------------ methods
    def greedy(self, sequence: str) -> List[str]:
        return _engine.predict_taxonomy_for_sequence(sequence)

    def min_edit(self, sequence: str) -> List[str]:
        return _engine.predict_valid_tree_min_edit(sequence).get("names", [])

    def beam(self, sequence: str, topn_bin: int = 5) -> List[str]:
        res = _engine.predict_top_k_valid_trees_ar_validated(
            sequence, topn_bin=topn_bin, k_return=1)
        return res[0]["names"] if res else []

    def leaf_reconstruct(self, sequence: str) -> List[str]:
        return self._leaf_reconstruct(self.greedy(sequence))

    def predict(self, sequence: str, method: str = "leaf_reconstruct",
                return_confidence: bool = False):
        """Predict one lineage. If return_confidence, returns
        {"lineage": [...], "confidence": float in (0,1]}.

        `confidence` is the model's confidence in the BROAD placement (domain -> class);
        higher means surer. It is only weakly predictive of full-lineage correctness and
        can be high even when wrong on out-of-distribution inputs -- see `confidence`."""
        fn = {"greedy": self.greedy, "min_edit": self.min_edit,
              "beam": self.beam, "leaf_reconstruct": self.leaf_reconstruct}.get(method)
        if fn is None:
            raise ValueError(f"unknown method {method!r}; choose from greedy/min_edit/beam/leaf_reconstruct")
        names = fn(sequence)
        if not return_confidence:
            return names
        return {"lineage": names, "confidence": self.confidence(sequence, names)}

    def confidence(self, sequence: str, names: List[str]) -> float:
        """Broad-placement confidence in (0, 1]: geometric-mean per-rank probability over
        the broad taxonomic ranks (domain -> phylum/class). Positively (if weakly)
        correlated with how much of the true lineage is recovered; NOT a guarantee of
        correctness (the model can be confidently wrong on novel/short sequences)."""
        return _engine.score_path_confidence(sequence, names)

    def predict_topk(self, sequence: str, k: int = 5, topn_bin: int = 8) -> List[dict]:
        """Return the top-k *valid* lineages via constrained beam search, each with a
        confidence. Sorted best-first.

        Each item: {"lineage": [...], "logprob": float,
                    "confidence": broad-placement confidence (0..1, see `confidence`),
                    "rel_prob": softmax weight of this tree among the k returned}.
        """
        res = _engine.predict_top_k_valid_trees_ar_validated(
            sequence, topn_bin=topn_bin, k_return=k)
        if not res:
            return []
        lps = [r["logprob"] for r in res]
        mx = max(lps)
        exps = [math.exp(lp - mx) for lp in lps]
        z = sum(exps) or 1.0
        out = []
        for r, e in zip(res, exps):
            out.append({
                "lineage": r["names"],
                "logprob": r["logprob"],
                "confidence": _engine.score_path_confidence(sequence, r["names"]),
                "rel_prob": e / z,
            })
        return out

    # ------------------------------------------------------------------ leaf-reconstruct internals
    def _build_tree_index(self):
        p2c = _engine._parent_to_children or {}
        c2p: Dict[str, Set[str]] = {}
        known: Set[str] = set()
        for parent, children in p2c.items():
            known.add(parent)
            for ch in children:
                known.add(ch)
                c2p.setdefault(ch, set()).add(parent)
        self._child_to_parents = c2p
        self._known_nodes = known

    def _leaf_reconstruct(self, greedy_names: List[str]) -> List[str]:
        if not greedy_names:
            return []
        norm = lambda s: " ".join(str(s).strip().lower().split())
        greedy_norm = {norm(n) for n in greedy_names}
        anchor = next((nm for nm in reversed(greedy_names) if nm in self._known_nodes), None)
        if anchor is None:
            return list(greedy_names)
        path = [anchor]
        seen = {anchor}
        cur = anchor
        for _ in range(80):  # cycle guard
            parents = self._child_to_parents.get(cur)
            if not parents:
                break
            inter = [p for p in parents if norm(p) in greedy_norm]
            pick = sorted(inter)[0] if inter else sorted(parents)[0]
            if pick in seen:
                break
            path.append(pick)
            seen.add(pick)
            cur = pick
        return list(reversed(path))
