"""
Convert the Lightning .ckpt into a compact, shareable model.safetensors + config.json
for HuggingFace / the public repo.

Keeps only the *trained* parameters:
  - decoder.*                       (the taxonomy decoder, incl. level_embedding)
  - feature_extractor.*lora_*       (ESM LoRA deltas, rank 8)
  - feature_extractor.{q,k,v,out}_proj.*, feature_extractor.mlp.*  (attention pooling head)
Drops the frozen ESM-2 base weights (feature_extractor.esm_model.* minus LoRA): those are
re-downloaded at load time from the `esm` package (Meta-licensed) and applied via LoRA.

Usage: python convert_checkpoint.py --ckpt <path.ckpt> --out <dir>
"""
import os
import sys
import json
import argparse

# the ckpt pickles a PhyloTokenizer in its hyper_parameters; make it importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file


def is_base_esm(key: str) -> bool:
    """True for frozen ESM base weights we DON'T ship (everything under esm_model except LoRA)."""
    return key.startswith("feature_extractor.esm_model.") and "lora_" not in key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to the Lightning .ckpt to convert")
    ap.add_argument("--out", default="hf_export", help="output dir for safetensors + config.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.ckpt} ...", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]

    keep = {}
    dropped = 0
    for k, v in sd.items():
        if k.startswith("decoder.") or (k.startswith("feature_extractor.") and not is_base_esm(k)):
            keep[k] = v.contiguous()
        else:
            dropped += 1

    n_params = sum(v.numel() for v in keep.values())
    print(f"Keeping {len(keep)} tensors ({n_params/1e6:.1f}M params); dropped {dropped} base-ESM tensors.")

    out_st = os.path.join(args.out, "model.safetensors")
    save_file(keep, out_st, metadata={"format": "pt"})
    sz = os.path.getsize(out_st) / 1e6
    print(f"Wrote {out_st} ({sz:.1f} MB)")

    # config: hyperparameters needed to rebuild the model
    hp = ck.get("hyper_parameters", {})
    level_emb = sd["decoder.level_embedding.weight"]
    config = {
        "esm_model_name": "esm2_t33_650M_UR50D",
        "embed_dim": int(sd["decoder.token_embedding.weight"].shape[1]),
        "vocab_size": int(sd["decoder.token_embedding.weight"].shape[0]),
        "taxonomy_seq_length": int(sd["decoder.positional_embedding.weight"].shape[0]) - 1,
        "num_levels": int(level_emb.shape[0]) - 1,
        "decoder_layers": 4,
        "decoder_heads": 8,
        "dropout": 0.1,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_target_modules": ["query", "key", "value"],
        "max_seq_len_esm": 1022,
        "bos_token_id": 1,            # CONT id; verified correct for greedy decoding
        "pad_token_id": int(hp.get("pad_token_id", 0)),
        "trained_steps": int(ck.get("global_step", 0)),
        "dtype": str(next(iter(keep.values())).dtype).replace("torch.", ""),
    }
    out_cfg = os.path.join(args.out, "config.json")
    with open(out_cfg, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {out_cfg}:\n{json.dumps(config, indent=2)}")


if __name__ == "__main__":
    main()
