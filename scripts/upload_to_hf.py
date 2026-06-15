"""
Upload the TaxoFormer model + assets to the HuggingFace Hub.

Prerequisite (one-time): authenticate with a write token:
    hf auth login            # or: huggingface-cli login

Then:
    python scripts/upload_to_hf.py --src /path/to/hf_export --repo msparsa/taxoformer

`hf_export/` must contain: model.safetensors, config.json, README.md (model card),
phylo2_mapping.json, parent_to_child_mapping.csv
"""
import os
import argparse
from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with model.safetensors, config.json, etc.")
    ap.add_argument("--repo", default="msparsa/taxoformer")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"Uploading {args.src} -> https://huggingface.co/{args.repo}")
    api.upload_folder(
        folder_path=args.src,
        repo_id=args.repo,
        repo_type="model",
        commit_message="Add TaxoFormer weights, taxonomy assets, and model card",
    )
    print("Done.")


if __name__ == "__main__":
    main()
