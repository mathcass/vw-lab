"""Download the two datasets at the exact revisions used in this study.

    uv run python -I -m vwjev.download

Each goes into its own directory under data/raw/ (treat the contents as
untrusted data; run any Python that reads them with -I).
"""

from huggingface_hub import snapshot_download

DATASETS = {
    # third-party BEV Decision Mix (license listed as "unknown")
    "bev-decision": ("avbiswas/bev-decision", "11ed9b5b879e7f9524e7c89c1037d498c54206f0"),
    # Fastino's fast-decisions dev split: our lockbox
    "fast-decisions": ("fastino/fast-decisions", "35afb79a0f611a1db44380e855ac2a821b0a4da1"),
}


def main() -> None:
    for local, (repo, rev) in DATASETS.items():
        path = snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=f"data/raw/{local}")
        print(f"{repo}@{rev[:8]} -> {path}")


if __name__ == "__main__":
    main()
