"""Download gated SAM 3 / SAM 3.1 weights using the HF token from .env.

Usage:
    python -m bebop_vision.download_sam3            # both versions
    python -m bebop_vision.download_sam3 sam3.1     # one version
"""

import os

REPOS = {
    "sam3": ("facebook/sam3", "sam3.pt"),
    "sam3.1": ("facebook/sam3.1", "sam3.1_multiplex.pt"),
}


def load_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    return line.strip().split("=", 1)[1]
    return None


def main():
    from huggingface_hub import hf_hub_download

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    token = load_token()
    if not token:
        raise SystemExit("HF_TOKEN not set. Put it in .env or export it first.")

    import sys

    versions = sys.argv[1:] or ["sam3", "sam3.1"]
    for version in versions:
        if version not in REPOS:
            raise SystemExit(f"unknown version {version}, choices: {list(REPOS)}")
        repo_id, filename = REPOS[version]
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.join(project_root, "weights"),
            token=token,
        )
        print(f"{version} ready at {path}")


if __name__ == "__main__":
    main()