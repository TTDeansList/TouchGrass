"""
One-time prefetch of models to local folders so labeling runs fully offline.
"""
from huggingface_hub import snapshot_download

def main():
    # Zero-shot NLI
    snapshot_download(
        "facebook/bart-large-mnli",
        local_dir="backend/models/bart-large-mnli",
        ignore_patterns=["*.safetensors.index.json", "*.h5"],  # keep it light
    )
    # Sentence-Transformers for relevancy
    snapshot_download(
        "sentence-transformers/all-MiniLM-L6-v2",
        local_dir="backend/models/all-MiniLM-L6-v2",
        ignore_patterns=["*.h5"],
    )
    print("[OK] Prefetched models into backend/models/")

if __name__ == "__main__":
    main()
