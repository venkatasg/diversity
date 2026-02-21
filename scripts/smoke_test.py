"""
Smoke test: run compute_all_metrics on a dozen sample sentences.
Uses 'all-MiniLM-L6-v2' as the embedding model (smaller than the default
Qwen/Qwen3-Embedding-0.6B) so the test finishes quickly without requiring
a multi-GB download.
"""

import json
import nltk

# Make sure the punkt tokeniser data is present before we import diversity.
nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)

from diversity import compute_all_metrics  # noqa: E402 — after nltk setup

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A swift auburn fox leaps gracefully over a sleeping hound.",
    "Scientists discovered a new species of deep-sea fish near the Mariana Trench.",
    "The stock market fell sharply after the unexpected interest rate announcement.",
    "She finished reading the novel in a single afternoon, captivated by every page.",
    "Engineers unveiled a bridge design that uses recycled ocean plastic as reinforcement.",
    "The symphony orchestra performed Beethoven's Ninth to a standing ovation.",
    "Researchers linked regular exercise to improved long-term memory retention.",
    "A magnitude-6.2 earthquake struck the coastal region but caused no major damage.",
    "The bakery on the corner sells sourdough loaves that sell out before noon.",
    "Children in the program learn basic coding concepts through interactive puzzles.",
    "The documentary explored the cultural significance of traditional weaving in the Andes.",
]

print(f"Corpus: {len(texts)} sentences\n")

results = compute_all_metrics(
    corpus=texts,
    embedding_model="all-MiniLM-L6-v2",
    verbose=True,
)

# Pretty-print everything except the raw per-doc score list.
# Cast numpy scalars to plain Python floats for JSON serialisation.
clean = {
    k: float(v) if hasattr(v, "item") else v
    for k, v in results.items()
    if k != "templates_per_token_scores"
}
print("\n--- Results ---")
print(json.dumps(clean, indent=2))
