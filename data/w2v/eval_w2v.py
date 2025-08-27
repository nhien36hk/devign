#!/usr/bin/env python3
import argparse
import glob
import os
from typing import Optional, Set, Tuple, List

try:
    import pandas as pd  # optional, only for coverage over token pickles
except Exception:
    pd = None
from gensim.models import Word2Vec
import numpy as np


def load_tokens_summary(tokens_dir: str, max_files: int = 50) -> Optional[Tuple[int, Set[str]]]:
    """
    Load up to max_files token pickle files and return:
    - total token count across sampled files
    - set of unique tokens across sampled files
    Returns None if no files found.
    """
    if pd is None:
        return None

    files = sorted(glob.glob(os.path.join(tokens_dir, "*.pkl")))
    if not files:
        return None

    total_tokens = 0
    unique_tokens: Set[str] = set()

    for path in files[:max_files]:
        try:
            df = pd.read_pickle(path)
        except Exception:
            continue

        if "tokens" not in df.columns:
            continue

        for tokens in df["tokens"]:
            if isinstance(tokens, list):
                total_tokens += len(tokens)
                unique_tokens.update(tokens)

    return total_tokens, unique_tokens


def compute_sampled_total_coverage(model: Word2Vec, tokens_dir: str, max_files: int = 50) -> Optional[Tuple[int, int]]:
    """
    Compute (in_vocab_total, total_tokens) across up to max_files token files.
    Returns None if no files found.
    """
    if pd is None:
        return None

    files = sorted(glob.glob(os.path.join(tokens_dir, "*.pkl")))
    if not files:
        return None

    kv = model.wv
    total_tokens = 0
    total_in_vocab = 0

    for path in files[:max_files]:
        try:
            df = pd.read_pickle(path)
        except Exception:
            continue

        series = df.get("tokens")
        if series is None:
            continue

        for tokens in series:
            if not isinstance(tokens, list):
                continue
            total_tokens += len(tokens)
            total_in_vocab += sum(1 for t in tokens if t in kv.key_to_index)

    return total_in_vocab, total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Word2Vec quality check for Devign tokens")
    default_model = os.path.join(os.path.dirname(__file__), "w2v.model")
    default_tokens_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tokens"))

    parser.add_argument("--model", type=str, default=default_model, help="Path to Word2Vec .model file")
    parser.add_argument(
        "--tokens_dir",
        type=str,
        default=default_tokens_dir,
        help="Directory with token .pkl files (optional, used for coverage)",
    )
    parser.add_argument("--sample_files", type=int, default=50, help="Max token files to sample for coverage")
    parser.add_argument("--show_top", type=int, default=10, help="Show top-N tokens by rank")
    parser.add_argument("--neighbors", type=int, default=5, help="Neighbors per shown token")
    parser.add_argument("--tokens", type=str, default="", help="Comma-separated tokens to inspect vectors")
    parser.add_argument("--pairs", type=str, default="", help="Comma-separated pairs tokenA:tokenB to compare")
    parser.add_argument("--print_dims", type=int, default=10, help="How many vector dims to print per token")
    parser.add_argument("--show_full", action="store_true", help="Print full vectors instead of head dims")
    parser.add_argument("--coherence_top", type=int, default=50, help="Top-N tokens for coherence score")
    parser.add_argument("--coherence_k", type=int, default=5, help="Nearest neighbors per token for coherence")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        return

    model = Word2Vec.load(args.model)
    kv = model.wv

    # Basic stats
    print(f"model_path: {args.model}")
    print(f"vector_size: {kv.vector_size}")
    print(f"vocab_size: {len(kv.key_to_index)}")
    print(
        "train_args:",
        {
            "window": getattr(model, "window", None),
            "sg": getattr(model, "sg", None),
            "negative": getattr(model, "negative", None),
            "min_count": getattr(model, "min_count", None),
        },
    )

    # Coverage (if tokens_dir exists)
    if args.tokens_dir and os.path.isdir(args.tokens_dir):
        summary = load_tokens_summary(args.tokens_dir, max_files=args.sample_files)
        if summary is not None:
            total_tokens, unique_tokens = summary
            in_vocab_unique = sum(1 for t in unique_tokens if t in kv.key_to_index)
            oov_unique = len(unique_tokens) - in_vocab_unique
            print(
                f"coverage_unique_sampled: in_vocab={in_vocab_unique} / unique={len(unique_tokens)} => "
                f"{(in_vocab_unique / max(1, len(unique_tokens))):.3f}"
            )

            totals = compute_sampled_total_coverage(model, args.tokens_dir, max_files=args.sample_files)
            if totals is not None:
                in_vocab_total, total_count = totals
                print(
                    f"coverage_total_sampled: in_vocab={in_vocab_total} / tokens={total_count} => "
                    f"{(in_vocab_total / max(1, total_count)):.3f}"
                )
        else:
            print(f"No token files found in {args.tokens_dir}")
    else:
        print("tokens_dir not provided or not a directory; skipping coverage.")

    # Nearest neighbors for top tokens
    top_tokens = kv.index_to_key[: max(0, args.show_top)]
    if top_tokens:
        print(f"top_tokens: {top_tokens}")
        for token in top_tokens[: args.neighbors]:
            try:
                sims = kv.most_similar(token, topn=5)
                neighbors = ", ".join([w for w, _ in sims])
                print(f"neighbors[{token}]: {neighbors}")
            except KeyError:
                continue

    # Print vectors for requested tokens
    selected_tokens: List[str] = [t.strip() for t in args.tokens.split(",") if t.strip()]
    if selected_tokens:
        print("\nSelected token vectors:")
        for t in selected_tokens:
            if t in kv.key_to_index:
                vec = kv[t]
                if args.show_full:
                    print(f"{t}: {np.array2string(vec, precision=4, separator=", ")}")
                else:
                    head = np.array2string(vec[: max(0, args.print_dims)], precision=4, separator=", ")
                    print(f"{t}[:{args.print_dims}]: {head} ...")
            else:
                print(f"{t}: <OOV>")

    # Cosine similarity for pairs
    if args.pairs:
        print("\nPairwise cosine similarity:")
        pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
        for p in pairs:
            if ":" not in p:
                continue
            a, b = [x.strip() for x in p.split(":", 1)]
            if a in kv.key_to_index and b in kv.key_to_index:
                va = kv[a]
                vb = kv[b]
                denom = (np.linalg.norm(va) * np.linalg.norm(vb))
                sim = float(np.dot(va, vb) / denom) if denom > 0 else float("nan")
                print(f"cos({a}, {b}) = {sim:.4f}")
            else:
                print(f"cos({a}, {b}) = <OOV>")

    # Simple coherence score: average NN similarity over top-N tokens
    topN = max(0, args.coherence_top)
    K = max(1, args.coherence_k)
    if topN > 0 and kv.index_to_key:
        tokens_for_coherence = kv.index_to_key[: min(topN, len(kv.index_to_key))]
        sims_accum = []
        for t in tokens_for_coherence:
            try:
                sims = kv.most_similar(t, topn=K)
                sims_accum.append(float(np.mean([s for _, s in sims])))
            except KeyError:
                continue
        if sims_accum:
            print(f"\ncoherence@{K} over top-{len(tokens_for_coherence)}: {float(np.mean(sims_accum)):.4f}")


if __name__ == "__main__":
    main()


