"""
recommender.py

Usage:
    python recommender.py

Prompts:
    Enter anime ratings (format: ID:rating, ID:rating, ...)

Requirements:
    - save_and_load_recommender.load_recommender(save_dir) available
    - save_and_load_cf_recommender.load_collab_model(save_dir) available
    - The saved files must have consistent anime IDs (UIDs). If you saved animes.csv
      in the same project, the script will attempt to load it for titles; otherwise
      it will print anime IDs where titles are unavailable.
"""

import os
import sys
import numpy as np
import pandas as pd

from save_and_load_cf_recommender import load_collab_model
from save_and_load_recommender import load_recommender


def parse_user_input(input_string, anime_to_idx):
    """
    Parse input_string like "33352:10, 5114:8, 1575:9" into {anime_id: rating}.
    anime_to_idx is used to validate anime IDs quickly (it maps anime_id -> idx).
    """
    user_ratings = {}
    if not input_string:
        return user_ratings

    pairs = input_string.split(',')
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue
        try:
            anime_id_str, rating_str = pair.split(':')
            anime_id = int(anime_id_str.strip())
            rating = float(rating_str.strip())
            if anime_id not in anime_to_idx:
                print(f"Warning: Anime ID {anime_id} not in content-model dataset — skipping.")
                continue
            if rating < 1 or rating > 10:
                print(f"Warning: Rating {rating} invalid (must be 1–10), skipping {anime_id}.")
                continue
            user_ratings[anime_id] = rating
        except ValueError:
            print(f"Warning: Unable to parse '{pair}', expected format ID:rating — skipping.")
            continue
    return user_ratings


def user_vector_content(user_ratings, anime_to_idx, item_matrix):
    """
    Build a user vector in content feature space as a weighted average of item vectors.
    item_matrix shape: (num_items, n_features)
    anime_to_idx maps anime_id -> index in item_matrix.
    """
    vec = np.zeros(item_matrix.shape[1], dtype=float)
    count = 0.0
    for anime_id, rating in user_ratings.items():
        idx = anime_to_idx.get(anime_id)
        if idx is None:
            continue
        vec += item_matrix[idx] * rating
        count += 1.0
    if count > 0:
        vec /= count
    return vec


def user_vector_collab(user_ratings, anime_id_to_idx, X):
    """
    Build a user vector in collaborative latent space as a weighted average of item latent vectors.
    X shape: (num_items_collab, num_features)
    anime_id_to_idx maps anime_id -> index in X.
    """
    num_features = X.shape[1]
    vec = np.zeros(num_features, dtype=float)
    count = 0.0
    for anime_id, rating in user_ratings.items():
        idx = anime_id_to_idx.get(anime_id)
        if idx is None:
            continue
        vec += X[idx] * rating
        count += 1.0
    if count > 0:
        vec /= count
    return vec


def align_collab_to_content_order(collab_preds, anime_id_to_idx, idx_to_anime, content_length):
    """
    Given collab_preds (length = number of collab items) and mappings,
    return an array of length content_length, where each entry corresponds
    to the content model's index order. If a content index maps to an anime
    not found in the collab model, put np.nan.
    - anime_id for content idx i is idx_to_anime[i]
    - collab index for anime_id is anime_id_to_idx.get(anime_id)
    """
    aligned = np.full(content_length, np.nan, dtype=float)
    for content_idx, anime_id in idx_to_anime.items():
        # idx_to_anime is expected to be a dict mapping content_idx -> anime_id
        collab_idx = anime_id_to_idx.get(anime_id)
        if collab_idx is not None and collab_idx < len(collab_preds):
            aligned[content_idx] = collab_preds[collab_idx]
        # else remain nan
    return aligned


def combine_predictions(content_preds, collab_preds_aligned, fallback_strategy='content', alpha=0.5):
    """
    Combine content_preds and collab_preds_aligned into a hybrid prediction.
    - content_preds: np.array (num_content_items,)
    - collab_preds_aligned: np.array (same length) with np.nan where unavailable
    fallback_strategy: 'content' or 'mean' or 'zero' — how to handle missing collab preds
    alpha: weight for content model (0..1)
    Returns: hybrid_preds (np.array)
    """
    assert 0.0 <= alpha <= 1.0
    # Prepare collab array filled where missing according to fallback
    collab_filled = collab_preds_aligned.copy()
    missing_mask = np.isnan(collab_filled)
    if np.any(missing_mask):
        if fallback_strategy == 'content':
            collab_filled[missing_mask] = content_preds[missing_mask]
        elif fallback_strategy == 'mean':
            mean_val = np.nanmean(collab_filled)
            if np.isnan(mean_val):
                mean_val = 0.0
            collab_filled[missing_mask] = mean_val
        elif fallback_strategy == 'zero':
            collab_filled[missing_mask] = 0.0
        else:
            collab_filled[missing_mask] = content_preds[missing_mask]

    hybrid = alpha * content_preds + (1.0 - alpha) * collab_filled
    return hybrid


def load_anime_titles(possible_paths=None, id_col_candidates=("uid", "anime_id", "id"), title_col_candidates=("title", "name")):
    """
    Try to load animes.csv (or similar) to produce anime_titles dict: anime_id -> title.
    If no file found, return empty dict.
    possible_paths: iterable of filenames to try. If None, try common names in cwd.
    """
    if possible_paths is None:
        possible_paths = ["data/animes.csv", "animes.csv", "data/anime.csv", "anime.csv"]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # try to find id and title columns
                id_col = next((c for c in id_col_candidates if c in df.columns), None)
                title_col = next((c for c in title_col_candidates if c in df.columns), None)
                if id_col and title_col:
                    return dict(zip(df[id_col].astype(int), df[title_col].astype(str)))
            except Exception:
                continue
    # not found or failed -> return empty dict
    return {}


def recommend_top_n(user_ratings, models, top_n=10, alpha=0.5, fallback_strategy='content'):
    """
    Core recommend flow. models is the dict returned by load_models() below.
    Returns list of (anime_id, title_or_none, predicted_score)
    """
    # content model data
    content = models['content']
    collab = models['collab']

    item_matrix = content['item_matrix']  # numpy array shape (num_content_items, n_features)
    anime_to_idx = content['anime_to_idx']  # anime_id -> content_idx
    idx_to_anime = content['idx_to_anime']  # content_idx -> anime_id (dict)

    # Build content user vector & predictions
    user_vec_content = user_vector_content(user_ratings, anime_to_idx, item_matrix)
    preds_content = item_matrix.dot(user_vec_content)  # shape (num_content_items,)

    # Build collab user vector & predictions
    X = collab['X']  # numpy array (num_collab_items, n_features)
    anime_id_to_idx = collab['anime_id_to_idx']  # anime_id -> collab_idx
    anime_means_array = collab['anime_means_array']  # length num_collab_items

    # If X is tensorflow.Variable convert to numpy:
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    user_vec_collab = user_vector_collab(user_ratings, anime_id_to_idx, X)
    # collab preds in collab index order
    preds_collab = X.dot(user_vec_collab) + np.array(anime_means_array)

    # Align collab preds into content index order
    content_length = preds_content.shape[0]
    collab_aligned = align_collab_to_content_order(preds_collab, anime_id_to_idx, idx_to_anime, content_length)

    # Combine
    hybrid = combine_predictions(preds_content, collab_aligned, fallback_strategy=fallback_strategy, alpha=alpha)

    # Filter out anime the user already rated
    filtered_indices = [i for i in np.argsort(hybrid)[::-1] if idx_to_anime.get(i) not in user_ratings]

    # Take top N after filtering
    top_idx = filtered_indices[:top_n]

    return [(idx_to_anime.get(i, None), hybrid[i]) for i in top_idx]



def load_models(content_dir="saved_recommender", collab_dir="models"):
    """
    Wrapper that loads both saved models and returns a dictionary with their assets.
    The exact keys must match what your save functions wrote.
    """
    # Content-based model loader: returns (model, user_matrix, item_matrix, anime_to_idx, user_to_idx, idx_to_anime, idx_to_user)
    try:
        model, user_matrix, item_matrix, anime_to_idx, user_to_idx, idx_to_anime, idx_to_user = load_recommender(content_dir)
    except Exception as e:
        print("Error loading content-based recommender:", e)
        raise

    # Ensure item_matrix and idx_to_anime are numpy-compatible
    item_matrix = np.array(item_matrix)
    # idx_to_anime might be list or dict; normalize to dict mapping content_idx -> anime_id
    if isinstance(idx_to_anime, dict):
        idx_to_anime_map = {int(k): int(v) for k, v in idx_to_anime.items()}
    elif isinstance(idx_to_anime, (list, tuple, np.ndarray)):
        idx_to_anime_map = {int(i): int(a) for i, a in enumerate(idx_to_anime)}
    else:
        raise ValueError("Unexpected idx_to_anime type returned by load_recommender")

    # Collaborative loader
    try:
        X_var, W_var, b_var, anime_means_array, anime_id_to_idx, user_id_to_idx = load_collab_model(collab_dir)
    except Exception as e:
        print("Error loading collaborative model:", e)
        raise

    # Convert TF variables to numpy arrays if necessary
    try:
        X_arr = np.array(X_var)
    except Exception:
        X_arr = X_var

    return {
        "content": {
            "model": model,
            "user_matrix": user_matrix,
            "item_matrix": item_matrix,
            "anime_to_idx": anime_to_idx,
            "user_to_idx": user_to_idx,
            "idx_to_anime": idx_to_anime_map,
            "idx_to_user": idx_to_user,
        },
        "collab": {
            "X": X_arr,
            "W": W_var,
            "b": b_var,
            "anime_means_array": np.array(anime_means_array),
            "anime_id_to_idx": anime_id_to_idx,
            "user_id_to_idx": user_id_to_idx
        }
    }


def main():
    print("Loading models...")
    models = load_models(content_dir="saved_recommender", collab_dir="models")
    print("Models loaded.\n")

    # Try to get anime titles for nicer output
    anime_titles = load_anime_titles()

    print("Enter your anime ratings in one line (format: ID:rating, ID:rating, ...).")
    print("Example: 33352:10, 5114:8, 1575:9")
    user_input = input("Ratings: ").strip()

    # Use content's anime_to_idx to validate IDs
    content_anime_to_idx = models['content']['anime_to_idx']
    user_ratings = parse_user_input(user_input, content_anime_to_idx)

    if len(user_ratings) == 0:
        print("No valid ratings provided. Exiting.")
        return

    # You can tune alpha here or prompt the user for it
    alpha = 0.5  # weight for content-based model (0..1)
    fallback_strategy = 'content'  # options: 'content', 'mean', 'zero'

    top = recommend_top_n(user_ratings, models, top_n=10, alpha=alpha, fallback_strategy=fallback_strategy)

    print("\nTop 10 recommendations (hybrid):\n")
    for rank, (anime_id, score) in enumerate(top, start=1):
        title = anime_titles.get(anime_id, None)
        if title is None:
            # If no title file loaded, try to display the ID itself
            title_display = f"Anime ID {anime_id}"
        else:
            title_display = f"{title} (ID {anime_id})"
        print(f"{rank:2d}. {title_display}")



if __name__ == "__main__":
    main()


