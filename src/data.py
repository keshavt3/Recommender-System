"""
Data loading and preprocessing for the anime recommender.
Handles the actual CSV column names: uid, profile, anime_uid, score
"""
import pandas as pd
import numpy as np


def load_data(animes_path="data/animes.csv", reviews_path="data/reviews.csv"):
    """
    Load and preprocess anime and reviews data.

    Returns:
        ratings_df: DataFrame with columns [anime_idx, user_idx, score, score_norm]
        anime_means: np.array of mean rating per anime
        anime_id_to_idx: dict mapping anime_uid -> index
        idx_to_anime_id: dict mapping index -> anime_uid
        user_id_to_idx: dict mapping profile -> index
        animes_df: original animes DataFrame (for titles)
    """
    animes_df = pd.read_csv(animes_path)
    reviews_df = pd.read_csv(reviews_path)

    # Create mappings from IDs to indices
    anime_id_to_idx = {aid: idx for idx, aid in enumerate(animes_df['uid'].unique())}
    idx_to_anime_id = {idx: aid for aid, idx in anime_id_to_idx.items()}
    user_id_to_idx = {uid: idx for idx, uid in enumerate(reviews_df['profile'].unique())}

    # Map to indices
    reviews_df['anime_idx'] = reviews_df['anime_uid'].map(anime_id_to_idx)
    reviews_df['user_idx'] = reviews_df['profile'].map(user_id_to_idx)

    # Compute mean rating per anime (for normalization)
    num_items = len(anime_id_to_idx)
    anime_means = np.zeros(num_items)
    means_dict = reviews_df.groupby('anime_idx')['score'].mean().to_dict()
    for idx, mean in means_dict.items():
        if not np.isnan(idx):
            anime_means[int(idx)] = mean

    # Normalize scores (subtract anime mean)
    reviews_df['score_norm'] = reviews_df.apply(
        lambda row: row['score'] - anime_means[int(row['anime_idx'])]
        if not np.isnan(row['anime_idx']) else np.nan,
        axis=1
    )

    # Keep only needed columns and drop NaN
    ratings_df = reviews_df[['anime_idx', 'user_idx', 'score', 'score_norm']].copy()
    ratings_df = ratings_df.dropna()
    ratings_df['anime_idx'] = ratings_df['anime_idx'].astype(int)
    ratings_df['user_idx'] = ratings_df['user_idx'].astype(int)

    return {
        'ratings_df': ratings_df,
        'anime_means': anime_means,
        'anime_id_to_idx': anime_id_to_idx,
        'idx_to_anime_id': idx_to_anime_id,
        'user_id_to_idx': user_id_to_idx,
        'animes_df': animes_df,
        'num_items': num_items,
        'num_users': len(user_id_to_idx),
    }


def get_anime_title(animes_df, anime_id):
    """Get anime title from anime_id (uid)."""
    match = animes_df[animes_df['uid'] == anime_id]
    if len(match) > 0:
        return match.iloc[0]['title']
    return f"Unknown (ID: {anime_id})"


def search_anime(animes_df, query, limit=10):
    """
    Search for anime by title (case-insensitive, matches all words).
    Returns list of (anime_id, title) tuples.
    """
    # Split query into words and match all of them
    words = query.lower().split()

    def matches_all_words(title):
        if pd.isna(title):
            return False
        title_lower = title.lower()
        return all(word in title_lower for word in words)

    matches = animes_df[animes_df['title'].apply(matches_all_words)]
    return [(row['uid'], row['title']) for _, row in matches.head(limit).iterrows()]
