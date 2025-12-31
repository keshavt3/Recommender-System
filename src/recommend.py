"""
Recommendation logic for collaborative filtering.
"""
import numpy as np


def predict_for_user(user_ratings, model):
    """
    Generate predictions for a new user based on their ratings.

    Args:
        user_ratings: Dict of {anime_id: rating} from user input
        model: Dict from load_model()

    Returns:
        np.array of predicted scores for all anime (indexed by anime_idx)
    """
    X = model['X'].numpy() if hasattr(model['X'], 'numpy') else model['X']
    anime_id_to_idx = model['anime_id_to_idx']
    anime_means = model['anime_means']
    num_features = X.shape[1]

    # Build user vector as weighted average of item vectors
    user_vec = np.zeros(num_features)
    total_weight = 0.0

    for anime_id, rating in user_ratings.items():
        idx = anime_id_to_idx.get(anime_id)
        if idx is not None:
            user_vec += X[idx] * rating
            total_weight += rating

    if total_weight > 0:
        user_vec /= total_weight

    # Predict scores for all items
    predictions = X.dot(user_vec) + anime_means

    # Clip to valid rating range
    predictions = np.clip(predictions, 1.0, 10.0)

    return predictions


def get_recommendations(user_ratings, model, top_n=10, min_ratings=10):
    """
    Get top-N recommendations for a user.

    Args:
        user_ratings: Dict of {anime_id: rating} from user input
        model: Dict from load_model()
        top_n: Number of recommendations to return
        min_ratings: Minimum number of ratings an anime must have

    Returns:
        List of (anime_id, predicted_score) tuples, sorted by score descending
    """
    predictions = predict_for_user(user_ratings, model)
    idx_to_anime_id = model['idx_to_anime_id']
    anime_means = model['anime_means']

    # Filter out anime with no/few ratings (anime_means == 0 means no ratings)
    # Also filter out already-rated anime
    rated_ids = set(user_ratings.keys())

    # Build list of (idx, score, popularity) for valid anime
    candidates = []
    for idx in range(len(predictions)):
        anime_id = idx_to_anime_id.get(idx)
        if anime_id is None or anime_id in rated_ids:
            continue
        # Skip anime with no ratings (anime_means == 0)
        if anime_means[idx] == 0:
            continue
        # Use anime_means as a proxy for popularity/quality
        # Higher mean = more popular/better rated
        popularity = anime_means[idx]
        candidates.append((idx, predictions[idx], popularity))

    # Sort by: prediction first, then popularity as tiebreaker
    # This surfaces popular anime when predictions are tied
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # Return top N
    recommendations = []
    for idx, score, _ in candidates[:top_n]:
        anime_id = idx_to_anime_id.get(idx)
        recommendations.append((anime_id, score))

    return recommendations
