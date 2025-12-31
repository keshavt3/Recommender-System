"""
Collaborative filtering model training.
Uses triplet-based matrix factorization with TensorFlow.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


# Default hyperparameters (tuned to reduce overfitting)
DEFAULT_CONFIG = {
    'num_features': 10,
    'lambda_': 100,
    'learning_rate': 0.1,
    'iterations': 300,
    'test_size': 0.2,
    'random_seed': 42,
}


def cost_function(X, W, b, anime_idx, user_idx, ratings, lambda_):
    """
    Collaborative filtering cost function (triplet form).

    Args:
        X: Item latent factors (num_items, num_features)
        W: User latent factors (num_users, num_features)
        b: User biases (1, num_users)
        anime_idx: Tensor of anime indices for each rating
        user_idx: Tensor of user indices for each rating
        ratings: Tensor of normalized ratings
        lambda_: Regularization strength
    """
    preds = tf.reduce_sum(
        tf.gather(X, anime_idx) * tf.gather(W, user_idx), axis=1
    ) + tf.gather(b[0], user_idx)

    err = preds - ratings
    J = 0.5 * tf.reduce_sum(err**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J


def calculate_rmse(X, W, b, df, anime_means):
    """Calculate RMSE on a ratings dataframe."""
    anime_idx = df['anime_idx'].values.astype(int)
    user_idx = df['user_idx'].values.astype(int)
    actual = df['score'].values

    X_np = X.numpy() if hasattr(X, 'numpy') else X
    W_np = W.numpy() if hasattr(W, 'numpy') else W
    b_np = b.numpy() if hasattr(b, 'numpy') else b

    preds = np.sum(X_np[anime_idx] * W_np[user_idx], axis=1) + b_np[0, user_idx] + anime_means[anime_idx]
    return np.sqrt(np.mean((preds - actual) ** 2))


def train_model(data, config=None, verbose=True):
    """
    Train collaborative filtering model.

    Args:
        data: Dict from load_data()
        config: Optional dict of hyperparameters
        verbose: Print training progress

    Returns:
        Dict with trained model parameters and metadata
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    ratings_df = data['ratings_df']
    anime_means = data['anime_means']
    num_items = data['num_items']
    num_users = data['num_users']

    # Train/test split
    train_df, test_df = train_test_split(
        ratings_df,
        test_size=cfg['test_size'],
        random_state=cfg['random_seed']
    )

    if verbose:
        print(f"Training data: {len(train_df):,} ratings")
        print(f"Test data: {len(test_df):,} ratings")
        print(f"Config: {cfg['num_features']} features, lambda={cfg['lambda_']}")
        print("-" * 50)

    # Initialize parameters
    tf.random.set_seed(cfg['random_seed'])
    X = tf.Variable(tf.random.normal((num_items, cfg['num_features']), dtype=tf.float64))
    W = tf.Variable(tf.random.normal((num_users, cfg['num_features']), dtype=tf.float64))
    b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64))

    # Convert to tensors
    anime_idx_t = tf.constant(train_df['anime_idx'].values, dtype=tf.int32)
    user_idx_t = tf.constant(train_df['user_idx'].values, dtype=tf.int32)
    ratings_t = tf.constant(train_df['score_norm'].values, dtype=tf.float64)

    optimizer = keras.optimizers.Adam(learning_rate=cfg['learning_rate'])

    best_test_rmse = float('inf')
    best_iter = 0

    for i in range(cfg['iterations']):
        with tf.GradientTape() as tape:
            cost = cost_function(X, W, b, anime_idx_t, user_idx_t, ratings_t, cfg['lambda_'])
        grads = tape.gradient(cost, [X, W, b])
        optimizer.apply_gradients(zip(grads, [X, W, b]))

        if verbose and i % 50 == 0:
            train_rmse = calculate_rmse(X, W, b, train_df, anime_means)
            test_rmse = calculate_rmse(X, W, b, test_df, anime_means)
            marker = " *" if test_rmse < best_test_rmse else ""
            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                best_iter = i
            print(f"Iter {i:3d} | Train: {train_rmse:.4f} | Test: {test_rmse:.4f}{marker}")

    train_rmse = calculate_rmse(X, W, b, train_df, anime_means)
    test_rmse = calculate_rmse(X, W, b, test_df, anime_means)

    if verbose:
        print("-" * 50)
        print(f"Final | Train: {train_rmse:.4f} | Test: {test_rmse:.4f}")
        print(f"Best test RMSE: {best_test_rmse:.4f} at iter {best_iter}")

    return {
        'X': X,
        'W': W,
        'b': b,
        'anime_means': anime_means,
        'anime_id_to_idx': data['anime_id_to_idx'],
        'idx_to_anime_id': data['idx_to_anime_id'],
        'user_id_to_idx': data['user_id_to_idx'],
        'config': cfg,
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'best_test_rmse': best_test_rmse,
        }
    }
