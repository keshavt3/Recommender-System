"""
Save and load collaborative filtering models.
"""
import os
import numpy as np
import tensorflow as tf
import pickle


DEFAULT_MODEL_DIR = "models"


def save_model(model, save_dir=DEFAULT_MODEL_DIR):
    """
    Save trained CF model to disk.

    Args:
        model: Dict returned by train_model()
        save_dir: Directory to save to
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save TensorFlow variables as numpy arrays
    np.savez(
        os.path.join(save_dir, "weights.npz"),
        X=model['X'].numpy(),
        W=model['W'].numpy(),
        b=model['b'].numpy(),
        anime_means=model['anime_means']
    )

    # Save mappings and config
    with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
        pickle.dump({
            'anime_id_to_idx': model['anime_id_to_idx'],
            'idx_to_anime_id': model['idx_to_anime_id'],
            'user_id_to_idx': model['user_id_to_idx'],
            'config': model['config'],
            'metrics': model.get('metrics', {}),
        }, f)

    print(f"Model saved to {save_dir}/")


def load_model(save_dir=DEFAULT_MODEL_DIR):
    """
    Load trained CF model from disk.

    Returns:
        Dict with model parameters and metadata
    """
    # Load weights
    data = np.load(os.path.join(save_dir, "weights.npz"))
    X = tf.Variable(data['X'], dtype=tf.float64)
    W = tf.Variable(data['W'], dtype=tf.float64)
    b = tf.Variable(data['b'], dtype=tf.float64)
    anime_means = data['anime_means']

    # Load metadata
    with open(os.path.join(save_dir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    print(f"Model loaded from {save_dir}/")

    return {
        'X': X,
        'W': W,
        'b': b,
        'anime_means': anime_means,
        **metadata
    }


def model_exists(save_dir=DEFAULT_MODEL_DIR):
    """Check if a saved model exists."""
    return (
        os.path.exists(os.path.join(save_dir, "weights.npz")) and
        os.path.exists(os.path.join(save_dir, "metadata.pkl"))
    )
