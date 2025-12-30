import numpy as np
import tensorflow as tf
import pickle
import os

def save_collab_model(X, W, b, anime_means_array, anime_id_to_idx, user_id_to_idx, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Save learned parameters
    np.savez(os.path.join(save_dir, "collab_weights.npz"),
             X=X.numpy(),
             W=W.numpy(),
             b=b.numpy(),
             anime_means_array=anime_means_array)
    
    # Save mappings
    with open(os.path.join(save_dir, "collab_mappings.pkl"), "wb") as f:
        pickle.dump({
            "anime_id_to_idx": anime_id_to_idx,
            "user_id_to_idx": user_id_to_idx
        }, f)
    
    print("Collaborative filtering model saved successfully.")

def load_collab_model(save_dir="models"):
    # Load weights
    data = np.load(os.path.join(save_dir, "collab_weights.npz"))
    X = tf.Variable(data["X"], dtype=tf.float64)
    W = tf.Variable(data["W"], dtype=tf.float64)
    b = tf.Variable(data["b"], dtype=tf.float64)
    anime_means_array = data["anime_means_array"]
    
    # Load mappings
    with open(os.path.join(save_dir, "collab_mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    
    anime_id_to_idx = mappings["anime_id_to_idx"]
    user_id_to_idx = mappings["user_id_to_idx"]
    
    print("Collaborative filtering model loaded successfully.")
    return X, W, b, anime_means_array, anime_id_to_idx, user_id_to_idx
