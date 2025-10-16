import joblib
import os
from tensorflow.keras.models import load_model

def save_recommender(model, save_dir, user_matrix, item_matrix,
                     user_to_idx, anime_to_idx, idx_to_user, idx_to_anime):
    """
    Saves the trained model and all necessary data for reloading later.
    """

    import os
    os.makedirs(save_dir, exist_ok=True)

    # --- 1️⃣ Save the Keras model ---
    model.save(os.path.join(save_dir, "anime_recommender.keras"))

    # --- 2️⃣ Save all data objects ---
    joblib.dump({
        "user_matrix": user_matrix,
        "item_matrix": item_matrix,
        "user_to_idx": user_to_idx,
        "anime_to_idx": anime_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_anime": idx_to_anime
    }, f"{save_dir}/recommender_data.pkl")

    print(f"✅ Model and data saved in: {save_dir}")


def load_recommender(save_dir):
    """
    Loads the trained model and all supporting data.
    Returns a tuple: (model, data_dict)
    """

    # --- 1️⃣ Load model ---
    model = load_model(os.path.join(save_dir, "anime_recommender.keras"))

    # --- 2️⃣ Load data ---
    data = joblib.load(f"{save_dir}/recommender_data.pkl")

    # --- 3️⃣ Extract everything from the saved dict ---
    user_matrix = data["user_matrix"]
    item_matrix = data["item_matrix"]
    anime_to_idx = data["anime_to_idx"]
    user_to_idx = data["user_to_idx"]
    idx_to_anime = data["idx_to_anime"]
    idx_to_user = data["idx_to_user"]

    print(f"✅ Model and data loaded from: {save_dir}")
    return model, user_matrix, item_matrix, anime_to_idx, user_to_idx, idx_to_anime, idx_to_user
