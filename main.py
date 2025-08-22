from src import load_anime_data, train_cf_model, predict_ratings, recommend_for_user
import pandas as pd

# Load data
Y, R, Ynorm, Ymean = load_anime_data()
animes_df = pd.read_csv("data/animes.csv")
#add in an intermediate step to split the data into training, validation, and test sets if needed
#Split data into training and test sets (60% train, 20% validation, 20% test)

# Train model
X, W, b = train_cf_model(Ynorm, R, num_features=10, lambda_=0.1, alpha=1e-2, epochs=200)

# Predict
predictions = predict_ratings(X, W, b, Ymean)

# Recommend for a specific user
user_id = 12345  # replace with an actual user_id from reviews.csv
# You’ll need the user_id_to_idx mapping from data_processing.py
#we can just return this is load_anime_data()
from src.data_processing import get_user_id_mapping
user_id_to_idx = get_user_id_mapping()

recommended_titles = recommend_for_user(predictions, user_id, R, animes_df, user_id_to_idx)
print("Recommended Anime:", recommended_titles)
#add in a different file prompting user for input
# and displaying recommendations based on user input so that we dont need to retrain the model every time



import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
#we need to create Y, R, Ynorm, Ymean from reviews.csv and animes.csv (num_users is number of unique profiles in reviews.csv, num_items is number of unique anime_ids in animes.csv)
#Y matrix is ratings, R matrix is presence of ratings, Ynorm is normalized ratings, Ymean is mean ratings for each item
#Y shape is (num_items, num_users), R shape is (num_items, num_users)

animes_df = pd.read_csv("data/animes.csv")
reviews_df = pd.read_csv("data/reviews.csv")

# Map IDs to indices
anime_id_to_idx = {aid: idx for idx, aid in enumerate(animes_df['uid'].unique())}
user_id_to_idx  = {uid: idx for idx, uid in enumerate(reviews_df['profile'].unique())}

# Convert to indices
reviews_df['anime_idx'] = reviews_df['anime_uid'].map(anime_id_to_idx)
reviews_df['profile_idx']  = reviews_df['profile'].map(user_id_to_idx)

# Compute mean per anime
anime_means = reviews_df.groupby('anime_idx')['score'].mean().to_dict()

num_items = animes_df.shape[0]   # total number of animes in catalog

anime_means_array = np.zeros(num_items)   # default fill with 0
for idx, mean in anime_means.items():
    anime_means_array[idx] = mean

# Normalize ratings (subtract per-anime mean)
reviews_df['score_norm'] = reviews_df.apply(
    lambda row: row['score'] - anime_means_array[row['anime_idx']],
    axis=1
)

# Keep only needed columns (triplet + normalized score)
ratings_df = reviews_df[['anime_idx', 'profile_idx', 'score', 'score_norm']].copy()

num_items = len(anime_id_to_idx)
num_users = len(user_id_to_idx) + 1  # +1 for the new user
num_features = 10

# Initialize parameters
X = tf.Variable(tf.random.normal((num_items, num_features), dtype=tf.float64), name='X')
W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')

#concatenate a new user's ratings to Y and R at the beginning
new_user_id = "new_user"
new_user_idx = num_users - 1

new_ratings = [
    (0, new_user_idx, 10),
    (50, new_user_idx, 8),
    (100, new_user_idx, 9),
    (150, new_user_idx, 7),
    (200, new_user_idx, 6),
    (250, new_user_idx, 8),
    (300, new_user_idx, 9),
    (350, new_user_idx, 7),
]

new_df = pd.DataFrame(new_ratings, columns=['anime_idx', 'user_idx', 'score'])

# normalize new ratings
new_df['score_norm'] = new_df.apply(
    lambda row: row['score'] - anime_means_array[row['anime_idx']],
    axis=1
)

ratings_df = pd.concat([ratings_df, new_df], ignore_index=True)
#change this code to simulate a new user with some ratings

# 1. Remove rows with NaNs in anime_idx, user_idx, or score_norm
ratings_df = ratings_df.dropna(subset=['anime_idx', 'user_idx', 'score_norm'])

def cofi_cost_func_triplet(X, W, b, anime_idx_tensor, user_idx_tensor, ratings_tensor, lambda_):
    preds = tf.reduce_sum(tf.gather(X, anime_idx_tensor) * tf.gather(W, user_idx_tensor), axis=1) + tf.gather(b[0], user_idx_tensor)
    err = preds - ratings_tensor
    J = 0.5 * tf.reduce_sum(err**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

# Convert to TensorFlow tensors with correct dtypes
anime_idx_tensor = tf.constant(ratings_df['anime_idx'].values, dtype=tf.int32)
user_idx_tensor  = tf.constant(ratings_df['user_idx'].values, dtype=tf.int32)
ratings_tensor   = tf.constant(ratings_df['score_norm'].values, dtype=tf.float64)

optimizer = keras.optimizers.Adam(learning_rate=1e-1)
iterations = 200
lambda_ = 1

for iter in range(iterations):
    with tf.GradientTape() as tape:
        cost_value = cofi_cost_func_triplet(
            X, W, b, anime_idx_tensor, user_idx_tensor, ratings_tensor, lambda_
        )
    grads = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))

    if iter % 20 == 0:
        print(f"Training loss at iteration {iter}: {cost_value.numpy():0.2f}")

# Get feature vector and bias for the new user
w_new = W[new_user_idx].numpy()   # shape (num_features,)
b_new = b[0, new_user_idx].numpy()

# Predictions for all items for this user
my_predictions = X.numpy().dot(w_new) + b_new + np.array(list(anime_means_array))
ix = np.argsort(my_predictions)[::-1]
print("\nNew user's actual ratings vs predicted ratings:")
for _, row in new_df.iterrows():
    idx = row['anime_idx']
    actual = row['score']
    predicted = my_predictions[idx]
    print(f"Anime idx {idx}: Actual = {actual}, Predicted = {predicted:.2f}")
print("\nTop 10 anime recommendations for the new user:")
for i in ix[:10]:
    anime_id = animes_df['uid'].iloc[i]
    title = animes_df['title'].iloc[i]
    print(f"Anime ID {anime_id}: {title} (Predicted Rating: {my_predictions[i]:.2f})")