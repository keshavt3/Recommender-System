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