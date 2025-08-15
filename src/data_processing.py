import pandas as pd
import numpy as np
from cf_model import normalizeRatings

def load_anime_data():
    """
    Load anime data from CSV files and preprocess it.
    Returns:
        Y (ndarray): Ratings matrix of shape (num_items, num_users)
        R (ndarray): Ratings presence matrix of shape (num_items, num_users)
        Ynorm (ndarray): Normalized ratings matrix
        Ymean (ndarray): Mean ratings for each item
    """
    animes_df = pd.read_csv("data/animes.csv")
    reviews_df = pd.read_csv("data/reviews.csv")
    
    anime_id_to_idx = {aid: idx for idx, aid in enumerate(animes_df['anime_id'])}
    user_ids = reviews_df['user_id'].unique()
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}

    num_items = len(anime_id_to_idx)
    num_users = len(user_id_to_idx)

    Y = np.zeros((num_items, num_users))
    R = np.zeros((num_items, num_users))

    for _, row in reviews_df.iterrows():
        i = anime_id_to_idx[row['anime_id']]
        j = user_id_to_idx[row['user_id']]
        Y[i, j] = row['score']
        R[i, j] = 1

    Ynorm, Ymean = normalizeRatings(Y, R)
    
    return Y, R, Ynorm, Ymean