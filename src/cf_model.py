import numpy as np
import pandas as pd
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras

def normalizeRatings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).
    Only include real ratings R(i,j)=1.
    [Ynorm, Ymean] = normalizeRatings(Y, R) normalized Y so that each movie
    has a rating of 0 on average. Unrated moves then have a mean rating (0)
    Returns the mean rating in Ymean.
    """
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R) 
    return(Ynorm, Ymean)

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

def train_cf_model(Y, R, num_features, lambda_, alpha, epochs):
    num_animes, num_users = Y.shape

    # Initialize parameters
    W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
    X = tf.Variable(tf.random.normal((num_animes, num_features),dtype=tf.float64),  name='X')
    b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

    optimizer = keras.optimizers.Adam(learning_rate=1e-1)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            J = cofi_cost_func_v(X, W, b, Y, R, lambda_)
        grads = tape.gradient(J, [X, W, b])

        optimizer.apply_gradients( zip(grads, [X,W,b]) )

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: cost = {J.numpy():.4f}")

    return X, W, b

def predict_ratings(X, W, b, Ymean):
    return X @ tf.transpose(W) + b + Ymean

def recommend_for_user(predictions, user_id, R, animes_df, user_id_to_idx):
    j = user_id_to_idx[user_id]
    user_ratings = predictions[:, j].numpy()
    already_rated = R[:, j] == 1
    user_ratings[already_rated] = -np.inf  # exclude rated items
    top_indices = np.argsort(user_ratings)[::-1][:10]
    return animes_df.iloc[top_indices]['title'].tolist()
