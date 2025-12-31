"""
Anime Recommender - Collaborative Filtering
"""
from .data import load_data, get_anime_title, search_anime
from .train import train_model, DEFAULT_CONFIG
from .model_io import save_model, load_model, model_exists
from .recommend import get_recommendations, predict_for_user
