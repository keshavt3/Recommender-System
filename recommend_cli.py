#!/usr/bin/env python3
"""
Interactive CLI for getting anime recommendations.

Usage:
    python recommend_cli.py
"""
import pandas as pd
from src import load_model, model_exists, get_recommendations, load_data, search_anime, get_anime_title


def search_and_select(animes_df, query):
    """Search for anime and let user select one."""
    results = search_anime(animes_df, query, limit=10)

    if not results:
        print(f"  No anime found matching '{query}'")
        return None

    print(f"\n  Found {len(results)} matches:")
    for i, (anime_id, title) in enumerate(results, 1):
        print(f"    {i}. {title}")

    while True:
        choice = input("  Select number (or 'skip'): ").strip().lower()
        if choice == 'skip' or choice == 's':
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                return results[idx][0]  # Return anime_id
            print("  Invalid selection, try again")
        except ValueError:
            print("  Enter a number or 'skip'")


def get_user_ratings(animes_df):
    """Interactive prompt to collect user ratings."""
    print("\n" + "=" * 60)
    print("RATE SOME ANIME")
    print("=" * 60)
    print("Search for anime you've watched and rate them 1-10.")
    print("Type 'done' when finished (need at least 3 ratings).\n")

    ratings = {}

    while True:
        query = input("Search anime (or 'done'): ").strip()

        if query.lower() == 'done':
            if len(ratings) < 3:
                print(f"  Need at least 3 ratings (you have {len(ratings)})")
                continue
            break

        if not query:
            continue

        anime_id = search_and_select(animes_df, query)
        if anime_id is None:
            continue

        title = get_anime_title(animes_df, anime_id)

        while True:
            rating_str = input(f"  Rate '{title}' (1-10): ").strip()
            try:
                rating = float(rating_str)
                if 1 <= rating <= 10:
                    ratings[anime_id] = rating
                    print(f"  Added: {title} = {rating}")
                    break
                print("  Rating must be 1-10")
            except ValueError:
                print("  Enter a number 1-10")

    return ratings


def display_recommendations(recommendations, animes_df):
    """Display recommendations nicely."""
    print("\n" + "=" * 60)
    print("YOUR RECOMMENDATIONS")
    print("=" * 60 + "\n")

    for i, (anime_id, score) in enumerate(recommendations, 1):
        title = get_anime_title(animes_df, anime_id)
        print(f"  {i:2d}. {title}")
        print(f"      Predicted score: {score:.1f}/10\n")


def main():
    # Check if model exists
    if not model_exists():
        print("No trained model found. Run 'python train_model.py' first.")
        return

    print("Loading model...")
    model = load_model()

    print("Loading anime data...")
    data = load_data()
    animes_df = data['animes_df']

    print(f"\nWelcome to the Anime Recommender!")
    print(f"Database: {data['num_items']:,} anime")

    # Get user ratings
    user_ratings = get_user_ratings(animes_df)

    print(f"\nYou rated {len(user_ratings)} anime. Generating recommendations...")

    # Get recommendations
    recommendations = get_recommendations(user_ratings, model, top_n=10)

    # Display
    display_recommendations(recommendations, animes_df)

    print("\nThanks for using the Anime Recommender!")


if __name__ == "__main__":
    main()
