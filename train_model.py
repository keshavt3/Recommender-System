#!/usr/bin/env python3
"""
Train the collaborative filtering model.

Usage:
    python train_model.py
    python train_model.py --features 10 --lambda 100 --iterations 300
"""
import argparse
from src import load_data, train_model, save_model


def main():
    parser = argparse.ArgumentParser(description="Train anime recommender model")
    parser.add_argument("--features", type=int, default=10, help="Number of latent features")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=100, help="Regularization strength")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=300, help="Training iterations")
    parser.add_argument("--output", default="models", help="Output directory")
    args = parser.parse_args()

    config = {
        'num_features': args.features,
        'lambda_': args.lambda_,
        'learning_rate': args.lr,
        'iterations': args.iterations,
    }

    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data['ratings_df']):,} ratings")
    print(f"Items: {data['num_items']:,} | Users: {data['num_users']:,}")
    print()

    print("Training model...")
    model = train_model(data, config=config)
    print()

    save_model(model, save_dir=args.output)

    print()
    print(f"Final metrics:")
    print(f"  Train RMSE: {model['metrics']['train_rmse']:.4f}")
    print(f"  Test RMSE:  {model['metrics']['test_rmse']:.4f}")


if __name__ == "__main__":
    main()
