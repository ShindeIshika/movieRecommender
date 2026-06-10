import pandas as pd
import numpy as np
import os

# Import all modules
from content_based import build_content_based_model, get_content_recommendations
from collaborative import build_user_user_cf, get_user_recommendations
from matrix_factorization import build_svd_model, get_svd_recommendations, save_model, load_model

def main():
    print("=" * 50)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("=" * 50)
    
    # Load data
    print("\nLoading data...")
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    
    print(f"Loaded {len(ratings)} ratings from {ratings['user_id'].nunique()} users")
    print(f"Total movies: {len(movies)}")
    
    # 1. CONTENT-BASED EXAMPLE
    print("\n" + "=" * 50)
    print("1. CONTENT-BASED FILTERING")
    print("=" * 50)
    
    cosine_sim, movies_df = build_content_based_model(movies.copy())
    recs = get_content_recommendations('Toy Story', cosine_sim, movies_df, 5)
    print("\nMovies similar to 'Toy Story':")
    for _, row in recs.iterrows():
        print(f"  - {row['title']} ({row['genres']})")
    
    # 2. COLLABORATIVE EXAMPLE
    print("\n" + "=" * 50)
    print("2. COLLABORATIVE FILTERING")
    print("=" * 50)
    
    user_item_matrix, user_similarity_df = build_user_user_cf(ratings, n_users=200)
    recs = get_user_recommendations(1, user_item_matrix, user_similarity_df, 5)
    print(f"\nRecommendations for User 1:")
    for movie_id in recs:
        title = movies[movies['movie_id'] == movie_id]['title'].values
        if len(title) > 0:
            print(f"  - {title[0]}")
    
    # 3. MATRIX FACTORIZATION EXAMPLE
    print("\n" + "=" * 50)
    print("3. MATRIX FACTORIZATION (SVD)")
    print("=" * 50)
    
    model_file = 'svd_model.pkl'
    
    # Check if model already exists
    if os.path.exists(model_file):
        print("Loading pre-trained model...")
        model = load_model(model_file)
        # Note: You'll need trainset for recommendations
        # For now, we'll retrain to get trainset
        ratings_subset = ratings.head(100000)
        model, trainset, rmse = build_svd_model(ratings_subset)
    else:
        print("Training new SVD model...")
        ratings_subset = ratings.head(100000)  # Use subset for speed
        model, trainset, rmse = build_svd_model(ratings_subset)
        save_model(model, model_file)
    
    print(f"\nModel RMSE: {rmse:.4f}")
    
    recs = get_svd_recommendations(1, model, trainset, movies, 5)
    print(f"\nRecommendations for User 1:")
    for title, score in recs:
        print(f"  - {title} (predicted: {score:.2f})")
    
    print("\n" + "=" * 50)
    print("Done! All three recommendation approaches work.")
    print("=" * 50)

if __name__ == "__main__":
    main()