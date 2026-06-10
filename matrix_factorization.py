import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import joblib
import os

def build_svd_model(ratings_df):
    """Matrix factorization using SVD"""
    # Prepare data for Surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        ratings_df[['user_id', 'movie_id', 'rating']], 
        reader
    )
    
    # Train-test split
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Train SVD model
    print("Training SVD model...")
    model = SVD(n_factors=100, n_epochs=20, random_state=42)
    model.fit(trainset)
    
    # Evaluate
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    
    return model, trainset, rmse

def get_svd_recommendations(user_id, model, trainset, movies_df, n=10):
    """Get top N recommendations for a user"""
    # Get all movie IDs
    all_movies = set(movies_df['movie_id'])
    
    # Get movies user has rated (handle if user not in trainset)
    try:
        inner_user_id = trainset.to_inner_uid(user_id)
        user_rated = [item for item in trainset.ur[inner_user_id]]
        rated_movies = [trainset.to_raw_iid(iid) for (iid, _) in user_rated]
    except:
        rated_movies = []
    
    # Predict ratings for unrated movies
    predictions = []
    for movie in all_movies:
        if movie not in rated_movies:
            pred = model.predict(user_id, movie).est
            predictions.append((movie, pred))
    
    # Sort and get top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = predictions[:n]
    
    # Get movie titles
    recommendations = []
    for movie_id, score in top_movies:
        title = movies_df[movies_df['movie_id'] == movie_id]['title'].values
        if len(title) > 0:
            recommendations.append((title[0], score))
    
    return recommendations

def save_model(model, filepath='svd_model.pkl'):
    """Save trained model to disk"""
    joblib.dump(model, filepath)
    print(f"Model saved to '{filepath}'")

def load_model(filepath='svd_model.pkl'):
    """Load trained model from disk"""
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        print(f"Model loaded from '{filepath}'")
        return model
    else:
        print(f"No model found at '{filepath}'. Train a model first.")
        return None

if __name__ == "__main__":
    print("Loading data...")
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    
    # Use subset for faster testing
    ratings_subset = ratings.head(50000)
    
    model, trainset, rmse = build_svd_model(ratings_subset)
    print(f"\nModel RMSE: {rmse:.4f}")
    
    print("\nRecommendations for User 1:")
    recs = get_svd_recommendations(1, model, trainset, movies, n=5)
    for title, score in recs:
        print(f"  - {title} (predicted rating: {score:.2f})")
    
    # Save the model
    save_model(model)
    
    # Test loading the model
    loaded_model = load_model()
    if loaded_model:
        print("\nTesting loaded model...")
        recs = get_svd_recommendations(1, loaded_model, trainset, movies, n=3)
        print("Recommendations from loaded model:")
        for title, score in recs:
            print(f"  - {title} (predicted rating: {score:.2f})")