import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def build_user_user_cf(ratings_df, n_users=500):
    """User-User collaborative filtering"""
    # Create user-item matrix
    user_item_matrix = ratings_df.pivot(
        index='user_id', 
        columns='movie_id', 
        values='rating'
    ).fillna(0)
    
    # Limit to top N users for memory
    user_item_matrix = user_item_matrix.head(n_users)
    
    # Compute user similarity
    print("Computing user similarity matrix...")
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    return user_item_matrix, user_similarity_df

def get_user_recommendations(user_id, user_item_matrix, user_similarity_df, n_recommendations=10):
    """Get recommendations for a user based on similar users"""
    if user_id not in user_similarity_df.columns:
        return f"User {user_id} not found"
    
    # Get similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]
    
    # Get movies liked by similar users
    similar_users_ratings = user_item_matrix.loc[similar_users.index]
    
    # Weight by similarity
    weighted_ratings = similar_users_ratings.T.dot(similar_users.values)
    
    # Get movies user hasn't rated
    user_rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recommendations = weighted_ratings[~weighted_ratings.index.isin(user_rated)]
    
    # Get top N
    top_movies = recommendations.sort_values(ascending=False).head(n_recommendations).index
    
    return top_movies.tolist()

if __name__ == "__main__":
    print("Loading ratings...")
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    
    print("Building collaborative filtering model...")
    user_item_matrix, user_similarity_df = build_user_user_cf(ratings)
    
    print("\nRecommendations for User 1:")
    recs = get_user_recommendations(1, user_item_matrix, user_similarity_df)
    
    # Get movie titles
    for movie_id in recs[:5]:
        title = movies[movies['movie_id'] == movie_id]['title'].values
        if len(title) > 0:
            print(f"  - {title[0]}")