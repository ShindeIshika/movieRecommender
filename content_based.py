import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def build_content_based_model(movies_df):
    """Build content-based recommendation model"""
    # Fill NaN genres
    movies_df['genres'] = movies_df['genres'].fillna('')
    
    # Convert genres to TF-IDF features
    tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
    genre_matrix = tfidf.fit_transform(movies_df['genres'])
    
    # Compute cosine similarity between all movies
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
    
    return cosine_sim, movies_df

def get_content_recommendations(movie_title, cosine_sim, movies_df, top_n=10):
    """Get top N similar movies to given movie"""
    # Find movie index
    matching = movies_df[movies_df['title'].str.contains(movie_title, case=False)]
    if len(matching) == 0:
        return f"Movie '{movie_title}' not found"
    
    idx = matching.index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N (excluding itself)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    return movies_df.iloc[movie_indices][['title', 'genres']]

if __name__ == "__main__":
    # Test the code
    print("Loading movies...")
    movies = pd.read_csv('data/movies.csv')
    print(f"Loaded {len(movies)} movies")
    
    print("\nBuilding content-based model...")
    cosine_sim, movies_df = build_content_based_model(movies)
    
    print("\nRecommendations for 'Toy Story':")
    recs = get_content_recommendations('Toy Story', cosine_sim, movies_df)
    print(recs)