import pandas as pd
import numpy as np
from surprise import SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    """Combines content-based and collaborative filtering"""
    
    def __init__(self, ratings_df, movies_df):
        self.ratings = ratings_df
        self.movies = movies_df
        self.alpha = 0.5  # Weight for collaborative part
        
        # Build content-based model
        print("Building content-based model...")
        tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
        genre_matrix = tfidf.fit_transform(movies_df['genres'].fillna(''))
        self.content_sim = cosine_similarity(genre_matrix)
        
        # Build collaborative model
        print("Building collaborative model...")
        from surprise import Dataset, Reader
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.cf_model = SVD(n_factors=100, random_state=42)
        self.cf_model.fit(trainset)
        
    def recommend(self, user_id, movie_id=None, n=10):
        """Hybrid recommendation"""
        if movie_id:
            # For 'similar movies' functionality
            content_scores = self.content_sim[movie_id]
            return self._get_top_movies(content_scores, n)
        else:
            # For user recommendations
            all_movies = set(self.movies['movie_id'])
            user_rated = set(self.ratings[self.ratings['user_id'] == user_id]['movie_id'])
            candidates = all_movies - user_rated
            
            scores = []
            for movie in candidates:
                # Collaborative score
                cf_score = self.cf_model.predict(user_id, movie).est
                # Content score (based on user's rated movies)
                content_score = self._get_content_score(user_id, movie)
                # Hybrid score
                hybrid_score = self.alpha * cf_score + (1 - self.alpha) * content_score
                scores.append((movie, hybrid_score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return self._get_movie_titles([m for m, _ in scores[:n]])
    
    def _get_content_score(self, user_id, movie_id):
        """Average similarity to user's rated movies"""
        user_movies = self.ratings[self.ratings['user_id'] == user_id]['movie_id'].values
        if len(user_movies) == 0:
            return 2.5  # Neutral score
        
        similarities = []
        for um in user_movies[:20]:  # Limit for speed
            if um < len(self.content_sim) and movie_id < len(self.content_sim):
                similarities.append(self.content_sim[um, movie_id])
        
        return np.mean(similarities) * 5 if similarities else 2.5
    
    def _get_top_movies(self, scores, n):
        indices = np.argsort(scores)[::-1][:n]
        return self._get_movie_titles(indices)
    
    def _get_movie_titles(self, movie_ids):
        titles = []
        for mid in movie_ids:
            title = self.movies[self.movies['movie_id'] == mid]['title'].values
            if len(title) > 0:
                titles.append(title[0])
        return titles

# Test it
print("Loading data...")
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

# Use subset for speed
ratings_sample = ratings.head(100000)

print("\nInitializing hybrid recommender...")
hybrid = HybridRecommender(ratings_sample, movies)

print("\nHybrid Recommendations for User 1:")
recs = hybrid.recommend(user_id=1, n=10)
for i, movie in enumerate(recs[:10], 1):
    print(f"  {i}. {movie}")