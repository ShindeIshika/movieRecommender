from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List

app = FastAPI(title="Movie Recommendation API", description="Multi-strategy recommendation engine")

# Load data and model at startup
print("Loading models...")
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')
model = joblib.load('svd_model.pkl')

class RecommendRequest(BaseModel):
    user_id: int
    n_recommendations: int = 10

class MovieRecommendation(BaseModel):
    title: str
    predicted_rating: float

@app.get("/")
def root():
    return {"message": "Movie Recommendation API", "algorithms": ["SVD", "Content-Based", "Collaborative"]}

@app.post("/recommend/svd", response_model=List[MovieRecommendation])
def recommend_svd(request: RecommendRequest):
    """Get recommendations using SVD matrix factorization"""
    user_id = request.user_id
    n = request.n_recommendations
    
    # Get all movies
    all_movies = set(movies['movie_id'])
    
    # Get user's rated movies
    user_rated = set(ratings[ratings['user_id'] == user_id]['movie_id'])
    candidates = all_movies - user_rated
    
    # Predict ratings
    predictions = []
    for movie in list(candidates)[:1000]:  # Limit for speed
        pred = model.predict(user_id, movie).est
        predictions.append((movie, pred))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Format response
    recommendations = []
    for movie_id, score in predictions[:n]:
        title = movies[movies['movie_id'] == movie_id]['title'].values
        if len(title) > 0:
            recommendations.append(MovieRecommendation(title=title[0], predicted_rating=round(score, 2)))
    
    return recommendations

@app.get("/movie/{movie_title}/similar")
def similar_movies(movie_title: str, n: int = 10):
    """Get similar movies using content-based filtering"""
    # Simple implementation
    matching = movies[movies['title'].str.contains(movie_title, case=False)]
    if len(matching) == 0:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    # Return top N similar by genre
    movie_genres = matching.iloc[0]['genres']
    similar = movies[movies['genres'].str.contains('|'.join(movie_genres.split('|')[:2]), na=False)]
    similar = similar[similar['movie_id'] != matching.iloc[0]['movie_id']].head(n)
    
    return [{"title": title, "genres": genres} for title, genres in zip(similar['title'], similar['genres'])]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)