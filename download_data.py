import pandas as pd
import urllib.request
import zipfile
import os

# Create data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download MovieLens 1M
print("Downloading MovieLens dataset...")
url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
urllib.request.urlretrieve(url, "data/ml-1m.zip")

# Extract
print("Extracting...")
with zipfile.ZipFile("data/ml-1m.zip", "r") as zip_ref:
    zip_ref.extractall("data/")

# Load ratings
ratings = pd.read_csv(
    'data/ml-1m/ratings.dat',
    sep='::',
    engine='python',
    names=['user_id', 'movie_id', 'rating', 'timestamp']
)

# Load movies
movies = pd.read_csv(
    'data/ml-1m/movies.dat',
    sep='::',
    engine='python',
    names=['movie_id', 'title', 'genres'],
    encoding='latin-1'
)

# Save as CSV for easier loading later
ratings.to_csv('data/ratings.csv', index=False)
movies.to_csv('data/movies.csv', index=False)

print(f"Done! Loaded {len(ratings)} ratings and {len(movies)} movies")
print(f"Users: {ratings['user_id'].nunique()}")
print(f"Movies: {len(movies)}")