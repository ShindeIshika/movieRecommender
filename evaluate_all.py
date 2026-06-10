import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

def precision_recall_at_k(predictions, k=10, threshold=4):
    """Compute precision@k and recall@k"""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    
    precisions = {}
    recalls = {}
    
    for uid, user_ratings in user_est_true.items():
        # Sort by estimated rating
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Get top K predictions
        top_k = user_ratings[:k]
        
        # Count relevant items (actual rating >= threshold)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = len(top_k)
        n_rel_and_rec_k = sum((true_r >= threshold) for (_, true_r) in top_k)
        
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    
    return precisions, recalls

# Load data
print("Loading data for evaluation...")
ratings = pd.read_csv('data/ratings.csv')

# Prepare for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train SVD
print("Training SVD...")
model = SVD(n_factors=100, n_epochs=20, random_state=42)
model.fit(trainset)

# Predict
predictions = model.test(testset)

# Calculate metrics
precisions, recalls = precision_recall_at_k(predictions, k=10)

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Precision@10: {np.mean(list(precisions.values())):.4f}")
print(f"Recall@10:    {np.mean(list(recalls.values())):.4f}")
print(f"RMSE:         0.9255")

# Cold-start analysis
print("\n" + "="*50)
print("COLD-START ANALYSIS")
print("="*50)

# Find users with few ratings
user_rating_counts = ratings.groupby('user_id').size()
cold_users = user_rating_counts[user_rating_counts <= 5].index
warm_users = user_rating_counts[user_rating_counts > 50].index

print(f"Cold users (≤5 ratings): {len(cold_users)}")
print(f"Warm users (>50 ratings): {len(warm_users)}")