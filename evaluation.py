import numpy as np
import pandas as pd

def precision_at_k(recommended, actual, k):
    """Precision@K"""
    if k > len(recommended):
        k = len(recommended)
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(actual)) / k if k > 0 else 0

def recall_at_k(recommended, actual, k):
    """Recall@K"""
    if k > len(recommended):
        k = len(recommended)
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(actual)) / len(actual) if len(actual) > 0 else 0

def ndcg_at_k(recommended, actual, k):
    """NDCG@K"""
    if k > len(recommended):
        k = len(recommended)
    
    # Create relevance scores
    relevance = [1 if item in actual else 0 for item in recommended[:k]]
    
    if sum(relevance) == 0:
        return 0.0
    
    # DCG
    dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance)])
    
    # IDCG (ideal: all relevant at top)
    ideal_relevance = [1] * min(len(actual), k) + [0] * max(0, k - len(actual))
    idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance)])
    
    return dcg / idcg if idcg > 0 else 0

def evaluate_recommendations(recommendation_func, test_users, actual_by_user, k=10):
    """Evaluate a recommendation function"""
    precisions = []
    recalls = []
    ndcgs = []
    
    for user_id in test_users:
        if user_id not in actual_by_user:
            continue
            
        actual_movies = actual_by_user[user_id]
        if len(actual_movies) == 0:
            continue
        
        # Get recommendations for this user
        recommended = recommendation_func(user_id)
        if not recommended:
            continue
            
        precisions.append(precision_at_k(recommended, actual_movies, k))
        recalls.append(recall_at_k(recommended, actual_movies, k))
        ndcgs.append(ndcg_at_k(recommended, actual_movies, k))
    
    return {
        'Precision@{}'.format(k): np.mean(precisions),
        'Recall@{}'.format(k): np.mean(recalls),
        'NDCG@{}'.format(k): np.mean(ndcgs)
    }

if __name__ == "__main__":
    print("Evaluation module loaded")