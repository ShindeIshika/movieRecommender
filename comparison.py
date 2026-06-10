import pandas as pd
import matplotlib.pyplot as plt

# Create comparison dataframe
comparison_data = {
    'Approach': ['Content-Based', 'Collaborative', 'Matrix Factorization'],
    'Strengths': [
        'No cold-start for new items,\nInterpretable',
        'Serendipity,\nCaptures user patterns',
        'Highest accuracy,\nHandles sparsity well'
    ],
    'Weaknesses': [
        'No serendipity,\nGenre-only',
        'Cold-start problem,\nScalability issues',
        'Less interpretable,\nRequires more data'
    ],
    'Best For': [
        'New users,\nSimilar items',
        'Established users,\nNiche tastes',
        'Production systems,\nHigh accuracy'
    ]
}

df = pd.DataFrame(comparison_data)
print("\n" + "="*70)
print("ALGORITHM COMPARISON")
print("="*70)
print(df.to_string(index=False))

# Save to CSV
df.to_csv('algorithm_comparison.csv', index=False)
print("\nSaved to 'algorithm_comparison.csv'")