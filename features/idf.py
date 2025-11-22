import pandas as pd
import numpy as np

print("\n" + "="*60)
print("⚖️  APPLYING IDF-BASED REWEIGHTING")
print("="*60)

total_movies = len(movies[movies['genres'] != "(no genres listed)"])
genre_weights = {}

for genre, count in genre_counts.items():
    idf_weight = np.log(total_movies / count)
    genre_weights[genre] = idf_weight

print("\nGenre Weights (IDF):")
sorted_weights = sorted(genre_weights.items(), key=lambda x: x[1], reverse=True)
print("\nTop 5 Highest Weights (rare genres - boosted):")
for genre, weight in sorted_weights[:5]:
    count = genre_counts[genre]
    print(f"   {genre:20s}: weight={weight:.3f} (appears in {count} movies)")

print("\nTop 5 Lowest Weights (common genres - reduced):")
for genre, weight in sorted_weights[-5:]:
    count = genre_counts[genre]
    print(f"   {genre:20s}: weight={weight:.3f} (appears in {count} movies)")
