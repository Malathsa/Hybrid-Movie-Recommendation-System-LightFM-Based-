import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

movies['genres'] = movies['genres'].str.split('|')
all_genres = set()
for genre_list in movies['genres']:
    if isinstance(genre_list, list):
        for genre in genre_list:
            if genre != "(no genres listed)":
                all_genres.add(genre)

top_tags = tags['tag'].value_counts().head(50).index.tolist()
print(f"\nTotal unique genres: {len(all_genres)}")
print(f"Top 50 tags: {top_tags[:10]}...")

# ===== 6) Prepare Dataset =====
RATING_THRESHOLD = 4.0
positive = ratings[ratings["rating"] >= RATING_THRESHOLD].copy()

print(f"\n{'='*60}")
print(f"DATASET PREPARATION")
print(f"{'='*60}")
print(f"Total ratings: {len(ratings)}")
print(f"Positive ratings (>= {RATING_THRESHOLD}): {len(positive)}")

all_users = ratings["userId"].unique()
all_items = ratings["movieId"].unique()

print(f"Unique users: {len(all_users)}")
print(f"Unique movies in ratings: {len(all_items)}")

movies_in_ratings = movies[movies['movieId'].isin(all_items)]
print(f"Movies common to both ratings and movies: {len(movies_in_ratings)}")

# Create Dataset
dataset = Dataset()
dataset.fit(
    users=all_users,
    items=all_items,
    item_features=list(all_genres) + top_tags
)
