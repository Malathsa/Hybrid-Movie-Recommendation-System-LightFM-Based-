import pandas as pd
from sklearn.model_selection import train_test_split

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
tags = pd.read_csv("tags.csv")

print(f"Ratings data: {ratings.shape}")
print(f"Movies data: {movies.shape}")
print(f"Tags data: {tags.shape}")
