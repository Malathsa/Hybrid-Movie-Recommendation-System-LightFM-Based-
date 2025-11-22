import pandas as pd

def major_minor_ratio_genres(movies_df):
    movies_df = movies_df.copy()
    movies_df["genres"] = movies_df["genres"].str.split("|")
    
    genre_counts = {}
    for genre_list in movies_df["genres"]:
        if isinstance(genre_list, list):
            for g in genre_list:
                if g != "(no genres listed)":
                    genre_counts[g] = genre_counts.get(g, 0) + 1
    
    genre_counts = pd.Series(genre_counts)
    
    major = genre_counts.max()
    minor = genre_counts.min()
    ratio = major / minor
    
    print("\n" + "="*60)
    print("📊 BIAS DETECTION - Major-Minor Ratio")
    print("="*60)
    print("🎭 Genre Counts (Top 10):")
    print(genre_counts.sort_values(ascending=False).head(10))
    print("\n📉 Genre Counts (Bottom 10):")
    print(genre_counts.sort_values().head(10))
    
    print(f"\n🟦 Major–Minor Ratio: {ratio:.2f}")
    print(f"Most common genre has: {major} movies")
    print(f"Least common genre has: {minor} movies")
    
    if ratio > 10:
        print("⚠️ Warning: Strong genre imbalance detected!")
    else:
        print("✅ Genre distribution is reasonably balanced.")
    
    return ratio, genre_counts

ratio, genre_counts = major_minor_ratio_genres(movies)
