import pandas as pd
from scipy.sparse import coo_matrix
from lightfm.data import Dataset

def prepare_features_with_reweighting(genre_weights, reweight_strength=1.0):
    """
    Build item features with genre reweighting
    
    Args:
        genre_weights: Dictionary of genre weights
        reweight_strength: Controls the strength of reweighting (0=no reweight, 1=full reweight)
    """
    # Build interactions
    interactions, _ = dataset.build_interactions(
        [(row.userId, row.movieId) for row in positive.itertuples(index=False)]
    )
    
    # Build weighted item features - USE DICTIONARY FORMAT
    item_features_list = []
    
    for movie_id in all_items:
        movie_genres = movies[movies['movieId'] == movie_id]['genres']
        
        weighted_features = {}  # Dictionary for weighted features
        if len(movie_genres) > 0:
            genres_str = movie_genres.iloc[0]
            if isinstance(genres_str, list):
                for genre in genres_str:
                    if genre != "(no genres listed)":
                        # Apply genre weight
                        base_weight = genre_weights.get(genre, 1.0)
                        # Apply reweight strength
                        weight = 1.0 + (base_weight - 1.0) * reweight_strength
                        # CORRECT FORMAT: dictionary {feature_name: weight}
                        weighted_features[genre] = weight
        
        item_features_list.append((movie_id, weighted_features))
    
    print(f"✅ Built weighted features for {len(item_features_list)} movies")
    
    item_features_matrix = dataset.build_item_features(item_features_list)
    
    return interactions, item_features_matrix

# Build features with reweighting
interactions, item_features = prepare_features_with_reweighting(
    genre_weights, 
    reweight_strength=0.7  # Adjust this value: 0=no reweight, 1=full reweight
)
