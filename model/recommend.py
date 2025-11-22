import numpy as np

def sample_recommendations(model, user_ids, item_features, dataset, n_items=5):
    """Generate movie recommendations using hybrid model with proper ID mapping"""
    
    user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
    
    available_movies = list(item_id_map.values())
    print(f"📊 Hybrid predictions for {len(available_movies)} movies")
    
    for user_id in user_ids:
        print(f"\n🔍 Generating recommendations for user {user_id}...")
        user_start = time.time()
        
        user_internal_id = user_id_map.get(user_id)
        if user_internal_id is None:
            print(f"❌ User {user_id} not found in dataset")
            continue
        
        scores = []
        original_movie_ids = []
        
        for movie_internal_id in available_movies:
            score = model.predict(
                np.array([user_internal_id], dtype=np.int32), 
                np.array([movie_internal_id], dtype=np.int32),
                item_features=item_features,
                num_threads=1
            )[0]
            scores.append(score)
            
            original_id = [k for k, v in item_id_map.items() if v == movie_internal_id][0]
            original_movie_ids.append(original_id)
        
        scores = np.array(scores)
        
        # Get top recommendations
        top_indices = np.argsort(-scores)[:n_items]
        top_movies = []
        
        for idx in top_indices:
            original_movie_id = original_movie_ids[idx]
            movie_data = movies[movies['movieId'] == original_movie_id]
            if len(movie_data) > 0:
                title = movie_data['title'].values[0]
                genres = movie_data['genres'].values[0]
                top_movies.append((title, genres, scores[idx]))
        
        print(f"🎬 User {user_id} - Top {n_items} Hybrid Recommendations:")
        for i, (title, genres, score) in enumerate(top_movies, 1):
            print(f"   {i}. {title}")
            print(f"      ⭐ Score: {score:.3f} | 🎭 {genres}")
        user_elapsed = time.time() - user_start
        print(f"⏱ Response time for user {user_id}: {user_elapsed:.3f} seconds")


# Show recommendations for first 3 users
print("\n" + "="*60)
print("🎯 HYBRID RECOMMENDATIONS (IDF-Weighted)")
print("="*60)
sample_users = list(all_users)[:3]
sample_recommendations(model_hybrid, sample_users, item_features, dataset)
