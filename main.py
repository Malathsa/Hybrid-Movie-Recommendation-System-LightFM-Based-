from data.load_data import load_ratings, load_movies
from features.idf import compute_genre_idf
from features.genre_features import build_genre_matrix
from features.interactions import build_interaction_matrix
from model.train import train_model
from model.evaluate import evaluate
from model.recommend import recommend_items

def run():
    ratings = load_ratings("ratings.csv")
    movies = load_movies("movies.csv")

    idf = compute_genre_idf(movies)
    item_features = build_genre_matrix(movies, idf)
    interactions = build_interaction_matrix(ratings, "userId", "movieId", "rating")

    # train/test split (من خلية رقم 9)
    train, test = ...
    
    model = train_model(train, item_features)
    print(evaluate(model, train, test))

    print(recommend_items(model, 1, interactions, item_features))

if name == "__main__":
    run()
  
