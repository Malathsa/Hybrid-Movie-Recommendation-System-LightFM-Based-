import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time


# PAGE CONFIG (must be first command)
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    layout="wide",
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');

/* Apply font */
html, body, [class*="css"], .stApp, input, textarea, select, option {
    font-family: 'Montserrat', sans-serif !important;
}

/* ---------- TITLES: ALL RED ---------- */
h1, h2, h3, h4, h5, h6 {
    color: #cc0000 !important;
    font-weight: 700 !important;
}

/* ---------- BUTTONS: Transparent + Red Border + Red Hover ---------- */
div.stButton > button {
    background-color: transparent !important;      /* no fill */
    color: #000000 !important;                     /* black text */
    border: 2px solid #cc0000 !important;          /* red border */
    padding: 8px 22px !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    transition: 0.25s ease-in-out;
}

div.stButton > button:hover {
    background-color: #cc0000 !important;          /* fill red */
    color: #ffffff !important;                     /* white text */
    cursor: pointer;
}

/* ---------- INPUTS: Transparent + Red Border ---------- */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[role="combobox"],
.stMultiSelect div[role="combobox"],
div[data-baseweb="select"] > div {
    background-color: transparent !important;      /* no background */
    color: #000000 !important;                     /* black text */
    border: 2px solid #cc0000 !important;          /* red border */
    border-radius: 6px !important;
    padding: 6px 10px !important;
}

/* Dropdown menu */
ul[role="listbox"] {
    background-color: #ffffff !important;
    border: 1px solid #cc0000 !important;
    color: #000000 !important;
}

ul[role="listbox"] > li:hover {
    background-color: #ffe5e5 !important;
    color: #cc0000 !important;
}

/* Table */
table {
    background-color: #ffffff !important;
    color: #000000 !important;
}
thead th {
    background-color: #f0f0f0 !important;
    color: #cc0000 !important;
    font-weight: 700 !important;
}

</style>
""", unsafe_allow_html=True)


# 1) LOAD MODEL + DATA

@st.cache_resource
def load_model():
    with open("lightfm_hybrid_checkpoint.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_dataset():
    with open("dataset_mapping.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_movies():
    return pd.read_csv("movies.csv")

@st.cache_data
def load_ratings():
    return pd.read_csv("ratings.csv")

@st.cache_data
def load_user_ids():
    ratings = load_ratings()
    return sorted(ratings["userId"].unique())


model_hybrid = load_model()
dataset = load_dataset()
movies = load_movies()
all_users = load_user_ids()


# 2) RECOMMENDER FUNCTIONS

def get_hybrid_recommendations_for_user(user_id, n_items=10):
    """Recommendations for an existing user (has User ID + history)."""
    user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
    user_internal_id = user_id_map.get(user_id)

    if user_internal_id is None:
        return pd.DataFrame(columns=["movieId", "title", "genres", "score"])

    item_internal_ids = np.array(list(item_id_map.values()), dtype=np.int32)
    original_movie_ids = np.array(list(item_id_map.keys()))
    user_ids_arr = np.full_like(item_internal_ids, user_internal_id)

    scores = model_hybrid.predict(user_ids_arr, item_internal_ids)

    recs = pd.DataFrame({
        "movieId": original_movie_ids,
        "score": scores
    }).merge(movies, on="movieId", how="left")

    recs = recs.sort_values("score", ascending=False).head(n_items)
    return recs.reset_index(drop=True)


def get_cold_start_recommendations(selected_genres, min_rating, popular_only, n_items=10):
    """Cold-start recommendations for a brand-new user."""
    ratings = load_ratings()

    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "num_ratings", "mean": "avg_rating"})
    )

    df = movies.merge(stats, on="movieId", how="left")
    df["num_ratings"] = df["num_ratings"].fillna(0)
    df["avg_rating"] = df["avg_rating"].fillna(0)

    # Filter by preferred genres
    if selected_genres:
        def match(gen):
            if pd.isna(gen):
                return False
            glist = [g.strip() for g in str(gen).split("|")]
            return any(g in glist for g in selected_genres)
        df = df[df["genres"].apply(match)]

    # Filter by minimum rating
    df = df[df["avg_rating"] >= min_rating]

    # Filter by popularity
    if popular_only:
        df = df[df["num_ratings"] >= 50]

    df = df.sort_values(["avg_rating", "num_ratings"], ascending=[False, False])
    return df.head(n_items).reset_index(drop=True)


# 3) UI PAGES

def page_known_user():
    st.header("Personalized Recommendations (Existing User)")
    st.write(
        "Already part of our system? Pick your User ID and let the recommender "
        "serve a fresh batch of movies tailored just for you"
    )

    selected_user = st.selectbox("Choose your User ID:", all_users)
    n_items = st.slider("How many recommendations do you want?", 5, 30, 10)

    if st.button("Show My Recommendations"):
        start = time.time()
        recs = get_hybrid_recommendations_for_user(selected_user, n_items)
        end = time.time()

        st.success(f"Generated in {end - start:.3f} seconds")

        if recs.empty:
            st.warning("No recommendations were found for this user. Try a different ID.")
        else:
            st.subheader("Your Top Movie Picks")
            st.dataframe(recs, use_container_width=True)


def page_cold_start():
    st.header("New Here? Let’s Discover Your Movie Taste!")
    st.write(
        "Answer these 3 quick questions, and we'll instantly build a custom movie list "
        "that matches your vibe"
    )

    # Q1: Favorite genres
    all_genres = sorted(
        {g.strip() for gs in movies["genres"].dropna() for g in str(gs).split("|")}
    )
    selected_genres = st.multiselect(
        "1) Which genres excite you the most?",
        options=all_genres
    )

    # Q2: Minimum rating
    min_rating = st.slider(
        "2) What’s the lowest average rating you're comfortable with?",
        0.0, 5.0, 3.5, 0.5
    )

    # Q3: Popularity preference
    popularity_pref = st.select_slider(
        "3) What kind of movies do you prefer?",
        options=["All movies (including hidden gems)", "Popular movies only"],
        value="Popular movies only"
    )
    popular_only = popularity_pref == "Popular movies only"

    n_items = st.slider("Number of recommendations:", 5, 30, 10)

    if st.button("Show My Movie Matches"):
        recs = get_cold_start_recommendations(
            selected_genres,
            min_rating,
            popular_only,
            n_items
        )

        if recs.empty:
            st.warning("No movies matched your filters. Try relaxing them a bit")
        else:
            st.success("Here are your personalized movie matches!")
            st.dataframe(recs, use_container_width=True)


# 4) MODE SELECTION WINDOW

def mode_selection_window():
    """Small centered window that appears first to choose the mode."""
    st.title("Hybrid Movie Recommendation System")
    st.caption("AI-powered movie suggestions tailored just for you")
    # Center the card using columns
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.markdown('<div class="mode-card">', unsafe_allow_html=True)
        st.markdown("## Pick a mode to get started")
        st.write(
            "Are you totally new or do you already have a User ID in our system?\n\n"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("New User (Cold Start)", key="btn_new_user"):
                st.session_state["mode"] = "cold_start"
                st.rerun()
        with col2:
            if st.button("Existing User (User ID)", key="btn_existing_user"):
                st.session_state["mode"] = "known_user"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# 5) MAIN CONTROLLER

def main():
    # If mode not chosen yet, show the small window first
    if "mode" not in st.session_state:
        mode_selection_window()
        return

    # Optional: small control in sidebar to switch mode
    with st.sidebar:
        st.subheader("Mode")
        if st.button("Change mode"):
            del st.session_state["mode"]
            st.rerun()

    # After choosing mode, go to the corresponding page
    if st.session_state["mode"] == "cold_start":
        page_cold_start()
    else:
        page_known_user()


if __name__ == "__main__":
    main()