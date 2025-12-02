import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

st.set_page_config(
    page_title="Hybrid Movie Recommender (LightFM)",
    page_icon="🎥",
    layout="wide",
)

# ============================
# 1) تحميل النموذج والبيانات
# ============================

@st.cache_resource
def load_model():
    with open("lightfm_hybrid_checkpoint.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_dataset():
    with open("dataset_mapping.pkl", "rb") as f:
        dataset = pickle.load(f)
    return dataset

@st.cache_data
def load_movies():
    # نفس movies.csv اللي استخدمتيه في التدريب
    return pd.read_csv("movies.csv")

@st.cache_data
def load_user_ids():
    # نستخدم ratings.csv بس عشان نجيب كل الـ userId المتاحة
    ratings = pd.read_csv("ratings.csv")
    return sorted(ratings["userId"].unique())

model_hybrid = load_model()
dataset = load_dataset()
movies = load_movies()
all_users = load_user_ids()

# ============================
# 2) دالة التوصيات
# ============================

def get_hybrid_recommendations_for_user(user_id, n_items=10):
    """
    ترجع DataFrame لأفضل n_items توصيات للمستخدم المحدد.
    تعتمد على model_hybrid + dataset + movies.
    """
    # نستخرج الخرائط من الـ dataset
    user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()

    # نحول userId الحقيقي إلى internal id
    user_internal_id = user_id_map.get(user_id)
    if user_internal_id is None:
        return pd.DataFrame(columns=["movieId", "title", "genres", "score"])

    # internal ids للأفلام + movieId الأصلية
    item_internal_ids = np.array(list(item_id_map.values()), dtype=np.int32)
    original_movie_ids = np.array(list(item_id_map.keys()))

    # نكرر user_internal_id بعدد الأفلام
    user_ids_arr = np.full_like(item_internal_ids, user_internal_id)

    # نتنبأ بالسكورات باستخدام النموذج الهجين
    scores = model_hybrid.predict(user_ids_arr, item_internal_ids)

    # نبني DataFrame بالتوصيات
    recs = pd.DataFrame({
        "movieId": original_movie_ids,
        "score": scores
    })

    # نربط مع جدول الأفلام للحصول على العناوين والأنواع
    recs = recs.merge(movies, on="movieId", how="left")

    # ترتيب تنازلي وأخذ أعلى n_items
    recs = recs.sort_values("score", ascending=False).head(n_items)

    # اختيار الأعمدة المهمة
    recs = recs[["movieId", "title", "genres", "score"]].reset_index(drop=True)
    return recs

# ============================
# 3) واجهة Streamlit
# ============================

def main():

    st.title("🎥 Hybrid Movie Recommendation System")
    st.caption("LightFM-based hybrid recommender – Streamlit Interface")

    # اختيار User ID
    selected_user = st.selectbox(
        "اختر User ID:",
        options=all_users,
        index=0
    )

    # عدد التوصيات
    n_items = st.slider("عدد التوصيات:", min_value=5, max_value=30, value=10, step=1)

    if st.button("احصل على التوصيات", type="primary"):
        start_time = time.time()
        recs = get_hybrid_recommendations_for_user(selected_user, n_items=n_items)
        elapsed = time.time() - start_time

        if recs.empty:
            st.warning("لم يتم العثور على توصيات لهذا المستخدم.")
        else:
            st.success(f"⏱️ تم توليد التوصيات في {elapsed:.3f} ثانية")
            st.subheader(f"🎬 أفضل {n_items} فيلم للمستخدم {selected_user}")
            st.dataframe(recs, use_container_width=True)


if __name__ == "__main__":
    main()