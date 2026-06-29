import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# LOAD & MERGE DATA
# ---------------------------------------------------------
anime_data = pd.read_csv("anime.csv")
rating_data = pd.read_csv("rating.csv")

rating_data = rating_data[rating_data['rating'] != -1]

anime_fulldata = pd.merge(
    anime_data,
    rating_data,
    on='anime_id',
    suffixes=['', '_user']
)

anime_fulldata = anime_fulldata.rename(
    columns={'name': 'anime_title', 'rating_user': 'user_rating'}
)

# ---------------------------------------------------------
# PREPARE METADATA FOR RECOMMENDER
# ---------------------------------------------------------
mean_user_rating = anime_fulldata.groupby('anime_id')['user_rating'].mean().reset_index()
mean_user_rating.rename(columns={'user_rating': 'user_mean_rating'}, inplace=True)

anime_meta = anime_data.merge(mean_user_rating, on='anime_id', how='left')
anime_meta['user_mean_rating'] = anime_meta['user_mean_rating'].fillna(anime_meta['rating'])

# Convert all columns used in text_features to string safely
anime_meta['genre'] = anime_meta['genre'].astype(str).fillna('')
anime_meta['type'] = anime_meta['type'].astype(str).fillna('')
anime_meta['user_mean_rating'] = anime_meta['user_mean_rating'].astype(str).fillna('')

# Replace literal "nan" strings created by astype(str)
anime_meta['genre'] = anime_meta['genre'].replace("nan", "")
anime_meta['type'] = anime_meta['type'].replace("nan", "")
anime_meta['user_mean_rating'] = anime_meta['user_mean_rating'].replace("nan", "")

# Build text_features
anime_meta['text_features'] = (
    anime_meta['genre'].fillna('') + ' ' +
    anime_meta['type'].fillna('') + ' ' +
    anime_meta['user_mean_rating'].fillna('')
)

# Final cleanup: remove any accidental NaN or None
anime_meta['text_features'] = anime_meta['text_features'].fillna('').astype(str)

# Verify zero NaN
print("Remaining NaN in text_features:", anime_meta['text_features'].isna().sum())

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(anime_meta['text_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(anime_meta.index, index=anime_meta['name']).drop_duplicates()

def recommend(title, top_n=10):
    if title not in indices.index:
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    anime_indices = [i[0] for i in sim_scores]
    return anime_meta.loc[
        anime_indices,
        ['anime_id', 'name', 'genre', 'type', 'rating', 'user_mean_rating', 'episodes', 'members']
    ]

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Anime Dashboard", layout="wide")

st.title("🎌 Anime Analytics & Recommendation Dashboard")

tabs = st.tabs(["🤖 Recommender", "🔍 Explore"])

# ---------------------------------------------------------
# TAB 1 — EDA
# ---------------------------------------------------------
with tabs[0]:
    st.header("🤖 Anime Recommendation System")

    selected_title = st.selectbox("Select an Anime", anime_meta['name'].unique())

    if st.button("Get Recommendations"):
        recs = recommend(selected_title, 10)
        st.subheader("Recommended Anime")
        st.write(recs)

# ---------------------------------------------------------
# TAB 4 — EXPLORE
# ---------------------------------------------------------
with tabs[1]:
    st.header("🔍 Explore Anime")

    title = st.text_input("Write something", key="user_text")

    if title != "":
        results = anime_meta[anime_meta['name'].str.contains(title, case=False)]

        st.write(results[['anime_id', 'name', 'genre', 'type', 'rating', 'episodes', 'members']])

    

        
