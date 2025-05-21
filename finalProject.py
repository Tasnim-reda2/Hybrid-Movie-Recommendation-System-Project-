import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split

# ------------------ Data Loading & Preprocessing ------------------

# Load datasets
movies = pd.read_csv(r"C:/Users/Acer/Downloads/movies.csv")
ratings = pd.read_csv(r"C:/Users/Acer/Downloads/ratings.csv")

# Fill missing genres
movies['genres'] = movies['genres'].fillna('')

# Merge ratings with movie info
data = pd.merge(ratings, movies, on='movieId')

# TF-IDF Vectorizer for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index of movies
movie_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# ------------------ Collaborative Filtering with Surprise ------------------

reader = Reader(rating_scale=(0.5, 5.0))
data_surprise = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = surprise_train_test_split(data_surprise, test_size=0.2, random_state=42)

svd_model = SVD()
svd_model.fit(trainset)

# ------------------ Evaluation Metrics ------------------
predictions = svd_model.test(testset)
rmse = accuracy.rmse(predictions, verbose=False)
mae = accuracy.mae(predictions, verbose=False)

# ------------------ Helper Functions ------------------

def predict_rating(user_id, movie_id):
    try:
        return svd_model.predict(user_id, movie_id).est
    except:
        return 0

def get_content_recommendations(title, top_n=10):
    idx = movie_indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    rec_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[rec_indices].tolist()

def hybrid_recommendations(user_id, title, top_n=10, alpha=0.5):
    content_recs = get_content_recommendations(title, top_n * 2)
    movie_ids = movies[movies['title'].isin(content_recs)]['movieId'].tolist()

    scores = []
    for movie_id in movie_ids:
        rating_score = predict_rating(user_id, movie_id)
        movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
        content_score = cosine_sim[movie_indices[title]][movie_indices[movie_title]]
        final_score = alpha * rating_score + (1 - alpha) * content_score
        scores.append((movie_id, final_score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    recommended_ids = [s[0] for s in scores]
    return movies[movies['movieId'].isin(recommended_ids)]['title'].tolist()

# ------------------ Streamlit UI ------------------

st.title("🎬 Hybrid Movie Recommendation System")

user_id = st.number_input("Enter User ID", min_value=1, max_value=ratings['userId'].max(), value=1)
movie_title = st.selectbox("Select a movie you like", sorted(movies['title'].unique()))
alpha = st.slider("Weight for Collaborative Filtering (0 = Content, 1 = CF)", 0.0, 1.0, 0.5)

if st.button("Get Recommendations"):
    recommendations = hybrid_recommendations(user_id, movie_title, top_n=10, alpha=alpha)
    if recommendations:
        st.subheader("Recommended Movies:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {str(rec)}")
    else:
        st.warning("No recommendations found. Try another movie.")

st.markdown("---")
st.subheader("📊 Model Evaluation")
st.write(f"**Root Mean Square Error (RMSE):** {rmse:.4f}")
st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
