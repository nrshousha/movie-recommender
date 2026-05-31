import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split
import streamlit as st

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(base_dir, '..', 'data', 'movies.csv')
    ratings_path = os.path.join(base_dir, '..', 'data', 'ratings.csv')
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    return movies, ratings

@st.cache_resource
def build_similarity(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    indices = pd.Series(movies.index, index=movies['title'])
    return cosine_sim, indices

@st.cache_resource
def build_collab_model(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, _ = surprise_split(data, test_size=0.2)
    model = SVD(random_state=42)
    model.fit(trainset)
    return model

def get_recommendations(title, movies, cosine_sim, indices, n=10):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

def hybrid_recommendations(user_id, title, movies, cosine_sim, indices, model, n=10):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    results = []
    for movie_idx, content_score in sim_scores:
        movie_id = movies.iloc[movie_idx]['movieId']
        collab_score = model.predict(uid=user_id, iid=movie_id).est
        hybrid_score = content_score + collab_score
        results.append((movie_idx, hybrid_score))
    results = sorted(results, key=lambda x: x[1], reverse=True)
    results = results[1:n+1]
    movie_indices = [i[0] for i in results]
    return movies['title'].iloc[movie_indices].tolist()