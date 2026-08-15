import streamlit as st
import requests
from src.recommender import load_data

# Only load movies data to populate the selectbox titles
movies, _ = load_data()

st.title("🎬 Movie Recommender")
st.write("Enter your user ID and a movie you like to get personalized recommendations.")

# User inputs
user_id = st.number_input("Your user ID:", min_value=1, max_value=668, step=1, value=1)

movie_list = movies['title'].tolist()
selected_movie = st.selectbox(
    "Search for a movie you like:",
    movie_list,
    index=None,
    placeholder="Type a movie name..."
)

n = st.slider("How many recommendations?", min_value=5, max_value=20, value=10)

# URL of our FastAPI backend
API_URL = "http://127.0.0.1:8000/recommend"

if st.button("Recommend"):
    if selected_movie is None:
        st.warning("Please select a movie first!")
    else:
        with st.spinner("Requesting recommendations from backend API..."):
            try:
                # Call the FastAPI server
                response = requests.post(
                    API_URL,
                    json={
                        "user_id": user_id,
                        "movie_title": selected_movie,
                        "num_recommendations": n
                    }
                )

                if response.status_code == 200:
                    recommendations = response.json()
                    st.subheader("Movies you might like:")
                    for i, movie in enumerate(recommendations, 1):
                        st.write(f"{i}. {movie}")
                else:
                    error_detail = response.json().get('detail', 'Unknown error occurred.')
                    st.error(f"Error from API ({response.status_code}): {error_detail}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the API server. Make sure your FastAPI backend is running on http://127.0.0.1:8000")