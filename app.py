import streamlit as st
from src.recommender import load_data, build_similarity, build_collab_model, hybrid_recommendations

# load everything
movies, ratings = load_data()
cosine_sim, indices = build_similarity(movies)

st.title("🎬 Movie Recommender")
st.write("Enter your user ID and a movie you like to get personalized recommendations.")

# user input
user_id = st.number_input("Your user ID:", min_value=1, max_value=668, step=1, value=1)

movie_list = movies['title'].tolist()
selected_movie = st.selectbox(
    "Search for a movie you like:",
    movie_list,
    index=None,
    placeholder="Type a movie name..."
)

n = st.slider("How many recommendations?", min_value=5, max_value=20, value=10)

if st.button("Recommend"):
    if selected_movie is None:
        st.warning("Please select a movie first!")
    else:
        with st.spinner("Finding recommendations..."):
            model = build_collab_model(ratings)
            recommendations = hybrid_recommendations(
                user_id, selected_movie, movies, cosine_sim, indices, model, n
            )
        st.subheader("Movies you might like:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")