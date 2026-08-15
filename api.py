from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Import your recommender logic
from src.recommender import (
    load_data,
    build_similarity,
    build_collab_model,
    hybrid_recommendations
)

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommender API",
    description="API for content-based, collaborative, and hybrid movie recommendations",
    version="1.0.0"
)

# Load data and models once at startup
movies, ratings = load_data()

# We call the functions directly to bypass streamlit's caching decorators
cosine_sim, indices = build_similarity(movies)
model = build_collab_model(ratings)


# Define the Request Body schema using Pydantic
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., ge=1, le=668, description="The ID of the user requesting recommendations")
    movie_title: str = Field(..., description="A movie title the user liked (e.g. 'Toy Story (1995)')")
    num_recommendations: int = Field(10, ge=1, le=50, description="Number of recommendations to return")


# Define the endpoint
@app.post("/recommend", response_model=List[str])
def recommend_movies(request: RecommendationRequest):
    # 1. Validate if movie title exists in our data
    matched_title = None
    if request.movie_title in indices:
        matched_title = request.movie_title
    else:
        # Fallback: find any movie that contains the query case-insensitively
        matches = movies[movies['title'].str.lower().str.contains(request.movie_title.lower(), na=False)]
        if not matches.empty:
            matched_title = matches.iloc[0]['title']

    if not matched_title:
        raise HTTPException(
            status_code=404,
            detail=f"Movie '{request.movie_title}' not found in the dataset. Please check the spelling."
        )

    try:
        # Generate recommendations using your hybrid model
        recommendations = hybrid_recommendations(
            user_id=request.user_id,
            title=matched_title,
            movies=movies,
            cosine_sim=cosine_sim,
            indices=indices,
            model=model,
            n=request.num_recommendations
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# root endpoint for health checks
@app.get("/")
def read_root():
    return {"message": "Movie Recommender API is up and running!"}
