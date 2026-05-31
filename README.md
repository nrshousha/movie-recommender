# Movie Recommendation System

A hybrid movie recommender that combines content-based filtering and collaborative filtering to deliver personalized movie recommendations.
<img width="1440" height="900" alt="Screenshot 2026-05-31 at 6 12 16 PM" src="https://github.com/user-attachments/assets/40b66804-0430-4940-8c6e-97e5f896cac4" />

## What it does

Enter your user ID and a movie you like, and the system returns the top N most similar movies tailored to your personal taste.

## How it works

Two approaches are combined into a hybrid score:

**Content-Based Filtering**
- Movie genres are vectorized using TF-IDF
- Cosine similarity is calculated between all movies
- Movies with similar genres score higher

**Collaborative Filtering**
- User ratings are processed using SVD (Singular Value Decomposition)
- Hidden patterns between users and movies are discovered from 105k ratings
- Predicted ratings are generated for movies a user hasn't seen yet

**Hybrid Score**
```
final_score = content_similarity + predicted_rating
```
Best of both worlds — similar genres AND matched to your personal taste.

## Project Structure

```
project 4/
├── data/
│   ├── movies.csv            # 10,329 movies with genres
│   └── ratings.csv           # 105,339 ratings from 668 users
├── notebooks/
│   └── exploration.ipynb     # experimentation and testing
├── src/
│   └── recommender.py        # TF-IDF, SVD, and hybrid logic
├── app.py                    # Streamlit web interface
└── requirements.txt
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

- Python
- scikit-learn (TF-IDF, cosine similarity)
- scikit-surprise (SVD collaborative filtering)
- Streamlit
- Pandas
- NumPy

## Dataset

MovieLens dataset — 10,329 movies, 105,339 ratings, 668 users.
