import streamlit as st
import pandas as pd
import ast
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("🎬 Movie Recommendation System")
st.caption("Content-Based Recommendation using TMDB 5000 Dataset 🤖")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    posters = pd.read_csv("movies_with_posters.csv")

    # Merge movies + credits
    movies = movies.merge(credits, on="title")

    # Merge posters
    movies = movies.merge(
        posters[['movie_id', 'poster']],
        on='movie_id',
        how='left'
    )

    movies = movies[['movie_id', 'title', 'overview',
                     'genres', 'keywords', 'cast', 'crew',
                     'poster', 'vote_average', 'release_date']]

    movies.dropna(subset=['title', 'overview'], inplace=True)

    return movies

movies_data = load_data()

# ---------------- HELPER FUNCTIONS ----------------
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

def convert_cast(text):
    L = []
    for i in ast.literal_eval(text)[:3]:
        L.append(i['name'])
    return L

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

# ---------------- CLEAN DATA ----------------
movies_data['genres'] = movies_data['genres'].apply(convert)
movies_data['keywords'] = movies_data['keywords'].apply(convert)
movies_data['cast'] = movies_data['cast'].apply(convert_cast)
movies_data['crew'] = movies_data['crew'].apply(fetch_director)

movies_data['genres'] = movies_data['genres'].apply(lambda x: " ".join(x))
movies_data['keywords'] = movies_data['keywords'].apply(lambda x: " ".join(x))
movies_data['cast'] = movies_data['cast'].apply(lambda x: " ".join(x))
movies_data['crew'] = movies_data['crew'].apply(lambda x: " ".join(x))

# ---------------- CREATE TAGS ----------------
movies_data['tags'] = (
    movies_data['genres'] + " " +
    movies_data['keywords'] + " " +
    movies_data['overview'] + " " +
    movies_data['cast'] + " " +
    movies_data['crew']
)

# ---------------- VECTORIZE ----------------
@st.cache_resource
def compute_similarity(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(data)
    return cosine_similarity(vectors)

similarity = compute_similarity(movies_data['tags'])

# ---------------- RECOMMEND FUNCTION (UNCHANGED LOGIC) ----------------
def recommend(movie_name, top_n=8):
    list_of_titles = movies_data['title'].tolist()
    close_match = difflib.get_close_matches(movie_name, list_of_titles)

    if not close_match:
        return None

    matched_movie = close_match[0]
    index = movies_data[movies_data.title == matched_movie].index[0]

    similarity_score = list(enumerate(similarity[index]))
    sorted_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []
    for movie in sorted_movies[1:top_n + 1]:
        movie_index = movie[0]
        similarity_percent = round(movie[1] * 100, 2)
        recommendations.append((movies_data.iloc[movie_index], similarity_percent))

    return matched_movie, recommendations

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Filters")

    top_n = st.slider("🎯 Number of Recommendations", 4, 20, 8)
    min_rating = st.slider("⭐ Minimum Rating", 0.0, 10.0, 0.0, 0.5)

    movies_data["year"] = pd.to_datetime(
        movies_data["release_date"], errors="coerce"
    ).dt.year

    min_year = int(movies_data["year"].min())
    max_year = int(movies_data["year"].max())

    year_range = st.slider(
        "📅 Release Year",
        min_year,
        max_year,
        (2000, max_year)
    )

    all_genres = sorted(set(" ".join(movies_data['genres']).split()))
    selected_genres = st.multiselect("🎭 Genre Filter", all_genres)

# ---------------- MOVIE INPUT ----------------
movie_name = st.selectbox(
    "🎥 Select a Movie",
    options=[""] + movies_data['title'].tolist()
)

# ---------------- SHOW TOP RATED IF NO SEARCH ----------------
if not movie_name:

    st.subheader("🔥 Top Rated Movies")

    top_movies = movies_data.sort_values(
        by="vote_average",
        ascending=False
    )

    valid_top_movies = []

    for _, movie in top_movies.iterrows():
        if (
            pd.notna(movie["poster"]) and
            str(movie["poster"]).startswith("http") and
            movie["vote_average"] >= min_rating and
            year_range[0] <= movie["year"] <= year_range[1]
        ):
            if selected_genres:
                movie_genres = movie["genres"].split()
                if not any(g in movie_genres for g in selected_genres):
                    continue
            valid_top_movies.append(movie)

        if len(valid_top_movies) >= 12:
            break

    for i in range(0, len(valid_top_movies), 4):
        row = st.columns(4)
        for col, movie in zip(row, valid_top_movies[i:i+4]):
            with col:
                st.image(movie["poster"], use_container_width=True)
                st.markdown(f"### {movie['title']}")
                st.caption(f"⭐ Rating: {movie['vote_average']}")
                with st.expander("📖 Description"):
                    st.write(movie["overview"])

# ---------------- RECOMMEND BUTTON ----------------
if st.button("🔍 Recommend") and movie_name:

    result = recommend(movie_name, top_n)

    if result is None:
        st.error("Movie not found ❌")
    else:
        matched_movie, recommendations = result
        st.success(f"Movies similar to **{matched_movie}** 🎉")

        valid_movies = []

        for movie, score in recommendations:
            if (
                pd.notna(movie["poster"]) and
                str(movie["poster"]).startswith("http") and
                movie["vote_average"] >= min_rating and
                year_range[0] <= movie["year"] <= year_range[1]
            ):
                if selected_genres:
                    movie_genres = movie["genres"].split()
                    if not any(g in movie_genres for g in selected_genres):
                        continue
                valid_movies.append((movie, score))

        if not valid_movies:
            st.warning("No movies with available posters match the filters.")

        for i in range(0, len(valid_movies), 4):
            row = st.columns(4)
            for col, item in zip(row, valid_movies[i:i+4]):
                movie, score = item
                with col:
                    st.image(movie["poster"], use_container_width=True)
                    st.markdown(f"### {movie['title']}")
                    st.caption(f"Similarity: {score}%")
                    with st.expander("📖 Description"):
                        st.write(movie["overview"])
