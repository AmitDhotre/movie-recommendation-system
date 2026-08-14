import streamlit as st
import pandas as pd
import numpy as np
import ast
import difflib
import random
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
#  PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="CineMatch — Movie Recommendation Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
#  CUSTOM CSS — dark cinematic theme
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

    .stApp {
        background: radial-gradient(circle at top left, #1b1033 0%, #0d0d1a 45%, #05050a 100%);
        color: #eaeaf5;
    }

    /* ---------- Top navbar (Netflix/Prime style) ---------- */
    .navbar-wrap {
        display: flex;
        align-items: center;
        margin-top: 0.2rem;
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-top: 6px;
    }
    .nav-logo {
        font-size: 1.9rem;
        line-height: 1;
        filter: drop-shadow(0 0 8px rgba(255,77,109,0.6));
    }
    .nav-brand-text {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ff4d6d, #ffb703, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .navbar-underline {
        height: 3px;
        border-radius: 3px;
        background: linear-gradient(90deg, #ff4d6d, #ffb703, #7b2ff7, #2ec4b6);
        margin: 0.6rem 0 1.6rem 0;
        opacity: 0.85;
    }

    /* nav buttons in the header (keys start with nav_) — uniform size/shape */
    div[class*="st-key-nav_"] {
        height: 56px;
    }
    div[class*="st-key-nav_"] button {
        background: rgba(255,255,255,0.03) !important;
        box-shadow: none !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        color: #cfcfe8 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border-radius: 14px !important;
        height: 56px !important;
        width: 100% !important;
        min-width: 0 !important;
    }
    div[class*="st-key-nav_"] button:hover {
        border-color: rgba(255,77,109,0.5) !important;
        color: #ffffff !important;
    }
    div[class*="st-key-nav_"] button[kind="primary"] {
        background: linear-gradient(90deg, #ff4d6d, #7b2ff7) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 6px 18px rgba(255,77,109,0.35) !important;
    }


    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 14px 18px;
        border-radius: 16px;
        backdrop-filter: blur(6px);
    }
    div[data-testid="stMetricValue"] { color: #ff9f7d; }

    .movie-card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 10px 10px 14px 10px;
        margin-bottom: 18px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(255, 77, 109, 0.25);
        border-color: rgba(255, 77, 109, 0.4);
    }
    .movie-title {
        font-weight: 700;
        font-size: 1.0rem;
        margin: 10px 2px 2px 2px;
        color: #ffffff;
        line-height: 1.25rem;
        min-height: 2.5rem;
    }
    .badge-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 2px; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.74rem; font-weight: 600; }
    .badge-rating { background: linear-gradient(90deg, #ffb703, #fb8500); color: #201400; }
    .badge-match  { background: linear-gradient(90deg, #4dd8c9, #2ec4b6); color: #012320; }
    .badge-year   { background: rgba(255,255,255,0.12); color: #dcdcff; }
    .badge-genre  { background: rgba(123,47,247,0.25); color: #d9c8ff; border: 1px solid rgba(123,47,247,0.4); }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #150c28 0%, #0a0a14 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .stButton>button {
        background: linear-gradient(90deg, #ff4d6d, #7b2ff7);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.55rem 1.2rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        box-shadow: 0 6px 18px rgba(255,77,109,0.35);
        width: 100%;
    }
    .stButton>button:hover { filter: brightness(1.12); }

    .ghost-btn button {
        background: rgba(255,255,255,0.08) !important;
        box-shadow: none !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }

    div[data-baseweb="tab-list"] { gap: 6px; }
    button[data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px 10px 0 0 !important;
        padding: 8px 18px !important;
        color: #cfcfe8 !important;
    }
    button[aria-selected="true"] { background: rgba(255, 77, 109, 0.18) !important; color: #ffffff !important; }

    .detail-hero {
        border-radius: 22px;
        padding: 1.8rem;
        background: linear-gradient(135deg, rgba(123,47,247,0.25), rgba(255,77,109,0.15));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.4rem;
    }
    .detail-title { font-size: 2.2rem; font-weight: 800; color: white; margin-bottom: 0.2rem; }
    .detail-overview { font-size: 1.05rem; line-height: 1.6rem; color: #e4e4f5; margin-top: 0.8rem; }
    .section-title { font-size: 1.3rem; font-weight: 700; margin: 1.4rem 0 0.6rem 0; color: #ffe0d6; }

    footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
#  LOAD DATA
# =====================================================================
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    posters = pd.read_csv("movies_with_posters.csv")

    movies = movies.merge(credits, on="title")
    movies = movies.merge(posters[["movie_id", "poster"]], on="movie_id", how="left")

    movies = movies[[
        "movie_id", "title", "overview", "genres", "keywords",
        "cast", "crew", "poster", "vote_average", "vote_count",
        "popularity", "release_date",
    ]]

    movies.dropna(subset=["title", "overview"], inplace=True)
    movies.reset_index(drop=True, inplace=True)
    return movies


movies_data_raw = load_data()


# =====================================================================
#  HELPERS
# =====================================================================
def convert(text):
    try:
        return [i["name"] for i in ast.literal_eval(text)]
    except Exception:
        return []


def convert_cast(text, n=5):
    try:
        return [i["name"] for i in ast.literal_eval(text)[:n]]
    except Exception:
        return []


def fetch_director(text):
    try:
        for i in ast.literal_eval(text):
            if i["job"] == "Director":
                return [i["name"]]
    except Exception:
        pass
    return []


@st.cache_data
def clean_data(df):
    df = df.copy()
    df["genres_list"] = df["genres"].apply(convert)
    df["keywords"] = df["keywords"].apply(convert)
    df["cast_full_list"] = df["cast"].apply(lambda x: convert_cast(x, n=6))
    df["cast_list"] = df["cast"].apply(lambda x: convert_cast(x, n=3))
    df["director_list"] = df["crew"].apply(fetch_director)

    df["genres"] = df["genres_list"].apply(lambda x: " ".join(x))
    df["keywords"] = df["keywords"].apply(lambda x: " ".join(x))
    df["cast"] = df["cast_list"].apply(lambda x: " ".join(x))
    df["crew"] = df["director_list"].apply(lambda x: " ".join(x))
    df["director"] = df["director_list"].apply(lambda x: x[0] if x else "Unknown")
    df["cast_display"] = df["cast_full_list"].apply(lambda x: ", ".join(x) if x else "—")

    df["tags"] = (
        df["genres"] + " " + df["keywords"] + " " +
        df["overview"] + " " + df["cast"] + " " + df["crew"]
    )

    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    return df


movies_data = clean_data(movies_data_raw)
movies_data["row_id"] = movies_data.index  # stable key for widgets / lookups


@st.cache_resource
def compute_similarity(data):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=15000)
    vectors = vectorizer.fit_transform(data)
    return cosine_similarity(vectors)


similarity = compute_similarity(movies_data["tags"])


def get_movie_by_row_id(row_id):
    return movies_data.loc[row_id]


def recommend_by_row_id(row_id, top_n=8):
    similarity_score = list(enumerate(similarity[row_id]))
    sorted_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    recommendations = []
    for idx, score in sorted_movies[1:top_n + 1]:
        similarity_percent = round(score * 100, 2)
        recommendations.append((movies_data.iloc[idx], similarity_percent))
    return recommendations


def recommend_by_title(movie_name, top_n=8):
    list_of_titles = movies_data["title"].tolist()
    close_match = difflib.get_close_matches(movie_name, list_of_titles)
    if not close_match:
        return None
    matched_movie = close_match[0]
    row_id = movies_data[movies_data.title == matched_movie].index[0]
    return matched_movie, recommend_by_row_id(row_id, top_n)


def passes_filters(movie, min_rating, year_range, selected_genres):
    if not (pd.notna(movie["poster"]) and str(movie["poster"]).startswith("http")):
        return False
    if movie["vote_average"] < min_rating:
        return False
    if pd.isna(movie["year"]) or not (year_range[0] <= movie["year"] <= year_range[1]):
        return False
    if selected_genres:
        movie_genres = movie["genres"].split()
        if not any(g in movie_genres for g in selected_genres):
            return False
    return True


# =====================================================================
#  SESSION STATE / NAVIGATION
# =====================================================================
defaults = {
    "view": "home",
    "selected_row_id": None,
    "home_shuffle_seed": random.randint(0, 999999),
    "home_count": 12,
    "reco_count": 8,
    "last_recommended_for": None,
    "active_tab": "discover",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def go_to_detail(row_id):
    st.session_state.selected_row_id = row_id
    st.session_state.view = "detail"


def go_home():
    st.session_state.view = "home"
    st.session_state.selected_row_id = None


def go_to_tab(tab_key):
    st.session_state.active_tab = tab_key
    st.session_state.view = "home"
    st.session_state.selected_row_id = None


# =====================================================================
#  CARD RENDERING
# =====================================================================
def render_card(movie, extra_badge=None, key_prefix="card"):
    poster = movie["poster"] if pd.notna(movie["poster"]) and str(movie["poster"]).startswith("http") \
        else "https://via.placeholder.com/300x450.png?text=No+Poster"
    year = int(movie["year"]) if pd.notna(movie["year"]) else "—"
    row_id = movie["row_id"]

    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
    st.image(poster, use_container_width=True)
    st.markdown(f'<div class="movie-title">{movie["title"]}</div>', unsafe_allow_html=True)

    badges = f'<span class="badge badge-rating">⭐ {movie["vote_average"]}</span>'
    badges += f'<span class="badge badge-year">📅 {year}</span>'
    if extra_badge is not None:
        badges += f'<span class="badge badge-match">🎯 {extra_badge}% match</span>'
    st.markdown(f'<div class="badge-row">{badges}</div>', unsafe_allow_html=True)

    with st.expander("📖 Quick description"):
        preview = movie["overview"]
        if len(preview) > 220:
            preview = preview[:220] + "…"
        st.write(preview)

    st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
    if st.button("▶ View Details", key=f"{key_prefix}_view_{row_id}"):
        go_to_detail(row_id)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def grid(items, is_recommendation=False, cols=4, key_prefix="grid"):
    for i in range(0, len(items), cols):
        row = st.columns(cols)
        chunk = items[i:i + cols]
        for col, item in zip(row, chunk):
            with col:
                if is_recommendation:
                    movie, score = item
                    render_card(movie, extra_badge=score, key_prefix=key_prefix)
                else:
                    render_card(item, key_prefix=key_prefix)


# =====================================================================
#  TOP HEADER (brand left, nav buttons right — like a real app header)
# =====================================================================
_on_home = st.session_state.view == "home"

st.markdown('<div class="navbar-wrap">', unsafe_allow_html=True)
head_brand, head_nav1, head_nav2, head_nav3 = st.columns([2.6, 1, 1, 1])

with head_brand:
    st.markdown(
        '<div class="nav-brand">'
        '<span class="nav-logo">🎬</span>'
        '<span class="nav-brand-text">CineMatch</span>'
        '</div>',
        unsafe_allow_html=True,
    )

with head_nav1:
    active = _on_home and st.session_state.active_tab == "discover"
    if st.button("🏠︎ Home", key="nav_discover", use_container_width=True,
                 type="primary" if active else "secondary"):
        go_to_tab("discover")
        st.rerun()

with head_nav2:
    active = _on_home and st.session_state.active_tab == "recommend"
    if st.button("🔍︎ Search", key="nav_recommend", use_container_width=True,
                 type="primary" if active else "secondary"):
        go_to_tab("recommend")
        st.rerun()

with head_nav3:
    active = _on_home and st.session_state.active_tab == "analytics"
    if st.button("📈 Analytics", key="nav_analytics", use_container_width=True,
                 type="primary" if active else "secondary"):
        go_to_tab("analytics")
        st.rerun()

st.markdown('</div><div class="navbar-underline"></div>', unsafe_allow_html=True)



# =====================================================================
#  SIDEBAR FILTERS
# =====================================================================
with st.sidebar:
    st.markdown("## ⚙️ Filters")
    min_rating = st.slider("⭐ Minimum rating", 0.0, 10.0, 0.0, 0.5)
    min_year = int(movies_data["year"].min())
    max_year = int(movies_data["year"].max())
    year_range = st.slider("📅 Release year", min_year, max_year, (2000, max_year))
    all_genres = sorted(set(" ".join(movies_data["genres"]).split()))
    selected_genres = st.multiselect("🎭 Genre filter", all_genres)

    st.markdown("---")
    if st.session_state.view == "detail":
        if st.button("🏠 Back to Home"):
            go_home()
            st.rerun()
    st.markdown("---")
    st.caption("Built with Streamlit · TF-IDF · Cosine Similarity")


# =====================================================================
#  DETAIL PAGE (opens when a movie card / "View Details" is clicked)
# =====================================================================
def render_detail_page():
    row_id = st.session_state.selected_row_id
    if row_id is None or row_id not in movies_data.index:
        st.warning("Movie not found. Returning home.")
        go_home()
        st.rerun()
        return

    movie = get_movie_by_row_id(row_id)
    poster = movie["poster"] if pd.notna(movie["poster"]) and str(movie["poster"]).startswith("http") \
        else "https://via.placeholder.com/400x600.png?text=No+Poster"
    year = int(movie["year"]) if pd.notna(movie["year"]) else "—"

    if st.button("← Back"):
        go_home()
        st.rerun()

    st.markdown('<div class="detail-hero">', unsafe_allow_html=True)
    col_poster, col_info = st.columns([1, 2.4])

    with col_poster:
        st.image(poster, use_container_width=True)

    with col_info:
        st.markdown(f'<div class="detail-title">{movie["title"]}</div>', unsafe_allow_html=True)

        genre_badges = "".join(
            f'<span class="badge badge-genre">{g}</span>' for g in movie["genres_list"]
        )
        vote_count = int(movie["vote_count"]) if pd.notna(movie["vote_count"]) else 0
        st.markdown(
            f'<div class="badge-row">'
            f'<span class="badge badge-rating">⭐ {movie["vote_average"]} ({vote_count} votes)</span>'
            f'<span class="badge badge-year">📅 {year}</span>'
            f'{genre_badges}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(f'<div class="detail-overview">{movie["overview"]}</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown(f"**🎬 Director:** {movie['director']}")
        st.markdown(f"**⭐ Cast:** {movie['cast_display']}")
        pop = movie["popularity"] if pd.notna(movie["popularity"]) else 0.0
        st.markdown(f"**🔥 Popularity score:** {pop:.1f}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎯 More Like This</div>', unsafe_allow_html=True)
    recs = recommend_by_row_id(row_id, top_n=8)
    valid_recs = [(m, s) for m, s in recs if passes_filters(m, min_rating, year_range, selected_genres)]

    if not valid_recs:
        st.info("No similar movies match your current sidebar filters — try loosening them.")
    else:
        grid(valid_recs, is_recommendation=True, cols=4, key_prefix=f"detail_{row_id}")


# =====================================================================
#  HOME PAGE
# =====================================================================
def render_home_page():
    active_tab = st.session_state.active_tab

    # ---------------- DISCOVER (random movies) ----------------
    if active_tab == "discover":
        top_row = st.columns([3, 1, 1])
        with top_row[0]:
            st.subheader("🎥 Trending Movies")
            st.caption("Discover the most popular and highly rated movies loved by audiences worldwide")
        with top_row[1]:
            st.write("")
            if st.button("🎥 Discover"):
                st.session_state.home_shuffle_seed = random.randint(0, 999999)
                st.session_state.home_count = 12
                st.rerun()
        with top_row[2]:
            st.write("")
            surprise = st.button("Watch Something")

        pool = movies_data[
            movies_data.apply(lambda r: passes_filters(r, min_rating, year_range, selected_genres), axis=1)
        ]

        if pool.empty:
            st.warning("No movies match the current filters. Try loosening them in the sidebar.")
        else:
            if surprise:
                pick = pool.sample(1).iloc[0]
                go_to_detail(pick["row_id"])
                st.rerun()

            shuffled = pool.sample(frac=1, random_state=st.session_state.home_shuffle_seed)
            visible = shuffled.iloc[: st.session_state.home_count]

            grid([row for _, row in visible.iterrows()], is_recommendation=False, cols=4, key_prefix="home")

            if len(shuffled) > len(visible):
                _, mid_col, _ = st.columns([2, 1, 2])
                with mid_col:
                    if st.button("⬇️ Load More Movies"):
                        st.session_state.home_count += 12
                        st.rerun()

    # ---------------- RECOMMEND ----------------
    elif active_tab == "recommend":
        st.subheader("🎞️ Find movies similar to your favorite")

        c1, c2 = st.columns([3, 1])
        with c1:
            movie_name = st.selectbox(
                "🎥 Select or search a movie",
                options=[""] + movies_data["title"].tolist(),
            )
        with c2:
            st.write("")
            st.write("")
            go = st.button("🔍 Search", use_container_width=True)

        if go and movie_name:
            st.session_state.last_recommended_for = movie_name
            st.session_state.reco_count = 8

        active_movie = st.session_state.last_recommended_for

        if active_movie:
            result = recommend_by_title(active_movie, top_n=max(st.session_state.reco_count, 20))

            if result is None:
                st.error("Movie not found ❌ try a different title.")
            else:
                matched_movie, recommendations = result
                st.success(f"Movies similar to **{matched_movie}** 🎉")

                valid_movies = [
                    (m, s) for m, s in recommendations
                    if passes_filters(m, min_rating, year_range, selected_genres)
                ]

                if not valid_movies:
                    st.warning("No movies with available posters match the filters.")
                else:
                    shown = valid_movies[: st.session_state.reco_count]
                    grid(shown, is_recommendation=True, cols=4, key_prefix="reco")

                    if len(valid_movies) > len(shown):
                        _, mid_col, _ = st.columns([2, 1, 2])
                        with mid_col:
                            if st.button("➕ Show More Similar Movies"):
                                st.session_state.reco_count += 8
                                st.rerun()
        else:
            st.info("👆 Pick a movie above and hit **Recommend** to see similar titles.")

    # ---------------- ANALYTICS ----------------
    elif active_tab == "analytics":
        st.subheader("📊 Movie Insights")

        genre_counts = pd.Series(" ".join(movies_data["genres"]).split()).value_counts().head(15)
        fig_genre = px.bar(
            genre_counts, x=genre_counts.values, y=genre_counts.index, orientation="h",
            labels={"x": "Number of movies", "y": "Genre"}, title="Top 15 Genres by Movie Count",
            color=genre_counts.values, color_continuous_scale="Sunset",
        )
        fig_genre.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_genre, use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            fig_rating = px.histogram(
                movies_data, x="vote_average", nbins=30, title="Rating Distribution",
                color_discrete_sequence=["#ff4d6d"],
            )
            fig_rating.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rating, use_container_width=True)

        with colB:
            movies_per_year = movies_data.dropna(subset=["year"]).groupby("year").size()
            movies_per_year = movies_per_year[movies_per_year.index >= 1970]
            fig_year = px.line(
                x=movies_per_year.index, y=movies_per_year.values,
                title="Movies Released per Year (since 1970)",
                labels={"x": "Year", "y": "Number of movies"},
            )
            fig_year.update_traces(line_color="#7b2ff7")
            fig_year.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_year, use_container_width=True)

        st.markdown("#### 🏆 Highest-Rated Directors (min. 3 movies)")
        dir_stats = (
            movies_data[movies_data["director"] != "Unknown"]
            .groupby("director")
            .agg(avg_rating=("vote_average", "mean"), movie_count=("title", "count"))
            .query("movie_count >= 3")
            .sort_values("avg_rating", ascending=False)
            .head(10)
            .reset_index()
        )
        dir_stats["avg_rating"] = dir_stats["avg_rating"].round(2)
        st.dataframe(dir_stats, use_container_width=True, hide_index=True)


# =====================================================================
#  ROUTER
# =====================================================================
if st.session_state.view == "detail":
    render_detail_page()
else:
    render_home_page()
