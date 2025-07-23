import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt


# Initialize session state
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

if "show_watchlist" not in st.session_state:
    st.session_state.show_watchlist = False

if "show_recommendations" not in st.session_state:
    st.session_state.show_recommendations = False


if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
query_params = st.query_params

def scroll_to_section(section_name):
    st.query_params["section"] = section_name

def get_recommendations_from_search(text, data, top_n=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    data = data.copy()
    data["content"] = data["content"].fillna("")

    # TF-IDF Vectorizer to convert text to vectors
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data["content"])

    # Transform the user's query
    query_vec = tfidf.transform([text])

    # Compute cosine similarity between query and all movies
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get top N matches
    top_indices = cosine_sim.argsort()[-top_n:][::-1]

    return data.iloc[top_indices]




st.set_page_config(page_title="CineMatch", layout="wide")
# Title and description
st.title("🎬 CineMatch: Movie & Series Recommender")
st.markdown("This app recommends the best movies and shows across platforms like Netflix, Prime Video, and IMDb based on your taste.")

# === Toggle buttons for Watchlist and Recommendations ===
header_col1, header_col2 = st.columns([6, 1])  # Adjust width ratios for spacing

with header_col2:
    if st.button("⭐ Show Watchlist", key="show_watchlist_btn"):
        st.session_state.show_watchlist = not st.session_state.get("show_watchlist", False)





# Watchlist Panel
if st.session_state.show_watchlist:
    st.markdown("## ⭐ Your Watchlist")

    if st.session_state.watchlist:
        for i, item in enumerate(st.session_state.watchlist):
            with st.expander(f"{item['Title']} ({item['Year']})"):
                st.write(f"**Platform:** {item['Platform']}")
                st.write(f"**Genre:** {item['Genre']}")
                st.write(f"**IMDb Rating:** {item['IMDb']}")
                if st.button(f"❌ Remove from Watchlist", key=f"remove_{i}"):
                    st.session_state.watchlist.pop(i)
                    st.rerun()

        # Download all as CSV
        df_watchlist = pd.DataFrame(st.session_state.watchlist)
        csv = df_watchlist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Watchlist as CSV",
            data=csv,
            file_name="my_watchlist.csv",
            mime="text/csv"
        )
    else:
        st.info("Your watchlist is empty.")









# ========== Load Datasets ==========
@st.cache_data
def load_data():
    # Load each dataset
    netflix = pd.read_csv("C:/Users/Administrator/Desktop/Project/Python/movie_recommendation_app/data/NetflixOriginals.csv").head(30)
    prime = pd.read_csv("data/Amazon Prime Movies.csv").head(15)
    imdb = pd.read_csv("data/IMDb Top 1000.csv").head(10)



    # Add platform column
    netflix["platform"] = "Netflix"
    prime["platform"] = "Amazon Prime"
    imdb["platform"] = "IMDb"

    # Rename columns to match format
    netflix = netflix.rename(columns={
        "title": "title",
        "listed_in": "genre",
        "release_year": "year"
    })
    prime = prime.rename(columns={
        "title": "title",
        "listed_in": "genre",
        "release_year": "year"
    })
    imdb["platform"] = "IMDb"
    imdb = imdb.rename(columns={
        "Series_Title": "title",
        "Genre": "genre",
        "IMDB_Rating": "imdb_rating",
        "Released_Year": "year",
        "Poster_Link": "poster"
    })

    # IMDb has rating, others don’t
    netflix["imdb_rating"] = None
    prime["imdb_rating"] = None

    netflix["poster"] = None
    prime["poster"] = None

    # Combine into one DataFrame
    combined = pd.concat([
        netflix[["title", "genre", "imdb_rating", "year", "platform", "poster"]],
        prime[["title", "genre", "imdb_rating", "year", "platform", "poster"]],
        imdb[["title", "genre", "imdb_rating", "year", "platform", "poster"]]
    ], ignore_index=True)

    # Clean year column
    combined = combined.dropna(subset=["year"])
    combined["year"] = combined["year"].astype(str).str.extract(r"(\d{4})").dropna().astype(int)

    combined["genre"] = combined["genre"].fillna("")
    combined["title"] = combined["title"].fillna("")
    combined["content"] = combined["title"] + " " + combined["genre"]

    return combined


data = load_data()

# ========== Sidebar Filters ==========
st.sidebar.header("🔍 Filter Options")

platform = st.sidebar.selectbox(
    "Select Platform",
    ["All", "Netflix", "Amazon Prime", "IMDb"]
)

genre = st.sidebar.multiselect(
    "Select Genre",
    sorted(list(set([g.strip() for sublist in data["genre"].dropna().str.split(",") for g in sublist])))
)

year_range = st.sidebar.slider("Select Release Year", 1980, 2025, (2000, 2024))









# === Movie Recommendations Section ===

st.markdown("## 👥 Movie Recommendations Based on Your Taste")

with st.expander("🎥 Recommendations", expanded=True):
    movie_list = data["title"].dropna().unique()
    selected_movie = st.selectbox("🎯 Choose a movie you like:", sorted(movie_list))


    if selected_movie:
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(data["content"])

            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

            indices = pd.Series(data.index, index=data['title']).drop_duplicates()
            idx = indices[selected_movie]

            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
            movie_indices = [i[0] for i in sim_scores]

            rec_cols = st.columns(2)
            for i, movie_idx in enumerate(movie_indices):
                with rec_cols[i % 2]:
                    row = data.iloc[movie_idx]
                    st.markdown(f"""
                        <div style="background-color:#f3f3f3;padding:15px;border-radius:10px;margin-bottom:20px;">
                        <h5>{row.title} ({row.year})</h5>
                        <b>Platform:</b> {row.platform}<br>
                        <b>Genre:</b> {row.genre}<br>
                        <b>IMDb Rating:</b> {row.imdb_rating if pd.notnull(row.imdb_rating) else 'N/A'}<br>
                        <a href="https://www.youtube.com/results?search_query={row.title.replace(' ', '+')}+trailer" target="_blank">▶️ Watch Trailer</a>
                        </div>
                    """, unsafe_allow_html=True)

st.markdown("## 🎭 Mood-Based Recommendation")

mood = st.selectbox("Choose your mood", [
    "None", "Happy", "Sad", "Romantic", "Thrilling", "Action", "Comedy", "Drama", "Scary", "Inspirational"
], index=0)

if mood != "None":
    st.markdown(f"### 🎬 Recommendations for mood: *{mood}*")

    mood_keywords = {
        "Happy": ["family", "feel-good", "musical", "joy", "animated"],
        "Sad": ["tragedy", "emotional", "loss"],
        "Romantic": ["romance", "love"],
        "Thrilling": ["thriller", "suspense", "mystery"],
        "Action": ["action", "adventure", "battle"],
        "Comedy": ["comedy", "funny", "satire"],
        "Drama": ["drama", "biography", "history"],
        "Scary": ["horror", "ghost", "supernatural"],
        "Inspirational": ["biopic", "motivational", "sports"],
    }

    keywords = mood_keywords.get(mood, [])
    mood_filtered = data[data["genre"].str.contains('|'.join(keywords), case=False, na=False)]

    if not mood_filtered.empty:
        mood_filtered = mood_filtered.sample(min(5, len(mood_filtered)))  # show up to 5
        mood_cols = st.columns(2)
        for i, row in enumerate(mood_filtered.itertuples()):
            with mood_cols[i % 2]:
                st.markdown(f"""
                    <div style="background-color:#f2f2f2;padding:15px;border-radius:10px;margin-bottom:20px;">
                    <h5>{row.title} ({row.year})</h5>
                    <b>Platform:</b> {row.platform}<br>
                    <b>Genre:</b> {row.genre}<br>
                    <b>IMDb Rating:</b> {row.imdb_rating if pd.notnull(row.imdb_rating) else 'N/A'}<br>
                    <a href="https://www.youtube.com/results?search_query={row.title.replace(' ', '+')}+trailer" target="_blank">▶️ Watch Trailer</a>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("😔 No movies found for that mood.")




# ========== Apply Filters ==========
filtered = data.copy()

if platform != "All":
    filtered = filtered[filtered["platform"] == platform]

if genre:
    filtered = filtered[filtered["genre"].str.contains('|'.join(genre), case=False, na=False)]

filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]



st.markdown("---")
st.markdown("## 📊 Data Insights and Analytics")

analytics_tabs = st.tabs(["📺 Platform Distribution", "🎭 Genre Popularity", "📅 Yearly Trend"])

# 📺 Platform Distribution
with analytics_tabs[0]:
    platform_counts = data["platform"].value_counts().reset_index()
    platform_counts.columns = ["Platform", "Count"]

    chart = alt.Chart(platform_counts).mark_bar().encode(
        x=alt.X("Platform", sort='-y'),
        y="Count",
        color="Platform"
    ).properties(
        width=600,
        height=400,
        title="Content by Platform"
    )

    st.altair_chart(chart)

# 🎭 Genre Popularity
with analytics_tabs[1]:
    # Flatten genre strings
    all_genres = data["genre"].dropna().str.split(",")
    all_genres = [genre.strip() for sublist in all_genres for genre in sublist]

    genre_counts = pd.Series(all_genres).value_counts().reset_index()
    genre_counts.columns = ["Genre", "Count"]
    top_genres = genre_counts.head(15)

    chart = alt.Chart(top_genres).mark_bar().encode(
        x=alt.X("Genre", sort='-y'),
        y="Count",
        color=alt.Color("Genre", legend=None)
    ).properties(
        width=600,
        height=400,
        title="Top 15 Genres"
    )

    st.altair_chart(chart)

# 📅 Yearly Trend
with analytics_tabs[2]:
    year_counts = data["year"].value_counts().sort_index().reset_index()
    year_counts.columns = ["Year", "Count"]

    chart = alt.Chart(year_counts).mark_area(
        line={'color':'steelblue'},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='lightblue', offset=0),
                   alt.GradientStop(color='white', offset=1)],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x="Year",
        y="Count"
    ).properties(
        width=600,
        height=400,
        title="Movies/Shows Released Over the Years"
    )

    st.altair_chart(chart)
# ========== Search Bar ==========
search_query = st.text_input("🔎 Search by Title")
# Apply search filter
# Apply search filter
if search_query:
    filtered = filtered[filtered["title"].str.contains(search_query, case=False, na=False)]

    if filtered.empty:
        st.warning("😕 No results found for your search.")

        st.markdown("### 🎯 You May Also Like:")
        recs = get_recommendations_from_search(search_query, data, top_n=5)

        rec_cols = st.columns(2)
        for i, row in enumerate(recs.itertuples()):
            with rec_cols[i % 2]:
                st.markdown(f"""
                    <div style="background-color:#f2f2f2;padding:15px;border-radius:10px;margin-bottom:20px;">
                    <h5>{row.title} ({row.year})</h5>
                    <b>Platform:</b> {row.platform}<br>
                    <b>Genre:</b> {row.genre}<br>
                    <b>IMDb Rating:</b> {row.imdb_rating if pd.notnull(row.imdb_rating) else 'N/A'}<br>
                    <a href="https://www.youtube.com/results?search_query={row.title.replace(' ', '+')}+trailer" target="_blank">▶️ Watch Trailer</a>
                    </div>
                """, unsafe_allow_html=True)



# ========== Display Movie Cards ==========
st.markdown(f"### 🎞️ {len(filtered)} Movies/Series Found")

cols = st.columns(3)

for idx, row in enumerate(filtered.itertuples()):
    with cols[idx % 3]:
        st.markdown(f"""
            <div style="background-color:#f9f9f9;padding:10px;border-radius:10px;margin-bottom:20px;">
                <h5>{row.title} ({row.year})</h5>
        """, unsafe_allow_html=True)

        # Poster image (if exists)
        if row.poster and pd.notna(row.poster):
            st.image(row.poster, width=220)

        # Show details
        st.markdown(f"""
        - **Platform**: {row.platform}  
        - **Genre**: {row.genre}  
        - **IMDb Rating**: {row.imdb_rating if pd.notnull(row.imdb_rating) else 'N/A'}
        """)

        # Trailer button (YouTube search)
        youtube_url = f"https://www.youtube.com/results?search_query={row.title.replace(' ', '+')}+trailer"
        st.markdown(f"[▶️ Watch Trailer]({youtube_url})", unsafe_allow_html=True)

        # Add to Watchlist button
        if st.button(f"➕ Add to Watchlist", key=f"add_{idx}"):
            st.session_state.watchlist.append({
                "Title": row.title,
                "Year": row.year,
                "Platform": row.platform,
                "Genre": row.genre,
                "IMDb": row.imdb_rating,
            })
            st.success(f"✅ Added '{row.title}' to Watchlist!")









