import os

import pandas as pd
import plotly.express as px
import streamlit as st

from netflix_utils import (
    build_dataset_context,
    build_search_index,
    cluster_content,
    country_counts,
    country_type_table,
    genre_counts,
    load_data,
    retrieve_titles,
)

st.set_page_config(
    page_title="Netflix Content Explorer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
    .hero {
        background: linear-gradient(135deg, #e50914 0%, #7a0b0b 100%);
        padding: 30px 34px;
        border-radius: 18px;
        margin-bottom: 18px;
        box-shadow: 0 12px 30px rgba(229, 9, 20, 0.25);
    }
    .hero h1 {color: #ffffff; margin: 0; font-size: 2.2rem; letter-spacing: 0.4px;}
    .hero p {color: #ffe1e1; margin: 8px 0 0; font-size: 1.02rem;}
    div[data-testid="stMetric"] {
        background: rgba(229, 9, 20, 0.07);
        border: 1px solid rgba(229, 9, 20, 0.25);
        border-radius: 14px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] {font-size: 0.82rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 6px;}
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 8px 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
]

SYSTEM_BASE = (
    "You are a sharp data analyst embedded in a Netflix exploratory-data-analysis dashboard. "
    "Answer only from the dataset summary and the retrieved titles supplied to you. "
    "Be concise, use short paragraphs or bullet points, cite concrete numbers, and never invent statistics. "
    "If the dataset does not contain the answer, say so plainly and suggest a useful alternative question. "
    "When recommending titles, explain in one line why each fits the request."
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


@st.cache_resource(show_spinner=False)
def get_index():
    df = get_data()
    return build_search_index(df)


@st.cache_data(show_spinner=False)
def get_context():
    return build_dataset_context(get_data())


@st.cache_resource(show_spinner="Clustering the catalogue...")
def get_clusters(n_clusters, types, year_range, ratings, countries):
    df = get_data()
    vectorizer, matrix = get_index()

    mask = (
        df["type"].isin(types)
        & df["release_year"].between(year_range[0], year_range[1])
        & df["rating"].isin(ratings)
    )
    if countries:
        mask = mask & df["primary_country"].isin(countries)

    subset = df[mask].reset_index(drop=True)
    if len(subset) < n_clusters:
        return subset, {}

    sub_matrix = matrix[mask.to_numpy()]
    return cluster_content(subset, sub_matrix, vectorizer, n_clusters)


def gemini_reply(api_key, model_name, messages, system_instruction):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(
            role="user" if message["role"] == "user" else "model",
            parts=[types.Part(text=message["content"])],
        )
        for message in messages
    ]
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.35,
        ),
    )
    return response.text or "I could not generate a response. Please try rephrasing."


df_all = get_data()

with st.sidebar:
    st.markdown("## 🎬 Netflix Explorer")
    st.caption("EDA · Country insights · Clustering · Gemini")
    st.divider()

    all_types = sorted(df_all["type"].unique())
    selected_types = st.multiselect("Content type", all_types, default=all_types)

    year_min = int(df_all["release_year"].min())
    year_max = int(df_all["release_year"].max())
    year_range = st.slider("Release year", year_min, year_max, (year_min, year_max))

    country_options = country_counts(df_all, 30).index.tolist()
    selected_countries = st.multiselect(
        "Primary country", country_options, placeholder="All countries"
    )

    rating_options = sorted(df_all["rating"].unique())
    selected_ratings = st.multiselect("Rating", rating_options, default=rating_options)

    st.divider()
    st.markdown("### 🤖 Gemini")
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    api_key = st.text_input(
        "API key",
        value=env_key,
        type="password",
        help="Create a free key at https://aistudio.google.com/app/apikey",
    )
    model_name = st.selectbox("Model", MODELS, index=0)
    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

st.markdown(
    """
    <div class="hero">
      <h1>🎬 Netflix Content Explorer</h1>
      <p>Interactive EDA, country intelligence, TV-vs-movie trends, text clustering
      and a Gemini-powered analyst for the Netflix catalogue.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = df_all[df_all["type"].isin(selected_types)]
df = df[df["release_year"].between(year_range[0], year_range[1])]
df = df[df["rating"].isin(selected_ratings)]
if selected_countries:
    df = df[df["primary_country"].isin(selected_countries)]

if df.empty:
    st.warning("No titles match the current filters. Adjust the sidebar to continue.")
    st.stop()

metric_cols = st.columns(5)
metric_cols[0].metric("Titles", f"{len(df):,}")
metric_cols[1].metric("Movies", f"{(df['type'] == 'Movie').sum():,}")
metric_cols[2].metric("TV Shows", f"{(df['type'] == 'TV Show').sum():,}")
metric_cols[3].metric("Countries", f"{country_counts(df, 10000).shape[0]:,}")
metric_cols[4].metric("Genres", f"{genre_counts(df, 10000).shape[0]:,}")

tab_overview, tab_countries, tab_trends, tab_clusters, tab_ai = st.tabs(
    ["📊 Overview", "🌍 Countries", "📈 TV vs Movies", "🧩 Clustering", "🤖 AI Analyst"]
)

with tab_overview:
    left, right = st.columns(2)

    with left:
        type_df = df["type"].value_counts().reset_index()
        type_df.columns = ["type", "count"]
        fig = px.pie(
            type_df,
            names="type",
            values="count",
            hole=0.55,
            title="Content mix",
            color="type",
            color_discrete_sequence=["#E50914", "#2B2B2B"],
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, width="stretch")

    with right:
        rating_df = df["rating"].value_counts().head(10).reset_index()
        rating_df.columns = ["rating", "count"]
        fig = px.bar(
            rating_df,
            x="count",
            y="rating",
            orientation="h",
            title="Most common ratings",
            color="count",
            color_continuous_scale="Reds",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)

    with left:
        genres = genre_counts(df, 15).reset_index()
        genres.columns = ["genre", "count"]
        fig = px.bar(
            genres,
            x="count",
            y="genre",
            orientation="h",
            title="Top 15 genres",
            color="count",
            color_continuous_scale="Reds",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    with right:
        releases = df["release_year"].value_counts().sort_index().reset_index()
        releases.columns = ["release_year", "count"]
        fig = px.area(
            releases,
            x="release_year",
            y="count",
            title="Titles by release year",
            color_discrete_sequence=["#E50914"],
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Dataset preview")
    st.dataframe(
        df[["title", "type", "primary_country", "release_year", "rating", "listed_in"]].head(50),
        width="stretch",
        hide_index=True,
    )

with tab_countries:
    left, right = st.columns(2)

    with left:
        top = country_counts(df, 15).reset_index()
        top.columns = ["country", "titles"]
        fig = px.bar(
            top,
            x="titles",
            y="country",
            orientation="h",
            title="Top 15 countries by catalogue size",
            color="titles",
            color_continuous_scale="Reds",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    with right:
        table = country_type_table(df, 10).reset_index()
        table.columns = ["country", "Movie", "TV Show"]
        melted = table.melt(id_vars="country", var_name="type", value_name="count")
        fig = px.bar(
            melted,
            x="country",
            y="count",
            color="type",
            barmode="stack",
            title="Movies vs TV shows in top 10 countries",
            color_discrete_sequence=["#E50914", "#2B2B2B"],
        )
        fig.update_layout(xaxis_tickangle=-35, legend_title="Type")
        st.plotly_chart(fig, width="stretch")

    world = country_counts(df, 500).reset_index()
    world.columns = ["country", "titles"]
    fig = px.choropleth(
        world,
        locations="country",
        locationmode="country names",
        color="titles",
        color_continuous_scale="Reds",
        title="Global content footprint",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Country breakdown")
    st.dataframe(table, width="stretch", hide_index=True)

with tab_trends:
    yearly = (
        df.dropna(subset=["year_added"])
        .groupby(["year_added", "type"])
        .size()
        .unstack(fill_value=0)
    )
    if not yearly.empty:
        yearly.index = yearly.index.astype(int)

    if yearly.empty:
        st.info("No dated titles for the current filters.")
    else:
        yearly_long = yearly.reset_index().melt(
            id_vars="year_added", var_name="type", value_name="titles"
        )
        fig = px.line(
            yearly_long,
            x="year_added",
            y="titles",
            color="type",
            markers=True,
            title="Titles added to Netflix per year, by type",
            color_discrete_sequence=["#E50914", "#2B2B2B"],
        )
        fig.update_layout(xaxis_title="Year added", yaxis_title="Titles")
        st.plotly_chart(fig, width="stretch")

        if "TV Show" in yearly.columns and "Movie" in yearly.columns:
            share = (yearly["TV Show"] / yearly.sum(axis=1) * 100).reset_index()
            share.columns = ["year_added", "tv_share"]

            fig = px.area(
                share,
                x="year_added",
                y="tv_share",
                title="Share of yearly additions that are TV shows (%)",
                color_discrete_sequence=["#E50914"],
            )
            fig.update_layout(yaxis_title="TV share (%)", xaxis_title="Year added")
            st.plotly_chart(fig, width="stretch")

            totals = yearly.sum(axis=1)
            complete = totals
            if len(complete) >= 2 and complete.iloc[-1] < 0.5 * complete.iloc[-2]:
                complete = complete.iloc[:-1]
            meaningful = complete[complete >= 50]

            if len(meaningful) >= 2:
                first_y = int(meaningful.index.min())
                last_y = int(meaningful.index.max())
                share_by_year = share.set_index("year_added")["tv_share"]
                first_share = float(share_by_year.loc[first_y])
                last_share = float(share_by_year.loc[last_y])
                delta = last_share - first_share

                tv_first = int(yearly.loc[first_y, "TV Show"])
                tv_last = int(yearly.loc[last_y, "TV Show"])
                movie_first = int(yearly.loc[first_y, "Movie"])
                movie_last = int(yearly.loc[last_y, "Movie"])

                cols = st.columns(3)
                cols[0].metric(f"TV share {first_y}", f"{first_share:.1f}%")
                cols[1].metric(f"TV share {last_y}", f"{last_share:.1f}%")
                cols[2].metric("Change", f"{delta:+.1f} pts")

                growth = (
                    f"Between {first_y} and {last_y}, TV-show additions went from "
                    f"{tv_first:,} to {tv_last:,}, while movie additions went from "
                    f"{movie_first:,} to {movie_last:,}."
                )
                if delta > 0:
                    st.success(
                        growth
                        + " TV is taking a larger share of new titles, supporting the brief "
                        "that Netflix has been leaning further into series."
                    )
                else:
                    st.info(
                        growth
                        + " The TV share of new additions has not risen over this window; "
                        "try narrowing the filters or comparing other years."
                    )
                st.caption(
                    "Years with fewer than 50 additions and an incomplete final year are "
                    "excluded from the headline comparison to avoid noisy percentages."
                )

        st.caption(
            "Trends use the date a title was added to Netflix, which reflects catalogue strategy "
            "better than the original release year."
        )

with tab_clusters:
    st.markdown("### 🧩 Clustering similar content")
    st.caption(
        "TF-IDF over title, description, cast, director and genres, reduced with SVD "
        "and grouped with KMeans."
    )

    n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=5)

    clustered, top_terms = get_clusters(
        n_clusters,
        tuple(selected_types),
        (year_range[0], year_range[1]),
        tuple(selected_ratings),
        tuple(selected_countries),
    )

    if clustered.empty or not top_terms:
        st.warning("Not enough titles match the filters to build clusters.")
    else:
        labels = {
            i: f"C{i} · {', '.join(terms[:3])}" for i, terms in top_terms.items()
        }
        clustered["cluster_label"] = clustered["cluster"].map(labels)

        fig = px.scatter(
            clustered,
            x="svd_x",
            y="svd_y",
            color="cluster_label",
            hover_data={
                "title": True,
                "type": True,
                "primary_country": True,
                "release_year": True,
                "listed_in": True,
                "svd_x": False,
                "svd_y": False,
            },
            title=f"{len(clustered):,} titles grouped into {n_clusters} clusters",
            opacity=0.7,
        )
        fig.update_layout(legend_title="Cluster", xaxis_title="SVD 1", yaxis_title="SVD 2")
        st.plotly_chart(fig, width="stretch")

        for i in range(n_clusters):
            subset = clustered[clustered["cluster"] == i]
            header = (
                f"Cluster {i} · {len(subset):,} titles · "
                + ", ".join(top_terms[i][:5])
            )
            with st.expander(header):
                st.markdown("**Top terms:** " + ", ".join(top_terms[i]))
                st.dataframe(
                    subset[
                        ["title", "type", "primary_country", "release_year", "rating", "listed_in"]
                    ].head(12),
                    width="stretch",
                    hide_index=True,
                )

        st.divider()
        st.subheader("🔎 Find similar titles")
        title_choice = st.selectbox(
            "Choose a title", sorted(clustered["title"].unique())
        )
        if st.button("Find similar titles", type="primary"):
            vectorizer, matrix = get_index()
            results = retrieve_titles(title_choice, vectorizer, matrix, df_all, 11)
            results = results[results["title"] != title_choice].head(10)
            st.dataframe(
                results[["title", "type", "primary_country", "release_year", "rating", "similarity"]],
                width="stretch",
                hide_index=True,
            )

with tab_ai:
    st.markdown("### 🤖 Ask the Netflix analyst")
    st.caption(
        "Gemini answers using a live summary of the catalogue plus the most relevant titles "
        "retrieved with TF-IDF. Add a key in the sidebar to enable it."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    suggestions = [
        "Summarise the three most important insights in this dataset.",
        "Has Netflix shifted from movies toward TV shows over time? Back it with numbers.",
        "Which countries produce the most TV shows, and how do they differ from movie-heavy markets?",
        "Recommend 5 TV shows similar to Dark and explain why.",
    ]

    st.markdown("**Try a prompt:**")
    suggestion_cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if suggestion_cols[i % 2].button(suggestion, width="stretch"):
            st.session_state.pending_prompt = suggestion
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("retrieved"):
                with st.expander("Titles retrieved for this answer"):
                    st.dataframe(
                        pd.DataFrame(message["retrieved"]),
                        width="stretch",
                        hide_index=True,
                    )

    typed = st.chat_input("Ask about the Netflix catalogue...")
    prompt = st.session_state.pop("pending_prompt", None) or typed

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        vectorizer, matrix = get_index()
        retrieved = retrieve_titles(prompt, vectorizer, matrix, df_all, 8)
        retrieved_rows = retrieved[
            ["title", "type", "primary_country", "release_year", "rating", "similarity"]
        ].to_dict("records")

        with st.chat_message("assistant"):
            if not api_key:
                answer = (
                    "**Gemini key not configured.** Add a `GEMINI_API_KEY` in the sidebar "
                    "for full language answers. Meanwhile, here are the closest catalogue "
                    "matches to your question:"
                )
                st.markdown(answer)
                st.dataframe(
                    retrieved[
                        ["title", "type", "primary_country", "release_year", "rating", "similarity"]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                retrieval_text = "\n".join(
                    f"- {row.title} | {row.type} | {row.primary_country} | "
                    f"{row.release_year} | {row.rating} | {row.listed_in}"
                    for row in retrieved.itertuples()
                )
                enriched = (
                    f"{prompt}\n\nRetrieved titles from the dataset that may be relevant:\n"
                    f"{retrieval_text}"
                )
                call_messages = st.session_state.messages[:-1] + [
                    {"role": "user", "content": enriched}
                ]
                system_instruction = f"{SYSTEM_BASE}\n\nDATASET SUMMARY:\n{get_context()}"

                with st.spinner("Gemini is analysing the catalogue..."):
                    try:
                        answer = gemini_reply(
                            api_key, model_name, call_messages, system_instruction
                        )
                    except Exception as exc:
                        answer = f"⚠️ Gemini request failed: `{exc}`"

                st.markdown(answer)
                with st.expander("Titles retrieved for this answer"):
                    st.dataframe(
                        retrieved[
                            ["title", "type", "primary_country", "release_year", "rating", "similarity"]
                        ],
                        width="stretch",
                        hide_index=True,
                    )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "retrieved": retrieved_rows,
            }
        )
