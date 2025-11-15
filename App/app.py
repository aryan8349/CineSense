import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_PATH = os.path.join(BASE_DIR, "../model/embeddings.npy")
FAISS_PATH = os.path.join(BASE_DIR, "../model/faiss_index.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "../data/netflix_data_with_links.csv")
LINKS_PATH = os.path.join(BASE_DIR, "../data/netflix_data_with_links.csv")
EDA_FOLDER = os.path.join(BASE_DIR, "../EDA outputs")

@st.cache_resource
def load_model():
    embeddings = np.load(EMBED_PATH).astype('float32')
    faiss.normalize_L2(embeddings)
    
    if os.path.exists(FAISS_PATH):
        with open(FAISS_PATH, "rb") as f:
            faiss_index = pickle.load(f)
    else:
        faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss_index.add(embeddings)
        with open(FAISS_PATH, "wb") as f:
            pickle.dump(faiss_index, f)
    
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()  

    links_df = pd.read_csv(LINKS_PATH)
    links_df.columns = links_df.columns.str.strip()

    return embeddings, faiss_index, df, links_df

embeddings, faiss_index, df, links_df = load_model()

@st.cache_data
def recommend_with_percentage(title, top_n=5, type_filter=None):
    title_lower = title.lower().strip()
    if title_lower not in df['title'].str.lower().values:
        return []

    idx = df[df['title'].str.lower() == title_lower].index[0]
    vec = embeddings[idx].reshape(1, -1)
    faiss.normalize_L2(vec)

    if type_filter:
        filtered_idx = df[df['type'] == type_filter].index.to_numpy()
        filtered_embeddings = embeddings[filtered_idx]
        faiss.normalize_L2(filtered_embeddings)

        index_temp = faiss.IndexFlatIP(filtered_embeddings.shape[1])
        index_temp.add(filtered_embeddings)
        D, I = index_temp.search(vec, min(top_n, len(filtered_embeddings)))
        top_indices = filtered_idx[I[0]]
        top_scores = D[0]
    else:
        D, I = faiss_index.search(vec, min(top_n + 1, len(df)))
        top_indices = I[0][1:top_n+1] if len(I[0]) > 1 else []
        top_scores = D[0][1:top_n+1] if len(D[0]) > 1 else []

    recommended = []
    for i, score in zip(top_indices, top_scores):
        poster = df.iloc[i]['poster_url'] if pd.notna(df.iloc[i]['poster_url']) else "https://via.placeholder.com/150"
        title_i = df.iloc[i]['title']
        link = links_df.loc[links_df['title'].str.lower() == title_i.lower(), 'movie_link'].values
        link = link[0] if len(link) > 0 else "#"
        recommended.append((title_i, poster, round(float(score)*100,1), link))
    return recommended

st.set_page_config(page_title="Netflix Recommender", page_icon="🎬", layout="wide")
st.title("🎬 CineSense")
st.markdown("---")

movie_input = st.text_input("Enter a movie you like:")

if st.button("Recommend"):
    if not movie_input.strip():
        st.warning("Enter a movie first")
    else:
        recs = recommend_with_percentage(movie_input, top_n=5)
        if not recs:
            st.info("Movie not found in dataset")
        else:
            st.subheader("Top Recommendations")
            cols = st.columns(5)
            for idx, (title, poster_url, similarity, link) in enumerate(recs):
                with cols[idx % 5]:
                    st.markdown(
                        f'<a href="{link}" target="_blank">'
                        f'<img src="{poster_url}" width="150"><br>{title}<br>{similarity}% similar</a>',
                        unsafe_allow_html=True
                    )


st.markdown("---")
st.header("📊 Exploratory Data Analysis")

if st.button("Show Type of Content"):
    type_of_content = df['type'].value_counts()  
    df_type = type_of_content.reset_index()
    df_type.columns = ['Content Type', 'Count']

    fig = px.pie(
        df_type,
        names='Content Type',
        values='Count',
        color='Content Type',
        color_discrete_map={
            'Movie': '#E50914',
            'TV Show': '#B0B0B0',
        },
        hole=0.1
    )
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(
        title_text="Type of Content",
        title_font_size=20,
        template='plotly_dark',
        legend=dict(
            title='Content Type',
            font=dict(size=14, color='white')
        )
    )

    st.plotly_chart(fig, use_container_width=True)

if st.button("Show Content Trend Over the Years"):
    content_trend = df.groupby(['release_year', 'type']).size().unstack(fill_value=0)

    content_trend = content_trend[['Movie', 'TV Show']]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=content_trend.index,
        y=content_trend['Movie'],
        mode='lines+markers',
        name='Movie',
        line=dict(color='#E50914', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(229,9,20,0.2)'  
    ))

    # TV Show line + fill
    fig.add_trace(go.Scatter(
        x=content_trend.index,
        y=content_trend['TV Show'],
        mode='lines+markers',
        name='TV Show',
        line=dict(color='#B0B0B0', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(176,176,176,0.2)'  
    ))

    fig.update_layout(
        title='Netflix Content Trend Over the Years',
        template='plotly_dark',
        title_font_size=20,
        xaxis_title='Release Year',
        yaxis_title='Count of Content',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white'),
        legend=dict(
            x=1.05,
            y=1,
            font=dict(size=12, color='white'),
            bgcolor='#222222',
            bordercolor='white',
            borderwidth=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

if st.button("Movies vs TV Shows Released Over the Years"):

    content_trend = df.groupby(['release_year', 'type']).size().unstack(fill_value=0)
    content_trend_reset = content_trend.reset_index()
    melted = content_trend_reset.melt(
        id_vars='release_year',
        value_vars=['Movie', 'TV Show'],
        var_name='Type',
        value_name='Count'
    )
    melted['release_year'] = melted['release_year'].astype(int)

    fig = px.bar(
        melted,
        x='release_year',
        y='Count',
        color='Type',
        barmode='group',
        color_discrete_map={"Movie": "#E50914", "TV Show": "#B0B0B0"},
        labels={'release_year':'Release Year', 'Count':'Count of Content'},
        title="Movies vs TV Shows Released Over the Years"
    )

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white', size=14),
        xaxis=dict(tickmode='linear', dtick=20),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5),
        legend=dict(
            title='Type',
            x=1.05,
            y=1,
            bgcolor='#111111',
            bordercolor='white',
            borderwidth=1,
            font=dict(color='white')
        )
    )

    st.plotly_chart(fig, use_container_width=True)


if st.button("Rating Distribution by Type"):
    rating_type = df.groupby(['type', 'rating']).size().reset_index(name='Counts')
    rating_type = rating_type.sort_values(by='Counts', ascending=True)

    fig = px.bar(
        rating_type,
        x='rating',
        y='Counts',
        color='type',
        barmode='group',
        color_discrete_map={'Movie': '#E50914', 'TV Show': '#B0B0B0'},
        labels={'rating': 'Rating', 'Counts': 'Counts'},
        title='Rating Distribution by Type'
    )

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white', size=14),
        xaxis=dict(
            tickangle=90, 
            title='Rating'
        ),
        yaxis=dict(
            title='Counts',
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        legend=dict(
            title='Type',
            x=1.05,
            y=1,
            bgcolor='#111111',
            bordercolor='white',
            borderwidth=1,
            font=dict(color='white')
        )
    )

    st.plotly_chart(fig, use_container_width=True)

if st.button("Content Added per Month (All Years)"):
    df['date_added'] = df['date_added'].astype(str).str.strip()
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df_month = df.dropna(subset=['date_added'])
    df_month['month'] = df_month['date_added'].dt.month_name()

    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]

    monthly_counts = (
        df_month['month'].value_counts()
        .reindex(month_order)
        .reset_index()
    )
    monthly_counts.columns = ['Month', 'Counts']

    fig = px.bar(
        monthly_counts,
        x='Month',
        y='Counts',
        text='Counts',
        color_discrete_sequence=['#E50914'],
        title='Content Added per Month (All Years)',
        height=700,
        width=1000
    )

    fig.update_traces(
        texttemplate='%{text}',
        textposition='outside',
        marker=dict(opacity=0.85)
    )

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white', size=14, family='Arial'),
        xaxis=dict(
            title=dict(text='Month', font=dict(size=16, color='white')),
            tickangle=45
        ),
        yaxis=dict(
            title=dict(text='Number of Content Added', font=dict(size=16, color='white')),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        title_font=dict(size=22, color='white', family='Arial Black'),
        margin=dict(l=80, r=50, t=100, b=100)
    )

    st.plotly_chart(fig, use_container_width=True)




if st.button("Country with Most Content"):
    country = df['country'].str.split(',').explode().str.strip()
    country = country[~country.str.contains('unknown', case=False, na=False)]
    country = country.dropna()

    all_country = country.value_counts().head(30)
    country_df = all_country.reset_index()
    country_df.columns = ['Country', 'Count']
    country_df = country_df.sort_values(by='Count', ascending=True)

    fig = px.bar(
        country_df,
        x='Count',
        y='Country',
        orientation='h',
        text='Count',
        color_discrete_sequence=['#E50914'],
        title='Country with Most Content'
    )

    fig.update_traces(
        texttemplate='%{text}',
        textposition='outside',
        marker=dict(opacity=0.85)
    )

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white', size=14),
        xaxis=dict(
            title='Count',
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(
            title='Country',
            categoryorder='total ascending'
        ),
        title_font=dict(size=20, color='white', family='Arial Black')
    )

    st.plotly_chart(fig, use_container_width=True)

if st.button("Movie Duration Distribution"):
    movies = df[df['type'] == 'Movie'].copy()

    movies['duration'] = movies['duration'].astype(str).str.extract('(\d+)')[0]
    movies['duration'] = pd.to_numeric(movies['duration'], errors='coerce')
    movies = movies.dropna(subset=['duration'])

    import numpy as np
    from scipy.stats import gaussian_kde
    import plotly.graph_objects as go

    nbins = 30
    hist_vals, bin_edges = np.histogram(movies['duration'], bins=nbins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    kde = gaussian_kde(movies['duration'])
    x_range = np.linspace(movies['duration'].min(), movies['duration'].max(), 500)
    kde_scaled = kde(x_range) * len(movies) * (bin_edges[1] - bin_edges[0])  # scale to match histogram

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist_vals,
        name='Count',
        marker_color='#E50914',
        opacity=0.7
    ))

    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_scaled,
        mode='lines',
        line=dict(color='white', width=3),
        name='KDE'
    ))

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        title=dict(text='Movie Duration Distribution', font=dict(size=24, color='white', family='Arial Black')),
        xaxis=dict(
            title=dict(text='Duration (Minutes)', font=dict(size=18, color='white')),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(
            title=dict(text='Frequency', font=dict(size=18, color='white')),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        legend=dict(font=dict(color='white')),
        margin=dict(l=80, r=50, t=120, b=80)
    )

    st.plotly_chart(fig, use_container_width=True)


if st.button("Content Type Distribution (All Categories)"):
    Type = df['listed_in'].str.split(',').explode().str.strip()
    
    total_type = Type.value_counts().reset_index()
    total_type.columns = ['Type', 'Count']
    total_type = total_type.sort_values(by='Count', ascending=True)

    fig = px.bar(
        total_type,
        x='Count',
        y='Type',
        orientation='h',
        color='Count',
        color_continuous_scale=['#B0B0B0', '#E50914'],  
        title='Content Type Distribution (All Categories)',
        height=900
    )

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white', size=14, family='Arial'),
        xaxis=dict(
            title=dict(text='Number of Titles', font=dict(size=16, color='white')),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(
            title=dict(text='Type / Genre', font=dict(size=16, color='white'))
        ),
        title_font=dict(size=22, color='white', family='Arial Black'),
        margin=dict(l=200, r=50, t=100, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)


if st.button("Top 20 Most Frequent Directors"):
    directors = (
        df.loc[df['director'].notna() & (df['director'] != 'Unknown'), 'director']
        .str.split(',')
        .explode()
        .str.strip()
    )

    director_counts = (
        directors.value_counts()
        .head(20)
        .sort_values(ascending=True)
        .reset_index()
    )
    director_counts.columns = ['Director', 'Count']

    fig = px.bar(
        director_counts,
        x='Count',
        y='Director',
        orientation='h',
        color='Count',
        color_continuous_scale=['#B0B0B0', '#E50914'],
        height=700
    )

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white', size=14, family='Arial'),
        title=dict(
            text='Top 20 Most Frequent Directors on Netflix',
            font=dict(size=22, color='white', family='Arial Black'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(
                text='Number of Shows/Movies',
                font=dict(size=16, color='white')
            ),
            tickfont=dict(size=12, color='white', family='Arial Black'),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(
            title=dict(
                text='Directors',
                font=dict(size=16, color='white')
            ),
            tickfont=dict(size=12, color='white', family='Arial Black')
        ),
        margin=dict(l=200, r=50, t=100, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)



if st.button("Top 20 Most Common Genres"):
    genres = df['listed_in'].dropna().str.split(',').explode().str.strip()

    genre_counts = genres.value_counts().head(20).sort_values(ascending=True).reset_index()
    genre_counts.columns = ['Genre', 'Count']

    fig = px.bar(
        genre_counts,
        x='Count',
        y='Genre',
        orientation='h',
        color='Count',
        color_continuous_scale=['#B0B0B0', '#E50914'],
        height=700
    )

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white', size=14, family='Arial'),
        title=dict(
            text='Top 20 Most Common Genres on Netflix',
            font=dict(size=22, color='white', family='Arial Black'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(
                text='Number of Titles',
                font=dict(size=16, color='white')
            ),
            tickfont=dict(size=12, color='white', family='Arial Black'),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(
            title=dict(
                text='Genres',
                font=dict(size=16, color='white')
            ),
            tickfont=dict(size=12, color='white', family='Arial Black')
        ),
        margin=dict(l=200, r=50, t=100, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)


if st.button("Top 5 Genre Trends (2010–Present)"):
    genre_trend = df.dropna(subset=['listed_in', 'release_year'])
    genre_trend = genre_trend.assign(
        genre=genre_trend['listed_in'].str.split(',')
    ).explode('genre')
    genre_trend['genre'] = genre_trend['genre'].str.strip()

    top5_genres = genre_trend['genre'].value_counts().head(5).index
    genre_trend = genre_trend[genre_trend['genre'].isin(top5_genres)]

    trend_data = genre_trend.groupby(['release_year', 'genre']).size().reset_index(name='Count')
    trend_data = trend_data[trend_data['release_year'] >= 2010]

    fig = px.line(
        trend_data,
        x='release_year',
        y='Count',
        color='genre',
        markers=True,
        title='Top 5 Genre Trends on Netflix (2010–Present)',
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white', size=14),
        xaxis=dict(
            title=dict(text='Release Year', font=dict(size=16, color='white')),
            dtick=1
        ),
        yaxis=dict(
            title=dict(text='Number of Titles', font=dict(size=16, color='white')),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        legend=dict(
            title='Genre',
            title_font=dict(size=14, color='white'),
            font=dict(size=12, color='white'),
            bgcolor='#111111',
            bordercolor='white',
            borderwidth=1
        ),
        title=dict(
            text='Top 5 Genre Trends on Netflix (2010–Present)',
            font=dict(size=22, color='white', family='Arial Black')
        ),
        margin=dict(l=80, r=50, t=120, b=80)
    )

    st.plotly_chart(fig, use_container_width=True)


if st.button("Average Movie Duration Over the Years"):
    movies = df[df['type'] == 'Movie'].copy()
    movies['duration'] = movies['duration'].astype(str).str.extract('(\d+)').astype(float)
    avg_duration = movies.groupby('release_year')['duration'].mean().reset_index()

    fig = px.line(
        avg_duration,
        x='release_year',
        y='duration',
        markers=True,
        title='Average Movie Duration Over the Years',
        color_discrete_sequence=['#E50914']  
    )

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white', size=14),
        xaxis=dict(
            title=dict(text='Release Year', font=dict(size=16, color='white')),
            dtick=1
        ),
        yaxis=dict(
            title=dict(text='Average Duration (Minutes)', font=dict(size=16, color='white')),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        title=dict(
            text='Average Movie Duration Over the Years',
            font=dict(size=20, color='white', family='Arial Black')
        ),
        margin=dict(l=80, r=50, t=100, b=80)
    )

    st.plotly_chart(fig, use_container_width=True)


if st.button("Top 10 Countries with Most Content"):

    country_df = df.dropna(subset=['country'])
    country_df = country_df.assign(
        country=country_df['country'].str.split(',')
    ).explode('country')
    country_df['country'] = country_df['country'].str.strip()

    top10_countries = country_df['country'].value_counts().head(10).index
    country_df = country_df[country_df['country'].isin(top10_countries)]

    country_counts = country_df.groupby(['country', 'type']).size().reset_index(name='Count')

    fig = px.bar(
        country_counts,
        x='country',
        y='Count',
        color='type',
        barmode='group',
        color_discrete_map={'Movie': '#E50914', 'TV Show': '#808080'},
        labels={'country':'Country', 'Count':'Number of Titles', 'type':'Type'},
        title='Top 10 Countries with Most Netflix Content'
    )

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white', size=14),
        xaxis=dict(title='Country', tickangle=45),
        yaxis=dict(title='Number of Titles', showgrid=True, gridcolor='gray', gridwidth=0.5),
        legend=dict(title='Type', title_font=dict(size=14, color='white'),
                    font=dict(size=12, color='white'),
                    bgcolor='#111111', bordercolor='white', borderwidth=1),
        title_font=dict(size=22, color='white', family='Arial Black'),
        margin=dict(l=80, r=50, t=120, b=120)
    )

    st.plotly_chart(fig, use_container_width=True)

if st.button("Movie vs TV Show Ratio by Country"):
    country_df_clean = df.dropna(subset=['country'])
    country_df_clean = country_df_clean.assign(
        country=country_df_clean['country'].str.split(',')
    ).explode('country')
    country_df_clean['country'] = country_df_clean['country'].str.strip()

    ratio_df = country_df_clean.groupby(['country', 'type']).size().unstack(fill_value=0)
    ratio_df['Total'] = ratio_df.sum(axis=1)
    ratio_df = ratio_df.sort_values('Total', ascending=False).head(10)
    
    ratio_df[['Movie', 'TV Show']] = ratio_df[['Movie', 'TV Show']].div(ratio_df['Total'], axis=0)
    ratio_df = ratio_df[['Movie', 'TV Show']] 

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=ratio_df.index,
        y=ratio_df['Movie'],
        name='Movie',
        marker_color='#E50914'
    ))
    
    fig.add_trace(go.Bar(
        x=ratio_df.index,
        y=ratio_df['TV Show'],
        name='TV Show',
        marker_color='#808080'
    ))

    fig.update_layout(
        barmode='stack',
        template='plotly_dark',
        title="Movie vs TV Show Ratio by Country",
        title_font=dict(size=22, color='white', family='Arial Black'),
        xaxis=dict(title='Country'),
        yaxis=dict(title='Proportion', tickformat='.0%', showgrid=True, gridcolor='gray', gridwidth=0.5),
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white', size=14),
        legend=dict(title='Type', title_font=dict(size=14, color='white'),
                    font=dict(size=12, color='white'),
                    bgcolor='#111111', bordercolor='white', borderwidth=1),
        margin=dict(l=80, r=50, t=120, b=100)
    )

    st.plotly_chart(fig, use_container_width=True)

if st.button("Correlation Heatmap"):
    # Encode features
    encoded = df.copy()
    encoded['type'] = encoded['type'].map({'Movie': 1, 'TV Show': 0})
    encoded['has_rating'] = encoded['rating'].notna().astype(int)
    encoded['has_director'] = encoded['director'].notna().astype(int)
    encoded['has_cast'] = encoded['cast'].notna().astype(int)
    encoded['duration_num'] = encoded['duration'].str.extract('(\d+)').astype(float)
    encoded.loc[encoded['type'] == 0, 'duration_num'] = 0  

    corr_data = encoded[['type', 'duration_num', 'has_rating', 'has_director', 'has_cast']]
    corr_matrix = corr_data.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Reds',
        zmin=-1,
        zmax=1,
        showscale=True,
        hovertemplate='Correlation: %{z:.2f}<extra></extra>',
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"color":"white"}
    ))

    fig.update_layout(
        title="Correlation Heatmap (Encoded Netflix Features)",
        title_font=dict(size=18, color='white', family='Arial Black'),
        template='plotly_dark',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white', size=14)
    )

    st.plotly_chart(fig, use_container_width=True)

if st.button("Top Country-Specific Genres"):
    filtered_data = df[(df['country'].notna()) & (df['country'] != 'Unknown')].copy()
    filtered_data['genre_split'] = filtered_data['listed_in'].str.split(',')
    exploded_data = filtered_data.explode('genre_split')
    exploded_data['genre_split'] = exploded_data['genre_split'].str.strip()

    top_countries = exploded_data['country'].value_counts().head(10).index
    exploded_top = exploded_data[exploded_data['country'].isin(top_countries)]

    def top_genres_per_country(df, top_n=5):
        result = pd.DataFrame()
        for country in df['country'].unique():
            temp = df[df['country'] == country].copy()
            top_genres = temp['genre_split'].value_counts().head(top_n).index
            temp['genre_split'] = temp['genre_split'].apply(lambda x: x if x in top_genres else 'Others')
            result = pd.concat([result, temp])
        return result

    exploded_top = top_genres_per_country(exploded_top, top_n=5)

    genre_ratio = exploded_top.groupby(['country', 'genre_split']).size().unstack(fill_value=0)
    genre_ratio = genre_ratio.div(genre_ratio.sum(axis=1), axis=0)  

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    if 'Others' in genre_ratio.columns:
        colors = colors + ['#808080']  

    for i, genre in enumerate(genre_ratio.columns):
        fig.add_trace(go.Bar(
            x=genre_ratio.index,
            y=genre_ratio[genre],
            name=genre,
            marker_color=colors[i % len(colors)],
            text=(genre_ratio[genre]*100).round(1).astype(str) + '%',
            textposition='inside'
        ))

    fig.update_layout(
        barmode='stack',
        template='plotly_dark',
        title='Top 10 Country-Specific Genres',
        title_font_size=24,
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white', size=14),
        xaxis=dict(title='Country', title_font=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title='Proportion of Genres', title_font=dict(size=18), tickfont=dict(size=14)),
        legend=dict(title='Genre', title_font=dict(size=14), font=dict(size=12), bgcolor='#141414', bordercolor='white', borderwidth=1)
    )

    st.plotly_chart(fig, use_container_width=True)

if st.button("Country-Genre Correlation (Heatmap)"):
    data_clean = df[df['country'].notna() & (df['country'] != 'Unknown')].copy()

    data_clean['genre_split'] = data_clean['listed_in'].str.split(',')
    exploded_data = data_clean.explode('genre_split')
    exploded_data['genre_split'] = exploded_data['genre_split'].str.strip()

    top_countries = exploded_data['country'].value_counts().head(10).index
    exploded_top = exploded_data[exploded_data['country'].isin(top_countries)]

    def top_genres_per_country(df, top_n=5):
        result = pd.DataFrame()
        for country in df['country'].unique():
            temp = df[df['country'] == country].copy()
            top_genres = temp['genre_split'].value_counts().head(top_n).index
            temp['genre_split'] = temp['genre_split'].apply(lambda x: x if x in top_genres else 'Others')
            result = pd.concat([result, temp])
        return result

    exploded_top = top_genres_per_country(exploded_top, top_n=5)

    country_genre = exploded_top.groupby(['country','genre_split']).size().unstack(fill_value=0)
    country_genre_ratio = country_genre.div(country_genre.sum(axis=1), axis=0)
    country_genre_ratio = country_genre_ratio[country_genre_ratio.sum().sort_values(ascending=False).index]

    fig = go.Figure(data=go.Heatmap(
        z=country_genre_ratio.values,
        x=country_genre_ratio.columns,
        y=country_genre_ratio.index,
        colorscale='Reds',
        zmin=0,
        zmax=1,
        colorbar=dict(title="Proportion")
    ))

    for i, country in enumerate(country_genre_ratio.index):
        for j, genre in enumerate(country_genre_ratio.columns):
            val = country_genre_ratio.iloc[i, j]
            color = "white" if val > 0.5 else "black"
            fig.add_annotation(
                x=genre,
                y=country,
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(color=color),
                xanchor='center',
                yanchor='middle'
            )

    fig.update_layout(
        template='plotly_dark',
        title="Country-Genre Correlation (Top 10 Countries & Top 5 Genres)",
        title_font_size=20,
        xaxis_title="Genre",
        yaxis_title="Country",
        xaxis=dict(tickangle=45, tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12)),
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='white')
    )

    st.plotly_chart(fig, use_container_width=True)

if st.button("Content Duration Binning"):
    df_local = df.copy()
    df_local['has_rating'] = df_local['rating'].notna().astype(int)

    def duration_bin(duration_str):
        try:
            mins = int(duration_str.split(' ')[0])
            if mins < 60:
                return 'Short'
            elif 60 <= mins <= 120:
                return 'Medium'
            else:
                return 'Long'
        except:
            return 'Unknown'

    df_local['duration_bin'] = df_local['duration'].fillna('0 min').apply(duration_bin)
    df_filtered = df_local[df_local['duration_bin'] != 'Unknown']

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8,6))
    sns.countplot(
        x='duration_bin',
        data=df_filtered,
        palette='Reds_r',
        order=['Short','Medium','Long'],
        ax=ax
    )

    ax.set_title('Content Duration Binning', fontsize=18, color='white')
    ax.set_xlabel('Duration Category', color='white')
    ax.set_ylabel('Number of Titles', color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#141414')
    ax.grid(alpha=0.3, linestyle='--', color='gray')

    st.pyplot(fig)

 