import streamlit as st

st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
import streamlit_shadcn_ui as ui
from dotenv import load_dotenv
import os
import faiss
from sentence_transformers import SentenceTransformer
import spacy
import string
from nltk.corpus import stopwords
import nltk
import json

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

encore_path = 'en_core_web_sm-3.8.0/'

nltk.download('stopwords')

# Load NLP resources
nlp = spacy.load(encore_path)
stop_words = set(stopwords.words("english"))

@st.cache_resource
def load_data():
    """Load and cache the paper details DataFrame from MongoDB."""
    # Connect to MongoDB
    client = MongoClient(DATABASE_URL)  # Adjust the URI if necessary
    db = client["paperDB"]  # Replace with your MongoDB database name
    collection = db["paper"]  # Replace with your MongoDB collection name

    projection = {'Title': 1, 'Abstract': 1, 'Subject':1, 'Doi':1, 'Source_Date_Year':1, 'Authors':1, '_id': 0}

    # Fetch all documents from the collection
    papers = collection.find({}, projection) # You can apply queries if needed'

    # Convert MongoDB cursor to DataFrame
    df = pd.DataFrame(papers)

    return df

# Cache FAISS index and SentenceTransformer model (global resources)
@st.cache_resource
def load_faiss_index():
    """Load and cache the FAISS index."""
    return faiss.read_index("faiss_index.index")

@st.cache_resource
def load_model():
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_color_dict():
    with open("color_dict.json", "r") as file:
        return json.load(file)

def preprocess_text(text):
    if not isinstance(text, str) or not text:
        return ""

    # Lowercase and remove punctuation
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize using spacy and remove stopwords and non-alphabetic tokens
    doc = nlp(text)

    # Lemmatization (filtering stopwords and non-alphabetic tokens)
    lemmatized_words = [
        token.lemma_ for token in doc if token.text not in stop_words
    ]

    return ' '.join(lemmatized_words)

def perform_similarity_search(query, model, index, df, k):
    """Perform similarity search using FAISS."""
    # Correct spelling and preprocess query
    processed_query = preprocess_text(query)
    
    # Embed the query
    query_embedding = model.encode([processed_query]).reshape(1, -1).astype("float32")
    
    # Search in FAISS index
    distances, indices = index.search(query_embedding, k)
    
    # Collect results
    results = []
    for i, idx in enumerate(indices[0]):
        result = {
            "title": df.iloc[idx]["Title"],
            "abstract": df.iloc[idx]["Abstract"],
            "subject": df.iloc[idx]["Subject"],
            "doi": df.iloc[idx].get("Doi", ""),
            "distance": distances[0][i],
        }
        results.append(result)
    return results

def apply_custom_css():
    st.markdown(
        """
        <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f5f5f7;
            color: #333;
        }
        button {
            background-color: #000;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        button:hover {
            background-color: #1e40af;
        }
        .paper-card {
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        .paper-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 10px;
            color: #000;
        }
        .paper-abstract {
            font-size: 1rem;
            line-height: 1.6;
            color: #555;
            margin-bottom: 15px;
        }
        .subject-badge {
            display: inline-block;
            background-color: #2563eb;
            color: white;
            padding: 5px 10px;
            font-size: 0.875rem;
            font-weight: 500;
            border-radius: 12px;
            margin-right: 5px;
            margin-top: 5px;
        }
        .subject-container {
            margin-bottom: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

color_dict = load_color_dict()

def get_badge_color(subject):
    """Get color from the dictionary based on the subject."""
    return color_dict.get(subject, "#d1d5db")  # Default gray if subject not found

def format_subjects(subjects):
    """Format subjects as badges with colors from color_dict."""
    badges = ""
    for subject in subjects:
        color = get_badge_color(subject.strip())
        badges += f'<span class="subject-badge" style="color: black; background-color: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.5);">{subject.strip()}</span>'

    return badges



# Apply custom CSS for consistent styling
apply_custom_css()

# Load cached resources
df = load_data()
index = load_faiss_index()
model = load_model()

# Sidebar setup
st.sidebar.header('Analysis Controls')

# Process author data for visualizations
author_data = df[['Source_Date_Year', 'Authors']].explode(column='Authors').reset_index(drop=True)
author_data['Authors'] = author_data['Authors'].apply(lambda x: x['Name'])
author_data = (
    author_data.groupby(['Source_Date_Year', 'Authors'])
    .size()
    .reset_index(name='Count')
    .sort_values(by=['Source_Date_Year', 'Count'], ascending=[True, False])
    .reset_index(drop=True)
)

# Tab navigation
tab = ui.tabs(
    options=["Paper Recommendation System", "Publications", "Trends Subject Area"],
    default_value="Paper Recommendation System",
    key="my_tab_state"
)

def plot_top_authors_by_year(data, year, color, col):
    filtered_data = data[data['Source_Date_Year'] == year]
    fig = px.bar(filtered_data.head(8), x="Authors", y="Count", color="Source_Date_Year")
    fig.update_traces(marker=dict(color=color))
    col.plotly_chart(fig)

# Publications Tab
if tab == "Publications":
    st.header("Publications Frequency")
    
    # Sidebar controls
    num_authors = st.sidebar.slider(
        'Select Number of Authors:', min_value=5, max_value=20, value=10, step=1, key="num_authors"
    )
    selected_year = st.sidebar.selectbox(
        "Select Year", options=df["Source_Date_Year"].unique(), key="selected_year"
    )
    
    # Publications histogram
    st.subheader("Number of Publications Each Year")
    fig = px.histogram(
        df, x="Source_Date_Year", nbins=15,
        labels={"Source_Date_Year": "Year of Publication", "count": "Frequency of Publications"}
    )
    fig.update_traces(marker=dict(color='#FF3399'))
    st.plotly_chart(fig)

    # Top authors by publications
    st.subheader("Top Authors with the Most Publications")
    col1, col2, col3 = st.columns(3)

    with col1:
        plot_top_authors_by_year(author_data, "2018", '#FF0000', st)
        plot_top_authors_by_year(author_data, "2021", '#FF3399', st)

    with col2:
        plot_top_authors_by_year(author_data, "2019", '#FF8000', st)
        plot_top_authors_by_year(author_data, "2022", '#00CCCC', st)

    with col3:
        plot_top_authors_by_year(author_data, "2020", '#0000FF', st)
        plot_top_authors_by_year(author_data, "2023", '#D1C62B', st)

    # Authors table
    st.subheader("Top Authors Table")
    ui.table(
        data=author_data[author_data['Source_Date_Year'] == selected_year][["Authors", "Source_Date_Year", "Count"]].head(num_authors),
        maxHeight=300
    )

# Trends Subject Area Tab
elif tab == "Trends Subject Area":
    st.header("Author Publication Trends and Frequencies")
    
    # Sidebar controls
    num_authors = st.sidebar.slider(
        'Select Number of Authors:', min_value=5, max_value=20, value=10, step=1, key="num_authors"
    )
    selected_year = st.sidebar.selectbox(
        "Select Year", options=df["Source_Date_Year"].unique(), key="selected_year"
    )
    
    # Process subject data
    subject_data = (
        df[['Source_Date_Year', 'Subject', 'Authors']]
        .explode(column='Authors')
        .reset_index(drop=True)
    )
    subject_data['Authors'] = subject_data['Authors'].apply(lambda x: x['Name'])
    subject_data = subject_data.explode(column='Subject')
    subject_data = (
        subject_data.groupby(['Source_Date_Year', 'Subject', 'Authors'])
        .size()
        .reset_index(name='Count')
        .sort_values(by=['Source_Date_Year', 'Count'], ascending=[True, False])
        .reset_index(drop=True)
    )

    # Plot trends by subject
    aggregated_subjects = (
        subject_data.groupby(['Source_Date_Year', 'Subject'])
        .size()
        .reset_index(name='Count')
        .sort_values(by=['Count'], ascending=False)
        .reset_index(drop=True)
    )
    filtered_subjects = aggregated_subjects[aggregated_subjects['Count'] > 1000]
    
    fig = px.bar(
        filtered_subjects, x="Subject", y="Count", color="Source_Date_Year", barmode='stack'
    )
    st.plotly_chart(fig)

    # Show top subjects by year
    show_top_subjects = ui.switch(default_checked=True, label="Show Top Subjects Each Year", key="show_subjects")
    if show_top_subjects:
        st.subheader("Top Subjects Each Year")
        col1, col2, col3 = st.columns(3)

        with col1:
            plot_top_authors_by_year(aggregated_subjects, "2018", '#FF0000', st)
            plot_top_authors_by_year(aggregated_subjects, "2021", '#FF3399', st)

        with col2:
            plot_top_authors_by_year(aggregated_subjects, "2019", '#FF8000', st)
            plot_top_authors_by_year(aggregated_subjects, "2022", '#00CCCC', st)

        with col3:
            plot_top_authors_by_year(aggregated_subjects, "2020", '#0000FF', st)
            plot_top_authors_by_year(aggregated_subjects, "2023", '#D1C62B', st)

    # Subjects table
    st.subheader("Top Subjects Table")
    ui.table(
        data=aggregated_subjects[aggregated_subjects["Source_Date_Year"] == selected_year][["Subject", "Source_Date_Year", "Count"]].head(num_authors),
        maxHeight=300
    )

# Paper Recommendation System Tab
elif tab == "Paper Recommendation System":
    st.markdown("<h1 style='text-align: center;'>📚 Paper Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Discover academic papers that match your research interests.</p>", unsafe_allow_html=True)

    # Query input
    query = st.text_input("🔍 Enter your research query:")
    
    # Number of results selection
    k = st.radio("Select Number of Results to Show", options=[5, 10, 20, 50, 100], index=0)

    if query:
        # Perform similarity search
        results = perform_similarity_search(query, model, index, df, k)
        
        st.markdown(f"<h2>🔎 Recommended Papers ({k} results)</h2>", unsafe_allow_html=True)
        for result in results:
            subject_badges = format_subjects(result["subject"])
            st.markdown(
                f"""
                <div class="paper-card">
                    <div class="paper-title">{result['title']}</div>
                    <div class="subject-container">{subject_badges}</div>
                    <div class="paper-abstract">{result['abstract']}</div>
                    <div style="margin-top: 10px;">
                        {"<a href='https://doi.org/" + result['doi'] + "' target='_blank'><button>🔗 Visit Page</button></a>" if result['doi'] else "<p style='color: #999;'>🚫 No Link Available</p>"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
