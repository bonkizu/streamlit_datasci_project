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
from bson import ObjectId

@st.cache_resource
def load_data():
    """Load and cache the paper details DataFrame from MongoDB."""
    load_dotenv()

    DATABASE_URL = os.getenv('DATABASE_URL')

    # Connect to MongoDB
    client = MongoClient(DATABASE_URL)  # Adjust the URI if necessary
    db = client["paperDB"]  # Replace with your MongoDB database name
    collection = db["paper"]  # Replace with your MongoDB collection name

    # projection = {'Subject':1, 'Source_Date_Year':1, 'Authors':1, '_id': 0}

    # # Fetch all documents from the collection
    # papers = collection.find({}, projection) # You can apply queries if needed'

    # # Convert MongoDB cursor to DataFrame
    # df = pd.DataFrame(papers)

    # subject_author = df[['Source_Date_Year', 'Authors']].explode(column='Authors').reset_index(drop=True)
    # subject_author['Authors'] = subject_author['Authors'].apply(lambda x: x['Name'])
    # subject_author = subject_author.groupby(['Source_Date_Year', 'Authors']).size().reset_index(name='Count')
    # subject_author = subject_author.sort_values(by=['Source_Date_Year', 'Count'], ascending=[True, False]).reset_index(drop=True)

    # # Subject data
    # subject_subject = df[['Source_Date_Year', 'Subject', 'Authors']].explode(column='Authors').reset_index(drop=True)
    # subject_subject['Authors'] = subject_subject['Authors'].apply(lambda x: x['Name'])
    # subject_subject = subject_subject.explode(column='Subject')
    # subject_subject = subject_subject.groupby(['Source_Date_Year', 'Subject']).size().reset_index(name='Count')
    # subject_subject = subject_subject.sort_values(by=['Source_Date_Year', 'Count'], ascending=[True, False]).reset_index(drop=True)

    # print("Load Data")

    subject_author = pd.read_pickle('subject_author.pkl')
    subject_subject = pd.read_pickle('subject_subject.pkl')
    year_counts = pd.read_pickle('year_counts.pkl')

    return subject_author, subject_subject, year_counts, collection

# Cache FAISS index and SentenceTransformer model (global resources)
@st.cache_resource
def load_faiss_index():
    """Load and cache the FAISS index."""
    return faiss.read_index("faiss_index.index")

@st.cache_resource
def load_model():
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

# @st.cache_resource
@st.cache_resource
def load_color_dict():
    with open("color_dict.json", "r") as file:
        return json.load(file)
    
@st.cache_resource
def load_index_dict():
    with open("index_id_map.json", "r") as file:
        return json.load(file)

def preprocess_text(text):
    if not isinstance(text, str) or not text:
        return ""

    encore_path = 'en_core_web_sm-3.8.0/'

    nltk.download('stopwords')

    # Load NLP resources
    nlp = spacy.load(encore_path)
    stop_words = set(stopwords.words("english"))

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

def perform_similarity_search(query, model, index, k, collection):
    """Perform similarity search using FAISS."""
    # Correct spelling and preprocess query
    processed_query = preprocess_text(query)
    
    # Embed the query
    query_embedding = model.encode([processed_query]).reshape(1, -1).astype("float32")
    
    # Search in FAISS index
    distances, indices = index.search(query_embedding, k)
    
    # Collect results
    # results = []
    # for i, idx in enumerate(indices[0]):
    #     result = {
    #         "title": df.iloc[idx]["Title"],
    #         "abstract": df.iloc[idx]["Abstract"],
    #         "subject": df.iloc[idx]["Subject"],
    #         "doi": df.iloc[idx].get("Doi", ""),
    #         "distance": distances[0][i],
    #     }
    #     results.append(result)
    # return results

    index_dict = load_index_dict()

    results = []
    for i, idx in enumerate(indices[0]):
        # Get the MongoDB _id from df and convert to ObjectId
        document_id = index_dict[str(idx)]
        
        # Query MongoDB to get the document by _id
        document = collection.find_one({"_id": ObjectId(document_id)})
        
        # Prepare result with document details
        if document:
            result = {
                "title": document.get("Title", ""),
                "abstract": document.get("Abstract", ""),
                "subject": document.get("Subject", ""),
                "doi": document.get("Doi", ""),
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

def get_badge_color(subject):
    """Get color from the dictionary based on the subject."""
    return load_color_dict().get(subject, "#d1d5db")  # Default gray if subject not found

def format_subjects(subjects):
    """Format subjects as badges with colors from color_dict."""
    badges = ""
    for subject in subjects:
        color = get_badge_color(subject.strip())
        badges += f'<span class="subject-badge" style="color: black; background-color: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.5);">{subject.strip()}</span>'

    return badges

# def preprocess_author_data(df):
#     """Preprocess data for authors and subjects."""
#     # Author data
#     subject_author = df[['Source_Date_Year', 'Authors']].explode(column='Authors').reset_index(drop=True)
#     subject_author['Authors'] = subject_author['Authors'].apply(lambda x: x['Name'])
#     subject_author = subject_author.groupby(['Source_Date_Year', 'Authors']).size().reset_index(name='Count')
#     subject_author = subject_author.sort_values(by=['Source_Date_Year', 'Count'], ascending=[True, False]).reset_index(drop=True)

#     # Subject data
#     subject_subject = df[['Source_Date_Year', 'Subject', 'Authors']].explode(column='Authors').reset_index(drop=True)
#     subject_subject['Authors'] = subject_subject['Authors'].apply(lambda x: x['Name'])
#     subject_subject = subject_subject.explode(column='Subject')
#     subject_subject = subject_subject.groupby(['Source_Date_Year', 'Subject']).size().reset_index(name='Count')
#     subject_subject = subject_subject.sort_values(by=['Source_Date_Year', 'Count'], ascending=[True, False]).reset_index(drop=True)

    
#     return subject_author, subject_subject

def main():

    apply_custom_css()

    # Load cached resources
    subject_author, subject_subject, year_counts, collection = load_data()
    index = load_faiss_index()
    model = load_model()

    years = ["2018", "2019", "2020", "2021", "2022", "2023"]
    colors = ['#FF0000', '#FF8000', '#0000FF', '#FF3399', '#00CCCC', '#D1C62B']

    st.sidebar.header('Analysis Controls')

    tab = ui.tabs(options=["Paper Recommendation System", "Publications", "Trends Subject Area"], default_value="Paper Recommendation System", key="my_tab_state")
    if tab == "Publications":
        st.header("Publications Frequency")
        slectNumber = st.sidebar.slider('Select Number of Authors:', min_value=5, max_value=20, value=10, key="pub_slider_state", step=1)
        selectYear = st.sidebar.selectbox("Select Year", options=subject_author["Source_Date_Year"].unique(), key="pub_selectbox_state")

        st.subheader("Number of Publications each Year")
        fig = px.histogram(year_counts, x="Source_Date_Year", nbins=15, labels={
            "Source_Date_Year": "Year of Publication",
            "count": "Frequency of Publications"
        })
        fig.update_traces(marker=dict(color='#FF3399'))
        st.plotly_chart(fig)

        st.subheader("Top Authors with Most Publications")
        col1, col2, col3 = st.columns(3)

        for i, year in enumerate(years):
            col = [col1, col2, col3][i % 3]
            with col:
                data = subject_author[subject_author['Source_Date_Year'] == year]
                fig = px.bar(data.head(8), x="Authors", y="Count", color="Source_Date_Year", color_discrete_sequence=[colors[i]])
                st.plotly_chart(fig, key=year)
        
        st.subheader("Table of Top Authors")
        ui.table(data=subject_author[subject_author["Source_Date_Year"] == selectYear][["Authors", "Source_Date_Year", "Count"]].head(slectNumber), maxHeight=300)


    
    elif tab == "Trends Subject Area":
        st.header("Author Publication Trends and Frequencies")
        slectNumber = st.sidebar.slider('Select Number of Subjects:', min_value=5, max_value=20, value=10, key="trends_slider_state", step=1)
        selectYear = st.sidebar.selectbox("Select Year", options=subject_subject["Source_Date_Year"].unique(), key="trends_selectbox_state")

        st.subheader("Subject Trends Across Years")
        b = subject_subject[subject_subject["Count"] > 1000]
        fig = px.bar(b, x="Subject", y="Count", color="Source_Date_Year", barmode="stack")
        st.plotly_chart(fig)

        if ui.switch(default_checked=True, label="Show Top Subjects by Year", key="trends_switch"):
            st.subheader("Top Subjects by Year")
            col1, col2, col3 = st.columns(3)
            for i, year in enumerate(years):
                col = [col1, col2, col3][i % 3]
                with col:
                    data = subject_subject[subject_subject['Source_Date_Year'] == year]
                    fig = px.bar(data.head(8), x="Subject", y="Count", color="Source_Date_Year", color_discrete_sequence=[colors[i]])
                    st.plotly_chart(fig, key=year)

        st.subheader("Table of Top Subjects")
        ui.table(data=subject_subject[subject_subject["Source_Date_Year"] == selectYear][["Subject", "Source_Date_Year", "Count"]].head(slectNumber), maxHeight=300)


    elif tab == "Paper Recommendation System":
        # App title
        st.markdown("<h1 style='text-align: center;'>📚 Paper Recommendation System</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666;'>Discover academic papers that match your research interests and provide valuable insights.</p>", unsafe_allow_html=True)


        # Query input
        query = st.text_input("🔍 Enter your research query:")
        
        # Checkboxes for number of results to display
        k = st.radio(
            "Select number of results to show",
            options=[5, 10, 20, 50, 100],
            index=0,  # Default to 5
        )

        if query:
            # Perform similarity search for the selected k value
            results = perform_similarity_search(query, model, index, k, collection)

            st.markdown(f"<h2 style='margin-top: 20px;'>🔎 Recommended Papers ({k} results)</h2>", unsafe_allow_html=True)
            for result in results:
                subject_badges = format_subjects(result["subject"])
                st.markdown(
                    f"""
                    <div class="paper-card">
                        <div class="paper-title">{result['title']}</div>
                        <div class="subject-container">{subject_badges}</div>
                        <div class="paper-abstract">{result['abstract']}</div>
                        <div style="margin-top: 10px;">
                            {"<a href='https://doi.org/" + result['doi'] + "' target='_blank' style='text-decoration: none; color: white;'><button>🔗 Visit page</button></a>" if result['doi'] != "" else "<p style='color: #999;'>🚫 No Link available</p>"}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

if __name__ == "__main__":
    main()