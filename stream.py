import streamlit as st
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
import string
from nltk.corpus import stopwords
import nltk
import json
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import zipfile

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

encore_path = 'en_core_web_sm-3.8.0/'

nltk.download('stopwords')

# Load NLP resources
nlp = spacy.load(encore_path)
stop_words = set(stopwords.words("english"))

# Cache DataFrame (serializable)
@st.cache_resource
def load_dataframe():
    """Load and cache the paper details DataFrame from MongoDB."""
    # Connect to MongoDB
    client = MongoClient(DATABASE_URL)  # Adjust the URI if necessary
    db = client["paperDB"]  # Replace with your MongoDB database name
    collection = db["paper"]  # Replace with your MongoDB collection name

    # Fetch all documents from the collection
    papers = collection.find()  # You can apply queries if needed

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
    
color_dict = load_color_dict()

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

def main():
    """Streamlit app main function."""
    apply_custom_css()

    # Load cached resources
    df = load_dataframe()
    index = load_faiss_index()
    model = load_model()

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
        results = perform_similarity_search(query, model, index, df, k)

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
