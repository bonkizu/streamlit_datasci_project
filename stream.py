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
import networkx as nx
from pyvis.network import Network
import pydeck as pdk
import tempfile
from typing import Dict, Optional, Tuple
from itertools import combinations

@st.cache_resource
def load_data():
    """Load and cache the paper details DataFrame from MongoDB."""
    load_dotenv()

    DATABASE_URL = os.getenv('DATABASE_URL')

    # Connect to MongoDB
    client = MongoClient(DATABASE_URL)  # Adjust the URI if necessary
    db = client["paperDB"]  # Replace with your MongoDB database name
    collection = db["paper"]  # Replace with your MongoDB collection name

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

@st.cache_resource
def load_lang():
    encore_path = 'en_core_web_sm-3.8.0/'

    nltk.download('stopwords')

    # Load NLP resources
    nlp = spacy.load(encore_path)
    stop_words = set(stopwords.words("english"))

    return nlp, stop_words

def preprocess_text(text):
    if not isinstance(text, str) or not text:
        return ""
    
    nlp, stop_words = load_lang()

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

@st.cache_data
def load_city_country_coordinate_data():
    return pd.read_csv('city_country_coordinate.csv')

@st.cache_data
def load_coauthor_edges_data():
    return pd.read_csv('coauthor_series.csv')

def get_top_affiliation_coordinate_df(top_city_country_amount):
    top_affiliation_df = load_city_country_coordinate_data().head(top_city_country_amount)
    total_top_papers = sum(top_affiliation_df['papers'])
    top_affiliation_df['papers'] = top_affiliation_df['papers'] / total_top_papers
    top_affiliation_df.rename(columns={'papers':'local_paper_portions'}, inplace=True)
    return top_affiliation_df

def get_coauthor_series(edge_amount):
    return load_coauthor_edges_data().head(edge_amount)

class NetworkVisualizer:
    def __init__(self, G: nx.Graph):
        self.G = G
        self.colors = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", 
            "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
            "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000"
        ]

    def _get_layout(self, layout_type: str, G: nx.Graph, k_space: float = 2.0):
        """Calculate layout positions with adjustable spacing"""
        if layout_type == "spring":
            k = 1/np.sqrt(len(G.nodes())) * k_space
            return nx.spring_layout(G, k=k, iterations=50, seed=42)
        elif layout_type == "kamada_kawai":
            return nx.kamada_kawai_layout(G)
        elif layout_type == "circular":
            return nx.circular_layout(G)
        elif layout_type == "random":
            return nx.random_layout(G, seed=42)
        else:
            return nx.spring_layout(G)

    def create_interactive_network(
        self, 
        communities: Optional[Dict] = None,
        layout: str = "spring",
        centrality_metric: str = "degree",
        scale_factor: float = 1000,
        node_spacing: float = 2.0,
        node_size_range: Tuple[int, int] = (10, 50),
        show_edges: bool = True, 
        font_size: int = 14 
    ) -> str:
        # Get layout positions with adjustable spacing
        pos = self._get_layout(layout, self.G, node_spacing)
        
        # Scale positions
        pos = {node: (coord[0] * scale_factor, coord[1] * scale_factor) 
               for node, coord in pos.items()}

        # Calculate centrality
        try:
            if centrality_metric == "degree":
                centrality = nx.degree_centrality(self.G)
            elif centrality_metric == "betweenness":
                centrality = nx.betweenness_centrality(self.G)
            elif centrality_metric == "closeness":
                centrality = nx.closeness_centrality(self.G)
            else:  # pagerank
                centrality = nx.pagerank(self.G)
        except:
            centrality = nx.degree_centrality(self.G)
            st.warning(f"Failed to compute {centrality_metric} centrality, using degree centrality instead.")

        # Scale node sizes
        min_cent, max_cent = min(centrality.values()), max(centrality.values())
        min_size, max_size = node_size_range
        if max_cent > min_cent:
            size_scale = lambda x: min_size + (x - min_cent) * (max_size - min_size) / (max_cent - min_cent)
        else:
            size_scale = lambda x: (min_size + max_size) / 2

        # Create a copy of the graph to modify attributes
        G_vis = self.G.copy()

        # Prepare color map for communities if present
        if communities:
            unique_communities = sorted(set(communities.values()))
            color_map = {com: self.colors[i % len(self.colors)] 
                        for i, com in enumerate(unique_communities)}

        # Set node attributes all at once
        for node in G_vis.nodes():
            G_vis.nodes[node].update({
                'label': str(node),
                'size': size_scale(centrality[node]),
                'x': pos[node][0],
                'y': pos[node][1],
                'physics': False,
                'title': (f"Node: {node}\nDegree: {self.G.degree(node)}\nCommunity: {communities[node]}"
                         if communities else
                         f"Node: {node}\nDegree: {self.G.degree(node)}"),
                'color': color_map[communities[node]] if communities else None
            })

        # Create network
        nt = Network(
            height="720px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=self.G.is_directed()
        )

        # Convert from networkx to pyvis
        nt.from_nx(G_vis)
        
        # Disable physics
        nt.toggle_physics(False)

        # Set visualization options
        nt.set_options("""
        {
            "nodes": {
                "font": {"size": %d},
                "borderWidth": 2,
                "borderWidthSelected": 3,
                "shape": "dot"
            },
            "edges": {
                "color": {"color": "#666666"},
                "width": 1.5,
                "smooth": {
                    "type": "continuous",
                    "roundness": 0.5
                },
                "hidden": %s
            },
            "physics": {
                "enabled": false
            },
            "interaction": {
                "hover": true,
                "multiselect": true,
                "navigationButtons": true,
                "tooltipDelay": 100,
                "zoomView": true,
                "dragView": true,
                "zoomSpeed": 0.5,
                "minZoom": 1.0,
                "maxZoom": 2.5
            }
        }
        """ % (font_size, str(not show_edges).lower()))

        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as tmp:
            nt.save_graph(tmp.name)
            return tmp.name
        
@st.cache_data
def detect_communities(edges_str: str):
    """Community detection"""
    # Recreate graph from edges string
    edges = eval(edges_str)
    G = nx.Graph(edges)
    return list(nx.community.greedy_modularity_communities(G))

def main():

    apply_custom_css()

    # Load cached resources
    subject_author, subject_subject, year_counts, collection = load_data()
    index = load_faiss_index()
    model = load_model()

    years = ["2018", "2019", "2020", "2021", "2022", "2023"]
    colors = ['#FF0000', '#FF8000', '#0000FF', '#FF3399', '#00CCCC', '#D1C62B']

    st.sidebar.header('Analysis Controls')

    tab = ui.tabs(options=["Paper Recommendation System", "Publications", "Trends Subject Area", "Visualization"], default_value="Paper Recommendation System", key="my_tab_state")
    if tab == "Publications":
        st.header("Publications Frequency")
        slectNumber = st.sidebar.slider('Select Number of Authors:', min_value=5, max_value=20, value=10, key="pub_slider_state", step=1)
        selectYear = st.sidebar.selectbox("Select Year", options=subject_author["Source_Date_Year"].unique(), key="pub_selectbox_state")

        st.subheader("Number of Publications each Year")
        fig = px.histogram(year_counts, x="Source_Date_Year", y="count", nbins=15, labels={
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
    elif tab == "Visualization":
        st.title("Data Visualization")

        with st.sidebar: 
            # Top Affiliation Sidebar
            st.subheader("Top Affiliation Spatial Visualization")
            top_city_country_amount = st.slider(
                "Top (n) Authors' Affiliation City & Country", 
                min_value=1, 
                max_value=100, 
                value=10
            )
            top_affiliation_coordinate_df = get_top_affiliation_coordinate_df(top_city_country_amount)

            # Co-authorship Sidebar
            st.subheader("Co-Author Network Visualization Options")
            coauthor_edge_amount = st.slider(
                "Co-authoship Edges Amount", 
                min_value=1, 
                max_value=500, 
                step=1
            )
            coauthor_series = get_coauthor_series(coauthor_edge_amount)
            G = nx.Graph()

            for coauthor_edge in coauthor_series['Authors']:
                coauthor_edge = eval(coauthor_edge)
                G.add_edge(coauthor_edge[0], coauthor_edge[1])

            layout_option = st.selectbox(
                "Layout Algorithm",
                ["spring", "kamada_kawai", "circular", "random"]
            )
            centrality_option = st.selectbox(
                "Node Size By", 
                ["degree", "betweenness", "closeness", "pagerank"]
            )

            # Size Controls
            scale_factor = st.slider(
                "Graph Size", 
                min_value=500, 
                max_value=3000, 
                value=1000,
                step=100,
                help="Adjust the overall size of the graph"
            )
            if layout_option == "spring":
                node_spacing = st.slider(
                    "Node Spacing",
                    min_value=1.0,
                    max_value=20.0,
                    value=5.0,
                    step=1.0,
                    help="Adjust the spacing between nodes (only for spring layout)"
                )
            else:
                node_spacing = 2.0

            node_size_range = st.slider(
                "Node Size Range",
                min_value=5,
                max_value=200,
                value=(10, 50),
                step=5,
                help="Set the minimum and maximum node sizes"
            )
            font_size = st.slider(
                "Label Font Size",
                min_value=8,
                max_value=40,
                value=16,
                step=2,
                help="Adjust the font size of node labels"
            )
            show_edges = st.toggle(
                "Show Edges",
                value=True,  
                help="Toggle edge visibility"
            )

            show_communities = st.checkbox("Detect Communities")
            communities = None

            if show_communities:
                try:
                    edges_str = str(list(G.edges()))
                    communities_iter = detect_communities(edges_str)
                    communities = {node: idx for idx, community in enumerate(communities_iter) for node in community}
                except Exception as e:
                    st.warning(f"Could not detect communities: {str(e)}")

        # Tabs for Spatial and Network Visualization
        tab_spatial, tab_network = st.tabs(["Spatial Visualization", "Network Visualization"])

        with tab_spatial:
            st.header("Top Authors' Affiliation City & Country")

            st.subheader('Scatter Plot Spatial Analysis')

            view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1, pitch=0)

            scatterplot_layer = pdk.Layer(
                "ScatterplotLayer",
                top_affiliation_coordinate_df,
                get_position=['longitude', 'latitude'],
                opacity=0.8,
                get_radius=200000,
                get_fill_color=[30, 0, 255],
                pickable=True
            )

            st.pydeck_chart(pdk.Deck(layers=[scatterplot_layer], initial_view_state=view_state, map_style="light"))

            st.subheader('Heatmap Spatial Analysis')
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                top_affiliation_coordinate_df,
                get_position=['longitude', 'latitude'],
                opacity=0.8,
                get_weight="local_paper_portions",
                color_range=[
                    [150, 150, 200],
                    [80, 80, 200],
                    [60, 60, 225],
                    [30, 0, 255]
                ],
                pickable=True
            )

            st.pydeck_chart(pdk.Deck(layers=[heatmap_layer], initial_view_state=view_state, map_style="light"))

            st.header('Top Affiliation City & Country Table')
            top_affiliation_table = top_affiliation_coordinate_df.copy()
            top_affiliation_table.rename(columns={'local_paper_portions':'papers'}, inplace=True)
            top_affiliation_table['papers'] = top_affiliation_table['papers'] * sum(load_city_country_coordinate_data().head(top_city_country_amount)['papers'])
            st.dataframe(top_affiliation_table)
            
        with tab_network:
            st.header("First 40 Papers Co-authorship Network Visualization")
            network_visualizer = NetworkVisualizer(G)
            html_file = network_visualizer.create_interactive_network(
                communities=communities,
                layout=layout_option,
                centrality_metric=centrality_option,
                scale_factor=scale_factor,
                node_spacing=node_spacing,
                node_size_range=node_size_range,
                show_edges=show_edges,
                font_size=font_size
            )
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=800)
            os.unlink(html_file)

if __name__ == "__main__":
    main()