import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
import pandas as pd
import numpy as np
import math
import streamlit_shadcn_ui as ui
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

@st.cache_resource
def load_data():
    """Load and cache the paper details DataFrame from MongoDB."""
    # Connect to MongoDB
    client = MongoClient(DATABASE_URL)  # Adjust the URI if necessary
    db = client["paperDB"]  # Replace with your MongoDB database name
    collection = db["paper"]  # Replace with your MongoDB collection name

    projection = {'Title': 1, 'Abstract': 1, 'Subject':1, 'Doi':1, 'Source.Date.Year':1, 'Authors':1, '_id': 0}

    # Fetch all documents from the collection
    # papers = collection.find({}, projection) # You can apply queries if needed'

    # Convert MongoDB cursor to DataFrame
    df = pd.DataFrame(papers)

    return df

st.set_page_config(layout="wide")
df = load_data()
st.sidebar.header('Analysis Controls')
subject_author = df[['Source.Date.Year', 'Authors']].explode(column='Authors').reset_index(drop=True)
subject_author['Authors'] = subject_author['Authors'].apply(lambda x : x['Name'])
subject_author = subject_author.groupby(['Source.Date.Year', 'Authors']).size().reset_index(name='Count')
subject_author = subject_author.sort_values(by=['Source.Date.Year', 'Count'], ascending=[True, False])
subject_author = subject_author.reset_index(drop=True)


# data = {"Source.Date.Year": df['Source.Date.Year']}
# my_df = pd.DataFrame(data)

# year_counts = my_df["Source.Date.Year"].value_counts().reset_index()
# year_counts.columns = ['Year', 'Count']
# df_sorted = year_counts.sort_values("Year", ascending=True).reset_index(drop=True)
# _, col, _ = st.columns([1,5,2])
# with col:
tab = ui.tabs(options=["Publications", "Trends Subject Area"], default_value="Publications", key="my_tab_state")

if tab == "Publications":
    st.header("Publications Frequency")
    slectNumber = st.sidebar.slider('Select Number of Authors:',min_value=5, max_value=20, value=10, key="my_slider_state", step=1)
    selectYear = st.sidebar.selectbox("Select Year", options=df["Source.Date.Year"].unique(), key="my_selectbox_state")
    st.subheader("Number of Publications each Year")
    fig = px.histogram(df, x="Source.Date.Year",nbins=15, labels={
            "Source.Date.Year": "Year of Publication",  # เปลี่ยน label ของแกน X
            "count": "Frequency of Publications"       # เปลี่ยน label ของแกน Y
    })
    fig.update_traces(marker=dict(color='#FF3399'))
    st.plotly_chart(fig)
    
    st.subheader("Top Authors that have Published the Most Publications")
    

    col1,col2,col3 = st.columns(3)

    with col1:
        myfirst = subject_author[subject_author['Source.Date.Year'] == "2018"]
        fig = px.bar(myfirst.head(8), x="Authors",y="Count", color="Source.Date.Year")
        fig.update_traces(marker=dict(color='#FF0000'))
        st.plotly_chart(fig)

        myfirst = subject_author[subject_author['Source.Date.Year'] == "2021"]
        fig = px.bar(myfirst.head(8), x="Authors",y="Count", color="Source.Date.Year")
        fig.update_traces(marker=dict(color='#FF3399'))
        st.plotly_chart(fig)
    with col2:
        mysecond = subject_author[subject_author['Source.Date.Year'] == "2019"]
        fig1 = px.bar(mysecond.head(8), x="Authors",y="Count", color="Source.Date.Year",)
        fig1.update_traces(marker=dict(color='#FF8000'))
        st.plotly_chart(fig1)
        
        mysecond = subject_author[subject_author['Source.Date.Year'] == "2022"]
        fig1 = px.bar(mysecond.head(8), x="Authors",y="Count", color="Source.Date.Year",)
        fig1.update_traces(marker=dict(color='#00CCCC'))
        st.plotly_chart(fig1)
    with col3:
        mysecond = subject_author[subject_author['Source.Date.Year'] == "2020"]
        fig1 = px.bar(mysecond.head(8), x="Authors",y="Count", color="Source.Date.Year",)
        fig1.update_traces(marker=dict(color='#0000FF'))
        st.plotly_chart(fig1) 

        mysecond = subject_author[subject_author['Source.Date.Year'] == "2023"]
        fig2 = px.bar(mysecond.head(8), x="Authors",y="Count", color="Source.Date.Year",)
        fig2.update_traces(marker=dict(color='#D1C62B'))
        st.plotly_chart(fig2)
    st.subheader("Table of Top Authors that have Published the Most Publications")
    ui.table(data=subject_author[subject_author['Source.Date.Year'] == selectYear][["Authors", "Source.Date.Year", "Count"]].head(slectNumber), maxHeight=300)

   
elif tab == "Trends Subject Area":
    st.header("Author Publication Trends and Frequencies")
    slectNumber = st.sidebar.slider('Select Number of Authors:',min_value=5, max_value=20, value=10, key="my_slider_state", step=1)
    selectYear = st.sidebar.selectbox("Select Year", options=df["Source.Date.Year"].unique(), key="my_selectbox_state")
    colors = {'setosa': '#FF4B4B', 'versicolor': '#4B4BFF', 'virginica': '#4BFF4B'}
    
    subject_author = df[['Source.Date.Year','Subject', 'Authors']].explode(column='Authors').reset_index(drop=True)
    subject_author['Authors'] = subject_author['Authors'].apply(lambda x : x['Name'])
    subject_author = subject_author.explode(column='Subject')
    
    subject_author = subject_author.groupby(['Source.Date.Year', 'Subject', 'Authors']).size().reset_index(name='Count')
    subject_author = subject_author.sort_values(by=['Source.Date.Year', 'Count'], ascending=[True, False])
    subject_author = subject_author.reset_index(drop=True)
    subject_author = subject_author[['Source.Date.Year', 'Subject', 'Count']] 
    
    skater = subject_author.groupby(['Source.Date.Year' ,'Subject']).size().reset_index(name='Count')
    skater = skater.sort_values(by=['Count'], ascending=[False])
    
    b = skater[skater['Count'] > 1000]
    b = b.sort_values(by=['Source.Date.Year'], ascending=[False])
    b = b.reset_index(drop=True)
    
    
    
    # selectSubject = st.sidebar.selectbox("Select Subject", options=subject_author["Subject"].unique(), key="my_slider_state")
    fig = px.bar(b, x="Subject",y="Count", color="Source.Date.Year",color_discrete_map=colors, barmode='stack')
    # fig.update_traces(marker=dict(color='#FF0000'))
    st.plotly_chart(fig)
    switch_value = ui.switch(default_checked=True, label="show top subject each year", key="switch1")
    if switch_value == True:
        st.header("Top subjects that have pay attention each year")
        col1,col2,col3 = st.columns(3)
        
        with col1:
            myfirst = skater[skater['Source.Date.Year'] == "2018"]
            fig = px.bar(myfirst.head(8), x="Subject",y="Count", color="Source.Date.Year")
            fig.update_traces(marker=dict(color='#FF0000'))
            st.plotly_chart(fig)

            myfirst = skater[skater['Source.Date.Year'] == "2021"]
            fig = px.bar(myfirst.head(8), x="Subject",y="Count", color="Source.Date.Year")
            fig.update_traces(marker=dict(color='#FF3399'))
            st.plotly_chart(fig)
        with col2:
            mysecond = skater[skater['Source.Date.Year'] == "2019"]
            fig1 = px.bar(mysecond.head(8), x="Subject",y="Count", color="Source.Date.Year",)
            fig1.update_traces(marker=dict(color='#FF8000'))
            st.plotly_chart(fig1)

            mysecond = skater[skater['Source.Date.Year'] == "2022"]
            fig1 = px.bar(mysecond.head(8), x="Subject",y="Count", color="Source.Date.Year",)
            fig1.update_traces(marker=dict(color='#00CCCC'))
            st.plotly_chart(fig1)
        with col3:
            mysecond = skater[skater['Source.Date.Year'] == "2020"]
            fig2 = px.bar(mysecond.head(8), x="Subject",y="Count", color="Source.Date.Year",)
            fig2.update_traces(marker=dict(color='#0000FF'))
            st.plotly_chart(fig2)    

            mysecond = skater[skater['Source.Date.Year'] == "2023"]
            fig1 = px.bar(mysecond.head(8), x="Subject",y="Count", color="Source.Date.Year",)
            fig1.update_traces(marker=dict(color='#D1C62B'))
            st.plotly_chart(fig1)

    st.subheader("Table of Top subjects that have pay attention")
    ui.table(data=skater[skater["Source.Date.Year"] == selectYear][["Subject", "Source.Date.Year", "Count"]].head(slectNumber), maxHeight=300)