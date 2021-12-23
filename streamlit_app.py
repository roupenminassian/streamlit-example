import pandas as pd
import streamlit as st

import pickle
from rank_bm25 import BM25Okapi

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

In the meantime, below is an example of what you can do with just a few lines of code:
"""

#Load documents
input = st.file_uploader('')
    
if input is None:
    st.write("Or use sample dataset to try the application")
    sample = st.checkbox("Download sample data from GitHub")

    try:
        if sample:
            st.markdown("""[download_link](https://gist.github.com/roupenminassian/0a17d0bf8a6410dbb1b9d3f42462c063)""")
    
    except:
        pass

else:
    with open("test.txt","rb") as fp:# Unpickling
        contents = pickle.load(fp)
  
    #Preparing model
    tokenized_corpus = [doc.split(" ") for doc in contents]

    bm25 = BM25Okapi(tokenized_corpus)
    
    user_input = st.text_input('Seed Text (can leave blank)')

    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)

    if st.button('Generate Text'):
        generated_test = bm25.get_top_n(tokenized_query, contents, n=1)
        st.write(generated_text)
