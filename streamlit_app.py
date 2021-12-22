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
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  with open(uploaded_file,"rb") as fp:# Unpickling
    contents = pickle.load(fp)
    st.write(contents)

#uploaded_file = st.file_uploader(contents)
  
#Preparing model
tokenized_corpus = [doc.split(" ") for doc in contents]

bm25 = BM25Okapi(tokenized_corpus)

user_input = st.text_input('Seed Text (can leave blank)')

tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)

if st.button('Generate Text'):
    generated_test = bm25.get_top_n(tokenized_query, contents, n=1)
    st.write(generated_text)
