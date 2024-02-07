import streamlit as st
import spacy
from spacy import displacy

import requests, tarfile

# URL of the model file on GitHub
model_url = 'https://github.com/roupenminassian/streamlit-example/blob/master/medical-ner.tar.gz'

# Download and extract the model
r = requests.get(model_url, allow_redirects=True)
open(model_url, 'wb').write(r.content)

# Extract the tar.gz file
tar = tarfile.open(model_url, "r:gz")
tar.extractall()
tar.close()

nlp = spacy.load("working/medical-ner")

# Streamlit page setup
st.title("Named Entity Recognition with spaCy")

# Text input
text_input = st.text_area("Enter text here to extract named entities", "Sample text: MEDICAL REPORT. The Sheriff as a condition of granting sick leave with pay, may require medical evidence of sickness or injury acceptable to the Sheriff’s Office when the employee is absent for more than three consecutive working days or when the agency/department head determines within his/her discretion that there are indications of excessive use of sick leave or sick leave abuse.", height=200)

# Process the text input
doc = nlp(text_input)

# Display entities
st.write("Named Entities:")
if doc.ents:
    for ent in doc.ents:
        st.write(ent.text, ent.label_)
else:
    st.write("No named entities found.")

# Using displaCy to visualize entities
st.write("Visualized Named Entities:")
html = displacy.render(doc, style="ent")
st.write(html, unsafe_allow_html=True)
