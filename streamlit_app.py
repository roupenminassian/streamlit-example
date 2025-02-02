############ 1. IMPORTING LIBRARIES ############

# Import streamlit, requests for API calls, and pandas and numpy for data manipulation

import streamlit as st
import requests
import pandas as pd
import numpy as np
from streamlit_tags import st_tags  # to add labels on the fly!
from annotated_text import annotated_text


############ 2. SETTING UP THE PAGE LAYOUT AND TITLE ############

# `st.set_page_config` is used to display the default layout width, the title of the app, and the emoticon in the browser tab.

st.set_page_config(
    layout="centered", page_title="Medical NER Text Classifier"
)

############ CREATE THE LOGO AND HEADING ############

# We create a set of columns to display the logo and the heading next to each other.


c1, c2 = st.columns([0.32, 2])

# The heading will be on the right.

with c2:

    st.caption("")
    st.title("Medical NER Text Classifier")


# We need to set up session state via st.session_state so that app interactions don't reset the app.

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False


############ SIDEBAR CONTENT ############

# For elements to be displayed in the sidebar, we need to add the sidebar element in the widget.

# We create a text input field for users to enter their API key.

API_KEY = st.secrets["my_secret"]["HF_KEY"]

# Adding the HuggingFace API inference URL.
API_URL_1 = "https://ppsk5964bte2mrza.us-east-1.aws.endpoints.huggingface.cloud"
API_URL_2 = "https://kxo555j5ehh3adrf.us-east-1.aws.endpoints.huggingface.cloud"

# Now, let's create a Python dictionary to store the API headers.
headers = {"Accept" : "application/json","Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json" }


st.sidebar.markdown("---")


# Let's add some info about the app to the sidebar.

st.sidebar.write(
    """

App created by [Roupen Minassian](https://github.com/roupenminassian) using [Streamlit](https://streamlit.io/) and [HuggingFace](https://huggingface.co/inference-api)

"""
)


############ TABBED NAVIGATION ############

# First, we're going to create a tabbed navigation for the app via st.tabs()
# tabInfo displays info about the app.
# tabMain displays the main app.

MainTab, InfoTab = st.tabs(["Main", "Info"])

with InfoTab:

    st.subheader("What is Streamlit?")
    st.markdown(
        "[Streamlit](https://streamlit.io) is a Python library that allows the creation of interactive, data-driven web applications in Python."
    )

    st.subheader("What is HuggingFace?")
    st.markdown(
        "[HuggingFace](https://huggingface.co/) is a leading provider of state-of-the-art natural language processing models and tools."
    )


with MainTab:

    # Then, we create a intro text for the app, which we wrap in a st.markdown() widget.

    st.write("")
    st.markdown(
        """

     Instantly analyze medical text or doctor's notes with a specialized NER classifier!

    """
    )

    st.write("")

    # Now, we create a form via `st.form` to collect the user inputs.

    # All widget values will be sent to Streamlit in batch.
    # It makes the app faster!

    with st.form(key="my_form"):

        ############ ST TAGS ############

        # We initialize the st_tags component with default "labels"

        # Here, we want to classify the text into one of the following user intents:
        # Transactional
        # Informational
        # Navigational

        # The block of code below is to display some text samples to classify.
        # This can of course be replaced with your own text samples.

        # MAX_KEY_PHRASES is a variable that controls the number of phrases that can be pasted:
        # The default in this app is 50 phrases. This can be changed to any number you like.

        MAX_KEY_PHRASES = 50

        pre_defined_keyphrases = "John Doe, a 45-year-old male with a history of hypertension and diabetes, presented with a 3-month history of persistent lower back pain radiating to the left leg, exacerbated by prolonged sitting or standing. Physical examination revealed tenderness in the lower lumbar region without neurological deficits. The patient's medications include Lisinopril and Metformin, with occasional ibuprofen for pain. There's no significant family history contributing to the current condition. Considering the clinical presentation, a lumbar strain is suspected, and conservative management with physical therapy and NSAIDs is recommended, with a follow-up in 4 weeks to assess progress."

        # The block of code below displays a text area
        # So users can paste their phrases to classify

        text = st.text_area(
            # Instructions
            "Enter medical text to classify",
            # 'sample' variable that contains our keyphrases.
            pre_defined_keyphrases,
            # The height
            height=200,
            # The tooltip displayed when the user hovers over the text area.
            help="Enter medical text, such as prescriptions or patient conditions, for accurate classification.",
            key="1",
        )

        submit_button = st.form_submit_button(label="Submit")

    ############ CONDITIONAL STATEMENTS ############

    # Now, let us add conditional statements to check if users have entered valid inputs.
    # E.g. If the user has pressed the 'submit button without text, without labels, and with only one label etc.
    # The app will display a warning message.

    if not submit_button and not st.session_state.valid_inputs_received:
        st.stop()

    elif submit_button and not text:
        st.warning("❄️ There is no keyphrases to classify")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button or st.session_state.valid_inputs_received:

        if submit_button:

            # The block of code below if for our session state.
            # This is used to store the user's inputs so that they can be used later in the app.

            st.session_state.valid_inputs_received = True

        ############ MAKING THE API CALL ############

        # First, we create a Python function to construct the API call.

        def query(payload):
            response = requests.post(API_URL_1, headers=headers, json=payload)
            return response.json()

        def query_2(payload):
            response = requests.post(API_URL_2, headers=headers, json=payload)
            return response.json()
            
        # The function will send an HTTP POST request to the API endpoint.
        # This function has one argument: the payload
        # The payload is the data we want to send to HugggingFace when we make an API request

        # The payload is composed of:
        #   1. the keyphrase
        #   2. the labels
        #   3. the 'wait_for_model' parameter set to "True", to avoid timeouts!

        api_json_output = query(
                {
                    "inputs": text,
                    "parameters": {"aggregation_strategy": "average", "stride": 1}
                }
            )

        medication_entities = [entity for entity in api_json_output if entity["entity_group"] == "Medication"]
        medication_entities_string = ', '.join([entity['word'] for entity in medication_entities])

        st.success("✅ Done!")

        st.caption("")
        st.markdown("### Check the results!")
        st.caption("")

        # st.write(df)

        ############ DATA WRANGLING ON THE RESULTS ############
        # Various data wrangling to get the data in the right format!

        # Function to format annotated text
        def format_annotated_text(text, entities):
            annotated = []
            last_end = 0
            for entity in entities:
                annotated.append(text[last_end:entity["start"]])
                annotated.append((text[entity["start"]:entity["end"]], entity["entity_group"]))
                last_end = entity["end"]
            annotated.append(text[last_end:])
            return annotated
        
        # Display annotated text
        annotated_text(*format_annotated_text(text, api_json_output))
