import streamlit as st
import os
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
import pinecone

# Load environment variables
load_dotenv()

# Watsonx setup
watsonx_creds = Credentials(
    url=os.getenv("WATSONX_ENDPOINT"),
    apikey=os.getenv("WATSONX_API_KEY")
)

model_id = os.getenv("WATSONX_MODEL_ID")
project_id = os.getenv("WATSONX_PROJECT_ID")

model = ModelInference(
    model_id=model_id,
    credentials=watsonx_creds,
    project_id=project_id
)

# Pinecone setup
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")  # Change env if needed
index = pinecone.Index(pinecone_index_name)

# Streamlit UI
st.title("🧠 EduTutor AI App")

user_input = st.text_area("Enter your question:", "")

if st.button("Submit"):
    with st.spinner("Generating response..."):
        result = model.generate(prompt=user_input, max_tokens=200)
        response = result.get("results")[0]["generated_text"]
        st.success(response)
