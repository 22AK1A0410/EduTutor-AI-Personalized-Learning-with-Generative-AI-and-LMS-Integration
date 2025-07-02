import os
import streamlit as st
from dotenv import load_dotenv
import pinecone
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.watsonx_client import WatsonxClient

# Load environment variables
load_dotenv()

# Watsonx Setup
watsonx = WatsonxClient(
    url=os.getenv("WATSONX_ENDPOINT"),
    apikey=os.getenv("WATSONX_API_KEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID")
)

model = ModelInference(
    model_id=os.getenv("WATSONX_MODEL_ID"),
    params={"decoding_method": "greedy", "max_new_tokens": 200},
    client=watsonx
)

# Pinecone Setup
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

# Streamlit UI
st.title("🎓 EduTutor AI")
query = st.text_input("Ask your study question:")

if st.button("Get Answer") and query:
    # Embed the query (dummy 1536 vector used here; replace with real embedder)
    dummy_embedding = [0.01] * 1536
    result = index.query(vector=dummy_embedding, top_k=3, include_metadata=True)

    # Collect retrieved context
    context = "\n".join([item['metadata']['text'] for item in result['matches']])

    # Prompt for Watsonx
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Get response from Watsonx
    try:
        response = model.generate_text(prompt=prompt)
        st.markdown("### 📘 Answer:")
        st.write(response)
    except Exception as e:
        st.error(f"Error from Watsonx: {e}")