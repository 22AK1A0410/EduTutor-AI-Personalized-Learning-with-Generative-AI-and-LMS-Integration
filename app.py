import streamlit as st
from ibm_watsonx_ai.foundation_models import ModelInference

# Watsonx credentials and settings
model_id = "ibm/granite-13b-instruct-v2"
project_id = "ef457d57-bdd1-49aa-9e80-53bd5b3afbe8" 
credentials = {
    "url": "https://eu-de.ml.cloud.ibm.com", 
    "apikey": "S1zl_mgeQVqjItIpyy8d6-ZQmA0oP82sJ0x-d5AiPhpk"  
}

# Streamlit UI
st.title("EduTutor AI")

question = st.text_input("Ask your question:")

if st.button("Get Answer") and question.strip() != "":
    model = ModelInference(
        model_id="ibm/granite-13b-instruct-v2",
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 500
        },
        project_id="ef457d57-bdd1-49aa-9e80-53bd5b3afbe8",
        credentials=credentials
    )
    
    response = model.generate(question)
    answer = response["results"][0]["generated_text"]
    st.write(answer)
