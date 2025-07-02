import streamlit as st
from ibm_watsonx_ai.foundation_models import ModelInference

# Watsonx credentials and settings
model_id = "ibm/granite-3-8b-instruct"
project_id = "d92838f9-a4f7-4728-91c8-0ce485d35bce" 
credentials = {
    "url": "https://eu-de.ml.cloud.ibm.com", 
    "apikey": "b5Gu5kQpy1-gIVKlujwF5yOSK8vuMYQERhURhPuD_4ju"  
}

# Streamlit UI
st.title("EduTutor AI")

question = st.text_input("Ask your question:")

if st.button("Get Answer") and question.strip() != "":
    model = ModelInference(
        model_id=model_id,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 500
        },
        project_id=project_id,
        credentials=credentials
    )

    response = model.generate(question)
    answer = response["results"][0]["generated_text"]
    st.write(answer)
