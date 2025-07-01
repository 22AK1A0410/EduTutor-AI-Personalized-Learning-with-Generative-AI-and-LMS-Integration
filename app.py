from ibm_watsonx_ai.foundation_models import ModelInference

model_id = "ibm/granite-3-8b-instruct"  
project_id = "ef457d57-bdd1-49aa-9e80-53bd5b3afbe8" 

credentials = {
    "url": "https://eu-de.ml.cloud.ibm.com",  
    "apikey": "S1zl_mgeQVqjItIpyy8d6-ZQmA0oP82sJ0x-d5AiPhpk"  
}

model = ModelInference(
    model_id=model_id,
    params={"decoding_method": "greedy"},  
    project_id=project_id,
    credentials=credentials
)

prompt = "Explain the concept of ai"

response = model.generate(prompt)

print("Generated Text:",response)