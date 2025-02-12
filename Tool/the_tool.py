import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

def load_dataset(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    prompts = []
    for category, items in data.items():
        for text, score in items.items():
            prompts.append({"text": text, "score": score})
    return prompts

def preprocess_prompts(prompts, model):
    embeddings = model.encode([p["text"] for p in prompts], convert_to_tensor=True)
    return embeddings

def classify_prompt(new_prompt, model, embeddings, prompts):
    new_embedding = model.encode(new_prompt, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(new_embedding, embeddings)[0]
    best_match_idx = np.argmax(similarities.cpu().numpy())
    best_match_score = similarities[best_match_idx].cpu().item()
    
    if best_match_score < 0.5:
        raise ValueError("The prompt must be a request to generate Python code.")
    
    return prompts[best_match_idx]["score"], best_match_score

# UI with Streamlit
st.title("Prompt Classification with LLM")
st.write("Enter a prompt to get the assigned score and similarity compared to existing data.")

model = SentenceTransformer("all-MiniLM-L6-v2")
json_file = "dataset.json"
prompts = load_dataset(json_file)
embeddings = preprocess_prompts(prompts, model)

new_prompt = st.text_area("Enter your prompt:")
if st.button("Classify"):
    if new_prompt:
        try:
            score, similarity = classify_prompt(new_prompt, model, embeddings, prompts)
            st.metric(label="Assigned Score", value=round(score, 2))
            st.metric(label="Similarity", value=f"{similarity:.2f}")
        except ValueError as e:
            st.error(str(e))
    else:
        st.warning("Please enter a valid prompt.")