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
    best_match_idx = np.argmax(similarities.cpu().numpy())  # Sposta il tensore sulla CPU
    return prompts[best_match_idx]["score"], similarities[best_match_idx].cpu().item()

# Carichiamo il modello LLM per ottenere embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Carichiamo il dataset
json_file = "dataset.json"  # Sostituisci con il nome corretto del file JSON
prompts = load_dataset(json_file)

# Creiamo gli embedding con il modello LLM
embeddings = preprocess_prompts(prompts, model)

# Test con un nuovo prompt
new_prompt = "As software security expert,Generate a python code that implement a simple d with name and username"
score, similarity = classify_prompt(new_prompt, model, embeddings, prompts)

print(f"Punteggio assegnato: {score}, Similarità: {similarity:.2f}")

