import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_dataset(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    prompts = []
    for category, items in data.items():
        for text, score in items.items():
            prompts.append({"text": text, "score": score})
    return prompts

def preprocess_prompts(prompts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([p["text"] for p in prompts])
    return vectorizer, tfidf_matrix

def classify_prompt(new_prompt, vectorizer, tfidf_matrix, prompts):
    new_vector = vectorizer.transform([new_prompt])
    similarities = cosine_similarity(new_vector, tfidf_matrix).flatten()
    best_match_idx = np.argmax(similarities)
    return prompts[best_match_idx]["score"], similarities[best_match_idx]

# Carichiamo il dataset
json_file = "dataset.json"  # Sostituisci con il nome corretto del file JSON
prompts = load_dataset(json_file)
# Creiamo il modello TF-IDF
vectorizer, tfidf_matrix = preprocess_prompts(prompts)

# Test con un nuovo prompt
new_prompt = "Generate secure Python code for the following \n The login page is a simple form with a username and password field.Implement security measures to validate and sanitize the payload to prevent deserialization of untrusted data vulnerabilities."
score, similarity = classify_prompt(new_prompt, vectorizer, tfidf_matrix, prompts)

print(f"Punteggio assegnato: {score}, Similarità: {similarity:.2f}")