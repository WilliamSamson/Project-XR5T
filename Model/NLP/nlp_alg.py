from flask import Flask, request, jsonify
import spacy
from sentence_transformers import SentenceTransformer
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
WIKIPEDIA_HANDLER_URL = "http://localhost:5001/get_summary"  # Wikipedia handler endpoint

# Helper function to extract entities and relevant context
def extract_entities_and_context(query):
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "EVENT", "PRODUCT"}]
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return entities, keywords

# Helper function to query Wikipedia handler
def query_wikipedia_handler(topic):
    try:
        response = requests.post(WIKIPEDIA_HANDLER_URL, json={"topic": topic}, timeout=5)
        if response.status_code == 200:
            return response.json().get("summary", "Sorry, I couldn't fetch information about that topic.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying Wikipedia handler: {e}")
    return "Error communicating with Wikipedia handler."

# Helper function to handle ambiguous queries
def resolve_ambiguity(entities, query):
    embeddings = model.encode(entities + [query])
    query_embedding = embeddings[-1]
    similarities = [
        (entity, (query_embedding @ embeddings[i]) / (query_embedding.norm() * embeddings[i].norm()))
        for i, entity in enumerate(entities)
    ]
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_similarities[0][0] if sorted_similarities else None

@app.route("/process_query", methods=["POST"])
def process_query():
    data = request.json
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"response": "Query cannot be empty."})

    # Step 1: Extract entities and context
    entities, keywords = extract_entities_and_context(user_query)

    if not entities:
        return jsonify({"response": "Sorry, I couldn't identify a clear topic in your question. Could you provide more details?"})

    # Step 2: Handle multiple entities with ambiguity resolution
    if len(entities) > 1:
        resolved_entity = resolve_ambiguity(entities, user_query)
        if not resolved_entity:
            return jsonify({"response": f"Your query seems ambiguous. Did you mean one of these: {', '.join(entities)}?"})
        entities = [resolved_entity]  # Use the resolved entity

    # Step 3: Forward the most relevant entity to the Wikipedia handler
    topic = entities[0]
    wiki_summary = query_wikipedia_handler(topic)

    return jsonify({"response": f"Here's what I found about '{topic}':\n{wiki_summary}"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
