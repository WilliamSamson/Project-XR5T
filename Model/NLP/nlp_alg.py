from flask import Flask, request, jsonify
import spacy
from sentence_transformers import SentenceTransformer
import requests

app = Flask(__name__)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
WIKIPEDIA_HANDLER_URL = "http://localhost:5001/get_summary"  # Wikipedia handler endpoint

@app.route("/process_query", methods=["POST"])
def process_query():
    data = request.json
    user_query = data.get("query", "")

    # Step 1: Extract entities
    doc = nlp(user_query)
    entities = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "EVENT", "PRODUCT"}]

    if not entities:
        return jsonify({"response": "Sorry, I couldn't identify the topic of your question. Could you clarify?"})

    # Step 2: If multiple entities, ask for clarification
    if len(entities) > 1:
        return jsonify({"response": f"Your query seems ambiguous. Did you mean one of these: {', '.join(entities)}?"})

    # Step 3: Forward entity to the Wikipedia handler
    topic = entities[0]
    response = requests.post(WIKIPEDIA_HANDLER_URL, json={"topic": topic})
    if response.status_code == 200:
        wiki_summary = response.json().get("summary", "Sorry, I couldn't fetch information about that topic.")
        return jsonify({"response": f"Here's what I found about '{topic}':\n{wiki_summary}"})
    return jsonify({"response": "Error communicating with Wikipedia handler."})

if __name__ == "__main__":
    app.run(port=5000)
