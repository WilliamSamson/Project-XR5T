from flask import Flask, request, jsonify
import wikipediaapi

app = Flask(__name__)

# Initialize Wikipedia API with proper user-agent
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='ProjectXR5T/1.0 (contact: your-email@example.com)'
)

@app.route("/get_summary", methods=["POST"])
def get_summary():
    data = request.json
    topic = data.get("topic", "").strip()

    # Validate the topic
    if not topic:
        return jsonify({"error": "Topic is missing or empty."}), 400

    # Fetch summary from Wikipedia
    page = wiki.page(topic)
    if page.exists():
        return jsonify({"summary": page.summary})
    return jsonify({"error": "Page not found for the given topic."}), 404

if __name__ == "__main__":
    # Run the Flask app on port 5001
    app.run(port=5001)
