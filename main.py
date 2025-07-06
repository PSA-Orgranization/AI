from flask import Flask, request, jsonify 
from App.gpt_service import generate_response_with_rag_context_ROADMAP, generate_response_with_rag_context_General 

app = Flask(__name__)

@app.route("/generate_roadmap", methods=["POST"])
def generate_response_roadmap():
    prompt = request.form.get("prompt")
    session_id = request.form.get("session_id", None, type=int)
    username = request.form.get("username", "default_user")
    email = request.form.get("email", "default@example.com")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    response = generate_response_with_rag_context_ROADMAP(
        user_input=prompt,
        session_id=session_id,
        username=username,
        email=email
    )
    return jsonify({"response": response})

@app.route("/generate", methods=["POST"])
def generate_response_general():
    prompt = request.form.get("prompt")
    session_id = request.form.get("session_id", None, type=int)
    username = request.form.get("username", "default_user")
    email = request.form.get("email", "default@example.com")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    response = generate_response_with_rag_context_General(
        user_input=prompt,
        session_id=session_id,
        username=username,
        email=email
    )
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
