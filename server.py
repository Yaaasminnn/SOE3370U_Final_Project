from flask import Flask, request, jsonify, send_from_directory
import os
import bot

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ------------------ FRONTEND ------------------

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)

# ------------------- API -------------------

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("voltages", [])

    if len(data) != 21:
        return jsonify({"error": "21 voltage inputs required"}), 400

    soh = bot.predict_soh(data)

    if soh is None:
        return jsonify({"error": "Prediction failed"}), 500

    return jsonify({
        "soh": round(soh, 3),
        "status": bot.soh_status(soh)
    })


@app.route("/chat", methods=["POST"])
def chat():
    message = request.json.get("message", "")
    reply = bot.askGemini(message)

    return jsonify({"reply": reply})

# ------------------- MAIN -------------------

if __name__ == "__main__":
    app.run(debug=True)
