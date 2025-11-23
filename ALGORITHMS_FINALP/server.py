from flask import Flask, request, jsonify, send_from_directory
import bot

app = Flask(__name__)

# Home route
@app.get("/")
def home():
    return send_from_directory(".", "index.html")

# Serve CSS
@app.get("/style.css")
def css():
    return send_from_directory(".", "style.css")

# Predict SOH
@app.post("/predict")
def predict():
    data = request.json.get("voltages", [])
    if len(data) != 21:
        return jsonify({"error": "21 voltage inputs required"}), 400

    soh = bot.predict_soh(data)
    if soh is None:
        return jsonify({"error": "Prediction failed"}), 500

    return jsonify({
        "soh": soh,
        "status": bot.soh_status(soh)
    })

# Chat endpoint
@app.post("/chat")
def chat():
    message = request.json.get("message", "")
    reply = bot.askGemini(message)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
