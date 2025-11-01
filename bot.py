
# SOFE3370 Final Project – Linear Regression for Battery SOH

import os
import joblib
import pandas as pd
from google import genai

# API Key Setup
key = open("key", "r").read(),strip()
os.environ["GEMINI_API_KEY"] = key.read()

client = genai.Client()
model = "gemini-2.5-flash"

# Load Trained Model
MODEL_PATH = "result/linear_regression_model.pkl"

try: 
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully \n")
except Exception as e:
    print(f"Could not load model. Make sure it's saved as {MODEL_PATH}. Error: {e}")
    model = None

# Helper Functions
def predict_soh(input_data):
    """Predict battery SOH from input voltage values."""
    if model is None:
        return None
    try:
        dataframe = pd.DataFrame([input_data])
        soh_predict = model.predict(dataframe)[0]
        return float(soh_predict)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def soh_status(soh, threshold=0.6):
    """Classify battery health based on threshold"""
    if soh < threshold:
        return "The battery has is critical ⚠️"
    else:
        return "The battery is healthy ✅"

def askGemini(prompt):
    """Send a query to the Gemini API."""
    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
        return response.text
    except Exception as e:
        return f"Error contacting Gemini: {e}"

# Chatbot Interaction
print("Battery Health Chatbot")
print("Type 'check battery' to test SOH or ask any general battery questions.")
print("Type 'exit' to quit.\n")

while True:
    userInput = input("You: ").strip().lower()

    if userInput == "exit":
        print("Goodbye!")
        break

    elif "check battery" in userInput:
        try:
            print("\nEnter average voltage readings for 21 cells (U1–U21).")
            voltages = []
            for i in range(1, 22):
                val = float(input(f"U{i}: "))
                voltages.append(val)

            soh = predict_soh(voltages)
            if soh is not None:
                print(f"\nPredicted SOH: {soh:.3f}")
                print(soh_status(soh))
            else:
                print("Could not predict SOH.")
        except ValueError:
            print("Please enter valid numbers.")
        print()

    else:
        reply = askGemini(userInput)
        print(f"Bot: {reply}\n")
