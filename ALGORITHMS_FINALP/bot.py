# SOFE3370 Final Project – Battery Health Chatbot

import os
import joblib
import pandas as pd

# API Key Setup
try:
    if os.path.exists("key"):
        key = open("key", "r").read().strip()
        os.environ["GEMINI_API_KEY"] = key
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model_name = "models/gemini-2.5-flash"
        GEMINI_AVAILABLE = True
        print("Gemini connection ready.\n")
    else:
        GEMINI_AVAILABLE = False
        print("Gemini key file not found. Running in offline mode.\n")
except Exception as e:
    GEMINI_AVAILABLE = False
    print(f"Gemini connection could not be initialized: {e}\n")

# Load the trained regression model
MODEL_PATH = "results/battery_health_regression.pkl"

try:
    regression_model = joblib.load(MODEL_PATH)
    print("Linear Regression model loaded successfully.\n")
except Exception as e:
    regression_model = None
    print(f"Could not load model from {MODEL_PATH}\nError: {e}")

# Helper Functions
def predict_soh(input_data):
    if regression_model is None:
        return None
    try:
        df = pd.DataFrame([input_data])
        pred = regression_model.predict(df)[0]
        return float(pred)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def soh_status(soh, threshold=0.6):
    if soh < threshold:
        return "The battery health is CRITICAL. It may need replacement or maintenance."
    else:
        return "The battery is HEALTHY and operating normally."

def askGemini(prompt):
    if not GEMINI_AVAILABLE:
        return "(Offline mode) Gemini not available."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# CLI Chatbot
if __name__ == "__main__":
    print("Battery Health Chatbot")
    print("Type 'check battery' to predict SOH, or ask general battery questions.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip().lower()

        if user_input == "exit":
            print("Goodbye.")
            break

        elif "check battery" in user_input:
            print("\nEnter voltage readings for 21 cells (U1–U21).")
            readings = []
            try:
                for i in range(1, 22):
                    val = float(input(f"U{i}: "))
                    readings.append(val)
                soh = predict_soh(readings)
                if soh is None:
                    print("Could not compute prediction.")
                else:
                    print(f"\nPredicted SOH: {soh:.3f}")
                    print(soh_status(soh))
            except ValueError:
                print("Invalid input. Please enter only numeric values.\n")

        else:
            reply = askGemini(user_input)
            print(f"Bot: {reply}\n")

