# SOFE3370 Final Project – Battery Health Chatbot

import os
import joblib
import pandas as pd
import google.generativeai as genai

# API Key Setup
try:
    if os.path.exists("key"):
        key = open("key", "r").read().strip()
        os.environ["GEMINI_API_KEY"] = key

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


# As required by the project document
def soh_status(soh, threshold=0.6):
    if soh < threshold:
        return "The battery has a problem."
    else:
        return "The battery is healthy."


def askGemini(prompt):
    if not GEMINI_AVAILABLE:
        return "(Offline mode) Gemini not available."

    try:
        model = genai.GenerativeModel(model_name)

        structured_prompt = f"""
        Answer in a short and structured format:
        - Max 4 bullet points.
        - Short sentences.
        - No long paragraphs.
        - No markdown symbols like **, ### or lists.
        - Keep it concise and clear.

        User question:
        {prompt}
        """

        response = model.generate_content(structured_prompt)
        text = response.text.strip()

        # Only show voltage info if the question is about voltage, range, or inputs
        keywords = ["voltage", "volt", "range", "input", "value", "values", "battery range"]

        if any(word in prompt.lower() for word in keywords):

            voltage_info = """

Voltage reference (project values):
Normal cell range: 3.3 V – 3.6 V
Healthy examples: 3.45 – 3.60 V
Low / problem range: 3.0 – 3.3 V
Invalid (outside dataset): < 2.9 V or > 3.8 V
"""
            return text + voltage_info

        return text

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
            print(f"\nBot:\n{reply}\n")
