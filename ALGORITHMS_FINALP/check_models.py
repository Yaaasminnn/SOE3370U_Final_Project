import google.generativeai as genai
import os

# Cargar la clave
key = open("key", "r").read().strip()
os.environ["GEMINI_API_KEY"] = key
genai.configure(api_key=key)

# Listar modelos disponibles
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
