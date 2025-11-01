from google import genai
import os

# API key
key = open("key", "r")
os.environ["GEMINI_API_KEY"] = key.read()

client = genai.Client()
model = "gemini-2.5-flash"

response = client.models.generate_content(
    model=model, contents="Hello World"
)
print(response.text)
