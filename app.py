import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

VECTOR_DB_PATH = "faiss_store_pdfs.pkl"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("⚠️ Warning: GOOGLE_API_KEY not found in environment, using fallback key.")
    GOOGLE_API_KEY = ""  # Replace with your real key or env var

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-002",
    google_api_key=GOOGLE_API_KEY
)

vectorstore = None
if os.path.exists(VECTOR_DB_PATH):
    with open(VECTOR_DB_PATH, "rb") as f:
        vectorstore = pickle.load(f)
    print(f"✅ Loaded vector DB from {VECTOR_DB_PATH}")
else:
    print(f"⚠️ Vector DB file '{VECTOR_DB_PATH}' not found. Please create it before running the API.")

# Corrected prompt template using 'summaries' instead of 'context'
template = """
You are a helpful assistant specialized in providing information about the company KMTEC Ltd.
Answer the question based ONLY on the provided documents.

If the question is NOT related to KMTEC Ltd or its services, reply:
"Please ask company related questions only."

If the answer cannot be found in the documents, reply:
"I don't know."

Question: {question}
========
{summaries}
========
Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question", "summaries"]
)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "🔍 Chatbot API is running."})

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' in JSON body"}), 400

        if vectorstore is None:
            return jsonify({"error": "Vector DB not loaded"}), 500

        question = data["question"]

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            combine_prompt=prompt,
            return_source_documents=True,
        )

        result = chain({"question": question}, return_only_outputs=True)

        answer = result.get("answer", "").strip()

        # Post-process fallback if LLM returns empty or generic no-answer text
        if not answer or answer.lower() in ["", "no answer generated."]:
            answer = "I don't know."

        # Extra safeguard: if answer includes irrelevant notice, fix wording
        if "please ask company related" in answer.lower():
            answer = "Please ask company related questions only."

        return jsonify({
            "answer": answer,
            "sources": result.get("sources", "")
        })

    except Exception as e:
        print(f"❌ Exception in /ask: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
