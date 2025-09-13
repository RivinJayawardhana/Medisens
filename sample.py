import os
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# ⚠️ Page config MUST be first Streamlit call
st.set_page_config(page_title=" Chatbot", page_icon="🤖", layout="centered")

# Load environment variables
load_dotenv()

VECTOR_DB_PATH = "faiss_store_pdfs.pkl"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.warning("⚠️ GOOGLE_API_KEY not found in environment. Please set it in your .env file.")
    GOOGLE_API_KEY = ""  # fallback if needed

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-002",
    google_api_key=GOOGLE_API_KEY
)

# Load Vector DB
vectorstore = None
if os.path.exists(VECTOR_DB_PATH):
    with open(VECTOR_DB_PATH, "rb") as f:
        vectorstore = pickle.load(f)
    st.sidebar.success(f"✅ Loaded vector DB from {VECTOR_DB_PATH}")
else:
    st.sidebar.error(f"⚠️ Vector DB file '{VECTOR_DB_PATH}' not found. Please create it before running the app.")

# Prompt template
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

# Streamlit UI
st.title("🤖 Chatbot")
st.markdown("Ask questions")

# Input box
question = st.text_input("💬 Enter your question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("⚠️ Please enter a question.")
    elif vectorstore is None:
        st.error("❌ Vector DB not loaded. Please make sure faiss_store_pdfs.pkl exists.")
    else:
        with st.spinner("Thinking... 🤔"):
            try:
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(),
                    combine_prompt=prompt,
                    return_source_documents=True,
                )

                result = chain({"question": question}, return_only_outputs=True)
                answer = result.get("answer", "").strip()

                # Post-process
                if not answer or answer.lower() in ["", "no answer generated."]:
                    answer = "I don't know."

                if "please ask company related" in answer.lower():
                    answer = "Please ask company related questions only."

                # Display results
                st.subheader("📝 Answer")
                st.write(answer)

                sources = result.get("sources", "")
                if sources:
                    st.subheader("📚 Sources")
                    st.write(sources)

            except Exception as e:
                st.error(f"❌ Error: {e}")
