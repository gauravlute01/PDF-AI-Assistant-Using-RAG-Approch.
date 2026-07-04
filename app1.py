from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import os
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import google.generativeai as genai
from dotenv import load_dotenv

# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file"
    )

# =====================================================
# GEMINI SETUP
# =====================================================

genai.configure(
    api_key=GEMINI_API_KEY
)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# =====================================================
# FLASK SETUP
# =====================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# =====================================================
# GLOBAL VARIABLES
# =====================================================

chunks = []
index = None

# =====================================================
# EMBEDDING MODEL
# =====================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =====================================================
# PDF READING
# =====================================================

def load_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =====================================================
# TEXT CHUNKING
# =====================================================

def split_text_words(
        text,
        chunk_size=200,
        overlap=20
):

    words = text.split()

    result = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = words[start:end]

        result.append(
            " ".join(chunk)
        )

        start += (chunk_size - overlap)

    return result


# =====================================================
# CREATE VECTOR DATABASE
# =====================================================

def process_pdf(file_path):

    global chunks
    global index

    text = load_pdf(file_path)

    chunks = split_text_words(text)

    if len(chunks) == 0:
        raise ValueError(
            "No text found in PDF"
        )

    embeddings = embedding_model.encode(
        chunks
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        np.ascontiguousarray(
            embeddings
        )
    )

    print(
        f"Loaded {len(chunks)} chunks"
    )


# =====================================================
# GEMINI RESPONSE
# =====================================================

def ask_llm(prompt):

    try:

        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1024
            }
        )

        return response.text

    except Exception as e:

        print(
            "Gemini Error:",
            str(e)
        )

        return f"Error: {str(e)}"


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =====================================================
# PDF UPLOAD
# =====================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_pdf():

    if "pdf" not in request.files:

        return jsonify({
            "error": "No PDF uploaded"
        })

    file = request.files["pdf"]

    if file.filename == "":

        return jsonify({
            "error": "No file selected"
        })

    filename = secure_filename(
        file.filename
    )

    if not filename.lower().endswith(
            ".pdf"
    ):

        return jsonify({
            "error": "Only PDF files allowed"
        })

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(file_path)

    process_pdf(file_path)

    return jsonify({
        "message":
        "PDF uploaded and indexed successfully"
    })


# =====================================================
# ASK QUESTION
# =====================================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask_question():

    global chunks
    global index

    if index is None:

        return jsonify({
            "answer":
            "Please upload a PDF first."
        })

    data = request.get_json()

    question = data.get(
        "question",
        ""
    )

    if question.strip() == "":

        return jsonify({
            "answer":
            "Please enter a question."
        })

    query_embedding = embedding_model.encode(
        [question]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    k = 4

    distances, indices = index.search(
        np.ascontiguousarray(
            query_embedding
        ),
        k
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx < len(chunks):
            retrieved_chunks.append(
                chunks[idx]
            )

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are an academic tutor.

Answer ONLY using the supplied context.

If answer is unavailable in context,
reply with:

I don't know.

CONTEXT:
{context}

QUESTION:
{question}

Provide:
1. Definition
2. Explanation
3. Important points
4. Conclusion

Answer:
"""

    answer = ask_llm(
        prompt
    )

    return jsonify({
        "answer": answer
    })


# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )