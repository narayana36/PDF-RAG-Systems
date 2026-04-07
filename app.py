import streamlit as st 
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import openai
import pdfplumber
from dotenv import load_dotenv

# 1. Load keys from your .env file
load_dotenv()

# 2. Configure OpenAI library for OpenRouter
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = os.getenv("OPENROUTER_API_KEY")


#Load pre-trained sentence Transformer model
model=SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    try:
        # pdfplumber can open file-like objects directly from Streamlit
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None
        
    return text if text.strip() else None

    

# Step 2: split the text into chunks
def split_into_chunks(text, chunk_size=100):
    #Handle cases where text might be None or empty
    if not text:
        return []
        
    sentences = text.split('.')
    chunks = []
    current_chunk = ''
    
    for sentence in sentences:
        # clean up extra whitespace
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + '. '
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

# Step 3: Embed the chunks using SentenceTransformer
def embed_chunks(chunks):
    # Ensure chunks isn't empty to avoid shape errors
    if not chunks:
        return np.array([]).reshape(0, 0).astype('float32')
    embeddings = model.encode(chunks)
    return np.array(embeddings).astype('float32')

# Step 4: Function to generate answer using LLM
def generate_answer(contexts, query):
    contexts_string = "\n".join([f"- {context}" for context in contexts])
    # initalize openai api key 
    response = openai.ChatCompletion.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Using the context below, provide detailed points to answer the question:\n\nContext:\n{contexts_string}\n\nQuestion:{query}"}
        ]
    )
    return response.choices[0].message['content']

# Streamlit app
st.title("PDF RAG System")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF", type='pdf')

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    if pdf_text:
        st.success("PDF successfully loaded!")
        
        # Split text into chunks
        chunks = split_into_chunks(pdf_text)
        
        if chunks:
            # Embed the chunks and create FAISS index
            embedding_matrix = embed_chunks(chunks)
            
            # Check dimensions safely
            if embedding_matrix.ndim == 1:
                embedding_matrix = embedding_matrix.reshape(1, -1)
            
            d = embedding_matrix.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(embedding_matrix)
            st.success("Text successfully split into chunks and embedded!")
            
            # Query input
            query = st.text_input("Enter your query:")
            
            if query:
                # Embed the query
                query_embedding = model.encode([query]).astype('float32')
                
                # Perform FAISS search
                k = min(5, len(chunks)) # Ensure k isn't larger than available chunks
                distances, indices = index.search(query_embedding, k)
                
                # Gather context
                contexts = [chunks[i] for i in indices[0] if i != -1]
                
                # Generate and display answer
                with st.spinner("Generating answer..."):
                    answer = generate_answer(contexts, query)
                
                st.subheader("Generated Answer:")
                st.write(answer)
                
        else:
            st.error("Could not split text into chunks. Please check the PDF content.")
    else:
        st.error("Failed to extract text from PDF.")
else:
    st.info("Please upload a PDF file to begin.")
