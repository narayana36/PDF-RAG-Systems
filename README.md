# PDF Retrieval Augmentation Generator RAG Systems 
Project Overview This project is designed extract and understand information from PDFs data using Natural Language processing techinques. This system let's user upload an PDF and then ask questions about the content. It return relevant text of answer from PDF and generate a detailed, multi-point answer using a large lanauage model (LLM), specifically gpt-3.5 or gpt - 4.
Key components of this project: PDF text Extraction: Extract data from PDF Text chunking: text spliting into smaller, meaningful chunks Text Embedding: chunks of text convert into numberical embedding from SentenceTransformer model. FAISS Index: A search index used to efficient most relevant chunks based on the user query. LLM Integration: Gpt-3.5 or GPt -4 to generate answer detrailed and multi-point response based on the most relevant chunks of the document.
Detailed expaination of each step:
PDF Extraction this is the first step of  
