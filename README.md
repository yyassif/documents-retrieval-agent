# Chat With Multiple PDF Documents using Conversational RAG on CPU with LLAMA2, Langchain ChromaDB

This is a Conversational Retrieval Augmented Generation (RAG) Knowledge Base Chat built on top of LLAMA2 (Embeddings & Model), Langchain and ChromaDB and orchestrated by FastAPI framework to provide and Endpoint for easy communication.

---

## Quickstart

### Conversational RAG runs offline on local CPU

1. Setup a virtual environement & Install the requirements:

```{python}
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt`
```

2. Copy your PDF files to the `documents` folder.

3. Run the FastAPI server, to process and ingest your data on start with the LLM RAG and return the answer:

```{python}
python main.py "What is the invoice number value?"
```
