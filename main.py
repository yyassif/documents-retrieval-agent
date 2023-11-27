from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import warnings
import shutil
import torch
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    Load all the necessary models and data once the server starts.
    """
    app.directory = os.getenv("DATA_PATH")
    app.documents = load_docs(app.directory)
    print(f"Loaded {len(app.documents)} documents from {app.directory}")
    app.docs = split_docs(app.documents, chunk_size=int(os.getenv("CHUNK_OVERLAP")), chunk_overlap=int(os.getenv("CHUNK_OVERLAP")))

    app.embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDINGS"),
        model_kwargs={'device': os.getenv("DEVICE")},
        encode_kwargs={'normalize_embeddings': os.getenv("NORMALIZE_EMBEDDINGS") == "True"}
    )
    shutil.rmtree(os.getenv("VECTOR_DB"), ignore_errors=True)
    
    app.collection_name = os.getenv("COLLECTION_NAME")
    app.collection_metadata = { "hnsw:space": os.getenv("VECTOR_SPACE") }
    app.persist_directory = os.getenv("VECTOR_DB")

    app.vector_store = Chroma.from_documents(
        documents=app.docs,
        embedding=app.embeddings,
        collection_name=app.collection_name,
        collection_metadata=app.collection_metadata,
        persist_directory=app.persist_directory
    )
    app.vector_store.persist()
    print(f"Vector store created at {app.persist_directory}")

    app.model_name = os.getenv("LLAMA_PATH")
    app.tokenizer = AutoTokenizer.from_pretrained(app.model_name, token=os.getenv("HF_TOKEN"))
    app.model = AutoModelForCausalLM.from_pretrained(app.model_name,
                                                    device_map='auto',
                                                    torch_dtype=torch.float16,
                                                    token=os.getenv("HF_TOKEN"),
                                                    load_in_4bit=True)
    app.pipeline = pipeline("text-generation",
                            model=app.model,
                            tokenizer=app.tokenizer,
                            torch_dtype=torch.bfloat16,
                            device_map='auto',
                            max_new_tokens=512,
                            min_new_tokens=-1,
                            top_k=30)
    app.llm = HuggingFacePipeline(pipeline=app.pipeline, model_kwargs={ 'temperature':0 })
    app.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    app.chain = ConversationalRetrievalChain.from_llm(llm=app.llm,
                                                      retriever=app.vector_store.as_retriever(search_kwargs={'k': int(os.getenv("NUM_RESULTS"))}),
                                                      verbose=True,
                                                      memory=app.memory)


@app.get("/query/{question}")
async def query_chain(question: str):
    """
    Query the `ConversationalRetrievalChain` llama model based with a given question and returns the answer.
    """
    result = app.chain({ "question": question })
    answer = result["answer"]
    return { "answer": answer }

def load_docs(directory: str):
    """
    Load PDF documents using `PyPDFLoader` from the given directory.
    """
    loader = DirectoryLoader(directory, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size=1000, chunk_overlap=30):
    """
    Split the documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs