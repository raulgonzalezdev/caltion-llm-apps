import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os
import re

st.title("Chat with Webpage 🌐")
st.caption("This app allows you to chat with a webpage using local llama3 and RAG")

# Genera el nombre del directorio de persistencia desde una URL
def generate_persist_directory(url):
    clean_url = re.sub(r'https?://', '', url)
    clean_url = re.sub(r'^[^/]+', '', clean_url)
    clean_url = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', clean_url)
    directory = f'vectorstore_{clean_url}' if clean_url else 'vectorstore_default'
    return directory

def rag_chain(question):
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Proceso para obtener URL y manejar la base de datos vectorial
webpage_url = st.text_input("Enter Webpage URL", type="default")
if webpage_url:
    persist_dir = generate_persist_directory(webpage_url)
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    vectorstore = Chroma(collection_name="my_collection", embedding_function=OllamaEmbeddings(model="llama3"), persist_directory=persist_dir)
    if not os.listdir(persist_dir):  # Si el directorio está vacío, significa que no hay datos guardados
        loader = WebBaseLoader(webpage_url)
        docs = loader.load()
        if docs:
            st.write(f"Loaded {len(docs)} documents from the webpage.")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
            splits = text_splitter.split_documents(docs)
            added_ids = vectorstore.add_documents(splits)
            if added_ids:
                st.success(f"Added {len(added_ids)} documents to the vectorstore successfully!")
            else:
                st.error("Failed to add documents to the vectorstore.")
        else:
            st.error("No documents were loaded from the webpage.")
    else:
        st.success("Vectorstore loaded from existing data successfully!")

    # Si vectorstore está cargado, mostrar input para preguntas
    prompt_container = st.empty()
    prompt = prompt_container.text_input("Ask any question about the webpage")
    if prompt:
        result = rag_chain(prompt)
        st.write(result)
