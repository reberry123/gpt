import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore

st.set_page_config(
    page_title="FullstackGPT | DocumentGPT",
    page_icon="📃"
)

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file, api_key):
    file_content = file.read()
    os.makedirs(".cache/files", exist_ok=True)
    os.makedirs(".cache/embeddings", exist_ok=True)
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )

    docs = loader.load_and_split(text_splitter=splitter)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system",
        """
        Answer the question using ONLY the following context. If you don't know the answer
        just say you don't know. DON'T make anything up.

        Context: {context}
        """),
    ("human", "{question}")
])
st.title("DocumentGPT")

st.markdown("""
### Welcome!
            
Use this chatbot to ask questions to an AI about your files!
            
Upload your file and Google API Key!
""")

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file",
                            type=["pdf", "txt", "docx"])
    api_key = st.text_input("Your Google API Key")

if file and api_key:
    llm = ChatGoogleGenerativeAI(
        model= "gemini-1.5-flash",
        google_api_key= api_key,
        temperature= 0.1,
    )

    retriever = embed_file(file, api_key)
    chain = {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    } | prompt | llm

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()

    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        
        with st.chat_message("ai"):
            message_box = st.empty()
            response = ""
            for token in chain.stream(message):
                if hasattr(token, 'content'):
                    response += token.content
                    message_box.markdown(response + "▌")
            message_box.markdown(response)
            save_message(response, "ai")
else:
    st.session_state["messages"] = []