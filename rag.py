
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

# Load environment variables (optional)
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="FAISS RAG PDF Chat", layout="wide")
st.title("📄 Conversational RAG with FAISS + PDF Chat")
st.write("Upload PDFs and chat with their content using FAISS + Groq + LangChain.")

# Groq API key input
api_key = st.text_input("🔐 Enter your Groq API Key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")
    session_id = st.text_input("💬 Session ID:", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("📎 Upload PDF(s)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)

        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = splitter.split_documents(documents)

        # Use FAISS instead of Chroma
        vectorstore = FAISS.from_documents(splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Prompts
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and latest user question, rephrase it into a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )

        system_prompt = (
            "You are an assistant for question-answering tasks. Use the following context to answer the question. "
            "If unsure, say you don't know. Keep responses concise.\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        # Chat interface
        st.divider()
        st.subheader("Ask a question about the documents:")

        user_input = st.chat_input("Type your question here...")
        if user_input:
            with st.spinner("Thinking..."):
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                st.chat_message("assistant").write(response["answer"])
