import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
# from langchain.chat_models import ChatOpenAI
import google.generativeai as genai
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import HuggingFaceHub


from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# only use if you have credits in OpenAI
def get_vectorstore_openai(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vectorstore

def get_vectorstore_instruct(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vectorstore

def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI() 
    # llm = genai.GenerativeModel()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xl", task="text-generation", model_kwargs={"temperature": 0.5, "max_length": 512})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    
    mem = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=mem
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

def verify_google_api_key():
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY não encontrada no arquivo .env")
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Erro ao configurar a chave de API do Gemini: {e}")
        st.stop()

def main():
    load_dotenv()
    verify_google_api_key()

    st.set_page_config(page_title="docuract", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Docuract Hub :books:")
    # st.text_input("Interact with your documents")

    user_question = st.text_input("Interaja com seus documentos:")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "Olá, Docuract. Tudo bem com você?"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Olá humano, tudo certo! Como posso ajudá-lo?"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Documents")
        pdf_docs = st.file_uploader("Upload", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # no credits in openai
                # vectorestore = get_vectorstore(text_chunks)

                vectorstore = get_vectorstore_instruct(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore)

    
    # st.session_state.conversation
                


    
if __name__ == '__main__':
    main()