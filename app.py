import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

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


def main():
    load_dotenv()
    st.set_page_config(page_title="docuract", page_icon=":books:")
    
    st.header("Docuract Hub :books:")
    st.text_input("Interact with your documents")

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
        
                


    
if __name__ == '__main__':
    main()