import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

from PIL import Image
OPENAI_API_KEY= ""

# Custom CSS to change the sidebar and main page background colors
st.markdown(
    """
    <style>
    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #f0f8ff;  /* Light blue for sidebar */
    }

    /* Main page background color */
    .stApp {
        background-color: #f5f5f5;  /* Light grey for main page */
    }

    /* Text color and style in sidebar */
    [data-testid="stSidebar"] .css-1lcbmhc, [data-testid="stSidebar"] .css-145kmo2 {
        color: #333333;  /* Dark grey text color */
    }

    /* Button and file uploader styles */
    button {
        background-color: #4CAF50 !important;  /* Green buttons */
        color: white !important;
    }
    .stFileUploader label {
        color: #4CAF50 !important;  /* Matching file uploader label */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load images
sid_bear_image = Image.open('images/header.jpg')  # Add the path to Sid Bear's image
header_image = Image.open('images/Aigen.jpg')  # Replace with the header image path

# Set header section
st.image(header_image, use_column_width=True)
st.header("Welcome to Chatbot GenAI - Powered by Python")

# Sidebar section
with st.sidebar:
    st.title("Chat Interface")
    
    # Upload PDF file
    file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    # Display Sid Bear image
    st.image(sid_bear_image, caption="Meet Sid Bear!", use_column_width=True)

# If PDF is uploaded, process the file
if file is not None:
    st.subheader("Uploaded PDF Information:")
    
    # PDF processing
    pdf_reader = PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text += page.extract_text()
        #st.write(text)
 
    #break it to chunks

    text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len

    )
    chunks = text_splitter.split_text(text)
    st.write(chunks)
      # generating embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)
    # get user question
    user_question = st.text_input("Type Your question here")

    # do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)

        #define the LLM
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo"
        )

        #output results
        #chain -> take the question, get relevant document, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)


else:
    st.info("Please upload a PDF file to start")

# Footer
st.markdown("### Thank you for using Chatbot GenAI! Enjoy your experience with Sid Bear.")



  