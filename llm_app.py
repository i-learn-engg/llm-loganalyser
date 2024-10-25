import hashlib
import os
import openai
import streamlit as st
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Function to retrieve secrets, handling nested keys in st.secrets and fallback to environment variables
def get_secret(key_path):
    if st.secrets:
        keys = key_path.split(".")
        secret = st.secrets
        try:
            for key in keys:
                secret = secret[key]
            return secret
        except KeyError:
            pass  # Fallback to environment if not found in secrets
    return os.getenv(key_path.replace(".", "_").upper())

# Fetch API keys and environment settings
openai_api_key = get_secret("OPENAI_API_KEY")
pinecone_api_key = get_secret("PINECONE_API_KEY")
pinecone_env = get_secret("PINECONE_ENVIRONMENT")

# Initialize OpenAI API key
openai.api_key = openai_api_key

# Initialize Pinecone client with API key and environment
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# Define the index name
index_name = "log-analysis-index"

# Check if the Pinecone index already exists; if not, create it
if index_name not in pinecone.list_indexes():
    # Create a new index with the specified dimension and metric
    pinecone.create_index(
        name=index_name, 
        dimension=1536,  # For OpenAI embeddings
        metric="cosine"
    )

# Connect to the Pinecone index
index = pinecone.Index(index_name)

# Create embeddings using OpenAI
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Set up Pinecone vector store for the embeddings
vectorstore = PineconeStore(index, embeddings.embed_query, "text")

# Dictionary to store processed file hashes
processed_files = {}

# Function to compute hash of the uploaded file
def compute_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Function to load and process .log files
def load_and_process_logs(log_file):
    try:
        # Compute hash of the file
        file_hash = compute_file_hash(log_file)
        
        # Check if file has already been processed
        if file_hash in processed_files:
            st.warning("This file has already been processed!")
            return
        
        with open(log_file, "r") as file:
            log_data = file.read()

        # Split log data into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(log_data)

        # Store the embedded text in Pinecone
        vectorstore.add_texts(texts)

        # Mark the file as processed
        processed_files[file_hash] = log_file

        st.success("Logs uploaded and indexed successfully!")
    except Exception as e:
        st.error(f"Error processing log file: {str(e)}")

# Function to create the QA chain using LangChain
def create_qa_chain():
    try:
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

# Function to handle user questions and retrieve answers
def chat_with_logs(question, qa_chain):
    try:
        response = qa_chain.run(question)
        return response
    except Exception as e:
        st.error(f"Error in retrieving answer: {str(e)}")
        return None

# Function to validate the file extension
def validate_file_extension(file_name):
    if not file_name.endswith('.log'):
        raise ValueError("Invalid file format. Only '.log' files are allowed.")

# Main function to run Streamlit app
def main():
    st.title("Log Analysis Q&A Chat")
   
    # Initialize session state to store chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Section to upload log file (specifically accepting .log files)
    uploaded_file = st.file_uploader("Upload .log file", type=["log"])

    if uploaded_file is not None:
        # Validate the file extension
        validate_file_extension(uploaded_file.name)
        
        # Display file size in MB
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.write(f"Uploaded file size: {file_size_mb:.2f} MB")

        with open("log.log", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")

        # Load and process the .log file
        load_and_process_logs("log.log")

    # Initialize the QA chain
    qa_chain = create_qa_chain()

    # Section to ask questions
    st.subheader("Ask Questions Based on the Logs")
    user_input = st.text_input("Ask a question about the logs:")

    # Submit button to handle questions
    if st.button("Submit"):
        if user_input and qa_chain:
            response = chat_with_logs(user_input, qa_chain)
            if response:
                # Store the question and response in chat history
                st.session_state.chat_history.append((user_input, response))
        
        elif not user_input:
            st.warning("Please enter a question before submitting.")

    # Display the chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            st.write(f"**Q{i+1}:** {question}")
            st.write(f"**A{i+1}:** {answer}")

if __name__ == "__main__":
    main()