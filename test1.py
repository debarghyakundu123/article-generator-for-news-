import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)
if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    try:
        data = loader.load()
        print("Data loaded successfully:", data)  # Print loaded data
        print("Data length:", len(data))  # Print length of loaded data
    except Exception as e:
        print("Error loading data:", e)
        st.error("Error loading data. Please check the URLs and try again.")
        st.stop()

    # Check URLs
    print("Provided URLs:")
    for url in urls:
        print(url)

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    try:
        docs = text_splitter.split_documents(data)
        print("Documents split successfully.")
        print("Number of documents:", len(docs))  # Print number of split documents
    except Exception as e:
        print("Error splitting documents:", e)
        st.error("Error splitting documents.")
        st.stop()


    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    print("Embeddings created successfully.")
    if docs:
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)
        print("Embedding vector built successfully.")
        
        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)
            print("FAISS index saved successfully.")
    else:
        st.error("No documents found. Please check the URLs and try again.")


query = main_placeholder.text_input("Question: ")
import openai

# Set up OpenAI API credentials
openai.api_key = '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'

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            # Initialize the OpenAI model
            model = "gpt-3.5-turbo"  # Replace with your desired model
            # Generate completion using OpenAI library
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You: " + query},
                    {"role": "user", "content": ""},
                ],
                max_tokens=100,
                temperature=0.7,
                stop=["\n"]
            )



            # Display the answer
            st.header("Answer")
            st.write(completion.choices[0]['message']['content'].strip())

