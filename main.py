import os
import streamlit as st
import pickle
import time
import requests
from bs4 import BeautifulSoup
import openai

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
if process_url_clicked:
    # Load data from the URLs
    articles = []
    for url in urls:
        main_placeholder.text(f"Loading content from {url}...")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Parse the HTML content
                soup = BeautifulSoup(response.content, 'html.parser')
                # Find all the paragraphs in the article
                paragraphs = soup.find_all('p')
                # Combine paragraphs to form the article content
                article_content = '\n'.join([p.text.strip() for p in paragraphs])
                articles.append(article_content)
            else:
                st.error(f"Error loading article from {url}. Please check the URL and try again.")
        except Exception as e:
            st.error(f"Error loading article from {url}: {e}")

    if articles:
        main_placeholder.text("Articles loaded successfully.")

        # Save the articles to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(articles, f)
            main_placeholder.text("Articles saved successfully.")
    else:
        st.error("No articles found. Please check the URLs and try again.")

query = st.text_input("Question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            articles = pickle.load(f)
            # Initialize the OpenAI model
            openai.api_key = os.getenv("OPENAI_API_KEY")
            model = "text-davinci-002"  # Use the Davinci model for question answering
            completions = []
            for article_content in articles:
                # Generate completion using OpenAI library
                completion = openai.Completion.create(
                    engine=model,
                    prompt=f"Question: {query}\nContext: {article_content}\nAnswer:",
                    max_tokens=100,
                    temperature=0.7
                )
                completions.append(completion.choices[0].text.strip())

            # Display the answers
            st.header("Answers")
            for i, answer in enumerate(completions):
                st.write(f"Article {i+1}: {answer}")
