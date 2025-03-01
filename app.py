import streamlit as st
from transformers import pipeline
from modules.search import semantic_search  # Assuming you have a semantic_search function
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load a local LLM (choose a smaller model if needed)
generator = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

def query_llm(context, question):
    prompt = f"""
    Based on the following qualifications and skills, list the key technical skills and qualifications needed for this career:

    Context: {context}

    Answer:
    """
    # Generate a response using the local model
    response = generator(prompt, max_length=200, min_length=50, do_sample=False)

    return response[0]["summary_text"]

def main():
    st.title("LLM Query Application")

    # User input: Query and Question
    query = st.text_input("Enter your query:", "Engineer with Python experience")
    question = st.text_input("Enter your question:", "Summarize the skills")

    # Display the query and question
    st.write("Query:", query)
    st.write("Question:", question)

    if query and question:
        # Perform semantic search
        search_results = semantic_search(query)  # Perform the semantic search

        # Combine the metadata or chunks for context
        retrieved_chunks = " ".join([chunk["text"] for chunk in search_results])  # Adjust if necessary
        st.write("Retrieved Chunks:", retrieved_chunks)

        # Query the LLM with context and question
        response = query_llm(retrieved_chunks, question)

        # Display the LLM response
        st.write("LLM Response:", response)

if __name__ == "__main__":
    main()

