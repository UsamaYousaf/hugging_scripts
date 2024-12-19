import os
import streamlit as st
from utils.api import query_huggingface
from utils.memory import title_memory, script_memory
from utils.templates import title_template, script_template
from utils.chains import title_chain, script_chain
from utils.wiki import fetch_wikipedia
from components.sidebar import configure_sidebar
from components.display import display_results

# Configure Sidebar
mode, temperature = configure_sidebar()

if mode == "Content Generator":
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎥 YouTube Content Generator</h1>", unsafe_allow_html=True)
    prompt = st.text_input("Enter a topic for YouTube content:", placeholder="E.g., Artificial Intelligence")

    if prompt:
        with st.spinner("Generating YouTube Title..."):
            title = title_chain.run({"topic": prompt})
            st.success("Title generated successfully!")

        with st.spinner("Fetching Wikipedia Research..."):
            wiki_research = fetch_wikipedia(prompt)
            st.success("Wikipedia research completed!")

        with st.spinner("Generating YouTube Script..."):
            script = script_chain.run({"title": title, "wikipedia_research": wiki_research})
            st.success("Script generated successfully!")

        display_results(title, script, wiki_research)

elif mode == "Test API":
    st.markdown("<h1 style='text-align: center; color: #4B9CD3;'>🧪 Test Hugging Face API</h1>", unsafe_allow_html=True)
    test_query = st.text_input("Enter a query to test the API:", placeholder="E.g., What is the capital of France?")

    if st.button("Submit Query"):
        with st.spinner("Testing API..."):
            response = query_huggingface(test_query, temperature)
            st.subheader("API Response:")
            st.markdown(f"<div style='background-color: #E8F8F5; color: #333; padding: 15px; border-radius: 10px;'>{response}</div>", unsafe_allow_html=True)