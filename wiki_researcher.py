import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from huggingface_hub import InferenceClient
import requests

# Constants
API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b"

# Set up headers for Hugging Face API
headers = {
    "authorization": f"Bearer {st.secrets['auth_token']}",
    "content-type": "application/json"
}

# Function to query Hugging Face API
def query_huggingface(prompt):
    """Send a text prompt to the Hugging Face API and return the response."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 512,
            "temperature": 0.1
        }
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

# Sidebar Navigation
st.sidebar.image("https://huggingface.co/front/assets/huggingface_logo.svg", width=150)
st.sidebar.title("Hugging Face Explorer")
mode = st.sidebar.radio("Choose Mode", options=["Generate Content", "Test API"], index=0)

if mode == "Generate Content":
    # Main Content Generation
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🦜🔗 Wiki Research with Hugging Face API</h1>", unsafe_allow_html=True)
    prompt = st.text_input('Enter a topic for YouTube content:', placeholder="E.g., Artificial Intelligence")

    # Prompt Templates
    title_template = PromptTemplate(input_variables=['topic'], template='Generate a YouTube video title about {topic}')
    script_template = PromptTemplate(input_variables=['title', 'wikipedia_research'],
                                      template='Create a YouTube video script based on the title: {title} and the following Wikipedia research: {wikipedia_research}')

    # Memory for Conversations
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

    # Wikipedia Utility
    wiki = WikipediaAPIWrapper()

    if prompt:
        with st.spinner("Generating content..."):
            # Generate Title
            title_query = title_template.format(topic=prompt)
            title_response = query_huggingface(title_query)
            title = title_response[0]["generated_text"] if title_response else "Error generating title"
            st.success("Title generated successfully!")
            title_memory.save_context({'topic': prompt}, {'generated_title': title})

            # Fetch Wikipedia Research
            wiki_research = wiki.run(prompt)
            st.success("Wikipedia research completed!")

            # Generate Script
            script_query = script_template.format(title=title, wikipedia_research=wiki_research)
            script_response = query_huggingface(script_query)
            script = script_response[0]["generated_text"] if script_response else "Error generating script"
            st.success("Script generated successfully!")
            script_memory.save_context({'title': title}, {'generated_script': script})

        # Display Results
        st.subheader("Generated YouTube Title")
        st.markdown(f"<h3 style='color: #2ECC71;'>{title}</h3>", unsafe_allow_html=True)

        st.subheader("Generated YouTube Script")
        st.markdown(f"<div style='background-color: #F5F5F5; color: #333; padding: 10px;'>{script}</div>", unsafe_allow_html=True)

        # Expanders for History
        with st.expander("Title History"):
            st.info(title_memory.buffer)
        with st.expander("Script History"):
            st.info(script_memory.buffer)
        with st.expander("Wikipedia Research"):
            st.info(wiki_research)

elif mode == "Test API":
    # Test API Functionality
    st.markdown("<h1 style='text-align: center; color: #4B9CD3;'>Test Hugging Face InferenceClient API</h1>", unsafe_allow_html=True)

    # Initialize InferenceClient
    api_key = st.secrets["auth_token"]
    client = InferenceClient(api_key=api_key)

    # Test Query
    test_query = st.text_input("Enter a query for the API:", placeholder="E.g., What is the capital of France?")
    if st.button("Submit Query"):
        with st.spinner("Processing API call..."):
            try:
                messages = [{"role": "user", "content": test_query}]
                completion = client.chat.completions.create(
                    model="google/gemma-7b-it",
                    messages=messages,
                    max_tokens=500
                )
                response = completion.choices[0].message["content"]
                st.subheader("API Response")
                st.markdown(f"<div style='background-color: #E8F8F5; color: #333; padding: 10px;'>{response}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
