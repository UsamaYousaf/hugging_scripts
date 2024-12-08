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
            "max_length": 200,
            "temperature": 0.2,
            "repetition_penalty": 1.2,
            "top_k": 50,
            "top_p": 0.9
        }
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

# Preprocess Wikipedia research
def preprocess_wikipedia_research(research):
    """Condense Wikipedia research to the first 3-4 sentences."""
    sentences = research.split(". ")
    return ". ".join(sentences[:4]) + "."

# Validate model outputs
def validate_output(output):
    """Ensure the generated output is clear and free of irrelevant content."""
    if "double comparison" in output or "unclear comparison" in output:
        return "Error: The model generated irrelevant content. Please try again with a refined prompt."
    return output

# Sidebar Navigation
st.sidebar.image("https://huggingface.co/front/assets/huggingface_logo.svg", width=150)
st.sidebar.title("Hugging Face Explorer")
mode = st.sidebar.radio("Choose Mode", options=["Generate Content", "Test API"], index=0)

if mode == "Generate Content":
    # Main Content Generation
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🦜🔗 Wiki Research with Hugging Face API</h1>", unsafe_allow_html=True)
    prompt = st.text_input('Enter a topic for YouTube content:', placeholder="E.g., Artificial Intelligence")

    # Prompt Templates
    title_template = PromptTemplate(input_variables=['topic'], template='Generate a concise and engaging YouTube video title about: {topic}')
    script_template = PromptTemplate(
        input_variables=['title', 'wikipedia_research'],
        template=(
            "Write a clear, structured YouTube video script based on this title: {title}. "
            "Use the following Wikipedia research: {wikipedia_research}. "
            "Make it informative, engaging, and free of repetition or unrelated details."
        )
    )

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
            title = validate_output(title_response[0]["generated_text"] if title_response else "Error generating title")
            st.success("Title generated successfully!")
            title_memory.save_context({'topic': prompt}, {'generated_title': title})

            # Fetch Wikipedia Research
            wiki_research = preprocess_wikipedia_research(wiki.run(prompt))
            st.success("Wikipedia research completed!")

            # Generate Script
            script_query = script_template.format(title=title, wikipedia_research=wiki_research)
            script_response = query_huggingface(script_query)
            script = validate_output(script_response[0]["generated_text"] if script_response else "Error generating script")
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
