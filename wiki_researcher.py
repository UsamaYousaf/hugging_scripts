import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.llms.base import LLM
import requests

# Constants
API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b"
HEADERS = {
    "authorization": f"Bearer {st.secrets['auth_token']}",
    "content-type": "application/json"
}

# Function to query Hugging Face API
def query_huggingface(prompt, temperature):
    """Send a text prompt to the Hugging Face API and return the response."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 200,
            "temperature": temperature,
            "top_k": 50,
            "top_p": 0.9
        }
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        return result[0]["generated_text"] if result else "Error generating response."
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return "Error fetching response."

# Custom Hugging Face LLM class
class HuggingFaceLLM(LLM):
    def _call(self, prompt: str, stop: None = None) -> str:
        return query_huggingface(prompt, temperature=0.7)

    @property
    def _identifying_params(self):
        return {"model": "google/gemma-7b"}

    @property
    def _llm_type(self):
        return "custom_huggingface"

# Initialize LLM
llm = HuggingFaceLLM()

# Prompt Templates
title_template = PromptTemplate(
    input_variables=["topic"],
    template="Generate a concise and engaging YouTube video title about: {topic}."
)

script_template = PromptTemplate(
    input_variables=["title", "wikipedia_research"],
    template=(
        "Write a clear, structured YouTube video script based on this title: {title}. "
        "Use the following Wikipedia research: {wikipedia_research}. "
        "Make it informative, engaging, and easy to follow."
    )
)

# Memory
title_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
script_memory = ConversationBufferMemory(input_key="title", memory_key="chat_history")

# Wikipedia utility
wiki = WikipediaAPIWrapper()

# Chains
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, memory=script_memory)

# Streamlit App Layout
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose Mode", options=["Content Generator", "Test API"], index=0)

# Temperature Slider
st.sidebar.markdown("### Temperature Selection")
temperature = st.sidebar.slider(
    "Select Temperature (Creativity Level)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help=(
        "Low temperature (0.1-0.4) results in focused, deterministic outputs. "
        "High temperature (0.8-1.0) produces more creative but less predictable responses."
    )
)

# Main Content Section
if mode == "Content Generator":
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎥 YouTube Content Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555;'>Generate YouTube video titles and scripts effortlessly using Hugging Face's Gemma model.</p>", unsafe_allow_html=True)

    prompt = st.text_input("Enter a topic for YouTube content:", placeholder="E.g., Artificial Intelligence")

    if prompt:
        with st.spinner("Generating YouTube Title..."):
            title = title_chain.run({"topic": prompt})
            st.success("Title generated successfully!")

        with st.spinner("Fetching Wikipedia Research..."):
            wiki_research = wiki.run(prompt)
            st.success("Wikipedia research completed!")

        with st.spinner("Generating YouTube Script..."):
            script = script_chain.run({"title": title, "wikipedia_research": wiki_research})
            st.success("Script generated successfully!")

        # Display Results
        st.subheader("Generated YouTube Title:")
        st.markdown(f"<h3 style='color: #2ECC71;'>{title}</h3>", unsafe_allow_html=True)

        st.subheader("Generated YouTube Script:")
        st.markdown(f"<div style='background-color: #FAFAD2; color: #333; padding: 15px; border-radius: 10px;'>{script}</div>", unsafe_allow_html=True)

        # Expanders for History
        with st.expander("Title History"):
            st.write(title_memory.buffer)

        with st.expander("Script History"):
            st.write(script_memory.buffer)

        with st.expander("Wikipedia Research"):
            st.write(wiki_research)

# Test API Section
elif mode == "Test API":
    st.markdown("<h1 style='text-align: center; color: #4B9CD3;'>🧪 Test Hugging Face API</h1>", unsafe_allow_html=True)
    test_query = st.text_input("Enter a query to test the API:", placeholder="E.g., What is the capital of France?")

    if st.button("Submit Query"):
        with st.spinner("Testing API..."):
            try:
                response = query_huggingface(test_query, temperature)
                st.subheader("API Response:")
                st.markdown(f"<div style='background-color: #E8F8F5; color: #333; padding: 15px; border-radius: 10px;'>{response}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
