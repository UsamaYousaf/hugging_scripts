import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.schema import BaseLanguageModel
import requests

# Hugging Face API details
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
            "max_length": 512,
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

# Custom LLM wrapper for Hugging Face API
class CustomGemmaLLM(BaseLanguageModel):
    def _call(self, prompt: str, stop: None = None) -> str:
        response = query_huggingface(prompt)
        if response and "generated_text" in response[0]:
            return response[0]["generated_text"]
        return "Error generating response"

# Initialize the custom LLM
llm = CustomGemmaLLM()

# App framework
st.markdown(
    "<h1 style='text-align: center; color: #FF4B4B;'>🦜🔗 Wiki Research with Hugging Face API</h1>",
    unsafe_allow_html=True
)
prompt = st.text_input('Enter a topic for YouTube content:', placeholder="E.g., Artificial Intelligence")

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='Generate a concise and engaging YouTube video title about: {topic}'
)
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
        title = llm._call(title_query)
        st.success("Title generated successfully!")
        title_memory.save_context({'topic': prompt}, {'generated_title': title})

        # Fetch Wikipedia Research
        wiki_research = wiki.run(prompt)
        st.success("Wikipedia research completed!")

        # Generate Script
        script_query = script_template.format(title=title, wikipedia_research=wiki_research)
        script = llm._call(script_query)
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
