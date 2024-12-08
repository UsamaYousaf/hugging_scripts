# Import dependencies
import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
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
            "max_length": 200,
            "temperature": 0.9,
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

# App framework
st.title("🦜🔗 YouTube GPT Creator with Gemma")
prompt = st.text_input("Plug in your prompt here:", placeholder="E.g., Artificial Intelligence")

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template="Write me a YouTube video title about {topic}."
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template=(
        "Write me a YouTube video script based on this title: TITLE: {title} "
        "while leveraging this Wikipedia research: {wikipedia_research}. "
        "Make it engaging, clear, and informative."
    )
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Chains
title_chain = LLMChain(
    llm=lambda topic: validate_output(query_huggingface(title_template.format(topic=topic))[0]["generated_text"]),
    prompt=title_template,
    verbose=True,
    output_key='title',
    memory=title_memory
)
script_chain = LLMChain(
    llm=lambda inputs: validate_output(
        query_huggingface(script_template.format(title=inputs['title'], wikipedia_research=inputs['wikipedia_research']))[0]["generated_text"]
    ),
    prompt=script_template,
    verbose=True,
    output_key='script',
    memory=script_memory
)

# Wikipedia utility
wiki = WikipediaAPIWrapper()

# Generate content
if prompt:
    with st.spinner("Generating YouTube content..."):
        # Generate title
        title = title_chain.run(prompt)

        # Wikipedia research
        wiki_research = preprocess_wikipedia_research(wiki.run(prompt))

        # Generate script
        script = script_chain.run(title=title, wikipedia_research=wiki_research)

        # Display results
        st.subheader("Generated YouTube Title")
        st.write(f"**{title}**")

        st.subheader("Generated YouTube Script")
        st.markdown(f"<div style='background-color: #F5F5F5; color: #333; padding: 10px;'>{script}</div>", unsafe_allow_html=True)

        # Display history in expanders
        with st.expander("Title History"):
            st.info(title_memory.buffer)

        with st.expander("Script History"):
            st.info(script_memory.buffer)

        with st.expander("Wikipedia Research"):
            st.info(wiki_research)
