import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from huggingface_hub import InferenceClient
import requests

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b"
headers = {
    "authorization": f"Bearer {st.secrets['auth_token']}",
    "content-type": "application/json"
}

# Function to query Hugging Face API
def query_huggingface(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 512,
            "temperature": 0.1
        }
    }
    print(headers)
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()

# Add a toggle to enable/disable main code
run_main_code = st.checkbox("Run Main Code", value=False)

if run_main_code:
    # App framework
    st.title(f"🦜🔗 Wiki Research with Hugging Face API")
    prompt = st.text_input('Enter your prompt here')

    # Prompt templates
    title_template = PromptTemplate(
        input_variables=['topic'],
        template='Generate a YouTube video title about {topic}'
    )

    script_template = PromptTemplate(
        input_variables=['title', 'wikipedia_research'],
        template='Create a YouTube video script based on the title: {title} and the following Wikipedia research: {wikipedia_research}'
    )

    # Memory
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

    # Wikipedia utility
    wiki = WikipediaAPIWrapper()

    # Generate and display content if there's a prompt
    if prompt:
        # Generate a title using Hugging Face API
        title_query = title_template.format(topic=prompt)
        title_response = query_huggingface(title_query)
        title = title_response[0]["generated_text"] if title_response else "Error generating title"

        # Update title memory
        title_memory.save_context({'topic': prompt}, {'generated_title': title})

        # Wikipedia research
        wiki_research = wiki.run(prompt)

        # Generate a script using Hugging Face API
        script_query = script_template.format(title=title, wikipedia_research=wiki_research)
        script_response = query_huggingface(script_query)
        script = script_response[0]["generated_text"] if script_response else "Error generating script"

        # Update script memory
        script_memory.save_context({'title': title}, {'generated_script': script})

        # Display results
        st.write(f"**Title:** {title}")
        st.write(f"**Script:**\n{script}")

        # Expanders for history
        with st.expander('Title History'):
            st.info(title_memory.buffer)
        with st.expander('Script History'):
            st.info(script_memory.buffer)
        with st.expander('Wikipedia Research'):
            st.info(wiki_research)
else:
    # Test InferenceClient API
    st.title("Testing Hugging Face InferenceClient API")
    
    # Initialize InferenceClient with API key from secrets
    api_key = st.secrets["auth_token"]
    client = InferenceClient(api_key=api_key)

    # Define a test query
    messages = [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]

    try:
        # Make an API call
        completion = client.chat.completions.create(
            model="google/gemma-7b-it",
            messages=messages,
            max_tokens=500
        )

        # Display the result
        st.write("**Response:**")
        st.write(completion.choices[0].message["content"])
    except Exception as e:
        # Handle errors gracefully
        st.error(f"An error occurred: {e}")
