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
run_main_code = st.sidebar.radio(
    "Choose Mode",
    options=["Main Code", "Test API"],
    index=0
)

# Add a logo and title to the sidebar
st.sidebar.image("https://huggingface.co/front/assets/huggingface_logo.svg", width=150)
st.sidebar.title("Hugging Face Explorer")

if run_main_code == "Main Code":
    # App framework
    st.markdown(
        """
        <h1 style='text-align: center; color: #FF4B4B;'>🦜🔗 Wiki Research with Hugging Face API</h1>
        """,
        unsafe_allow_html=True
    )
    prompt = st.text_input('Enter your topic here:', placeholder="E.g., Artificial Intelligence")

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
        with st.spinner("Generating YouTube Title..."):
            # Generate a title using Hugging Face API
            title_query = title_template.format(topic=prompt)
            title_response = query_huggingface(title_query)
            title = title_response[0]["generated_text"] if title_response else "Error generating title"
            st.success("Title generated!")

        with st.spinner("Fetching Wikipedia Research..."):
            wiki_research = wiki.run(prompt)
            st.success("Research completed!")

        with st.spinner("Generating YouTube Script..."):
            # Generate a script using Hugging Face API
            script_query = script_template.format(title=title, wikipedia_research=wiki_research)
            script_response = query_huggingface(script_query)
            script = script_response[0]["generated_text"] if script_response else "Error generating script"
            st.success("Script generated!")

        # Update title memory
        title_memory.save_context({'topic': prompt}, {'generated_title': title})
        # Update script memory
        script_memory.save_context({'title': title}, {'generated_script': script})

        # Display results
        st.subheader("Generated YouTube Title:")
        st.markdown(f"<h3 style='color: #2ECC71;'>{title}</h3>", unsafe_allow_html=True)

        st.subheader("Generated YouTube Script:")
        st.markdown(f"<div style='background-color: #F5F5F5; color: #333; padding: 10px;'>{script}</div>", unsafe_allow_html=True)

        # Expanders for history
        with st.expander('Title History'):
            st.info(title_memory.buffer)
        with st.expander('Script History'):
            st.info(script_memory.buffer)
        with st.expander('Wikipedia Research'):
            st.info(wiki_research)
else:
    # Test InferenceClient API
    st.markdown(
        """
        <h1 style='text-align: center; color: #4B9CD3;'>Testing Hugging Face InferenceClient API</h1>
        """,
        unsafe_allow_html=True
    )
    
    # Initialize InferenceClient with API key from secrets
    api_key = st.secrets["auth_token"]
    client = InferenceClient(api_key=api_key)

    # Define a test query
    messages = [
        {
            "role": "user",
            "content": st.text_input(
                "Enter a query for the API:",
                placeholder="E.g., What is the capital of France?"
            )
        }
    ]

    if st.button("Submit Query"):
        try:
            # Make an API call
            with st.spinner("Processing API Call..."):
                completion = client.chat.completions.create(
                    model="google/gemma-7b-it",
                    messages=messages,
                    max_tokens=500
                )

            # Display the result
            st.subheader("API Response:")
            st.markdown(
                f"<div style='background-color: #E8F8F5; padding: 10px;'>{completion.choices[0].message['content']}</div>",
                unsafe_allow_html=True
            )
        except Exception as e:
            # Handle errors gracefully
            st.error(f"An error occurred: {e}")
