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
def query_huggingface(prompt, temperature=0.1):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 512,
            "temperature": temperature
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()

# Add a toggle to enable/disable main code
run_main_code = st.sidebar.radio(
    "Choose Mode",
    options=["Main Code", "Test API"],
    index=0
)

# Sidebar controls
st.sidebar.image("https://huggingface.co/front/assets/huggingface_logo.svg", width=150)
st.sidebar.title("Hugging Face Explorer")
temperature = st.sidebar.slider("Set Temperature", 0.1, 1.0, 0.7)

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
            title_response = query_huggingface(title_query, temperature)
            title = title_response[0]["generated_text"] if title_response else "Error generating title"
            st.success("Title generated!")

        with st.spinner("Fetching Wikipedia Research..."):
            wiki_research = wiki.run(prompt)
            st.success("Research completed!")

        with st.spinner("Generating YouTube Script..."):
            # Generate a script using Hugging Face API
            script_query = script_template.format(title=title, wikipedia_research=wiki_research)
            script_response = query_huggingface(script_query, temperature)
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

        # Summarization Feature
        if st.button("Summarize Script"):
            with st.spinner("Summarizing script..."):
                summary_response = query_huggingface(f"Summarize the following script:\n{script}", temperature)
                summary = summary_response[0]["generated_text"] if summary_response else "Error summarizing script"
                st.write(f"**Summary:** {summary}")

        # Sentiment Analysis
        if st.button("Analyze Sentiment"):
            with st.spinner("Analyzing sentiment..."):
                sentiment_response = query_huggingface(f"Analyze the sentiment of this text:\n{script}", temperature)
                sentiment = sentiment_response[0]["generated_text"] if sentiment_response else "Error analyzing sentiment"
                st.write(f"**Sentiment Analysis Result:** {sentiment}")

        # Keyword Extraction
        if st.button("Extract Keywords"):
            with st.spinner("Extracting keywords..."):
                keywords_response = query_huggingface(f"Extract key topics and keywords:\n{script}", temperature)
                keywords = keywords_response[0]["generated_text"] if keywords_response else "Error extracting keywords"
                st.write(f"**Keywords:** {keywords}")

        # Language Translation
        language = st.selectbox("Translate Script To:", ["French", "Spanish", "German"])
        if st.button(f"Translate Script to {language}"):
            with st.spinner(f"Translating script to {language}..."):
                translation_prompt = f"Translate the following script to {language}:\n{script}"
                translation_response = query_huggingface(translation_prompt, temperature)
                translation = translation_response[0]["generated_text"] if translation_response else "Error translating script"
                st.write(f"**Translated Script ({language}):**\n{translation}")

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
