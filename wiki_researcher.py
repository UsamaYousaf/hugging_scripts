from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
import requests
import streamlit as st

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
            "temperature": 0.1,
            "top_k": 50,
            "top_p": 0.9
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    return result[0]["generated_text"] if result else "Error generating response."

# Custom LLM class for Hugging Face API
class HuggingFaceLLM(LLM):
    def _call(self, prompt: str, stop: None = None) -> str:
        return query_huggingface(prompt)
    
    @property
    def _identifying_params(self):
        return {"model": "google/gemma-7b"}
    
    @property
    def _llm_type(self):
        return "custom_huggingface"

# Initialize custom LLM
llm = HuggingFaceLLM()

# Prompt templates
title_template = PromptTemplate(
    input_variables=["topic"],
    template="Write a concise and engaging YouTube video title about {topic}."
)

script_template = PromptTemplate(
    input_variables=["title", "wikipedia_research"],
    template=(
        "Write a detailed, structured YouTube video script based on this title: {title}. "
        "Incorporate the following Wikipedia research: {wikipedia_research}. Make it clear and engaging."
    )
)

# Memory
title_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
script_memory = ConversationBufferMemory(input_key="title", memory_key="chat_history")

# Wikipedia utility
wiki = WikipediaAPIWrapper()

# Streamlit UI
st.title("🦜🔗 YouTube Video Generator with Hugging Face")

prompt = st.text_input("Enter a topic for YouTube content:", placeholder="E.g., Artificial Intelligence")

if prompt:
    # Title generation
    title_chain = LLMChain(llm=llm, prompt=title_template, memory=title_memory, verbose=True)
    title = title_chain.run(prompt)

    # Wikipedia research
    wiki_research = wiki.run(prompt)

    # Script generation
    script_chain = LLMChain(llm=llm, prompt=script_template, memory=script_memory, verbose=True)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    # Display results
    st.subheader("Generated YouTube Title")
    st.write(title)

    st.subheader("Generated YouTube Script")
    st.write(script)

    # Expanders for history
    with st.expander("Title History"):
        st.write(title_memory.buffer)
    with st.expander("Script History"):
        st.write(script_memory.buffer)
    with st.expander("Wikipedia Research"):
        st.write(wiki_research)
