from langchain.prompts import PromptTemplate

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