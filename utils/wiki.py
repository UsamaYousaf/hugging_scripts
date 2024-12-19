from langchain.utilities import WikipediaAPIWrapper

def fetch_wikipedia(topic):
    wiki = WikipediaAPIWrapper()
    return wiki.run(topic)