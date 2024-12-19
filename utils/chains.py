from langchain.chains import LLMChain
from utils.templates import title_template, script_template
from utils.memory import title_memory, script_memory
from langchain.llms.base import LLM
from utils.api import query_huggingface

class HuggingFaceLLM(LLM):
    def _call(self, prompt: str, stop: None = None) -> str:
        return query_huggingface(prompt, temperature=0.7)

    @property
    def _identifying_params(self):
        return {"model": "google/gemma-7b"}

    @property
    def _llm_type(self):
        return "custom_huggingface"

llm = HuggingFaceLLM()

title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, memory=script_memory)