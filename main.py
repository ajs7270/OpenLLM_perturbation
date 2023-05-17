from pathlib import Path

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from datasets import Dataset

# Load SVAMP dataset
svamp = Dataset(Path("data/processed/svamp/dev.json"))

# Load LlamaCpp
llm = LlamaCpp()
