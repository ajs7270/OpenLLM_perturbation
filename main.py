from pathlib import Path

from langchain.llms import LlamaCpp
from datasets import Dataset
from core import CoT, PoT

# Load SVAMP dataset
svamp = Dataset(Path("data/SVAMP.json"))

# Load LlamaCpp
llm = LlamaCpp(
    model_path="./models/llama/7B/ggml-model-f4.bin",
    n_ctx=2048,
)

for problem in svamp:
    print(problem)
    output = CoT(llm=llm, problem=problem)
    print(output)

