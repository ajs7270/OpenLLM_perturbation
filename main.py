import json
from pathlib import Path

from langchain.llms import LlamaCpp
from datasets import Dataset
from core import CoT, PoT, PhP

MODEL_SIZE = "7B"

# Load SVAMP dataset
svamp = Dataset(Path("data/SVAMP.json"))

# Load LlamaCpp
llm = LlamaCpp(
    model_path=f"./models/llama/{MODEL_SIZE}/ggml-model-f4.bin",
    n_ctx=2048,
)

CoT_correct = 0
PoT_correct = 0
PHP_correct = 0

for i, problem in enumerate(svamp):
    cot_output = CoT(llm=llm, problem=problem)
    pot_output = PoT(llm=llm, problem=problem)
    php_output = PhP(llm=llm, problem=problem)

    if cot_output == problem.answer:
        CoT_correct += 1
    if pot_output == problem.answer:
        PoT_correct += 1
    if php_output == problem.answer:
        PHP_correct += 1

    print(f"current corrects PoT: {PoT_correct}, CoT: {CoT_correct}, PHP: {PHP_correct}")

# Save result json
with open(Path(f"result_{MODEL_SIZE}.json"), 'w') as f:
    json.dump({
        "CoT": CoT_correct,
        "PoT": PoT_correct,
        "PhP": PHP_correct,
    }, f)
