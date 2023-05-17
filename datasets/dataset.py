import re
import json

import torch.utils.data as data

from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

BASE_PATH = Path(__file__).parent.parent


@dataclass
class Problem:
    passage: str
    question: str
    answer: str

class Dataset(data.Dataset):
    def __init__(self,
                 data_path: Path = Path("data/SVAMP.json"),
                 ):
        with open(Path(BASE_PATH, data_path), 'r') as f:
            self.orig_dataset = json.load(f)

        self.data = []
        for problem_dict in tqdm(self.orig_dataset, desc="Converting Problem to Features "):
            passage = problem_dict['Body']
            question = problem_dict['Question']
            answer = float(problem_dict['Answer'])

            if float.is_integer(answer):
                answer = int(answer)

            problem = Problem(passage=passage, question=question, answer=answer)
            self.data.append(problem)

    def __getitem__(self, index) -> Problem:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

if __name__ == "__main__":
    dataset = Dataset()
    print(dataset[0])