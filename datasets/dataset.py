import re
import json

import torch.utils.data as data

from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

BASE_PATH = Path(__file__).parent.parent


@dataclass
class Problem:
    question: str
    answer: str

class Dataset(data.Dataset):
    def __init__(self,
                 data_path: Path = Path("data/processed/svamp/dev.json"),
                 ):
        with open(Path(BASE_PATH, data_path), 'r') as f:
            self.orig_dataset = json.load(f)

        self.data = []
        for problem_dict in tqdm(self.orig_dataset, desc="Converting Problem to Features "):
            question = problem_dict['question']
            numbers = problem_dict['numbers']
            answer = float(problem_dict['answer'])

            if float.is_integer(answer):
                answer = int(answer)

            # Replace numbers with a placeholder
            # question = 'If Bryan has number0 books in each of his number1 bookshelves , how many books does he have in total ?'
            # numbers = ['56.0', '9.0']
            for i, number in enumerate(numbers):
                number = float(number)
                if float.is_integer(number):
                    number = int(number)
                question = question.replace(f'number{i}', str(number))

            assert not re.match(f'number\d', question) , f"question: {question}"
            problem = Problem(question=question, answer=answer)
            self.data.append(problem)

    def __getitem__(self, index) -> Problem:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    dataset = Dataset()
    print(dataset[0])