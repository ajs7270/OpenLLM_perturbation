import os
import re
import json
import pandas as pd
import copy
from pathlib import Path
from typing import Tuple, Optional, Union
from dataclasses import dataclass

BASE_PATH = Path(__file__).parent.parent


class Equation:
    def __init__(self, raw: Optional[str] = None, type: str = "prefix"):
        self.equation = None

        if raw is not None:
            if type == "prefix":
                self.equation = self.prefix2equation(raw)
            elif type == "formula":
                self.equation = self.formular2eqation(raw)

    def getList(self) -> Optional[list[list[str]]]:
        return self.equation

    def getOperator(self) -> Optional[list[str]]:
        if self.equation is None:
            return None

        operator = []
        for e in self.equation:
            operator.append(e[0])

        return operator

    def getArgument(self) -> Optional[list[list[str]]]:
        if self.equation is None:
            return None

        argument = []
        for e in self.equation:
            argument.append(e[1:])

        return argument

    def formular2eqation(self, formula: str) -> list[list[str]]:
        equation = []
        formula = formula.strip("|")  # | 이 마지막에 있는 경우도 있고 없는 경우도 있으므로
        formula = formula.replace("(", ",")  # 괄호를 ,로 바꿔서 ,로 split 하자
        formula = formula.replace(")", "")
        formula = formula.split("|")

        for f in formula:
            entities = f.split(",")

            for i in range(len(entities)):

                # const가 const_100.0 형태로 되어 있으면 const_100 으로 바꿔줌
                if re.match("const_\d+", entities[i]):
                    num = float(entities[i].split("_")[1])
                    if float.is_integer(num):
                        entities[i] = "const_" + str(int(num))

            equation.append(entities)

        return equation

    def prefix2equation(self, prefix: str) -> list[list[str]]:
        @dataclass
        class Operator:
            data: str

        @dataclass
        class Operand:
            data : str

        def checkLeaf(cur: int, prefix: list[str]) -> bool:
            if cur + 2 >= len(prefix):
                return False

            if type(prefix[cur]) == Operator and type(prefix[cur + 1]) == Operand and type(prefix[cur+2]) == Operand:
                return True
            else:
                return False

        operator_dict = {
            "+": "add",
            "-": "subtract",
            "*": "multiply",
            "/": "divide",
        }

        equation = []

        # 1. operator를 구분해야 함. operator의 종류 : +, -, *, /
        # 2. operend를 구분해야 함. operend의 종류 : #0, #1, ... n1, n2, ... 100.0(숫자) 등
        prefix = prefix.replace("number", "n")

        prefix_list = []
        for p in prefix.split(" "):
            if re.match(r"[+\-/*]", p):
                prefix_list.append(Operator(p))
            else:
                #숫자가 등장하면 CONST_XXX 형태로 변경
                if re.match(r"\d+\.\d+", p):
                    num = float(p)
                    if float.is_integer(num):
                        p = "const_" + str(int(num))
                    else:
                        p = "const_" + str(num)

                prefix_list.append(Operand(p))

        # (operator, operand, operand) => #number
        result_cnt = 0
        while len(prefix_list) != 1:
            temp = []
            cur = 0
            while cur < len(prefix_list):
                if checkLeaf(cur, prefix_list):
                    equation.append([operator_dict[prefix_list[cur].data], prefix_list[cur+1].data, prefix_list[cur+2].data])
                    temp.append(Operand("#" + str(result_cnt)))
                    result_cnt += 1
                    cur += 2
                else:
                    temp.append(prefix_list[cur])
                cur += 1

            prefix_list = temp

        return equation

def getSameNumberIdx(numbers: list[str]) -> list[list[int]]:
    same_number_idx = []
    for num in set(numbers):
        idx = [i for i, n in enumerate(numbers) if n == num]
        if len(idx) != 1:
            same_number_idx.append(idx)

    return sorted(same_number_idx)

def start_later(problem, iter, prev_iter_end):
  indet = ['a', 'b', 'c', 'x', 'y', 'z', 'w', 'k', 'm', 'n', 'p', 'q', 'i', 'j', 'l', 't', 'u', 'v', 'd', 'e']
  blk = 0 # 0~2 값, - n 꼴이면 start 위치를 두 칸 미뤄서 n부터 보게 한다.
  start,end = iter.start(), iter.end() 
  if problem[start] == "-":                 
    if start != 0 and problem[+ start-2] in indet and start-2 == 0: #  x - 3 이 시작일 때
      if problem[start+1] == " ": 
        blk=2
      else :
        blk=1
    if problem[start-2] in indet and start != 0 and problem[start-3]  ==  " " : # 시작점이 문장 처음이 아니고,  x - 5처럼 앞에 미지수가 있을 경우                      
      if problem[start+1] == " ":       # x -5 와 x - 5 구분              
        blk=2
      else :                        
        blk=1
    if iter.start() == prev_iter_end + 1:            # 3 - 5 처럼 number(k)의 시작과 number(k-1)의 끝이 1칸 차이면 뺄셈으로 친다.     
      if problem[start+1] == " ": #3 -5와 3 - 5 구분 (3-5는 없는듯) 
        blk =2
      else:
        blk = 1
  return blk

class Problem:
    def __init__(self, problem: str, numbers: list[str], equation: Equation):
        self.context = None
        self.question = None
        self.numbers = numbers
        self.same_number_idx = getSameNumberIdx(numbers)
        self.equation = equation.getList()
        self.golden_op = equation.getOperator()
        self.golden_argument = equation.getArgument()

        if "number0" not in problem:
            problem = self.toNumProblem(problem)
        self.context, self.question = problem2CQ(problem)

    def toNumProblem(self, problem: str) -> str:
        original_problem = copy.deepcopy(problem)
        append_idx = 0 # n이 number{i}꼴로 바뀌면서 생기는 string 길이 변화 해소
        prev_iter_end = 0 # 바로 전 iter의 iter.end()
        blk = 0 #
        for i, iter in enumerate(re.finditer(r'(?:[-][ ]?)?\d+(?:\.\d+|(?:,\d\d\d)+)?', problem)):  
            append_idx += blk            
            blk = start_later(original_problem, iter, prev_iter_end)        
            problem = problem[:append_idx + iter.start() + blk] + f"number{i}" + problem[append_idx + iter.end():]
            append_idx += len(f"number{i}") - len(iter.group()) 
            prev_iter_end = iter.end()
        return problem

    def __repr__(self):
        return f"Problem(context={self.context}, question={self.question}, " \
               f"numbers={self.numbers}, equation={self.equation}, " \
               f"golden_op={self.golden_op}, golden_argument={self.golden_argument})"

#json serialization class (for json.dump)
class ProblemEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Problem):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)

# counts the number of consecutive dots and returns the correct number of "|~|" symbols
def replace_dots(match):
    num_dots = len(match.group(0)) // 2
    return " |~|" * num_dots

# divides the problem into context and question
# before dividing sentences using '.' , replace '.' with '|~|' if it is not a sentence ending(e.g., a.m., h.c.f.)
def problem2sentences(problem: str) -> [str]:
    replacements = [
        (" \. \)$", " |~| )"), #ends with . )
        (" \. \]$", " |~| ]"), #ends with . ]
        ("p \. m \.", "p |~| m |~|"),
        ("p \. m ", "p |~| m "),
        ("a \. m \.", "a |~| m |~|"),
        ("a \. m$", "a |~| m"),
        ("a \. m ", "a |~| m "),
        ("l \. c \. m \.", "l |~| c |~| m |~|"),
        ("l \. c \. m ", "l |~| c |~| m "),
        ("h \. c \. f \.", "h |~| c |~| f |~|"),
        ("h \. c \. f ", "h |~| c |~| f "),
        ("c \. i \.", "c |~| i |~|"),
        ("c \. i ", "c |~| i "),
        ("\% p \. a \. ", "% p |~| a |~| "),
        ("\% \. p \. a \.", "% |~| p |~| a |~|"),
        ("\% p \. a", "% p |~| a"),
        ("\% pa \. a ", "% pa |~| a "),
        ("p \. c \. p \. a \.", "p |~| c |~| p |~| a |~|"),
        ("p \. c \. p \. a", "p |~| c |~| p |~| a"),
        ("washington \. d \. c \. ", "washington |~| d |~| c |~| "),
        ("washington d \. c", "washington d |~| c"),
        (" d \. c", " d |~| c"),
        ("sq \. m \. ", "sq |~| m |~| "),
        ("sq \. ", "sq |~| "),
        ("g \. p \.", "g |~| p |~|"),
        ("g \. p ", "g |~| p "),
        (" rs \.", " rs |~| "),
        (" no \. ", " no |~| "),
        ("v \. c \. r \.", "v |~| c |~| r |~|"),
        ("s \. i \. ", "s |~| i |~| "),
        (" s \. i ", " s |~| i "),
        ("km / hr \.", "km / hr |~|"),
        ("t \. v \.", "t |~| v |~|"),
        ("t \. v ", "t |~| v "),
        ("cc \. ", "cc |~| "),
        ("prob . ", "prob |~| "),
        ("mr \. ", "mr |~| "),
        (" ms \. ", " ms |~| "),
        ("Mrs\. ", "Mrs|~| "),
        ("Mr\. ", "Mr|~| "),
        ("\. \?", "|~| ?"), # when sentence ends with ". ?"
        (r"(.) \. (.) \. ", r"\1 |~| \2 |~| "), # all cases like "a . b . "-> "a |~| b |~| " are abbreviations.
        ("\. '$", "|~| '"), # when sentence ends with ". '" -> "|~| '"
        (r"([\=\+\-x/]) \. number", r"\1 |~| number"), # change decimal point ". number00" -> "|~| number00"
    ]

    problem = re.sub("( \.){2,}", replace_dots, problem)  # 2 or more consecutive dots " . . . ." -> " |~| |~| |~| |~|"
    for pattern, replacement in replacements:
        problem = re.sub(pattern, replacement, problem)

    # strip the last dot and split using "."
    sentences = problem.strip().strip(".").split(".")
    return sentences

def concat_after_qmark(sentences):

    new_sentences = []
    for i, sentence in enumerate(sentences):
        #check if there is a question mark        
        if "?" in sentence:
            #concatenate the rest of the sentences
            rest = " ".join(sentences[i:])
            new_sentences.append(rest)
            break
        new_sentences.append(sentence)
    return new_sentences

def problem2CQ(problem : str) -> Tuple[str, str]:

    sentences = concat_after_qmark(problem2sentences(problem))
    context, question = ".".join(sentences[:-1]) + ".", sentences[-1].strip()

    # restore "|~|" -> "."
    context = re.sub("(\|\~\|)", ".", context)
    question = re.sub("(\|\~\|)", ".", question)
    return context, question

# 문제에 등장한 문자열 그대로 추출하는 함수 => 따라서 후처리를 통해 숫자만 추출해야할 필요가 생길 수 있음
def extractNum(problem : str):
    # 문제에 등장하는 숫자의 종류
    # 숫자 종류: 10000 (자연수), 1,000,000 (쉼표가 있는 숫자), 1.5 (소수점이 있는 숫자), - 4 or -4(부호가 있는 숫자)
    numbers = re.findall(r'(?:[-][ ]?)?\d+(?:\.\d+|(?:,\d\d\d)+)?', problem)
    prev_iter_end = 0
    for i, iter in enumerate(re.finditer(r'(?:[-][ ]?)?\d+(?:\.\d+|(?:,\d\d\d)+)?', problem)):
      numbers[i]= numbers[i][start_later(problem, iter, prev_iter_end) :]
      numbers[i] = numbers[i].replace(" ", "")
      prev_iter_end = iter.end()          
    return numbers            


def getConstantList(problem_list: list[Problem]) -> list[str]:
    constant_list = set()
    for p in problem_list:
        for e in p.equation:
            for op in e:
                if re.match(r"const_\S+", op):
                    constant_list.add(op)

    return list(constant_list)

# get the number of operands for each operator
def getOperatorDict(problem_list: list[Problem], operator_dict: dict) -> dict[str:set[int]]:
    for p in problem_list:
        for e in p.equation:
            if e[0] in operator_dict:
                operator_dict[e[0]].add(len(e) - 1)
            else:
                operator_dict[e[0]] = {len(e) - 1}
    return operator_dict

#mathqa preprocessing
def preprocess_mathqa(file_path : str = "data/raw/mathqa", save_path : str = "data/processed/mathqa"):
    train_path = Path(BASE_PATH, file_path, "train.json")
    dev_path = Path(BASE_PATH, file_path, "dev.json")
    test_path = Path(BASE_PATH, file_path, "test.json")

    dataset_path = [train_path, dev_path, test_path]

    constant_list = []
    operator_dict = {}

    max_numbers_size = 0
    max_operators_size = 0

    for path in dataset_path:
        print(f"preprocessing {path}...")
        with open(path, 'r') as f:
            problem_list = []

            data = json.load(f)
            print(f"number of problems: {len(data)}")
            for problem in data:
                problem_text = problem["Problem"]
                numbers = extractNum(problem["Problem"])
                equation = Equation(problem["linear_formula"], type="formula")

                problem = Problem(problem_text, numbers, equation)
                problem_list.append(problem)

                # Get Max number and operator size
                max_numbers_size = max(max_numbers_size, len(problem.numbers))
                max_operators_size = max(max_operators_size, len(problem.equation))

        processed_path = Path(BASE_PATH, save_path, f"{path.stem}.json")

        if not os.path.exists(processed_path.parent):
            os.makedirs(processed_path.parent)
        with open(processed_path, 'w') as f:
            json.dump(problem_list, f, indent=4, cls=ProblemEncoder)

        # Get Constant List
        constant_list += getConstantList(problem_list)

        # Get Operator Dict(key: operator name, value: list[num of operands]])
        operator_dict = getOperatorDict(problem_list, operator_dict)

    config = {}

    # Save Max Number Size
    config["max_numbers_size"] = max_numbers_size
    config["max_operators_size"] = max_operators_size
    # Save Constant List
    constant_list = sorted(list(set(constant_list)))
    config["constant_list"] = constant_list
    # Save Operator Dict
    operator_dict = {k: list(v) for k, v in operator_dict.items()}
    config["operator_dict"] = operator_dict

    config_list_path = Path(BASE_PATH, save_path, "config.json")
    with open(config_list_path, 'w') as f:
        json.dump(config, f, indent=4)


#svamp preprocessing
def preprocess_svamp(file_path : str = "data/raw/mawps-asdiv-a_svamp", save_path : str = "data/processed/svamp"):
    train_path = Path(BASE_PATH, file_path, "train.csv")
    dev_path = Path(BASE_PATH, file_path, "dev.csv")

    dataset_path = [train_path, dev_path]

    constant_list = []
    operator_dict = {}

    max_numbers_size = 0
    max_operators_size = 0

    for path in dataset_path:
        print(f"preprocessing {path}...")
        data = pd.read_csv(path)

        problem_list = []

        print(f"number of problems: {len(data)}")
        for problem in data.itertuples():
            problem_text = problem.Question
            numbers = problem.Numbers.split()
            equation = Equation(problem.Equation, type="prefix")

            problem = Problem(problem_text, numbers, equation)
            problem_list.append(problem)

            # Get Max number and operator size
            max_numbers_size = max(max_numbers_size, len(problem.numbers))
            max_operators_size = max(max_operators_size, len(problem.equation))

        processed_path = Path(BASE_PATH, save_path, f"{path.stem}.json")

        if not os.path.exists(processed_path.parent):
            os.makedirs(processed_path.parent)
        with open(processed_path, 'w') as f:
            json.dump(problem_list, f, indent=4, cls=ProblemEncoder)

        # Get Constant List
        constant_list += getConstantList(problem_list)

        # Get Operator List
        operator_dict = getOperatorDict(problem_list, operator_dict)

    config = {}

    # Save Max Number Size
    config["max_numbers_size"] = max_numbers_size
    config["max_operators_size"] = max_operators_size
    # Save Constant List
    constant_list = sorted(list(set(constant_list)))
    config["constant_list"] = constant_list
    # Save Operator List
    operator_dict = {k: list(v) for k, v in operator_dict.items()}
    config["operator_dict"] = operator_dict

    config_list_path = Path(BASE_PATH, save_path, "config.json")
    with open(config_list_path, 'w') as f:
        json.dump(config, f, indent=4)


#mawps preprocessing

if __name__ == "__main__":
    preprocess_mathqa("data/raw/mathqa", "data/processed/mathqa")
    preprocess_svamp("data/raw/mawps-asdiv-a_svamp", "data/processed/svamp")