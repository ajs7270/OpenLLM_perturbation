import re
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

from torchnlp.encoders import LabelEncoder
from transformers import AutoTokenizer, AutoConfig

from dataclasses import dataclass

from typing import List

BASE_PATH = Path(__file__).parent.parent


@dataclass
class Feature:
    # B : Batch size
    # S : Max length of tokenized problem text (Source)
    # T : Max length of tokenized equation (Target)
    # A : Max le ngth of operands (Arity)
    input_ids: torch.Tensor         # [B, S]
    attention_mask: torch.Tensor    # [B, S]
    question_mask: torch.Tensor     # [B, S]
    number_mask: torch.Tensor       # [B, S]
    operator_label: torch.tensor    # [B, T]
    operand_label: torch.tensor     # [B, T, A], type : constant, number in problem, previous step result
    equation_label: torch.Tensor    # [B, T, A+1]
    equation_mask: torch.Tensor     # [B, T, A+1] 0: padding, 1: operator, 2: operand

@dataclass
class Problem:
    context: str
    question: str
    numbers: list[str]
    same_number_idx: list[list[int]]
    equation: list[list[str]]
    golden_op: list[str]
    golden_argument: list[list[str]]


class Dataset(data.Dataset):
    def __init__(self,
                 data_path: Path = Path("data/processed/mathqa/train.json"),
                 config_path: Path = Path("data/processed/mathqa/config.json"),
                 pretrained_model_name: str = "roberta-base",
                 ):
        with open(Path(BASE_PATH, data_path), 'r') as f:
            self.orig_dataset = json.load(f)
        with open(Path(BASE_PATH, config_path), 'r') as f:
            self.config = json.load(f)
        # added 'PAD' token to operator and constant list
        operator_list = ['PAD'] + list(self.config['operator_dict'].keys())
        constant_list = ['PAD'] + list(self.config['constant_list'])
        self.pretrained_model_name = pretrained_model_name
        self.operator_encoder = LabelEncoder(operator_list,
                                             reserved_labels=['unknown'], unknown_index=0)
        self.operand_encoder = LabelEncoder(self._get_available_operand_list(constant_list),
                                            reserved_labels=['unknown'], unknown_index=0)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.plm_config = AutoConfig.from_pretrained(self.pretrained_model_name)
        self.constant_ids = [self.tokenizer(constant, return_tensors="pt").input_ids[0][1:-1]
                             for constant in constant_list]  # exclude first and last <s>, </s> tokens
        self.operator_ids = [self.tokenizer(operator, return_tensors="pt").input_ids[0][1:-1]
                             for operator in operator_list]  # exclude first and last <s>, </s> tokens

        # 문제에 등장하는 모든 숫자를 <quant>로 치환한 후 tokenize하기 때문에 tokenize가 끝난 후 숫자의 위치를 찾기 위해 사용
        # this idea is from Deductive MWP
        # AnReu/math_pretrained_bert, does not include spaces in tokenization only exclude first and last token
        if self.pretrained_model_name in ["AnReu/math_pretrained_bert"]:
            self.quant_list_ids = self.tokenizer(" <quant> ", return_tensors="pt").input_ids[0][1:-1]
        # witiko/mathberta, use 'Ġ' as space, exclude 2 of beggining and end.
        elif self.pretrained_model_name in ["wikito/mathberta"]:
            self.quant_list_ids = self.tokenizer(" <quant> ", return_tensors="pt").input_ids[0][2:-2]
                # roberta use 'Ġ' as space, but only concatenated in front of other tokens, or at the end independantly
            # exclude 1 beggining(<s>) and 2 end(Ġ, <\s>).
        elif self.pretrained_model_name in ["microsoft/deberta-v3-large", "microsoft/deberta-v2-xlarge", "microsoft/deberta-v3-base"]:
            self.quant_list_ids = self.tokenizer(" <quant> ", return_tensors="pt").input_ids[0][2:-1]
        else:
            self.quant_list_ids = self.tokenizer(" <quant> ", return_tensors="pt").input_ids[0][1:-2]

        self.features = []
        for problem_dict in tqdm(self.orig_dataset, desc="Converting Problem to Features "):
            problem = Problem(**problem_dict)
            feature = self._convert_to_feature(problem)
            self.features.append(feature)

    def _convert_to_feature(self, problem: Problem) -> Feature:
        # ~~~~ number0 ~~~~ number2 ~~~~~~~ 문장을
        # ~~~~ <quant> ~~~~ <quant> ~~~~~~~ 로 치환
        problem_question = self._num2quent(problem.question)
        problem_context = self._num2quent(problem.context)

        # tokenize
        tokenized_problem = self.tokenizer(problem_context, problem_question, return_tensors="pt").input_ids
        tokenized_context = self.tokenizer(problem_context, return_tensors="pt").input_ids
        # 첫번째는 SOS, 마지막은 EOS 토큰이므로 제외시킴
        number_tensors = [self.tokenizer(number, return_tensors="pt").input_ids[:, 1:-1] for number in problem.numbers]

        tokenized_problem, question_mask, number_mask, num_count = self._translate2number(tokenized_problem,
                                                                                          tokenized_context,
                                                                                          number_tensors,
                                                                                          self.quant_list_ids,
                                                                                          self.pretrained_model_name)
        assert num_count == len(number_tensors), "number의 개수가 맞지 않음 {} != {}\n" \
                                                 "number list : {}\n" \
                                                 "tokenized problem: {}\n" \
                                                 "tokenized result : {}\n" \
                                                 "problem context : {}\n" \
                                                 "problem question : {}" \
            .format(num_count, len(number_tensors), number_tensors, tokenized_problem,
                    self.tokenizer.convert_ids_to_tokens(tokenized_problem[0]), problem_context, problem_question)

        attention_mask = torch.ones_like(tokenized_problem)

        assert tokenized_problem.shape[1] == question_mask.shape[1] == number_mask.shape[1] == \
               attention_mask.shape[1], \
            "tokenized_problem.shape[1]: {}\n" \
            "question_mask.shape[1]: {}\n" \
            "number_mask.shape[1]: {}\n" \
            "attention_mask.shape[1]: {}\n" \
            "모든 shape 같아야 합니다." \
                .format(tokenized_problem.shape[1], question_mask.shape[1], number_mask.shape[1],
                        attention_mask.shape[1])

        # equation label
        operator_label, operand_label, equation_label, equation_mask = self._convert_equation_label(problem.equation)

        assert len(operator_label.shape) == 2 and len(operand_label.shape) == 3, \
            "dimension of operator_label must be 2, operand_label must be 3"

        assert operator_label.shape[:2] == operand_label.shape[:2], \
            "operator_label.shape[0]: {}, operator_label.shape[1]: {}\n" \
            "operand_label.shape[0]: {}, operand_label.shape[1]: {}\n" \
            "operator_label, operand_label must have same 1st dim" \
                .format(operator_label.shape[0], operator_label.shape[1], operand_label.shape[0],
                        operand_label.shape[1])

        return Feature(input_ids=tokenized_problem,
                       attention_mask=attention_mask,
                       question_mask=question_mask,
                       number_mask=number_mask,
                       operator_label=operator_label,
                       operand_label=operand_label,
                       equation_label=equation_label,
                       equation_mask=equation_mask)

    @staticmethod
    def _translate2number(tokenized_problem, tokenized_context, number_tensors, quant_list_ids=None, model_name=None):
        # AnReu/math_pretrained_bert only uses [SEP] once between sentences
        if model_name == "AnReu/math_pretrained_bert":
            tokenized_problem = torch.cat(
                [tokenized_problem[:, :tokenized_context.shape[1] - 1],
                 tokenized_problem[:, tokenized_context.shape[1]:]],
                dim=1)
        else:
            tokenized_problem = torch.cat(
                [tokenized_problem[:, :tokenized_context.shape[1] - 1],
                 tokenized_problem[:, tokenized_context.shape[1] + 1:]],
                dim=1)

        # tokenized_problem[0, :tokenized_context.shape[1]].tolist()
        question_mask = torch.zeros_like(tokenized_problem)
        question_mask[:, tokenized_context.shape[1] - 1:] = 1
        number_mask = torch.zeros_like(tokenized_problem)

        # <quant>를 number_tensors로 치환
        num_count = 0
        cur = 0
        while cur < len(tokenized_problem[0]) - len(quant_list_ids) + 1:
            if torch.equal(tokenized_problem[0][cur:cur + len(quant_list_ids)], quant_list_ids):
                # number_mask에 숫자의 등장순서에 따라 1,2,3으로 마스킹
                number_mask = torch.cat([number_mask[:, :cur],
                                         torch.full(number_tensors[num_count].shape, num_count + 1),
                                         number_mask[:, cur + len(quant_list_ids):]], dim=1)
                # question_mask 사이즈 조정
                question_mask = torch.cat([question_mask[:, :cur],
                                           torch.full(number_tensors[num_count].shape, question_mask[0, cur]),
                                           question_mask[:, cur + len(quant_list_ids):]], dim=1)
                # number_tensors로 치환
                tokenized_problem = torch.cat([tokenized_problem[:, :cur],
                                               number_tensors[num_count],
                                               tokenized_problem[:, cur + len(quant_list_ids):]], dim=1)

                cur += len(number_tensors[num_count][0]) - len(quant_list_ids)
                num_count += 1
            cur += 1

        return tokenized_problem, question_mask, number_mask, num_count

    @staticmethod
    def _num2quent(problem_text: str):
        # 문제에 등장하는 모든 number변수를 " <quant> "로 치환
        append_idx = 0
        for find_number in re.finditer("number\d+", problem_text):
            if find_number.start() == 0:
                problem_text = " " + problem_text
                append_idx += 1

            if find_number.end() + append_idx >= len(problem_text):
                problem_text = problem_text + " "

            l_space = "" if problem_text[find_number.start() + append_idx - 1] == " " else " "
            r_space = "" if problem_text[find_number.end() + append_idx] == " " else " "
            problem_text = problem_text[:find_number.start() + append_idx] + l_space + "<quant>" + r_space + \
                problem_text[find_number.end() + append_idx:]

            append_idx = append_idx + len("<quant>") - len(find_number.group())

        return problem_text

    def _convert_equation_label(self, equation: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # maximum arity of equation: max(operands + 1(operator), for all equation[i])
        max_arity = 0
        for i in range(len(equation)):
            max_arity = max(max_arity, len(equation[i]) - 1)
        # [T]
        operator_label = torch.full([len(equation)], self.operator_encoder.encode("PAD").item())
        operand_label = torch.full([len(equation), max_arity], self.operand_encoder.encode("PAD").item())
        equation_label = torch.full([len(equation), max_arity + 1], self.operator_encoder.encode("PAD").item())
        equation_mask = torch.zeros([len(equation), max_arity + 1])
        for i in range(len(equation)):
            for j in range(len(equation[i])):
                if j == 0:
                    equation_mask[i][j] = 1
                    equation_label[i][j] = self.operator_encoder.encode(equation[i][j])
                    operator_label[i] = self.operator_encoder.encode(equation[i][j])
                else:
                    equation_mask[i][j] = 2
                    equation_label[i][j] = self.operand_encoder.encode(equation[i][j])
                    operand_label[i][j - 1] = self.operand_encoder.encode(equation[i][j])

        # [T] -> [1, T], [T, A] -> [1, T, A], [T, A+1] -> [1, T, A+1], [T, A+1] -> [1, T, A+1]
        return operator_label.unsqueeze(dim=0), operand_label.unsqueeze(dim=0), equation_label.unsqueeze(dim=0), equation_mask.unsqueeze(dim=0)

    def _get_available_operand_list(self, constant_list: List[str]) -> List[str]:
        ret = []

        max_numbers_size = self.config["max_numbers_size"]
        max_step_size = self.config["max_operators_size"]

        ret += constant_list
        ret += [f"n{i}" for i in range(max_numbers_size)]
        ret += [f"#{i}" for i in range(max_step_size - 1)]

        assert len(ret) == max_numbers_size + (max_step_size - 1) + len(constant_list), \
            "length of ret: {}, max_numbers_size: {}, max_operators_size: {}, len(constant_list): {}\n" \
            "length of available operand list must be equal to the sum of each component" \
            .format(len(ret), max_numbers_size, max_step_size, len(constant_list))

        return ret

    def __getitem__(self, index) -> Feature:
        return self.features[index]

    def __len__(self) -> int:
        return len(self.features)

    def collate_function(self, batch: list[Feature]) -> Feature:
        bsz = len(batch)
        max_input_ids = self.plm_config.max_position_embeddings
        max_operators_size = self.config["max_operators_size"]
        max_operator_operands_size = max(map(max, self.config['operator_dict'].values()))

        input_ids = torch.full((bsz, max_input_ids), self.tokenizer.pad_token_id)
        attention_mask = torch.full((bsz, max_input_ids), 0)
        question_mask = torch.full((bsz, max_input_ids), 0)
        number_mask = torch.full((bsz, max_input_ids), 0)
        operator_label = torch.full((bsz, max_operators_size), self.operator_encoder.encode("PAD").item())
        operand_label = torch.full((bsz, max_operators_size, max_operator_operands_size),
                                   self.operand_encoder.encode("PAD").item())
        equation_label = torch.full((bsz, max_operators_size, max_operator_operands_size+1), self.operator_encoder.encode("PAD").item())
        equation_mask = torch.zeros((bsz, max_operators_size, max_operator_operands_size+1))
        for i in range(bsz):
            input_ids[i, :batch[i].input_ids.shape[1]] = batch[i].input_ids[0]
            attention_mask[i, :batch[i].attention_mask.shape[1]] = batch[i].attention_mask[0]
            question_mask[i, :batch[i].question_mask.shape[1]] = batch[i].question_mask[0]
            number_mask[i, :batch[i].number_mask.shape[1]] = batch[i].number_mask[0]
            operator_label[i, :batch[i].operator_label.shape[1]] = batch[i].operator_label[0]
            operand_label[i, :batch[i].operand_label.shape[1], :batch[i].operand_label.shape[2]] = \
                batch[i].operand_label[0]
            equation_label[i, :batch[i].equation_label.shape[1], :batch[i].equation_label.shape[2]] = \
                batch[i].equation_label[0]
            equation_mask[i, :batch[i].equation_mask.shape[1], :batch[i].equation_mask.shape[2]] = \
                batch[i].equation_mask[0]
        return Feature(input_ids, attention_mask, question_mask, number_mask, operator_label, operand_label,
                       equation_label, equation_mask)

    @property
    def pad_id(self):
        assert self.operand_encoder.encode("PAD").item() == self.operator_encoder.encode("PAD").item()

        return self.operand_encoder.encode("PAD").item()