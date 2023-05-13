from unittest import TestCase

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import re
from datasets.dataset import Dataset, Problem, Feature
import torch.nn.functional as F
import torch


class TestDataset(TestCase):
    datasets = []
    # roberta-large, roberta-base, npm, npm-single, AnReu/math_pretrained_roberta have same tokenizer
    # models = ["roberta-large", "roberta-base", "facebook/npm", "facebook/npm-single", "witiko/mathberta",
    # "AnReu/math_pretrained_bert", "AnReu/math_pretrained_roberta"]
    models = ["roberta-base"]

    for model in models:
        print("preparing datasets for model: ", model)
        datasets.append(Dataset(data_path="data/processed/mathqa/test.json",
                                config_path="data/processed/mathqa/config.json",
                                pretrained_model_name=model))


    def _prepare_test_translate2number(self, dataset, problem) -> (list, list, list, list, list):
        model_name = dataset.pretrained_model_name
        tokenizer = dataset.tokenizer
        feature = dataset._convert_to_feature(problem)
        tokenized_problem = feature.input_ids
        question_mask = feature.question_mask
        number_mask = feature.number_mask

        decoded_num_list = []
        decoded_sentence = ""
        tokenized_question = ""

        # get tokens of cur_num, decode them and append to num_list
        for cur_num in range(max(number_mask[0])):
            mask = number_mask[0] == (cur_num + 1)
            tokenized_num = torch.masked_select(tokenized_problem[0], mask)
            # witiko/mathberta adds a space at the beggining of every sentence, so we need to remove it
            if model_name in ["witiko/mathberta"]:
                decoded_num_list.append(tokenizer.decode(tokenized_num).strip())
            else:
                decoded_num_list.append(tokenizer.decode(tokenized_num))

        # strip whitespaces from problem and decoded tokenized_problem, also removed "<s>", "</s>"
        for k in range(len(tokenized_problem[0])):
            decoded_sentence += tokenizer.decode(tokenized_problem[0][k])
        # AnReu/math_pretrained_bert has ## in front of tokens that are not the first token of a word
        # and it uses [CLS] and [SEP] as special tokens
        if model_name in ["AnReu/math_pretrained_bert"]:
            decoded_sentence = decoded_sentence.replace("[CLS]", "").replace("[SEP]", "").replace("##", "").replace(" ", "")
        else:
            decoded_sentence = decoded_sentence.replace("<s>", "").replace("</s>", "").replace(" ", "")
        problem_sentence = problem.context + problem.question
        for k, num in enumerate(decoded_num_list):
            problem_sentence = problem_sentence.replace(f"number{k}", num).replace(" ", "")

        # strip whitespaces from question and decoded tokenized question, also removed "<s>", "</s>"
        for k in range(len(question_mask[0])):
            if question_mask[0][k] != 0:
                tokenized_question += tokenizer.decode(tokenized_problem[0][k])
        # AnReu/math_pretrained_bert has ## in front of tokens that are not the first token of a word
        # and it uses [CLS] and [SEP] as special tokens
        if model_name in ["AnReu/math_pretrained_bert"]:
            tokenized_question = tokenized_question.replace("[CLS]", "").replace("[SEP]", "").replace("##", "").replace(" ", "")
        else:
            tokenized_question = tokenized_question.replace(" ", "").replace("</s>", "")
        question_sentence = problem.question
        for k, num in enumerate(decoded_num_list):
            question_sentence = question_sentence.replace(f"number{k}", num).replace(" ", "")

        return decoded_num_list, decoded_sentence, problem_sentence, tokenized_question, question_sentence

    def test_translate2number(self):
        print(f"check models: ", end="")
        for i, dataset in enumerate(self.datasets):
            print("\033[32m" + self.models[i] + "\033[0m", end=", ")
            # print(dataset.tokenizer(" <quant> ", return_tensors="pt").input_ids[0])
            # print(dataset.quant_list_ids)
            # print(dataset.tokenizer.convert_ids_to_tokens(dataset.tokenizer(" <quant> ", return_tensors="pt").input_ids[0]))
            # print(dataset.tokenizer.convert_ids_to_tokens(dataset.quant_list_ids))
            problem_dict1 = {
                'context': 'sophia finished number0 / number1 of a book . she calculated that she finished number2 more pages than she has yet to read .',
                'question': 'how long is her book ?',
                'numbers': ['23838', '32131313', '90'],
                'equation': [['divide', 'n0', 'n1'], ['subtract', 'const_1', '#0'], ['divide', 'n2', '#1']],
                'golden_op': ['divide', 'subtract', 'divide'],
                'golden_argument': [['n0', 'n1'], ['const_1', '#0'], ['n2', '#1']],
                'same_number_idx': [[0, 1], [2]],
            }
            problem1 = Problem(**problem_dict1)
            decoded_num_list, decoded_problem, original_problem, decoded_question, original_question = \
                self._prepare_test_translate2number(dataset, problem1)
            # check equality of decoded tokenized numbers and problem.numbers
            self.assertEqual(decoded_num_list, problem1.numbers)
            # check equality of decoded sentence and original problem sentence
            self.assertEqual(decoded_problem, original_problem)
            # check whether question_mask is correct after stripping whitespaces
            self.assertEqual(decoded_question, original_question)

            problem_dict2 = {
                "context": "a mixture of number0 liters of milk and water contains number1 % water .",
                "question": "how much water should be added to this so that water may be number2 % in the new mixture ?",
                "numbers": ["40", "10", "20"],
                "same_number_idx": [],
                "equation": [["divide", "n2", "const_100"], ["divide", "n1", "const_100"]],
                "golden_op": ["divide", "divide", "subtract", "divide", "multiply", "multiply", "subtract", "divide"],
                "golden_argument": [["n2", "const_100"], ["n1", "const_100"], ["const_100", "n2"], ["#2", "const_100"]]
            }
            problem2 = Problem(**problem_dict2)

            decoded_num_list, decoded_problem, original_problem, decoded_question, original_question = \
                self._prepare_test_translate2number(dataset, problem2)
            # check equality of decoded tokenized numbers and problem.numbers
            self.assertEqual(decoded_num_list, problem2.numbers)
            # check whether question_mask is correct after stripping whitespaces
            self.assertEqual(decoded_problem, original_problem)
            # check equality of decoded sentence and original problem sentence
            self.assertEqual(decoded_question, original_question)

            problem_dict3 = {
                "context": "in a certain lottery , the probability that a number between number0 and number1 , "
                           "inclusive , is drawn is number2 / number3 .",
                "question": "if the probability that a number number4 or larger is drawn is number5 / number6 , what is the probability that a number less than or equal to number7 is drawn ?",
                "numbers": ["14", "20", "1434", "6", "1414414", "2", "3", "20"],
                "same_number_idx": [[0, 1], [2, 3], [4, 5, 6], [7]],
                "equation": [["divide", "n2", "n3"], ["divide", "n5", "n6"], ["subtract", "const_1", "#0"],
                             ["divide", "n4", "n1"], ["subtract", "const_1", "#1"], ["divide", "n0", "n1"],
                             ["subtract", "const_1", "#2"], ["divide", "n7", "n1"]],
                "golden_op": ["divide", "divide", "subtract", "divide", "subtract", "divide", "subtract", "divide"],
                "golden_argument": [["n2", "n3"], ["n5", "n6"], ["const_1", "#0"], ["n4", "n1"], ["const_1", "#1"],
                                    ["n0", "n1"], ["const_1", "#2"], ["n7", "n1"]]
            }
            problem3 = Problem(**problem_dict3)

            decoded_num_list, decoded_problem, original_problem, decoded_question, original_question = \
                self._prepare_test_translate2number(dataset, problem3)
            # check equality of decoded tokenized numbers and problem.numbers
            self.assertEqual(decoded_num_list, problem3.numbers)
            # check equality of decoded sentence and original problem sentence
            self.assertEqual(decoded_problem, original_problem)
            # check whether question_mask is correct after stripping whitespaces
            self.assertEqual(decoded_question, original_question)

            problem_dict4 = {
                "context": "i . x + number0 y + number1 z = number2 ii . x + y - z = number3 iii .",
                "question": "number4 x + number5 y - z = number6 what is the value of y in the system above ?",
                "numbers": ["234", "31444", "772", "055", "0", "2", "1"],
                "same_number_idx": [[0, 1], [2, 3], [4, 5, 6]],
                "equation": [["divide", "n0", "n1"], ["divide", "n2", "n3"], ["divide", "n4", "n5"],
                             ["divide", "n6", "n7"], ["subtract", "const_1", "#0"], ["subtract", "const_1", "#1"],
                             ["subtract", "const_1", "#2"], ["subtract", "const_1", "#3"], ["divide", "n0", "n1"],
                             ["divide", "n2", "n3"], ["divide", "n4", "n5"], ["divide", "n6", "n7"]],
                "golden_op": ["divide", "divide", "divide", "divide", "subtract", "subtract", "subtract", "subtract",
                              "divide", "divide", "divide", "divide"],
                "golden_argument": [["n0", "n1"], ["n2", "n3"], ["n4", "n5"], ["n6", "n7"], ["const_1", "#0"],
                                    ["const_1", "#1"], ["const_1", "#2"], ["const_1", "#3"], ["n0", "n1"], ["n2", "n3"],
                                    ["n4", "n5"], ["n6", "n7"]]
            }

            problem4 = Problem(**problem_dict4)

            decoded_num_list, decoded_problem, original_problem, decoded_question, original_question = \
                self._prepare_test_translate2number(dataset, problem4)
            # check equality of decoded tokenized numbers and problem.numbers
            self.assertEqual(decoded_num_list, problem4.numbers)
            # check equality of decoded sentence and original problem sentence
            self.assertEqual(decoded_problem, original_problem)
            # check whether question_mask is correct after stripping whitespaces
            self.assertEqual(decoded_question, original_question)

            problem_dict5 = {
                "context": ".",
                "question": "if grapes are number0 % water and raisins are number1 % water , then how many kilograms did a quantity of raisins , which currently weighs number2 kilograms , weigh when all the raisins were grapes ? ( assume that the only difference between their raisin - weight and their grape - weight is water that evaporated during their transformation . )",
                "numbers": ["90", "233333330", "10"],
                "same_number_idx": [[0, 1], [2]],
                "equation": [["divide", "n0", "n1"], ["divide", "n2", "const_100"]],
                "golden_op": ["divide", "divide"],
                "golden_argument": [["n0", "n1"], ["n2", "const_100"]]
            }
            problem5 = Problem(**problem_dict5)
            decoded_num_list, decoded_problem, original_problem, decoded_question, original_question = \
                self._prepare_test_translate2number(dataset, problem5)
            # check equality of decoded tokenized numbers and problem.numbers
            self.assertEqual(decoded_num_list, problem5.numbers)
            # check equality of decoded sentence and original problem sentence
            self.assertEqual(decoded_problem, original_problem)
            # check whether question_mask is correct after stripping whitespaces
            self.assertEqual(decoded_question, original_question)

            problem_dict6 = {
                "context": "the volume of a sphere with radius r is ( number0 / number1 ) * pi * r ^ number2 and the surface area is number3 * pi * r ^ number4 .",
                "question": "if a sperical balloon has a volume of number5 pi cubic centimeters , what is hte surface area of the balloon in square centimeters ?",
                "numbers": ["4", "3", "3", "4", "3", "12345"],
                "same_number_idx": [[0, 1], [2, 3, 4], [5]],
                "equation": [["divide", "n0", "n1"], ["divide", "n2", "const_100"], ["divide", "n3", "const_100"],
                             ["divide", "n4", "const_100"], ["divide", "n5", "const_100"]],
                "golden_op": ["divide", "divide", "divide", "divide", "divide"],
                "golden_argument": [["n0", "n1"], ["n2", "const_100"], ["n3", "const_100"], ["n4", "const_100"],
                                    ["n5", "const_100"]]
            }
            problem6 = Problem(**problem_dict6)
            decoded_num_list, decoded_problem, original_problem, decoded_question, original_question = \
                self._prepare_test_translate2number(dataset, problem6)
            # check equality of decoded tokenized numbers and problem.numbers
            self.assertEqual(decoded_num_list, problem6.numbers)
            # check equality of decoded sentence and original problem sentence
            self.assertEqual(decoded_problem, original_problem)
            # check whether question_mask is correct after stripping whitespaces
            self.assertEqual(decoded_question, original_question)

    def test_collate_function_all_dataset(self):
        print(f"check models: ", end="")
        for dataset in self.datasets:
            print("\033[32m" + dataset.pretrained_model_name + "\033[0m", end=", ")
            dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=dataset.collate_function, drop_last=True)
            for i, batch in enumerate(dataloader):
                # check if batch size is correct
                self.assertEqual(batch.input_ids.shape[0], 5)
                self.assertEqual(batch.attention_mask.shape[0], 5)
                self.assertEqual(batch.question_mask.shape[0], 5)
                self.assertEqual(batch.number_mask.shape[0], 5)
                self.assertEqual(batch.operator_label.shape[0], 5)
                self.assertEqual(batch.operand_label.shape[0], 5)
                self.assertEqual(batch.equation_label.shape[0], 5)
                self.assertEqual(batch.equation_mask.shape[0], 5)
                # check if sequence length is correct
                self.assertEqual(batch.input_ids.shape[1], batch.attention_mask.shape[1])
                self.assertEqual(batch.input_ids.shape[1], batch.question_mask.shape[1])
                self.assertEqual(batch.input_ids.shape[1], batch.number_mask.shape[1])

    def test_collate_function(self):
        print(f"check models: ", end="")
        for dataset in self.datasets:
            print("\033[32m" + dataset.pretrained_model_name + "\033[0m", end=", ")
            max_input_ids = dataset.plm_config.max_position_embeddings
            max_operators_size = dataset.config["max_operators_size"]
            max_operator_operands_size = max(map(max, dataset.config['operator_dict'].values()))

            input_ids = torch.Tensor([1, 2, 3, 4, 5, 6]).view(1, 6)
            attention_mask = torch.Tensor([1, 1, 1, 1, 1, 1]).view(1, 6)
            question_mask = torch.Tensor([0, 0, 0, 1, 1, 1]).view(1, 6)
            number_mask = torch.Tensor([0, 0, 1, 0, 0, 0]).view(1, 6)
            equation_label = torch.Tensor([7, 8, 9, 10, 11, 12]).view(1, 2, 3)
            operator_label = equation_label[:, :, 0]
            operand_label = equation_label[:, :, 1:]
            equation_mask = torch.Tensor([1, 2, 2, 1, 2, 2]).view(1, 2, 3)
            feature1 = Feature(input_ids, attention_mask, question_mask, number_mask, operator_label, operand_label, equation_label, equation_mask)
            feature2 = Feature(-input_ids, -attention_mask, -question_mask, -number_mask, -operator_label, -operand_label, -equation_label, -equation_mask)
            collated = dataset.collate_function([feature1, feature2])

            input_ids_ans = torch.LongTensor([[1, 2, 3, 4, 5, 6], [-1, -2, -3, -4, -5, -6]]).view(2, 6)
            attention_mask_ans = torch.LongTensor([[1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, -1]]).view(2, 6)
            question_mask_ans = torch.LongTensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, -1, -1, -1]]).view(2, 6)
            number_mask_ans = torch.LongTensor([[0, 0, 1, 0, 0, 0], [0, 0, -1, 0, 0, 0]]).view(2, 6)
            equation_label_ans = torch.LongTensor([[7, 8, 9, 10, 11, 12], [-7, -8, -9, -10, -11, -12]]).view(2, 2, 3)
            equation_mask_ans = torch.LongTensor([[1, 2, 2, 1, 2, 2], [-1, -2, -2, -1, -2, -2]]).view(2, 2, 3)

            input_ids_ans = F.pad(input_ids_ans, (0, max_input_ids-6), value = dataset.tokenizer.pad_token_id)
            attention_mask_ans = F.pad(attention_mask_ans, (0, max_input_ids-6), value = 0)
            question_mask_ans = F.pad(question_mask_ans, (0, max_input_ids-6), value = 0)
            number_mask_ans = F.pad(number_mask_ans, (0, max_input_ids-6), value = 0)
            equation_label_ans = F.pad(equation_label_ans, (0, max_operator_operands_size - 2, 0, max_operators_size - 2), value = dataset.tokenizer.pad_token_id)
            operator_label_ans = equation_label_ans[:, :, 0]
            operand_label_ans = equation_label_ans[:, :, 1:]
            equation_mask_ans = F.pad(equation_mask_ans, (0, max_operator_operands_size - 2, 0, max_operators_size - 2), value = 0)

            self.assertEqual(collated.input_ids.tolist(), input_ids_ans.tolist())
            self.assertEqual(collated.attention_mask.tolist(), attention_mask_ans.tolist())
            self.assertEqual(collated.question_mask.tolist(), question_mask_ans.tolist())
            self.assertEqual(collated.number_mask.tolist(), number_mask_ans.tolist())
            self.assertEqual(collated.operator_label.tolist(), operator_label_ans.tolist())
            self.assertEqual(collated.operand_label.tolist(), operand_label_ans.tolist())
            self.assertEqual(collated.equation_label.tolist(), equation_label_ans.tolist())
            self.assertEqual(collated.equation_mask.tolist(), equation_mask_ans.tolist())
