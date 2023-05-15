from unittest import TestCase
from preprocess import extractNum, problem2CQ, Equation, Problem, getSameNumberIdx
from pathlib import Path

class ProblemTest(TestCase):
    def test_problem2cq(self):
        # mathqa sample
        problem1 = "oshua and jose work at an auto repair center with 3 other workers . for a survey on health care insurance , 2 of the 6 workers will be randomly chosen to be interviewed . what is the probability that joshua and jose will both be chosen ?"
        context1 = "oshua and jose work at an auto repair center with 3 other workers . for a survey on health care insurance , 2 of the 6 workers will be randomly chosen to be interviewed ."
        question1 = "what is the probability that joshua and jose will both be chosen ?"
        self.assertEqual(problem2CQ(problem1), (context1, question1))

        # svamp sample
        problem2 = "every day ryan spends number0 hours on learning english and some more hours on learning chinese . if he spends number1 hours more on learning english than on learning chinese how many hours does he spend on learning chinese ?"
        context2 = "every day ryan spends number0 hours on learning english and some more hours on learning chinese ."
        question2 = "if he spends number1 hours more on learning english than on learning chinese how many hours does he spend on learning chinese ?"
        self.assertEqual(problem2CQ(problem2), (context2, question2))

        # mawps and asdiv sample
        problem3 = "A chef needs to cook number0 potatoes . He has already cooked number1 . If each potato takes number2 minutes to cook , how long will it take him to cook the rest ?"
        context3 = "A chef needs to cook number0 potatoes . He has already cooked number1 ."
        question3 = "If each potato takes number2 minutes to cook , how long will it take him to cook the rest ?"
        self.assertEqual(problem2CQ(problem3), (context3, question3))

        # mawps and asdiv sample 변형 (question에 소숫점이 등장하는 문제)
        problem4 = "A chef needs to cook 9.0 potatoes . He has already cooked 7.0 . If each potato takes 3.0 minutes to cook , how long will it take him to cook the rest ?"
        context4 = "A chef needs to cook number0 potatoes . He has already cooked number1 ."
        question4 = "If each potato takes number2 minutes to cook , how long will it take him to cook the rest ?"
        numbers4 = ["9.0", "7.0", "3.0"]
        problem4 = Problem(problem4, numbers4, Equation(""))
        self.assertEqual(problem4.context, context4)
        self.assertEqual(problem4.question, question4)

        # 질문이 ?가 아닌 .으로 끝나는 경우
        problem5 = "average age of students of an adult school is 40 years . 120 new students whose average age is 32 years joined the school . as a result the average age is decreased by 4 years . find the number of students of the school after joining of the new students ."
        context5 = "average age of students of an adult school is 40 years . 120 new students whose average age is 32 years joined the school . as a result the average age is decreased by 4 years ."
        question5 = "find the number of students of the school after joining of the new students"
        self.assertEqual(problem2CQ(problem5), (context5, question5))


    def test_extract_num(self):
        # 숫자 종류: 10000 (자연수), 1,000,000 (쉼표가 있는 숫자), 1.5 (소수점이 있는 숫자), - 4 or -4(부호가 있는 숫자)
        self.assertEqual(extractNum("es - 4 ≤ x ≤ 5 and"), ["-4", "5"])
        self.assertEqual(extractNum("es -4 ≤ x ≤ 5 and"), ["-4", "5"])
        self.assertEqual(extractNum("and 6 ≤ y ≤ 16 . -4 ho"), ["6", "16", "-4"])
        self.assertEqual(extractNum("atest 6 - digit  divided by 6 , 7 , 8 , 9 ,"), ["6", "6", "7", "8", "9"])
        self.assertEqual(extractNum(" sum of a number and its square is 20 , what i"), ["20"])
        self.assertEqual(extractNum("d $ 5,000 to open -123 haha - 123"), ["5,000", "-123", "-123"])
        self.assertEqual(extractNum("and 6 ≤ y ≤ 16 . -4 ho"), ["6", "16", "-4"])

        #TODO : 추가적인 테스팅

    def test_to_num_problem(self):
        # mathqa sample
        context1 = "oshua and jose work at an auto repair center with 3 other workers . for a survey on health care insurance , 2 of the 6 workers will be randomly chosen to be interviewed ."
        question1 = "what is the probability that joshua and jose will both be chosen ?"
        numbers1 = ["3", "2", "6"]
        problem1 = Problem(context1 + question1, numbers1, Equation(""))
        self.assertEqual(problem1.context,
                         "oshua and jose work at an auto repair center with number0 other workers . for a survey on health care insurance , number1 of the number2 workers will be randomly chosen to be interviewed .")
        self.assertEqual(problem1.question, "what is the probability that joshua and jose will both be chosen ?")

        # svamp sample
        context2 = "every day ryan spends number0 hours on learning english and some more hours on learning chinese ."
        question2 = "if he spends number1 hours more on learning english than on learning chinese how many hours does he spend on learning chinese ?"
        numbers2 = ["11", "23"]
        problem2 = Problem(context2 + question2, numbers2, Equation(""))
        self.assertEqual(problem2.context,
                         "every day ryan spends number0 hours on learning english and some more hours on learning chinese .")
        self.assertEqual(problem2.question,
                         "if he spends number1 hours more on learning english than on learning chinese how many hours does he spend on learning chinese ?")

        # mawps and asdiv sample
        context3 = "A chef needs to cook number0 potatoes . He has already cooked number1 ."
        question3 = "If each potato takes number2 minutes to cook , how long will it take him to cook the rest ?"
        numbers3 = ["9.0", "7.0", "3.0"]
        problem3 = Problem(context3 + question3, numbers3, Equation(""))
        self.assertEqual(problem3.context, "A chef needs to cook number0 potatoes . He has already cooked number1 .")
        self.assertEqual(problem3.question,
                         "If each potato takes number2 minutes to cook , how long will it take him to cook the rest ?")

        # mawps and asdiv sample 변형 (question에 소숫점이 등장하는 문제)
        context4 = "A chef needs to cook 9.0 potatoes . He has already cooked 7.0 ."
        question4 = "If each potato takes 3.0 minutes to cook , how long will it take him to cook the rest ?"
        numbers4 = ["9.0", "7.0", "3.0"]
        problem4 = Problem(context4 + question4, numbers4, Equation(""))
        self.assertEqual(problem4.context, "A chef needs to cook number0 potatoes . He has already cooked number1 .")
        self.assertEqual(problem4.question,
                         "If each potato takes number2 minutes to cook , how long will it take him to cook the rest ?")

        # TODO: mathqa sample 변형
        context1 = "oshua and jose work at an auto repair center with 3 other workers . for a survey on health care insurance , 0 of the 36 workers will be randomly chosen to be interviewed ."
        question1 = "what is the probability that joshua and jose will both be chosen ?"
        numbers1 = ["3", "0", "36"]
        problem1 = Problem(context1 + question1, numbers1, Equation(""))
        self.assertEqual(problem1.context,
                         "oshua and jose work at an auto repair center with number0 other workers . for a survey on health care insurance , number1 of the number2 workers will be randomly chosen to be interviewed .")
        self.assertEqual(problem1.question, "what is the probability that joshua and jose will both be chosen ?")

    def test_get_same_number_idx(self):
        self.assertEqual(getSameNumberIdx(["1", "2", "3", "4", "5"]), [])
        self.assertEqual(getSameNumberIdx(["1", "2", "1", "4", "1"]), [[0,2,4]])
        self.assertEqual(getSameNumberIdx(["1", "2", "1", "4", "1", "2"]), [[0,2,4], [1,5]])
        self.assertEqual(getSameNumberIdx(["0.3333","0.3333"]), [[0,1]])

class TestEquation(TestCase):
    def test_formular2eqation(self):
        linear_formula1 = "add(n1,const_1)|"
        equation1 = Equation(linear_formula1, type="formula")
        self.assertEqual(equation1.getList(), [["add", "n1", "const_1"]])

        linear_formula2 = "multiply(n0,n1)|divide(n2,#0)"
        equation2 = Equation(linear_formula2, type="formula")
        self.assertEqual(equation2.getList(), [["multiply", "n0", "n1"], ["divide", "n2", "#0"]])

        linear_formula3 = "divide(n0,n1)|multiply(const_1,const_1000)|divide(#1,#0)|subtract(#2,n1)|"
        equation3 = Equation(linear_formula3, type="formula")
        self.assertEqual(equation3.getList(),
                         [["divide", "n0", "n1"], ["multiply", "const_1", "const_1000"], ["divide", "#1", "#0"],
                          ["subtract", "#2", "n1"]])

        linear_formula4 = "add(n1,n2)|add(n3,#0)|subtract(const_100,#1)|multiply(n0,#2)|divide(#3,const_100)"
        equation4 = Equation(linear_formula4, type="formula")
        self.assertEqual(equation4.getList(),
                         [["add", "n1", "n2"], ["add", "n3", "#0"], ["subtract", "const_100", "#1"],
                          ["multiply", "n0", "#2"], ["divide", "#3", "const_100"]])

    def test_get_operator_argument(self):
        linear_formula1 = "add(n1,const_1)|"
        equation1 = Equation(linear_formula1, type="formula")
        self.assertEqual(equation1.getOperator(), ["add"])
        self.assertEqual(equation1.getArgument(), [["n1", "const_1"]])

        linear_formula2 = "multiply(n0,n1)|divide(n2,#0)"
        equation2 = Equation(linear_formula2, type="formula")
        self.assertEqual(equation2.getOperator(), ["multiply", "divide"])
        self.assertEqual(equation2.getArgument(), [["n0", "n1"], ["n2", "#0"]])

        linear_formula3 = "divide(n0,n1)|multiply(const_1,const_1000)|divide(#1,#0)|subtract(#2,n1)|"
        equation3 = Equation(linear_formula3, type="formula")
        self.assertEqual(equation3.getOperator(), ["divide", "multiply", "divide", "subtract"])
        self.assertEqual(equation3.getArgument(), [["n0", "n1"], ["const_1", "const_1000"], ["#1", "#0"], ["#2", "n1"]])

        linear_formula4 = "add(n1,n2)|add(n3,#0)|subtract(const_100,#1)|multiply(n0,#2)|divide(#3,const_100)"
        equation4 = Equation(linear_formula4, type="formula")
        self.assertEqual(equation4.getOperator(), ["add", "add", "subtract", "multiply", "divide"])
        self.assertEqual(equation4.getArgument(),
                         [["n1", "n2"], ["n3", "#0"], ["const_100", "#1"], ["n0", "#2"], ["#3", "const_100"]])

    def test_prefix2equation(self):
        converter = Equation()

        prefix1 = "* number0 number1"
        self.assertEqual(converter.prefix2equation(prefix1), [["multiply", "n0", "n1"]])

        prefix2 = "* number0 + number1 number2"
        self.assertEqual(converter.prefix2equation(prefix2), [["add", "n1", "n2"], ["multiply", "n0", "#0"]])

        prefix3 = "+ + number0 number1 number2"
        self.assertEqual(converter.prefix2equation(prefix3), [["add", "n0", "n1"], ["add", "#0", "n2"]])

        prefix4 = "* / - number1 number0 number0 100.0"
        self.assertEqual(converter.prefix2equation(prefix4),
                         [["subtract", "n1", "n0"], ["divide", "#0", "n0"], ["multiply", "#1", "const_100"]])


class Test(TestCase):
    def check_svamp_number_type(self):
        # number가 중복된 경우 문제를
        # number0 number1 number0 형태로 치환하는지 (X)
        # number0 number1 number2 형태로 치환하는지 (O) => 이렇게 치환했음으로 몇번째 number들이 같은 숫자인지 체크할 필요가 있음
        BASE_PATH = Path(__file__).parent.parent
        file_path: str = "data/raw/mawps-asdiv-a_svamp"
        train_path = Path(BASE_PATH, file_path, "train.csv")
        dev_path = Path(BASE_PATH, file_path, "dev.csv")

        dataset_path = [train_path, dev_path]
        import re
        import pandas as pd
        for path in dataset_path:
            with open(path, 'r') as f:
                data = pd.read_csv(path)

                for problem in data.itertuples():
                    numlen = len(problem.Numbers.split())
                    self.assertEqual(numlen, len(re.findall(r"number\d+", problem.Question)))
                    if numlen != len(extractNum(problem.Question)):
                        print(problem)
                        print(extractNum(problem.Question))


    def check_same_nuber_svamp(self):
        # number가 중복으로 출현하는지 확인 => 중복으로 출현함
        BASE_PATH = Path(__file__).parent.parent
        file_path: str = "data/raw/mawps-asdiv-a_svamp"
        train_path = Path(BASE_PATH, file_path, "train.csv")
        dev_path = Path(BASE_PATH, file_path, "dev.csv")

        dataset_path = [train_path, dev_path]
        import re
        import pandas as pd
        for path in dataset_path:
            with open(path, 'r') as f:
                data = pd.read_csv(path)
                for problem in data.itertuples():
                    num_list = problem.Numbers.split()
                    if len(num_list) != len(set(num_list)):
                        print(problem)
