from langchain import PromptTemplate, LLMChain
import func_timeout
import re
from collections import defaultdict

CoT_template = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
Q: {question}
"""


PoT_template = """
Read the following passages to answer questions with Python code, store the result as a 'ans' variable:

# Passage: James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop.
# Question: How many red stickers does James have?
original_red_stickers = 93
used_red_stickers = 31
ans = original_red_stickers - used_red_stickers

# Passage: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars.
# Question: How much do you have to pay to buy for each egg?
original_egg_price_in_dollars = 80
discount_dollars = 29
ans = original_egg_price_in_dollars - discount_dollars

# Passage: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books.
# Question: How many books did danny have at first?
num_books_bought_at_store = 5
num_books_now = 25
ans = num_books_now - num_books_bought_at_store

# Passage: There were 108 chickens and 20 sheeps at the farm, some of chickens and sheeps were sold. There are 87 chickens and 18 sheeps left now.
# Question: How many chickens were sold?
num_chicken_before = 108
num_chicken_now = 87
ans = num_chicken_before - num_chicken_now

# Passage: Katty scored 2 goals on monday, 8 goals on tuesday and 9 goals on wednesday.
# Question: How many did Katty score on monday and wednesday?
num_goals_on_monday = 2
num_goals_on_wednesday = 9
ans = num_goals_on_monday + num_goals_on_wednesday

# Passage: There are 5 girls and 4 boys in the Masquerade, 12 more girls and 7 more boys joined. 
# Question: How many more girls than boys are in the Masquerade?
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
ans = total_girls - total_boys

# Passage: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creasm, and he has 2 ice creams left now. 
# Question: How much ice creasm did Getty purchase?
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
ans = total_ice_creams - num_ice_creams_bought_by_joseph

# Passage: {passage}
# Question : {question}
"""

PhP_template = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? (Hint: The answer is near to 6).
A: We know the Answer Hints: 6. With the Answer Hints: 6, we will answer the question. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? (Hint: The answer is near to 10, 8).
A: We know the Answer Hints: 10, 8. With the Answer Hints: 10, 8, we will answer the question. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? (Hint: The answer is near to 30, 35).
A: We know the Answer Hints: 30, 35. With the Answer Hints: 30, 35, we will answer the question. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? (Hint: The answer near to 8, 12).
A: We know the Answer Hints: 8, 12. With the Answer Hints: 8, 12, we will answer the question. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? (Hint: The answer is near to 9, 5).
A: We know the Answer Hints: 9, 5. With the Answer Hints: 9, 5, we will answer the question. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? (Hint: The answer is near to 20).
A: We know the Answer Hints: 20. With the Answer Hints: 20, we will answer the question. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? (Hint: The answer is near to 45).
A: We know the Answer Hints: 45. With the Answer Hints: 45, we will answer the question. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? (Hint: The answer is near to 8).
A: We know the Answer Hints: 8. With the Answer Hints: 8, we will answer the question. The answer is 8.
Q: {question}
"""

PhP_CoT_template = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? (Hint: The answer is near to 6).
A: We know the Answer Hints: 6. With the Answer Hints: 6, we will answer the question. There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = 6 trees that were planted. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? (Hint: The answer is near to 10, 8).
A: We know the Answer Hints: 10, 8. With the Answer Hints: 10, 8, we will answer the question. There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = 5 cars are in the parking lot. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? (Hint: The answer is near to 30, 35).
A: We know the Answer Hints: 30, 35. With the Answer Hints: 30, 35, we will answer the question. Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left in total. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? (Hint: The answer near to 8, 12).
A: We know the Answer Hints: 8, 12. With the Answer Hints: 8, 12, we will answer the question. Jason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? (Hint: The answer is near to 9, 5).
A: We know the Answer Hints: 9, 5. With the Answer Hints: 9, 5, we will answer the question. Shawn started with 5 toys. He then got 2 toys each from his mom and dad. So he got 2 * 2 = 4 more toys. Now he has 5 + 4 = 9 toys. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? (Hint: The answer is near to 20).
A: We know the Answer Hints: 20. With the Answer Hints: 20, we will answer the question. There were originally 9 computers. For each day from monday to thursday, 5 more computers were installed. So 4 * 5 = 20 computers were added. Now 9 + 20 = 29 computers are now in the server room. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? (Hint: The answer is near to 45).
A: We know the Answer Hints: 45. With the Answer Hints: 45, we will answer the question. Michael started with 58 golf balls. He lost 23 on Tuesday, and lost 2 more on wednesday. So he had 58 - 23 = 35 at the end of Tuesday, and 35 - 2 = 33 at the end of wednesday. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? (Hint: The answer is near to 8).
A: We know the Answer Hints: 8. With the Answer Hints: 8, we will answer the question. Olivia had 23 dollars. She bought 5 bagels for 3 dollars each. So she spent 5 * 3 = 15 dollars. Now she has 23 - 15 = 8 dollars left. The answer is 8.
Q: {question}
"""


def CoT(llm, problem, n=1):
    prompt = PromptTemplate(template=CoT_template, input_variables=["question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    print("problem:")
    print(prompt.format(question=problem.passage + ' ' + problem.question))

    answers = defaultdict(int)
    for i in range(n):
        output = llm_chain.run(problem.passage + ' ' + problem.question)
        print("output:")
        print(output)
        ans = float(re.findall(r"\d+\.\d+|\d+", output)[-1])
        answers[ans] += 1
        print(answers)

    ans = sorted(answers.items(), key=lambda x: x[1], reverse=True)[0][0]

    if ans.is_integer():
        return int(ans)

    return ans


def safe_execute(code_string: str, keys=None):
    def execute(x):
        try:
            exec(x)
            locals_ = locals()
            if keys is None:
                return locals_.get('ans', None)
            else:
                return [locals_.get(k, None) for k in keys]
        except Exception:
            return None
    try:
        ans = func_timeout.func_timeout(5, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None

    return ans


def PoT(llm, problem, n=1):
    prompt = PromptTemplate(template=PoT_template, input_variables=["passage", "question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    print("problem:")
    print(prompt.format(passage=problem.passage, question=problem.question))

    answers = defaultdict(int)
    for i in range(n):
        output = llm_chain.run(passage=problem.passage, question=problem.question)
        print("output:")
        print(output)
        ans = safe_execute(output) #TODO should check output
        if ans:
            ans = float(ans)
            answers[ans] += 1
        print(answers)

    ans = sorted(answers.items(), key=lambda x: x[1], reverse=True)[0][0]

    if ans.is_integer():
        return int(ans)

    return ans


def PhP(llm, problem, prompt_option="PhP"):

    if prompt_option == "PhP":
        prompt = PromptTemplate(template=PhP_template, input_variables=["question"])
    else:
        prompt = PromptTemplate(template=PhP_CoT_template, input_variables=["question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    prev = None
    cnt = 0
    question = problem.passage + ' ' + problem.question
    hint = ""
    while cnt < 100:  # Ask 100 times in maximum
        cnt += 1
        print("Loop {}:".format(cnt))

        output = llm_chain.run(question + hint)
        ans = float(re.findall(r"\d+\.\d+|\d+", output)[-1])

        if ans == prev:
            break

        prev = ans

        if cnt == 1:
            hint += " (Hint: The answer is near to {}).".format(ans)
        else:
            hint = hint[:-2] + ", {}).".format(ans)

        print("Hint:", hint)
        print("Answer: {}".format(ans))

    if prev.is_integer():
        return int(prev)

    return prev






