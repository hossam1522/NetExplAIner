from netexplainer.dataset import Dataset
from netexplainer.llm import LLM_GEMINI, LLM_QWEN_2_5_32B
from netexplainer.evaluator import Evaluator
import time

dataset = Dataset('netexplainer/downloads/data.pcap', 'netexplainer/questions/questions.yaml', 128)
#llm = LLM_GEMINI(dataset.processed_file)
llm = LLM_QWEN_2_5_32B(dataset.processed_file)
evaluator = Evaluator(dataset)

results = []

for question in dataset.questions_subquestions.keys():
    subquestions = llm.get_subquestions(question)
    answers = []
    for subquestion in subquestions:
        time.sleep(120)
        answer = llm.answer_subquestion(subquestion)
        answers.append(answer)

    time.sleep(60)
    final_answer = llm.get_final_answer(question, subquestions, answers)

    time.sleep(60)
    subquestions_eval = evaluator.evaluate_subquestions(question, subquestions)
    time.sleep(60)
    answers_eval = evaluator.evaluate_answer(question, final_answer)

    results.append({
        "question": question,
        "subquestions_eval": subquestions_eval,
        "answers_eval": answers_eval
    })

    print(f"Question: {question}")
    print(f"Subquestions Evaluation: {subquestions_eval}")
    print(f"Answers Evaluation: {answers_eval}")
