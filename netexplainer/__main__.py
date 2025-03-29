from netexplainer.dataset import Dataset
from netexplainer.llm import LLM_GEMINI
import time

dataset = Dataset('netexplainer/downloads/data.pcap', 'netexplainer/questions/questions.yaml')
llm = LLM_GEMINI(dataset.processed_file)

question = "How many unique communicators are present in the trace?"

subquestions = llm.get_subquestions(question)
answers = []
for subquestion in subquestions:
    time.sleep(30)
    print(subquestion)
    answer = llm.answer_subquestion(subquestion)
    print(answer)
    answers.append(answer)

print(llm.get_final_answer(question, subquestions, answers))
