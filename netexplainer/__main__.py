from netexplainer.dataset import Dataset
from netexplainer.llm import LLM_GEMINI
import time


dataset = Dataset('netexplainer/downloads/data.pcap', 'netexplainer/questions/questions.yaml')
llm = LLM_GEMINI(dataset.processed_file)

results = []
template = """You are a network analyst that answer questions about network traces.
Use the following network trace to answer the questions.
If you don't know the answer, just say that you don't know. Keep the answer as concise as possible.
Question: {question}
Traces: {traces}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

for sub_question in sub_questions:
    time.sleep(60)
    print(sub_question)
    context = llm.file[0].page_content
    answer = (prompt | llm.model | StrOutputParser()).invoke({"traces": context,
                                                                    "question": sub_question})
    print(answer)
    results.append(answer)

context = llm.format_qa_pairs(sub_questions, results)

template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
time.sleep(20)

final_chain = (
    prompt
    | llm.model
    | StrOutputParser()
)

final_aswer = final_chain.invoke({"context":context,"question":question})

print(final_aswer)
