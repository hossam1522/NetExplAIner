from netexplainer.dataset import Dataset
from netexplainer.llm import LLM_GEMINI
import time


dataset = Dataset('netexplainer/downloads/data.pcap', 'netexplainer/questions/questions.yaml')
llm = LLM_GEMINI(dataset.processed_file)

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
