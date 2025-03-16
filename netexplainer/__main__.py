from netexplainer.dataset import Dataset
from netexplainer.llm import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain import hub
import time


def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""

    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()



def main():
    dataset = Dataset('downloads')
    processed_files = dataset.process_files()

    for file in processed_files:
        llm = LLM(file)

        vector_store = Chroma.from_documents(documents=llm.file,
                                            embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))

        retriever = vector_store.as_retriever(search_kwargs={"k": 1})

        template = """You are a network analyst that generates multiple sub-questions related to an input question about a network trace. \n
        The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
        Generate queries related to: {question} \n
        The output should be ONLY a list of sub-questions splited with '\ n'. \n"""
        prompt_decomposition = ChatPromptTemplate.from_template(template)

        generate_queries_decomposition = ( prompt_decomposition | llm.model | StrOutputParser() | (lambda x: x.split("\n")))

        question = "How many unique communicators are present in the trace?"

        sub_questions = generate_queries_decomposition.invoke({"question":question})
        rag_results = []
        prompt_rag = hub.pull("rlm/rag-prompt")

        for sub_question in sub_questions:
            time.sleep(30)
            print(sub_question)
            retrieved_docs = retriever.invoke(sub_question)
            answer = (prompt_rag | llm.model | StrOutputParser()).invoke({"context": retrieved_docs,
                                                                            "question": sub_question})
            print(answer)
            rag_results.append(answer)

        context = format_qa_pairs(sub_questions, rag_results)

        template = """Here is a set of Q+A pairs:

        {context}

        Use these to synthesize an answer to the question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        time.sleep(20)

        final_rag_chain = (
            prompt
            | llm.model
            | StrOutputParser()
        )

        final_aswer = final_rag_chain.invoke({"context":context,"question":question})

        print(final_aswer)

if __name__ == '__main__':
    main()
