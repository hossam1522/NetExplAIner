from dataset import Dataset
from llm import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain import hub

def main():
    dataset = Dataset('downloads')
    processed_files = dataset.process_files()
    llm = LLM(processed_files[0])
    #answer = llm.model.invoke({"question": "What is the total number of packets in the trace?"})
    #messages = [
    #    (f"{llm.file[0].page_content[0:1250000]}"
    #    ),
    #    ("What is the total number of packets in the trace?")
    #]
    #answer = (llm.model | StrOutputParser()).invoke(messages)
    #print(answer)
    #print(len(llm.file[0].page_content))

    vector_store = Chroma.from_documents(documents=llm.file, 
                                         embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))
    
    retriever = vector_store.as_retriever()

    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)

    generate_queries_decomposition = ( prompt_decomposition | llm.model | StrOutputParser() | (lambda x: x.split("\n")))

    question = "How many unique communicators are present in the trace?"

    sub_questions = generate_queries_decomposition.invoke({"question":question})

    prompt_rag = hub.pull("rlm/rag-prompt")

    #for sub_question in sub_questions:
    #    print(sub_question)
    #    retrieved_docs = retriever.invoke(sub_question)
    #    answer = (prompt_rag | llm.model | StrOutputParser()).invoke({"context": retriever,
    #                                                                    "question": sub_question})
    #    print(answer)

    print((prompt_rag | llm.model | StrOutputParser()).invoke({"context": retriever.invoke(question),
                                                                "question": question}))


if __name__ == '__main__':
    main()