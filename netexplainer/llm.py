import os
import math
import numexpr
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Calculate expression using Python's numexpr library.
    USE ONLY IF NECESSARY.

    Expression should be a single line mathematical expression
    that solves the problem.

    Examples:
        "37593 * 67" for "37593 times 67"
        "37593**(1/5)" for "37593^(1/5)"
    """
    local_dict = {"pi": math.pi, "e": math.e}
    return str(
        numexpr.evaluate(
            expression.strip(),
            global_dict={},  # restrict access to globals
            local_dict=local_dict,  # add common mathematical functions
        )
    )

class LLM:
    def __init__(self, data_path: str):
        """
        Initialize the LLM object with the file provided
        Args:
            data_path (str): The path of the file to process
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'The path {data_path} does not exist')
        elif not os.path.isfile(data_path):
            raise FileExistsError(f'The path {data_path} is not a file, please provide a file')
        elif not data_path.endswith('.txt'):
            raise TypeError(f'The file {data_path} is not a text file, please provide a txt file')

        load_dotenv()
        self.model = None
        loader = TextLoader(data_path)
        self.file = loader.load()

    def get_subquestions(self, question: str) -> list:
        """
        Get sub-questions from the LLM
        Args:
            question (str): The question to process
        Returns:
            list: A list of sub-questions
        """
        template = """You are a network analyst that generates multiple sub-questions related to an input question about a network trace provided in text format.
        I do not need the answer to the question. The ouput should only contain the sub-questions. Be as simple as possible. 
        3 sub-questions as maximum. The sub-questions cannot answer directly the input question.
        Input question: {question}"""
        prompt_decomposition = ChatPromptTemplate.from_template(template)

        generate_queries_decomposition = ( prompt_decomposition | self.model | StrOutputParser() | (lambda x: x.split("\n")))
        sub_questions = generate_queries_decomposition.invoke({"question":question})

        return sub_questions

    def answer_subquestion(self, question: str) -> str:
        """
        Answer the sub-question using the LLM
        Args:
            question (str): The question to process
        Returns:
            str: The answer to the question
        """
        template = """You are a network analyst that answer questions about network traces.
        Use the following network trace to answer the questions.
        If you don't know the answer, just say that you don't know. Keep the answer as concise as possible.
        Question: {question}
        Traces: {traces}
        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            prompt
            | self.model
            | StrOutputParser()
        )

        answer = chain.invoke({"traces": self.file[0].page_content, "question": question})

        return answer

    def format_qa_pairs(self, questions: list, answers: list) -> str:
        """
        Format the questions and answers into a string
        Args:
            questions (list): The list of questions
            answers (list): The list of answers
        Returns:
            str: The formatted string
        """
        formatted_string = ""
        for i, (question, answer) in enumerate(zip(questions, answers), start=1):
            formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
        return formatted_string.strip()

    def get_final_answer(self, question:str, subquestions: list, answers: list) -> str:
        """
        Combine the questions and answers to get a final answer
        Args:
            question (str): The question to process
            subquestions (list): The list of sub-questions
            answers (list): The list of answers to the sub-questions
        Returns:
            str: The final answer
        """
        template = """Here is a set of Q+A pairs:
        {context}
        Use these to synthesize an answer to the question: {question}"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            prompt
            | self.model
            | StrOutputParser()
        )
        final_answer = chain.invoke({"context": self.format_qa_pairs(subquestions, answers), "question": question})
        return final_answer


class LLM_GEMINI(LLM):
    """
    Class for Google Gemini LLM
    """
    def __init__(self, data_path: str, tools: bool = False):
        """
        Initialize the LLM object with the file provided
        Args:
            data_path (str): The path of the file to process
        """
        super().__init__(data_path)
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
        )

        if not tools:
            self.model = llm
        else:
            self.model = llm.bind_tools(
                tools=[calculator],
            )

class LLM_QWEN_2_5_32B(LLM):
    """
    Class for Qwen 2.5 32B LLM
    """
    def __init__(self, data_path: str, tools: bool = False):
        """
        Initialize the LLM object with the file provided
        Args:
            data_path (str): The path of the file to process
        """
        super().__init__(data_path)
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

        llm = ChatGroq(
            model="qwen-2.5-32b",
            temperature=0,
        )

        if not tools:
            self.model = llm
        else:
            self.model = llm.bind_tools(
                tools=[calculator],
            )


class LLM_LLAMA_3_8B(LLM):
    """
    Class for Llama 3.3 70B Versatile LLM
    """
    def __init__(self, data_path: str, tools: bool = False):
        """
        Initialize the LLM object with the file provided
        Args:
            data_path (str): The path of the file to process
        """
        super().__init__(data_path)
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

        llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
        )

        if not tools:
            self.model = llm
        else:
            self.model = llm.bind_tools(
                tools=[calculator],
            )

class LLM_MISTRAL_SABA_24B(LLM):
    """
    Class for Mistral Saba 24B LLM
    """
    def __init__(self, data_path: str, tools: bool = False):
        """
        Initialize the LLM object with the file provided
        Args:
            data_path (str): The path of the file to process
        """
        super().__init__(data_path)
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

        llm = ChatGroq(
            model="mistral-saba-24b",
            temperature=0,
        )

        if not tools:
            self.model = llm
        else:
            self.model = llm.bind_tools(
                tools=[calculator],
            )

class LLM_GEMMA_3(LLM):
    """
    Class for Google Gemma 3 LLM
    """
    def __init__(self, data_path: str, tools: bool = False):
        """
        Initialize the LLM object with the file provided
        Args:
            data_path (str): The path of the file to process
        """
        super().__init__(data_path)
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        llm = ChatGoogleGenerativeAI(
            model="gemma-3-27b-it",
            temperature=0,
            max_tokens=None,
            timeout=None,
        )

        if not tools:
            self.model = llm
        else:
            self.model = llm.bind_tools(
                tools=[calculator],
            )

class LLM_MISTRAL_7B(LLM):
    """
    Class for Mistral 7B LLM
    """
    def __init__(self, data_path: str, tools: bool = False):
        """
        Initialize the LLM object with the file provided
        Args:
            data_path (str): The path of the file to process
        """
        super().__init__(data_path)
        os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="open-mistral-7b",
            temperature=0,
            timeout=30,
        )

        if not tools:
            self.model = llm
        else:
            self.model = llm.bind_tools(
                tools=[calculator],
            )

models = {
    "gemini-2.0-flash": LLM_GEMINI,
    "qwen-2.5-32b": LLM_QWEN_2_5_32B,
    "llama3-8b-8192": LLM_LLAMA_3_8B,
    "mistral-saba-24b": LLM_MISTRAL_SABA_24B,
    "gemma-3-27b": LLM_GEMMA_3,
    "mistral-7b": LLM_MISTRAL_7B,
}
