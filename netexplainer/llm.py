import os
import math
import numexpr
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv
from netexplainer.logger import configure_logger
from langchain_community.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, BaseMessage
from langchain_ollama import ChatOllama

warnings.filterwarnings("ignore", category=DeprecationWarning)
configure_logger(name="llm", filepath=Path(__file__).parent / "data/evaluation/netexplainer.log")
logger = logging.getLogger("llm")


@tool
def calculator(expression: str) -> str:
    """Calculate expression using Python's numexpr library.

    Expression should be a single line mathematical expression
    that solves the problem.

    Examples:
        "37593 * 67" for "37593 times 67"
        "37593**(1/5)" for "37593^(1/5)"
    """
    logger.debug(f"Calculator tool called with expression: {expression}")
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
            logger.error(f'The path {data_path} does not exist')
            raise FileNotFoundError(f'The path {data_path} does not exist')
        elif not os.path.isfile(data_path):
            logger.error(f'The path {data_path} is not a file, please provide a file')
            raise FileExistsError(f'The path {data_path} is not a file, please provide a file')
        elif not data_path.endswith('.txt'):
            logger.error(f'The file {data_path} is not a text file, please provide a txt file')
            raise TypeError(f'The file {data_path} is not a text file, please provide a txt file')

        load_dotenv()
        self.llm = None
        self.model = None
        self.tools = False
        loader = TextLoader(data_path)
        self.file = loader.load()

    def call_llm(self, messages: list[BaseMessage], tools: bool = False) -> str:
        """
        Call the LLM with the provided messages and return the response.
        Args:
            messages (list[BaseMessage]): The list of messages to process
            tools (bool): Whether to use tools or not
        Returns:
            str: The response from the LLM
        """
        if tools:
            response = self.llm_with_tools.invoke(messages)
        else:
            response = self.llm.invoke(messages)

        if response.tool_calls:
            tool_responses = []
            for tool_call in response.tool_calls:
                if tool_call['name'] == "calculator":
                    result = calculator.invoke(tool_call['args']['expression'])
                    tool_responses.append(
                        ToolMessage(
                            content=result,
                            name=tool_call['name'],
                            tool_call_id=tool_call['id']
                        )
                    )

            messages.append(response)
            messages.extend(tool_responses)

            final_response = self.llm.invoke(messages)
            return final_response.content
        else:
            return response.content

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
        messages = {"question": question}

        sub_questions = self.call_llm(prompt_decomposition.format_messages(**messages), tools=self.tools)
        sub_questions = [q.strip() for q in sub_questions.split('\n') if q.strip()]

        logger.debug(f"Model: {self.model}, Question: {question}, Sub-questions generated: {sub_questions}")
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
        DON'T GIVE FUNCTIONS OR CODE, ONLY THE ANSWER. Use the calculator tool if needed.
        Question: "{question}"
        Trace:
        {traces}"""
        prompt = ChatPromptTemplate.from_template(template)
        messages = {"traces": self.file[0].page_content, "question": question}

        answer = self.call_llm(prompt.format_messages(**messages), tools=self.tools)

        logger.debug(f"Model: {self.model}, Question: {question}, Answer: {answer}")
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
        template = """DO NOT GIVE FUNCTIONS OR CODE, ONLY THE ANSWER.
        Here is a set of Q+A pairs:
        {context}
        Use these to synthesize an answer to the question: {question}
        Use the calculator tool if needed."""
        prompt = ChatPromptTemplate.from_template(template)
        messages = {"context": self.format_qa_pairs(subquestions, answers), "question": question}

        final_answer = self.call_llm(prompt.format_messages(**messages), tools=self.tools)

        logger.debug(f"Model: {self.model}, Question: {question}, Final answer: {final_answer}")
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

        self.model = "gemini-2.0-flash"
        self.tools = tools

        llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0,
            max_tokens=None,
            timeout=None,
        )

        self.llm = llm
        if tools:
            self.llm_with_tools = llm.bind_tools(
                tools=[calculator]
            )
            logger.debug("Using Gemini 2.0 Flash LLM with tools")
        else:
            logger.debug("Using Gemini 2.0 Flash LLM without tools")

class LLM_QWEN_2_5_7B(LLM):
    """
    Class for Qwen2.5 7B LLM
    """
    def __init__(self, data_path: str, tools: bool = False):
        """
        Initialize the LLM object with the file provided
        Args:
            data_path (str): The path of the file to process
        """
        super().__init__(data_path)

        self.model = "qwen2.5"
        self.tools = tools

        llm = ChatOllama(
            model=self.model,
            num_ctx=32768,
        )

        self.llm = llm
        if tools:
            self.llm_with_tools = llm.bind_tools(
                tools=[calculator]
            )
            logger.debug("Using Qwen2.5 7B LLM with tools")
        else:
            logger.debug("Using Qwen2.5 7B LLM without tools")

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

        self.model = "gemma-3-27b-it"
        self.tools = tools

        llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0,
            max_tokens=None,
            timeout=None,
        )

        self.llm = llm
        if tools:
            self.llm_with_tools = llm.bind_tools(
                tools=[calculator]
            )
            logger.debug("Using Gemma 3 LLM with tools")
        else:
            logger.debug("Using Gemma 3 LLM without tools")

class LLM_LLAMA2_7B(LLM):
    """
    Class for Llama 2 7B LLM
    """
    def __init__(self, data_path: str, tools: bool = False):
        """
        Initialize the LLM object with the file provided
        Args:
            data_path (str): The path of the file to process
        """
        super().__init__(data_path)

        self.model = "llama2"
        self.tools = tools

        llm = ChatOllama(
            model=self.model,
        )

        self.llm = llm
        if tools:
            self.llm_with_tools = llm.bind_tools(
                tools=[calculator]
            )
            logger.debug("Using Llama 2 7B LLM with tools")
        else:
            logger.debug("Using Llama 2 7B LLM without tools")

class LLM_MISTRAL_7B(LLM):
    """
    Class for Mistral 7B LLM using Ollama
    """
    def __init__(self, data_path: str, tools: bool = False):
        """
        Initialize the LLM object with the file provided
        Args:
            data_path (str): The path of the file to process
        """
        super().__init__(data_path)

        self.model = "mistral"
        self.tools = tools

        llm = ChatOllama(
            model=self.model,
            num_ctx=32768,
        )

        self.llm = llm
        if tools:
            self.llm_with_tools = llm.bind_tools(
                tools=[calculator]
            )
            logger.debug("Using Mistral 7B LLM using Ollama with tools")
        else:
            logger.debug("Using Mistral 7B LLM using Ollama without tools")

class LLM_LLAMA3_8B(LLM):
    """
    Class for Llama3.1 8B LLM
    """
    def __init__(self, data_path: str, tools: bool = False):
        """
        Initialize the LLM object with the file provided
        Args:
            data_path (str): The path of the file to process
        """
        super().__init__(data_path)

        self.model = "llama3.1"
        self.tools = tools

        llm = ChatOllama(
            model=self.model,
            num_ctx=128000,
        )

        self.llm = llm
        if tools:
            self.llm_with_tools = llm.bind_tools(
                tools=[calculator]
            )
            logger.debug("Using Llama3.1 8B LLM with tools")
        else:
            logger.debug("Using Llama3.1 8B LLM without tools")

class LLM_GEMMA3_12B_Ollama(LLM):
    """
    Class for Gemma3 12B LLM using Ollama
    """
    def __init__(self, data_path: str, tools: bool = False):
        """
        Initialize the LLM object with the file provided
        Args:
            data_path (str): The path of the file to process
        """
        super().__init__(data_path)

        self.model = "gemma3:12b"
        self.tools = tools

        llm = ChatOllama(
            model=self.model,
            num_ctx=128000,
        )

        self.llm = llm
        if tools:
            self.llm_with_tools = llm.bind_tools(
                tools=[calculator]
            )
            logger.debug("Using Gemma3 12B LLM using Ollama with tools")
        else:
            logger.debug("Using Gemma3 12B LLM using Ollama without tools")

"""
This dictionary maps model names to their respective LLM classes and
if windows context size is small or big.
"""
models = {
    "gemini-2.0-flash": (LLM_GEMINI, "big"),
    "qwen2.5-7b": (LLM_QWEN_2_5_7B, "big"),
    "gemma-3-27b": (LLM_GEMMA_3, "big"),
    "llama2-7b": (LLM_LLAMA2_7B, "small"),
    "mistral-7b": (LLM_MISTRAL_7B, "big"),
    "llama3.1-8b": (LLM_LLAMA3_8B, "big"),
    "gemma-3-12b-ollama": (LLM_GEMMA3_12B_Ollama, "big"),
}
