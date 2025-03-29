import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI


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
        loader = TextLoader(data_path)
        self.file = loader.load()

    def format_qa_pairs(self, questions: list, answers: list) -> str:
        """Format Q and A pairs"""
        formatted_string = ""
        for i, (question, answer) in enumerate(zip(questions, answers), start=1):
            formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
        return formatted_string.strip()

class LLM_GEMINI(LLM):
    """
    Class for Google Gemini LLM
    """
    def __init__(self, data_path: str):
        """
        Initialize the LLM object with the file provided
        Args:
            data_path (str): The path of the file to process
        """
        super().__init__(data_path)
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )