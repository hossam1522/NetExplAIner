import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI


class LLM:
    def __init__(self, data_path):
        load_dotenv()
        loader = TextLoader(data_path)
        self.file = loader.load()

class LLM_GEMINI(LLM):
    """
    Class for Google Gemini LLM
    """
    def __init__(self, data_path):
        super().__init__(data_path)
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )