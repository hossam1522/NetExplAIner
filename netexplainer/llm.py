import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_google_genai import ChatGoogleGenerativeAI


class LLM:
    def __init__(self, data_path):
        load_dotenv()
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
        )

        loader = DirectoryLoader(data_path)
        self.docs = loader.load()
