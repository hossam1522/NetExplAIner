import unittest
import os
from unittest.mock import patch, mock_open, MagicMock
from netexplainer.llm import LLM, models, calculator
from langchain_core.runnables import Runnable
from langchain.agents import AgentType

class TestLLM(unittest.TestCase):
    @patch("netexplainer.llm.TextLoader")
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test", "GROQ_API_KEY": "test"})
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    @patch("netexplainer.llm.load_dotenv")
    def setUp(self, mock_load_dotenv, mock_exists, mock_isfile, mock_loader):
        self.mock_file_content = "Sample trace data\nLine1\nLine2"
        mock_loader.return_value.load.return_value = [MagicMock(page_content=self.mock_file_content)]
        
        with patch("builtins.open", mock_open(read_data=self.mock_file_content)):
            self.llm = LLM("dummy.txt")
            self.llm.model = MagicMock()

    @patch("netexplainer.llm.load_dotenv")
    def test_init_valid_file(self, mock_load_dotenv):
        with patch("os.path.exists", return_value=True), \
             patch("os.path.isfile", return_value=True), \
             patch("builtins.open", mock_open(read_data=self.mock_file_content)):
            llm = LLM("valid.txt")
            self.assertEqual(llm.file[0].page_content, self.mock_file_content)

    @patch("os.path.exists", return_value=False)
    def test_init_file_not_found(self, mock_exists):
        with self.assertRaises(FileNotFoundError):
            LLM("missing.txt")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=False)
    def test_init_not_a_file(self, mock_isfile, mock_exists):
        with self.assertRaises(FileExistsError):
            LLM("directory")

class TestLLMSubclasses(unittest.TestCase):
    @patch("netexplainer.llm.initialize_agent")
    @patch("netexplainer.llm.ChatGoogleGenerativeAI")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    @patch("netexplainer.llm.load_dotenv")
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test"})
    def test_gemini_init(self, mock_load_dotenv, mock_isfile, mock_exists, mock_model, mock_init_agent):
        mock_llm_instance = MagicMock(spec=Runnable)
        mock_model.return_value = mock_llm_instance

        with patch("netexplainer.llm.TextLoader"), \
             patch("builtins.open", mock_open(read_data="data")):
            llm = models["gemini-2.0-flash"][0]("dummy.txt", tools=True)

            mock_init_agent.assert_called_once_with(
                tools=[calculator],
                llm=mock_llm_instance,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            )

    @patch("netexplainer.llm.initialize_agent")
    @patch("netexplainer.llm.ChatOllama")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    @patch("netexplainer.llm.load_dotenv")
    def test_mistral_init(self, mock_load_dotenv, mock_isfile, mock_exists, mock_model, mock_init_agent):
        mock_llm_instance = MagicMock(spec=Runnable)
        mock_model.return_value = mock_llm_instance

        with patch("netexplainer.llm.TextLoader"), \
             patch("builtins.open", mock_open(read_data="data")):
            llm = models["mistral-7b"][0]("dummy.txt", tools=True)

            mock_init_agent.assert_called_once_with(
                tools=[calculator],
                llm=mock_llm_instance,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            )

class TestCalculatorTool(unittest.TestCase):
    def test_calculator_valid(self):
        result = calculator("2 + 3 * 4")
        self.assertEqual(result, "14")

    def test_calculator_math_constants(self):
        result = calculator("pi * 2")
        self.assertAlmostEqual(float(result), 6.283185, places=5)

if __name__ == '__main__':
    unittest.main()
