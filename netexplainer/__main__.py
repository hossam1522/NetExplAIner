from netexplainer.dataset import Dataset
from netexplainer.llm import LLM_GEMINI


dataset = Dataset('netexplainer/downloads/data.pcap', 'netexplainer/questions/questions.yaml')
llm = LLM_GEMINI(dataset.processed_file)
