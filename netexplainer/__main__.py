from netexplainer.dataset import Dataset
from netexplainer.llm import models
from netexplainer.evaluator import Evaluator
from netexplainer.scraper import Scraper
import argparse
import yaml
import os
import time

QUESTIONS_PATH = "netexplainer/data/questions.yaml"
with open(QUESTIONS_PATH, 'r') as file:
    data = yaml.safe_load(file)
    models_to_evaluate = data['models']

def evaluate_without_tools() -> None:
    """
    Evaluate the LLM without tools
    """
    for model in models_to_evaluate:
        for file in os.listdir("netexplainer/data/cleaned/"):
            if file.endswith(".txt"):
                continue
            dataset = Dataset(os.path.join("netexplainer/data/cleaned/", file), QUESTIONS_PATH)
            llm = models[f"{model}"](dataset.processed_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download-data", action="store_true", help="Download network files from Wireshark samples")
    group.add_argument("--clean-data", type=int, metavar="<N>", help="Keep network files with a maximum of N packets")

    args = parser.parse_args()

    if args.download_data:
        scraper = Scraper()
        scraper.download_captures()
    elif args.clean_data:
        max_packets = args.clean_data
        scraper = Scraper()
        scraper.clean_raw_data(max_packets=max_packets)

    evaluate_without_tools()
