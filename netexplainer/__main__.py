import argparse
import yaml
import os
import sys
import time
import pandas as pd
from netexplainer.dataset import Dataset
from netexplainer.llm import models
from netexplainer.evaluator import Evaluator
from netexplainer.scraper import Scraper

QUESTIONS_PATH = "netexplainer/data/questions.yaml"

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
            evaluator = Evaluator(dataset)

            results = []

            for question in dataset.questions_subquestions.keys():
                time.sleep(2)
                subquestions = llm.get_subquestions(question)
                answers = []
                for subquestion in subquestions:
                    time.sleep(2)
                    answer = llm.answer_subquestion(subquestion)
                    answers.append(answer)

                time.sleep(2)
                final_answer = llm.get_final_answer(question, subquestions, answers)

                time.sleep(2)
                subquestions_eval = evaluator.evaluate_subquestions(question, subquestions)
                time.sleep(2)
                answers_eval = evaluator.evaluate_answer(question, final_answer)

                results.append({
                    "question": question,
                    "subquestions": subquestions,
                    "answers": answers,
                    "final_answer": final_answer,
                    "subquestions_eval": subquestions_eval,
                    "answer_eval": answers_eval
                })

                print(f"Question: {question}")
                print(f"Subquestions Evaluation: {subquestions_eval}")
                print(f"Answers Evaluation: {answers_eval}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download-data", action="store_true", help="Download network files from Wireshark samples")
    group.add_argument("--clean-data", type=int, metavar="<N>", help="Keep network files with a maximum of N packets")

    args = parser.parse_args()

    if args.download_data:
        scraper = Scraper()
        scraper.download_captures()
        sys.exit(0)
    elif args.clean_data:
        max_packets = args.clean_data
        scraper = Scraper()
        scraper.clean_raw_data(max_packets=max_packets)
        sys.exit(0)

    with open(QUESTIONS_PATH, 'r') as file:
        data = yaml.safe_load(file)
        models_to_evaluate = data['models']

    evaluate_without_tools()
