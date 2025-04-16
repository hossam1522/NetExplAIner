import argparse
import yaml
import os
import sys
import time
import plotly.express as px
from netexplainer.dataset import Dataset
from netexplainer.llm import models
from netexplainer.evaluator import Evaluator
from netexplainer.scraper import Scraper

QUESTIONS_PATH = "netexplainer/data/questions.yaml"

def generate_pie_charts(results: list) -> None:
    """
    Generate pie charts from the evaluation results.

    Args:
        results (list): List of evaluation results.
    """
    from collections import defaultdict

    model_data = defaultdict(list)
    for result in results:
        model_data[result["model"]].append(result["answer_eval"])

    for model, evaluations in model_data.items():
        counts = {
            "Correct (YES)": 0,
            "Incorrect (NO)": 0,
            "Problematic (PROBLEM)": 0
        }

        for eval in evaluations:
            if eval == "YES":
                counts["Correct (YES)"] += 1
            elif eval == "NO":
                counts["Incorrect (NO)"] += 1
            else:
                counts["Problematic (PROBLEM)"] += 1

        total = sum(counts.values())
        if total == 0:
            continue

        labels = list(counts.keys())
        values = [round((count/total)*100, 1) for count in counts.values()]

        fig = px.pie(
            names=labels,
            values=values,
            title=f"{model}",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.write_image(f"netexplainer/data/pie_charts/{model}/answers_pie_chart.png")

def evaluate_without_tools() -> list:
    """
    Evaluates the models without using any tools.
    """
    all_results = []

    for model in models_to_evaluate:
        for file in os.listdir("netexplainer/data/cleaned/"):
            if file.endswith(".txt"):
                continue

            try:
                dataset = Dataset(os.path.join("netexplainer/data/cleaned/", file), QUESTIONS_PATH)
                llm = models[f"{model}"](dataset.processed_file)
                evaluator = Evaluator(dataset)

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

                    try:
                        subquestions_eval = evaluator.evaluate_subquestions(question, subquestions)
                    except Exception as e:
                        subquestions_eval = "ERROR"

                    try:
                        answers_eval = evaluator.evaluate_answer(question, final_answer)
                    except Exception as e:
                        answers_eval = "PROBLEM"

                    all_results.append({
                        "model": model,
                        "file": file,
                        "question": question,
                        "subquestions_eval": subquestions_eval,
                        "answer_eval": answers_eval
                    })

                    print(f"Model: {model}, File: {file}, Question: {question}, Subquestions Eval: {subquestions_eval}, Answer Eval: {answers_eval}")

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                all_results.append({
                    "model": model,
                    "file": file,
                    "question": None,
                    "subquestions_eval": "ERROR",
                    "answer_eval": "ERROR"
                })
                continue

    return all_results


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

    results = evaluate_without_tools()
    generate_pie_charts(results)
