import argparse
import yaml
import os
import sys
import time
import re
import plotly.express as px
import plotly.graph_objects as go
from netexplainer.dataset import Dataset
from netexplainer.llm import models
from netexplainer.evaluator import Evaluator
from netexplainer.scraper import Scraper

QUESTIONS_PATH = "netexplainer/data/questions.yaml"

def generate_model_subquestions_chart(results: list) -> None:
    """
    Generate radar charts for the subquestions similarity evaluation.

    Args:
        results (list): List of evaluation results.
    """
    from collections import defaultdict

    model_data = defaultdict(lambda: defaultdict(list))

    for result in results:
        model = result["model"]
        question = result["question"]
        sub_eval = result["subquestions_eval"]

        if not question or sub_eval == "ERROR":
            continue

        try:
            similarity = float(sub_eval)
            model_data[model][question].append(similarity)
        except:
            continue

    for model, questions in model_data.items():
        sorted_questions = sorted(
            questions.keys(),
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )

        avg_values = []
        valid_questions = []

        for q in sorted_questions:
            avg = sum(questions[q]) / len(questions[q])
            avg_values.append(round(avg, 1))
            valid_questions.append("Question " + str(len(valid_questions) + 1))

        fig = go.Figure(data=go.Scatterpolar(
            r=avg_values,
            theta=valid_questions,
            fill='toself',
            line=dict(color='royalblue'),
            name="Similarity"
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(tickfont=dict(size=12))
            ),
            title=f"{model}",
            showlegend=True,
            width=800,
            height=600
        )

        dir_path = f"netexplainer/data/evaluation/{model}/"
        os.makedirs(dir_path, exist_ok=True)
        fig.write_image(f"{dir_path}radar_subquestions_similarity.png")

def generate_bar_charts(results: list) -> None:
    """
    Generate grouped bar charts for correct/incorrect answers per question for each model.
    """
    from collections import defaultdict

    model_question_data = defaultdict(lambda: defaultdict(lambda: {'YES': 0, 'NO': 0}))

    for result in results:
        model = result["model"]
        question = result["question"]
        eval = result["answer_eval"]

        if eval in ['YES', 'NO']:
            model_question_data[model][question][eval] += 1

    for model, questions in model_question_data.items():
        sorted_questions = sorted(
            questions.keys(),
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )

        yes_values = []
        no_values = []
        q_labels = []

        for q in sorted_questions:
            yes_values.append(questions[q]['YES'])
            no_values.append(questions[q]['NO'])
            match = re.search(r'\d+', q)
            q_labels.append("Question " + str(len(q_labels) + 1) if match else q)

        fig = go.Figure(data=[
            go.Bar(name='Correct (YES)', x=q_labels, y=yes_values, marker_color='#4CAF50'),
            go.Bar(name='Incorrect (NO)', x=q_labels, y=no_values, marker_color='#F44336')
        ])

        fig.update_layout(
            barmode='group',
            title=f"{model}",
            xaxis_title="Questions",
            yaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            width=1200,
            height=600,
            margin=dict(t=60)
        )

        dir_path = f"netexplainer/data/evaluation/{model}/"
        os.makedirs(dir_path, exist_ok=True)
        fig.write_image(f"{dir_path}grouped_bar_answers.png")

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
        dir_path = f"netexplainer/data/evaluation/{model}/"
        os.makedirs(dir_path, exist_ok=True)
        fig.write_image(f"{dir_path}answers_pie_chart.png")

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
                    for _ in range(5):
                        try:
                            subquestions = llm.get_subquestions(question)

                            answers = []
                            for subquestion in subquestions:
                                time.sleep(1.5)
                                answer = llm.answer_subquestion(subquestion)
                                answers.append(answer)

                            time.sleep(1.5)
                            final_answer = llm.get_final_answer(question, subquestions, answers)

                            try:
                                time.sleep(1.5)
                                subquestions_eval = evaluator.evaluate_subquestions(question, subquestions)
                            except Exception as e:
                                subquestions_eval = "ERROR"

                            try:
                                time.sleep(1.5)
                                answers_eval = evaluator.evaluate_answer(question, final_answer)
                            except Exception as e:
                                answers_eval = "PROBLEM"

                            if subquestions_eval != "ERROR" and answers_eval != "PROBLEM":
                                break

                        except Exception as e:
                            print(f"Error processing question {question} in file {file}: {e}")

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
    generate_bar_charts(results)
    generate_model_subquestions_chart(results)
