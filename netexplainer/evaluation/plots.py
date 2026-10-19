"""Per model/condition charts (pie, grouped bars, sub-question radar).

Requires the optional ``plots`` extra (plotly + kaleido).
"""

import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def _question_order(question: str, index) -> int:
    if index is not None:
        return int(index)
    match = re.search(r"\d+", question)
    return int(match.group()) if match else 0


def _group(records: list[dict]) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for r in records:
        groups[(r["model"], r["condition"])].append(r)
    return groups


def _questions(items: list[dict]) -> list[str]:
    index = {r["question"]: r.get("question_index") for r in items}
    return sorted(index, key=lambda q: _question_order(q, index[q]))


def pie_chart(model, condition, items, out_dir: Path):
    import plotly.express as px

    counts = {"Correct (YES)": 0, "Incorrect (NO)": 0, "Problematic (PROBLEM)": 0}
    names = {"YES": "Correct (YES)", "NO": "Incorrect (NO)"}
    for r in items:
        counts[names.get(r["verdict"], "Problematic (PROBLEM)")] += 1
    total = sum(counts.values())
    values = [round(c / total * 100, 2) for c in counts.values()]
    with open(out_dir / "answers_pie_chart.txt", "w") as f:
        f.write(f"Model: {model} ({condition})\nCorrect and incorrect answers:\n")
        for label, value in zip(counts, values):
            f.write(f"{label}: {value}%\n")
    fig = px.pie(
        names=list(counts),
        values=values,
        title=f"Correct and incorrect answers: {model} ({condition})",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.write_image(out_dir / "answers_pie_chart.png")


def bar_chart(model, condition, items, out_dir: Path):
    import plotly.graph_objects as go

    questions = _questions(items)
    yes = [
        sum(r["verdict"] == "YES" for r in items if r["question"] == q)
        for q in questions
    ]
    no = [
        sum(r["verdict"] == "NO" for r in items if r["question"] == q)
        for q in questions
    ]
    labels = [f"Question {i}" for i in range(1, len(questions) + 1)]
    with open(out_dir / "answers.txt", "w") as f:
        f.write(f"Model: {model} ({condition})\nCorrect and incorrect answers:\n")
        for q, y, n in zip(questions, yes, no):
            f.write(f"{q}: Correct: {y}, Incorrect: {n}\n")
    fig = go.Figure(
        data=[
            go.Bar(name="Correct (YES)", x=labels, y=yes, marker_color="#4CAF50"),
            go.Bar(name="Incorrect (NO)", x=labels, y=no, marker_color="#F44336"),
        ]
    )
    fig.update_layout(
        barmode="group",
        title=f"Correct and incorrect answers by question: {model} ({condition})",
        xaxis_title="Questions",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        width=1200,
        height=600,
        margin=dict(t=60),
    )
    fig.write_image(out_dir / "grouped_bar_answers.png")


def radar_chart(model, condition, items, out_dir: Path):
    import plotly.graph_objects as go

    items = [r for r in items if r.get("subq_similarity") is not None]
    if not items:
        return
    questions = _questions(items)
    avg = []
    for q in questions:
        values = [r["subq_similarity"] for r in items if r["question"] == q]
        avg.append(round(sum(values) / len(values), 2))
    with open(out_dir / "subquestions_similarity.txt", "w") as f:
        f.write(f"Model: {model} ({condition})\nSubquestions similarity:\n")
        for q, a in zip(questions, avg):
            f.write(f"{q}: {a}\n")
    fig = go.Figure(
        data=go.Scatterpolar(
            r=avg,
            theta=[f"Question {i}" for i in range(1, len(questions) + 1)],
            fill="toself",
            line=dict(color="royalblue"),
            name="Similarity",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        title=f"Subquestions similarity: {model} ({condition})",
        showlegend=True,
        width=800,
        height=600,
    )
    fig.write_image(out_dir / "radar_subquestions_similarity.png")


def generate_plots(records: list[dict], out_dir: Path) -> None:
    """Write the charts of every (model, condition) to ``out_dir/<model>_<condition>/``."""
    for (model, condition), items in _group(records).items():
        target = Path(out_dir) / f"{model}_{condition.replace('+', '_')}"
        target.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Generating charts for {model} ({condition}) in {target}")
        pie_chart(model, condition, items, target)
        bar_chart(model, condition, items, target)
        radar_chart(model, condition, items, target)
