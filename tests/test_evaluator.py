import unittest
import os
from unittest.mock import patch, call
from netexplainer.evaluator import Evaluator

class TestEvaluatorCharts(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator()
        self.mock_results = [
            {"model": "model1", "question": "Q1", "answer_eval": "YES"},
            {"model": "model1", "question": "Q1", "answer_eval": "NO"},
            {"model": "model1", "question": "Q2", "answer_eval": "YES"},
            {"model": "model1", "question": "Q2", "answer_eval": "PROBLEM"},
            {"model": "model2", "question": "Q3", "answer_eval": "NO"},
        ]

    @patch("netexplainer.evaluator.px.pie")
    @patch("netexplainer.evaluator.os.makedirs", wraps=os.makedirs)
    @patch("plotly.graph_objects.Figure.write_image")
    def test_generate_pie_charts(self, mock_write_image, mock_makedirs, mock_pie):
        self.evaluator.generate_pie_charts(self.mock_results)
        mock_makedirs.assert_has_calls([
            call("netexplainer/data/evaluation/model1/", exist_ok=True),
            call("netexplainer/data/evaluation/model2/", exist_ok=True)
        ], any_order=True)

    @patch("netexplainer.evaluator.go.Figure")
    @patch("netexplainer.evaluator.os.makedirs", wraps=os.makedirs)
    @patch("plotly.graph_objects.Figure.write_image")
    def test_generate_bar_charts(self, mock_write_image, mock_makedirs, mock_fig):
        self.evaluator.generate_bar_charts(self.mock_results)

        call_args = mock_fig.call_args_list[0][1]
        self.assertEqual(len(call_args['data']), 2)
        self.assertEqual(call_args['data'][0].name, 'Correct (YES)')
        self.assertEqual(call_args['data'][1].name, 'Incorrect (NO)')

    @patch("netexplainer.evaluator.go.Figure")
    @patch("netexplainer.evaluator.os.makedirs", wraps=os.makedirs)
    @patch("plotly.graph_objects.Figure.write_image")
    def test_radar_charts(self, mock_write_image, mock_makedirs, mock_fig):
        radar_results = [
            {"model": "modelA", "question": "Q5", "subquestions_eval": "80%"},
            {"model": "modelA", "question": "Q5", "subquestions_eval": "60"},
        ]
        self.evaluator.generate_model_subquestions_chart(radar_results)

        call_args = mock_fig.call_args_list[0][1]
        self.assertEqual(call_args['data']['r'], (70.0,))
        self.assertEqual(call_args['data']['theta'], ("Question 1",))

    @patch("netexplainer.evaluator.os.makedirs", wraps=os.makedirs)
    @patch("plotly.graph_objects.Figure.write_image")
    def test_directory_creation_with_tools(self, mock_write_image, mock_makedirs):
        self.evaluator.generate_pie_charts([{"model": "gemma", "answer_eval": "YES"}], tools=True)
        mock_makedirs.assert_called_with("netexplainer/data/evaluation/gemma_tools/", exist_ok=True)

if __name__ == "__main__":
    unittest.main()
