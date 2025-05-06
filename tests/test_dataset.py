import unittest
import os
from unittest.mock import patch, mock_open, MagicMock
from netexplainer.dataset import Dataset

class TestDataset(unittest.TestCase):
    @patch("netexplainer.dataset.check_output")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    @patch("netexplainer.dataset.rdpcap")
    def setUp(self, mock_rdpcap, mock_isfile, mock_exists, mock_check_output):
        mock_check_output.return_value = b"1\t0.0\tSrc\tDst\tHTTP\t100\tMocked Data"
        mock_rdpcap.return_value = MagicMock()

        self.mock_questions_content = """
        questions:
          - question: "Sample question"
            subquestions: ["Sub1", "Sub2"]
        """
        with patch("builtins.open", mock_open(read_data=self.mock_questions_content)):
            self.dataset = Dataset("dummy.pcap", "dummy_questions.yaml", "big")

    @patch("netexplainer.dataset.check_output", return_value=b"Mocked Data")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    @patch("netexplainer.dataset.rdpcap")
    def test_init(self, mock_rdpcap, mock_isfile, mock_exists, mock_check_output):
        mock_rdpcap.return_value = MagicMock()
        with patch("builtins.open", mock_open(read_data=self.mock_questions_content)):
            dataset = Dataset("dummy.pcap", "dummy_questions.yaml", "big")
            self.assertEqual(dataset._Dataset__path, os.path.abspath("dummy.pcap"))

    @patch("netexplainer.dataset.check_output", return_value=b"Mocked Data")
    @patch("builtins.open", new_callable=mock_open)
    def test_process_file(self, mock_file, mock_check_output):
        processed_path = self.dataset._Dataset__process_file("dummy.pcap", "big")
        self.assertTrue(processed_path.endswith(".txt"))
        mock_file.assert_called_once_with(processed_path, 'w')
        handle = mock_file()
        handle.write.assert_any_call("No.|Time|Source|Destination|Protocol|Length|Info\n")
        handle.write.assert_any_call("Mocked Data\n")

    def test_clean_cap_format(self):
        mocked_data = "1\t0.0\t192.168.1.1\t192.168.1.2\tTCP\t54\t[SYN]"
        result = self.dataset._Dataset__clean_cap_format(mocked_data, "big")
        self.assertIn("|", result)

    @patch("os.path.isfile")
    @patch("os.path.exists")
    @patch("netexplainer.dataset.rdpcap")
    def test_missing_questions_file(self, mock_rdpcap, mock_exists, mock_isfile):
        mock_exists.side_effect = lambda x: True if x == "dummy.pcap" else False
        mock_isfile.side_effect = lambda x: True if x == "dummy.pcap" else False
        mock_rdpcap.return_value = MagicMock()
        with self.assertRaises(FileNotFoundError):
            Dataset("dummy.pcap", "missing.yaml", "big")

if __name__ == '__main__':
    unittest.main()
