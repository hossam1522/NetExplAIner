import unittest
import os
from unittest.mock import patch, mock_open
from netexplainer.dataset import Dataset


class TestDataset(unittest.TestCase):
    @patch("os.listdir")
    def setUp(self, mock_listdir):
        mock_listdir.return_value = []
        self.dataset = Dataset("dummy_path")

    def test_init(self):
        """Test if the Dataset object is initialized correctly."""
        self.assertEqual(self.dataset._Dataset__path, os.path.dirname(os.path.abspath("netexplainer/dataset.py")) + '/dummy_path')

    @patch("os.listdir")
    def test_get_files_from_path(self, mock_listdir):
        """Test if __get_files_from_path correctly identifies network files (mocked)."""
        mock_listdir.return_value = ["test1.pcap", "test2.pcapng", "test3.cap", "not_net_file.txt"]
        expected_files = ["test1.pcap", "test2.pcapng", "test3.cap"]
        actual_files = self.dataset._Dataset__get_files_from_path()
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        mock_listdir.return_value = ["not_net_file1.txt", "not_net_file2.log"]
        self.assertEqual(self.dataset._Dataset__get_files_from_path(), [])

        mock_listdir.return_value = []
        self.assertEqual(self.dataset._Dataset__get_files_from_path(), [])


    @patch("netexplainer.dataset.Dataset._Dataset__cap_to_str")
    @patch("builtins.open", new_callable=mock_open)
    def test_process_file(self, mock_open_func, mock_cap_to_str):
        """Test if __process_file correctly processes a single file (mocked rdpcap)."""
        mock_cap_to_str.return_value = "mock packet 1\nmock packet 2"
        file_path = "dummy_dir/test1.pcap"
        txt_file = self.dataset._Dataset__process_file(file_path)
        self.assertTrue(txt_file.endswith(".txt"))
        expected_txt_file_path = "dummy_dir/test1.txt"
        self.assertEqual(txt_file, expected_txt_file_path)
        mock_cap_to_str.assert_called_once_with(file_path)
        mock_open_func.assert_called_once_with(expected_txt_file_path, 'w')


    @patch("netexplainer.dataset.Dataset._Dataset__cap_to_str")
    @patch("os.listdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_process_files(self, mock_open_func, mock_listdir, mock_cap_to_str):
        """Test process_files with mocked rdpcap, listdir, and open."""
        mock_cap_to_str.return_value = "mock packet 1"
        mock_listdir.return_value = ["test1.pcap", "test2.pcapng", "test3.cap"]

        self.dataset = Dataset("dummy_path")
        processed_files = self.dataset.process_files()
        expected_txt_files = [
            os.path.join(os.path.dirname(os.path.abspath("netexplainer/dataset.py")), "dummy_path", "test1.txt"),
            os.path.join(os.path.dirname(os.path.abspath("netexplainer/dataset.py")), "dummy_path", "test2.txt"),
            os.path.join(os.path.dirname(os.path.abspath("netexplainer/dataset.py")), "dummy_path", "test3.txt")
        ]
        self.assertEqual(sorted(processed_files), sorted(expected_txt_files))

        expected_rdpcap_calls = [
            unittest.mock.call(os.path.join(os.path.dirname(os.path.abspath("netexplainer/dataset.py")), "dummy_path", "test1.pcap")),
            unittest.mock.call(os.path.join(os.path.dirname(os.path.abspath("netexplainer/dataset.py")), "dummy_path", "test2.pcapng")),
            unittest.mock.call(os.path.join(os.path.dirname(os.path.abspath("netexplainer/dataset.py")), "dummy_path", "test3.cap"))
        ]
        mock_cap_to_str.assert_has_calls(expected_rdpcap_calls, any_order=True)

        expected_open_calls = [
            unittest.mock.call(os.path.join(os.path.dirname(os.path.abspath("netexplainer/dataset.py")), "dummy_path", "test1.txt"), 'w'),
            unittest.mock.call(os.path.join(os.path.dirname(os.path.abspath("netexplainer/dataset.py")), "dummy_path", "test2.txt"), 'w'),
            unittest.mock.call(os.path.join(os.path.dirname(os.path.abspath("netexplainer/dataset.py")), "dummy_path", "test3.txt"), 'w')
        ]
        mock_open_func.assert_has_calls(expected_open_calls, any_order=True)


if __name__ == '__main__':
    unittest.main()
