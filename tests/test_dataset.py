from unittest import mock
from netexplainer.dataset import Dataset

def test_dataset():
    with mock.patch('os.listdir') as mock_listdir:
        mock_listdir.return_value = ['file1.pcap', 'file2.pcapng', 'file3.cap', 'file4.txt']
        dataset = Dataset('path')
        assert dataset.process_files() == ['path/file1.txt', 'path/file2.txt', 'path/file3.txt']
        mock_listdir.assert_called_once_with('path')
        assert mock_listdir.call_count == 1
        assert mock_listdir.call_args == mock.call('path')