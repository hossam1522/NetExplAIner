from unittest.mock import patch, MagicMock, call
from netexplainer.dataset import Dataset

def test_dataset():
    with patch('os.listdir') as mock_listdir, \
         patch('scapy.all.rdpcap') as mock_rdpcap, \
         patch('builtins.open', create=True) as mock_open:
        
        # Configure the mocks
        mock_rdpcap.return_value = MagicMock()
        mock_listdir.return_value = ['file1.pcap', 'file2.pcapng', 'file3.cap', 'file4.txt']
        mock_open.return_value = MagicMock()
        
        # Execute the test
        dataset = Dataset('path')
        assert dataset.process_files() == ['path/file1.txt', 'path/file2.txt', 'path/file3.txt']
        
        # Verify the calls
        mock_listdir.assert_called_once_with('path')
        assert mock_rdpcap.call_count == 3
        mock_rdpcap.assert_has_calls([
            call('path/file1.pcap'),
            call('path/file2.pcapng'),
            call('path/file3.cap')
        ])