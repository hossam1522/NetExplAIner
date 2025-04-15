import os
import pytest
from unittest.mock import patch, MagicMock, Mock
from scapy.error import Scapy_Exception

# Update this import to match your actual project structure
from netexplainer.scraper import Scraper, DATASET_PATH, CLEANED_PATH

def test_get_download_urls():
    """Test URL parsing logic"""
    mock_response = Mock()
    mock_response.text = '''
    <a href="test1.pcap">Link1</a>
    <a href="/test2.cap">Link2</a>
    <a href="https://external.com/test3.pcapng">Link3</a>
    '''
    
    expected_urls = {
        'https://wiki.wireshark.org/test1.pcap',
        'https://wiki.wireshark.org/test2.cap',
        'https://external.com/test3.pcapng'
    }

    with patch('requests.get', return_value=mock_response):
        scraper = Scraper()
        assert set(scraper.download_urls) == expected_urls

def test_download_captures(tmpdir):
    """Test file downloading functionality"""
    test_urls = ['http://example.com/file.pcap']
    
    with patch.object(Scraper, '_Scraper__get_download_urls', return_value=test_urls):
        mock_response = Mock()
        mock_response.content = b'file content'
        
        with patch('requests.get', return_value=mock_response), \
             patch('netexplainer.scraper.DATASET_PATH', str(tmpdir)):
            
            scraper = Scraper()
            scraper.download_captures()
            
            assert os.listdir(str(tmpdir)) == ['file.pcap']
            with open(os.path.join(str(tmpdir), 'file.pcap'), 'rb') as f:
                assert f.read() == b'file content'

def test_clean_raw_data_basic(tmpdir):
    """Test basic file filtering"""
    raw_path = tmpdir.mkdir("raw")
    cleaned_path = tmpdir.mkdir("cleaned")

    (raw_path / "valid.pcap").write("")
    (raw_path / "invalid.txt").write("")

    mock_packets = MagicMock()
    mock_packets.__len__.return_value = 5

    with patch('netexplainer.scraper.rdpcap', return_value=mock_packets) as mock_rdpcap, \
         patch('netexplainer.scraper.DATASET_PATH', str(raw_path)), \
         patch('netexplainer.scraper.CLEANED_PATH', str(cleaned_path)):

        scraper = Scraper()
        scraper.clean_raw_data(max_packets=10, data_path=str(raw_path))

        mock_rdpcap.assert_called_once_with(os.path.join(str(raw_path), "valid.pcap"))
        assert os.listdir(str(cleaned_path)) == ['valid.pcap']

def test_clean_raw_data_packet_count(tmpdir):
    """Test packet count filtering"""
    raw_path = tmpdir.mkdir("raw")
    cleaned_path = tmpdir.mkdir("cleaned")
    (raw_path / "small.pcap").write("")
    (raw_path / "large.pcap").write("")

    mock_packets = MagicMock()
    def mock_rdpcap(path):
        mock = MagicMock()
        if "small" in path:
            mock.__len__.return_value = 5
        else:
            mock.__len__.return_value = 15
        return mock

    with patch('netexplainer.scraper.rdpcap', side_effect=mock_rdpcap), \
         patch('netexplainer.scraper.DATASET_PATH', str(raw_path)), \
         patch('netexplainer.scraper.CLEANED_PATH', str(cleaned_path)):

        scraper = Scraper()
        scraper.clean_raw_data(max_packets=10, data_path=str(raw_path))

        assert os.listdir(str(cleaned_path)) == ['small.pcap']

def test_clean_raw_data_existing_dir(tmpdir, capsys):
    """Test existing directory handling"""
    raw_path = tmpdir.mkdir("raw")
    cleaned_path = tmpdir.mkdir("cleaned")
    (cleaned_path / "existing.txt").write("")

    with patch('netexplainer.scraper.DATASET_PATH', str(raw_path)), \
         patch('netexplainer.scraper.CLEANED_PATH', str(cleaned_path)):

        scraper = Scraper()
        scraper.clean_raw_data(max_packets=10, data_path=str(raw_path))

        captured = capsys.readouterr()
        assert "already exists and is not empty" in captured.out
        assert len(os.listdir(str(cleaned_path))) == 1

def test_clean_raw_data_error_handling(tmpdir, capsys):
    """Test error handling during processing"""
    raw_path = tmpdir.mkdir("raw")
    cleaned_path = tmpdir.mkdir("cleaned")
    (raw_path / "corrupted.pcap").write("invalid data")

    with patch('scapy.all.rdpcap', side_effect=Scapy_Exception("Invalid file")), \
         patch('netexplainer.scraper.DATASET_PATH', str(raw_path)), \
         patch('netexplainer.scraper.CLEANED_PATH', str(cleaned_path)):

        scraper = Scraper()
        scraper.clean_raw_data(max_packets=10, data_path=str(raw_path))

        captured = capsys.readouterr()
        assert "Error processing file corrupted.pcap" in captured.out
        assert len(os.listdir(str(cleaned_path))) == 0
