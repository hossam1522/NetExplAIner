import os
import re
import requests
import shutil
from scapy.all import rdpcap

DATASET_PATH = os.path.join(os.getcwd(), "data/raw")

class Scraper:
    def __init__(self):
        """
        Initialize the Scraper object and fetch download URLs.
        """
        self.download_urls = self.__get_download_urls()

    def __get_download_urls(self) -> list:
        """
        Fetch download URLs from the Wireshark Sample Captures page.

        Returns:
            list: A list of unique download URLs for sample captures.
        """
        response = requests.get("https://wiki.wireshark.org/SampleCaptures")

        pattern = r'href\s*=\s*["\']([^"\']*?\.(?:cap|pcap|pcapng))["\']'
        matches = re.findall(pattern, response.text, re.IGNORECASE)
        
        capture_downloads = []
        base_url = "https://wiki.wireshark.org"

        for href in matches:
            if href.startswith(('http://', 'https://')):
                capture_downloads.append(href)
            else:
                corrected = f"{base_url}{href}" if href.startswith('/') else f"{base_url}/{href}"
                capture_downloads.append(corrected)

        return list(set(capture_downloads))

    def download_captures(self) -> None:
        """
        Download sample captures from the URLs fetched by __get_download_urls.
        """
        download_dir = DATASET_PATH
        os.makedirs(download_dir, exist_ok=True)

        for url in self.download_urls:
            try:
                response = requests.get(url)
                filename = url.split("/")[-1]
                filepath = os.path.join(download_dir, filename)

                print(f"Downloading {filename}")
                with open(filepath, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")

    def clean_raw_data(data_path: str = DATASET_PATH, max_packets: int = 128) -> None:
        """
        Clean the raw data by filtering out files that are not in the correct format.

        Args:
            data_path (str): The path to the raw data directory.
            max_packets (int): The maximum number of packets allowed in a capture file.
        """
        cleaned_path = os.path.join(os.getcwd(), "data/cleaned")
        try:
            os.mkdir(cleaned_path)
        except Exception as e:
            print(f"Error creating directory: {e}")
            return

        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
            if file.lower().endswith((".cap", ".pcap", ".pcapng")):
                try:
                    packets = rdpcap(file_path)
                    cap_len = len(packets)

                    if cap_len < 1:
                        print(f"File {file} has less than 1 packet. Skipping...")
                        continue
                    if cap_len > max_packets:
                        print(f"File {file} has more than {max_packets} packets. Skipping...")
                        continue

                    shutil.copy(
                        file_path,
                        os.path.join(cleaned_path, file)
                    )
                    print(f"File {file} successfully copied ({cap_len} packets)")

                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
            else:
                print(f"File {file} is not a capture file (.cap/.pcap/.pcapng)")
